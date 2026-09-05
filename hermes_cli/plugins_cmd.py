"""``hermes plugins`` CLI subcommand — install, update, remove, and list plugins."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Optional

# The staged plugin-update security transaction (artifact identity / staging /
# activation / settlement / post-accept policy) lives in plugin_update_txn; this
# facade binds its API at module level so ``plugins_cmd.<name>`` references and
# monkeypatch seams keep working (leaf-sibling rule: plugin_update_txn never
# imports this module at module level).
from hermes_cli.plugin_update_txn import (
    PluginConsentDrift,
    _commit_staged_plugin_update,
    _consent_record,
    _decline_staged_update,
    _plugin_artifact_identity,
    _plugin_update_root,
    _review_live_drift,
    _run_plugin_update_diff_gate,
    _stage_plugin_update,
)
from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import noninteractive_git_env
from hermes_cli.cli_output import line_input
from hermes_cli.config import cfg_get
from hermes_cli.plugin_capabilities import _child_dict
from hermes_cli.secret_prompt import masked_secret_prompt
from utils import atomic_write_text

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _resolve_git_executable() -> Optional[str]:
    """Resolve a git binary for subprocess use when ``PATH`` may be minimal."""
    found = shutil.which("git")
    if found:
        return found
    if os.name == "nt":
        roots = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ]
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            roots.append(os.path.join(local, "Programs"))
        candidates = [os.path.join(r, "Git", sub, "git.exe") for r in roots for sub in ("cmd", "bin")]
    else:
        candidates = ["/usr/bin/git", "/usr/local/bin/git", "/bin/git"]
    return next((c for c in candidates if c and os.path.isfile(c)), None)


class PluginOperationError(Exception):
    """Recoverable plugin install/update failure (CLI exits; HTTP maps to 4xx)."""


class PluginScanBlocked(PluginOperationError):
    """Plugin failed the security scan and was not installed."""

    def __init__(self, message: str, scan_result=None):
        super().__init__(message)
        self.scan_result = scan_result


def _console():
    """A fresh Rich console (rich is imported lazily)."""
    from rich.console import Console
    return Console()


def _table(columns, **kwargs):
    """A Rich ``Table(**kwargs)`` with ``(header, style)`` *columns* added in order."""
    from rich.table import Table
    table = Table(**kwargs)
    for header, style in columns:
        table.add_column(header, style=style)
    return table


def _is_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _fail(console, message: str) -> None:
    """Print *message* and exit 1."""
    console.print(message)
    sys.exit(1)


def _ask_yes(prompt: str, reader=input) -> bool:
    """One y/N question; EOF / Ctrl-C count as "no"."""
    try:
        answer = reader(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


def _config_value(*keys: str, default: Any) -> Any:
    """Read ``keys`` from config.yaml; *default* on a missing key or any load failure."""
    try:
        from hermes_cli.config import load_config
        return cfg_get(load_config(), *keys, default=default)
    except Exception:
        return default


def _config_name_set(*keys: str) -> set:
    """A list-valued config key as a set (empty on any failure or non-list)."""
    value = _config_value(*keys, default=[])
    return set(value) if isinstance(value, list) else set()


def _config_str(*keys: str, default: str) -> str:
    """A string config key; empty/missing/failed reads coerce to *default*."""
    return _config_value(*keys, default=default) or default


def _write_config_value(section: str, key: str, value: Any) -> None:
    """Persist ``config[section][key] = value`` to config.yaml (creating the section)."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    config.setdefault(section, {})[key] = value
    save_config(config)


def _scan_on_install_enabled() -> bool:
    """Install/update-time security scanning; on by default, off via ``plugins.scan_on_install: false``."""
    return bool(_config_value("plugins", "scan_on_install", default=True))


def _scan_plugin_tree(plugin_dir: Path, identifier: str, *, force: bool, scan_decision_cb=None):
    """Scan *plugin_dir* and enforce the install policy.

    Verdicts: safe → proceed; caution → needs confirmation (``force=True`` or a truthy
    ``scan_decision_cb(result)``); dangerous → always blocked (:class:`PluginScanBlocked`).
    Returns the ScanResult, or None when scanning is disabled.
    """
    if not _scan_on_install_enabled():
        return None
    from tools.plugin_guard import format_scan_report, scan_plugin, should_allow_plugin_install
    result = scan_plugin(plugin_dir, source=identifier)
    allowed, reason = should_allow_plugin_install(result, force=force)

    if allowed is None and scan_decision_cb is not None:
        try:
            if scan_decision_cb(result):
                allowed = True
                reason = "Caution verdict accepted by user"
        except Exception:
            logger.exception("plugin scan decision callback failed")

    if allowed is not True:
        raise PluginScanBlocked(
            f"Security scan blocked plugin install: {reason}\n\n"
            f"{format_scan_report(result)}\n"
            "Review the findings above. Install only plugins from sources "
            "you trust. (Scanning can be configured via "
            "plugins.scan_on_install in config.yaml.)",
            scan_result=result)
    logger.info("plugin scan passed for %s: %s", plugin_dir.name, reason)
    return result


# Highest ``manifest_version`` this installer understands; breaking schema changes bump it.
_SUPPORTED_MANIFEST_VERSION = 1


def _plugins_dir() -> Path:
    """Return the user plugins directory, creating it if needed."""
    plugins = get_hermes_home() / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    return plugins


def _sanitize_plugin_name(name: str, plugins_dir: Path, *, allow_subdir: bool = False) -> Path:
    """Validate a plugin name and return the safe target path inside *plugins_dir*.

    Raises ``ValueError`` on traversal or a target outside the plugins directory. ``allow_subdir``
    permits forward slashes so category keys like ``observability/langfuse`` can be looked up
    (``..`` and backslashes stay rejected); install paths keep ``False`` — a clone lands top-level.
    """
    if allow_subdir and name:
        name = name.strip("/")
    if not name:
        raise ValueError("Plugin name must not be empty.")
    if name in {".", ".."}:
        raise ValueError(f"Invalid plugin name '{name}': must not reference the plugins directory itself.")
    for bad in ("\\", "..") if allow_subdir else ("/", "\\", ".."):
        if bad in name:
            raise ValueError(f"Invalid plugin name '{name}': must not contain '{bad}'.")

    target = (plugins_dir / name).resolve()
    plugins_resolved = plugins_dir.resolve()
    if target == plugins_resolved:
        raise ValueError(f"Invalid plugin name '{name}': resolves to the plugins directory itself.")
    if plugins_resolved not in target.parents:
        raise ValueError(f"Invalid plugin name '{name}': resolves outside the plugins directory.")
    return target


_GITHUB_BROWSER_SEGMENTS = {
    "actions", "blob", "commit", "commits", "issues", "pull", "pulls", "releases", "tree", "wiki",
}
_URL_SCHEMES = ("https://", "http://", "git@", "ssh://", "file://")


def _resolve_git_url(identifier: str) -> tuple[str, Optional[str]]:
    """Turn an identifier into a cloneable Git URL and optional subdirectory.

    ``http://`` and ``file://`` are accepted but trigger a security warning at install time.
    """
    if identifier.startswith(_URL_SCHEMES):
        if identifier.startswith("https://github.com/"):
            path = identifier[len("https://github.com/") :]
            path = path.split("?", 1)[0].split("#", 1)[0].strip("/")
            parts = path.split("/")
            if len(parts) >= 3 and all(parts[:2]) and parts[2] in _GITHUB_BROWSER_SEGMENTS:
                repo = parts[1].removesuffix(".git")
                subdir = None
                if parts[2] == "tree" and len(parts) >= 5:
                    subdir = "/".join(p for p in parts[4:] if p).strip("/") or None
                return f"https://github.com/{parts[0]}/{repo}.git", subdir

        # Explicit ``#subdir`` fragment — unambiguous for any scheme.
        if "#" in identifier:
            git_url, _, frag = identifier.partition("#")
            return git_url, (frag.strip("/") or None)
        # Natural ``.git/`` boundary (GitHub-style URLs).
        git_url, marker, subdir = identifier.partition(".git/")
        if marker:
            return git_url + ".git", (subdir.strip("/") or None)
        return identifier, None

    # owner/repo[/subdir...] shorthand
    parts = [p for p in identifier.strip("/").split("/") if p]
    if len(parts) >= 2:
        subdir = "/".join(parts[2:]).strip("/")
        return f"https://github.com/{parts[0]}/{parts[1]}.git", (subdir or None)
    raise ValueError(
        f"Invalid plugin identifier: '{identifier}'. "
        "Use a Git URL or 'owner/repo' shorthand (optionally with a subdirectory: "
        "'owner/repo/path/to/plugin').")


def _resolve_subdir_within(clone_root: Path, subdir: str) -> Path:
    """Resolve ``subdir`` inside ``clone_root``; ``..``, absolute paths and symlinks may not
    escape the clone. Raises ``PluginOperationError`` if it escapes, is missing, or is a file."""
    clone_root = clone_root.resolve()
    candidate = (clone_root / subdir).resolve()
    if candidate != clone_root and clone_root not in candidate.parents:
        raise PluginOperationError(f"Plugin subdirectory '{subdir}' escapes the repository.")
    if not candidate.exists():
        raise PluginOperationError(f"Plugin subdirectory '{subdir}' does not exist in the repository.")
    if not candidate.is_dir():
        raise PluginOperationError(f"Plugin subdirectory '{subdir}' is not a directory.")
    return candidate


def _repo_name_from_url(url: str) -> str:
    """Repo name from a Git URL (last path component; ssh-style ``git@host:repo`` splits on ':')."""
    name = url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]
    if ":" in name:
        name = name.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return name


def _native_manifest_file(plugin_dir: Path) -> Optional[Path]:
    """``plugin.yaml`` (or ``plugin.yml``) under *plugin_dir*, or None when neither exists."""
    return next((p for p in (plugin_dir / "plugin.yaml", plugin_dir / "plugin.yml") if p.exists()), None)


def _has_portable_manifest(plugin_dir: Path) -> bool:
    """True when ``plugin.json`` exists (or is a symlink, even dangling) under *plugin_dir*."""
    portable_file = plugin_dir / "plugin.json"
    return portable_file.exists() or portable_file.is_symlink()


def _load_yaml_manifest(manifest_file: Path):
    """``yaml.safe_load`` of *manifest_file* (``{}`` when empty); raises on any read/parse error."""
    import yaml
    with open(manifest_file, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_manifest(plugin_dir: Path) -> dict:
    """Read a native or portable manifest, preferring native YAML."""
    manifest_file = _native_manifest_file(plugin_dir)
    if manifest_file is None:
        if not _has_portable_manifest(plugin_dir):
            return {}
        try:
            from hermes_cli.agent_plugins import read_agent_plugin_manifest
            return read_agent_plugin_manifest(plugin_dir)[0]
        except Exception as e:
            logger.warning("Failed to read plugin.json in %s: %s", plugin_dir, e)
            return {}
    try:
        return _load_yaml_manifest(manifest_file)
    except Exception as e:
        logger.warning("Failed to read plugin.yaml in %s: %s", plugin_dir, e)
        return {}


def _looks_like_plugin_dir(target: Path) -> bool:
    """True when *target* has a native/portable manifest or a package ``__init__.py``."""
    return (
        _native_manifest_file(target) is not None
        or (target / "plugin.json").exists()
        or (target / "__init__.py").exists())


def _copy_example_files(plugin_dir: Path, console) -> None:
    """Copy ``*.example`` files to their real names (``config.yaml.example`` -> ``config.yaml``),
    never overwriting an existing file so reinstall keeps user config."""
    for example_file in plugin_dir.glob("*.example"):
        real_name = example_file.stem
        real_path = plugin_dir / real_name
        if real_path.exists():
            continue
        try:
            shutil.copy2(example_file, real_path)
            console.print(f"[dim]  Created {real_name} from {example_file.name}[/dim]")
        except OSError as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to copy {example_file.name}: {e}")


def _missing_env_specs(manifest: dict) -> list[dict]:
    """``requires_env`` entries (plain names or ``{name, description, url, secret}`` dicts,
    normalised to dicts; nameless entries dropped) whose variable is unset in ``~/.hermes/.env``."""
    env_specs = [
        {"name": entry} if isinstance(entry, str) else entry
        for entry in manifest.get("requires_env") or []
        if isinstance(entry, str) or (isinstance(entry, dict) and entry.get("name"))
    ]
    if not env_specs:
        return []
    from hermes_cli.config import get_env_value
    return [s for s in env_specs if not get_env_value(s["name"])]


def _print_python_dependencies(manifest: dict, console) -> None:
    """Print declared ``python_dependencies`` with an install hint — Hermes never auto-installs
    plugin pip dependencies.

    See #64165.
    See #15220, #64165.
    """
    deps = manifest.get("python_dependencies") or []
    if not isinstance(deps, list):
        return
    deps = [d.strip() for d in deps if isinstance(d, str) and d.strip()]
    if not deps:
        return
    plugin_name = manifest.get("name", "this plugin")
    console.print(f"\n[bold]{plugin_name}[/bold] declares Python dependencies (not installed automatically):")
    for dep in deps:
        console.print(f"  - {dep}")
    console.print(
        f"[dim]Install them yourself if needed: pip install {' '.join(repr(d) for d in deps)}[/dim]\n")


def _prompt_plugin_env_vars(manifest: dict, console) -> None:
    """Prompt for unset ``requires_env`` variables and save the answers to the user's ``.env``."""
    missing = _missing_env_specs(manifest)
    if not missing:
        return
    from hermes_cli.config import save_env_value
    from hermes_constants import display_hermes_home
    plugin_name = manifest.get("name", "this plugin")
    console.print(f"\n[bold]{plugin_name}[/bold] requires the following environment variables:\n")
    for spec in missing:
        name = spec["name"]
        desc = spec.get("description", "")
        url = spec.get("url", "")
        console.print(f"  {name}" + (f" — {desc}" if desc else ""))
        if url:
            console.print(f"  [dim]Get yours at: {url}[/dim]")
        try:
            value = (masked_secret_prompt if spec.get("secret", False) else line_input)(f"  {name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n[dim]  Skipped (you can set these later in {display_hermes_home()}/.env)[/dim]")
            return

        if value:
            save_env_value(name, value)
            os.environ[name] = value
            console.print(f"  [green]✓[/green] Saved to {display_hermes_home()}/.env")
        else:
            console.print(f"  [dim]  Skipped (set {name} in {display_hermes_home()}/.env later)[/dim]")

    console.print()


def _display_after_install(plugin_dir: Path, identifier: str) -> None:
    """Show after-install.md if it exists, otherwise a default message."""
    from rich.markdown import Markdown
    from rich.panel import Panel
    console = _console()
    after_install = plugin_dir / "after-install.md"
    if after_install.exists():
        body, title = Markdown(after_install.read_text(encoding="utf-8")), None
    else:
        body = f"[green bold]Plugin installed:[/] {identifier}\n[dim]Location:[/] {plugin_dir}"
        title = "✓ Installed"
    console.print()
    console.print(Panel(body, border_style="green", title=title, expand=False))
    console.print()


def _require_installed_plugin(name: str, plugins_dir: Path, console) -> Path:
    """The plugin path if it exists; else exit 1 (invalid name, or a listing of installed plugins)."""
    try:
        target = _sanitize_plugin_name(name, plugins_dir, allow_subdir=True)
    except ValueError as e:
        _fail(console, f"[red]Error:[/red] {e}")
    if not target.exists():
        installed = ", ".join(d.name for d in plugins_dir.iterdir() if d.is_dir()) or "(none)"
        _fail(console, f"[red]Error:[/red] Plugin '{name}' not found in {plugins_dir}.\nInstalled plugins: {installed}")
    return target


# ── Install metadata + git plumbing ─────────────────────────────────────────────────────────

_EXACT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _install_metadata_path() -> Path:
    return get_hermes_home() / "plugins" / ".install-metadata.json"


def _read_install_metadata() -> dict[str, dict[str, object]]:
    """Read profile-local, non-secret plugin source metadata from disk."""
    path = _install_metadata_path()
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PluginOperationError(f"Could not read plugin install metadata: {exc}") from exc
    if not isinstance(value, dict):
        raise PluginOperationError("Plugin install metadata must be a JSON object.")
    return value


def _write_install_metadata(metadata: dict[str, dict[str, object]]) -> None:
    """Atomically replace the profile-local plugin install metadata sidecar."""
    path = _install_metadata_path()
    atomic_write_text(
        path, json.dumps(metadata, indent=2, sort_keys=True) + "\n", tmp_prefix=f"{path.name}.tmp-")


# ── Artifact (content) consent records ──────────────────────────────────────
# Moved to hermes_cli.plugin_update_txn.py (the single update owner). The
# consent/identity helpers are imported at module level below; install still
# records the baseline through _consent_record/_plugin_artifact_identity at
# tree-swap time, and the update gate compares against it.

def _normalize_exact_revision(ref: str) -> str:
    """Lowercase a full 40-hex commit SHA; anything else is a PluginOperationError."""
    if not isinstance(ref, str) or not _EXACT_COMMIT_RE.fullmatch(ref):
        raise PluginOperationError("--ref must be a full 40-character commit SHA.")
    return ref.lower()


def _safe_git_error(result: subprocess.CompletedProcess, source_url: str = "") -> str:
    """Diagnosable Git output without echoing embedded credentials."""
    from agent.redact import redact_sensitive_text
    error = (result.stderr or result.stdout or "").strip()
    if source_url:
        error = error.replace(source_url, _scrub_git_url(source_url))
    return redact_sensitive_text(error)


def _git_or_raise(
    git_exe: str, repo: Path, *args: str, failure_prefix: str, timeout: int = 60, source_url: str = ""
) -> subprocess.CompletedProcess:
    """Run git in *repo*; on a non-zero exit raise PluginOperationError(prefix + scrubbed error)."""
    result = _run_plugin_git(git_exe, repo, *args, timeout=timeout)
    if result.returncode != 0:
        raise PluginOperationError(failure_prefix + _safe_git_error(result, source_url))
    return result


def _git_head_revision(repo: Path, git_exe: str) -> str:
    return _git_or_raise(
        git_exe, repo, "rev-parse", "HEAD", timeout=15,
        failure_prefix="Could not determine installed Git revision:\n",
    ).stdout.strip().lower()


def _checkout_exact_revision(repo: Path, git_exe: str, revision: str) -> None:
    """Fetch and detach at one immutable commit, then verify the resulting HEAD."""
    for verb, args, failure_prefix in (
        ("fetch", ("fetch", "--depth", "1", "origin", revision), f"Git commit '{revision}' could not be fetched:\n"),
        ("checkout", ("checkout", "--detach", revision), f"Git checkout of commit '{revision}' failed:\n"),
    ):
        try:
            _git_or_raise(git_exe, repo, *args, failure_prefix=failure_prefix)
        except subprocess.TimeoutExpired as exc:
            raise PluginOperationError(f"Git {verb} of commit '{revision}' timed out after 60 seconds.") from exc
    actual = _git_head_revision(repo, git_exe)
    if actual != revision:
        raise PluginOperationError(
            f"Checked-out revision '{actual}' does not match requested commit '{revision}'.")


def _scrub_git_url(git_url: str) -> str:
    """Strip credentials and query/fragment data from an HTTP Git URL."""
    parsed = urllib.parse.urlsplit(git_url)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    return git_url


def _canonical_source(git_url: str, subdir: Optional[str]) -> str:
    scrubbed = _scrub_git_url(git_url)
    return f"{scrubbed}#{subdir}" if subdir else scrubbed


def _scrub_cloned_origin(repo: Path, git_exe: str, git_url: str) -> None:
    """Ensure credentials used for cloning do not survive in ``.git/config``."""
    scrubbed = _scrub_git_url(git_url)
    if scrubbed != git_url:
        _git_or_raise(
            git_exe, repo, "remote", "set-url", "origin", scrubbed, timeout=15,
            failure_prefix="Could not sanitize installed Git remote:\n", source_url=git_url)


def _check_manifest_version(manifest: dict, plugin_name: str) -> None:
    """Reject manifests declaring a newer ``manifest_version`` than this installer supports."""
    mv = manifest.get("manifest_version")
    if mv is None:
        return
    try:
        mv_int = int(mv)
    except (ValueError, TypeError):
        raise PluginOperationError(
            f"Plugin '{plugin_name}' has invalid manifest_version '{mv}' (expected an integer).",
        ) from None
    if mv_int > _SUPPORTED_MANIFEST_VERSION:
        from hermes_cli.config import recommended_update_command
        raise PluginOperationError(
            f"Plugin '{plugin_name}' requires manifest_version {mv}, "
            f"but this installer only supports up to {_SUPPORTED_MANIFEST_VERSION}. "
            f"Run {recommended_update_command()} to update Hermes.",
        ) from None


def _clone_plugin_repo(tmp_clone: Path, git_url: str, revision: Optional[str]) -> str:
    """Shallow-clone *git_url* into *tmp_clone* (detached at *revision* when given), scrub any
    credentials from the recorded origin, and return the installed HEAD SHA."""
    git_exe = _resolve_git_executable()
    if not git_exe:
        raise PluginOperationError("git is not installed or not in PATH.")
    clone_args = ["clone", "--depth", "1", *(["--no-checkout"] if revision else []), git_url, str(tmp_clone)]
    try:
        result = _run_plugin_git(git_exe, tmp_clone.parent, *clone_args)
    except FileNotFoundError as e:
        raise PluginOperationError("git is not installed or not in PATH.") from e
    except subprocess.TimeoutExpired as e:
        raise PluginOperationError("Git clone timed out after 60 seconds.") from e
    if result.returncode != 0:
        raise PluginOperationError(f"Git clone failed:\n{_safe_git_error(result, git_url)}")
    _scrub_cloned_origin(tmp_clone, git_exe, git_url)
    if revision:
        _checkout_exact_revision(tmp_clone, git_exe, revision)
    return _git_head_revision(tmp_clone, git_exe)


def _read_manifest_for_install(plugin_dir: Path) -> dict:
    """Manifest of a freshly cloned tree. Unlike :func:`_read_manifest`, a broken portable
    ``plugin.json`` is an install error (not a silent ``{}``) and its diagnostics are logged."""
    if _native_manifest_file(plugin_dir) is not None or not _has_portable_manifest(plugin_dir):
        return _read_manifest(plugin_dir)
    try:
        from hermes_cli.agent_plugins import read_agent_plugin_manifest
        manifest, diagnostics = read_agent_plugin_manifest(plugin_dir)
    except Exception as exc:
        raise PluginOperationError(f"Portable plugin manifest validation failed: {exc}") from exc
    for diagnostic in diagnostics:
        logger.warning("Agent Plugin install: %s", diagnostic.message)
    return manifest


def _swap_in_plugin(tmp_target: Path, target: Path, backup: Path, old_metadata: dict, new_metadata: dict) -> None:
    """Move the validated clone into place and persist metadata; on any failure restore the
    previous tree (if one was replaced) and the previous metadata sidecar, then re-raise."""
    replaced_existing = target.exists()
    if replaced_existing:
        os.replace(target, backup)
    try:
        os.replace(tmp_target, target)
        _write_install_metadata(new_metadata)
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        if replaced_existing and backup.exists():
            os.replace(backup, target)
        if old_metadata:
            _write_install_metadata(old_metadata)
        else:
            _install_metadata_path().unlink(missing_ok=True)
        raise


def _install_plugin_core(
    identifier: str,
    *,
    force: bool,
    ref: Optional[str] = None,
    scan_decision_cb=None,
) -> tuple[Path, dict, str]:
    """Clone a Git plugin and atomically record its source and exact revision."""
    requested_revision = _normalize_exact_revision(ref) if ref is not None else None
    try:
        git_url, subdir = _resolve_git_url(identifier)
    except ValueError as e:
        raise PluginOperationError(str(e)) from e

    plugins_dir = _plugins_dir()
    source = _canonical_source(git_url, subdir)
    old_metadata = _read_install_metadata()

    # Reinstalling the same pinned source retains its pin, even if its plugin
    # directory was manually removed. Moving a pin requires an explicit --ref.
    if requested_revision is None:
        pins = [e for e in old_metadata.values() if e.get("source") == source and e.get("pinned") is True]
        if len(pins) == 1 and isinstance(pins[0].get("revision"), str):
            requested_revision = _normalize_exact_revision(pins[0]["revision"])

    with tempfile.TemporaryDirectory(prefix=".install-", dir=plugins_dir) as tmp:
        tmp_clone = Path(tmp) / "plugin"
        installed_revision = _clone_plugin_repo(tmp_clone, git_url, requested_revision)
        tmp_target = _resolve_subdir_within(tmp_clone, subdir) if subdir else tmp_clone
        manifest = _read_manifest_for_install(tmp_target)
        plugin_name = manifest.get("name") or (
            subdir.rstrip("/").rsplit("/", 1)[-1] if subdir else _repo_name_from_url(git_url))
        try:
            target = _sanitize_plugin_name(plugin_name, plugins_dir)
        except ValueError as e:
            raise PluginOperationError(str(e)) from e
        _check_manifest_version(manifest, plugin_name)
        # Scan BEFORE anything is moved into place; raises PluginScanBlocked when blocked.
        _scan_plugin_tree(tmp_target, identifier, force=force, scan_decision_cb=scan_decision_cb)

        if target.exists() and not force:
            raise PluginOperationError(
                f"Plugin '{plugin_name}' already exists. Use force reinstall "
                f"or run `hermes plugins update {plugin_name}`.")
        prior = old_metadata.get(plugin_name)
        if target.exists() and requested_revision is None and isinstance(prior, dict) and prior.get("pinned") is True:
            raise PluginOperationError(
                f"Plugin '{plugin_name}' is pinned. Reinstall it with an explicit "
                "--ref <40-character commit SHA> to change its source or revision.")

        # Consent baseline: the artifact identity at the exact revision being
        # installed (canonical git tree id for git clones; whole-tree sha256 for
        # subdir/manual trees without a checkout). Recorded atomically with the
        # tree swap so a baseline always exists; the update gate compares the
        # candidate's identity against it.
        installed_kind, installed_artifact = _plugin_artifact_identity(
            tmp_target, is_git=(tmp_target / ".git").exists(), git_exe=_resolve_git_executable())
        new_metadata = {
            **old_metadata,
            plugin_name: {
                "pinned": requested_revision is not None,
                "revision": installed_revision,
                "source": source,
                "consent": _consent_record(
                    installed_kind, installed_artifact, installed_revision,
                    scope="reinstall" if target.exists() else "install"),
            },
        }
        _swap_in_plugin(tmp_target, target, Path(tmp) / "previous-plugin", old_metadata, new_metadata)

    if not _looks_like_plugin_dir(target):
        logger.warning("%s has no plugin.yaml / __init__.py; may not be a valid plugin", plugin_name)
    _copy_example_files(target, _console())
    installed_manifest = _read_manifest(target)
    return target, installed_manifest, installed_manifest.get("name") or target.name


def _looks_like_bare_index_name(identifier: str) -> bool:
    """True for a bare plugin name (no slash, no URL scheme) — resolved via the community index."""
    return "/" not in identifier and "\\" not in identifier and not identifier.startswith(_URL_SCHEMES)


def _resolve_index_name(identifier: str, console) -> tuple[str, Optional[str]]:
    """Resolve a bare plugin name to ``(install_identifier, pinned_ref)``; exit 1 when unknown or
    ambiguous. The ref is only pinned when it is an exact 40-char SHA; tags are advisory output."""
    from hermes_cli.plugin_index import SECURITY_FOOTER, load_index, resolve_name
    entries, source = load_index()
    entry, candidates = resolve_name(entries, identifier)
    if entry is None:
        if len(candidates) > 1:
            console.print(
                f"[red]Error:[/red] Plugin name '{identifier}' is ambiguous in the "
                f"community index ({source}). Candidates:")
            for c in candidates:
                console.print(f"  {c.name}  →  {c.install_identifier}")
            _fail(console, "Re-run with the exact name or the owner/repo identifier.")
        _fail(console, (
            f"[red]Error:[/red] Plugin '{identifier}' was not found in the "
            f"community index ({source}). Use `hermes plugins search <term>` to "
            "browse, or install directly with an owner/repo identifier."))

    pinned_ref: Optional[str] = None
    if entry.ref and _EXACT_COMMIT_RE.fullmatch(entry.ref):
        pinned_ref = entry.ref.lower()
    elif entry.ref:
        console.print(
            f"[dim]Index pins ref '{entry.ref}' (not an exact commit SHA); "
            "installing the default branch head instead.[/dim]")
    console.print(
        f"[dim]Resolved '{entry.name}' via community index ({source}) → "
        f"{entry.install_identifier}"
        + (f" @ {pinned_ref[:12]}[/dim]" if pinned_ref else "[/dim]"))
    console.print(f"[dim]{SECURITY_FOOTER}[/dim]")
    return entry.install_identifier, pinned_ref


def cmd_install(
    identifier: str,
    force: bool = False,
    enable: Optional[bool] = None,
    ref: Optional[str] = None,
) -> None:
    """Install a plugin from a Git URL, owner/repo shorthand, or index name.

    Bare names resolve through the community index (an explicit ``--ref`` beats the index pin).
    *enable* None prompts "Enable now? [y/N]"; True/False skip the prompt.
    """
    console = _console()
    if _looks_like_bare_index_name(identifier):
        identifier, index_ref = _resolve_index_name(identifier, console)
        if ref is None:
            ref = index_ref

    try:
        git_url, _subdir = _resolve_git_url(identifier)
    except ValueError as e:
        _fail(console, f"[red]Error:[/red] {e}")
    if git_url.startswith(("http://", "file://")):
        console.print(
            "[yellow]Warning:[/yellow] Using insecure/local URL scheme. "
            "Consider using https:// or git@ for production installs.")

    console.print(f"[dim]Cloning {git_url}{f' (subdir: {_subdir})' if _subdir else ''}...[/dim]")

    def _interactive_scan_decision(scan_result) -> bool:
        """Prompt the user to accept a caution-verdict plugin."""
        from tools.plugin_guard import format_scan_report
        console.print()
        console.print("[yellow]⚠ Security scan flagged this plugin:[/yellow]")
        console.print(format_scan_report(scan_result))
        return _is_tty() and _ask_yes("  Install anyway? Only continue if you trust the source. [y/N]: ")

    try:
        target, installed_manifest, installed_name = _install_plugin_core(
            identifier, force=force, ref=ref, scan_decision_cb=_interactive_scan_decision)
    except PluginOperationError as e:
        _fail(console, f"[red]{'Blocked' if isinstance(e, PluginScanBlocked) else 'Error'}:[/red] {e}")
    if not _looks_like_plugin_dir(target):
        console.print(
            f"[yellow]Warning:[/yellow] {installed_name} doesn't contain plugin.yaml, "
            f"plugin.json, or __init__.py. It may not be a valid Hermes plugin.")
    _prompt_plugin_env_vars(installed_manifest, console)
    _print_python_dependencies(installed_manifest, console)
    _display_after_install(target, identifier)

    if enable is None:
        enable = _is_tty() and _ask_yes(f"  Enable '{installed_name}' now? [y/N]: ")
    if enable:
        _set_plugin_enabled(installed_name, enable=True)
        console.print(f"[green]✓[/green] Plugin [bold]{installed_name}[/bold] enabled.")
    else:
        console.print(
            f"[dim]Plugin installed but not enabled. "
            f"Run `hermes plugins enable {installed_name}` to activate.[/dim]")

    # Non-interactive installs and declines leave declared capabilities ungranted (fail closed).
    declared_caps = _declared_capabilities_from_manifest(installed_manifest, installed_name)
    if declared_caps:
        _run_capability_consent(console, installed_name, declared_caps, context="install")
    console.print("[dim]Restart the gateway for the plugin to take effect:[/dim]")
    console.print("[dim]  hermes gateway restart[/dim]")
    console.print()


def cmd_update(name: str, *, accept_update: bool = False, accept_caution: bool = False) -> None:
    """Update an installed plugin through the staged candidate-tree transaction.

    The remote is fetched into a private quarantine tree; the live checkout is
    never mutated before authorization. Any artifact change (canonical git tree
    id drift — bytes, mode, path, type) is reviewed and requires an explicit
    accept (TTY ``y`` / ``--accept-update``); the candidate is then committed
    under the per-plugin lock with rollback. The security scan runs on the
    exact staged candidate BEFORE promotion: dangerous → refused (the live
    tree stays at the consented revision), caution → adopted only with an
    explicit keep-enabled decision (TTY ``y`` / ``--accept-caution``; non-TTY
    without the flag fails closed — nothing is adopted). Capability
    re-consent settles through the same surface-neutral policy the Dashboard
    uses. Decline / interrupt / non-TTY-without-flag leave the live tree at
    the last consented revision. If the live tree already drifted from its
    recorded consent, the drifted state is reviewed instead of silently
    laundered by a no-op update.
    """
    console = _console()
    target = _require_installed_plugin(name, _plugins_dir(), console)
    key = target.name

    console.print(f"[dim]Updating {name}...[/dim]")
    try:
        staged = _stage_plugin_update(name, target)
    except PluginConsentDrift as drift:
        if drift.dirty:
            # The operator's own uncommitted edits make the live executable tree
            # differ from the consented artifact. Never disable or destroy them:
            # abort, keep the plugin enabled as-is, and say what to do next.
            console.print(
                f"[yellow]✗ Update not run:[/yellow] [bold]{name}[/bold] has uncommitted "
                f"tracked changes, so its live tree is not the consented artifact. "
                f"[dim]Commit or stash the local changes inside {target}, then re-run "
                f"`hermes plugins update {name}`. The live plugin was not modified.[/dim]")
            return
        _review_live_drift(
            console, target, key, name, drift,
            accept_update=accept_update, accept_caution=accept_caution)
        return
    except PluginOperationError as exc:
        _fail(console, f"[red]Error:[/red] {exc}")
        return  # unreachable (_fail exits); satisfies the flow analyzer

    if staged.get("unchanged"):
        console.print(f"[green]✓[/green] Plugin [bold]{name}[/bold] is already up to date.")
        return

    token = staged["review_token"]
    old_revision = staged.get("old_revision")
    candidate_artifact = staged["candidate_artifact"]
    old_artifact_id = staged.get("old_artifact_id") or None
    new_manifest = staged["new_manifest"]

    accepted, caution_accepted = _run_plugin_update_diff_gate(
        console,
        _plugin_update_root() / token,  # review runs against the staged candidate
        name,
        old_revision,
        old_artifact_id,
        candidate_artifact,
        staged.get("old_manifest") or {},
        new_manifest,
        td_signature=staged.get("td_signature", False),
        accept_update=accept_update,
        accept_caution=accept_caution)
    if not accepted:
        _decline_staged_update(console, key, name, token, old_revision)
        return

    try:
        policy = _commit_staged_plugin_update(
            name, target, token, accept_caution=caution_accepted, console=console)
    except PluginOperationError as exc:
        _fail(console, f"[red]Error:[/red] {exc}")
        return  # unreachable (_fail exits); satisfies the flow analyzer

    if not policy.committed:
        # Refusal (scan-blocked dangerous, or caution without an explicit keep decision
        # reaching commit): render the SAME policy outcome the Dashboard returns — a
        # refused candidate is never promoted and the live tree stays at the last
        # consented revision.
        console.print()
        console.print(f"[red bold]✗ Update refused.[/red bold] {policy.reason}")
        _render_scan_findings(console, policy)
        console.print(
            f"[dim]Plugin [bold]{name}[/bold] stays on the previously consented "
            f"revision {old_revision or 'HEAD'} — no changes were adopted.[/dim]")
        return

    # Caution findings were rendered by the update gate when a caution verdict resolved
    # (--accept-caution / TTY y), so a committed-caution candidate does not print the
    # report twice. Safe commits carry no findings; only a resolution path that did NOT
    # pass through the gate's report (dashboard/race) would render here.
    if not caution_accepted:
        _render_scan_findings(console, policy)

    # Re-consent when the candidate declares capabilities the granted set lacks;
    # additions stay ungranted until the user says yes (fail closed). See #64228.
    # The owner already settled declared-set bookkeeping; only genuinely pending
    # capabilities reach this interactive tail (Dashboard leaves them pending).
    plugin_id = new_manifest.get("name") or target.name
    declared_caps = _declared_capabilities_from_manifest(new_manifest, plugin_id)
    if declared_caps and policy.pending_capabilities:
        _run_capability_consent(console, plugin_id, declared_caps, context="update")

    console.print(f"[green]✓[/green] Plugin [bold]{name}[/bold] updated.")
    console.print(f"[dim]Now at revision {policy.candidate_revision}.[/dim]")
    if staged.get("output"):
        console.print(f"[dim]{staged['output']}[/dim]")


def _render_scan_findings(console, policy) -> None:
    """Render a commit policy's scan findings (caution report / refusal report)."""
    if not policy.scan_findings or policy.scan_verdict is None:
        return
    if policy.scan_blocked:
        console.print("[yellow]⚠ Security scan findings:[/yellow]")
    else:
        console.print(f"[yellow]⚠ Security scan flagged the updated plugin:[/yellow] {policy.reason}")
    for finding in policy.scan_findings:
        loc = finding.get("file") or ""
        if finding.get("line"):
            loc = f"{loc}:{finding['line']}"
        console.print(
            f"  [bold]{finding.get('severity') or '?'}[/bold] "
            f"{finding.get('category') or '?'} {loc} — {finding.get('description') or ''}")
    console.print(
        "[dim]Review the findings above. Scanning is configured via "
        "plugins.scan_on_install in config.yaml.[/dim]")


def _remove_plugin_core(target: Path) -> None:
    """Remove one plugin and its metadata without splitting their state."""
    metadata = _read_install_metadata()
    if target.name not in metadata:
        shutil.rmtree(target)
        return
    updated = {k: v for k, v in metadata.items() if k != target.name}
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.remove-", dir=target.parent))
    backup = staging / "plugin"
    os.replace(target, backup)
    try:
        _write_install_metadata(updated)
    except Exception:
        try:
            os.replace(backup, target)
        except OSError as restore_exc:
            raise PluginOperationError(
                f"Plugin metadata update failed and '{target.name}' could not be "
                f"restored automatically; recovery copy remains at {backup}."
            ) from restore_exc
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging)


def cmd_remove(name: str) -> None:
    """Remove an installed plugin by name."""
    console = _console()
    plugins_dir = _plugins_dir()
    target = _require_installed_plugin(name, plugins_dir, console)
    try:
        _remove_plugin_core(target)
    except (OSError, PluginOperationError) as exc:
        _fail(console, f"[red]Error:[/red] Could not remove plugin '{name}': {exc}")
    console.print()
    console.print(f"[red]✗[/red] Plugin [bold]{name}[/bold] removed from {plugins_dir}")
    console.print()


# ``plugins.disabled`` is an explicit deny-list that wins over the ``plugins.enabled`` allow-list.
_get_disabled_set = functools.partial(_config_name_set, "plugins", "disabled")
_get_enabled_set = functools.partial(_config_name_set, "plugins", "enabled")


def _save_disabled_set(disabled: set) -> None:
    _write_config_value("plugins", "disabled", sorted(disabled))


def _save_enabled_set(enabled: set) -> None:
    _write_config_value("plugins", "enabled", sorted(enabled))


def _save_plugin_sets(enabled: set, disabled: set) -> None:
    _save_enabled_set(enabled)
    _save_disabled_set(disabled)


_BASIC_AUTH_PLUGIN_KEYS = frozenset({"basic", "dashboard_auth/basic"})


def ensure_basic_auth_plugin_enabled_in_config(cfg: dict) -> bool:
    """Drop the bundled basic dashboard-auth plugin from ``plugins.disabled`` in *cfg*.

    ``hermes setup`` / ``hermes plugins disable basic`` can park it there while
    ``dashboard.basic_auth`` is configured, and password auth then silently fails.
    Returns True when modified.
    """
    plugins_cfg = cfg.get("plugins")
    disabled = plugins_cfg.get("disabled") if isinstance(plugins_cfg, dict) else None
    if not isinstance(disabled, list) or not (set(disabled) & _BASIC_AUTH_PLUGIN_KEYS):
        return False
    plugins_cfg["disabled"] = sorted(set(disabled) - _BASIC_AUTH_PLUGIN_KEYS)
    return True


def _discard_key_and_leaf(names: set, key: str) -> None:
    """Drop *key* and its bare leaf (``observability/langfuse`` -> ``langfuse``) from *names*, so a
    stale legacy bare-name entry can't keep vetoing the canonical key."""
    names.discard(key)
    names.discard(key.split("/")[-1])


def _set_plugin_enabled(name: str, *, enable: bool) -> None:
    """Move *name* between the enabled allow-list and the disabled deny-list and persist both."""
    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    (enabled.add if enable else enabled.discard)(name)
    (disabled.discard if enable else disabled.add)(name)
    _save_plugin_sets(enabled, disabled)


def _resolve_plugin_key(name: str) -> Optional[str]:
    """Canonical registry key for a manifest name / directory name / path key, or ``None``.
    The single normalization point so enable/disable write the key ``PluginManager`` gates on."""
    resolved = _resolve_plugin_key_and_source(name)
    return resolved[0] if resolved else None


def _find_plugin_entry(name: str) -> Optional[tuple]:
    """First discovered ``(name, version, description, source, dir_path, key)`` entry whose
    manifest name or canonical key equals *name*."""
    return next((entry for entry in _discover_all_plugins() if name in (entry[0], entry[5])), None)


def _resolve_plugin_key_and_source(name: str) -> Optional[tuple]:
    """Resolve *name* to ``(canonical_key, source)`` or ``None``. Exact key/manifest-name match
    first; then a bare leaf match (``langfuse`` -> ``observability/langfuse``) only when unique,
    so a same-named nested plugin is never picked silently."""
    entries = _discover_all_plugins()
    for entry in entries:
        if name in (entry[0], entry[5]):
            return (entry[5], entry[3])
    leaf_matches = [(entry[5], entry[3]) for entry in entries if name == entry[5].split("/")[-1]]
    return leaf_matches[0] if len(leaf_matches) == 1 else None


def _set_plugin_entry_flag(plugin_id: str, key: str, value: bool) -> None:
    """Write ``plugins.entries.<plugin_id>.<key> = value`` into config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    entry = _child_dict(_child_dict(_child_dict(config, "plugins"), "entries"), plugin_id)
    entry[key] = bool(value)
    save_config(config)


def cmd_enable(name: str, allow_tool_override: Optional[bool] = None) -> None:
    """Add a plugin to the enabled allow-list (and remove it from disabled).

    Non-bundled plugins are asked about the privileged ``allow_tool_override`` grant;
    tri-state: ``True``/``False`` skip the prompt, ``None`` asks. Bundled plugins are trusted.
    """
    from hermes_cli.relay_plugin_cutover import LEGACY_RELAY_PLUGIN_KEYS, RELAY_PLUGINS_CONFIG_ENV
    console = _console()

    def _refuse_legacy_relay(plugin: str) -> None:
        if plugin in LEGACY_RELAY_PLUGIN_KEYS:
            _fail(console, (
                f"[red]Plugin '{plugin}' was removed.[/red] Relay lifecycle is owned "
                f"by Hermes core; configure {RELAY_PLUGINS_CONFIG_ENV} instead."))

    _refuse_legacy_relay(name)
    resolved = _resolve_plugin_key_and_source(name)
    if resolved is None:
        _fail(console, f"[red]Plugin '{name}' is not installed or bundled.[/red]")
    key, source = resolved
    _refuse_legacy_relay(key)

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    if key in enabled and key not in disabled:
        console.print(f"[dim]Plugin '{key}' is already enabled.[/dim]")
    else:
        enabled.add(key)
        # The loader's disable check matches BOTH the canonical key (``web/firecrawl``) and the
        # manifest name (``web-firecrawl``); a stale entry under either form would silently veto
        # this enable ("explicit disable wins"), so drop the key, its bare leaf, and the name.
        _discard_key_and_leaf(disabled, key)
        manifest_name = next((e[0] for e in _discover_all_plugins() if e[5] == key), None)
        if manifest_name is not None:
            disabled.discard(manifest_name)
        _save_plugin_sets(enabled, disabled)
        console.print(f"[green]✓[/green] Plugin [bold]{key}[/bold] enabled. Takes effect on next session.")

    # Built-in tool override is a privileged grant; bundled plugins are trusted.
    if source == "bundled":
        return
    # When the manifest declares capabilities the consent screen is the canonical grant path
    # (it covers tools.override too); the legacy prompt then only runs on an explicit flag.
    # See #64228.
    declared_caps = _declared_capabilities_for_key(key)
    if declared_caps:
        _run_capability_consent(console, key, declared_caps, context="enable")
        if allow_tool_override is not None:
            _resolve_tool_override_grant(console, key, allow_tool_override)
        return
    _resolve_tool_override_grant(console, key, allow_tool_override)


# ── Capability consent flow ──────────────────────────────────────────────────


# ── Capability consent flow (#64228) ─────────────────────────────────────────
def _declared_capabilities_from_manifest(manifest: dict, plugin_name: str = "?") -> list:
    """Extract + normalize the ``capabilities:`` declaration from a manifest."""
    from hermes_cli.plugin_capabilities import parse_declared_capabilities
    return parse_declared_capabilities((manifest or {}).get("capabilities"), plugin_name)


def _declared_capabilities_for_key(key: str) -> list:
    """Read the declared capabilities for an installed/bundled plugin by key."""
    entry = _find_plugin_entry(key)
    if entry is None:
        return []
    if entry[3] == "entrypoint":
        from hermes_cli.plugins import discover_entrypoint_manifests
        for manifest in discover_entrypoint_manifests():
            if key in (manifest.key, manifest.name):
                return list(manifest.capabilities)
        return []
    if not entry[4]:
        return []
    return _declared_capabilities_from_manifest(_read_manifest(Path(entry[4])), entry[0])


def _run_capability_consent(console, plugin_id: str, declared: list, *, context: str = "install") -> bool:
    """Show the capability consent screen and record the decision; True when granted.

    On consent the pending capabilities are granted under
    ``plugins.entries.<id>.granted_capabilities`` with a hash of the declared set. On decline —
    or in ANY non-interactive context — they stay ungranted (fail closed) and the plugin must
    degrade via ``ctx.has_capability()``. Consent + audit, NOT a sandbox.
    """
    from hermes_cli.plugin_capabilities import CAPABILITY_REGISTRY, pending_capabilities, record_consent
    pending = pending_capabilities(plugin_id, declared)
    if not pending:
        # Refresh the consent hash so a later declaration change is detected.
        if declared:
            record_consent(plugin_id, [], declared)
        return True

    verb = "requests" if context == "install" else "now requests"
    console.print(f"\n  [yellow]Plugin [bold]{plugin_id}[/bold] {verb} the following capabilities:[/yellow]")
    for cap in pending:
        spec = CAPABILITY_REGISTRY.get(cap)
        console.print(f"    [bold]{cap}[/bold] — {spec.description if spec else ''}")
    console.print(
        "  [dim]Granting trusts the plugin author with these host surfaces. "
        "This is consent, not a sandbox — plugins run as regular Python "
        "in-process.[/dim]")

    if not _is_tty():
        console.print(
            "  [yellow]Non-interactive session: capabilities NOT granted "
            "(fail closed).[/yellow] Run "
            f"`hermes plugins capabilities {plugin_id}` to review and "
            f"`hermes plugins enable {plugin_id}` to grant interactively.")
        return False

    if _ask_yes("  Grant these capabilities? [y/N] ", console.input):
        record_consent(plugin_id, pending, declared)
        console.print(
            f"  [green]✓[/green] Granted: {', '.join(pending)} "
            f"([dim]plugins.entries.{plugin_id}.granted_capabilities[/dim])")
        return True

    console.print(
        f"  [dim]Declined. {plugin_id} stays enabled with these capabilities "
        "off; it should degrade gracefully (ctx.has_capability()). Re-run "
        f"`hermes plugins enable {plugin_id}` to grant later.[/dim]")
    return False


def cmd_capabilities(name: Optional[str] = None) -> None:
    """``hermes plugins capabilities [<id>]`` — declared vs granted."""
    from hermes_cli.plugin_capabilities import (
        CAPABILITY_REGISTRY,
        granted_capabilities,
        plugin_capability_granted,
    )
    console = _console()
    rows = []
    for entry in _discover_all_plugins():
        key = entry[5] or entry[0]
        if name is not None and name not in (key, entry[0]):
            continue
        declared = _declared_capabilities_for_key(key)
        granted = granted_capabilities(key)
        # Effective state includes grants live via deprecated allow_* keys.
        effective = {cap for cap in CAPABILITY_REGISTRY if plugin_capability_granted(key, cap)}
        if not declared and not effective and name is None:
            continue
        rows.append((key, entry[3], declared, granted, effective))

    if name is not None and not rows:
        _fail(console, f"[red]Plugin '{name}' is not installed or bundled.[/red]")
    if not rows:
        console.print("[dim]No plugins declare or hold capabilities.[/dim]")
        return

    for key, source, declared, granted, effective in sorted(rows):
        console.print(f"[bold]{key}[/bold] [dim]({source})[/dim]")
        if not declared:
            console.print("  declared: [dim](none)[/dim]")
        for cap in declared:
            if cap not in effective:
                mark = "[yellow]not granted[/yellow]"
            elif cap in granted:
                mark = "[green]granted[/green]"
            else:
                mark = "[green]granted[/green] [dim](via legacy allow_* key — deprecated)[/dim]"
            console.print(f"  {cap}: {mark}")
        for cap in sorted(effective - set(declared)):
            console.print(f"  {cap}: [green]granted[/green] [dim](not declared in manifest)[/dim]")


def _resolve_tool_override_grant(console, key: str, allow_tool_override: Optional[bool]) -> None:
    """Resolve and persist the ``allow_tool_override`` grant for a plugin."""
    if allow_tool_override is None:
        # Default NO: a blind Enter or a non-interactive stdin denies safely.
        allow_tool_override = _ask_yes(
            "[yellow]Allow this plugin to replace built-in tools "
            "(e.g. shell_exec, write_file)?[/yellow]\n"
            "  This is a privileged capability: an override can intercept "
            "everything the agent routes through that tool.\n"
            "  Grant it? [y/N] ",
            console.input,
        )
    _set_plugin_entry_flag(key, "allow_tool_override", allow_tool_override)
    if allow_tool_override:
        console.print(
            f"[green]✓[/green] Granted [bold]{key}[/bold] permission to "
            "override built-in tools "
            f"([dim]plugins.entries.{key}.allow_tool_override: true[/dim]).")
    else:
        console.print(
            f"[dim]{key} may not override built-in tools. Re-run "
            f"`hermes plugins enable {key} --allow-tool-override` to grant "
            "this later.[/dim]")


def cmd_disable(name: str) -> None:
    """Remove a plugin from the enabled allow-list (and add to disabled)."""
    console = _console()
    key = _resolve_plugin_key(name)
    if key is None:
        _fail(console, f"[red]Plugin '{name}' is not installed or bundled.[/red]")
    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    if key not in enabled and key in disabled:
        console.print(f"[dim]Plugin '{key}' is already disabled.[/dim]")
        return
    # Also drop a stale legacy bare-name entry so it can't keep a nested plugin loading.
    _discard_key_and_leaf(enabled, key)
    disabled.add(key)
    _save_plugin_sets(enabled, disabled)
    console.print(
        f"[yellow]\u2298[/yellow] Plugin [bold]{key}[/bold] disabled. Takes effect on next session.")


def _read_manifest_info(d: Path, prefix: str):
    """Read a native or portable manifest and return display metadata."""
    manifest_file = _native_manifest_file(d)
    if manifest_file is None:
        if not _has_portable_manifest(d):
            return None
        try:
            from hermes_cli.agent_plugins import read_agent_plugin_manifest
            manifest = read_agent_plugin_manifest(d)[0]
            name = manifest["name"]
        except Exception:
            return None
    else:
        # Unreadable YAML (or no yaml module) degrades to the directory name, silently.
        try:
            manifest = _load_yaml_manifest(manifest_file)
        except Exception:
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
        name = manifest.get("name", d.name)
    key = f"{prefix}/{d.name}" if prefix else name
    return name, manifest.get("version", ""), manifest.get("description", ""), key


def _is_portable_plugin_dir(dir_path) -> bool:
    """True for an Agent Plugins v1 package (``plugin.json`` only; native ``plugin.yaml`` wins)."""
    try:
        d = Path(dir_path)
        return d.is_dir() and _native_manifest_file(d) is None and _has_portable_manifest(d)
    except OSError:
        return False


# Manifest kinds active-by-default when bundled (backends auto-load, platforms register lazily,
# model providers go through providers/ discovery). Standalone/exclusive kinds stay opt-in.
_BUNDLED_DEFAULT_ON_KINDS = frozenset({"backend", "platform", "model-provider"})


def _bundled_default_on(dir_path) -> bool:
    """True when a bundled plugin is active without a ``plugins.enabled`` entry (portable
    ``plugin.json`` packages have no kind, so never)."""
    manifest_file = _native_manifest_file(Path(dir_path))
    if manifest_file is None:
        return False
    try:
        kind = str(_load_yaml_manifest(manifest_file).get("kind", "standalone")).strip().lower()
        return kind in _BUNDLED_DEFAULT_ON_KINDS
    except Exception:
        return False


def _scan_level(base: Path, source: str, skip_names: set, prefix: str, depth: int, seen: dict) -> None:
    """Recursive directory scan matching PluginManager._scan_directory_level."""
    if not base.is_dir():
        return
    for d in sorted(base.iterdir()):
        if not d.is_dir() or (depth == 0 and skip_names and d.name in skip_names):
            continue
        info = _read_manifest_info(d, prefix)
        if info is None:
            if depth == 0:
                _scan_level(d, source, set(), f"{prefix}/{d.name}" if prefix else d.name, 1, seen)
            continue
        name, version, description, key = info
        if key in seen and source == "bundled":
            continue
        src_label = "git" if source == "user" and (d / ".git").exists() else source
        seen[key] = (name, version, description, src_label, d, key)


def _discover_all_plugins() -> list:
    """``(name, version, description, source, dir_path, key)`` for every plugin the loader sees,
    in ``PluginManager.discover_and_load`` order: bundled, user, entry points (later wins)."""
    seen: dict = {}
    # memory/, context_engine/ and model-providers/ load through dedicated registries, not the
    # PluginManager opt-in surface, so listing them as toggleable plugins would mislead.
    from hermes_cli.plugins import discover_entrypoint_manifests, get_bundled_plugins_dir
    for base, source, skip in (
        (get_bundled_plugins_dir(), "bundled", {"memory", "context_engine", "model-providers"}),
        (_plugins_dir(), "user", set()),
    ):
        _scan_level(base, source, skip, "", 0, seen)
    # Entry-point plugins are installed as Python packages, so they have no plugin directory.
    for m in discover_entrypoint_manifests():
        seen[m.name] = (m.name, m.version, m.description, "entrypoint", m.path, m.name)
    return list(seen.values())


def _plugin_status(name: str, enabled: set, disabled: set, key: str = "") -> str:
    """User-facing activation state for a plugin name or key."""
    names = {name, key}
    return "disabled" if names & disabled else "enabled" if names & enabled else "not enabled"


def _filter_plugin_entries(entries: list, args: Any, enabled: set, disabled: set) -> list:
    """Apply ``hermes plugins list`` CLI filters."""
    filtered = entries
    if getattr(args, "no_bundled", False) or getattr(args, "user", False):
        filtered = [entry for entry in filtered if entry[3] != "bundled"]
    if getattr(args, "enabled", False):
        filtered = [
            entry for entry in filtered
            if _plugin_status(entry[0], enabled, disabled, key=entry[5]) == "enabled"
        ]
    return filtered


_STATUS_MARKUP = {"disabled": "[red]disabled[/red]", "enabled": "[green]enabled[/green]"}


def cmd_list(args: Any | None = None) -> None:
    """List all plugins (bundled + user) with enabled/disabled state."""
    console = _console()
    entries = _discover_all_plugins()
    if not entries:
        console.print("[dim]No plugins installed.[/dim]")
        console.print("[dim]Install with:[/dim] hermes plugins install owner/repo")
        return

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    entries = _filter_plugin_entries(entries, args, enabled, disabled)
    rows = [
        (name, _plugin_status(name, enabled, disabled, key=key), str(version), description, source)
        for name, version, description, source, _dir, key in entries
    ]

    if getattr(args, "json", False):
        keys = ("name", "status", "version", "description", "source")
        print(json.dumps([dict(zip(keys, row)) for row in rows], indent=2))
        return

    if getattr(args, "plain", False):
        for name, status, version, _description, source in rows:
            print(f"{status:12} {source:8} {version:8} {name}")
        return

    if not entries:
        console.print("[dim]No plugins matched the selected filters.[/dim]")
        return

    table = _table(
        (("Name", "bold"), ("Status", None), ("Version", "dim"), ("Description", None), ("Source", "dim")),
        title="Plugins", show_lines=False)
    for name, status_name, version, description, source in rows:
        status = _STATUS_MARKUP.get(status_name, "[yellow]not enabled[/yellow]")
        table.add_row(name, status, version, description, source)
    console.print()
    console.print(table)
    console.print()
    # Load-derived warnings ride on the live manager (this process already ran discovery when a
    # plugin is loaded here); manifest-only scans have no registration data to compare (G2-2a),
    # and drift-gate fail-open advisories live only on the loaded record (G4-3).
    try:
        from hermes_cli.plugins import get_plugin_manager
        loaded_info = {p["key"]: p for p in get_plugin_manager().list_plugins()}
    except Exception:
        loaded_info = {}
    for name, _version, _description, _source, _dir, key in entries:
        info = loaded_info.get(key) or loaded_info.get(name) or {}
        undeclared = info.get("undeclared_hooks")
        if undeclared:
            console.print(
                f"[yellow]⚠ plugin '{name}' registered undeclared hook(s): "
                f"{', '.join(undeclared)} — declare them in the manifest's "
                f"provides_hooks (see the load log).[/yellow]")
        advisory = info.get("load_advisory")
        if advisory:
            console.print(f"[yellow]⚠ plugin '{name}': {advisory}[/yellow]")
    console.print("[dim]Compact view:[/dim] hermes plugins list --plain --no-bundled")
    console.print("[dim]Interactive toggle:[/dim] hermes plugins")
    console.print("[dim]Enable/disable:[/dim] hermes plugins enable/disable <name>")
    console.print("[dim]Plugins are opt-in by default — only 'enabled' plugins load.[/dim]")


# ── Provider plugin discovery helpers ───────────────────────────────────────────────────────


def _discover_memory_providers() -> list[tuple[str, str]]:
    """``[(name, description), ...]`` for available memory providers."""
    try:
        from plugins.memory import discover_memory_providers
        return [(name, desc) for name, desc, _avail in discover_memory_providers()]
    except Exception:
        return []


def _discover_context_engines() -> list[tuple[str, str]]:
    """``[(name, description), ...]`` for repo-shipped context engines plus the plugin-registered
    one (``ctx.register_context_engine``); repo-shipped descriptions win on a name collision."""
    engines: dict[str, str] = {}
    try:
        from plugins.context_engine import discover_context_engines
        for name, desc, _avail in discover_context_engines():
            engines.setdefault(name, desc)
    except Exception:
        pass
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_context_engine
        discover_plugins()
        plugin_engine = get_plugin_context_engine()
        if plugin_engine and getattr(plugin_engine, "name", None):
            engines.setdefault(plugin_engine.name, "installed plugin")
    except Exception:
        pass
    return list(engines.items())


# memory.provider ("" = built-in) and context.engine config accessors.
_get_current_memory_provider = functools.partial(_config_str, "memory", "provider", default="")
_get_current_context_engine = functools.partial(_config_str, "context", "engine", default="compressor")
_save_memory_provider = functools.partial(_write_config_value, "memory", "provider")
_save_context_engine = functools.partial(_write_config_value, "context", "engine")


# (title, default label, default name, current-value reader, discovery fn, saver) per provider
# category. Readers/savers are looked up at call time so module-level patching still applies.
_PROVIDER_CATEGORY_SPECS = (
    ("Memory Provider", "built-in", "", lambda: _get_current_memory_provider(),
     lambda: _discover_memory_providers(), lambda v: _save_memory_provider(v)),
    ("Context Engine", "compressor", "compressor", lambda: _get_current_context_engine(),
     lambda: _discover_context_engines(), lambda v: _save_context_engine(v)),
)


def _configure_category_spec(spec) -> bool:
    """Radio picker for one ``_PROVIDER_CATEGORY_SPECS`` row: the built-in default first, then the
    discovered choices; a current value not among them is appended as ``(not found)``. Saves and
    returns True when the choice changed."""
    from hermes_cli.curses_ui import curses_radiolist
    title, default_label, default_name, current, discover, save = spec
    current = current()
    choices = discover()
    names = [default_name] + [name for name, _desc in choices]
    items = [f"{default_label} (default)"] + [f"{name} \u2014 {desc}" if desc else name for name, desc in choices]
    if current not in names:
        names.append(current)
        items.append(f"{current} (not found)")
    selected = max(i for i, name in enumerate(names) if name == current)
    new_value = names[curses_radiolist(title=f"{title} (select one)", items=items, selected=selected)]
    if new_value == current:
        return False
    save(new_value)
    return True


def _provider_categories() -> list:
    """``[(title, current_label, configure_fn), ...]`` rows for the composite UI."""
    return [(s[0], s[3]() or s[1], functools.partial(_configure_category_spec, s)) for s in _PROVIDER_CATEGORY_SPECS]


# ── Composite plugins UI ────────────────────────────────────────────────────────────────────


def cmd_show(name: str) -> None:
    """Show details for a single plugin, including declared emits/listens."""
    console = _console()
    match = _find_plugin_entry(name)
    if match is None:
        console.print(f"[red]Plugin '{name}' not found.[/red]")
        _fail(console, "[dim]List installed plugins:[/dim] hermes plugins list")

    pname, version, description, source, dir_path, key = match
    manifest = _read_manifest(Path(dir_path)) if dir_path else {}
    emits = manifest.get("emits") or []
    listens = manifest.get("listens") or []
    status = _plugin_status(pname, _get_enabled_set(), _get_disabled_set(), key=key)
    console.print()
    console.print(f"[bold]{pname}[/bold]" + (f" [dim]v{version}[/dim]" if version else ""))
    if description:
        console.print(description)
    console.print(f"[dim]Status:[/dim] {status}")
    console.print(f"[dim]Source:[/dim] {source}")
    console.print(f"[dim]Key:[/dim] {key}")
    console.print("[dim]Emits:[/dim] " + (", ".join(emits) if emits else "[dim](none)[/dim]"))
    console.print("[dim]Listens:[/dim] " + (", ".join(listens) if listens else "[dim](none)[/dim]"))
    console.print()


def cmd_toggle() -> None:
    """Interactive composite UI — general plugins + provider plugin categories."""
    console = _console()
    entries = _discover_all_plugins()
    enabled_set = _get_enabled_set()
    disabled_set = _get_disabled_set()

    # Track by CANONICAL KEY, not manifest name: the loader and enable/disable all gate on the
    # key (``web/firecrawl``) while the name may differ (``web-firecrawl``); persisting the bare
    # name let plugins.disabled drift so "explicit disable wins" kept a plugin off forever.
    plugin_keys = [entry[5] for entry in entries]
    # Keys keep every surface aligned. See #40190.
    plugin_labels = [
        (f"{name} \u2014 {description}" if description else name) + (" [bundled]" if source == "bundled" else "")
        for name, _version, description, source, _d, _key in entries
    ]
    # Selected when enabled AND not disabled; the legacy bare name counts on either side.
    plugin_selected = {
        i for i, (name, _v, _desc, _src, _d, key) in enumerate(entries)
        if {key, name} & enabled_set and not ({key, name} & disabled_set)
    }
    categories = _provider_categories()

    if not sys.stdin.isatty():
        console.print("[dim]Interactive mode requires a terminal.[/dim]")
        return
    try:
        import curses
        _run_composite_ui(curses, plugin_keys, plugin_labels, plugin_selected, disabled_set, categories, console)
    except ImportError:
        _run_composite_fallback(plugin_keys, plugin_labels, plugin_selected, disabled_set, categories, console)


def _persist_plugin_selection(plugin_keys, chosen, disabled) -> tuple[bool, set]:
    """Save the composite UI's checkbox state; returns ``(changed, new_enabled)``.

    Unchecked plugins go to the disabled-list (so they stay off even if something auto-enables
    them) under the canonical key ONLY, so the list can't drift from what ``cmd_enable`` clears.
    Re-checking also drops any stale legacy bare-leaf disable.
    """
    # See #40190.
    # Persist by canonical key only — never the bare manifest name — so the disabled-list stays aligned with
    # cmd_enable / PluginManager (#40190).
    new_enabled: set = set()
    new_disabled: set = set(disabled)  # preserve existing disabled state for unseen plugins
    for i, key in enumerate(plugin_keys):
        if i in chosen:
            new_enabled.add(key)
            _discard_key_and_leaf(new_disabled, key)
        else:
            new_disabled.add(key)

    changed = new_enabled != _get_enabled_set() or new_disabled != disabled
    if changed:
        _save_plugin_sets(new_enabled, new_disabled)
    return changed, new_enabled


def _run_composite_ui(curses, plugin_keys, plugin_labels, plugin_selected, disabled, categories, console):
    """Custom curses screen with checkboxes + category action rows."""
    from hermes_cli.curses_ui import _addnstr, flush_stdin
    chosen = set(plugin_selected)
    n_plugins, n_categories = len(plugin_keys), len(categories)
    total_items = n_plugins + n_categories  # navigable rows (headers/separator are skipped)
    providers_changed = False
    nav = {  # key -> new cursor, given (cursor, page_size)
        key: move
        for keys, move in (
            ((curses.KEY_UP, ord("k")), lambda c, p: (c - 1) % total_items),
            ((curses.KEY_DOWN, ord("j")), lambda c, p: (c + 1) % total_items),
            ((curses.KEY_NPAGE, ord("f")), lambda c, p: min(total_items - 1, c + p)),
            ((curses.KEY_PPAGE, ord("b")), lambda c, p: max(0, c - p)),
            ((curses.KEY_HOME,), lambda c, p: 0),
            ((curses.KEY_END,), lambda c, p: total_items - 1),
        )
        for key in keys
    }

    def _init_colors():
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            gray = 8 if curses.COLORS > 8 else curses.COLOR_WHITE
            for pair, fg in ((1, curses.COLOR_GREEN), (2, curses.COLOR_YELLOW), (3, curses.COLOR_CYAN), (4, gray)):
                curses.init_pair(pair, fg, -1)

    def _attr(base, pair):
        return base | curses.color_pair(pair) if curses.has_colors() else base

    def _row(text, idx, cursor, pair):
        """One navigable body row: arrow marker + bold color when *idx* is the cursor."""
        arrow = "\u2192" if idx == cursor else " "
        return (f" {arrow} {text}", _attr(curses.A_BOLD, pair) if idx == cursor else curses.A_NORMAL)

    def _configure_category(ci):
        """Leave curses, run the category's picker, refresh its row, re-enter curses."""
        nonlocal providers_changed
        curses.endwin()
        cat_name, _cat_cur, cat_fn = categories[ci]
        if cat_fn():
            providers_changed = True
            categories[ci] = (cat_name, _provider_categories()[ci][1], cat_fn)
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)
        _init_colors()
        curses.curs_set(0)
        return stdscr

    def _body_lines(cursor, scroll_offset, visible_rows):
        """Body rows as (text, attr); "" is a blank separator."""
        lines = []
        if n_plugins > 0:
            lines.append(("  General Plugins", _attr(curses.A_BOLD, 2)))
            for i in range(scroll_offset, min(n_plugins, scroll_offset + max(visible_rows, 0))):
                check = "\u2713" if i in chosen else " "
                lines.append(_row(f"[{check}] {plugin_labels[i]}", i, cursor, 1))
        lines.append(("", curses.A_NORMAL))
        if n_categories > 0:
            lines.append(("  Provider Plugins", _attr(curses.A_BOLD, 2)))
            lines += [
                _row(f"  {cat_name:<24} \u25b8 {cat_current}", n_plugins + ci, cursor, 3)
                for ci, (cat_name, cat_current, _cat_fn) in enumerate(categories)
            ]
        return lines

    def _draw(stdscr):
        curses.curs_set(0)
        _init_colors()
        cursor = scroll_offset = 0
        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()
            _addnstr(stdscr, 0, 0, "Plugins", max_x - 1, _attr(curses.A_BOLD, 2))
            _addnstr(
                stdscr, 1, 0, "  ↑↓/j/k navigate  PgUp/PgDn page  SPACE toggle  ENTER configure/confirm  ESC done",
                max_x - 1, curses.A_DIM)
            visible_rows = max_y - 4
            if cursor < scroll_offset:
                scroll_offset = cursor
            elif cursor >= scroll_offset + visible_rows:
                scroll_offset = cursor - visible_rows + 1
            lines = _body_lines(cursor, scroll_offset, visible_rows)
            for y, (text, attr) in enumerate(lines[: max(0, max_y - 4)], start=3):
                if text:
                    _addnstr(stdscr, y, 0, text, max_x - 1, attr)
            stdscr.refresh()
            key = stdscr.getch()

            if key in nav:
                if total_items > 0:  # (with no rows, every motion leaves cursor at 0)
                    cursor = nav[key](cursor, max(1, max_y - 5))
            elif key == ord(" ") or key in {curses.KEY_ENTER, 10, 13}:
                if cursor >= n_plugins:
                    # Provider category — launch sub-screen (SPACE and ENTER alike)
                    if cursor - n_plugins < n_categories:
                        stdscr = _configure_category(cursor - n_plugins)
                elif key == ord(" "):
                    chosen.symmetric_difference_update({cursor})
                else:
                    return  # ENTER on a plugin checkbox — confirm and exit
            elif key in {27, ord("q")}:
                return  # plugin changes are saved on exit

    curses.wrapper(_draw)
    flush_stdin()

    changed, new_enabled = _persist_plugin_selection(plugin_keys, chosen, disabled)
    if changed:
        console.print(
            f"\n[green]\u2713[/green] General plugins: {len(new_enabled)} enabled, "
            f"{len(plugin_keys) - len(new_enabled)} disabled.")
    elif n_plugins > 0:
        console.print("\n[dim]General plugins unchanged.[/dim]")
    if providers_changed:
        console.print(
            f"[green]\u2713[/green] Memory provider: [bold]{_get_current_memory_provider() or 'built-in'}[/bold]  "
            f"Context engine: [bold]{_get_current_context_engine()}[/bold]")
    if n_plugins > 0 or providers_changed:
        console.print("[dim]Changes take effect on next session.[/dim]")
    console.print()


def _run_composite_fallback(plugin_keys, plugin_labels, plugin_selected, disabled, categories, console):
    """Text-based fallback for the composite plugins UI."""
    from hermes_cli.colors import Colors, color
    print(color("\n  Plugins", Colors.YELLOW))
    if plugin_keys:
        chosen = set(plugin_selected)
        print(color("\n  General Plugins", Colors.YELLOW))
        print(color("  Toggle by number, Enter to confirm.\n", Colors.DIM))
        while True:
            for i, label in enumerate(plugin_labels):
                marker = color("[\u2713]", Colors.GREEN) if i in chosen else "[ ]"
                print(f"  {marker} {i + 1:>2}. {label}")
            print()
            try:
                val = input(color("  Toggle # (or Enter to confirm): ", Colors.DIM)).strip()
                if not val:
                    break
                idx = int(val) - 1
                if 0 <= idx < len(plugin_keys):
                    chosen.symmetric_difference_update({idx})
            except (ValueError, KeyboardInterrupt, EOFError):
                return
            print()
        _persist_plugin_selection(plugin_keys, chosen, disabled)

    if categories:
        print(color("\n  Provider Plugins", Colors.YELLOW))
        for ci, (cat_name, cat_current, _cat_fn) in enumerate(categories):
            print(f"  {ci + 1}. {cat_name} [{cat_current}]")
        print()
        try:
            val = input(color("  Configure # (or Enter to skip): ", Colors.DIM)).strip()
            if val:
                ci = int(val) - 1
                if 0 <= ci < len(categories):
                    categories[ci][2]()
        except (ValueError, KeyboardInterrupt, EOFError):
            pass
    print()


def dashboard_install_plugin(identifier: str, *, force: bool, enable: bool) -> dict[str, Any]:
    """Non-interactive install for the web dashboard. Returns a JSON-serializable dict."""
    warnings: list[str] = []
    try:
        if _resolve_git_url(identifier)[0].startswith(("http://", "file://")):
            warnings.append("Insecure URL scheme; prefer https:// or git@ for production installs.")
    except ValueError:
        pass
    try:
        target, installed_manifest, installed_name = _install_plugin_core(identifier, force=force)
    except PluginScanBlocked as exc:
        fields = ("pattern_id", "severity", "category", "file", "line", "description")
        return {
            "ok": False, "error": str(exc), "scan_blocked": True,
            "scan_verdict": getattr(exc.scan_result, "verdict", "dangerous"),
            "scan_findings": [
                {k: getattr(f, k) for k in fields}
                for f in (exc.scan_result.findings if exc.scan_result is not None else ())
            ],
        }
    except PluginOperationError as exc:
        return {"ok": False, "error": str(exc)}

    if enable:
        _set_plugin_enabled(installed_name, enable=True)
    ap = target / "after-install.md"
    return {
        "ok": True, "plugin_name": installed_name, "warnings": warnings,
        "missing_env": [s["name"] for s in _missing_env_specs(installed_manifest)],
        "after_install_path": str(ap) if ap.exists() else None, "enabled": enable,
    }


def _get_plugin_toolset_key(name: str) -> Optional[str]:
    """Toolset key a plugin registers its tools under, or None: from the live registry (plugin
    already loaded), else from ``provides_tools`` in plugin.yaml looked up in the registry."""
    try:
        from tools.registry import registry
    except Exception:
        return None

    def _first_toolset(tool_names) -> Optional[str]:
        return next((e.toolset for t in tool_names if (e := registry.get_entry(t)) and e.toolset), None)

    def _from_loaded_plugin() -> Optional[str]:
        from hermes_cli.plugins import discover_plugins, get_plugin_manager
        discover_plugins()  # idempotent — ensures plugins are loaded
        for _key, loaded in get_plugin_manager()._plugins.items():
            if loaded.manifest.name == name or _key == name:
                return _first_toolset(loaded.tools_registered)
        return None

    def _from_manifest_on_disk() -> Optional[str]:
        from hermes_cli.plugins import get_bundled_plugins_dir
        return next((
            toolset for base in (get_bundled_plugins_dir(), _plugins_dir())
            if base.is_dir() and (base / name).is_dir()
            and (toolset := _first_toolset(_read_manifest(base / name).get("provides_tools") or []))
        ), None)

    for lookup in (_from_loaded_plugin, _from_manifest_on_disk):
        try:
            if toolset := lookup():
                return toolset
        except Exception:
            continue
    return None


def _toggle_plugin_toolset(name: str, *, enable: bool) -> None:
    """Add/remove a plugin's toolset in ``platform_toolsets`` for all platforms (no-op when the
    plugin provides no tools)."""
    toolset_key = _get_plugin_toolset_key(name)
    if not toolset_key:
        return
    from hermes_cli.config import load_config, save_config
    config = load_config()
    platform_toolsets = _child_dict(config, "platform_toolsets")
    changed = False
    for ts_list in platform_toolsets.values():
        if isinstance(ts_list, list) and enable != (toolset_key in ts_list):
            (ts_list.append if enable else ts_list.remove)(toolset_key)
            changed = True
    # Enabling with no platform lists yet: seed "cli" at minimum.
    if enable and not changed and not platform_toolsets:
        platform_toolsets["cli"] = [toolset_key]
        changed = True
    if changed:
        save_config(config)


def dashboard_set_agent_plugin_enabled(name: str, *, enabled: bool) -> dict[str, Any]:
    """Enable or disable a plugin in ``config.yaml`` (runtime allow/deny lists)."""
    if _resolve_plugin_key(name) is None:
        return {"ok": False, "error": f"Plugin '{name}' is not installed or bundled."}
    en = _get_enabled_set()
    dis = _get_disabled_set()
    if ((name in en and name not in dis) if enabled else (name not in en and name in dis)):
        return {"ok": True, "name": name, "unchanged": True}
    _set_plugin_enabled(name, enable=enabled)
    _toggle_plugin_toolset(name, enable=enabled)
    return {"ok": True, "name": name, "unchanged": False}


def _user_installed_plugin_dir(name: str) -> Optional[Path]:
    """Resolved path under ``~/.hermes/plugins/<name>`` if it exists."""
    try:
        target = _sanitize_plugin_name(name, _plugins_dir(), allow_subdir=True)
    except ValueError:
        return None
    return target if target.is_dir() else None


def dashboard_update_user_plugin(name: str, *, review_token: Optional[str] = None) -> dict[str, Any]:
    """Stage an update, or accept an exact revision after Dashboard review.

    Rides the SAME staged transaction/accept carrier as ``cmd_update`` (one
    review-token path for CLI + Dashboard — no second pull-first route). Without
    a token the remote is fetched into the private quarantine and the live tree
    stays untouched: the result carries ``review_required`` + ``review_token`` +
    the review payload for the Dashboard surface. With the token the staged
    candidate is revalidated and promoted atomically.
    """
    target = _user_installed_plugin_dir(name)
    if target is None:
        return {"ok": False, "error": f"Plugin '{name}' was not found under {_plugins_dir()}."}
    try:
        if review_token:
            policy = _commit_staged_plugin_update(name, target, review_token)
            if not policy.committed:
                # One surface-neutral policy outcome: a dangerous candidate is
                # refused at commit (scan_blocked) — the live tree stays at the
                # last consented revision and the plugin is NOT promoted.
                return {
                    "ok": False,
                    "name": name,
                    "accepted": False,
                    "error": policy.reason,
                    "scan_blocked": policy.scan_blocked,
                    "scan_verdict": policy.scan_verdict,
                    "scan_findings": policy.scan_findings,
                    "pending_capabilities": policy.pending_capabilities,
                }
            return {
                "ok": True,
                "name": name,
                "accepted": True,
                "unchanged": False,
                "review_required": False,
                "candidate_revision": policy.candidate_revision,
                "candidate_artifact": policy.candidate_artifact,
                "outcome": policy.outcome,
                "scan_verdict": policy.scan_verdict,
                "scan_findings": policy.scan_findings,
                "pending_capabilities": policy.pending_capabilities,
                "capabilities_changed": policy.capabilities_changed,
            }
        staged = _stage_plugin_update(name, target)
        return {
            "ok": True,
            "name": name,
            "output": staged.get("output") or "",
            "unchanged": bool(staged.get("unchanged")),
            "review_required": bool(staged.get("review_required")),
            **{k: staged[k] for k in (
                "review_token", "candidate_revision", "candidate_artifact",
                "changed_files", "td_signature",
            ) if k in staged},
        }
    except PluginConsentDrift as exc:
        return {
            "ok": False,
            "error": f"{exc} Run `hermes plugins update {name}` (interactive) to review the "
            "live tree against its recorded consent.",
        }
    except (PluginOperationError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}


def _run_plugin_git(git_exe: str, target: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run one git command inside a plugin checkout (non-interactive)."""
    return subprocess.run(
        [git_exe, *args], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
        cwd=str(target), stdin=subprocess.DEVNULL, env=noninteractive_git_env())


def dashboard_remove_user_plugin(name: str) -> dict[str, Any]:
    """Delete a plugin tree under ``~/.hermes/plugins/`` only."""
    plugins_dir = _plugins_dir()
    if any(n == name and src == "bundled" for n, _ver, _d, src, _path, _key in _discover_all_plugins()):
        return {"ok": False, "error": "Bundled plugins cannot be removed from the dashboard."}
    target = _user_installed_plugin_dir(name)
    if target is None:
        return {"ok": False, "error": f"Plugin '{name}' was not found under {plugins_dir}."}
    try:
        _remove_plugin_core(target)
    except (OSError, PluginOperationError) as exc:
        return {"ok": False, "error": f"Could not remove plugin '{name}': {exc}"}
    return {"ok": True, "name": name}


def cmd_plugin_doctor(target: str = ".", *, ci: bool = False) -> None:
    """Validate one plugin through runtime discovery and registration."""
    from hermes_cli.plugin_dev import doctor_plugin
    report = doctor_plugin(target)
    _console().print(report.format_text())
    if ci and not report.ok:
        raise SystemExit(1)


def cmd_search(
    term: str = "",
    *,
    json_output: bool = False,
    capability: Optional[str] = None,
    refresh: bool = False,
) -> None:
    """Search the community plugin index (fuzzy on name/description/tags)."""
    from hermes_cli.plugin_index import SECURITY_FOOTER, load_index, search_index
    console = _console()
    entries, source = load_index(refresh=refresh)
    results = search_index(entries, term, capability=capability)
    if json_output:
        print(json.dumps(
            {"source": source, "query": term, "results": [e.to_dict() for e in results], "note": SECURITY_FOOTER},
            indent=2))
        return

    if not results:
        console.print(f"[yellow]No plugins matched '{term}'[/yellow] [dim](index source: {source})[/dim]")
        return

    table = _table(
        (("Name", "bold"), ("Description", None), ("Author", None), ("Tags", "dim")),
        title=f"Community plugins ({len(results)} match{'es' if len(results) != 1 else ''})")
    for e in results:
        desc = e.description if len(e.description) <= 70 else e.description[:67] + "..."
        table.add_row(e.name, desc, e.author, ", ".join(e.tags))
    console.print(table)
    console.print(f"[dim]Index source: {source}. Install: hermes plugins install <name>[/dim]")
    console.print(f"[dim]{SECURITY_FOOTER}[/dim]")


def _tri_state_flag(args, yes_attr: str, no_attr: str) -> Optional[bool]:
    """Map an argparse ``--x`` / ``--no-x`` pair to True / False / None (neither given)."""
    return True if getattr(args, yes_attr, False) else (False if getattr(args, no_attr, False) else None)


def _action_pack(args):
    from hermes_cli.plugin_packs import pack_command
    pack_command(args)


def cmd_compat(args: Any | None = None) -> None:
    """``hermes plugins compat`` — which installed plugins import paths scheduled for removal, and where."""
    import sys
    from pathlib import Path
    from hermes_cli.plugin_compat import (
        ALLOW_KEY, COMPAT_REMOVAL, compat_report, removal_in_effect, scan_plugin, summary_lines)
    console = _console()
    path = getattr(args, "path", None)
    if path:
        hits = scan_plugin(Path(path).expanduser().resolve())
        report = {Path(path).name: hits} if hits else {}
    else:
        report = compat_report(force=True)
    if getattr(args, "json", False):
        print(json.dumps({"removal_date": COMPAT_REMOVAL, "in_effect": removal_in_effect(),
                          "plugins": {k: [h.__dict__ for h in v] for k, v in report.items()}}, indent=2))
        sys.exit(1 if report else 0)
    if not report:
        console.print(f"[green]✓ No enabled plugin imports paths scheduled for removal on {COMPAT_REMOVAL}.[/green]")
        return
    head, tail = summary_lines(report)
    console.print(f"[bold {'red' if removal_in_effect() else 'yellow'}]{head}[/]")
    console.print(f"[dim]{tail}[/dim]")
    for name, hits in sorted(report.items()):
        table = _table(((f"{name}  ({len(hits)} import{'s' if len(hits) != 1 else ''})", "bold"), ("old path", "yellow"), ("new path", "green")),
                       title=None, show_lines=False)
        for h in hits:
            table.add_row(f"{h.file}:{h.line}", h.old, h.new)
        console.print()
        console.print(table)
    console.print()
    console.print(f"[dim]After {COMPAT_REMOVAL} these plugins are not loaded. Update them, or force-load with "
                  f"plugins.{ALLOW_KEY}: true in config.yaml (the old paths still break once the compat layer is reverted).[/dim]")
    sys.exit(1)


# Tri-state flags: neither --x nor --no-x given == None == interactive prompt.
_PLUGIN_ACTIONS = {
    "install": lambda args: cmd_install(
        args.identifier,
        force=getattr(args, "force", False),
        enable=_tri_state_flag(args, "enable", "no_enable"),
        ref=getattr(args, "ref", None)),
    "search": lambda args: cmd_search(
        getattr(args, "term", "") or "",
        json_output=getattr(args, "json", False),
        capability=getattr(args, "capability", None),
        refresh=getattr(args, "refresh", False)),
    "update": lambda args: cmd_update(
        args.name,
        accept_update=getattr(args, "accept_update", False),
        accept_caution=getattr(args, "accept_caution", False)),
    "remove": lambda args: cmd_remove(args.name),
    "rm": lambda args: cmd_remove(args.name),
    "uninstall": lambda args: cmd_remove(args.name),
    "enable": lambda args: cmd_enable(
        args.name,
        allow_tool_override=_tri_state_flag(args, "allow_tool_override", "no_allow_tool_override")),
    "disable": lambda args: cmd_disable(args.name),
    "capabilities": lambda args: cmd_capabilities(getattr(args, "name", None)),
    "list": lambda args: cmd_list(args),
    "ls": lambda args: cmd_list(args),
    "doctor": lambda args: cmd_plugin_doctor(args.target, ci=getattr(args, "ci", False)),
    "compat": lambda args: cmd_compat(args),
    "pack": _action_pack,
    "show": lambda args: cmd_show(args.name),
    "info": lambda args: cmd_show(args.name),
    None: lambda args: cmd_toggle(),
}


def plugins_command(args) -> None:
    """Dispatch hermes plugins subcommands."""
    action = getattr(args, "plugins_action", None)
    handler = _PLUGIN_ACTIONS.get(action)
    if handler is None:
        _fail(_console(), f"[red]Unknown plugins action: {action}[/red]")
    handler(args)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import importlib.metadata  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
