"""``hermes plugins`` CLI subcommand — install, update, remove, and list plugins.

Plugins are installed from Git repositories into ``~/.hermes/plugins/``.
Supports full URLs and ``owner/repo`` shorthand (resolves to GitHub).

After install, if the plugin ships an ``after-install.md`` file it is
rendered with Rich Markdown.  Otherwise a default confirmation is shown.
"""

from __future__ import annotations

import functools
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import noninteractive_git_env
from hermes_cli.config import cfg_get
from hermes_cli.plugin_activation import PluginActivationState
from hermes_cli.secret_prompt import masked_secret_prompt
from utils import env_var_enabled, fast_safe_load

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _resolve_git_executable() -> Optional[str]:
    """Resolve a git binary for subprocess use when ``PATH`` may be minimal.

    Matches other Hermes subprocess resolution: :func:`shutil.which` first,
    then common Git for Windows install paths and POSIX defaults.
    """
    found = shutil.which("git")
    if found:
        return found
    if os.name == "nt":
        prog = os.environ.get("ProgramFiles", r"C:\Program Files")
        prog_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local = os.environ.get("LOCALAPPDATA", "")
        candidates = [
            os.path.join(prog, "Git", "cmd", "git.exe"),
            os.path.join(prog, "Git", "bin", "git.exe"),
            os.path.join(prog_x86, "Git", "cmd", "git.exe"),
            os.path.join(prog_x86, "Git", "bin", "git.exe"),
        ]
        if local:
            candidates.extend(
                (
                    os.path.join(local, "Programs", "Git", "cmd", "git.exe"),
                    os.path.join(local, "Programs", "Git", "bin", "git.exe"),
                )
            )
    else:
        candidates = ["/usr/bin/git", "/usr/local/bin/git", "/bin/git"]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


class PluginOperationError(Exception):
    """Recoverable plugin install/update failure (CLI exits; HTTP maps to 4xx)."""


class PluginActivationConflictError(PluginOperationError):
    """Requested activation cannot be represented without enabling another key."""


_LEGACY_MUTATION_GROUP = object()
_HIDDEN_MUTATION_GROUP = object()
_BASIC_AUTH_MUTATION_GROUP = object()


# Minimum manifest version this installer understands.
# Plugins may declare ``manifest_version: 1`` in plugin.yaml;
# future breaking changes to the manifest schema bump this.
_SUPPORTED_MANIFEST_VERSION = 1


def _plugins_dir() -> Path:
    """Return the user plugins directory, creating it if needed."""
    plugins = get_hermes_home() / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    return plugins


def _sanitize_plugin_name(
    name: str,
    plugins_dir: Path,
    *,
    allow_subdir: bool = False,
) -> Path:
    """Validate a plugin name and return the safe target path inside *plugins_dir*.

    Raises ``ValueError`` if the name contains path-traversal sequences or would
    resolve outside the plugins directory.

    ``allow_subdir=True`` permits a single forward slash inside *name* so
    category-namespaced plugin keys like ``observability/langfuse`` or
    ``image_gen/openai`` (the registry keys emitted by ``_discover_all_plugins``)
    can be looked up. ``..`` and backslash are still rejected, leading and
    trailing slashes are stripped, and the resolved target must still live
    inside *plugins_dir*. Install paths leave this at the default ``False``
    because a freshly-cloned plugin always lands top-level under
    ``~/.hermes/plugins/<name>/``.
    """
    if not name:
        raise ValueError("Plugin name must not be empty.")

    if allow_subdir:
        name = name.strip("/")
        if not name:
            raise ValueError("Plugin name must not be empty.")

    if name in {".", ".."}:
        raise ValueError(
            f"Invalid plugin name '{name}': must not reference the plugins directory itself."
        )

    # Reject obvious traversal characters
    bad_chars = ("\\", "..") if allow_subdir else ("/", "\\", "..")
    for bad in bad_chars:
        if bad in name:
            raise ValueError(f"Invalid plugin name '{name}': must not contain '{bad}'.")

    target = (plugins_dir / name).resolve()
    plugins_resolved = plugins_dir.resolve()

    if target == plugins_resolved:
        raise ValueError(
            f"Invalid plugin name '{name}': resolves to the plugins directory itself."
        )

    try:
        target.relative_to(plugins_resolved)
    except ValueError:
        raise ValueError(
            f"Invalid plugin name '{name}': resolves outside the plugins directory."
        )

    return target


_GITHUB_BROWSER_SEGMENTS = {
    "actions",
    "blob",
    "commit",
    "commits",
    "issues",
    "pull",
    "pulls",
    "releases",
    "tree",
    "wiki",
}


def _resolve_git_url(identifier: str) -> tuple[str, Optional[str]]:
    """Turn an identifier into a cloneable Git URL and optional subdirectory.

    Returns ``(git_url, subdir)`` where ``subdir`` is the path within the
    cloned repository that contains the plugin (``None`` when the plugin lives
    at the repo root).

    Accepted formats:
    - Full URL: https://github.com/owner/repo.git
    - Full URL: git@github.com:owner/repo.git
    - Full URL: ssh://git@github.com/owner/repo.git
    - Browser URL: https://github.com/owner/repo/tree/main/path
      →  (https://github.com/owner/repo.git, "path")
    - Shorthand: owner/repo  →  https://github.com/owner/repo.git
    - Shorthand w/ subdir: owner/repo/path/to/plugin
      →  (https://github.com/owner/repo.git, "path/to/plugin")
    - Full URL w/ subdir (``.git`` boundary):
      https://github.com/owner/repo.git/path/to/plugin
      →  (https://github.com/owner/repo.git, "path/to/plugin")
    - Any URL w/ explicit subdir fragment (works for every scheme, incl.
      ``file://`` and ssh): <url>#path/to/plugin
      →  (<url>, "path/to/plugin")

    NOTE: ``http://`` and ``file://`` schemes are accepted but will trigger a
    security warning at install time.
    """
    # Already a URL.
    if identifier.startswith(("https://", "http://", "git@", "ssh://", "file://")):
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
        marker = ".git/"
        idx = identifier.find(marker)
        if idx != -1:
            git_url = identifier[: idx + len(".git")]
            subdir = identifier[idx + len(marker) :].strip("/")
            return git_url, (subdir or None)
        return identifier, None

    # owner/repo[/subdir...] shorthand
    parts = [p for p in identifier.strip("/").split("/") if p]
    if len(parts) >= 2:
        owner, repo = parts[0], parts[1]
        subdir = "/".join(parts[2:]).strip("/")
        git_url = f"https://github.com/{owner}/{repo}.git"
        return git_url, (subdir or None)

    raise ValueError(
        f"Invalid plugin identifier: '{identifier}'. "
        "Use a Git URL or 'owner/repo' shorthand (optionally with a subdirectory: "
        "'owner/repo/path/to/plugin')."
    )


def _resolve_subdir_within(clone_root: Path, subdir: str) -> Path:
    """Resolve ``subdir`` inside ``clone_root``, rejecting path traversal.

    Guards against ``..`` segments, absolute paths, and symlinks that would
    escape the cloned repository. Returns the resolved directory path.
    Raises ``PluginOperationError`` if the path escapes the clone, doesn't
    exist, or is not a directory.
    """
    clone_root = clone_root.resolve()
    candidate = (clone_root / subdir).resolve()

    # The resolved candidate must stay within the clone root.
    if candidate != clone_root and clone_root not in candidate.parents:
        raise PluginOperationError(
            f"Plugin subdirectory '{subdir}' escapes the repository.",
        )

    if not candidate.exists():
        raise PluginOperationError(
            f"Plugin subdirectory '{subdir}' does not exist in the repository.",
        )
    if not candidate.is_dir():
        raise PluginOperationError(
            f"Plugin subdirectory '{subdir}' is not a directory.",
        )

    return candidate


def _repo_name_from_url(url: str) -> str:
    """Extract the repo name from a Git URL for the plugin directory name."""
    # Strip trailing .git and slashes
    name = url.rstrip("/")
    if name.endswith(".git"):
        name = name[:-4]
    # Get last path component
    name = name.rsplit("/", 1)[-1]
    # Handle ssh-style urls: git@github.com:owner/repo
    if ":" in name:
        name = name.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return name


def _read_manifest(plugin_dir: Path) -> dict:
    """Read plugin.yaml and return the parsed dict, or empty dict."""
    manifest_file = plugin_dir / "plugin.yaml"
    if not manifest_file.exists():
        return {}
    try:
        import yaml

        with open(manifest_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to read plugin.yaml in %s: %s", plugin_dir, e)
        return {}


def _copy_example_files(plugin_dir: Path, console) -> None:
    """Copy any .example files to their real names if they don't already exist.

    For example, ``config.yaml.example`` becomes ``config.yaml``.
    Skips files that already exist to avoid overwriting user config on reinstall.
    """
    for example_file in plugin_dir.glob("*.example"):
        real_name = example_file.stem  # e.g. "config.yaml" from "config.yaml.example"
        real_path = plugin_dir / real_name
        if not real_path.exists():
            try:
                shutil.copy2(example_file, real_path)
                console.print(
                    f"[dim]  Created {real_name} from {example_file.name}[/dim]"
                )
            except OSError as e:
                console.print(
                    f"[yellow]Warning:[/yellow] Failed to copy {example_file.name}: {e}"
                )


def _missing_requires_env_names(manifest: dict) -> list[str]:
    """Return declared ``requires_env`` names that are unset in ``~/.hermes/.env``."""
    requires_env = manifest.get("requires_env") or []
    if not requires_env:
        return []

    from hermes_cli.config import get_env_value

    env_specs: list[dict] = []
    for entry in requires_env:
        if isinstance(entry, str):
            env_specs.append({"name": entry})
        elif isinstance(entry, dict) and entry.get("name"):
            env_specs.append(entry)

    return [s["name"] for s in env_specs if s.get("name") and not get_env_value(s["name"])]


def _prompt_plugin_env_vars(manifest: dict, console) -> None:
    """Prompt for required environment variables declared in plugin.yaml.

    ``requires_env`` accepts two formats:

    Simple list (backwards-compatible)::

        requires_env:
          - MY_API_KEY

    Rich list with metadata::

        requires_env:
          - name: MY_API_KEY
            description: "API key for Acme service"
            url: "https://acme.com/keys"
            secret: true

    Already-set variables are skipped.  Values are saved to the user's ``.env``.
    """
    requires_env = manifest.get("requires_env") or []
    if not requires_env:
        return

    from hermes_cli.config import get_env_value, save_env_value  # noqa: F811
    from hermes_constants import display_hermes_home

    # Normalise to list-of-dicts
    env_specs: list[dict] = []
    for entry in requires_env:
        if isinstance(entry, str):
            env_specs.append({"name": entry})
        elif isinstance(entry, dict) and entry.get("name"):
            env_specs.append(entry)

    # Filter to only vars that aren't already set
    missing = [s for s in env_specs if not get_env_value(s["name"])]
    if not missing:
        return

    plugin_name = manifest.get("name", "this plugin")
    console.print(f"\n[bold]{plugin_name}[/bold] requires the following environment variables:\n")

    for spec in missing:
        name = spec["name"]
        desc = spec.get("description", "")
        url = spec.get("url", "")
        secret = spec.get("secret", False)

        label = f"  {name}"
        if desc:
            label += f" — {desc}"
        console.print(label)
        if url:
            console.print(f"  [dim]Get yours at: {url}[/dim]")

        try:
            if secret:
                value = masked_secret_prompt(f"  {name}: ").strip()
            else:
                value = input(f"  {name}: ").strip()
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
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console()
    after_install = plugin_dir / "after-install.md"

    if after_install.exists():
        content = after_install.read_text(encoding="utf-8")
        md = Markdown(content)
        console.print()
        console.print(Panel(md, border_style="green", expand=False))
        console.print()
    else:
        console.print()
        console.print(
            Panel(
                f"[green bold]Plugin installed:[/] {identifier}\n"
                f"[dim]Location:[/] {plugin_dir}",
                border_style="green",
                title="✓ Installed",
                expand=False,
            )
        )
        console.print()


def _display_removed(name: str, plugins_dir: Path) -> None:
    """Show confirmation after removing a plugin."""
    from rich.console import Console

    console = Console()
    console.print()
    console.print(f"[red]✗[/red] Plugin [bold]{name}[/bold] removed from {plugins_dir}")
    console.print()


def _require_installed_plugin(name: str, plugins_dir: Path, console) -> Path:
    """Return the plugin path if it exists, or exit with an error listing installed plugins."""
    target = _sanitize_plugin_name(name, plugins_dir, allow_subdir=True)
    if not target.exists():
        installed = ", ".join(d.name for d in plugins_dir.iterdir() if d.is_dir()) or "(none)"
        console.print(
            f"[red]Error:[/red] Plugin '{name}' not found in {plugins_dir}.\n"
            f"Installed plugins: {installed}"
        )
        sys.exit(1)
    return target


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _install_plugin_core(identifier: str, *, force: bool) -> tuple[Path, dict, str]:
    """Clone Git plugin into ``~/.hermes/plugins``.

    Returns ``(target_dir, installed_manifest, canonical_name)``.
    Raises ``PluginOperationError`` on failure.
    """
    import tempfile

    try:
        git_url, subdir = _resolve_git_url(identifier)
    except ValueError as e:
        raise PluginOperationError(str(e)) from e

    plugins_dir = _plugins_dir()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_clone = Path(tmp) / "plugin"

        git_exe = _resolve_git_executable()
        if not git_exe:
            raise PluginOperationError("git is not installed or not in PATH.")

        try:
            result = subprocess.run(
                [git_exe, "clone", "--depth", "1", git_url, str(tmp_clone)],
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=60,
                stdin=subprocess.DEVNULL,
                env=noninteractive_git_env(),
            )
        except FileNotFoundError as e:
            raise PluginOperationError(
                "git is not installed or not in PATH.",
            ) from e
        except subprocess.TimeoutExpired as e:
            raise PluginOperationError(
                "Git clone timed out after 60 seconds.",
            ) from e

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            raise PluginOperationError(f"Git clone failed:\n{err}")

        # Resolve the directory within the clone that holds the plugin.
        if subdir:
            tmp_target = _resolve_subdir_within(tmp_clone, subdir)
        else:
            tmp_target = tmp_clone

        manifest = _read_manifest(tmp_target)
        plugin_name = manifest.get("name") or (
            subdir.rstrip("/").rsplit("/", 1)[-1] if subdir else _repo_name_from_url(git_url)
        )

        try:
            target = _sanitize_plugin_name(plugin_name, plugins_dir)
        except ValueError as e:
            raise PluginOperationError(str(e)) from e

        mv = manifest.get("manifest_version")
        if mv is not None:
            try:
                mv_int = int(mv)
            except (ValueError, TypeError):
                raise PluginOperationError(
                    f"Plugin '{plugin_name}' has invalid manifest_version "
                    f"'{mv}' (expected an integer).",
                ) from None
            if mv_int > _SUPPORTED_MANIFEST_VERSION:
                from hermes_cli.config import recommended_update_command

                raise PluginOperationError(
                    f"Plugin '{plugin_name}' requires manifest_version {mv}, "
                    f"but this installer only supports up to {_SUPPORTED_MANIFEST_VERSION}. "
                    f"Run {recommended_update_command()} to update Hermes.",
                ) from None

        if target.exists():
            if not force:
                raise PluginOperationError(
                    f"Plugin '{plugin_name}' already exists. Use force reinstall "
                    f"or run `hermes plugins update {plugin_name}`.",
                )
            shutil.rmtree(target)

        shutil.move(str(tmp_target), str(target))

    has_yaml = (target / "plugin.yaml").exists() or (target / "plugin.yml").exists()
    if not has_yaml and not (target / "__init__.py").exists():
        logger.warning(
            "%s has no plugin.yaml / __init__.py; may not be a valid plugin",
            plugin_name,
        )

    from rich.console import Console

    _copy_example_files(target, Console())
    installed_manifest = _read_manifest(target)
    installed_name = installed_manifest.get("name") or target.name
    return target, installed_manifest, installed_name


def cmd_install(
    identifier: str,
    force: bool = False,
    enable: Optional[bool] = None,
) -> None:
    """Install a plugin from a Git URL or owner/repo shorthand.

    After install, prompt "Enable now? [y/N]" unless *enable* is provided
    (True = auto-enable without prompting, False = install disabled).
    """
    from rich.console import Console

    console = Console()

    try:
        git_url, _subdir = _resolve_git_url(identifier)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if git_url.startswith(("http://", "file://")):
        console.print(
            "[yellow]Warning:[/yellow] Using insecure/local URL scheme. "
            "Consider using https:// or git@ for production installs.",
        )

    if _subdir:
        console.print(f"[dim]Cloning {git_url} (subdir: {_subdir})...[/dim]")
    else:
        console.print(f"[dim]Cloning {git_url}...[/dim]")

    try:
        target, installed_manifest, installed_name = _install_plugin_core(
            identifier,
            force=force,
        )
    except PluginOperationError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if not (target / "plugin.yaml").exists() and not (target / "plugin.yml").exists() and not (
        target / "__init__.py"
    ).exists():
        console.print(
            f"[yellow]Warning:[/yellow] {installed_name} doesn't contain plugin.yaml "
            f"or __init__.py. It may not be a valid Hermes plugin.",
        )

    _prompt_plugin_env_vars(installed_manifest, console)

    _display_after_install(target, identifier)

    should_enable = enable
    if should_enable is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                answer = input(
                    f"  Enable '{installed_name}' now? [y/N]: ",
                ).strip().lower()
                should_enable = answer in {"y", "yes"}
            except (EOFError, KeyboardInterrupt):
                should_enable = False
        else:
            should_enable = False

    if should_enable:
        try:
            _key, _source, _already, _repaired, activation_warning = (
                _enable_plugin_in_config(installed_name)
            )
        except PluginOperationError as exc:
            _invalidate_provider_discovery()
            console.print(
                f"[red]Plugin installed but could not be enabled safely:[/red] {exc}"
            )
            sys.exit(1)
        console.print(
            f"[green]✓[/green] Plugin [bold]{installed_name}[/bold] enabled.",
        )
        if activation_warning:
            console.print(f"[yellow]Warning:[/yellow] {activation_warning}")
    else:
        console.print(
            f"[dim]Plugin installed but not enabled. "
            f"Run `hermes plugins enable {installed_name}` to activate.[/dim]",
        )

    _invalidate_provider_discovery()

    console.print("[dim]Restart the gateway for the plugin to take effect:[/dim]")
    console.print("[dim]  hermes gateway restart[/dim]")
    console.print()


def cmd_update(name: str) -> None:
    """Update an installed plugin by pulling latest from its git remote."""
    from rich.console import Console

    console = Console()
    plugins_dir = _plugins_dir()

    try:
        target = _require_installed_plugin(name, plugins_dir, console)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if not (target / ".git").exists():
        console.print(
            f"[red]Error:[/red] Plugin '{name}' was not installed from git "
            f"(no .git directory). Cannot update."
        )
        sys.exit(1)

    console.print(f"[dim]Updating {name}...[/dim]")

    ok, output = _git_pull_plugin_dir(target)
    if not ok:
        console.print(f"[red]Error:[/red] {output}")
        sys.exit(1)

    # Same stale-bytecode class as the main checkout (#6207/#60242): the
    # pull just changed .py files under this plugin dir, so drop any
    # __pycache__ compiled from the previous revision.
    _clear_plugin_bytecode(target)

    # Copy any new .example files
    _copy_example_files(target, console)
    _invalidate_provider_discovery()

    out = output.strip()
    if "Already up to date" in out:
        console.print(
            f"[green]✓[/green] Plugin [bold]{name}[/bold] is already up to date."
        )
    else:
        console.print(f"[green]✓[/green] Plugin [bold]{name}[/bold] updated.")
        console.print(f"[dim]{out}[/dim]")


def cmd_remove(name: str) -> None:
    """Remove an installed plugin by name."""
    from rich.console import Console

    console = Console()
    plugins_dir = _plugins_dir()

    try:
        target = _require_installed_plugin(name, plugins_dir, console)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    shutil.rmtree(target)
    _invalidate_provider_discovery()
    _display_removed(name, plugins_dir)


def _invalidate_provider_discovery() -> None:
    """Refresh provider-derived surfaces after plugin files or policy change."""
    try:
        from providers import invalidate_provider_discovery

        invalidate_provider_discovery()
    except Exception:
        logger.debug("Provider discovery refresh failed", exc_info=True)


def _get_disabled_set() -> set:
    """Read the disabled plugins set from config.yaml.

    An explicit deny-list. A plugin name here never loads, even if also
    listed in ``plugins.enabled``.
    """
    from hermes_cli.config import load_plugin_activation_state

    return set(load_plugin_activation_state().disabled)


def _save_disabled_set(disabled: set) -> None:
    """Write the disabled plugins list to config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    if "plugins" not in config:
        config["plugins"] = {}
    config["plugins"]["disabled"] = sorted(disabled)
    save_config(config)
    _invalidate_provider_discovery()


_BASIC_AUTH_PLUGIN_NAME = "basic"
_BASIC_AUTH_PLUGIN_KEY = "dashboard_auth/basic"
_BASIC_AUTH_PLUGIN_KEYS = frozenset(
    {_BASIC_AUTH_PLUGIN_NAME, _BASIC_AUTH_PLUGIN_KEY}
)


def ensure_basic_auth_plugin_enabled_in_config(cfg: dict) -> bool:
    """Re-enable the bundled basic dashboard-auth plugin in *cfg*.

    ``hermes setup`` / ``hermes plugins disable basic`` can park the plugin
    in ``plugins.disabled`` while ``dashboard.basic_auth`` is configured.
    The basic provider is a bundled backend that still respects the
    deny-list, so password auth silently fails until the block is removed.

    Returns True when ``plugins.disabled`` was modified.
    """
    plugins_cfg = cfg.get("plugins")
    if not isinstance(plugins_cfg, dict):
        return False
    disabled = plugins_cfg.get("disabled")
    if not isinstance(disabled, list):
        return False
    if not disabled:
        return False
    # Password setup is another activation mutation surface.  Inventory every
    # identity consumer before clearing the bundled provider's deny so a
    # hidden project or legacy provider that shares ``basic`` remains blocked.
    candidates = _strict_plugin_activation_candidates()
    target_indexes = [
        index
        for index, entry in enumerate(candidates)
        if entry[0] == _BASIC_AUTH_PLUGIN_NAME
        and entry[3] == "bundled"
        and entry[5] == _BASIC_AUTH_PLUGIN_KEY
    ]
    if len(target_indexes) != 1:
        raise PluginActivationConflictError(
            "Cannot safely enable the bundled basic auth plugin because its "
            "activation identity could not be verified."
        )

    # This is an implicit enable performed by password setup, not an explicit
    # request to enable every candidate under the canonical key.  Isolate the
    # bundled provider from user/project overrides too: preserve their block
    # with a unique alias, or fail if the identities are indistinguishable.
    target_index = target_indexes[0]
    same_key_indexes = [
        index
        for index, entry in enumerate(candidates)
        if entry[5] == _BASIC_AUTH_PLUGIN_KEY
    ]
    same_key_identities = {
        identity
        for index in same_key_indexes
        for identity in PluginActivationState.identities(
            name=candidates[index][0],
            key=candidates[index][5],
        )
    }
    if not (set(disabled) & same_key_identities):
        return False
    if len(same_key_indexes) != 1:
        # Runtime applies every candidate-level deny as a canonical-key-wide
        # veto.  No replacement deny can keep an override blocked while also
        # loading bundled Basic, so reject instead of activating the override
        # or reporting success while the whole key remains disabled.
        raise PluginActivationConflictError(
            "Cannot safely enable the bundled basic auth plugin because "
            "another plugin candidate uses the same activation key."
        )

    identity_groups: dict[object, set[str]] = {
        _BASIC_AUTH_PLUGIN_KEY: set(_BASIC_AUTH_PLUGIN_KEYS)
    }
    for index, entry in enumerate(candidates):
        if index == target_index:
            continue
        identity_groups[(_BASIC_AUTH_MUTATION_GROUP, index)] = set(
            PluginActivationState.identities(name=entry[0], key=entry[5])
        )
    updated_disabled = set(disabled)
    _clear_plugin_activation_denies(
        updated_disabled,
        key=_BASIC_AUTH_PLUGIN_KEY,
        identity_groups=identity_groups,
    )
    plugins_cfg["disabled"] = sorted(updated_disabled)
    return True


def _get_enabled_set() -> set:
    """Read the enabled plugins allow-list from config.yaml.

    Plugins are opt-in: only names here are loaded. Returns ``set()`` if
    the key is missing (same behaviour as "nothing enabled yet").
    """
    from hermes_cli.config import load_plugin_activation_state

    enabled = load_plugin_activation_state().enabled
    return set(enabled or ())


def _save_enabled_set(enabled: set) -> None:
    """Write the enabled plugins list to config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    if "plugins" not in config:
        config["plugins"] = {}
    config["plugins"]["enabled"] = sorted(enabled)
    save_config(config)


def _resolve_plugin_entry(name: str, entries: list) -> Optional[tuple]:
    """Resolve *name* within ordered entries, preserving key ambiguity rules."""

    def _one_key(matches: list) -> Optional[tuple]:
        if len({entry[5] for entry in matches}) != 1:
            return None
        # Entries are in runtime precedence order. Multiple copies of one
        # canonical key therefore resolve to the prospective highest-priority
        # candidate when callers pass the raw candidate inventory.
        return matches[-1]

    key_match = _one_key([entry for entry in entries if name == entry[5]])
    if key_match is not None:
        return key_match

    manifest_match = _one_key([entry for entry in entries if name == entry[0]])
    if manifest_match is not None:
        return manifest_match

    return _one_key(
        [entry for entry in entries if name == entry[5].split("/")[-1]]
    )


def _resolve_plugin_key(name: str) -> Optional[str]:
    """Resolve a user-supplied plugin identifier to its canonical registry key.

    Accepts either the bare manifest name (``nemo_relay``), the directory
    name, or the full path-derived key (``observability/nemo_relay``) and
    returns the canonical key the loader gates on (``manifest.key`` or, for a
    flat plugin, the bare name). Returns ``None`` when no plugin matches.

    This is the single normalization point so ``hermes plugins enable`` /
    ``disable`` write the same key that ``PluginManager`` matches against —
    nested category plugins (e.g. ``observability/nemo_relay``) included.
    """
    entry = _resolve_plugin_entry(name, _discover_plugin_runtime_candidates())
    return entry[5] if entry is not None else None


def _resolve_plugin_key_and_source(
    name: str,
    *,
    for_enable: bool = False,
) -> Optional[tuple]:
    """Resolve *name* to ``(key, source, manifest_name, kind)`` or ``None``.

    Mirrors :func:`_resolve_plugin_key`'s normalization but also returns the
    plugin's source (``"bundled"``, ``"user"``, ``"project"``, ...) so the
    enable path can tell whether a built-in-override consent prompt is needed.
    Normal management actions resolve the current effective winner. Explicit
    enable instead resolves the prospective highest-precedence candidate: the
    config mutation can then activate an installed external override even when
    an active bundled fallback currently owns the management row.
    """
    entries = (
        _discover_plugin_runtime_candidates(
            include_inactive_project=True,
        )
        if for_enable
        else _discover_all_plugins()
    )
    entry = _resolve_plugin_entry(name, entries)
    if entry is None:
        return None
    if for_enable:
        # A manifest alias can identify a lower-priority bundled fallback even
        # when an external candidate owns the same canonical key.  Enabling is
        # a group-level grant, so always return that key's prospective runtime
        # winner rather than silently mutating consent for the fallback copy.
        entry = next(
            candidate
            for candidate in reversed(entries)
            if candidate[5] == entry[5]
        )
    return entry[5], entry[3], entry[0], entry[6]


def _plugin_runtime_identity_groups(
    candidates: Optional[list] = None,
) -> dict[str, set[str]]:
    """Map each canonical key to every name/key identity runtime checks."""
    entries = (
        _discover_plugin_runtime_candidates()
        if candidates is None
        else candidates
    )
    groups: dict[str, set[str]] = {}
    for entry in entries:
        key = entry[5]
        groups.setdefault(key, {key}).add(entry[0])
    return groups


def _plugin_mutation_identity_groups(candidates: list) -> dict[object, set[str]]:
    """Keep independent legacy providers separate during config mutations."""
    groups: dict[object, set[str]] = {}
    legacy_index = 0
    for entry in candidates:
        key = entry[5]
        if entry[3] == "legacy":
            group_key: object = (_LEGACY_MUTATION_GROUP, legacy_index)
            legacy_index += 1
        else:
            group_key = key
        groups.setdefault(group_key, {key}).add(entry[0])
    return groups


def _plugin_candidate_fingerprint(entry: tuple) -> tuple[str, str, str, str]:
    """Return stable source/path/key/name identity across inventory scans."""
    return entry[3], str(entry[4] or ""), entry[5], entry[0]


def _plugin_composite_identity_groups(
    runtime_candidates: list,
    activation_candidates: list,
) -> dict[object, set[str]]:
    """Keep candidates outside the current runtime scope as hidden groups."""
    groups: dict[object, set[str]] = dict(
        _plugin_runtime_identity_groups(runtime_candidates)
    )
    current = {_plugin_candidate_fingerprint(entry) for entry in runtime_candidates}
    hidden_index = 0
    for entry in activation_candidates:
        if entry[3] != "legacy" and _plugin_candidate_fingerprint(entry) in current:
            continue
        groups[(_HIDDEN_MUTATION_GROUP, hidden_index)] = {entry[5], entry[0]}
        hidden_index += 1
    return groups


def _strict_plugin_activation_candidates() -> list:
    """Return a complete mutation inventory or reject the config write."""
    try:
        return _discover_plugin_activation_candidates(
            include_inactive_project=True,
            strict=True,
        )
    except Exception as exc:
        raise PluginActivationConflictError(
            "Cannot safely update plugin activation because the complete "
            "plugin inventory could not be verified."
        ) from exc


def _nonconflicting_plugin_identity(
    key: object,
    runtime_identities: Iterable[str],
    forbidden: set[str],
) -> Optional[str]:
    """Prefer *key*, then a stable alias that does not affect another group."""
    identities = set(runtime_identities)
    preferred = (key,) if key in identities else ()
    return next(
        (
            identity
            for identity in (*preferred, *sorted(identities - {key}))
            if identity not in forbidden
        ),
        None,
    )


def _clear_plugin_activation_denies(
    disabled: set,
    *,
    key: str,
    identity_groups: dict[object, set[str]],
) -> None:
    """Clear one group's denies without changing any other group's block state.

    A deny identity can be shared by multiple candidates.  When clearing it
    for the target, replace its effect for every other previously blocked
    group.  Replacement identities may only touch groups that were already
    blocked, so preserving one group cannot accidentally disable a third.
    """
    target_runtime_identities = set(identity_groups.get(key, {key}))
    blocked_identities = set(disabled) & target_runtime_identities
    if not blocked_identities:
        return

    previously_unblocked_identities = {
        identity
        for other_key, identities in identity_groups.items()
        if other_key != key and not (set(disabled) & identities)
        for identity in identities
    }

    preserve_identities: set[str] = set()
    for other_key, other_runtime_identities in identity_groups.items():
        if other_key == key:
            continue
        if not (blocked_identities & other_runtime_identities):
            continue
        if (set(disabled) - target_runtime_identities) & other_runtime_identities:
            continue
        replacement = _nonconflicting_plugin_identity(
            other_key,
            other_runtime_identities,
            target_runtime_identities | previously_unblocked_identities,
        )
        if replacement is None:
            # Do not mutate ``disabled`` before this validation completes:
            # callers must be able to reject the enable with zero persistence.
            raise PluginActivationConflictError(
                f"Cannot enable plugin '{key}' without also enabling "
                f"'{other_key}': all runtime activation identities overlap."
            )
        preserve_identities.add(replacement)

    post_cleanup_disabled = (
        set(disabled) - target_runtime_identities
    ) | preserve_identities
    for other_key, other_runtime_identities in identity_groups.items():
        if other_key == key:
            continue
        was_blocked = bool(set(disabled) & other_runtime_identities)
        remains_blocked = bool(post_cleanup_disabled & other_runtime_identities)
        if was_blocked == remains_blocked:
            continue
        raise PluginActivationConflictError(
            f"Cannot enable plugin '{key}' without changing activation for "
            f"'{other_key}'."
        )

    disabled.difference_update(target_runtime_identities)
    disabled.update(preserve_identities)


def _disable_plugin_activation(
    enabled: set[str],
    disabled: set[str],
    *,
    key: str,
    identity_groups: dict[object, set[str]],
) -> None:
    """Disable one canonical group without mutating colliding groups."""
    target_identities = set(identity_groups.get(key, {key}))
    other_identities = {
        identity
        for other_key, identities in identity_groups.items()
        if other_key != key
        for identity in identities
    }
    deny_identity = _nonconflicting_plugin_identity(
        key,
        target_identities,
        other_identities,
    )
    if deny_identity is None:
        raise PluginActivationConflictError(
            f"Cannot disable plugin '{key}' independently: all runtime "
            "activation identities overlap."
        )

    # Shared grants belong to the other group too, so retain them.  The unique
    # deny above still blocks this group, while only target-exclusive grants
    # can be removed without changing another plugin's activation state.
    enabled.difference_update(target_identities - other_identities)
    disabled.add(deny_identity)


def _set_plugin_entry_flag(plugin_id: str, key: str, value: bool) -> None:
    """Write ``plugins.entries.<plugin_id>.<key> = value`` into config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    plugins_cfg = config.setdefault("plugins", {})
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
        config["plugins"] = plugins_cfg
    entries = plugins_cfg.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        plugins_cfg["entries"] = entries
    entry = entries.setdefault(plugin_id, {})
    if not isinstance(entry, dict):
        entry = {}
        entries[plugin_id] = entry
    entry[key] = bool(value)
    save_config(config)


def _plugin_is_enabled_by_default(*, source: str, kind: str) -> bool:
    """Return whether activation policy trusts this source/kind by default."""
    return PluginActivationState().status(
        source=source,
        kind=kind,
    ) == "enabled"


def _clear_default_plugin_activation_grants(
    enabled: set[str],
    *,
    key: str,
    candidates: list[tuple],
) -> bool:
    """Remove stale grants for one default-on bundled plugin.

    Older management paths could persist either a canonical key or a
    manifest name in the source-agnostic allow-list.  A bundled default does
    not need that grant, and retaining it would also authorize a future
    external candidate with the same runtime identity.

    Shared identities are retained whenever any consent-requiring candidate
    consumes them, including candidates under the same canonical key.
    """
    default_state = PluginActivationState()
    target_default_identities: set[str] = set()
    consent_identities: set[str] = set()
    for entry in candidates:
        identities = set(
            PluginActivationState.identities(name=entry[0], key=entry[5])
        )
        enabled_by_default = default_state.status(
            source=entry[3],
            kind=entry[6],
        ) == "enabled"
        if entry[5] == key and enabled_by_default:
            target_default_identities.update(identities)
        if not enabled_by_default:
            consent_identities.update(identities)

    stale_grants = enabled & (target_default_identities - consent_identities)
    if not stale_grants:
        return False

    enabled.difference_update(stale_grants)
    return True


def _apply_plugin_enable(
    *,
    key: str,
    source: str,
    manifest_name: str,
    kind: str,
    enabled: set[str],
    disabled: set[str],
) -> tuple[bool, bool, Optional[str]]:
    """Apply one enable transaction and return (already, repaired, warning)."""
    runtime_candidates = _discover_plugin_runtime_candidates(
        include_inactive_project=True,
    )
    runtime_groups = _plugin_runtime_identity_groups(runtime_candidates)
    runtime_groups.setdefault(key, {key}).add(manifest_name)

    enabled_by_default = _plugin_is_enabled_by_default(
        source=source,
        kind=kind,
    )
    already_enabled = _resolved_plugin_status(
        manifest_name,
        enabled,
        disabled,
        runtime_identities=runtime_groups.setdefault(key, {key}),
        key=key,
        source=source,
        kind=kind,
    ) == "enabled"
    if already_enabled:
        if not enabled_by_default or not (enabled & runtime_groups[key]):
            return True, False, None
        try:
            candidates = _strict_plugin_activation_candidates()
        except PluginActivationConflictError as exc:
            logger.warning("Skipped legacy activation-grant repair for '%s': %s", key, exc)
            return (
                True,
                False,
                "Skipped legacy activation-grant cleanup because the complete "
                "plugin inventory could not be verified.",
            )
        repaired = _clear_default_plugin_activation_grants(
            enabled,
            key=key,
            candidates=candidates,
        )
        return True, repaired, None

    candidates = _strict_plugin_activation_candidates()
    identity_groups = _plugin_mutation_identity_groups(candidates)
    identity_groups.setdefault(key, {key}).add(manifest_name)
    repaired = (
        _clear_default_plugin_activation_grants(
            enabled,
            key=key,
            candidates=candidates,
        )
        if enabled_by_default
        else False
    )

    if not enabled_by_default:
        candidate_identities = set(
            PluginActivationState.identities(name=manifest_name, key=key)
        )
        if not (enabled & candidate_identities):
            other_identities = {
                identity
                for other_key, identities in identity_groups.items()
                if other_key != key
                for identity in identities
            }
            grant_identity = _nonconflicting_plugin_identity(
                key,
                candidate_identities,
                other_identities,
            )
            if grant_identity is None:
                raise PluginActivationConflictError(
                    f"Cannot enable plugin '{key}' independently: all runtime "
                    "activation identities overlap."
                )
            enabled.add(grant_identity)
    _clear_plugin_activation_denies(
        disabled,
        key=key,
        identity_groups=identity_groups,
    )
    return False, repaired, None


def _apply_plugin_disable(
    *,
    key: str,
    source: str,
    manifest_name: str,
    kind: str,
    enabled: set[str],
    disabled: set[str],
) -> bool:
    """Apply one disable transaction and return whether activation changed."""
    runtime_candidates = _discover_plugin_runtime_candidates()
    runtime_groups = _plugin_runtime_identity_groups(runtime_candidates)
    runtime_groups.setdefault(key, {key}).add(manifest_name)
    if _resolved_plugin_status(
        manifest_name,
        enabled,
        disabled,
        runtime_identities=runtime_groups[key],
        key=key,
        source=source,
        kind=kind,
    ) == "disabled":
        return False

    candidates = _strict_plugin_activation_candidates()
    identity_groups = _plugin_mutation_identity_groups(candidates)
    identity_groups.setdefault(key, {key}).add(manifest_name)
    _disable_plugin_activation(
        enabled,
        disabled,
        key=key,
        identity_groups=identity_groups,
    )
    return True


def _enable_plugin_in_config(
    name: str,
) -> tuple[str, str, bool, bool, Optional[str]]:
    """Resolve, safely enable, and persist one plugin activation transaction."""
    resolved = _resolve_plugin_key_and_source(name, for_enable=True)
    if resolved is None:
        raise PluginOperationError(f"Plugin '{name}' is not installed or bundled.")
    key, source, manifest_name, kind = resolved
    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    already_enabled, repaired, warning = _apply_plugin_enable(
        key=key,
        source=source,
        manifest_name=manifest_name,
        kind=kind,
        enabled=enabled,
        disabled=disabled,
    )
    if not already_enabled:
        _save_enabled_set(enabled)
        _save_disabled_set(disabled)
    elif repaired:
        _save_enabled_set(enabled)
    return key, source, already_enabled, repaired, warning


def cmd_enable(name: str, allow_tool_override: Optional[bool] = None) -> None:
    """Add a plugin to the enabled allow-list (and remove it from disabled).

    For non-bundled plugins, prompt the operator about granting the
    privileged ``allow_tool_override`` capability (replacing built-in tools
    like ``shell_exec`` / ``write_file``). ``allow_tool_override`` is a
    tri-state: ``True`` grants without prompting, ``False`` declines without
    prompting, ``None`` (default) asks interactively. Bundled plugins are
    trusted and never prompted.
    """
    from rich.console import Console

    console = Console()
    # Discover the plugin — check installed (user) AND bundled, including
    # nested category plugins — and normalize to its canonical registry key.
    try:
        key, source, already_enabled, _repaired, repair_warning = (
            _enable_plugin_in_config(name)
        )
    except PluginOperationError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    if not already_enabled:
        console.print(
            f"[green]✓[/green] Plugin [bold]{key}[/bold] enabled. "
            "Takes effect on next session."
        )
    else:
        console.print(f"[dim]Plugin '{key}' is already enabled.[/dim]")

    if repair_warning:
        console.print(f"[yellow]Warning:[/yellow] {repair_warning}")

    # Built-in tool override is a privileged grant. Bundled plugins ship with
    # Hermes core and are trusted; every other source needs operator opt-in.
    if source == "bundled":
        return

    _resolve_tool_override_grant(console, key, allow_tool_override)


def _resolve_tool_override_grant(
    console, key: str, allow_tool_override: Optional[bool]
) -> None:
    """Resolve and persist the ``allow_tool_override`` grant for a plugin.

    ``allow_tool_override`` tri-state: True grants, False declines, None
    prompts interactively (defaulting to deny on a non-interactive stdin).
    """
    if allow_tool_override is None:
        # Interactive consent. Default to NO so a blind Enter doesn't grant
        # a privileged capability, and a non-interactive stdin denies safely.
        prompt = (
            "[yellow]Allow this plugin to replace built-in tools "
            "(e.g. shell_exec, write_file)?[/yellow]\n"
            "  This is a privileged capability: an override can intercept "
            "everything the agent routes through that tool.\n"
            "  Grant it? [y/N] "
        )
        try:
            answer = console.input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        allow_tool_override = answer in {"y", "yes"}

    plugin_id = key
    _set_plugin_entry_flag(plugin_id, "allow_tool_override", allow_tool_override)
    if allow_tool_override:
        console.print(
            f"[green]✓[/green] Granted [bold]{key}[/bold] permission to "
            "override built-in tools "
            f"([dim]plugins.entries.{plugin_id}.allow_tool_override: true[/dim])."
        )
    else:
        console.print(
            f"[dim]{key} may not override built-in tools. Re-run "
            f"`hermes plugins enable {key} --allow-tool-override` to grant "
            "this later.[/dim]"
        )


def cmd_disable(name: str) -> None:
    """Remove a plugin from the enabled allow-list (and add to disabled)."""
    from rich.console import Console

    console = Console()
    resolved = _resolve_plugin_key_and_source(name)
    if resolved is None:
        console.print(f"[red]Plugin '{name}' is not installed or bundled.[/red]")
        sys.exit(1)
    key, source, manifest_name, kind = resolved

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    try:
        changed = _apply_plugin_disable(
            key=key,
            source=source,
            manifest_name=manifest_name,
            kind=kind,
            enabled=enabled,
            disabled=disabled,
        )
    except PluginActivationConflictError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    if not changed:
        console.print(f"[dim]Plugin '{key}' is already disabled.[/dim]")
        return

    _save_enabled_set(enabled)
    _save_disabled_set(disabled)
    console.print(
        f"[yellow]\u2298[/yellow] Plugin [bold]{key}[/bold] disabled. "
        "Takes effect on next session."
    )


def _plugin_exists(name: str) -> bool:
    """Return True if a plugin with *name* (bare name or key) exists."""
    return _resolve_plugin_key(name) is not None


def _read_manifest_info(d: Path, prefix: str, *, strict: bool = False):
    """Read manifest metadata used by side-effect-free plugin listing.

    Returns None if no manifest file exists.
    """
    manifest_file = d / "plugin.yaml"
    if not manifest_file.exists():
        manifest_file = d / "plugin.yml"
    if not manifest_file.exists():
        return None
    name = d.name
    version = ""
    description = ""
    kind = "standalone"
    try:
        with open(manifest_file, encoding="utf-8") as manifest_stream:
            manifest = fast_safe_load(manifest_stream) or {}
        if not isinstance(manifest, dict):
            raise ValueError("plugin manifest must be a mapping")
        raw_name = manifest.get("name", d.name)
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("plugin manifest name must be a non-empty string")
        name = raw_name.strip()
        version = manifest.get("version", "")
        description = manifest.get("description", "")
        raw_kind = manifest.get("kind", "standalone")
        if not isinstance(raw_kind, str) or not raw_kind.strip():
            raise ValueError("plugin manifest kind must be a non-empty string")
        kind = raw_kind.strip().lower()
        if kind not in {
            "standalone",
            "backend",
            "exclusive",
            "platform",
            "model-provider",
        }:
            raise ValueError(f"unknown plugin kind: {raw_kind}")
    except Exception:
        if strict:
            raise
    key = f"{prefix}/{d.name}" if prefix else name
    return name, version, description, key, kind


def _scan_level(
    base: Path,
    source: str,
    skip_names: set,
    prefix: str,
    depth: int,
    seen: dict,
    candidates: Optional[list] = None,
    *,
    strict: bool = False,
) -> None:
    """Recursive directory scan matching PluginManager._scan_directory_level.

    Populates *seen* with key ->
    (name, version, description, source, dir, key, kind).
    """
    if not base.is_dir():
        return
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if depth == 0 and skip_names and d.name in skip_names:
            continue
        info = _read_manifest_info(d, prefix, strict=strict)
        if info is not None:
            name, version, description, key, kind = info
            src_label = source
            if source == "user" and (d / ".git").exists():
                src_label = "git"
            entry = (name, version, description, src_label, d, key, kind)
            if candidates is not None:
                candidates.append(entry)
            if key in seen and source == "bundled":
                continue
            seen[key] = entry
            continue
        if depth >= 1:
            continue
        sub_prefix = f"{prefix}/{d.name}" if prefix else d.name
        _scan_level(
            d,
            source,
            set(),
            sub_prefix,
            depth + 1,
            seen,
            candidates,
            strict=strict,
        )


def _discover_plugin_candidates(
    *,
    include_inactive_project: bool = False,
    strict_inventory: bool = False,
) -> list:
    """Return every discoverable plugin candidate in runtime priority order.

    Unlike :func:`_discover_all_plugins`, this retains lower-priority copies
    that share a canonical key. Runtime-adjacent consumers use it to select
    the highest-priority *active* copy instead of letting an installed but
    inactive user/project copy suppress an active bundled fallback.
    """
    seen: dict = {}
    candidates: list = []

    from hermes_cli.plugins import get_bundled_plugins_dir

    repo_plugins = get_bundled_plugins_dir()
    if strict_inventory and not repo_plugins.is_dir():
        raise FileNotFoundError(f"Bundled plugins directory not found: {repo_plugins}")
    _scan_level(
        repo_plugins,
        "bundled",
        {"memory", "context_engine", "platforms"},
        "",
        0,
        seen,
        candidates,
        strict=strict_inventory,
    )
    _scan_level(
        repo_plugins / "platforms",
        "bundled",
        set(),
        "",
        0,
        seen,
        candidates,
        strict=strict_inventory,
    )
    _scan_level(
        _plugins_dir(),
        "user",
        set(),
        "",
        0,
        seen,
        candidates,
        strict=strict_inventory,
    )

    if (
        include_inactive_project
        or env_var_enabled("HERMES_ENABLE_PROJECT_PLUGINS")
    ):
        _scan_level(
            Path.cwd() / ".hermes" / "plugins",
            "project",
            set(),
            "",
            0,
            seen,
            candidates,
            strict=strict_inventory,
        )

    entrypoints = (
        _discover_entrypoint_plugins(strict=True)
        if strict_inventory
        else _discover_entrypoint_plugins()
    )
    for name, version, description, path in entrypoints:
        candidates.append(
            (
                name,
                version,
                description,
                "entrypoint",
                path,
                name,
                "standalone",
            )
        )
    if strict_inventory and not any(
        entry[3] == "bundled"
        and PluginActivationState().status(
            source=entry[3],
            kind=entry[6],
        ) == "enabled"
        for entry in candidates
    ):
        raise RuntimeError("Bundled plugin inventory contains no default-on plugins")
    return candidates


def _sorted_plugin_candidates(candidates: list) -> list:
    """Return candidates in the source precedence used by plugin runtimes."""
    source_rank = {
        "bundled": 0,
        "user": 1,
        "git": 1,
        "project": 2,
        "entrypoint": 3,
        "legacy": 4,
    }
    return sorted(candidates, key=lambda entry: source_rank.get(entry[3], 5))


def _discover_plugin_runtime_candidates(
    *,
    include_inactive_project: bool = False,
    strict: bool = False,
) -> list:
    """Return manifested and manifestless candidates used by plugin runtimes."""
    if include_inactive_project or strict:
        candidates = _discover_plugin_candidates(
            include_inactive_project=include_inactive_project,
            strict_inventory=strict,
        )
    else:
        candidates = _discover_plugin_candidates()

    roots = [(_plugins_dir() / "model-providers", "user")]
    if include_inactive_project or env_var_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
        roots.append(
            (Path.cwd() / ".hermes" / "plugins" / "model-providers", "project")
        )
    for root, source in roots:
        if not root.is_dir():
            continue
        try:
            children = sorted(root.iterdir())
        except OSError:
            if strict:
                raise
            continue
        for child in children:
            if (
                not child.is_dir()
                or child.name.startswith(("_", "."))
                or not (child / "__init__.py").is_file()
                or (child / "plugin.yaml").is_file()
                or (child / "plugin.yml").is_file()
            ):
                continue
            key = f"model-providers/{child.name}"
            source_label = (
                "git" if source == "user" and (child / ".git").exists() else source
            )
            candidates.append((key, "", "", source_label, child, key, "model-provider"))

    return _sorted_plugin_candidates(candidates)


def _discover_plugin_activation_candidates(
    *,
    include_inactive_project: bool = False,
    strict: bool = False,
) -> list:
    """Return every current or compatibility activation-identity consumer.

    Runtime candidates are extended with trusted-installation
    ``providers/*.py`` compatibility modules without importing plugin code.
    Strict mode is used before destructive consent migrations.
    """
    candidates = _discover_plugin_runtime_candidates(
        include_inactive_project=include_inactive_project,
        strict=strict,
    )

    providers_dir = Path(__file__).resolve().parent.parent / "providers"
    try:
        provider_modules = list(providers_dir.glob("*.py"))
        provider_modules.extend(
            child
            for child in providers_dir.iterdir()
            if child.is_dir() and (child / "__init__.py").is_file()
        )
        provider_modules.sort()
    except OSError:
        if strict:
            raise
        provider_modules = []
    for module_path in provider_modules:
        module_name = module_path.name if module_path.is_dir() else module_path.stem
        if module_name == "base" or module_name.startswith("_"):
            continue
        candidates.append(
            (
                module_name,
                "",
                "",
                "legacy",
                module_path,
                f"model-providers/{module_name}",
                "model-provider",
            )
        )

    return _sorted_plugin_candidates(candidates)


def _resolve_plugin_entry_winners(
    entries: list,
    activation: PluginActivationState,
) -> list[tuple]:
    """Return ``(entry, status)`` winners under the canonical runtime policy."""
    from hermes_cli.plugins import resolve_plugin_candidate_winner

    grouped: dict[str, list] = {}
    for entry in entries:
        grouped.setdefault(entry[5], []).append(entry)

    winners: list[tuple] = []
    for candidates in grouped.values():
        selection = resolve_plugin_candidate_winner(
            candidates,
            lambda entry: activation.status(
                name=entry[0],
                key=entry[5],
                source=entry[3],
                kind=entry[6],
            ),
        )
        if selection is not None:
            winners.append(selection)
    return winners


def _select_active_plugin_entries(entries: list, activation: PluginActivationState) -> list:
    """Select the highest-priority active candidate for each canonical key."""
    return [
        entry
        for entry, status in _resolve_plugin_entry_winners(entries, activation)
        if status == "enabled"
    ]


def _discover_all_plugins() -> list:
    """Return the effective runtime/introspection winner for each plugin key.

    Source precedence alone is insufficient: an installed but inactive user or
    project override cannot hide an active bundled fallback.  Resolve the full
    candidate groups through the same canonical helper as ``PluginManager``.
    When a group has no active candidate, retain the helper's inactive winner
    so management surfaces still have one useful row to report and toggle.
    """
    from hermes_cli.config import load_plugin_activation_state

    return [
        entry
        for entry, _status in _resolve_plugin_entry_winners(
            _discover_plugin_runtime_candidates(),
            load_plugin_activation_state(),
        )
    ]


def _discover_entrypoint_plugins(
    *,
    strict: bool = False,
) -> list[tuple[str, str, str, str]]:
    """Return plugin entries advertised through ``hermes_agent.plugins``.

    Entry-point plugins are installed as Python packages, so they do not have a
    plugin directory under ``~/.hermes/plugins``. Include package metadata here
    so ``hermes plugins list`` can show and enable them.
    """
    from hermes_cli.plugins import ENTRY_POINTS_GROUP

    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            group_eps = list(eps.select(group=ENTRY_POINTS_GROUP))
        elif isinstance(eps, dict):
            group_eps = list(eps.get(ENTRY_POINTS_GROUP, []))
        else:
            group_eps = [ep for ep in list(eps) if ep.group == ENTRY_POINTS_GROUP]
    except Exception as exc:
        if strict:
            raise
        logger.debug("Entry-point plugin discovery failed: %s", exc)
        return []

    entries: list[tuple[str, str, str, str]] = []
    for ep in group_eps:
        try:
            name = ep.name
            value = ep.value
        except Exception as exc:
            if strict:
                raise
            logger.debug("Skipping invalid plugin entry point: %s", exc)
            continue

        version = ""
        description = ""
        try:
            dist = getattr(ep, "dist", None)
            metadata = getattr(dist, "metadata", None)
            if metadata is not None:
                version = str(getattr(dist, "version", "") or "")
                description = str(metadata.get("Summary", "") or "")
        except Exception as exc:
            logger.debug("Plugin entry-point metadata unavailable: %s", exc)
        entries.append((name, version, description, value))
    return entries


def _plugin_status(
    name: str,
    enabled: set,
    disabled: set,
    key: str = "",
    *,
    source: str = "",
    kind: str = "standalone",
    aliases: Iterable[str] = (),
) -> str:
    """Return the shared runtime activation state for one plugin."""
    state = PluginActivationState(
        enabled=frozenset(enabled),
        disabled=frozenset(disabled),
        safe_mode=env_var_enabled("HERMES_SAFE_MODE"),
    )
    return state.status(
        name=name,
        key=key,
        source=source,
        kind=kind,
        aliases=aliases,
    )


def _resolved_plugin_status(
    name: str,
    enabled: set,
    disabled: set,
    *,
    runtime_identities: Iterable[str],
    key: str,
    source: str,
    kind: str,
) -> str:
    """Return one candidate's exact status with canonical-group deny semantics."""
    status = _plugin_status(
        name,
        enabled,
        disabled,
        key=key,
        source=source,
        kind=kind,
    )
    if status != "disabled" and set(runtime_identities) & disabled:
        return "disabled"
    return status


def _filter_plugin_entries(
    entries: list,
    args: Any,
    enabled: set,
    disabled: set,
    *,
    identity_groups: Optional[dict[str, set[str]]] = None,
) -> list:
    """Apply ``hermes plugins list`` CLI filters."""
    groups = identity_groups
    if groups is None:
        groups = {}
        for entry in entries:
            groups.setdefault(entry[5], {entry[5]}).add(entry[0])
    filtered = entries
    if getattr(args, "no_bundled", False) or getattr(args, "user", False):
        filtered = [entry for entry in filtered if entry[3] != "bundled"]
    if getattr(args, "enabled", False):
        filtered = [
            entry for entry in filtered
            if _resolved_plugin_status(
                entry[0],
                enabled,
                disabled,
                runtime_identities=groups.get(entry[5], {entry[5], entry[0]}),
                key=entry[5],
                source=entry[3],
                kind=entry[6],
            ) == "enabled"
        ]
    return filtered


def cmd_list(args: Any | None = None) -> None:
    """List all plugins (bundled + user) with enabled/disabled state."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    entries = _discover_all_plugins()
    if not entries:
        console.print("[dim]No plugins installed.[/dim]")
        console.print("[dim]Install with:[/dim] hermes plugins install owner/repo")
        return

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    identity_groups = _plugin_runtime_identity_groups()
    entries = _filter_plugin_entries(
        entries,
        args,
        enabled,
        disabled,
        identity_groups=identity_groups,
    )
    manifest_name_counts: dict[str, int] = {}
    for entry in entries:
        manifest_name_counts[entry[0]] = manifest_name_counts.get(entry[0], 0) + 1

    def _display_name(name: str, key: str) -> str:
        if manifest_name_counts.get(name, 0) > 1:
            return f"{name} [{key}]"
        return name

    def _status_for(name: str, key: str, source: str, kind: str) -> str:
        return _resolved_plugin_status(
            name,
            enabled,
            disabled,
            runtime_identities=identity_groups.get(key, {key, name}),
            key=key,
            source=source,
            kind=kind,
        )

    if getattr(args, "json", False):
        payload = [
            {
                "name": name,
                "key": key,
                "status": _status_for(name, key, source, kind),
                "version": str(version),
                "description": description,
                "source": source,
            }
            for name, version, description, source, _dir, key, kind in entries
        ]
        print(json.dumps(payload, indent=2))
        return

    if getattr(args, "plain", False):
        for name, version, _description, source, _dir, key, kind in entries:
            status = _status_for(name, key, source, kind)
            print(
                f"{status:12} {source:8} {str(version):8} "
                f"{_display_name(name, key)}"
            )
        return

    if not entries:
        console.print("[dim]No plugins matched the selected filters.[/dim]")
        return

    table = Table(title="Plugins", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Version", style="dim")
    table.add_column("Description")
    table.add_column("Source", style="dim")

    for name, version, description, source, _dir, key, kind in entries:
        status_name = _status_for(name, key, source, kind)
        if status_name == "disabled":
            status = "[red]disabled[/red]"
        elif status_name == "enabled":
            status = "[green]enabled[/green]"
        else:
            status = "[yellow]not enabled[/yellow]"
        table.add_row(_display_name(name, key), status, str(version), description, source)

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Compact view:[/dim] hermes plugins list --plain --no-bundled")
    console.print("[dim]Interactive toggle:[/dim] hermes plugins")
    console.print("[dim]Enable/disable:[/dim] hermes plugins enable/disable <name>")
    console.print("[dim]Plugins are opt-in by default — only 'enabled' plugins load.[/dim]")


# ---------------------------------------------------------------------------
# Provider plugin discovery helpers
# ---------------------------------------------------------------------------


def _discover_memory_providers() -> list[tuple[str, str]]:
    """Return [(name, description), ...] for available memory providers."""
    try:
        from plugins.memory import discover_memory_providers
        return [(name, desc) for name, desc, _avail in discover_memory_providers()]
    except Exception:
        return []


def _discover_context_engines() -> list[tuple[str, str]]:
    """Return [(name, description), ...] for available context engines.

    Includes repo-shipped engines from ``plugins/context_engine/`` AND
    plugin-registered engines (third-party engines installed as Hermes
    plugins via ``ctx.register_context_engine``). Repo-shipped descriptions
    win when a plugin-registered engine collides on name.
    """
    engines: list[tuple[str, str]] = []
    seen: set[str] = set()

    try:
        from plugins.context_engine import discover_context_engines
        for name, desc, _avail in discover_context_engines():
            if name not in seen:
                engines.append((name, desc))
                seen.add(name)
    except Exception:
        pass

    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_context_engine
        discover_plugins()
        plugin_engine = get_plugin_context_engine()
        if plugin_engine and getattr(plugin_engine, "name", None) and plugin_engine.name not in seen:
            engines.append((plugin_engine.name, "installed plugin"))
    except Exception:
        pass

    return engines


def _get_current_memory_provider() -> str:
    """Return the current memory.provider from config (empty = built-in)."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return cfg_get(config, "memory", "provider", default="") or ""
    except Exception:
        return ""


def _get_current_context_engine() -> str:
    """Return the current context.engine from config."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return cfg_get(config, "context", "engine", default="compressor") or "compressor"
    except Exception:
        return "compressor"


def _save_memory_provider(name: str) -> None:
    """Persist memory.provider to config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    if "memory" not in config:
        config["memory"] = {}
    config["memory"]["provider"] = name
    save_config(config)


def _save_context_engine(name: str) -> None:
    """Persist context.engine to config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    if "context" not in config:
        config["context"] = {}
    config["context"]["engine"] = name
    save_config(config)


def _configure_memory_provider() -> bool:
    """Launch a radio picker for memory providers. Returns True if changed."""
    from hermes_cli.curses_ui import curses_radiolist

    current = _get_current_memory_provider()
    providers = _discover_memory_providers()

    # Build items: "built-in" first, then discovered providers
    items = ["built-in (default)"]
    names = [""]  # empty string = built-in
    selected = 0

    for name, desc in providers:
        names.append(name)
        label = f"{name} \u2014 {desc}" if desc else name
        items.append(label)
        if name == current:
            selected = len(items) - 1

    # If current provider isn't in discovered list, add it
    if current and current not in names:
        names.append(current)
        items.append(f"{current} (not found)")
        selected = len(items) - 1

    choice = curses_radiolist(
        title="Memory Provider (select one)",
        items=items,
        selected=selected,
    )

    new_provider = names[choice]
    if new_provider != current:
        _save_memory_provider(new_provider)
        return True
    return False


def _configure_context_engine() -> bool:
    """Launch a radio picker for context engines. Returns True if changed."""
    from hermes_cli.curses_ui import curses_radiolist

    current = _get_current_context_engine()
    engines = _discover_context_engines()

    # Build items: "compressor" first (built-in), then discovered engines
    items = ["compressor (default)"]
    names = ["compressor"]
    selected = 0

    for name, desc in engines:
        names.append(name)
        label = f"{name} \u2014 {desc}" if desc else name
        items.append(label)
        if name == current:
            selected = len(items) - 1

    # If current engine isn't in discovered list and isn't compressor, add it
    if current != "compressor" and current not in names:
        names.append(current)
        items.append(f"{current} (not found)")
        selected = len(items) - 1

    choice = curses_radiolist(
        title="Context Engine (select one)",
        items=items,
        selected=selected,
    )

    new_engine = names[choice]
    if new_engine != current:
        _save_context_engine(new_engine)
        return True
    return False


# ---------------------------------------------------------------------------
# Composite plugins UI
# ---------------------------------------------------------------------------


def cmd_toggle() -> None:
    """Interactive composite UI — general plugins + provider plugin categories."""
    from rich.console import Console

    console = Console()

    # -- General plugins discovery (bundled + user) --
    entries = _discover_all_plugins()
    candidates = _discover_plugin_runtime_candidates()
    identity_groups = _plugin_runtime_identity_groups(candidates)
    try:
        activation_candidates = _strict_plugin_activation_candidates()
    except PluginActivationConflictError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    mutation_identity_groups = _plugin_composite_identity_groups(
        candidates,
        activation_candidates,
    )
    enabled_set = _get_enabled_set()
    disabled_set = _get_disabled_set()

    # Track by CANONICAL KEY (``key``), not the manifest name. The loader
    # (PluginManager) and ``cmd_enable``/``cmd_disable`` all gate on the
    # canonical key (``web/firecrawl``), while the manifest name may differ
    # (``web-firecrawl``). Persisting the bare name here caused the two
    # forms to drift: the menu would write ``web-firecrawl`` to
    # plugins.disabled, but ``hermes plugins enable web/firecrawl`` cleared
    # only the key — so "explicit disable wins" kept a bundled backend off
    # forever (pi314's #40190 symptom). Keys keep every surface aligned.
    plugin_keys = []
    plugin_labels = []
    plugin_sources = []
    plugin_kinds = []
    plugin_identities = []
    plugin_grant_identities = []
    plugin_selected = set()

    for i, (name, _version, description, source, _d, key, kind) in enumerate(entries):
        label = f"{name} \u2014 {description}" if description else name
        if source == "bundled":
            label = f"{label} [bundled]"
        plugin_keys.append(key)
        plugin_labels.append(label)
        plugin_sources.append(source)
        plugin_kinds.append(kind)
        runtime_identities = set(identity_groups.get(key, {key}))
        runtime_identities.add(name)
        plugin_identities.append(runtime_identities)
        plugin_grant_identities.append(
            set(PluginActivationState.identities(name=name, key=key))
        )
        # Use the same source/kind policy as runtime discovery. Bundled
        # backends, platforms, and model profiles are on by default; every
        # non-bundled plugin remains opt-in and explicit disable always wins.
        is_on = _resolved_plugin_status(
            name,
            enabled_set,
            disabled_set,
            runtime_identities=plugin_identities[-1],
            key=key,
            source=source,
            kind=kind,
        ) == "enabled"
        if is_on:
            plugin_selected.add(i)

    # -- Provider categories --
    current_memory = _get_current_memory_provider() or "built-in"
    current_context = _get_current_context_engine()
    categories = [
        ("Memory Provider", current_memory, _configure_memory_provider),
        ("Context Engine", current_context, _configure_context_engine),
    ]

    has_plugins = bool(plugin_keys)
    has_categories = bool(categories)

    if not has_plugins and not has_categories:
        console.print("[dim]No plugins installed and no provider categories available.[/dim]")
        console.print("[dim]Install with:[/dim] hermes plugins install owner/repo")
        return

    # Non-TTY fallback
    if not sys.stdin.isatty():
        console.print("[dim]Interactive mode requires a terminal.[/dim]")
        return

    # Launch the composite curses UI
    try:
        import curses
        _run_composite_ui(
            curses,
            plugin_keys,
            plugin_labels,
            plugin_sources,
            plugin_kinds,
            plugin_selected,
            disabled_set,
            categories,
            console,
            plugin_identities=plugin_identities,
            plugin_grant_identities=plugin_grant_identities,
            enabled=enabled_set,
            identity_groups=mutation_identity_groups,
            activation_candidates=activation_candidates,
        )
    except ImportError:
        _run_composite_fallback(
            plugin_keys,
            plugin_labels,
            plugin_sources,
            plugin_kinds,
            plugin_selected,
            disabled_set,
            categories,
            console,
            plugin_identities=plugin_identities,
            plugin_grant_identities=plugin_grant_identities,
            enabled=enabled_set,
            identity_groups=mutation_identity_groups,
            activation_candidates=activation_candidates,
        )


def _run_composite_ui(
    curses,
    plugin_keys,
    plugin_labels,
    plugin_sources,
    plugin_kinds,
    plugin_selected,
    disabled,
    categories,
    console,
    plugin_identities=None,
    plugin_grant_identities=None,
    enabled=None,
    identity_groups=None,
    activation_candidates=None,
):
    """Custom curses screen with checkboxes + category action rows."""
    from hermes_cli.curses_ui import flush_stdin

    chosen = set(plugin_selected)
    n_plugins = len(plugin_keys)
    # Total rows: plugins + separator + categories
    # separator is not navigable
    n_categories = len(categories)
    total_items = n_plugins + n_categories  # navigable items

    result_holder = {"plugins_changed": False, "providers_changed": False}

    def _draw(stdscr):
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_CYAN, -1)
            curses.init_pair(4, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1)  # dim gray
        cursor = 0
        scroll_offset = 0

        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()

            # Header
            try:
                hattr = curses.A_BOLD
                if curses.has_colors():
                    hattr |= curses.color_pair(2)
                stdscr.addnstr(0, 0, "Plugins", max_x - 1, hattr)
                stdscr.addnstr(
                    1, 0,
                    "  ↑↓/j/k navigate  PgUp/PgDn page  SPACE toggle  ENTER configure/confirm  ESC done",
                    max_x - 1, curses.A_DIM,
                )
            except curses.error:
                pass

            # Build display rows
            # Row layout:
            #   [plugins section header] (not navigable, skipped in scroll math)
            #   plugin checkboxes (navigable, indices 0..n_plugins-1)
            #   [separator] (not navigable)
            #   [categories section header] (not navigable)
            #   category action rows (navigable, indices n_plugins..total_items-1)

            visible_rows = max_y - 4
            if cursor < scroll_offset:
                scroll_offset = cursor
            elif cursor >= scroll_offset + visible_rows:
                scroll_offset = cursor - visible_rows + 1

            y = 3  # start drawing after header

            # Determine which items are visible based on scroll
            # We need to map logical cursor positions to screen rows
            # accounting for non-navigable separator/headers


            # --- General Plugins section ---
            if n_plugins > 0:
                # Section header
                if y < max_y - 1:
                    try:
                        sattr = curses.A_BOLD
                        if curses.has_colors():
                            sattr |= curses.color_pair(2)
                        stdscr.addnstr(y, 0, "  General Plugins", max_x - 1, sattr)
                    except curses.error:
                        pass
                    y += 1

                plugin_start = scroll_offset
                plugin_stop = min(n_plugins, scroll_offset + max(visible_rows, 0))
                for i in range(plugin_start, plugin_stop):
                    if y >= max_y - 1:
                        break
                    check = "\u2713" if i in chosen else " "
                    arrow = "\u2192" if i == cursor else " "
                    line = f" {arrow} [{check}] {plugin_labels[i]}"
                    attr = curses.A_NORMAL
                    if i == cursor:
                        attr = curses.A_BOLD
                        if curses.has_colors():
                            attr |= curses.color_pair(1)
                    try:
                        stdscr.addnstr(y, 0, line, max_x - 1, attr)
                    except curses.error:
                        pass
                    y += 1

            # --- Separator ---
            if y < max_y - 1:
                y += 1  # blank line

            # --- Provider Plugins section ---
            if n_categories > 0 and y < max_y - 1:
                try:
                    sattr = curses.A_BOLD
                    if curses.has_colors():
                        sattr |= curses.color_pair(2)
                    stdscr.addnstr(y, 0, "  Provider Plugins", max_x - 1, sattr)
                except curses.error:
                    pass
                y += 1

                for ci, (cat_name, cat_current, _cat_fn) in enumerate(categories):
                    if y >= max_y - 1:
                        break
                    cat_idx = n_plugins + ci
                    arrow = "\u2192" if cat_idx == cursor else " "
                    line = f" {arrow}   {cat_name:<24} \u25b8 {cat_current}"
                    attr = curses.A_NORMAL
                    if cat_idx == cursor:
                        attr = curses.A_BOLD
                        if curses.has_colors():
                            attr |= curses.color_pair(3)
                    try:
                        stdscr.addnstr(y, 0, line, max_x - 1, attr)
                    except curses.error:
                        pass
                    y += 1

            stdscr.refresh()
            key = stdscr.getch()

            if key in {curses.KEY_UP, ord("k")}:
                if total_items > 0:
                    cursor = (cursor - 1) % total_items
            elif key in {curses.KEY_DOWN, ord("j")}:
                if total_items > 0:
                    cursor = (cursor + 1) % total_items
            elif key in {curses.KEY_NPAGE, ord("f")}:
                if total_items > 0:
                    cursor = min(total_items - 1, cursor + max(1, max_y - 5))
            elif key in {curses.KEY_PPAGE, ord("b")}:
                if total_items > 0:
                    cursor = max(0, cursor - max(1, max_y - 5))
            elif key == curses.KEY_HOME:
                cursor = 0
            elif key == curses.KEY_END:
                cursor = max(0, total_items - 1)
            elif key == ord(" "):
                if cursor < n_plugins:
                    # Toggle general plugin
                    chosen.symmetric_difference_update({cursor})
                else:
                    # Provider category — launch sub-screen
                    ci = cursor - n_plugins
                    if 0 <= ci < n_categories:
                        curses.endwin()
                        _cat_name, _cat_cur, cat_fn = categories[ci]
                        changed = cat_fn()
                        if changed:
                            result_holder["providers_changed"] = True
                            # Refresh current values
                            categories[ci] = (
                                _cat_name,
                                _get_current_memory_provider() or "built-in" if ci == 0
                                else _get_current_context_engine(),
                                cat_fn,
                            )
                        # Re-enter curses
                        stdscr = curses.initscr()
                        curses.noecho()
                        curses.cbreak()
                        stdscr.keypad(True)
                        if curses.has_colors():
                            curses.start_color()
                            curses.use_default_colors()
                            curses.init_pair(1, curses.COLOR_GREEN, -1)
                            curses.init_pair(2, curses.COLOR_YELLOW, -1)
                            curses.init_pair(3, curses.COLOR_CYAN, -1)
                            curses.init_pair(4, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1)
                        curses.curs_set(0)
            elif key in {curses.KEY_ENTER, 10, 13}:
                if cursor < n_plugins:
                    # ENTER on a plugin checkbox — confirm and exit
                    result_holder["plugins_changed"] = True
                    return
                else:
                    # ENTER on a category — same as SPACE, launch sub-screen
                    ci = cursor - n_plugins
                    if 0 <= ci < n_categories:
                        curses.endwin()
                        _cat_name, _cat_cur, cat_fn = categories[ci]
                        changed = cat_fn()
                        if changed:
                            result_holder["providers_changed"] = True
                            categories[ci] = (
                                _cat_name,
                                _get_current_memory_provider() or "built-in" if ci == 0
                                else _get_current_context_engine(),
                                cat_fn,
                            )
                        stdscr = curses.initscr()
                        curses.noecho()
                        curses.cbreak()
                        stdscr.keypad(True)
                        if curses.has_colors():
                            curses.start_color()
                            curses.use_default_colors()
                            curses.init_pair(1, curses.COLOR_GREEN, -1)
                            curses.init_pair(2, curses.COLOR_YELLOW, -1)
                            curses.init_pair(3, curses.COLOR_CYAN, -1)
                            curses.init_pair(4, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1)
                        curses.curs_set(0)
            elif key in {27, ord("q")}:
                # Save plugin changes on exit
                result_holder["plugins_changed"] = True
                return

    curses.wrapper(_draw)
    flush_stdin()

    # Persist by canonical key. Unchecked plugins are written to the
    # disabled-list so they stay off even if a future plugin auto-enables
    # itself — but we ONLY ever write the canonical key (never the bare
    # manifest name), so the disabled-list can't drift out of sync with
    # what ``cmd_enable`` clears or what PluginManager gates on (#40190).
    new_enabled, new_disabled = _composite_activation_sets(
        plugin_keys,
        plugin_sources,
        plugin_kinds,
        chosen,
        disabled,
        plugin_identities=plugin_identities,
        plugin_grant_identities=plugin_grant_identities,
        enabled=enabled,
        identity_groups=identity_groups,
        activation_candidates=activation_candidates,
    )

    prev_enabled = _get_enabled_set() if enabled is None else set(enabled)
    enabled_changed = new_enabled != prev_enabled
    disabled_changed = new_disabled != disabled

    if enabled_changed or disabled_changed:
        _save_enabled_set(new_enabled)
        _save_disabled_set(new_disabled)
        console.print(
            f"\n[green]\u2713[/green] General plugins: {len(chosen)} enabled, "
            f"{len(plugin_keys) - len(chosen)} disabled."
        )
    elif n_plugins > 0:
        console.print("\n[dim]General plugins unchanged.[/dim]")

    if result_holder["providers_changed"]:
        new_memory = _get_current_memory_provider() or "built-in"
        new_context = _get_current_context_engine()
        console.print(
            f"[green]\u2713[/green] Memory provider: [bold]{new_memory}[/bold]  "
            f"Context engine: [bold]{new_context}[/bold]"
        )

    if n_plugins > 0 or result_holder["providers_changed"]:
        console.print("[dim]Changes take effect on next session.[/dim]")
    console.print()


def _run_composite_fallback(
    plugin_keys,
    plugin_labels,
    plugin_sources,
    plugin_kinds,
    plugin_selected,
    disabled,
    categories,
    console,
    plugin_identities=None,
    plugin_grant_identities=None,
    enabled=None,
    identity_groups=None,
    activation_candidates=None,
):
    """Text-based fallback for the composite plugins UI."""
    from hermes_cli.colors import Colors, color

    print(color("\n  Plugins", Colors.YELLOW))

    # General plugins
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

        # Persist by canonical key only — never the bare manifest name — so
        # the disabled-list stays aligned with cmd_enable / PluginManager
        # (#40190).
        new_enabled, new_disabled = _composite_activation_sets(
            plugin_keys,
            plugin_sources,
            plugin_kinds,
            chosen,
            disabled,
            plugin_identities=plugin_identities,
            plugin_grant_identities=plugin_grant_identities,
            enabled=enabled,
            identity_groups=identity_groups,
            activation_candidates=activation_candidates,
        )
        prev_enabled = _get_enabled_set() if enabled is None else set(enabled)
        if new_enabled != prev_enabled or new_disabled != disabled:
            _save_enabled_set(new_enabled)
            _save_disabled_set(new_disabled)

    # Provider categories
    if categories:
        print(color("\n  Provider Plugins", Colors.YELLOW))
        for ci, (cat_name, cat_current, cat_fn) in enumerate(categories):
            print(f"  {ci + 1}. {cat_name} [{cat_current}]")
        print()
        try:
            val = input(color("  Configure # (or Enter to skip): ", Colors.DIM)).strip()
            if val:
                ci = int(val) - 1
                if 0 <= ci < len(categories):
                    categories[ci][2]()  # call the configure function
        except (ValueError, KeyboardInterrupt, EOFError):
            pass

    print()


def _composite_activation_sets(
    plugin_keys,
    plugin_sources,
    plugin_kinds,
    chosen,
    disabled,
    *,
    plugin_identities=None,
    plugin_grant_identities=None,
    enabled=None,
    identity_groups=None,
    activation_candidates=None,
):
    """Apply a composite selection without mutating undisplayed groups.

    Bundled backends, platforms, and model providers are active by default, so
    checking one is not an operator grant and must not enter
    ``plugins.enabled``. Bundled standalone plugins and every external plugin
    remain opt-in. Existing grants and explicit-block state for undisplayed
    compatibility providers are preserved exactly.
    """
    if not (len(plugin_keys) == len(plugin_sources) == len(plugin_kinds)):
        raise ValueError("plugin keys, sources, and kinds must have matching lengths")
    if plugin_identities is None:
        plugin_identities = [()] * len(plugin_keys)
    elif len(plugin_identities) != len(plugin_keys):
        raise ValueError("plugin identities and keys must have matching lengths")
    if plugin_grant_identities is None:
        plugin_grant_identities = plugin_identities
    elif len(plugin_grant_identities) != len(plugin_keys):
        raise ValueError("plugin grant identities and keys must have matching lengths")

    runtime_identities_by_row = [
        set(PluginActivationState.identities(key=key, aliases=aliases))
        for key, aliases in zip(plugin_keys, plugin_identities)
    ]
    grant_identities_by_row = [
        set(PluginActivationState.identities(key=key, aliases=aliases))
        for key, aliases in zip(plugin_keys, plugin_grant_identities)
    ]

    groups = {
        group_key: set(identities)
        for group_key, identities in (identity_groups or {}).items()
    }
    for key, identities in zip(plugin_keys, runtime_identities_by_row):
        groups.setdefault(key, {key}).update(identities)

    original_enabled = set(enabled or ())
    original_disabled = set(disabled)
    new_enabled = set(original_enabled)
    new_disabled = set(original_disabled)
    displayed_keys = set(plugin_keys)
    selected_keys = {key for i, key in enumerate(plugin_keys) if i in chosen}
    unselected_keys = displayed_keys - selected_keys
    consent_keys = {
        entry[5]
        for entry in (activation_candidates or ())
        if not _plugin_is_enabled_by_default(source=entry[3], kind=entry[6])
    }
    selected_consent_keys = {
        key
        for i, (key, source, kind) in enumerate(
            zip(plugin_keys, plugin_sources, plugin_kinds)
        )
        if i in chosen
        and (
            key in consent_keys
            or not _plugin_is_enabled_by_default(source=source, kind=kind)
        )
    }
    grant_protected_keys = (set(groups) - displayed_keys) | selected_consent_keys

    try:
        if activation_candidates:
            for i, (key, source, kind) in enumerate(
                zip(plugin_keys, plugin_sources, plugin_kinds)
            ):
                if i in chosen and _plugin_is_enabled_by_default(
                    source=source,
                    kind=kind,
                ):
                    _clear_default_plugin_activation_grants(
                        new_enabled,
                        key=key,
                        candidates=activation_candidates,
                    )

        # Revoke grants consumed only by rows the operator unchecked. Grants
        # shared with a selected or hidden provider remain untouched.
        for identity in tuple(new_enabled):
            owners = {
                group_key
                for group_key, identities in groups.items()
                if identity in identities
            }
            if owners and not (owners & grant_protected_keys):
                new_enabled.discard(identity)

        # A batch may intentionally grant one shared identity when every
        # current consumer is selected. Hidden consumers always make it unsafe.
        forbidden_grant_identities = {
            identity
            for group_key, identities in groups.items()
            if group_key not in selected_keys
            for identity in identities
        }
        for i, (key, source, kind) in enumerate(
            zip(plugin_keys, plugin_sources, plugin_kinds)
        ):
            if i not in chosen or _plugin_is_enabled_by_default(
                source=source,
                kind=kind,
            ):
                continue
            candidate_identities = grant_identities_by_row[i]
            if new_enabled & candidate_identities:
                continue
            grant_identity = _nonconflicting_plugin_identity(
                key,
                candidate_identities,
                {
                    identity
                    for group_key, identities in groups.items()
                    if group_key != key
                    for identity in identities
                },
            )
            if grant_identity is None:
                grant_identity = _nonconflicting_plugin_identity(
                    key,
                    candidate_identities,
                    forbidden_grant_identities,
                )
            if grant_identity is None:
                raise PluginActivationConflictError(
                    f"Cannot enable plugin '{key}' independently from an "
                    "undisplayed or unchecked provider."
                )
            new_enabled.add(grant_identity)

        desired_blocked = {
            group_key: (
                False
                if group_key in selected_keys
                else True
                if group_key in unselected_keys
                else bool(original_disabled & identities)
            )
            for group_key, identities in groups.items()
        }
        desired_unblocked_identities = {
            identity
            for group_key, identities in groups.items()
            if not desired_blocked[group_key]
            for identity in identities
        }
        new_disabled.difference_update(desired_unblocked_identities)

        for group_key, identities in groups.items():
            if not desired_blocked[group_key] or new_disabled & identities:
                continue
            deny_identity = _nonconflicting_plugin_identity(
                group_key,
                identities,
                desired_unblocked_identities,
            )
            if deny_identity is None:
                raise PluginActivationConflictError(
                    f"Cannot disable plugin '{group_key}' without also "
                    "disabling a selected or previously unblocked provider."
                )
            new_disabled.add(deny_identity)

        for group_key, identities in groups.items():
            if bool(new_disabled & identities) != desired_blocked[group_key]:
                raise PluginActivationConflictError(
                    f"Cannot preserve activation state for '{group_key}'."
                )
    except PluginActivationConflictError as exc:
        logger.warning(
            "Cannot apply composite plugin selection independently: %s; "
            "keeping the existing activation policy fail-closed",
            exc,
        )
        return original_enabled, original_disabled

    return new_enabled, new_disabled


def dashboard_install_plugin(
    identifier: str,
    *,
    force: bool,
    enable: bool,
) -> dict[str, Any]:
    """Non-interactive install for the web dashboard. Returns a JSON-serializable dict."""
    warnings: list[str] = []
    try:
        git_url, _subdir = _resolve_git_url(identifier)
        if git_url.startswith(("http://", "file://")):
            warnings.append(
                "Insecure URL scheme; prefer https:// or git@ for production installs.",
            )
    except ValueError:
        pass

    try:
        target, installed_manifest, installed_name = _install_plugin_core(
            identifier,
            force=force,
        )
    except PluginOperationError as exc:
        return {"ok": False, "error": str(exc)}

    missing_env = _missing_requires_env_names(installed_manifest)
    enabled_after_install = False
    if enable:
        try:
            _key, _source, _already, _repaired, activation_warning = (
                _enable_plugin_in_config(installed_name)
            )
        except PluginOperationError as exc:
            warnings.append(f"Plugin installed but could not be enabled safely: {exc}")
        else:
            enabled_after_install = True
            if activation_warning:
                warnings.append(activation_warning)
    _invalidate_provider_discovery()

    hint: str | None = None
    ap = target / "after-install.md"
    if ap.exists():
        hint = str(ap)

    return {
        "ok": True,
        "plugin_name": installed_name,
        "enabled": enabled_after_install,
        "warnings": warnings,
        "missing_env": missing_env,
        "after_install_path": hint,
    }


def _get_plugin_toolset_key(name: str) -> Optional[str]:
    """Return the toolset key a plugin registers its tools under, or None.

    Queries the live tool registry — the plugin must already be loaded.
    Falls back to reading ``provides_tools`` from plugin.yaml and looking
    up the toolset from the registry for the first tool name found.
    """
    try:
        from tools.registry import registry
    except Exception:
        return None

    # Check the plugin manager for tools this plugin registered
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_manager
        discover_plugins()  # idempotent — ensures plugins are loaded
        manager = get_plugin_manager()
        for _key, loaded in manager._plugins.items():
            if loaded.manifest.name == name or _key == name:
                for tool_name in loaded.tools_registered:
                    entry = registry.get_entry(tool_name)
                    if entry and entry.toolset:
                        return entry.toolset
                break
    except Exception:
        pass

    # Fallback: read provides_tools from manifest on disk and query registry
    try:
        from hermes_cli.plugins import get_bundled_plugins_dir
        for base in (get_bundled_plugins_dir(), _plugins_dir()):
            if not base.is_dir():
                continue
            candidate = base / name
            if candidate.is_dir():
                manifest = _read_manifest(candidate)
                for tool_name in manifest.get("provides_tools") or []:
                    entry = registry.get_entry(tool_name)
                    if entry and entry.toolset:
                        return entry.toolset
    except Exception:
        pass

    return None


def _toggle_plugin_toolset(name: str, *, enable: bool) -> None:
    """Add or remove a plugin's toolset from platform_toolsets for all platforms.

    Only acts if the plugin actually provides tools (has a toolset key).
    """
    toolset_key = _get_plugin_toolset_key(name)
    if not toolset_key:
        return

    from hermes_cli.config import load_config, save_config

    config = load_config()
    platform_toolsets = config.get("platform_toolsets")
    if not isinstance(platform_toolsets, dict):
        platform_toolsets = {}
        config["platform_toolsets"] = platform_toolsets

    changed = False
    for platform, ts_list in platform_toolsets.items():
        if not isinstance(ts_list, list):
            continue
        if enable:
            if toolset_key not in ts_list:
                ts_list.append(toolset_key)
                changed = True
        elif toolset_key in ts_list:
            ts_list.remove(toolset_key)
            changed = True

    # If enabling and no platforms have toolset lists yet, add to "cli" at minimum
    if enable and not changed and not platform_toolsets:
        platform_toolsets["cli"] = [toolset_key]
        changed = True

    if changed:
        save_config(config)


def dashboard_set_agent_plugin_enabled(name: str, *, enabled: bool) -> dict[str, Any]:
    """Enable or disable a plugin in ``config.yaml`` (runtime allow/deny lists).

    For plugins that provide tools (toolsets), also toggles the toolset in
    ``platform_toolsets`` so the agent actually sees the tools in sessions.
    """
    resolved = _resolve_plugin_key_and_source(name, for_enable=enabled)
    if resolved is None:
        return {"ok": False, "error": f"Plugin '{name}' is not installed or bundled."}
    key, source, manifest_name, kind = resolved

    en = _get_enabled_set()
    dis = _get_disabled_set()
    if enabled:
        try:
            (
                already_enabled,
                repaired_default_grants,
                repair_warning,
            ) = _apply_plugin_enable(
                key=key,
                source=source,
                manifest_name=manifest_name,
                kind=kind,
                enabled=en,
                disabled=dis,
            )
        except PluginActivationConflictError as exc:
            return {
                "ok": False,
                "name": name,
                "key": key,
                "error": str(exc),
            }
        if already_enabled:
            if repaired_default_grants:
                _save_enabled_set(en)
        else:
            _save_enabled_set(en)
            _save_disabled_set(dis)
            _toggle_plugin_toolset(key, enable=True)
        result = {
            "ok": True,
            "name": name,
            "key": key,
            "unchanged": already_enabled and not repaired_default_grants,
        }
        if repair_warning:
            result["warning"] = repair_warning
        return result

    try:
        changed = _apply_plugin_disable(
            key=key,
            source=source,
            manifest_name=manifest_name,
            kind=kind,
            enabled=en,
            disabled=dis,
        )
    except PluginActivationConflictError as exc:
        return {
            "ok": False,
            "name": name,
            "key": key,
            "error": str(exc),
        }
    if not changed:
        return {"ok": True, "name": name, "key": key, "unchanged": True}

    _save_enabled_set(en)
    _save_disabled_set(dis)
    _toggle_plugin_toolset(key, enable=False)
    return {"ok": True, "name": name, "key": key, "unchanged": False}


def _user_installed_plugin_dir(name: str) -> Optional[Path]:
    """Resolve a plugin identifier to its installed user-tree directory."""
    plugins_dir = _plugins_dir()
    try:
        plugins_root = plugins_dir.resolve()
    except (OSError, RuntimeError):
        return None

    key = _resolve_plugin_key(name)
    if key is not None:
        for entry in reversed(_discover_plugin_runtime_candidates()):
            if entry[5] != key or entry[3] not in {"user", "git"}:
                continue
            try:
                target = Path(entry[4]).resolve()
                target.relative_to(plugins_root)
            except (OSError, RuntimeError, TypeError, ValueError):
                return None
            return target if target.is_dir() else None

    # Compatibility for old clients that send a directory name rather than a
    # canonical key (including an ambiguous manifest-name collision).
    try:
        target = _sanitize_plugin_name(name, plugins_dir, allow_subdir=True)
    except ValueError:
        return None
    return target if target.is_dir() else None


def dashboard_update_user_plugin(name: str) -> dict[str, Any]:
    """``git pull`` inside ``~/.hermes/plugins/<name>``."""
    target = _user_installed_plugin_dir(name)
    if target is None:
        return {
            "ok": False,
            "error": f"Plugin '{name}' was not found under {_plugins_dir()}.",
        }

    if not (target / ".git").exists():
        return {
            "ok": False,
            "error": f"Plugin '{name}' is not a git checkout; cannot pull updates.",
        }

    ok, msg = _git_pull_plugin_dir(target)
    if not ok:
        return {"ok": False, "error": msg}

    # Sibling of the CLI ``hermes plugins update`` path: drop bytecode
    # compiled from the pre-pull plugin revision.
    _clear_plugin_bytecode(target)

    from rich.console import Console

    _copy_example_files(target, Console())
    _invalidate_provider_discovery()
    unchanged = "Already up to date" in msg
    return {"ok": True, "name": name, "output": msg, "unchanged": unchanged}


def _clear_plugin_bytecode(target: Path) -> int:
    """Remove ``__pycache__`` dirs under a just-updated plugin checkout.

    Plugin dirs live outside the main repo, so the launch-time checkout
    fingerprint sweep in ``hermes_cli.main`` never covers them. After a
    ``git pull`` changes a plugin's ``.py`` files, stale bytecode here can
    produce the same ImportError class as #6207/#60242 in whichever
    process imports the plugin next. Never raises.
    """
    removed = 0
    try:
        for cache_dir in target.rglob("__pycache__"):
            if not cache_dir.is_dir():
                continue
            try:
                shutil.rmtree(cache_dir)
                removed += 1
            except OSError:
                pass
    except OSError:
        pass
    return removed


def _git_pull_plugin_dir(target: Path) -> tuple[bool, str]:
    git_exe = _resolve_git_executable()
    if not git_exe:
        return False, "git is not installed or not in PATH."
    try:
        result = subprocess.run(
            [git_exe, "pull", "--ff-only"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=60,
            cwd=str(target),
            stdin=subprocess.DEVNULL,
            env=noninteractive_git_env(),
        )
    except FileNotFoundError:
        return False, "git is not installed or not in PATH."
    except subprocess.TimeoutExpired:
        return False, "Git pull timed out after 60 seconds."

    if result.returncode != 0:
        err = (result.stderr or "").strip() or result.stdout.strip()
        return False, err or "git pull failed."
    return True, result.stdout.strip()


def dashboard_remove_user_plugin(name: str) -> dict[str, Any]:
    """Delete a plugin tree under ``~/.hermes/plugins/`` only."""
    plugins_dir = _plugins_dir()
    resolved = _resolve_plugin_key_and_source(name)
    if resolved is not None and resolved[1] == "bundled":
        return {
            "ok": False,
            "error": "Bundled plugins cannot be removed from the dashboard.",
        }

    target = _user_installed_plugin_dir(name)
    if target is None:
        return {
            "ok": False,
            "error": f"Plugin '{name}' was not found under {plugins_dir}.",
        }

    shutil.rmtree(target)
    _invalidate_provider_discovery()
    return {"ok": True, "name": name}


def plugins_command(args) -> None:
    """Dispatch hermes plugins subcommands."""
    action = getattr(args, "plugins_action", None)

    if action == "install":
        # Map argparse tri-state: --enable=True, --no-enable=False, neither=None (prompt)
        enable_arg = None
        if getattr(args, "enable", False):
            enable_arg = True
        elif getattr(args, "no_enable", False):
            enable_arg = False
        cmd_install(
            args.identifier,
            force=getattr(args, "force", False),
            enable=enable_arg,
        )
    elif action == "update":
        cmd_update(args.name)
    elif action in {"remove", "rm", "uninstall"}:
        cmd_remove(args.name)
    elif action == "enable":
        # Tri-state: --allow-tool-override=True, --no-allow-tool-override=False,
        # neither=None (interactive prompt for non-bundled plugins).
        allow_override = None
        if getattr(args, "allow_tool_override", False):
            allow_override = True
        elif getattr(args, "no_allow_tool_override", False):
            allow_override = False
        cmd_enable(args.name, allow_tool_override=allow_override)
    elif action == "disable":
        cmd_disable(args.name)
    elif action in {"list", "ls"}:
        cmd_list(args)
    elif action is None:
        cmd_toggle()
    else:
        from rich.console import Console

        Console().print(f"[red]Unknown plugins action: {action}[/red]")
        sys.exit(1)
