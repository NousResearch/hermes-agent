"""``hermes plugins`` CLI subcommand — install, update, remove, and list plugins.

Facade: shared primitives (errors, console/config helpers, manifest reading, discovery, enable/disable
selection) and the dispatch table live here; each verb family lives in a ``plugins_cmd_<topic>.py``
sibling (install, update, remove, git, capabilities, toggle, listing, catalog)."""

from __future__ import annotations

import functools
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, NoReturn, Optional

from hermes_constants import get_hermes_home
from hermes_cli.config import cfg_get
from hermes_cli.plugin_capabilities import _child_dict
# Tests patch these two on the facade; the install/remove siblings read them through it.
from hermes_cli.secret_prompt import masked_secret_prompt  # noqa: F401
from utils import rmtree_readonly  # noqa: F401

# Topical siblings. The facade re-exports what other modules, tests and the old updater import from
# ``hermes_cli.plugins_cmd``; sibling bodies read those names back through the facade at call time.
from hermes_cli.plugins_cmd_capabilities import (  # noqa: F401
    _declared_capabilities_for_key, _declared_capabilities_from_manifest, _resolve_tool_override_grant,
    _run_capability_consent, cmd_capabilities,
)
from hermes_cli.plugins_cmd_git import (  # noqa: F401
    _EXACT_COMMIT_RE, _canonical_source, _checkout_exact_revision, _clone_plugin_repo, _git_head_revision,
    _git_or_raise, _git_pull_plugin_dir, _git_resolve_commit, _normalize_exact_revision, _pin_annotation,
    _read_install_metadata, _run_plugin_git, _safe_git_error, _scrub_cloned_origin, _update_install_record,
    _write_install_metadata, pinned_revision,
)
from hermes_cli.plugins_cmd_install import (  # noqa: F401
    _check_manifest_version, _consent_python_deps, _display_after_install, _install_plugin_core,
    _install_plugin_python_deps, _prompt_plugin_env_vars, _python_dependency_summary,
    _read_manifest_for_install, cmd_install, dashboard_install_plugin,
)
from hermes_cli.plugins_cmd_listing import (  # noqa: F401
    _filter_plugin_entries, cmd_compat, cmd_list, cmd_show,
)
from hermes_cli.plugins_cmd_remove import (  # noqa: F401
    _remove_plugin_core, cmd_remove, dashboard_remove_user_plugin,
)
from hermes_cli.plugins_cmd_toggle import (  # noqa: F401
    _discover_context_engines, _persist_plugin_selection, _provider_categories, _run_composite_fallback,
    cmd_toggle,
)
from hermes_cli.plugins_cmd_update import (  # noqa: F401
    _clear_plugin_bytecode, cmd_adopt, cmd_check_updates, cmd_trust_update_url, cmd_update,
    dashboard_update_user_plugin,
)

logger = logging.getLogger(__name__)
_DEFAULT_CLONE_TIMEOUT_SECONDS = 300
_MAX_CLONE_TIMEOUT_SECONDS = 3600
_CLONE_TIMEOUT_HINT = "On a slow connection, raise plugins.clone_timeout_seconds in config.yaml."


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


def _fail(console, message: str) -> NoReturn:
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


def _clone_timeout_seconds() -> int:
    """Deadline for plugin clone and pinned fetch, scoped to the active profile."""
    value = _config_value("plugins", "clone_timeout_seconds", default=_DEFAULT_CLONE_TIMEOUT_SECONDS)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        logger.warning("plugins.clone_timeout_seconds must be a positive integer; using %ss",
                       _DEFAULT_CLONE_TIMEOUT_SECONDS)
        return _DEFAULT_CLONE_TIMEOUT_SECONDS
    if value > _MAX_CLONE_TIMEOUT_SECONDS:
        logger.warning("plugins.clone_timeout_seconds exceeds %ss; clamping", _MAX_CLONE_TIMEOUT_SECONDS)
        return _MAX_CLONE_TIMEOUT_SECONDS
    return value


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


def _scan_plugin_tree(plugin_dir: Path, identifier: str, *, force: bool, scan_decision_cb=None,
                      reviewed_pin: bool = False):
    """Scan *plugin_dir* and enforce the install policy.

    Verdicts: safe → proceed; caution → needs confirmation (``force=True`` or a truthy
    ``scan_decision_cb(result)``); dangerous → always blocked (:class:`PluginScanBlocked`).
    *reviewed_pin* marks a tree checked out at a curated-catalog sha: that exact tree passed
    the same scanner at admission with a human reading the caution findings, so caution is
    accepted without a prompt (the Desktop has none). Dangerous still blocks — a signature
    added after review is exactly the case the backstop exists for.
    Returns the ScanResult, or None when scanning is disabled.
    """
    if not _scan_on_install_enabled():
        return None
    from tools.plugin_guard import format_scan_report, scan_plugin, should_allow_plugin_install
    result = scan_plugin(plugin_dir, source=identifier)
    allowed, reason = should_allow_plugin_install(result, force=force)
    if allowed is None and reviewed_pin:
        allowed, reason = True, "Caution verdict accepted: reviewed catalog pin"

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

    # owner/repo[/subdir...] or owner/repo#subdir shorthand (the catalog spells subdirs with ``#``).
    identifier, _, fragment = identifier.partition("#")
    parts = [p for p in identifier.strip("/").split("/") if p]
    if len(parts) >= 2:
        subdir = "/".join([*parts[2:], *fragment.split("/")]).strip("/")
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
    from pm.plugin_declarations import native_manifest_file

    try:
        return native_manifest_file(plugin_dir)
    except ValueError as exc:
        raise PluginOperationError(str(exc)) from exc


def _has_portable_manifest(plugin_dir: Path) -> bool:
    """True when ``plugin.json`` exists (or is a symlink, even dangling) under *plugin_dir*."""
    portable_file = plugin_dir / "plugin.json"
    return portable_file.exists() or portable_file.is_symlink()


def _load_yaml_manifest(manifest_file: Path):
    """``yaml.safe_load`` of *manifest_file* (``{}`` when empty); raises on any read/parse error."""
    from pm.plugin_declarations import read_native_manifest

    return read_native_manifest(manifest_file)


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


def _clone_failure_message(git_url: str, git_error: str) -> str:
    """Plain-words clone failure: what to check (address, network, private repo), raw git text last.

    The text reaches ``_fail`` -> Rich ``console.print``: escape the git output so ``[...]`` in it is
    not parsed as markup."""
    from rich.markup import escape
    return (f"Could not download the plugin from {git_url}. Check the address (browse the catalog "
            "with `hermes plugins search`), check your internet connection, or, if the repository "
            "is private, sign in first with `gh auth login` (or set GITHUB_TOKEN in your .env).\n"
            f"Details: {escape(git_error.strip())}")


def _require_installed_plugin(name: str, plugins_dir: Path, console) -> Path:
    """The plugin path if it exists; else exit 1 (invalid name, or a listing of installed plugins)."""
    try:
        target = _sanitize_plugin_name(name, plugins_dir, allow_subdir=True)
    except ValueError as e:
        _fail(console, f"[red]Error:[/red] {e}")
    if not target.exists():
        _fail(console, _unknown_plugin_message(name, downloaded_only=True))
    return target


def _unknown_plugin_message(name: str, *, downloaded_only: bool = False) -> str:
    """``No plugin named ...`` with the exact-name rule and the two commands that resolve it."""
    scope = (" This command only works on downloaded plugins; bundled ones can only be enabled or disabled."
             if downloaded_only else " Bundled plugins can only be enabled or disabled.")
    return (f"[red]No plugin named '{name}'.[/red] Run `hermes plugins list` to see the exact names "
            f"(nested plugins use their full key, e.g. web/firecrawl).{scope} "
            "To add one: `hermes plugins install <owner/repo>`.")


# ``plugins.disabled`` is an explicit deny-list that wins over the ``plugins.enabled`` allow-list.
_get_disabled_set = functools.partial(_config_name_set, "plugins", "disabled")
_get_enabled_set = functools.partial(_config_name_set, "plugins", "enabled")


def _save_enabled_set(enabled: set) -> None:
    """Frozen old-updater import: never resurrect a raw plugin-selection write."""
    from hermes_cli._old_updater import stop_for_relaunch

    stop_for_relaunch()


def _plugin_selection_version() -> str:
    from hermes_cli.runtime_state import _digest
    return _digest(get_hermes_home() / "config.yaml") or "missing"


def _admit_and_save_plugin_sets(
    enabled: set, disabled: set, *, extra_dirs=(), console=None, action: str = "enable", expected_config=None,
    plugin: Optional[str] = None,
) -> None:
    """ONE admission authority for proposed enabled/disabled sets (C13):
    the candidate union is resolved against the ACTIVE environment and
    the config commits inside the same worker-owned PM transaction — a refusal or a config-write failure
    publishes nothing: previous config bytes AND previous environment
    stay exactly in place. Raises :class:`AdmissionRefused` (UI callers
    catch and surface it — admission never auto-disables to fit); a resolver
    conflict raises its :class:`DependencyConflict` subclass naming *plugin*."""
    from rich.markup import escape

    from hermes_cli.plugins_admission import AdmissionRefused, DependencyConflict, admit_plugin_set_change

    try:
        admit_plugin_set_change(
            enabled, disabled, active_plugins_dir=_plugins_dir(), extra_dirs=extra_dirs, expected_config=expected_config,
            plugin=plugin,
        )
    except DependencyConflict as exc:
        # `hermes pm install` cannot fix a conflict, so the retry hint below would mislead here.
        if console is not None:
            console.print(f"[red]✗[/red] {escape(str(exc))}")
            console.print("[dim]config.yaml and the active environment are unchanged.[/dim]")
        raise
    except AdmissionRefused as exc:
        if console is not None:
            console.print(f"[red]✗[/red] {action} refused: {exc}")
            console.print(
                "[dim]config.yaml and the active environment are unchanged. "
                "Run `hermes pm install` to resolve dependencies, then retry.[/dim]"
            )
        raise



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


def _plugin_aliases(key: str) -> set:
    """Every spelling a config list may hold for *key*: the key, its bare leaf and the manifest name.
    The loader matches BOTH the canonical key (``web/firecrawl``) and the manifest name
    (``web-firecrawl``), so a stale entry under any form vetoes an enable ("explicit disable wins")."""
    names = {key, key.split("/")[-1]}
    names.update(e[0] for e in _discover_all_plugins() if e[5] == key)
    return names


def _activate_key(key: str, *, enable: bool, console=None) -> bool:
    """Transactionally persist canonical *key*, purging aliases from the opposing list.

    False when the lists already say so (nothing written). PM owns the selection and dependency
    publication together, so every CLI/dashboard activation surface goes through the same admission
    transaction instead of writing ``config.yaml`` directly."""
    enabled, disabled = _get_enabled_set(), _get_disabled_set()
    aliases = _plugin_aliases(key)
    target, other = (enabled, disabled) if enable else (disabled, enabled)
    if key in target and not (aliases & other):
        return False
    _set_plugin_enabled(key, enable=enable, aliases=aliases, console=console)
    return True


def _forget_plugin_config(aliases: set) -> dict[str, Any]:
    """Drop every trace of a removed plugin (its :func:`_plugin_aliases`, taken BEFORE the tree went)
    from config.yaml: allow/deny-list entries, ``plugins.entries.<id>`` grants and a ``memory.provider``
    selection naming it. A later reinstall under the same name must start from the "Enable now?"
    decision, not inherit a stale enable or grant (#54336); a dangling ``memory.provider`` would make
    the next agent init re-clone the plugin from the catalog, silently undoing the uninstall.
    Returns ``{"cleared_memory_provider": True}`` when the selection was reset."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    changed = False
    result: dict[str, Any] = {}
    plugins_cfg = config.get("plugins")
    if isinstance(plugins_cfg, dict):
        for list_key in ("enabled", "disabled"):
            names = plugins_cfg.get(list_key)
            if isinstance(names, list) and aliases & set(names):
                plugins_cfg[list_key] = sorted(set(names) - aliases)
                changed = True
        entries = plugins_cfg.get("entries")
        if isinstance(entries, dict) and aliases & set(entries):
            for alias in aliases & set(entries):
                del entries[alias]
            changed = True
    memory_cfg = config.get("memory")
    if isinstance(memory_cfg, dict) and str(memory_cfg.get("provider") or "").strip() in aliases:
        memory_cfg["provider"] = ""
        changed, result = True, {"cleared_memory_provider": True}
    if changed:
        save_config(config)
    return result


def _set_plugin_enabled(name: str, *, enable: bool, aliases=(), console=None) -> None:
    """Submit the command's delta with the version of the selection it read."""
    from pm.plugins_state import read_home_selection

    expected_config = _plugin_selection_version()
    config = read_home_selection(get_hermes_home()) or {}
    plugins = config.get("plugins") or {}
    enabled = set(plugins.get("enabled") or ())
    disabled = set(plugins.get("disabled") or ())
    removed = disabled if enable else enabled
    _discard_key_and_leaf(removed, name)
    removed.difference_update(aliases)
    (enabled if enable else disabled).add(name)
    _admit_and_save_plugin_sets(enabled, disabled, console=console,
                               action=f"{'Enable' if enable else 'Disable'} '{name}'",
                               expected_config=expected_config, plugin=name if enable else None)


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


def cmd_enable(name: str, allow_tool_override: Optional[bool] = None) -> None:
    """Add a plugin to the enabled allow-list (and remove it from disabled).

    Non-bundled plugins request consent for declared capabilities. The legacy
    ``allow_tool_override`` grant changes only with an explicit True/False flag;
    None leaves it unchanged. Bundled plugins are trusted.
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
        _fail(console, _unknown_plugin_message(name))
    key, source = resolved
    _refuse_legacy_relay(key)
    if source != "bundled":
        # Activating recalled code is the same act as installing it (`plugins/AGENTS.md`: kill list).
        from hermes_cli import plugins_cmd_catalog as catalog
        try:
            catalog.refuse_if_installed_removed(key, _user_installed_plugin_dir(key.rsplit("/", 1)[-1]))
        except PluginOperationError as exc:
            _fail(console, f"[red]Error:[/red] {exc}")

    if _activate_key(key, enable=True, console=console):
        from hermes_cli.plugins_activation import activate_plugin_now, activation_hint
        console.print(f"[green]✓[/green] Plugin [bold]{key}[/bold] enabled. Takes effect on next session.")
        console.print(f"[dim]{activation_hint(activate_plugin_now(key, in_process=False))}[/dim]")
    else:
        console.print(f"[dim]Plugin '{key}' is already enabled.[/dim]")

    # Built-in tool override is a privileged grant; bundled plugins are trusted.
    if source == "bundled":
        return
    # When the manifest declares capabilities the consent screen is the canonical grant path
    # (it covers tools.override too); the legacy prompt then only runs on an explicit flag.
    # See #64228.
    declared_caps = _declared_capabilities_for_key(key)
    if declared_caps:
        _run_capability_consent(console, key, declared_caps, context="enable")
    # Enabling a plugin is not a request for undeclared privileges. Keep existing
    # grants unchanged unless the operator explicitly grants or revokes one.
    if allow_tool_override is not None:
        _resolve_tool_override_grant(console, key, allow_tool_override)


def cmd_disable(name: str) -> None:
    """Remove a plugin from the enabled allow-list (and add to disabled)."""
    console = _console()
    key = _resolve_plugin_key(name)
    if key is None:
        _fail(console, _unknown_plugin_message(name))
    if not _activate_key(key, enable=False, console=console):
        console.print(f"[dim]Plugin '{key}' is already disabled.[/dim]")
        return
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
    try:
        children = sorted(base.iterdir())
    except OSError as exc:
        logger.warning("Skipping unreadable plugin directory %s: %s", base, exc)
        return
    for d in children:
        try:
            if not d.is_dir() or (depth == 0 and skip_names and d.name in skip_names):
                continue
            info = _read_manifest_info(d, prefix)
        except (OSError, PluginOperationError) as exc:
            # The PM declaration reader wraps unreadable manifests for install callers;
            # listing still skips them rather than hiding every other plugin.
            logger.warning("Skipping unreadable plugin directory %s: %s", d, exc)
            continue
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
    in ``PluginManager.discover_and_load`` order: bundled, user, then entry points — which never
    displace a directory plugin of the same key (see ``PluginManager._discover_and_load_inner``)."""
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
        seen.setdefault(m.name, (m.name, m.version, m.description, "entrypoint", m.path, m.name))
    return list(seen.values())


def _category_active_names() -> set:
    """Provider names switched on through ``<category>.provider`` config rather than
    ``plugins.enabled`` (the live memory provider), so status never calls them "not enabled"."""
    return {n for n in (_get_current_memory_provider(),) if n}


def _plugin_status(name: str, enabled: set, disabled: set, key: str = "", *, source: str = "",
                   dir_path=None, active: "frozenset | set" = frozenset()) -> str:
    """User-facing activation state for a plugin name or key. Mirrors ``gate_manifest``: an explicit
    disable wins, then the allow-list, then the activations that need no list entry — bundled
    backends/platforms/model providers (*source* + *dir_path*) and category-selected providers
    (*active*, see :func:`_category_active_names`)."""
    names = {name, key}
    if names & disabled:
        return "disabled"
    if names & enabled or names & active:
        return "enabled"
    if source == "bundled" and dir_path is not None and _bundled_default_on(dir_path):
        return "enabled"
    return "not enabled"


# ── Provider category config accessors ──────────────────────────────────────────────────────


# memory.provider ("" = built-in) and context.engine config accessors.
_get_current_memory_provider = functools.partial(_config_str, "memory", "provider", default="")
_get_current_context_engine = functools.partial(_config_str, "context", "engine", default="compressor")
_save_memory_provider = functools.partial(_write_config_value, "memory", "provider")
_save_context_engine = functools.partial(_write_config_value, "context", "engine")


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
    from hermes_cli.toolset_validation import parse_platform_toolsets_value
    config = load_config()
    platform_toolsets = _child_dict(config, "platform_toolsets")
    changed = False
    for platform, raw in list(platform_toolsets.items()):
        # A list-literal string (older `hermes config set`) is the user's real selection; toggling
        # it re-saves the entry as a proper list so the string never persists.
        ts_list = parse_platform_toolsets_value(raw)
        if ts_list is not None and enable != (toolset_key in ts_list):
            (ts_list.append if enable else ts_list.remove)(toolset_key)
            platform_toolsets[platform] = ts_list
            changed = True
    # Enabling with no platform lists yet: seed "cli" at minimum.
    if enable and not changed and not platform_toolsets:
        platform_toolsets["cli"] = [toolset_key]
        changed = True
    if changed:
        save_config(config)


def dashboard_set_agent_plugin_enabled(name: str, *, enabled: bool) -> dict[str, Any]:
    """Enable or disable a plugin in ``config.yaml`` (runtime allow/deny lists). *name* may be the
    canonical key, the manifest name or a unique bare leaf; the canonical key is what gets written
    (the loader never matches a bare leaf, and a stale key in ``disabled`` outranks a manifest-name
    entry in ``enabled`` — so writing the raw identifier reported success while the plugin stayed off)."""
    key = _resolve_plugin_key(name)
    if key is None:
        return {"ok": False, "error": f"Plugin '{name}' is not installed or bundled."}
    from hermes_cli.plugins_admission import AdmissionRefused

    try:
        changed = _activate_key(key, enable=enabled)
    except AdmissionRefused as exc:
        return {
            "ok": False,
            "error": str(exc),
            "name": key,
            "unchanged": True,
            "restart_required": False,
        }
    if changed:
        _toggle_plugin_toolset(key, enable=enabled)
    if changed and enabled:
        # Load it now, here and in the running gateway; ``activation`` tells the UI what is live vs
        # deferred, and ``restart_required`` only survives when no gateway answered (#87770).
        from hermes_cli.plugins_activation import activate_plugin_now
        return {"ok": True, "name": key, "unchanged": False, **activate_plugin_now(key)}
    # Disable is config-only: there is no un-wire primitive, so a running gateway keeps the plugin's
    # handlers until restart and every UI says so — #71595/#54941.
    return {"ok": True, "name": key, "unchanged": not changed, "restart_required": changed}


def _user_installed_plugin_dir(name: str) -> Optional[Path]:
    """Resolved path under ``~/.hermes/plugins/<name>`` if it exists."""
    try:
        target = _sanitize_plugin_name(name, _plugins_dir(), allow_subdir=True)
    except ValueError:
        return None
    return target if target.is_dir() else None


def cmd_plugin_doctor(target: str = ".", *, ci: bool = False) -> None:
    """Validate one plugin through runtime discovery and registration."""
    from hermes_cli.plugin_dev import doctor_plugin
    report = doctor_plugin(target)
    _console().print(report.format_text())
    if ci and not report.ok:
        raise SystemExit(1)


def _tri_state_flag(args, yes_attr: str, no_attr: str) -> Optional[bool]:
    """Map an argparse ``--x`` / ``--no-x`` pair to True / False / None (neither given)."""
    return True if getattr(args, yes_attr, False) else (False if getattr(args, no_attr, False) else None)


def _catalog():
    from hermes_cli import plugins_cmd_catalog
    return plugins_cmd_catalog


def _action_pack(args):
    from hermes_cli.plugin_packs import pack_command
    pack_command(args)


# Tri-state flags: neither --x nor --no-x given == None == interactive prompt.
_PLUGIN_ACTIONS = {
    "install": lambda args: cmd_install(
        args.identifier,
        force=getattr(args, "force", False),
        enable=_tri_state_flag(args, "enable", "no_enable"),
        ref=getattr(args, "ref", None),
        allow_removed=getattr(args, "allow_removed", False),
        no_deps=getattr(args, "no_deps", False),
        yes_deps=getattr(args, "yes_deps", False)),
    "search": lambda args: _catalog().cmd_search(
        getattr(args, "term", "") or "", json_output=getattr(args, "json", False)),
    "browse": lambda args: _catalog().cmd_search(""),
    "validate": lambda args: _catalog().cmd_validate(
        args.path, as_json=getattr(args, "json", False), install_deps=getattr(args, "install_deps", False)),
    "update": lambda args: cmd_update(args.name),
    "adopt": lambda args: cmd_adopt(args.name),
    "trust-update-url": lambda args: cmd_trust_update_url(args.name),
    "check-updates": lambda args: cmd_check_updates(args),
    "check": lambda args: cmd_check_updates(args),
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
    "info": lambda args: _catalog().cmd_info(args.name),
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
