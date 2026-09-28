"""hermes.feature_disabled.count: a user turns off something Hermes ships on, or removes something shipped.

One chokepoint sees every write: ``save_config`` hands over the raw config it replaced and the one it
wrote, and :func:`config_transitions` diffs only what the user moved away from the shipped default
(or back to it): default-on toolsets, ``skills.disabled``, ``plugins.disabled`` and default-``True``
booleans. Names are public only when shipped (toolset key, bundled/catalog skill, bundled/catalog
plugin, ``DEFAULT_CONFIG`` key path; never a value); anything else reads ``custom``. The store counts
each (kind, name, event) at most once per day.

The surface is the entry point that is running: ``set_process_surface`` from the ``hermes`` command
dispatch (``tools`` / ``config`` / ``skills`` / ``plugins`` / chat slash commands / the web server) and
the TUI gateway. A process with no surface (setup wizard, updates, migrations) records nothing: those
writes are Hermes applying choices, not a user turning something off.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

CUSTOM = "custom"
_COMMAND_SURFACES = {
    None: "cli_slash", "chat": "cli_slash", "tools": "cli_tools", "config": "cli_config",
    "skills": "cli_config", "plugins": "cli_config",
    "dashboard": "web", "serve": "web", "gui": "web", "desktop": "web",
}
_SETTING_KINDS = {"memory": "memory", "curator": "curator", "compression": "compression"}
_OFF = frozenset({"false", "off", "no", "0", "none", ""})
_process_surface: str | None = None


def set_process_surface(command: Any) -> None:
    """Called once by the ``hermes`` dispatch (``command`` is the subcommand, None for chat) or the
    TUI gateway (``"tui_gateway"``). Unknown commands clear it."""
    global _process_surface
    _process_surface = "tui_gateway" if command == "tui_gateway" else _COMMAND_SURFACES.get(command)


def current_surface() -> str | None:
    surface = _process_surface
    if surface == "web":
        from .shared_metrics_setup import web_setup_surface

        return web_setup_surface()
    if surface == "tui_gateway":
        try:
            from tui_gateway.server import _resolve_session_platform

            return _resolve_session_platform()
        except Exception:
            return "tui"
    return surface


# ---- names -----------------------------------------------------------------------------------

@functools.cache
def _bundled_plugin_names() -> frozenset[str]:
    from .shared_metrics_catalog import _REPO_ROOT

    root = _REPO_ROOT / "plugins"
    try:
        return frozenset(
            p.parent.name.lower() for p in [*root.glob("*/plugin.yaml"), *root.glob("*/*/plugin.yaml")]
        )
    except OSError:
        return frozenset()


def plugin_metric_name(raw: Any) -> str:
    from .shared_metrics_catalog import _norm, _safe, plugin_catalog_names

    name = _norm(raw).rsplit("/", 1)[-1]
    return name if name and (name in _safe(_bundled_plugin_names) or name in _safe(plugin_catalog_names)) else CUSTOM


def skill_metric_name(raw: Any) -> str:
    from .shared_metrics_catalog import skill_metric_name as catalog_skill_name

    return catalog_skill_name(raw)


@functools.cache
def default_on_settings() -> frozenset[str]:
    """Dotted ``DEFAULT_CONFIG`` paths whose shipped value is ``True`` (the closed setting vocabulary)."""
    from hermes_cli.config import DEFAULT_CONFIG

    from .shared_metrics_contract import FEATURE_DISABLED_NAME_MAX_LENGTH, _metric_identifier

    paths: set[str] = set()

    def walk(node: Any, prefix: str) -> None:
        for key, value in node.items():
            if not isinstance(key, str) or key.startswith("_"):
                continue
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                walk(value, f"{path}.")
            elif value is True and _metric_identifier(path, max_length=FEATURE_DISABLED_NAME_MAX_LENGTH) == path:
                paths.add(path)

    walk(DEFAULT_CONFIG, "")
    return frozenset(paths)


# ---- transitions -----------------------------------------------------------------------------

def _get(config: Any, path: str) -> Any:
    node = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return True  # absent = the shipped default (True for every tracked path)
        node = node[part]
    return node


def _on(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in _OFF
    return bool(value)


def _setting_transitions(old: dict, new: dict) -> Iterable[tuple[str, str, str]]:
    for path in sorted(default_on_settings()):
        before, after = _on(_get(old, path)), _on(_get(new, path))
        if before != after:
            yield _SETTING_KINDS.get(path.split(".", 1)[0], "setting"), path, "re_enabled" if after else "disabled"


def _names(config: Any, section: str, key: str, *, per_platform: str | None = None) -> set[str]:
    from agent.skill_utils import parse_config_string_list

    block = config.get(section) if isinstance(config, dict) else None
    if not isinstance(block, dict):
        return set()
    names = set(parse_config_string_list(block.get(key)) or ())
    platforms = block.get(per_platform) if per_platform else None
    if isinstance(platforms, dict):
        for value in platforms.values():
            names |= set(parse_config_string_list(value) or ())
    return {str(n) for n in names}


def _plugin_kind(name: str) -> str:
    """A messaging-platform plugin (bundled or catalog) reports as a platform, other plugins as plugins."""
    from .shared_metrics_catalog import _safe, bundled_platform_names, catalog_platform_names

    platforms = _safe(bundled_platform_names) | _safe(catalog_platform_names)
    return "platform" if name.strip().lower().rsplit("/", 1)[-1] in platforms else "plugin"


def _list_transitions(old_names: set[str], new_names: set[str], kind: str, namer) -> Iterable[tuple[str, str, str]]:
    for names, event in ((new_names - old_names, "disabled"), (old_names - new_names, "re_enabled")):
        for name in sorted(names):
            yield (_plugin_kind(name) if kind == "plugin" else kind), namer(name), event


def _toolset_transitions(old: dict, new: dict) -> Iterable[tuple[str, str, str]]:
    def scope(config: dict) -> tuple[Any, Any]:
        agent = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        return config.get("platform_toolsets"), agent.get("disabled_toolsets")

    if scope(old) == scope(new):
        return
    from hermes_cli.tools_config import _configurable_keys, _get_platform_tools

    platforms = {"cli"}
    for config in (old, new):
        if isinstance(config.get("platform_toolsets"), dict):
            platforms |= {str(p) for p in config["platform_toolsets"]}
    shipped = _configurable_keys()
    for platform in sorted(platforms):
        default = _get_platform_tools({}, platform, include_default_mcp_servers=False) & shipped
        before = _get_platform_tools(old, platform, include_default_mcp_servers=False)
        after = _get_platform_tools(new, platform, include_default_mcp_servers=False)
        for toolset in sorted(default):
            if (toolset in before) != (toolset in after):
                yield "toolset", toolset, "re_enabled" if toolset in after else "disabled"


def config_transitions(old: Any, new: Any) -> list[tuple[str, str, str]]:
    """(kind, name, event) for each shipped-on thing the write turned off or back on (deduped)."""
    old = old if isinstance(old, dict) else {}
    new = new if isinstance(new, dict) else {}
    found: list[tuple[str, str, str]] = []
    for batch in (
        _setting_transitions(old, new),
        _list_transitions(_names(old, "skills", "disabled", per_platform="platform_disabled"),
                          _names(new, "skills", "disabled", per_platform="platform_disabled"), "skill", skill_metric_name),
        _list_transitions(_names(old, "plugins", "disabled"), _names(new, "plugins", "disabled"), "plugin",
                          plugin_metric_name),
        _toolset_transitions(old, new),
    ):
        try:
            found.extend(t for t in batch if t not in found)
        except Exception:
            logger.debug("Feature-disabled transitions skipped", exc_info=True)
    return found


# ---- recording -------------------------------------------------------------------------------

def _emit(transitions: Iterable[tuple[str, str, str]], surface: str) -> None:
    from .relay_shared_metrics import record_process_mark
    from .shared_metrics_contract import FEATURE_DISABLED_MARK

    for kind, name, event in transitions:
        record_process_mark(FEATURE_DISABLED_MARK, {"event": event, "kind": kind, "name": name, "surface": surface})


def record_config_saved(old_raw: Any, new_config: Any) -> None:
    """``save_config``'s hook, after the write and outside its lock. Never raises."""
    if _process_surface is None:
        return
    try:
        from .relay_shared_metrics import enabled

        if not enabled() or (surface := current_surface()) is None:
            return
        _emit(config_transitions(old_raw, new_config), surface)
    except Exception:
        logger.debug("Feature-disabled not recorded", exc_info=True)


def recording_raw_config_write(config_path: Any, user_config: Any, write) -> None:
    """``hermes config set/unset`` write the raw file directly (not via ``save_config``): diff the file
    they replace against what they write. The write itself runs unguarded, so its errors are the caller's."""
    before = None
    if _process_surface is not None:
        try:
            from .relay_shared_metrics import enabled

            if enabled():
                from hermes_cli.config import read_user_config_raw

                before = read_user_config_raw(config_path)
        except Exception:
            logger.debug("Config snapshot for feature-disabled skipped", exc_info=True)
    write(config_path, user_config)
    if before is not None:
        record_config_saved(before, user_config)


def record_skill_removed(name: Any) -> None:
    """A catalog skill (bundled/optional) was uninstalled; third-party hub skills were never shipped."""
    if _process_surface is None:
        return
    try:
        from .relay_shared_metrics import enabled

        if not enabled() or (surface := current_surface()) is None:
            return
        if (metric_name := skill_metric_name(name)) != CUSTOM:
            _emit([("skill", metric_name, "disabled")], surface)
    except Exception:
        logger.debug("Skill removal not recorded", exc_info=True)
