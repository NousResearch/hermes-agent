"""Plugin namespaced settings bridge retained under the CLI/config owner."""

from __future__ import annotations

import re
from typing import Any, Mapping

from plugin_runtime.state import _locked_plugin_state

_PLUGIN_SETTING_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_PLUGIN_SETTING_RESERVED_ROOTS = frozenset({"model", "plugins", "security", "settings"})


def _plugin_relative_segments(key: str) -> tuple[str, ...]:
    """Validate/split a plugin-relative settings key before any config read."""
    if not isinstance(key, str):
        raise ValueError("Expected a plugin-relative config key string")
    segments = tuple(key.split("."))
    invalid = not key or "/" in key or "\\" in key or segments[0].lower() in _PLUGIN_SETTING_RESERVED_ROOTS
    if invalid or not all(_PLUGIN_SETTING_SEGMENT_RE.fullmatch(segment) for segment in segments):
        raise ValueError(
            "Expected a plugin-relative config key such as 'endpoint' or "
            "'retry.policy'; global, cross-plugin, and traversal paths are forbidden"
        )
    return segments


def _nested_plugin_value(root: object, segments: tuple[str, ...], default: Any) -> Any:
    """Walk segments through nested mappings; default on the first miss."""
    current = root
    for segment in segments:
        if not isinstance(current, Mapping) or segment not in current:
            return default
        current = current[segment]
    return current


def _nested_plugin_mapping(segments: tuple[str, ...], value: Any) -> dict[str, Any]:
    """Wrap value in nested single-key dicts, outermost first."""
    nested: Any = value
    for segment in reversed(segments):
        nested = {segment: nested}
    return nested


def _plugin_settings_entry(config: object, plugin_id: str) -> Mapping[str, Any] | None:
    """Return plugins.entries.<plugin_id> as a mapping, else None."""
    entry = _nested_plugin_value(config, ("plugins", "entries", plugin_id), None)
    return entry if isinstance(entry, Mapping) else None


def save_plugin_setting(plugin_id: str, segments: tuple[str, ...], value: Any) -> None:
    """Atomically write one value under plugins.entries.<plugin_id>.settings."""
    from hermes_cli import config as config_mod
    from hermes_cli import managed_scope

    if config_mod.is_managed():
        raise PermissionError("Plugin settings cannot be changed in a managed install")
    full_path = ("plugins", "entries", plugin_id, "settings", *segments)
    dotted_path = ".".join(full_path)
    if managed_scope.is_key_managed(dotted_path):
        raise PermissionError(f"Plugin setting {dotted_path!r} is administrator-managed")
    partial = _nested_plugin_mapping(full_path[:4], _nested_plugin_mapping(segments, value))
    with _locked_plugin_state(config_mod.get_config_path()), config_mod._CONFIG_LOCK:
        config_mod.read_user_config_raw()
        config_mod.save_config(partial, preserve_keys={full_path}, merge_existing=True)
