"""Narrow temporary bridge from plugin runtime to the legacy config owner.

Configuration ownership is intentionally deferred while #122245 is active.  Plugin runtime
must centralize the temporary dependency here instead of importing :mod:`hermes_cli.config`
throughout the new package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def load_plugin_config() -> dict[str, Any]:
    """Load the canonical merged Hermes config for plugin-runtime policy."""
    from hermes_cli.config import load_config

    return load_config()


def load_plugin_config_readonly() -> dict[str, Any]:
    """Load the user-facing read-only config view for policy checks."""
    from hermes_cli.config import load_config_readonly

    return load_config_readonly()


def read_enabled_plugins() -> set[str] | None:
    """Return the plugins.enabled allow-list; None means missing or malformed."""
    try:
        config = load_plugin_config()
        plugins = config.get("plugins") if isinstance(config, dict) else None
        enabled = plugins.get("enabled") if isinstance(plugins, dict) else None
        return set(enabled) if isinstance(enabled, list) else None
    except Exception:
        return None


def read_disabled_plugins() -> set[str]:
    """Return the plugins.disabled deny-list; failures and malformed values are empty."""
    try:
        config = load_plugin_config()
        plugins = config.get("plugins") if isinstance(config, dict) else None
        disabled = plugins.get("disabled", []) if isinstance(plugins, dict) else []
        return set(disabled) if isinstance(disabled, list) else set()
    except Exception:
        return set()


def read_plugin_load_timeout_seconds() -> Any:
    """Return the raw configured plugin load timeout, or None when absent/unreadable."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        plugins = config.get("plugins") if isinstance(config, dict) else None
        if not isinstance(plugins, dict):
            return None
        return plugins.get("load_timeout_seconds")
    except Exception:
        return None


def read_hook_callback_timeout_seconds() -> Any:
    """Return the raw configured hook callback timeout, or None when absent/unreadable."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        plugins = config.get("plugins") if isinstance(config, dict) else None
        if not isinstance(plugins, dict):
            return None
        return plugins.get("hook_callback_timeout")
    except Exception:
        return None


def read_plugin_settings(plugin_id: str) -> Mapping[str, Any]:
    """Return one plugin's effective settings/config mapping for runtime validation."""
    try:
        config = load_plugin_config()
        plugins = config.get("plugins") if isinstance(config, dict) else None
        entries = plugins.get("entries") if isinstance(plugins, dict) else None
        entry = entries.get(plugin_id) if isinstance(entries, dict) else None
        if not isinstance(entry, Mapping):
            return {}
        raw = entry.get("settings")
        if not isinstance(raw, Mapping):
            raw = entry.get("config")  # migration fallback mirroring PluginContext.get_config
        return raw if isinstance(raw, Mapping) else {}
    except Exception:
        return {}


def save_plugin_config(config: dict[str, Any]) -> None:
    """Persist plugin-owned config changes through the canonical config writer."""
    from hermes_cli.config import save_config

    save_config(config)


def plugin_setting_segments(key: str) -> tuple[str, ...]:
    """Validate and split one plugin-relative settings key."""
    from hermes_cli.plugins_state import _plugin_relative_segments

    return _plugin_relative_segments(key)


def read_plugin_setting(
    plugin_id: str, segments: tuple[str, ...], default: Any = None,
) -> Any:
    """Read one plugin setting, preferring settings over the legacy config subtree."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.plugins_state import _nested_plugin_value, _plugin_settings_entry

    entry = _plugin_settings_entry(load_config_readonly() or {}, plugin_id)
    if entry is None:
        return default
    missing = object()
    value = _nested_plugin_value(entry.get("settings"), segments, missing)
    if value is not missing:
        return value
    return _nested_plugin_value(entry.get("config"), segments, default)


def write_plugin_setting(plugin_id: str, segments: tuple[str, ...], value: Any) -> None:
    """Persist one plugin setting through the canonical config-owned writer."""
    from hermes_cli.plugins_state import save_plugin_setting

    save_plugin_setting(plugin_id, segments, value)


def read_plugin_mcp_allowlist(plugin_id: str) -> list[str]:
    """Return the operator-granted MCP server allowlist; unreadable config denies all."""
    try:
        from hermes_cli.plugins_state import _plugin_settings_entry

        entry = _plugin_settings_entry(load_plugin_config() or {}, plugin_id) or {}
        raw = entry.get("mcp_allowlist")
        return [str(item) for item in raw] if isinstance(raw, list) else []
    except Exception:
        return []


def plugin_gateway_injection_allowed(plugin_id: str) -> bool:
    """Return whether a plugin may inject gateway/TUI session messages; failures deny."""
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.plugins_state import _plugin_settings_entry

        entry = _plugin_settings_entry(load_config_readonly() or {}, plugin_id) or {}
        return entry.get("allow_gateway_injection") is True
    except Exception:
        return False


def load_plugin_config_for_home(home: Path) -> dict[str, Any]:
    """Load merged plugin config while pinned to one manager-owned Hermes home."""
    from plugin_runtime.scope import plugin_home_scope

    with plugin_home_scope(home):
        return load_plugin_config()


def read_running_hermes_version() -> str:
    """Return the canonical running code's base release version.

    Compatibility policy remains runtime-owned; the CLI-owned identity
    resolver is reached only through this narrow bridge.
    """
    from hermes_version import __version__

    return str(__version__)
