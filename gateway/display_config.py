"""Per-platform display/verbosity configuration resolver.

Provides ``resolve_display_setting()`` — the single entry-point for reading
display settings with platform-specific overrides and sensible defaults.

Resolution order (first non-None wins):
    1. ``display.platforms.<platform>.chat_types.<chat_type>.<key>``
       — explicit per-chat-type user override
    2. ``display.platforms.<platform>.<key>``  — explicit per-platform user override
    3. ``display.<key>``                       — global user setting
    4. ``_PLATFORM_DEFAULTS[<platform>][<key>]``  — built-in sensible default
    5. ``_GLOBAL_DEFAULTS[<key>]``              — built-in global default

Exception: ``display.streaming`` is CLI-only.  Gateway streaming follows the
top-level ``streaming`` config unless ``display.platforms.<platform>.streaming``
sets an explicit per-platform override.

Backward compatibility: ``display.tool_progress_overrides`` is still read as a
fallback for ``tool_progress`` when no ``display.platforms`` entry exists.  A
config migration (version bump) automatically moves the old format into the new
``display.platforms`` structure.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Overrideable display settings and their global defaults
# ---------------------------------------------------------------------------
# These are the settings that can be configured per-platform.
# Other display settings (compact, personality, skin, etc.) are CLI-only
# and don't participate in per-platform resolution.

_GLOBAL_DEFAULTS: dict[str, Any] = {
    "tool_progress": "all",
    "show_reasoning": False,
    "tool_preview_length": 0,
    "streaming": None,  # None = follow top-level streaming config
}

# ---------------------------------------------------------------------------
# Sensible per-platform defaults — tiered by platform capability
# ---------------------------------------------------------------------------
# Tier 1 (high): Supports message editing, typically personal/team use
# Tier 2 (medium): Supports editing but often workspace/customer-facing
# Tier 3 (low): No edit support — each progress msg is permanent
# Tier 4 (minimal): Batch/non-interactive delivery

_TIER_HIGH = {
    "tool_progress": "all",
    "show_reasoning": False,
    "tool_preview_length": 40,
    "streaming": None,  # follow global
}

_TIER_MEDIUM = {
    "tool_progress": "new",
    "show_reasoning": False,
    "tool_preview_length": 40,
    "streaming": None,
}

_TIER_LOW = {
    "tool_progress": "off",
    "show_reasoning": False,
    "tool_preview_length": 40,
    "streaming": False,
}

_TIER_MINIMAL = {
    "tool_progress": "off",
    "show_reasoning": False,
    "tool_preview_length": 0,
    "streaming": False,
}

_PLATFORM_DEFAULTS: dict[str, dict[str, Any]] = {
    # Tier 1 — full edit support, personal/team use
    "telegram":    _TIER_HIGH,
    "discord":     _TIER_HIGH,

    # Tier 2 — edit support, often customer/workspace channels
    # Slack: tool_progress off by default — Bolt posts cannot be edited like CLI;
    # "new"/"all" spam permanent lines in channels (hermes-agent#14663).
    "slack":           {**_TIER_MEDIUM, "tool_progress": "off"},
    "mattermost":      _TIER_MEDIUM,
    "matrix":          _TIER_MEDIUM,
    "feishu":          _TIER_MEDIUM,

    # Tier 3 — no edit support, progress messages are permanent
    "signal":          _TIER_LOW,
    "whatsapp":        _TIER_MEDIUM,  # Baileys bridge supports /edit
    "bluebubbles":     _TIER_LOW,
    "weixin":          _TIER_LOW,
    "wecom":           _TIER_LOW,
    "wecom_callback":  _TIER_LOW,
    "dingtalk":        _TIER_LOW,

    # Tier 4 — batch or non-interactive delivery
    "email":           _TIER_MINIMAL,
    "sms":             _TIER_MINIMAL,
    "webhook":         _TIER_MINIMAL,
    "homeassistant":   _TIER_MINIMAL,
    "api_server":      {**_TIER_HIGH, "tool_preview_length": 0},
}

# Canonical set of per-platform overrideable keys (for validation).
OVERRIDEABLE_KEYS = frozenset(_GLOBAL_DEFAULTS.keys())


def resolve_display_setting(
    user_config: dict,
    platform_key: str,
    setting: str,
    fallback: Any = None,
    *,
    chat_type: str | None = None,
) -> Any:
    """Resolve a display setting with per-platform override support.

    Parameters
    ----------
    user_config : dict
        The full parsed config.yaml dict.
    platform_key : str
        Platform config key (e.g. ``"telegram"``, ``"slack"``).  Use
        ``_platform_config_key(source.platform)`` from gateway/run.py.
    setting : str
        Display setting name (e.g. ``"tool_progress"``, ``"show_reasoning"``).
    fallback : Any
        Fallback value when the setting isn't found anywhere.
    chat_type : str | None
        Optional source chat type (e.g. ``"dm"``, ``"group"``,
        ``"channel"``, ``"thread"``).  When provided, chat-type overrides
        under ``display.platforms.<platform>.chat_types`` take precedence.

    Returns
    -------
    The resolved value, or *fallback* if nothing is configured.
    """
    display_cfg = user_config.get("display") or {}

    # 1. Explicit per-chat-type override
    # (display.platforms.<platform>.chat_types.<chat_type>.<key>)
    platforms = display_cfg.get("platforms") or {}
    plat_overrides = platforms.get(platform_key)
    if isinstance(plat_overrides, dict):
        chat_type_key = _normalise_chat_type(chat_type)
        chat_types = plat_overrides.get("chat_types")
        if chat_type_key and isinstance(chat_types, dict):
            chat_overrides = chat_types.get(chat_type_key)
            if isinstance(chat_overrides, dict):
                val = chat_overrides.get(setting)
                if val is not None:
                    return _normalise(setting, val)

        # 2. Explicit per-platform override (display.platforms.<platform>.<key>)
        val = plat_overrides.get(setting)
        if val is not None:
            return _normalise(setting, val)

    # 2b. Backward compat: display.tool_progress_overrides.<platform>
    if setting == "tool_progress":
        legacy = display_cfg.get("tool_progress_overrides")
        if isinstance(legacy, dict):
            val = legacy.get(platform_key)
            if val is not None:
                return _normalise(setting, val)

    # 3. Global user setting (display.<key>).  Skip display.streaming because
    # that key controls only CLI terminal streaming; gateway token streaming is
    # governed by the top-level streaming config plus per-platform overrides.
    if setting != "streaming":
        val = display_cfg.get(setting)
        if val is not None:
            return _normalise(setting, val)

    # 4. Built-in platform default
    plat_defaults = _PLATFORM_DEFAULTS.get(platform_key)
    if plat_defaults:
        val = plat_defaults.get(setting)
        if val is not None:
            return val

    # 5. Built-in global default
    val = _GLOBAL_DEFAULTS.get(setting)
    if val is not None:
        return val

    return fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(setting: str, value: Any) -> Any:
    """Normalise YAML quirks (bare ``off`` → False in YAML 1.1)."""
    if setting == "tool_progress":
        if value is False:
            return "off"
        if value is True:
            return "all"
        return str(value).lower()
    if setting in ("show_reasoning", "streaming"):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    if setting == "tool_preview_length":
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return value


def _normalise_chat_type(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    aliases = {
        "direct": "dm",
        "private": "dm",
    }
    return aliases.get(key, key)
