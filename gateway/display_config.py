"""Per-platform display/verbosity resolver (``resolve_display_setting``).

Resolution order, first non-None wins: ``display.platforms.<platform>.<key>`` →
``display.<key>`` → ``_PLATFORM_DEFAULTS[platform][key]`` (a plugin platform: its registered
``display_tier`` + ``display_overrides``) → ``_GLOBAL_DEFAULTS[key]``.
Exception: ``display.streaming`` is CLI-only; gateway streaming follows the top-level
``streaming`` config unless a per-platform override sets it. Legacy
``display.tool_progress_overrides`` is still read as a ``tool_progress`` fallback.
"""

from __future__ import annotations

from typing import Any

# Settings configurable per-platform; other display settings are CLI-only.
_GLOBAL_DEFAULTS: dict[str, Any] = {
    "tool_progress": "all",
    "tool_progress_grouping": "accumulate",  # "accumulate" = edit one bubble; "separate" = one msg per tool
    "show_reasoning": False,
    "reasoning_style": "code",  # "code" (💭 **Reasoning:** + fence), "blockquote" ("> "), "subtext" ("-# " Discord)
    "tool_preview_length": 0,
    "streaming": None,  # None = follow top-level streaming config
    # Gateway-only assistant/status chatter; mobile platforms opt down to final-answer-first.
    "interim_assistant_messages": True,
    "suppress_warning_notifications": False,
    "long_running_notifications": True,
    "busy_ack_detail": True,
    "busy_steer_ack_enabled": True,  # busy_input_mode=steer echo; the text still lands in the run
    # Delete tool-progress / "⏳ Working" bubbles after a SUCCESSFUL final response where deletion is
    # supported (Telegram); failed runs keep them as breadcrumbs.
    "cleanup_progress": False,
    # Working-state text on text-rendering indicators (Slack assistant status): "full"/true = verb +
    # argument preview, "verb" = verb only (keeps paths out of shared channels), "off"/false = static.
    "live_status": "full",
}

# Tiers: HIGH = editing, personal/team use; MEDIUM = editing but customer-facing;
# LOW = no edit support (progress messages are permanent); MINIMAL = batch delivery.
_TIER_HIGH = {
    "tool_progress": "all", "show_reasoning": False, "tool_preview_length": 40,
    "streaming": None,  # follow global
    "interim_assistant_messages": True, "long_running_notifications": True, "busy_ack_detail": True,
}
_TIER_MEDIUM = {**_TIER_HIGH, "tool_progress": "new"}
_TIER_LOW = {
    **_TIER_HIGH, "tool_progress": "off", "streaming": False,
    "interim_assistant_messages": False, "long_running_notifications": False, "busy_ack_detail": False,
}
_TIER_MINIMAL = {**_TIER_LOW, "tool_preview_length": 0}

# Core-only surfaces. A plugin platform declares ``display_tier`` (+ optional ``display_overrides``)
# on its ``PlatformEntry``; _platform_defaults() reads it from the registry.
_PLATFORM_DEFAULTS: dict[str, dict[str, Any]] = {
    "webhook": _TIER_MINIMAL,
    "api_server": {**_TIER_HIGH, "tool_preview_length": 0},
}

_TIERS = {"high": _TIER_HIGH, "medium": _TIER_MEDIUM, "low": _TIER_LOW, "minimal": _TIER_MINIMAL}


def _platform_defaults(platform_key: str) -> dict[str, Any]:
    """Built-in defaults for *platform_key*: the table above, else the ``display_tier`` a plugin
    platform registered (``PlatformEntry.display_tier``) with its ``display_overrides`` on top."""
    if platform_key in _PLATFORM_DEFAULTS:
        return _PLATFORM_DEFAULTS[platform_key]
    try:
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform_key)
    except Exception:
        return {}
    tier = _TIERS.get(str(getattr(entry, "display_tier", "") or "").lower(), {})
    overrides = getattr(entry, "display_overrides", None)
    return {**tier, **overrides} if overrides else tier


# Canonical set of per-platform overrideable keys (for validation).
OVERRIDEABLE_KEYS = frozenset(_GLOBAL_DEFAULTS.keys())


def resolve_display_setting(user_config: dict, platform_key: str, setting: str, fallback: Any = None) -> Any:
    """Resolve a display setting with per-platform override support (see module docstring for order).

    ``platform_key`` is the platform config key (``"telegram"``; see ``_platform_config_key`` in
    gateway/run.py). Returns *fallback* when nothing is configured.
    """
    configured = _configured_display_value(user_config, platform_key, setting)
    if configured is not None:
        return _normalise(setting, configured)
    val = _platform_defaults(platform_key).get(setting)
    if val is None:
        val = _GLOBAL_DEFAULTS.get(setting)
    return fallback if val is None else val


def _configured_display_value(user_config: dict, platform_key: str, setting: str) -> Any:
    """First non-None operator value, without introducing tier defaults."""
    display_cfg = user_config.get("display")
    if not isinstance(display_cfg, dict):
        return None
    platforms = display_cfg.get("platforms")
    plat_overrides = platforms.get(platform_key) if isinstance(platforms, dict) else None
    if isinstance(plat_overrides, dict) and plat_overrides.get(setting) is not None:
        return plat_overrides[setting]
    if setting == "tool_progress":
        legacy = display_cfg.get("tool_progress_overrides")
        if isinstance(legacy, dict) and legacy.get(platform_key) is not None:
            return legacy[platform_key]
    if setting != "streaming":  # display.streaming is CLI-only
        return display_cfg.get(setting)
    return None


def resolve_tool_progress(user_config: dict, platform_key: str, env_mode: str | None = None) -> tuple[str, bool]:
    """Return (mode, explicit intent) from the same winning source.

    Non-None YAML wins over the legacy env bridge. Null inherits through to env,
    then tier defaults. A tier's off is not an operator request to disable cards.
    """
    configured = _configured_display_value(user_config, platform_key, "tool_progress")
    if configured is not None:
        return _normalise("tool_progress", configured), True
    if env_mode:
        return _normalise("tool_progress", env_mode), True
    return resolve_display_setting(user_config, platform_key, "tool_progress"), False


# --- Normalisation of YAML quirks (bare ``off`` → False in YAML 1.1, etc.) ---

_TRUTHY = {"true", "1", "yes", "on"}
_FALSY = {"false", "0", "no"}


def _norm_tristate(on: str, off: str, choices: set, extra_truthy: set = frozenset()):
    """Normaliser for bool-or-keyword settings: bools/truthy tokens → *on*, falsy → *off*, else a known choice or *on*."""
    def norm(value: Any) -> str:
        if isinstance(value, bool):
            return on if value else off
        val = str(value).strip().lower()
        if val in _FALSY:
            return off
        if val in _TRUTHY | extra_truthy:
            return on
        return val if val in choices else on
    return norm


def _norm_bool(value: Any) -> bool:
    return value.strip().lower() in _TRUTHY | {"raw", "verbose"} if isinstance(value, str) else bool(value)


def _norm_long_running(value: Any) -> Any:
    return "generic" if isinstance(value, str) and value.strip().lower() == "generic" else _norm_bool(value)


def _norm_suppress_warning_notifications(value: Any) -> bool:
    # Only an explicit, recognized opt-in may hide engine diagnostics.
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return False


def _norm_cleanup_progress(value: Any) -> bool:
    return value.lower() in _TRUTHY if isinstance(value, str) else bool(value)


def _norm_choice(choices: tuple[str, ...]) -> Any:
    def norm(value: Any) -> str:
        val = str(value).lower()
        return val if val in choices else choices[0]

    return norm


def _norm_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


_NORMALISERS: dict[str, Any] = {
    "tool_progress": _norm_tristate("all", "off", {"off", "new", "all", "verbose", "log"}),
    "show_reasoning": _norm_bool,
    "streaming": _norm_bool,
    "interim_assistant_messages": _norm_bool,
    "suppress_warning_notifications": _norm_suppress_warning_notifications,
    "long_running_notifications": _norm_long_running,
    "busy_ack_detail": _norm_bool,
    "busy_steer_ack_enabled": _norm_bool,
    "thinking_progress": _norm_bool,
    "cleanup_progress": _norm_cleanup_progress,
    "live_status": _norm_tristate("full", "off", {"full", "verb", "off"}, extra_truthy={"all"}),
    "tool_progress_grouping": _norm_choice(("accumulate", "separate")),
    "reasoning_style": _norm_choice(("code", "blockquote", "subtext")),
    "tool_preview_length": _norm_int,
}


def _normalise(setting: str, value: Any) -> Any:
    """Normalise a user-supplied value for *setting*; unknown settings pass through."""
    norm = _NORMALISERS.get(setting)
    return norm(value) if norm else value


# LOCAL's per-platform config key (``_platform_config_key(Platform.LOCAL)``): where a local session's
# ``/reasoning show|hide`` and ``/footer`` save, and what every local surface reads.
LOCAL_DISPLAY_KEY = "cli"
# The per-platform display settings a local surface takes from the local key over the global one.
_LOCAL_OVERLAY_KEYS = ("show_reasoning", "runtime_footer")


def local_display_config(display: Any) -> dict[str, Any]:
    """*display* (a ``config.yaml`` ``display`` block) as a local surface (classic CLI, Ink, Desktop,
    a local turn) resolves it: ``display.platforms.cli.<key>`` over ``display.<key>`` for the settings
    the local commands save there. No second key is written; the block is returned copied."""
    display = dict(display) if isinstance(display, dict) else {}
    platforms = display.get("platforms")
    local = platforms.get(LOCAL_DISPLAY_KEY) if isinstance(platforms, dict) else None
    if not isinstance(local, dict):
        return display
    for key in _LOCAL_OVERLAY_KEYS:
        value = local.get(key)
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(display.get(key), dict):
            value = {**display[key], **value}
        display[key] = _normalise(key, value)
    return display
