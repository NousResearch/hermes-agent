"""Per-platform display/verbosity configuration resolver.

Provides ``resolve_display_setting()`` — the single entry-point for reading
display settings with platform-specific overrides and sensible defaults.

Resolution order (first non-None wins):
    1. ``display.platforms.<platform>.<key>``  — explicit per-platform user override
    2. ``display.<key>``                       — global user setting
    3. ``_PLATFORM_DEFAULTS[<platform>][<key>]``  — built-in sensible default
    4. ``_GLOBAL_DEFAULTS[<key>]``              — built-in global default

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
    "tool_progress_grouping": "accumulate",  # "accumulate" = edit one bubble; "separate" = one msg per tool
    "show_reasoning": False,
    # How a reasoning/thinking summary is rendered when show_reasoning is on.
    #   "code"      -> 💭 **Reasoning:** + fenced code block (legacy default)
    #   "blockquote"-> each line prefixed with "> "
    #   "subtext"   -> each line prefixed with "-# " (Discord small grey subtext)
    # Discord defaults to "subtext"; everywhere else defaults to "code".
    "reasoning_style": "code",
    "tool_preview_length": 0,
    "streaming": None,  # None = follow top-level streaming config
    # Gateway-only assistant/status chatter controls. These default on for
    # back-compat, but mobile platforms can opt down to final-answer-first.
    "interim_assistant_messages": True,
    "long_running_notifications": True,
    "busy_ack_detail": True,
    # Whether busy_input_mode=steer sends a visible "Steered into current run"
    # acknowledgment after successfully injecting the user's mid-turn message.
    # Disable when the platform should steer silently (the text still lands in
    # the active run; only the confirmation echo is suppressed).
    "busy_steer_ack_enabled": True,
    # When true, delete tool-progress / "⏳ Working — N min" / status bubbles
    # after the final response lands on platforms that support message
    # deletion (e.g. Telegram). Off by default — progress is still shown
    # live, just cleaned up after success so the chat doesn't fill up with
    # stale breadcrumbs. Failed runs leave bubbles in place as breadcrumbs.
    "cleanup_progress": False,
    # Optional per-tool completion timing in gateway progress bubbles. When
    # enabled, each tool's existing progress row gets a compact duration suffix
    # appended in place (e.g. "💻 terminal: \"date\" · 10ms"), and a fallback
    # "✅ <tool> completed in <time>" row is emitted only if no matching start
    # row was found. Off by default to avoid extra chat noise; platforms/users
    # can opt in where timing visibility matters.
    "tool_completion_durations": False,
    # Surface a delegate_task child's OWN tool calls in gateway progress.
    # Child tool events are already relayed to the parent's progress callback
    # (tools/delegate_tool.py emits "subagent.tool"/"subagent.progress"), but
    # the gateway drops them by default — only the parent's delegate_task call
    # is shown. Three levels:
    #   "off"     — drop child tool events (default; original behavior)
    #   "batched" — render the batched "subagent.progress" summary only
    #               (one card per ~5 child tools; low spam)
    #   "full"    — render every individual "subagent.tool" event (noisy on
    #               long subagent runs, but a complete live trace)
    # A long subagent doing dozens of tools can post many permanent messages
    # on platforms with no message editing, so this stays off unless opted in.
    "subagent_tool_progress": "off",
    # Live, edited-in-place roster bubble for delegate_task subagents. When
    # "on", a single message shows how many children are running, each child's
    # short goal, per-agent elapsed time, and status (running/done/errored/
    # timed-out), collapsing to a one-liner at turn end. Independent of
    # subagent_tool_progress (membership + status come from subagent.start/
    # .complete lifecycle events; elapsed from a periodic registry poll). Only
    # renders on edit-capable adapters (Telegram, Discord); silent no-op on
    # platforms without message editing. Off by default.
    "subagent_roster": "off",
    # Minimum seconds between live roster bubble edits. Telegram enforces a
    # per-chat edit flood ceiling; a busy chat with other bubbles can trip
    # "Flood control exceeded" and freeze the roster's live timer. 10s stays
    # well under the ceiling. The final collapse (force) bypasses this throttle
    # so the terminal state always lands. Clamped to a 1.0s floor.
    "subagent_roster_interval": 10.0,
    # Surface the `todo` tool's plan/status card in gateway progress WITHOUT
    # turning on general tool_progress. Mirrors delegate_task_args: a single
    # tool's card is rendered even when tool_progress is "off" (the Telegram
    # default), so the multi-step plan + per-item status/timing shows up
    # without the rest of the tool noise. Binary on/off; off by default.
    # Only renders on edit-capable adapters; harmless no-op elsewhere.
    "todo_progress": "off",
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
    "interim_assistant_messages": True,
    "long_running_notifications": True,
    "busy_ack_detail": True,
}

_TIER_MEDIUM = {
    "tool_progress": "new",
    "show_reasoning": False,
    "tool_preview_length": 40,
    "streaming": None,
    "interim_assistant_messages": True,
    "long_running_notifications": True,
    "busy_ack_detail": True,
}

_TIER_LOW = {
    "tool_progress": "off",
    "show_reasoning": False,
    "tool_preview_length": 40,
    "streaming": False,
    "interim_assistant_messages": False,
    "long_running_notifications": False,
    "busy_ack_detail": False,
}

_TIER_MINIMAL = {
    "tool_progress": "off",
    "show_reasoning": False,
    "tool_preview_length": 0,
    "streaming": False,
    "interim_assistant_messages": False,
    "long_running_notifications": False,
    "busy_ack_detail": False,
}

_PLATFORM_DEFAULTS: dict[str, dict[str, Any]] = {
    # Tier 1 — full edit support, personal/team use
    # Telegram is usually a mobile inbox: keep tool_progress quiet and skip
    # the verbose busy-ack iteration counter, but DO surface real mid-turn
    # assistant commentary (interim_assistant_messages) and DO send periodic
    # heartbeats (long_running_notifications) so the user has signal between
    # turn start and final answer. Otherwise it looks like "typing..." for
    # 30 minutes with nothing happening. Opt in to verbose iteration detail
    # via display.platforms.telegram.busy_ack_detail / tool_progress.
    "telegram":    {
        **_TIER_HIGH,
        "tool_progress": "off",
        "busy_ack_detail": False,
    },
    # Discord has a native "subtext" primitive (-# small grey text) that reads
    # as metadata rather than content, so reasoning summaries default to it
    # here instead of the fenced code block used elsewhere.
    "discord":     {**_TIER_HIGH, "reasoning_style": "subtext"},

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
    # WhatsApp Cloud API: Meta added message editing in 2023 but the
    # Hermes Cloud adapter doesn't implement edit_message yet, so we
    # stay on TIER_LOW (tool_progress off) to avoid spamming each
    # status update as a separate message. Promote to TIER_MEDIUM once
    # Cloud's edit_message lands.
    "whatsapp_cloud":  _TIER_LOW,
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

    Returns
    -------
    The resolved value, or *fallback* if nothing is configured.
    """
    display_cfg = user_config.get("display") or {}

    # 1. Explicit per-platform override (display.platforms.<platform>.<key>)
    platforms = display_cfg.get("platforms") or {}
    plat_overrides = platforms.get(platform_key)
    if isinstance(plat_overrides, dict):
        val = plat_overrides.get(setting)
        if val is not None:
            return _normalise(setting, val)

    # 1b. Backward compat: display.tool_progress_overrides.<platform>
    if setting == "tool_progress":
        legacy = display_cfg.get("tool_progress_overrides")
        if isinstance(legacy, dict):
            val = legacy.get(platform_key)
            if val is not None:
                return _normalise(setting, val)

    # 2. Global user setting (display.<key>).  Skip display.streaming because
    # that key controls only CLI terminal streaming; gateway token streaming is
    # governed by the top-level streaming config plus per-platform overrides.
    if setting != "streaming":
        val = display_cfg.get(setting)
        if val is not None:
            return _normalise(setting, val)

    # 3. Built-in platform default
    plat_defaults = _PLATFORM_DEFAULTS.get(platform_key)
    if plat_defaults:
        val = plat_defaults.get(setting)
        if val is not None:
            return val

    # 4. Built-in global default
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
        val = str(value).strip().lower()
        if val in {"false", "0", "no"}:
            return "off"
        if val in {"true", "1", "yes", "on"}:
            return "all"
        return val if val in {"off", "new", "all", "verbose", "log"} else "all"
    if setting in {
        "show_reasoning",
        "streaming",
        "interim_assistant_messages",
        "long_running_notifications",
        "busy_ack_detail",
        "busy_steer_ack_enabled",
        "thinking_progress",
    }:
        if isinstance(value, str):
            val = value.strip().lower()
            if val == "generic" and setting == "long_running_notifications":
                return "generic"
            return val in {"true", "1", "yes", "on", "raw", "verbose"}
        return bool(value)
    if setting in {"cleanup_progress", "tool_completion_durations"}:
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "on"}
        return bool(value)
    if setting == "tool_progress_grouping":
        val = str(value).lower()
        return val if val in ("accumulate", "separate") else "accumulate"
    if setting == "reasoning_style":
        val = str(value).lower()
        return val if val in ("code", "blockquote", "subtext") else "code"
    if setting == "subagent_tool_progress":
        # Tri-state string flag. Accept legacy booleans for forgiveness:
        # True → "full" (show everything), False → "off". Unknown strings
        # collapse to "off" so a typo fails safe (quiet) rather than spamming.
        if value is True:
            return "full"
        if value is False:
            return "off"
        v = str(value).strip().lower()
        return v if v in {"off", "batched", "full"} else "off"
    if setting in {"subagent_roster", "todo_progress"}:
        # Binary on/off. Accept YAML 1.1 booleans (bare on->True, off->False).
        # Unknown strings fail safe to "off".
        if value is True:
            return "on"
        if value is False:
            return "off"
        v = str(value).strip().lower()
        return "on" if v in {"on", "true", "1", "yes"} else "off"
    if setting == "tool_preview_length":
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    if setting == "subagent_roster_interval":
        # Seconds between roster edits. Parse to float and clamp to a 1.0s
        # floor so a typo / 0 can never flood Telegram's edit rate limiter.
        try:
            secs = float(value)
        except (TypeError, ValueError):
            return 10.0
        return max(1.0, secs)
    return value
