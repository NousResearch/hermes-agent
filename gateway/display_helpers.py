"""Display / resolution helper functions for the gateway.

Extracted from ``gateway/run.py`` to reduce module size and improve cohesion.
All functions here are pure helpers with no side-effects beyond reading
``os.environ`` (for the env-var helpers).
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_TELEGRAM_COMMAND_MENTION_RE = re.compile(r"(?<![\w:/])/([A-Za-z0-9][A-Za-z0-9_-]*)")

_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT = 60 * 60


# ---------------------------------------------------------------------------
# Progress / thread resolution
# ---------------------------------------------------------------------------


def _resolve_progress_thread_id(
    platform: Any,
    source_thread_id: Any,
    event_message_id: Any,
    *,
    reply_in_thread: bool = True,
) -> Optional[str]:
    """Return thread/root ID that progress/status bubbles should target.

    ``reply_in_thread=False`` (Slack ``platforms.slack.extra.reply_in_thread``)
    disables the synthetic-thread fallback: progress messages must not create
    a thread the final flat reply would then inherit. A source.thread_id equal
    to the event's own message id is the adapter's synthetic session-keying
    thread, not a real thread — treat it as "no thread" too (#18859).
    """
    platform_value = getattr(platform, "value", platform)
    platform_key = str(platform_value or "").lower()
    if not reply_in_thread:
        if (
            source_thread_id
            and event_message_id
            and str(source_thread_id) == str(event_message_id)
        ):
            return None
        return str(source_thread_id) if source_thread_id else None
    if source_thread_id:
        return str(source_thread_id)
    if platform_key in {"slack", "mattermost"} and event_message_id:
        return str(event_message_id)
    return None


# ---------------------------------------------------------------------------
# Display-config helpers
# ---------------------------------------------------------------------------


def _has_platform_display_override(user_config: dict, platform_key: str, setting: str) -> bool:
    """Return True when display.platforms.<platform> explicitly sets setting."""
    display = user_config.get("display") if isinstance(user_config, dict) else None
    if not isinstance(display, dict):
        return False
    platforms = display.get("platforms")
    if not isinstance(platforms, dict):
        return False
    platform_cfg = platforms.get(platform_key)
    return isinstance(platform_cfg, dict) and setting in platform_cfg


def _resolve_gateway_display_bool(
    user_config: dict,
    platform_key: str,
    setting: str,
    *,
    default: bool = False,
    platform_value: Optional[str] = None,
    require_platform_override_for: set[str] | None = None,
) -> bool:
    """Resolve a boolean display setting with optional platform-only opt-in.

    Some display features expose assistant scratch text rather than deliberate
    user-facing output.  For high-noise threaded chat surfaces such as
    Mattermost, a global opt-in is too broad: they must be enabled with an
    explicit display.platforms.<platform>.<setting> override.

    ``platform_value`` is the caller-normalized platform key (lowercase) — the
    normalization helper itself stays in ``gateway.run`` to avoid duplicating
    ``_gateway_platform_value`` here (it is extracted separately by the
    platform-routing slice).
    """
    current_platform = platform_value or str(platform_key or "").strip().lower()
    platform_only = {
        str(candidate or "").strip().lower()
        for candidate in (require_platform_override_for or set())
    }
    if (
        current_platform in platform_only
        and not _has_platform_display_override(user_config, platform_key, setting)
    ):
        return False

    from gateway.display_config import resolve_display_setting

    value = resolve_display_setting(user_config, platform_key, setting, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    if value is None:
        return bool(default)
    return bool(value)


# ---------------------------------------------------------------------------
# Telegram command normalisation
# ---------------------------------------------------------------------------


def _telegramize_command_mentions(text: str, platform: Any) -> str:
    """Rewrite slash-command mentions to Telegram-valid command names.

    Telegram Bot API command names allow only lowercase letters, digits, and
    underscores.  Keep other platform renderings unchanged, but normalize
    Telegram help text so command mentions remain clickable/valid there.
    """
    platform_value = getattr(platform, "value", platform)
    if platform_value != "telegram":
        return text

    from hermes_cli.commands import _sanitize_telegram_name

    def _replace(match: re.Match[str]) -> str:
        sanitized = _sanitize_telegram_name(match.group(1))
        return f"/{sanitized}" if sanitized else match.group(0)

    return _TELEGRAM_COMMAND_MENTION_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Timestamp / freshness helpers
# ---------------------------------------------------------------------------


def _coerce_gateway_timestamp(value: Any) -> Optional[float]:
    """Best-effort conversion of stored gateway timestamps to epoch seconds.

    Missing/unparseable timestamps return None so legacy transcripts keep the
    historical auto-continue behaviour instead of being silently dropped.
    Accepts: datetime, epoch seconds (int/float), epoch milliseconds (when
    the magnitude exceeds year-2286), ISO-8601 strings (with or without a
    trailing ``Z``), and numeric strings.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, bool):  # bool is a subclass of int — skip it
        return None
    if isinstance(value, (int, float)):
        # Some platform events use milliseconds; Hermes state rows use seconds.
        return float(value) / 1000.0 if float(value) > 10_000_000_000 else float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
            return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _auto_continue_freshness_window() -> float:
    """Return the configured auto-continue freshness window in seconds.

    Thin wrapper that delegates to the canonical implementation in
    ``gateway.session`` (the single source of truth shared with the
    routing-time zombie gate in ``get_or_create_session``).  Reads
    ``HERMES_AUTO_CONTINUE_FRESHNESS`` (bridged from ``config.yaml``
    ``agent.gateway_auto_continue_freshness`` at gateway startup, same
    pattern as ``HERMES_AGENT_TIMEOUT``).  Falls back to the module default
    when unset or malformed.  Non-positive values disable the freshness gate
    (restores the pre-fix "always fresh" behaviour for users who want to opt
    out).  Kept here so existing call sites and test patches importing it
    from ``gateway.run`` continue to work.
    """
    from gateway.session import auto_continue_freshness_window
    return auto_continue_freshness_window()


# ---------------------------------------------------------------------------
# Auto-continue / startup-restore timing helpers
# ---------------------------------------------------------------------------


# Startup-restore inbound-gate drain bound: longer than a normal resume
# turn's first response yet short enough that one pathologically
# long resumed turn can't hold every channel's inbound queued for minutes.
# Override via ``config.yaml`` ``agent.gateway_startup_restore_drain_timeout``.
_STARTUP_RESTORE_DRAIN_TIMEOUT_SECS_DEFAULT = 30.0


def _startup_restore_drain_timeout_secs() -> float:
    """Max seconds ``_finish_startup_restore`` waits on boot auto-resume turns
    before releasing the inbound gate and draining the queue.

    While startup restore is in progress the gateway QUEUES every inbound
    message (``_queue_startup_restore_event``) instead of processing it, so no
    channel gets a reply until the gate opens.  The gate is opened by
    ``_finish_startup_restore``, which waits for the synthetic boot
    auto-resume turns to finish.  A single long resumed turn therefore held
    the gate shut for every channel — inbound piled up unanswered for as long
    as that one turn ran.

    This bounds that wait.  Duplicate-agent safety does NOT depend on the
    wait: ``_schedule_resume_pending_sessions`` claims each session's
    ``_running_agents`` slot SYNCHRONOUSLY (before the gate ever runs), so a
    message drained while a resume turn is still running queues behind that
    slot rather than spawning a second agent.  So on timeout we release the
    gate and let the slow turn finish in the background.

    Reads ``HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT`` (bridged from
    ``config.yaml`` ``agent.gateway_startup_restore_drain_timeout`` at gateway
    startup, same pattern as the other ``agent.*`` knobs).  Non-positive
    disables the bound (restores the historical "wait forever" behaviour).
    """
    raw = os.environ.get("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT")
    if raw is None or raw == "":
        return float(_STARTUP_RESTORE_DRAIN_TIMEOUT_SECS_DEFAULT)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(_STARTUP_RESTORE_DRAIN_TIMEOUT_SECS_DEFAULT)


def _float_env(name: str, default: float) -> float:
    """Read an env var as float, falling back to ``default`` on typos/empty.

    A misconfigured env var (e.g. ``HERMES_AGENT_TIMEOUT=abc``) must not
    crash the gateway or an agent turn.  Unset/empty also falls back.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)
