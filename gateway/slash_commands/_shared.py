"""Module-level state shared by the slash-command leaf modules.

Moved verbatim from ``gateway/slash_commands.py``. The single shared
``logger`` preserves the historical ``"gateway.run"`` log name."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent.i18n import t
from gateway.config import Platform

logger = logging.getLogger("gateway.run")

# Upper bound on the off-loop agent-resource cleanup during a /new or /reset
# (see _handle_reset_command). A stuck teardown must not block the event loop;
# past this the reset proceeds and the cleanup is left to finish (or leak) in
# its worker thread. (#35994)
_RESET_CLEANUP_TIMEOUT_S = 30.0

def _clean_str(value: Any) -> str:
    """Strip and return a non-empty string value, or empty string."""
    return value.strip() if isinstance(value, str) and value.strip() else ""

def _int_value(value: Any) -> int:
    """Safely coerce to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

def _model_switch_skew_guard() -> Optional[str]:
    """Refuse a model switch when the gateway is running stale code.

    A long-lived gateway holds its modules in memory from boot. If the checkout
    changed underneath it (e.g. a manual ``git pull``), switching models can hit
    a first-time lazy import on a new code path and crash on a stale cached
    dependency — the cryptic ``cannot import name 'env_float' from 'utils'``.
    Detect the drift and tell the user to restart instead.

    Intentionally scoped to model switching — the known, highest-risk trigger.
    Any first-time lazy import on a stale process is technically exposed; we
    don't guard every import site, only this one.
    """
    from gateway.code_skew import detect_code_skew

    skew = detect_code_skew()
    if not skew:
        return None
    boot_rev, disk_rev = skew
    return t(
        "gateway.model.error_prefix",
        error=(
            f"This gateway is running code from {boot_rev} but the checkout on "
            f"disk is now {disk_rev}. Switching models would risk a stale-module "
            f"crash — restart the gateway to load the new code: hermes gateway restart"
        ),
    )

def _home_thread_from_source(source) -> Optional[str]:
    """The thread id /sethome should persist on the home target, or None.

    Slack thread-per-message session keying stamps a top-level message's own
    id as ``source.thread_id`` (a session KEY, not a durable location).
    Persisting it would pin the HOME target itself to the ephemeral thread
    spawned around the /sethome message — every bare-platform delivery
    (``deliver="slack"``) would then land in that thread forever. Same
    recognition as cron origin capture: a Slack thread id equal to the
    message's own id is synthetic. A /sethome run inside a genuine thread
    (thread id = the parent's id, not this message's own) keeps that thread
    as the home target.
    """
    thread_id = getattr(source, "thread_id", None)
    if not thread_id:
        return None
    if (
        getattr(source, "platform", None) == Platform.SLACK
        and getattr(source, "message_id", None)
        and str(thread_id) == str(source.message_id)
    ):
        return None
    return str(thread_id)
