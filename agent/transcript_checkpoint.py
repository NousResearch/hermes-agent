"""Turn-boundary transcript checkpoint for agents that run without a session DB (#92079).

An agent constructed with a ``session_db`` (CLI, oneshot, gateway) already persists every
tool round to SQLite before its side effects run, so an abrupt exit loses at most the
in-flight call. An agent constructed without one (library use, ``batch_runner``, external
harnesses) has no such path: ``_flush_messages_to_session_db`` returns early and the
transcript exists only in memory until the caller serializes it at clean exit. SIGTERM from a
wall-clock limit, SIGKILL, OOM or node loss then lose the whole conversation.

A signal handler cannot close that gap (SIGKILL is uncatchable, OOM and node loss run no
cleanup), so the transcript is written to ``<hermes home>/transcripts/transcript_<session_id>.json``
every ``agent.transcript_checkpoint_every`` API calls (default 10, ``0`` disables). One file per
session, latest state wins, so disk stays bounded; the loss on an abrupt exit is bounded to the
calls since the last checkpoint.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("agent.conversation_loop")

DEFAULT_INTERVAL = 10

# Session ids become a file name; anything outside this set (path separators, ``..``) is refused.
_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def resolve_interval(raw: Any) -> int:
    """``agent.transcript_checkpoint_every`` as an interval: ``None`` → default, ``0`` disables,
    negatives clamp to ``0``. A malformed value keeps the default (protection stays on) and warns."""
    if raw is None:
        return DEFAULT_INTERVAL
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid agent.transcript_checkpoint_every=%r; expected an integer. Using %d.",
            raw, DEFAULT_INTERVAL,
        )
        return DEFAULT_INTERVAL


def needs_checkpoint(agent: Any) -> bool:
    """True when nothing else persists this agent's transcript incrementally."""
    return getattr(agent, "_session_db", None) is None and not getattr(agent, "_persist_disabled", False)


def maybe_write(agent: Any, messages: Any, api_call_count: int) -> None:
    """Write the checkpoint when ``api_call_count`` lands on the interval. Never raises."""
    interval = getattr(agent, "_transcript_checkpoint_every", DEFAULT_INTERVAL)
    if not interval or api_call_count % interval or not needs_checkpoint(agent):
        return
    write(agent, messages, api_call_count)


def write(agent: Any, messages: Any, api_call_count: int) -> bool:
    """Persist ``messages`` atomically; ``False`` (with a warning) when skipped or failed."""
    session_id = getattr(agent, "session_id", None)
    if not session_id:
        return False
    session_id = str(session_id)
    if not _SAFE_SESSION_ID.fullmatch(session_id):
        logger.warning("transcript checkpoint skipped: unsafe session id %r", session_id)
        return False
    try:
        from hermes_constants import get_hermes_home
        from utils import atomic_json_write

        dest_dir = get_hermes_home() / "transcripts"
        dest_dir.mkdir(parents=True, exist_ok=True)
        # atomic_json_write: unique mkstemp temp file, fsync, then replace, so a crash or a
        # concurrent writer never leaves truncated JSON in place. 0600: transcripts are private.
        # default=str keeps one non-JSON tool payload from disabling every later checkpoint.
        atomic_json_write(
            dest_dir / f"transcript_{session_id}.json",
            {
                "session_id": session_id,
                "api_call_count": api_call_count,
                "message_count": len(messages),
                "messages": list(messages),
            },
            indent=None, mode=0o600, fsync_dir=True, default=str,
        )
        return True
    except Exception as exc:  # health: allow BLE001 -- a checkpoint must never break the conversation
        # Loud on purpose: a checkpoint that is quietly not written looks like protection and is not.
        logger.warning("transcript checkpoint failed (session=%s): %s", session_id, type(exc).__name__)
        return False
