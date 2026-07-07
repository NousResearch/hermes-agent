"""Replay-history sanitization shared across resume code paths.

When a session's last turn dies mid-tool-loop — the process is killed by a
restart/shutdown command, a stale-timeout fires, or an interrupt lands before
the tool result is written — the persisted transcript can end with a dangling
``assistant(tool_calls)`` (no matching ``tool`` answer) or an interrupted
``assistant→tool`` block.  On resume the model sees that broken tail and
re-issues the unanswered call, producing an endless "thinking"/reboot loop
(#49201, #29086).

These pure helpers strip those tails before the history is replayed to the
model.  They were originally local to ``gateway/run.py`` (which fixed the
messaging-gateway path) and are extracted here so every resume surface — the
messaging gateway AND the TUI/WebUI gateway — shares the same cleanup instead
of the WebUI path silently skipping it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def is_interrupted_tool_result(content: Any) -> bool:
    """Return True if a tool result indicates the tool was interrupted."""
    if not isinstance(content, str):
        return False
    lowered = content.lower()
    if "[command interrupted]" in lowered:
        return True
    if "exit_code" in lowered and ("130" in lowered or "-1" in lowered):
        return "interrupt" in lowered
    return False


def strip_interrupted_tool_tails(
    agent_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Strip interrupted assistant→tool sequences from replay history.

    Older interrupted gateway turns can be followed by a queued real user
    message, so the interrupted assistant/tool block is not necessarily the
    final tail by the time we rebuild replay history.  Remove any contiguous
    assistant(tool_calls) + tool-result block that contains an interrupted tool
    result, while preserving successful tool-call sequences intact.
    """
    if not agent_history:
        return agent_history

    cleaned: List[Dict[str, Any]] = []
    i = 0
    n = len(agent_history)
    while i < n:
        msg = agent_history[i]
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            j = i + 1
            tool_results: List[Dict[str, Any]] = []
            while j < n and agent_history[j].get("role") == "tool":
                tool_results.append(agent_history[j])
                j += 1
            if tool_results and any(
                is_interrupted_tool_result(m.get("content", ""))
                for m in tool_results
            ):
                logger.debug(
                    "Stripping interrupted assistant→tool replay block "
                    "(indices %d–%d, tool_results=%d)",
                    i, j - 1, len(tool_results),
                )
                i = j
                continue
        if msg.get("role") == "tool" and is_interrupted_tool_result(msg.get("content", "")):
            logger.debug("Stripping orphan interrupted tool result from replay history")
            i += 1
            continue
        cleaned.append(msg)
        i += 1

    return cleaned


def strip_dangling_tool_call_tail(
    agent_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Strip a trailing ``assistant(tool_calls)`` block left with NO answers.

    When a tool call itself kills the gateway process (``docker restart``,
    ``systemctl restart``, ``kill``, ``hermes gateway restart``), the process
    is terminated by SIGKILL *mid-call* — before the tool result is ever
    written and before the orderly shutdown rewind
    (``_drop_trailing_empty_response_scaffolding``) can run.  The last thing
    persisted is the ``assistant`` message that issued the ``tool_calls``,
    with zero matching ``tool`` rows.

    On resume the model sees an unanswered tool call at the tail and naturally
    re-issues it — which restarts the gateway again, producing the infinite
    reboot loop in #49201.  ``strip_interrupted_tool_tails`` does not catch
    this because there is no tool result to inspect for an interrupt marker.

    This strips that dangling tail at the source so there is nothing for the
    model to re-execute.  It only acts when the tail is an
    ``assistant(tool_calls)`` whose calls have NO corresponding ``tool``
    results — a completed assistant→tool pair (any tool answers present) is
    left untouched so genuine mid-progress tool loops still resume.
    """
    if not agent_history:
        return agent_history

    last = agent_history[-1]
    if not (
        isinstance(last, dict)
        and last.get("role") == "assistant"
        and last.get("tool_calls")
    ):
        return agent_history

    logger.debug(
        "Stripping dangling unanswered assistant(tool_calls) tail "
        "(%d call(s)) — process likely killed mid-tool-call by a "
        "restart/shutdown command (#49201)",
        len(last.get("tool_calls") or []),
    )
    return agent_history[:-1]


def sanitize_replay_history(
    agent_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply both replay-tail strippers in the canonical order.

    Convenience entry point for resume code paths: removes interrupted
    assistant→tool blocks anywhere in the history, then removes a dangling
    unanswered ``assistant(tool_calls)`` tail.  Returns the same list object
    when there is nothing to strip.
    """
    if not agent_history:
        return agent_history
    return strip_dangling_tool_call_tail(strip_interrupted_tool_tails(agent_history))


# ──────────────────────────────────────────────────────────────────────
# Stale dangerous-confirmation text expiry (#59607)
# ──────────────────────────────────────────────────────────────────────

# How long a high-risk confirmation phrase remains valid.
# Short on purpose: dangerous side effects should not survive any restart
# or session resumption gap. The user can always re-confirm if needed.
_DANGEROUS_CONFIRMATION_EXPIRY_SECONDS = 60.0

# Confirmation phrases that unlock destructive host actions.
# Substring match (case-insensitive) so that user variants (e.g. trailing
# punctuation, additional context) still match. Add new patterns here when
# new high-risk actions are introduced.
_DANGEROUS_CONFIRMATION_PATTERNS: tuple = (
    "confirm forced restart",
    "confirm forced reboot",
    "confirm shutdown",
    "confirm reboot",
    "confirm power off",
    "yes, delete everything",
    "confirm wipe",
    "confirm factory reset",
    # i18n variants observed in the original incident
    "確認強制重開機",
    "確認強制重開",
    "確認重啟",
)

# Replacement text for an expired confirmation. Redacting in place (rather
# than deleting the message) preserves strict user/assistant role
# alternation in the replayed history.
_EXPIRED_CONFIRMATION_SENTINEL = (
    "[A high-risk confirmation previously given here has EXPIRED and must "
    "not be acted on. Ask the user to re-confirm explicitly before "
    "performing any destructive action.]"
)


def is_dangerous_confirmation(content: Any) -> bool:
    """Return True if a user-message text matches a known dangerous confirmation.

    Used by ``strip_stale_dangerous_confirmations`` to decide which
    transcript rows to expire. Substring + case-insensitive so that
    ``"Please confirm forced restart, the host is critical"`` still matches.
    """
    if not isinstance(content, str):
        return False
    text = content.strip().lower()
    return any(pattern in text for pattern in _DANGEROUS_CONFIRMATION_PATTERNS)


def strip_stale_dangerous_confirmations(
    agent_history: List[Dict[str, Any]],
    *,
    now: float,
    expiry_seconds: float = _DANGEROUS_CONFIRMATION_EXPIRY_SECONDS,
) -> List[Dict[str, Any]]:
    """Expire stale dangerous-confirmation text in user messages (#59607).

    When a high-risk side effect (e.g. host restart via ``shutdown.exe``)
    runs, the user's plain-text confirmation phrase is persisted in the
    conversation transcript.  If the host restart killed the gateway
    process before the assistant's tool result was written, the
    transcript tail ends on the assistant's text response — and the
    dangerous confirmation text remains in the user role.

    On the next inbound message — possibly a casual "are you there?" from
    the user minutes later — the LLM sees the stale confirmation and may
    interpret the new turn as a fresh re-confirmation, re-executing the
    destructive action.  This is the failure mode reported in #59607.

    Expired confirmations are REDACTED IN PLACE, not removed: deleting a
    user message from the incident tail (``user(confirm) →
    assistant("OK, restarting")``) would leave two consecutive assistant
    messages, violating the strict role-alternation invariant providers
    enforce.  The message survives with its role intact; only the trigger
    text is replaced by a sentinel that tells the model the confirmation
    has expired.

    Messages without a timestamp are left untouched (backward
    compatibility: legacy transcripts and in-memory test scaffolding have
    no timestamps).  User messages that contain dangerous confirmation
    text but are within the expiry window are also left untouched — they
    represent a fresh confirmation that has not yet been acted on.

    Complements 75ed07ace (which strips the *assistant* side of the
    broken tail) by handling the *user* side: a stale plain-text
    confirmation that the assistant has not yet responded to in a way
    the resume logic recognises.
    """
    if not agent_history:
        return agent_history

    cleaned: List[Dict[str, Any]] = []
    for msg in agent_history:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "user"
            and is_dangerous_confirmation(msg.get("content", ""))
        ):
            ts = msg.get("timestamp")
            if ts is not None and (now - float(ts)) > expiry_seconds:
                logger.debug(
                    "Redacting stale dangerous-confirmation text in user "
                    "message (age=%.1fs, expiry=%.1fs): %r",
                    now - float(ts),
                    expiry_seconds,
                    (msg.get("content") or "")[:80],
                )
                redacted = dict(msg)
                redacted["content"] = _EXPIRED_CONFIRMATION_SENTINEL
                cleaned.append(redacted)
                continue
        cleaned.append(msg)
    return cleaned
