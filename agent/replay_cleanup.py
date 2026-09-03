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

from agent.tool_dispatch_helpers import make_tool_result_message
from agent.tool_result_classification import tool_may_have_side_effect
from agent.turn_context import drop_stale_api_content

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
                calls = msg.get("tool_calls") or []
                if any(
                    tool_may_have_side_effect(
                        str((call.get("function") or {}).get("name") or "")
                    )
                    for call in calls
                ):
                    call_names = {
                        str(call.get("id") or call.get("call_id") or ""): str(
                            (call.get("function") or {}).get("name") or ""
                        )
                        for call in calls
                    }
                    cleaned.append(msg)
                    for tool_result in tool_results:
                        if not is_interrupted_tool_result(tool_result.get("content", "")):
                            cleaned.append(tool_result)
                            continue
                        recovered = dict(tool_result)
                        name = call_names.get(str(tool_result.get("tool_call_id") or ""), "")
                        recovered["effect_disposition"] = (
                            "unknown" if tool_may_have_side_effect(name) else "none"
                        )
                        recovered["content"] = (
                            "[Orphan recovery: interrupted side-effecting tool may have "
                            "executed; its effect is UNKNOWN. Inspect state before retrying.]"
                            if recovered["effect_disposition"] == "unknown"
                            else "[Orphan recovery: interrupted read-only tool did not complete.]"
                        )
                        cleaned.append(recovered)
                    i = j
                    continue
                logger.debug(
                    "Stripping interrupted read-only assistant→tool replay block "
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

    tool_calls = last.get("tool_calls") or []
    if any(
        tool_may_have_side_effect(
            str((call.get("function") or {}).get("name") or "")
        )
        for call in tool_calls
    ):
        recovered = list(agent_history)
        for call in tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name") or "unknown")
            call_id = str(call.get("id") or call.get("call_id") or "")
            disposition = "unknown" if tool_may_have_side_effect(name) else "none"
            content = (
                "[Orphan recovery: this tool may have executed before Hermes stopped; "
                "its effect is UNKNOWN. Inspect current state before retrying.]"
                if disposition == "unknown"
                else "[Orphan recovery: this read-only tool did not complete and had no effect.]"
            )
            recovered.append(make_tool_result_message(
                name, content, call_id, effect_disposition=disposition,
            ))
        logger.warning(
            "Recovered dangling side-effecting tool call(s) as UNKNOWN instead of erasing them"
        )
        return recovered

    logger.debug(
        "Stripping dangling unanswered read-only assistant(tool_calls) tail (%d call(s))",
        len(tool_calls),
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
# Durable orphan-result backfill (#lost-result-delivery)
# ──────────────────────────────────────────────────────────────────────

def collect_dangling_tool_call_ids(agent_history: List[Dict[str, Any]]) -> List[str]:
    """Return tool-call ids that have NO matching tool result, in history order.

    Unlike the tail strippers (which only fire on block tails with interrupt
    markers), this walks the FULL history: a session can accumulate dangling
    calls mid-transcript when the tool-delivery layer dies without writing
    the result row (e.g. a wedged pre-exec guard threads holding the GIL
    until the service is restarted). The resumed conversation sanitizes them
    in memory, but the durable transcript keeps the dangling call forever —
    and every TUI transcript view renders it as an eternally in-flight call.
    """
    dangling: List[str] = []
    seen: set = set()
    for msg in agent_history:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                cid = str(tc.get("id") or tc.get("call_id") or "").strip()
                if cid:
                    seen.add(cid)
        elif msg.get("role") == "tool":
            cid = str(msg.get("tool_call_id") or "").strip()
            seen.discard(cid)
    # preserve history order for the assistant calls, results appended after
    ordered: List[str] = []
    emitted: set = set()
    for msg in agent_history:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                cid = str(tc.get("id") or tc.get("call_id") or "").strip()
                if cid and cid in seen and cid not in emitted:
                    ordered.append(cid)
                    emitted.add(cid)
    return ordered


def backfill_orphan_tool_results(
    session_id: str,
    agent_history: List[Dict[str, Any]],
    session_db: Any,
) -> int:
    """Persist synthetic results for dangling tool calls — once, durably.

    The resume-time sanitizers (``strip_dangling_tool_call_tail`` /
    ``strip_interrupted_tool_tails``) repair the model-fed history in memory
    only; the synthetic ``[Orphan recovery: ...]`` rows never reach state.db,
    so the dangling call resurfaces on EVERY restart and the TUI transcript
    shows a never-ending in-flight tool call for that turn.

    This backfills the durable transcript: for each dangling call id it
    writes one synthetic tool row (same wording as the in-memory recovery,
    ``effect_disposition`` mirroring the stripper's side-effect logic) with
    the ORIGINAL assistant message's timestamp slot + 1µs so ordering stays
    below whatever the user sent next. Idempotent: calls that already have
    any tool row are skipped, so a re-run after a crash adds nothing.

    Returns the number of rows written. Never raises: a backfill failure
    must not fail the resume (the in-memory sanitizers still protect the
    model); the failure is logged and retried on the next resume.
    """
    if not agent_history or session_db is None:
        return 0
    try:
        dangling = [
            (cid, tool_name, ts_after)
            for cid, tool_name, ts_after in _dangling_with_metadata(
                agent_history
            )
            if not session_db.has_tool_result(session_id, cid)
        ]
        if not dangling:
            return 0
        written = 0
        for cid, tool_name, ts_after in dangling:
            content = (
                "[Orphan recovery: interrupted side-effecting tool may have "
                "executed; its effect is UNKNOWN. Inspect state before retrying.]"
                if tool_may_have_side_effect(tool_name)
                else "[Orphan recovery: interrupted read-only tool did not complete.]"
            )
            try:
                session_db.append_message(
                    session_id,
                    "tool",
                    content=content,
                    tool_name=tool_name or "unknown",
                    tool_call_id=cid,
                    effect_disposition=(
                        "unknown" if tool_may_have_side_effect(tool_name) else "none"
                    ),
                    timestamp=ts_after,
                )
                written += 1
                logger.info(
                    "Backfilled orphan tool result row for call %s (%s) in "
                    "session %s (durable lost-result repair)",
                    cid, tool_name, session_id,
                )
            except Exception:
                logger.exception(
                    "orphan-result backfill failed for call %s in session %s",
                    cid, session_id,
                )
        return written
    except Exception:
        logger.exception("orphan-result backfill scan failed; skipping repair")
        return 0


def _dangling_with_metadata(agent_history):
    """Yield (call_id, tool_name, timestamp_after_assistant) for dangling ids."""
    # First pass: collect which ids have results at all
    have_result: set = set()
    for msg in agent_history:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            cid = str(msg.get("tool_call_id") or "").strip()
            if cid:
                have_result.add(cid)
    # Second pass: walk in order, remember the assistant ts for each dangling id
    for msg in agent_history:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        calls = msg.get("tool_calls") or []
        if not calls:
            continue
        ts = msg.get("timestamp")
        for tc in calls:
            if not isinstance(tc, dict):
                continue
            cid = str(tc.get("id") or tc.get("call_id") or "").strip()
            if not cid or cid in have_result:
                continue
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            yield cid, name, ts


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
                # Drop the api_content sidecar: it carries the exact bytes
                # previously sent — i.e. the dangerous confirmation this
                # redaction exists to expire. Replaying it verbatim would
                # undo the redaction on the wire.
                drop_stale_api_content(redacted)
                cleaned.append(redacted)
                continue
        cleaned.append(msg)
    return cleaned
