"""Replay-history sanitization shared by EVERY resume surface (messaging gateway, TUI/WebUI gateway).

A turn that died mid-tool-loop (restart command, stale timeout, interrupt before the result was written)
persists a dangling ``assistant(tool_calls)`` or interrupted ``assistant→tool`` tail; on resume the model
re-issues the unanswered call → endless "thinking"/reboot loop. These pure helpers strip those tails."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agent.tool_dispatch_helpers import make_tool_result_message
from agent.tool_result_classification import tool_may_have_side_effect
from agent.turn_context import drop_stale_api_content

logger = logging.getLogger(__name__)

# Orphan-recovery notices: (side-effecting, read-only) for an interrupted block vs a dangling tail.
_INTERRUPTED_NOTICES = (
    "[Orphan recovery: interrupted side-effecting tool may have executed; its effect is UNKNOWN. Inspect state before retrying.]",
    "[Orphan recovery: interrupted read-only tool did not complete.]",
)
_DANGLING_NOTICES = (
    "[Orphan recovery: this tool may have executed before Hermes stopped; its effect is UNKNOWN. Inspect current state before retrying.]",
    "[Orphan recovery: this read-only tool did not complete and had no effect.]",
)


def is_interrupted_tool_result(content: Any) -> bool:
    """Return True if a tool result indicates the tool was interrupted."""
    if not isinstance(content, str):
        return False
    lowered = content.lower()
    return "[command interrupted]" in lowered or ("exit_code" in lowered and ("130" in lowered or "-1" in lowered) and "interrupt" in lowered)


def _call_name(call: Dict[str, Any]) -> str:
    return str((call.get("function") or {}).get("name") or "")


def _call_id(call: Dict[str, Any]) -> str:
    return str(call.get("id") or call.get("call_id") or "")


def _any_side_effecting(calls: List[Dict[str, Any]]) -> bool:
    return any(tool_may_have_side_effect(_call_name(call)) for call in calls)


def _orphan_recovery(name: str, notices: tuple) -> tuple:
    """(effect_disposition, content) for an interrupted/dangling call named ``name``."""
    if tool_may_have_side_effect(name):
        return "unknown", notices[0]
    return "none", notices[1]


def strip_interrupted_tool_tails(agent_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip interrupted assistant→tool blocks anywhere in history (a queued user message may follow one).
    Read-only blocks are dropped; blocks with a side-effecting call are KEPT with the interrupted results
    rewritten as orphan-recovery notices, since the effect may have happened."""
    if not agent_history:
        return agent_history
    cleaned: List[Dict[str, Any]] = []
    i, n = 0, len(agent_history)
    while i < n:
        msg = agent_history[i]
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            j = i + 1
            while j < n and agent_history[j].get("role") == "tool":
                j += 1
            tool_results = agent_history[i + 1:j]
            if any(is_interrupted_tool_result(m.get("content", "")) for m in tool_results):
                calls = msg.get("tool_calls") or []
                if _any_side_effecting(calls):
                    call_names = {_call_id(call): _call_name(call) for call in calls}
                    cleaned.append(msg)
                    for tool_result in tool_results:
                        if is_interrupted_tool_result(tool_result.get("content", "")):
                            name = call_names.get(str(tool_result.get("tool_call_id") or ""), "")
                            disposition, content = _orphan_recovery(name, _INTERRUPTED_NOTICES)
                            tool_result = {**tool_result, "effect_disposition": disposition, "content": content}
                        cleaned.append(tool_result)
                else:
                    logger.debug("Stripping interrupted read-only assistant→tool replay block (indices %d–%d, tool_results=%d)",
                                 i, j - 1, len(tool_results))
                i = j
                continue
        if msg.get("role") == "tool" and is_interrupted_tool_result(msg.get("content", "")):
            logger.debug("Stripping orphan interrupted tool result from replay history")
        else:
            cleaned.append(msg)
        i += 1
    return cleaned


def strip_dangling_tool_call_tail(agent_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip a trailing ``assistant(tool_calls)`` with NO answers — a call that killed the gateway itself
    (``docker restart``) left zero ``tool`` rows, invisible to ``strip_interrupted_tool_tails``. A partially
    answered block still resumes. Read-only tails are dropped; side-effecting ones get UNKNOWN-effect results.

    On resume the model sees an unanswered tool call at the tail and naturally re-issues it — which restarts
    the gateway again, producing the infinite reboot loop in #49201. ``strip_interrupted_tool_tails`` does
    not catch this because there is no tool result to inspect for an interrupt marker.
    """
    if not agent_history:
        return agent_history
    last = agent_history[-1]
    if not (isinstance(last, dict) and last.get("role") == "assistant" and last.get("tool_calls")):
        return agent_history
    tool_calls = last.get("tool_calls") or []
    if _any_side_effecting(tool_calls):
        recovered = list(agent_history)
        for call in tool_calls:
            name = _call_name(call) or "unknown"
            disposition, content = _orphan_recovery(name, _DANGLING_NOTICES)
            recovered.append(make_tool_result_message(name, content, _call_id(call), effect_disposition=disposition))
        logger.warning("Recovered dangling side-effecting tool call(s) as UNKNOWN instead of erasing them")
        return recovered
    logger.debug("Stripping dangling unanswered read-only assistant(tool_calls) tail (%d call(s))", len(tool_calls))
    return agent_history[:-1]


def sanitize_replay_history(agent_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Both strippers in canonical order (interrupted blocks, then dangling tail); same list object when nothing strips."""
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


# Short on purpose: a dangerous confirmation must not survive any restart or resume gap.
# ────────────────────────────────────────────────────────────────────── Stale dangerous-confirmation text
# expiry (#59607) ──────────────────────────────────────────────────────────────────────
_DANGEROUS_CONFIRMATION_EXPIRY_SECONDS = 60.0

# Phrases that unlock destructive host actions; case-insensitive substring match so trailing punctuation /
# extra context still matches. Includes i18n variants from the original incident.
_DANGEROUS_CONFIRMATION_PATTERNS: tuple = (
    "confirm forced restart", "confirm forced reboot", "confirm shutdown", "confirm reboot", "confirm power off",
    "yes, delete everything", "confirm wipe", "confirm factory reset",
    "確認強制重開機", "確認強制重開", "確認重啟",
)

# Redacting in place (not deleting the message) preserves strict user/assistant alternation in the replay.
_EXPIRED_CONFIRMATION_SENTINEL = (
    "[A high-risk confirmation previously given here has EXPIRED and must "
    "not be acted on. Ask the user to re-confirm explicitly before "
    "performing any destructive action.]"
)


def is_dangerous_confirmation(content: Any) -> bool:
    """True if user-message text contains a known dangerous confirmation phrase."""
    return isinstance(content, str) and any(pattern in content.strip().lower() for pattern in _DANGEROUS_CONFIRMATION_PATTERNS)


def strip_stale_dangerous_confirmations(
    agent_history: List[Dict[str, Any]],
    *,
    now: float,
    expiry_seconds: float = _DANGEROUS_CONFIRMATION_EXPIRY_SECONDS,
) -> List[Dict[str, Any]]:
    """Redact IN PLACE dangerous-confirmation text older than ``expiry_seconds`` in user messages: a confirmation
    surviving a restart reads as a fresh re-confirmation minutes later. Untimestamped messages (legacy
    transcripts, test scaffolding) are left untouched.

    See #59607.
    On the next inbound message — possibly a casual "are you there?" from the user minutes later — the LLM
    sees the stale confirmation and may interpret the new turn as a fresh re-confirmation, re-executing the
    destructive action. This is the failure mode reported in #59607.
    """
    if not agent_history:
        return agent_history
    cleaned: List[Dict[str, Any]] = []
    for msg in agent_history:
        ts = msg.get("timestamp") if isinstance(msg, dict) and msg.get("role") == "user" else None
        if ts is None or not is_dangerous_confirmation(msg.get("content", "")) or (now - float(ts)) <= expiry_seconds:
            cleaned.append(msg)
            continue
        logger.debug(
            "Redacting stale dangerous-confirmation text in user message (age=%.1fs, expiry=%.1fs): %r",
            now - float(ts), expiry_seconds, (msg.get("content") or "")[:80],
        )
        redacted = dict(msg)
        redacted["content"] = _EXPIRED_CONFIRMATION_SENTINEL
        # The api_content sidecar carries the exact bytes sent — the confirmation itself; replaying it would undo the redaction.
        drop_stale_api_content(redacted)
        cleaned.append(redacted)
    return cleaned
