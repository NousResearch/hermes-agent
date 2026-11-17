"""Replay-history sanitization shared by EVERY resume surface (messaging gateway, TUI/WebUI gateway).

A turn that died mid-tool-loop (restart command, stale timeout, interrupt before the result was written)
persists a dangling ``assistant(tool_calls)`` or interrupted ``assistant→tool`` tail; on resume the model
re-issues the unanswered call → endless "thinking"/reboot loop. These pure helpers strip those tails."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional

from agent.message_sanitization import coalesce_tool_call_id, tool_call_id_variants, tool_result_id_variants
from agent.tool_dispatch_helpers import make_tool_result_message
from agent.tool_result_classification import tool_may_have_side_effect
from agent.message_metadata import drop_typed_ask
from agent.turn_context import drop_stale_api_content
from hermes_cli.timefmt import coerce_epoch

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


# Every executor ends the killed run's ``output`` with a bracketed marker line: "[Command
# interrupted]" (tools/environments/, exit 130), "[Command interrupted - Modal ...]"
# (managed_modal.py, exit 130), "[execution interrupted ...]" (code_execution_tool.py, exit -1).
_INTERRUPT_MARKER_LINE = re.compile(r"^\[(?:command|execution) interrupted\b[^\n]*\]\s*$", re.IGNORECASE)


def is_interrupted_tool_result(content: Any) -> bool:
    """True only when the result has the executor's interrupt SHAPE: the marker is the last
    line of the output (JSON envelope with a non-zero exit code, or a bare text result). A
    marker quoted inside successful output — a grep hit, a doc example — is ordinary data;
    this runs on every live request, so a false positive rewrites real tool output."""
    if not isinstance(content, str):
        return False
    output = content
    if content.lstrip().startswith("{"):
        try:
            envelope = json.loads(content)
        except ValueError:
            return False
        if not isinstance(envelope, dict) or envelope.get("exit_code") in (0, None):
            return False
        output = envelope.get("output")
        if not isinstance(output, str):
            return False
    last_line = output.rstrip().rsplit("\n", 1)[-1]
    return _INTERRUPT_MARKER_LINE.match(last_line) is not None


def _call_name(call: dict[str, Any]) -> str:
    return str((call.get("function") or {}).get("name") or "")


def _any_side_effecting(calls: list[dict[str, Any]]) -> bool:
    return any(tool_may_have_side_effect(_call_name(call)) for call in calls)


def _orphan_recovery(name: str, notices: tuple) -> tuple:
    """(effect_disposition, content) for an interrupted/dangling call named ``name``."""
    if tool_may_have_side_effect(name):
        return "unknown", notices[0]
    return "none", notices[1]


def strip_interrupted_tool_tails(agent_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip interrupted assistant→tool blocks anywhere in history (a queued user message may follow one).
    Read-only blocks are dropped; blocks with a side-effecting call are KEPT with the interrupted results
    rewritten as orphan-recovery notices, since the effect may have happened."""
    if not agent_history:
        return agent_history
    cleaned: list[dict[str, Any]] = []
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
                    # Same pairing rule as the request sanitizer: any id spelling of the call matches.
                    call_names = {v: _call_name(call) for call in calls for v in tool_call_id_variants(call)}
                    cleaned.append(msg)
                    for tool_result in tool_results:
                        if is_interrupted_tool_result(tool_result.get("content", "")):
                            name = next((call_names[v] for v in tool_result_id_variants(tool_result.get("tool_call_id"))
                                         if v in call_names), "")
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


def strip_dangling_tool_call_tail(agent_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    results = unanswered_call_results(tool_calls)
    if results is not None:
        logger.warning("Recovered dangling side-effecting tool call(s) as UNKNOWN instead of erasing them")
        return list(agent_history) + results
    logger.debug("Stripping dangling unanswered read-only assistant(tool_calls) tail (%d call(s))", len(tool_calls))
    return agent_history[:-1]


def unanswered_call_results(calls: list[dict[str, Any]]) -> Optional[list[dict[str, Any]]]:
    """Orphan-recovery results for ``calls`` that never got an answer, or None when every call is
    read-only (erasing those loses nothing). One side-effecting call keeps the whole set: the model
    must learn the action may have happened, or it repeats it."""
    if not _any_side_effecting(calls):
        return None
    results = []
    for call in calls:
        name = _call_name(call) or "unknown"
        disposition, content = _orphan_recovery(name, _DANGLING_NOTICES)
        # ``call_id`` (not the Responses item ``id``) is what the provider pairs the output with.
        results.append(make_tool_result_message(name, content, coalesce_tool_call_id(call), effect_disposition=disposition))
    return results


def recover_unanswered_side_effect_calls(messages: list[dict]) -> tuple[list[dict], int]:
    """Repair pass before unanswered tool_calls are pruned: a call that may have had a side effect and
    got no result in its immediately following tool run is answered with an UNKNOWN-effect result
    after that run. A killed turn leaves exactly this shape (the call row is persisted before the tool
    runs), and erasing it lets the resumed model repeat an action that already happened (#49201). The
    result is a projection of the durable rows, regenerated on every load, so it is never flushed.
    Codex interims are exempt, as from the prune."""
    from agent.agent_runtime_helpers import _is_codex_interim
    from agent.context_compressor import _DB_PERSISTED_MARKER

    recovered_after: dict[int, list[dict]] = {}
    for i, msg in enumerate(messages):
        if not (isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls")
                and not _is_codex_interim(msg)):
            continue
        answered: set = set()
        run_end = i + 1
        while run_end < len(messages) and isinstance(messages[run_end], dict) and messages[run_end].get("role") == "tool":
            tid = (messages[run_end].get("tool_call_id") or "").strip()
            if tid:
                answered.update(tool_result_id_variants(tid))
            run_end += 1
        unanswered = [
            tc for tc in msg["tool_calls"]
            if isinstance(tc, dict) and not tool_call_id_variants(tc) & answered and coalesce_tool_call_id(tc)
        ]
        results = unanswered_call_results(unanswered) if unanswered else None
        if results:
            for result in results:
                result[_DB_PERSISTED_MARKER] = True
            recovered_after[run_end] = results
    repairs = len(recovered_after)
    if not repairs:
        return messages, 0
    repaired: list[dict] = []
    for i, msg in enumerate(messages):
        repaired.extend(recovered_after.pop(i, ()))
        repaired.append(msg)
    repaired.extend(recovered_after.pop(len(messages), ()))
    return repaired, repairs


def sanitize_replay_history(agent_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Both strippers in canonical order (interrupted blocks, then dangling tail); same list object when nothing strips."""
    if not agent_history:
        return agent_history
    return strip_dangling_tool_call_tail(strip_interrupted_tool_tails(agent_history))


def canonicalize_replay_history(
    agent_history: list[dict[str, Any]], *, now: Optional[float] = None
) -> list[dict[str, Any]]:
    """Apply every destructive replay transform in the shared, fixed order.

    Resume surfaces and the send path must serialize the same history bytes, or a
    resumed request diverges in the middle of the cached prefix.

    The input is never modified. ``now`` is the expiry clock; the send path passes the
    turn's admission time so every request in one turn sees the same bytes.
    """
    if not agent_history:
        return agent_history
    if now is None:
        now = time.time()
    cleaned = strip_interrupted_tool_tails(agent_history)
    cleaned = strip_dangling_tool_call_tail(cleaned)
    return strip_stale_dangerous_confirmations(cleaned, now=now)


# --- Stale dangerous-confirmation text expiry ---

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
    if not isinstance(content, str):
        return False
    lowered = content.strip().lower()
    return any(pattern in lowered for pattern in _DANGEROUS_CONFIRMATION_PATTERNS)


def strip_stale_dangerous_confirmations(
    agent_history: list[dict[str, Any]],
    *,
    now: float,
    expiry_seconds: float = _DANGEROUS_CONFIRMATION_EXPIRY_SECONDS,
) -> list[dict[str, Any]]:
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
    cleaned: list[dict[str, Any]] = []
    for msg in agent_history:
        ts = msg.get("timestamp") if isinstance(msg, dict) and msg.get("role") == "user" else None
        if ts is None or not is_dangerous_confirmation(msg.get("content", "")):
            cleaned.append(msg)
            continue
        # A present-but-untrustworthy stamp (corrupt, or issued in the future relative to
        # the admission clock) is treated as expired: its age is unknowable, and keeping the
        # text (plus its api_content sidecar) would replay a live confirmation.
        ts_f = coerce_epoch(ts, field="message timestamp")
        age = math.inf if ts_f is None else now - ts_f
        if 0 <= age <= expiry_seconds:
            cleaned.append(msg)
            continue
        logger.debug(
            "Redacting stale dangerous-confirmation text in user message (age=%.1fs, expiry=%.1fs): %r",
            age, expiry_seconds, (msg.get("content") or "")[:80],
        )
        redacted = dict(msg)
        redacted["content"] = _EXPIRED_CONFIRMATION_SENTINEL
        # The api_content sidecar carries the exact bytes sent — the confirmation itself; replaying it would undo the redaction.
        drop_stale_api_content(redacted)
        drop_typed_ask(redacted)  # the stamp holds the same confirmation text
        cleaned.append(redacted)
    return cleaned
