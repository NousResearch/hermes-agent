"""The clarify tool's whole question batch for a classic-CLI viewer's turn, in one request.

Main's HermesCLI answered ``clarify_callback(questions)`` with its batch panel: every question shown
at once, one active, answers revisitable until all are locked, ``{answers: {qid: raw | None},
outcome}``. The messaging path asks one card per question. For a classic-CLI local viewer the owner
sends the batch as the existing ``clarify`` server→client request (``ClarifyRequestParams`` →
``ClarifyResult``) through the same request/response machinery as the sudo prompt, so main's own panel
answers it unchanged, and blocks until it answers, the turn is interrupted, or the clarify timeout
(the same config both sides read) passes.
"""
from __future__ import annotations

_QUESTION_KEYS = ("qid", "question", "choices", "multi_select")


def ask_clarify_batch(turn, questions) -> dict:
    from gateway.session_sudo_prompt import ask_viewer_result
    from tools.clarify_gateway import get_clarify_timeout
    authority, session_id, _generation = turn._approval_owner
    live = authority.sessions.get(session_id)
    agent = (getattr(turn._ctx, "agent_holder", None) or [None])[0]
    interrupted = (lambda: bool(getattr(agent, "_interrupt_requested", False))) if agent is not None else (
        lambda: False)
    if live is None:
        return {"answers": {}, "outcome": "cancelled"}
    wire = [{key: entry.get(key) for key in _QUESTION_KEYS if entry.get(key) is not None} for entry in questions]
    timeout = get_clarify_timeout()
    result = ask_viewer_result(live.event_stream, session_id, "clarify", {"questions": wire},
                               cancelled=interrupted, timeout=timeout if timeout > 0 else None)
    if result is None:
        return {"answers": {}, "outcome": "cancelled" if interrupted() else "timed_out"}
    answers = result.get("answers")
    if not isinstance(answers, dict):
        return {"answers": {}, "outcome": "cancelled"}
    known = {entry["qid"] for entry in questions}
    clean = {qid: (value if isinstance(value, str) else None) for qid, value in answers.items() if qid in known}
    outcome = result.get("outcome") if result.get("outcome") in ("submitted", "timed_out", "cancelled") else "submitted"
    return {"answers": clean, "outcome": outcome}
