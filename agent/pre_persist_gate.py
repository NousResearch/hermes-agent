"""Shared pre-persist gate application for the turn loop's durable-write chokepoint.

An unvalidated answer must not become durable (SessionDB + this agent's own in-memory
history feeding the NEXT turn) before an injected host validator has approved it. The
gateway wires ``agent._pre_persist_gate`` / ``agent._pre_persist_gate_metadata`` from its
own ``pre_delivery_gate`` (see gateway/run_turn_runner.py, gateway/run.py's
``_init_evaluator_shadow``) only when ``gateway.evaluator_shadow.mode`` is ``"strict"``;
every other caller (CLI, TUI, cron) and shadow-mode gateway runs leave the hook unset, so
this is a no-op for them.

Called from agent/turn_finalizer.py::finalize_turn's persist step — the single point every
turn-loop exit path converges on before its durable write.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("agent.conversation_loop")

# Matches the gateway's own strict-mode delivery placeholder (agent/turn_finalizer.py) so a
# blocked answer reads identically whether the user sees it as the platform delivery, the
# durable transcript row, or the context the model itself will see on the next turn.
PRE_PERSIST_WITHHELD_TEXT = "⚠️ Response withheld because pre-delivery validation did not pass."


def apply_pre_persist_gate(
    agent: Any, final_response: Optional[str], *, target_msg: Optional[dict] = None,
) -> Optional[str]:
    """Evaluate ``final_response`` through ``agent._pre_persist_gate`` before it is persisted.

    Returns the (possibly withheld) text. When the verdict changes the text and
    ``target_msg`` is given (the transcript row about to be persisted, so its ``content``
    stays in lockstep with the returned value), ``target_msg["content"]`` is updated in
    place.

    A gate exception or a non-"passed" verdict is a safe stop — the withheld placeholder,
    never the unvalidated text — mirroring the fail-closed contract in
    gateway/evaluator_shadow.py (a returncode/schema mismatch already returns
    "inconclusive" there; both "blocked" and "inconclusive" are withheld here).
    """
    gate = getattr(agent, "_pre_persist_gate", None)
    if gate is None or not isinstance(final_response, str) or not final_response.strip():
        return final_response
    try:
        outcome = gate(final_response)
        decision = outcome.get("decision") if isinstance(outcome, dict) else None
    except Exception:
        logger.warning("pre-persist gate invocation failed — withholding as a safe stop", exc_info=True)
        decision = None
    if decision != "passed":
        gated_text = PRE_PERSIST_WITHHELD_TEXT
        if gated_text != final_response and isinstance(target_msg, dict):
            target_msg["content"] = gated_text
        return gated_text
    return final_response
