"""Scripted prelude: tool calls a prompt built-in plays before the turn's first model call.

A prelude is a generator owned by the built-in (``agent.initiate_setup_prompt.initiate_setup_prelude``
is the one today). Each step it yields ``(content, tool_name, args)``: the assistant text and the one
tool call that row carries. The loop runs it through the unchanged ``run_tool_round`` (row persisted
before the tool runs, interim text and tool events emitted, tool hooks fired) and sends the tool's
result string back into the generator, so a later step can use an earlier answer.

It runs before any provider call, so the rows sit between this turn's user message and the first
model call: nothing cached is mutated, the system prompt is untouched, no user row is added, and
roles alternate ``user -> assistant(tool_calls) -> tool -> ... -> assistant(model)``. No API call is
counted and no ``pre/post_api_request`` hook fires for a step.
"""

from __future__ import annotations

import json
from typing import Any, Generator, Optional, Tuple

from agent.message_sanitization import deterministic_call_id
from agent.transports.types import NormalizedResponse, build_tool_call
from agent.turn_tool_round import run_tool_round

Prelude = Generator[tuple[str, str, dict], Optional[str], None]


def run_scripted_prelude(agent: Any, s: Any, prelude: Prelude) -> Any:
    """Play ``prelude`` into the loop state ``s``; returns the last tool round's verdict, or None when
    no step ran. The caller ends the turn on ``return`` and skips the model on ``break``."""
    from agent.conversation_loop import _run_phase

    verdict, result = None, None
    while not agent._interrupt_requested:
        try:
            content, name, args = prelude.send(result)
        except StopIteration:
            break
        s.finish_reason = "tool_calls"
        s.assistant_message = NormalizedResponse(
            content=content, finish_reason="tool_calls",
            # Seeded by the card and its place in history: unique within the session, byte-stable on replay.
            tool_calls=[build_tool_call(deterministic_call_id(name, json.dumps(args, sort_keys=True), len(s.messages)),
                                        name, args)])
        verdict = _run_phase(run_tool_round, agent, s)
        if verdict.action != "continue":
            break
        tail = s.messages[-1] if s.messages else {}
        result = tail.get("content") if tail.get("role") == "tool" else None
    prelude.close()
    return verdict


def play_prelude(agent: Any, s: Any, prelude: Optional[Prelude]) -> tuple[str, Any]:
    """``("return", result)`` ends the turn now, ``("break", None)`` skips the model call, ``("run", None)`` runs
    the loop as usual (also when there is no prelude)."""
    verdict = run_scripted_prelude(agent, s, prelude) if prelude is not None else None
    if verdict is not None and verdict.action in ("return", "break"):
        return verdict.action, verdict.result
    return "run", None
