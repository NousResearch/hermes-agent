"""``reasoning.available`` carries extracted reasoning, never the reply text (#132477).

``normalize_model_response`` relays every assistant response through ``_relay_thinking``. The old
relay stripped reasoning *delimiters* out of the whole reply and forwarded whatever was left, so a
structured-reasoning model answering in plain text (no ``<think>``/``<REASONING_SCRATCHPAD>``
block) had its entire answer re-emitted as ``reasoning.available``. Desktop's ``message-stream.ts``
handles that event with ``appendReasoningDelta(text, replace=true)`` and renders it as the turn's
thinking block — on top of the native reasoning already streamed via ``reasoning.delta`` — so
reply fragments rendered twice (#132477).

The relay must fire only with genuinely extracted reasoning (structured ``reasoning`` /
``reasoning_content`` / ``reasoning_details`` fields or inline think blocks), stay silent when the
response already streamed native reasoning deltas, and keep the subagent ``_thinking`` first-line
preview intact.
"""
from types import SimpleNamespace
from typing import Any

from agent.agent_runtime_helpers import extract_reasoning
from agent.turn_response_intake import normalize_model_response


class _Transport:
    def normalize_response(self, response, strip_tool_prefix=False):
        return response


class _Agent:
    """Minimal stand-in exposing exactly what the intake path touches."""

    api_mode = "openai_chat"
    quiet_mode = True
    verbose_logging = False
    log_prefix = ""
    _delegate_depth = 0
    _incomplete_scratchpad_retries = 0
    _native_reasoning_streamed = False

    def __init__(self, delegate_depth: int = 0):
        self._delegate_depth = delegate_depth
        self.calls = []
        # Typed loosely so a test can swap in ``None`` (no callback registered).
        self.tool_progress_callback: Any = lambda *args: self.calls.append(args)
        self._extract_reasoning = lambda msg: extract_reasoning(self, msg)

    def _get_transport(self):
        return _Transport()

    def _vprint(self, *args, **kwargs):
        pass


def _run(agent, *, content, tool_calls=None, reasoning=None, reasoning_content=None):
    message = SimpleNamespace(
        content=content, finish_reason="stop", tool_calls=tool_calls, reasoning=reasoning,
        reasoning_content=reasoning_content, reasoning_details=None,
    )
    normalize_model_response(
        agent, response=message, messages=[], api_messages=[], conversation_history={},
        api_call_count=1, api_duration=0.0, api_start_time=0.0, api_request_id="r1",
        effective_task_id="t1", turn_id="t1",
    )
    return agent.calls


def test_plain_reply_is_not_relayed_as_reasoning():
    """#132477: a reply with no reasoning blocks must emit no ``reasoning.available`` at all."""
    agent = _Agent()
    calls = _run(agent, content="The answer is 42.")
    assert calls == []


def test_inline_thinking_block_is_relayed_without_the_reply():
    """Only the text inside the think block is reasoning; the visible reply is not."""
    agent = _Agent()
    calls = _run(
        agent,
        content="<REASONING_SCRATCHPAD>plan step one</REASONING_SCRATCHPAD>\nThe answer is 42.",
    )
    assert calls == [("reasoning.available", "_thinking", "plan step one", None)]
