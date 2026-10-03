"""``reasoning.available`` carries extracted reasoning, never the reply itself (#118934, #13007).

``normalize_model_response`` relays every non-empty assistant reply through ``_relay_thinking``.
The relay used to strip reasoning *delimiters* out of the whole reply and forward whatever was
left, so a model that answered in plain text (no ``<think>``/``<REASONING_SCRATCHPAD>`` block)
had its entire answer re-emitted as a ``reasoning.available`` block. Desktop and TUI render that
event as the turn's thinking block, which produced the two symptoms in #118934: the answer
rendered twice, and internal-looking reasoning content drawn straight from the visible reply.

The relay must fire only with genuinely extracted reasoning — inline think blocks or the
structured ``reasoning``/``reasoning_content``/``reasoning_details`` fields — and stay silent
otherwise. The subagent ``_thinking`` first-line preview is a separate lane and keeps working.
"""
from types import SimpleNamespace

from agent.agent_runtime_helpers import extract_reasoning
from agent.turn_response_intake import _relay_thinking, normalize_model_response


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

    def __init__(self, delegate_depth: int = 0):
        self._delegate_depth = delegate_depth
        self.calls = []
        self.tool_progress_callback = lambda *args: self.calls.append(args)
        self._extract_reasoning = lambda msg: extract_reasoning(self, msg)

    def _get_transport(self):
        return _Transport()

    def _vprint(self, *args, **kwargs):
        pass

    @property
    def reasoning_calls(self):
        return [c for c in self.calls if c[0] == "reasoning.available"]


def _run(agent, *, content, reasoning=None, reasoning_content=None):
    message = SimpleNamespace(
        content=content, finish_reason="stop", tool_calls=None, reasoning=reasoning,
        reasoning_content=reasoning_content, reasoning_details=None,
    )
    normalize_model_response(
        agent, response=message, messages=[], api_messages=[], conversation_history={},
        api_call_count=1, api_duration=0.0, api_start_time=0.0, api_request_id="r1",
        effective_task_id="t1", turn_id="t1",
    )
    return agent.calls


def test_plain_reply_text_is_not_relayed_as_reasoning():
    """A reply with no reasoning blocks must produce no ``reasoning.available`` at all."""
    agent = _Agent()
    calls = _run(agent, content="The answer is 42.")
    assert calls == []


def test_inline_reasoning_block_is_relayed_without_the_answer():
    """Only the text inside the think block is reasoning; the visible reply is not."""
    agent = _Agent()
    calls = _run(
        agent,
        content="<REASONING_SCRATCHPAD>plan step one</REASONING_SCRATCHPAD>\nThe answer is 42.",
    )
    assert calls == [("reasoning.available", "_thinking", "plan step one", None)]


def test_structured_reasoning_field_is_relayed_verbatim():
    """Non-streaming providers hand reasoning over as a field, not as content tags."""
    agent = _Agent()
    calls = _run(agent, content="Plain reply.", reasoning="secret plan")
    assert calls == [("reasoning.available", "_thinking", "secret plan", None)]


def test_reply_with_structured_reasoning_does_not_duplicate_the_reply():
    """The reply itself never rides along in the reasoning preview, even when reasoning exists."""
    agent = _Agent()
    calls = _run(
        agent,
        content="<think>scratch</think>\nVisible answer",
        reasoning_content="scratch",
    )
    assert len(agent.reasoning_calls) == 1
    preview = agent.reasoning_calls[0][2]
    assert "Visible answer" not in preview


def test_subagent_first_line_preview_still_reaches_the_parent():
    """Delegated agents keep their first-line ``_thinking`` progress preview."""
    agent = _Agent(delegate_depth=1)
    calls = _run(agent, content="First line of subagent work\nsecond line")
    assert calls == [("_thinking", "First line of subagent work")]


def test_relay_is_silent_without_a_callback():
    """No structured callback registered → nothing to relay, nothing raised."""
    agent = _Agent()
    agent.tool_progress_callback = None
    _relay_thinking(agent, "<think>hush</think>ok", SimpleNamespace(content="<think>hush</think>ok"))
    assert agent.calls == []
