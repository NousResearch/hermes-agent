"""#158684: a summarizer endpoint that refuses role-typed turns gets one flat retry on the SAME route.

The typed shape is a security property (#146750), so it stays the default; a route that answers
"tool calls and tool messages are not supported" takes the flat transcript instead of dropping every
compaction onto the session's main model.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor

TURNS = [
    {"role": "user", "content": "run echo 42"},
    {"role": "assistant", "content": "calling", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{\"command\": \"echo 42\"}"}},
    ]},
    {"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "42"},
    {"role": "assistant", "content": "the answer is 42"},
]


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="main-model", provider="main-provider", quiet_mode=True,
            base_url="https://main.example/v1", api_key="***",
            summary_model_override="tool-less-summarizer",
        )


def _tool_shape_400():
    error = Exception(
        "Error code: 400 - {'error': {'message': 'Tool calls and tool messages are not supported', "
        "'code': 400, 'metadata': {'provider_name': 'ScaleDown'}}}"
    )
    error.status_code = 400
    return error


def _ok(text="FLAT SUMMARY OK"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


def _refuses_typed():
    """Records each attempt's roles; refuses any request carrying a tool row."""
    seen = []

    def _call(**kwargs):
        roles = [m["role"] for m in kwargs["messages"]]
        seen.append(roles)
        if "tool" in roles:
            raise _tool_shape_400()
        return _ok()

    return seen, _call


def test_tool_shape_400_retries_flat_on_the_same_route():
    compressor = _compressor()
    seen, call = _refuses_typed()
    with patch("agent.context_compressor.call_llm", side_effect=call):
        summary = compressor._generate_summary(TURNS)

    assert summary is not None and "FLAT SUMMARY OK" in summary
    # Two _generate_summary attempts: typed, then flat. On the wire a typed attempt is itself two
    # requests (streaming, then the non-stream resend), which is why the stub-endpoint run logs three.
    assert len(seen) == 2, seen
    assert "tool" in seen[0] and "tool" not in seen[1]
    # Same aux route: no main-model fallback was armed.
    assert compressor.summary_model == "tool-less-summarizer"
    assert compressor._summary_model_fallen_back is False


@pytest.mark.parametrize("change", [
    {"summary_model": ""},          # fallback to main clears the aux model
    {"provider": "other-provider"},
    {"base_url": "https://other.example/v1"},
])
def test_flat_shape_belongs_to_the_route_that_refused_it(change):
    """Any component of the route identity changing gets the typed shape again."""
    compressor = _compressor()
    seen, call = _refuses_typed()
    with patch("agent.context_compressor.call_llm", side_effect=call):
        compressor._generate_summary(TURNS)
        seen.clear()
        for attr, value in change.items():
            setattr(compressor, attr, value)
        compressor._generate_summary(TURNS)

    assert "tool" in seen[0], seen


def test_rejected_shape_stays_flat_for_the_next_compaction():
    compressor = _compressor()
    seen, call = _refuses_typed()
    with patch("agent.context_compressor.call_llm", side_effect=call):
        compressor._generate_summary(TURNS)
        seen.clear()
        compressor._generate_summary(TURNS)

    assert len(seen) == 1, seen
    assert "tool" not in seen[0]


def test_flat_mode_400_falls_back_without_recursing():
    """`tail_mode: legacy` has nothing left to flatten: one rejected attempt, then today's fallback."""
    compressor = _compressor()
    compressor.tail_mode = "legacy"
    models = []

    def _call(**kwargs):
        models.append(kwargs.get("model"))
        raise _tool_shape_400()

    with patch("agent.context_compressor.call_llm", side_effect=_call):
        summary = compressor._generate_summary(TURNS)

    assert summary is None
    assert models == ["tool-less-summarizer", "main-model"]


def test_other_400_keeps_the_main_model_fallback():
    compressor = _compressor()
    models = []

    def _call(**kwargs):
        models.append(kwargs.get("model"))
        if len(models) == 1:
            raise_too_long = Exception("Error code: 400 - This model's maximum context length is 8192 tokens")
            raise_too_long.status_code = 400
            raise raise_too_long
        return _ok("MAIN SUMMARY OK")

    with patch("agent.context_compressor.call_llm", side_effect=_call):
        summary = compressor._generate_summary(TURNS)

    assert summary is not None and "MAIN SUMMARY OK" in summary
    assert models == ["tool-less-summarizer", "main-model"]

def _tool_call_reply():
    """A 200 reply that continues the agent's work instead of summarizing (local models, #158684)."""
    call = {"id": "c9", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[call]), finish_reason="tool_calls",
    )])


def test_tool_call_reply_retries_flat_on_the_same_route():
    """An empty reply carrying tool calls is a request-shape failure: flatten and retry, no main fallback."""
    compressor = _compressor()
    seen = []

    def _call(**kwargs):
        roles = [m["role"] for m in kwargs["messages"]]
        seen.append(roles)
        return _tool_call_reply() if "tool" in roles else _ok()

    with patch("agent.context_compressor.call_llm", side_effect=_call):
        summary = compressor._generate_summary(TURNS)

    assert summary is not None and "FLAT SUMMARY OK" in summary
    assert len(seen) == 2, seen
    assert "tool" in seen[0] and "tool" not in seen[1]
    assert compressor.summary_model == "tool-less-summarizer"
    assert compressor._summary_model_fallen_back is False
