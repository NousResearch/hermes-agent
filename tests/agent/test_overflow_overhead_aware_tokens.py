"""Overflow recovery handlers (413, context overflow, Anthropic long-context
tier 429) must hand the compressor an overhead-aware token estimate that counts
tool schemas, not a messages-only estimate (LCM issue 441). Also: the retry
diagnostics buffered during long-context recovery must flush correctly (muted
vs. unmuted) on terminal failure."""

import pytest

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Shared fixtures / helpers (mirrored from test_413_compression.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Short-circuit all time.sleep and jittered_backoff calls."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    from agent import retry_utils as _retry_utils
    monkeypatch.setattr(_retry_utils, "jittered_backoff", lambda *a, **k: 0.0)


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None, usage=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = SimpleNamespace(**usage) if usage else None
    return resp


@pytest.fixture()
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = True
        a.save_trajectories = False
        return a


def _prefill():
    return [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]


_SENTINEL_TOKENS = 987_654


def _make_413_error():
    err = Exception("Request entity too large")
    err.status_code = 413
    return err


def _make_context_overflow_error():
    """A 400 the classifier routes to context_overflow."""
    err = Exception(
        "Error code: 400 - {'error': {'message': "
        "\"This endpoint's maximum context length is 128000 tokens. "
        "However, you requested about 200000 tokens. "
        "Please reduce the length of the messages.\"}}"
    )
    err.status_code = 400
    return err


def _make_prompt_too_long_error():
    """Anthropic's 'prompt is too long' variant of context overflow."""
    err = Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'prompt is too long: 233153 tokens > 200000 maximum'}}"
    )
    err.status_code = 400
    return err


def _make_long_context_tier_error():
    """Build a 429 'extra usage required for long context requests' error."""
    err = Exception(
        "Error code: 429 - {'error': {'type': 'rate_limit_error', "
        "'message': 'Extra usage is required for long context requests. "
        "Please enable extra usage in your account settings.'}}"
    )
    err.status_code = 429
    return err


@pytest.mark.parametrize("muted", [False, True])
def test_long_context_retry_carrier_survives_failure_flush(agent, muted, monkeypatch):
    from gateway.warning_notifications import DiagnosticText
    agent._notification_config = {"display": {"suppress_warning_notifications": muted}}
    err = _make_long_context_tier_error()
    agent.client.chat.completions.create.side_effect = [err, _mock_response(content="Recovered")]
    captured = []
    original = agent._buffer_retry_message
    def capture(kind, text):
        captured.append((kind, text))
        original(kind, text)
    monkeypatch.setattr(agent, "_buffer_retry_message", capture)
    with (patch("agent.model_metadata.estimate_request_tokens_rough", return_value=_SENTINEL_TOKENS),
          patch.object(agent, "_compress_context", return_value=([{"role": "user", "content": "compressed"}], "compressed prompt")),
          patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
          patch.object(agent, "_cleanup_task_resources")):
        result = agent.run_conversation("hello", conversation_history=_prefill())
    assert result["final_response"] == "Recovered"
    assert captured
    assert any(kind == "status" for kind, _ in captured)
    assert all(kind == "vprint" or isinstance(text, DiagnosticText) for kind, text in captured), captured
    # Recovery clears buffered diagnostics; the same carriers must also work on
    # the terminal-failure flush instead of leaking their retry sibling copy.
    assert not agent._retry_status_buffer
    agent._retry_status_buffer = list(captured)
    printed, observed = [], []
    agent._print_fn = lambda *a, **k: printed.append(a)
    agent.status_callback = lambda *a: observed.append(a)
    agent._flush_status_buffer()
    assert bool(printed) is not muted
    assert observed


# A tool schema big enough (~10k tokens) to dwarf the tiny test conversation,
# so a messages-only estimate cannot reach the floor asserted below.
_HEAVY_TOOL_DESCRIPTION = "Heavy schema padding. " * 2_000
_TOOL_OVERHEAD_FLOOR = 5_000


@pytest.mark.parametrize(
    "make_error",
    [
        pytest.param(_make_413_error, id="413"),
        pytest.param(_make_context_overflow_error, id="context_overflow"),
        pytest.param(_make_prompt_too_long_error, id="prompt_too_long"),
        pytest.param(_make_long_context_tier_error, id="long_context_tier_429"),
    ],
)
def test_overflow_recovery_compresses_with_tool_overhead_counted(agent, make_error):
    """Each recovery path must pass approx_tokens that include tool-schema
    overhead: a messages-only estimate under-reports the request, so the
    compressor sizes its target wrong and the retry overflows again."""
    from agent.model_metadata import estimate_messages_tokens_rough

    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "heavy_tool",
                "description": _HEAVY_TOOL_DESCRIPTION,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    history = _prefill()
    assert estimate_messages_tokens_rough(history + [{"role": "user", "content": "hello"}]) < 500

    agent.client.chat.completions.create.side_effect = [
        make_error(),
        _mock_response(content="Recovered", finish_reason="stop"),
    ]
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        mock_compress.return_value = (
            [{"role": "user", "content": "compressed"}],
            "compressed prompt",
        )
        result = agent.run_conversation("hello", conversation_history=history)

    assert result["final_response"] == "Recovered"
    passed = [c.kwargs.get("approx_tokens") for c in mock_compress.call_args_list]
    assert passed, "recovery never invoked the compressor"
    assert any(t is not None and t >= _TOOL_OVERHEAD_FLOOR for t in passed), (
        f"compressor got messages-only estimates {passed}; tool schemas were not counted"
    )
