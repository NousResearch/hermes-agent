"""Anthropic long-context-tier (429) recovery: the retry diagnostics buffered
during recovery must flush correctly (muted vs. unmuted) on terminal failure."""

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
