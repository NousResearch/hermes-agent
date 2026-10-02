"""Unexpected-kwarg TypeError dispatch diagnostic (follow-up to #60821).

When the provider SDK rejects a kwarg at dispatch, the API-error handler logs the
api_kwargs keys plus the llm_request middleware trace so the injector (a plugin
replacing the request dict, or stale kwargs crossing an api_mode switch) is
identifiable from one log line.
"""

import logging
from types import SimpleNamespace

import pytest

import run_agent
from agent.turn_recovery import _format_unexpected_kwarg_diagnostics


class TestFormatUnexpectedKwargDiagnostics:
    def test_renders_sorted_kwargs_keys(self):
        rendered = _format_unexpected_kwarg_diagnostics(
            {"system": "x", "messages": [], "model": "m"}, []
        )
        assert "api_kwargs keys: [messages, model, system]" in rendered

    def test_renders_each_middleware_trace_entry(self):
        trace = [{"name": "skill-enforce", "source": "plugin"}, {"source": "nemo_relay"}, "raw-entry"]
        rendered = _format_unexpected_kwarg_diagnostics({"model": "m"}, trace)
        assert "llm_request middleware trace: skill-enforce; nemo_relay; raw-entry" in rendered

    def test_missing_inputs_render_placeholders(self):
        rendered = _format_unexpected_kwarg_diagnostics(None, None)
        assert "<unavailable>" in rendered
        assert "none applied" in rendered


@pytest.fixture
def chat_agent(monkeypatch):
    import time as _time

    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    agent = run_agent.AIAgent(
        model="tencent/hy3",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        quiet_mode=True,
        max_iterations=2,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    agent._disable_streaming = True
    return agent


def test_middleware_injected_kwarg_is_named_in_the_failure_log(chat_agent, monkeypatch, caplog):
    """The #60821 shape end to end: a plugin middleware writes the Anthropic-shape
    ``system`` key onto chat_completions kwargs and the SDK rejects it at dispatch."""

    def _inject_system(request, **_context):
        return SimpleNamespace(
            payload={**request, "system": "reminder"}, original_payload=request,
            changed=True, trace=[{"name": "skill-enforce", "source": "plugin"}],
        )

    def _sdk_rejects(api_kwargs):
        raise TypeError("Completions.create() got an unexpected keyword argument 'system'")

    monkeypatch.setattr("hermes_cli.middleware.apply_llm_request_middleware", _inject_system)
    monkeypatch.setattr(chat_agent, "_interruptible_api_call", _sdk_rejects)

    with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
        chat_agent.run_conversation("hi")

    diagnostics = [r.getMessage() for r in caplog.records if "Unexpected-kwarg TypeError" in r.getMessage()]
    assert diagnostics, "no unexpected-kwarg diagnostic was logged"
    assert "api_mode=chat_completions" in diagnostics[0]
    assert "system" in diagnostics[0].split("api_kwargs keys:")[1]
    assert "llm_request middleware trace: skill-enforce" in diagnostics[0]
