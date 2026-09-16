"""Standard (chat_completions/anthropic_messages) response-path coverage for
Patch B's usage-less accounting rule: a real API response with no usable
usage data still counts as exactly one API call, and it makes the session's
cumulative cost status sticky-"unknown" — even across later priced calls.

Integration-level (drives agent.run_conversation() through a mocked
Anthropic response, mirroring tests/run_agent/test_context_token_tracking.py)
because the behavior lives inline in conversation_loop.py's per-call
accounting block, not behind a separately importable helper. The Codex
app-server equivalent is covered directly in
tests/run_agent/test_codex_app_server_integration.py, where
_record_codex_app_server_usage is an importable function.
"""

import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


class _FakeAnthropicClient:
    def close(self):
        pass


def _patch_bootstrap(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [{
        "type": "function",
        "function": {"name": "t", "description": "t", "parameters": {"type": "object", "properties": {}}},
    }])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _make_agent(monkeypatch, *, model="claude-sonnet-4-5"):
    """An AIAgent whose next API response is swapped per-turn via
    ``holder["response_fn"]`` — lets one agent/session drive several turns
    with different usage shapes, which is what the sticky-status rule needs
    to be observed across (a single turn can't exercise "sticky")."""
    _patch_bootstrap(monkeypatch)
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        lambda k, b=None, **kwargs: _FakeAnthropicClient(),
    )

    holder: dict = {}

    class _A(run_agent.AIAgent):
        def __init__(self, *a, **kw):
            kw.update(skip_context_files=True, skip_memory=True, max_iterations=4)
            super().__init__(*a, **kw)
            self._cleanup_task_resources = self._persist_session = lambda *a, **k: None
            self._save_trajectory = lambda *a, **k: None

        def run_conversation(self, msg, conversation_history=None, task_id=None):
            self._interruptible_api_call = lambda kw: holder["response_fn"]()
            self._disable_streaming = True
            return super().run_conversation(msg, conversation_history=conversation_history, task_id=task_id)

    agent = _A(
        model=model,
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        provider="anthropic",
        api_mode="anthropic_messages",
    )
    return agent, holder


def _anthropic_resp(input_tok=100, output_tok=20, *, usage=True):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=input_tok, output_tokens=output_tok) if usage else None,
        model="claude-sonnet-4-5",
    )


def test_usage_less_response_counts_the_call_and_marks_status_unknown(monkeypatch):
    agent, holder = _make_agent(monkeypatch)
    holder["response_fn"] = lambda: _anthropic_resp(usage=False)

    agent.run_conversation("hi")

    assert agent.session_api_calls == 1
    assert agent.session_cost_status == "unknown"


def test_priced_call_reports_estimated_status(monkeypatch):
    """Baseline: claude-sonnet-4-5 has a real pricing entry, so a normal
    usage-bearing call reports "estimated", not "unknown" — the control for
    the sticky test below."""
    agent, holder = _make_agent(monkeypatch)
    holder["response_fn"] = lambda: _anthropic_resp(100, 20, usage=True)

    agent.run_conversation("hi")

    assert agent.session_api_calls == 1
    assert agent.session_cost_status == "estimated"
    assert agent.session_estimated_cost_usd > 0.0


def test_priced_parent_call_keeps_unknown_status_from_prior_subagent_cost(monkeypatch):
    agent, holder = _make_agent(monkeypatch)
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "subagent"
    holder["response_fn"] = lambda: _anthropic_resp(100, 20, usage=True)

    agent.run_conversation("hi")

    assert agent.session_api_calls == 1
    assert agent.session_cost_status == "unknown"


def test_usage_less_turn_sticks_session_status_unknown_even_after_a_later_priced_turn(monkeypatch):
    agent, holder = _make_agent(monkeypatch)

    holder["response_fn"] = lambda: _anthropic_resp(100, 20, usage=True)
    agent.run_conversation("turn one")
    assert agent.session_cost_status == "estimated"

    holder["response_fn"] = lambda: _anthropic_resp(usage=False)
    agent.run_conversation("turn two")
    assert agent.session_api_calls == 2
    assert agent.session_cost_status == "unknown"

    # A later, perfectly priced turn cannot undo the earlier gap in the
    # session's cumulative cost data.
    holder["response_fn"] = lambda: _anthropic_resp(100, 20, usage=True)
    agent.run_conversation("turn three")
    assert agent.session_api_calls == 3
    assert agent.session_cost_status == "unknown"
