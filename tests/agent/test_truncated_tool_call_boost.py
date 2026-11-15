"""Truncation retries must send a larger output budget than the failed request (#72770),
without exceeding the model's known output limit (#79715)."""
from typing import Any
from types import SimpleNamespace

import pytest

from agent.turn_iteration_prep import apply_retry_restarts
from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import _retry_truncated_tool_call


def _agent(max_tokens, requested_cap, **extra):
    return SimpleNamespace(
        max_tokens=max_tokens,
        _ephemeral_max_output_tokens=None,
        _buffer_vprint=lambda *a, **k: None,
        _requested_output_cap_from_api_kwargs=lambda kw: requested_cap,
        **extra,
    )


def _tool_call_budgets(agent, attempts=4):
    st = SimpleNamespace(agent=agent, truncated_tool_call_retries=0, is_stub=False)
    st.done = lambda action, result=None: action
    budgets = []
    for _ in range(attempts):
        assert _retry_truncated_tool_call(st, {}) == "continue"
        budgets.append(agent._ephemeral_max_output_tokens)
    return budgets


def _length_continuation_budgets(agent, attempts=4):
    budgets = []
    for n in range(1, attempts + 1):
        _retry = TurnRetryState()
        _retry.restart_with_length_continuation = True
        verdict = apply_retry_restarts(
            agent, _retry=_retry, response=None, interrupted=False, messages=[],
            conversation_history=[], user_message="hi", api_kwargs={}, current_turn_user_idx=0,
            final_response=None, retry_count=0, max_retries=3, api_call_count=1,
            restart_count=0, length_continue_retries=n,
            _preflight_compression_blocked=False, _turn_exit_reason="unknown",
        )
        assert verdict.action == "continue"
        budgets.append(agent._ephemeral_max_output_tokens)
    return budgets


SITES = [_tool_call_budgets, _length_continuation_budgets]


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("requested_cap", [32768, 65536])
def test_retry_raises_budget_above_large_requested_cap(site, requested_cap):
    budgets = site(_agent(None, requested_cap))
    assert budgets[0] > requested_cap
    assert max(budgets) <= requested_cap * 2


@pytest.mark.parametrize("site", SITES)
def test_boost_clamped_to_known_model_output_limit(site):
    # claude-sonnet-4-5 outputs at most 64000: never ask for more, and don't double a
    # request that already sits at the ceiling (the provider would 400).
    below = site(_agent(None, 40000, api_mode="anthropic_messages", model="claude-sonnet-4-5"))
    at_limit = site(_agent(None, 64000, api_mode="anthropic_messages", model="claude-sonnet-4-5"))
    assert below == [64000] * 4
    assert at_limit == [64000] * 4


def test_small_explicit_max_tokens_ladder_still_capped_at_floor():
    assert _tool_call_budgets(_agent(4096, None)) == [8192, 16384, 32768, 32768]


def test_retry_truncated_tool_call_sets_openrouter_cache_bypass_and_disables_reasoning(monkeypatch):
    from agent.chat_completion_helpers import _reasoning_config_for_wire
    import agent.conversation_loop as conv_loop
    from agent.turn_api_request import build_request_payload

    monkeypatch.setattr(
        conv_loop,
        "_redecorate_prompt_cache_for_provider",
        lambda ag, msgs, system_message=None, moa_prepared=None, tools_for_api=None: (
            msgs,
            moa_prepared,
            tools_for_api,
        ),
    )

    class _StubAgent:
        max_tokens = None
        model = "google/gemini-4-argon"
        provider = "openrouter"
        base_url = "https://openrouter.ai/api/v1"
        api_mode = "chat_completions"
        session_id = "test-session"
        platform = "cli"
        tools = []
        reasoning_config = {"enabled": True, "effort": "high"}
        _ephemeral_max_output_tokens = None
        _ephemeral_reasoning_off = False
        _empty_content_retries = 0
        _truncated_tool_call_retries = 0
        _pending_user_turn_headers = None
        _image_rejecting_models = set()
        _force_ascii_payload = False
        _last_observed_tool_names = None
        log_prefix = ""

        def _buffer_vprint(self, *a, **k):
            pass

        def _flush_status_buffer(self):
            pass

        def _vprint(self, *a, **k):
            pass

        def _cleanup_task_resources(self, tid):
            pass

        def _reset_stream_delivery_tracking(self):
            pass

        def _reapply_reasoning_echo_for_provider(self, msgs):
            pass

        def _requested_output_cap_from_api_kwargs(self, kw):
            return kw.get("max_tokens")

        def _is_openrouter_url(self):
            return True

        def _build_api_kwargs(self, api_messages, tools_for_api=None):
            rc = _reasoning_config_for_wire(self)
            out = {"model": self.model, "messages": api_messages, "extra_body": {"reasoning": rc}}
            if self._ephemeral_max_output_tokens is not None:
                out["max_tokens"] = self._ephemeral_max_output_tokens
                self._ephemeral_max_output_tokens = None
            return out

    agent = _StubAgent()

    def _build(call_no):
        return build_request_payload(
            agent,
            api_messages=[{"role": "user", "content": "hi"}],
            _moa_prepared_request=None,
            tools_for_api=[],
            system_message=None,
            api_call_count=call_no,
            api_request_id=f"req-{call_no}",
            effective_task_id="task-1",
            turn_id="turn-1",
        )

    # Negative control: initial request before truncation retry has no cache-bypass header
    # and keeps normal reasoning enabled.
    req0 = _build(1)
    assert req0.action == "fallthrough"
    assert "X-OpenRouter-Cache" not in (req0.api_kwargs.get("extra_headers") or {})
    assert req0.api_kwargs["extra_body"]["reasoning"] == {"enabled": True, "effort": "high"}

    # Trigger truncated tool call retry.
    st: Any = SimpleNamespace(agent=agent, truncated_tool_call_retries=0, is_stub=False)
    st.done = lambda action, result=None: action
    assert _retry_truncated_tool_call(st, {"max_tokens": 2048}) == "continue"
    assert agent._truncated_tool_call_retries == 1
    assert agent._ephemeral_reasoning_off is True
    assert agent._ephemeral_max_output_tokens == 4096

    # Next build_request_payload consumes _ephemeral_reasoning_off and injects X-OpenRouter-Cache: false.
    req1 = _build(2)
    assert req1.action == "fallthrough"
    assert req1.api_kwargs["extra_headers"]["X-OpenRouter-Cache"] == "false"
    assert req1.api_kwargs["extra_body"]["reasoning"] == {"enabled": False, "effort": "none"}
    assert req1.api_kwargs["max_tokens"] == 4096
    assert agent._ephemeral_reasoning_off is False

    # Terminal exhaustion (5th truncation) resets both retry flags.
    st.truncated_tool_call_retries = 4
    st.effective_task_id = "task-1"
    st.messages = []
    st.end_turn = lambda resp, cleanup=False, failure=None: "end_turn"
    assert _retry_truncated_tool_call(st, {"max_tokens": 4096}) == "end_turn"
    assert agent._truncated_tool_call_retries == 0
    assert agent._ephemeral_reasoning_off is False
