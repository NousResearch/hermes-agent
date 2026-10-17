"""Machinery turns get a zero empty-response retry budget.

An injected internal-notification turn (``display_kind=internal_notification``,
#82888 machinery lane) is not owed a reply. When the model answers it with an
empty completion, the ordinary recovery ladder's paid retries re-bill the full
input for a notification nobody is waiting on: #135816 logged six notification
turns burning twelve requests (~$8.95 estimated) where every paid retry came
back empty too. The free repair paths (post-tool nudge, thinking prefill,
fallback activation) still run — only the paid retry count collapses, and only
for machinery turns; human turns keep the full budget.
"""

from types import SimpleNamespace

import pytest

from agent import empty_response_guard as guard
from agent.turn_context import _reset_per_turn_agent_state, _stage_turn_user_message
from agent.turn_empty_response import (
    MACHINERY_TURN_DISPLAY_KIND,
    MACHINERY_TURN_DISPLAY_KINDS,
    _reset_turn_scoped_state,
    _retry_empty,
)
from gateway.response_filters import MACHINERY_DISPLAY_KINDS as gateway_machinery_kinds


def _agent(**overrides):
    base = dict(
        model="openai/gpt-6-astra",
        provider="nous",
        api_mode="chat_completions",
        base_url=None,
        api_key=None,
        _empty_content_retries=0,
        _thinking_prefill_retries=0,
        max_iterations=40,
        quiet_mode=True,
        session_id="test-session",
        platform="test",
        _turn_origin=None,
        _compression_warning=False,
        _pending_startup_notices=None,
        _run_budget_started_at=None,
        run_budget_seconds=None,
        _stream_context_scrubber=None,
        _stream_think_scrubber=None,
        _todo_store=SimpleNamespace(has_items=lambda: False),
        _memory_store=SimpleNamespace(reset_consolidation_failures=None),
        _tool_guardrails=SimpleNamespace(reset_for_turn=lambda: None),
        _hydrate_todo_store=lambda history: None,
        _interrupt_requested=False,
        _touch_activity=lambda label: None,
        _adopt_nous_key_before_expiry=lambda: None,
        diagnostics=[],
    )
    base.update(overrides)
    agent = SimpleNamespace(**base)
    agent._buffer_diagnostic_status = lambda text: agent.diagnostics.append(text)
    return agent


def _empty_response(prompt_tokens=25_900):
    """A zero-output empty completion whose usage normalizes and prices fine —
    the cost-aware budget path must not be what is under test here."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        total_tokens=prompt_tokens,
    )
    return SimpleNamespace(usage=usage, choices=[], model="openai/gpt-6-astra")


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    """The paid retry's backoff must never actually wait in tests."""
    monkeypatch.setattr("time.sleep", lambda *_: None)


class TestMachineryEmptyRetryBudget:
    def test_internal_notification_turn_skips_paid_retries(self):
        agent = _agent()
        action, interrupt, _ = _retry_empty(
            agent, _empty_response(), "stop", True,
            display_kind=MACHINERY_TURN_DISPLAY_KIND,
            messages=[], conversation_history=None, api_call_count=1,
        )
        assert action is None  # no retry is scheduled
        assert interrupt is None
        assert agent._empty_content_retries == 0
        assert len(agent.diagnostics) == 0  # no "retrying (1/3)" chatter either

    def test_human_turn_keeps_full_budget_and_retries(self):
        agent = _agent()
        action, _, _ = _retry_empty(
            agent, _empty_response(), "stop", True,
            display_kind=None,
            messages=[], conversation_history=None, api_call_count=1,
        )
        assert action == "continue"
        assert agent._empty_content_retries == 1

    def test_only_machinery_kinds_collapse(self):
        agent = _agent()
        action, _, _ = _retry_empty(
            agent, _empty_response(), "stop", True,
            display_kind="background_result",
            messages=[], conversation_history=None, api_call_count=1,
        )
        assert action == "continue"
        assert agent._empty_content_retries == 1

    def test_non_empty_candidate_unaffected(self):
        """A reasoning-only response routes through prefill, not this cut."""
        agent = _agent()
        action, _, _ = _retry_empty(
            agent, _empty_response(), "stop", False,
            display_kind=MACHINERY_TURN_DISPLAY_KIND,
            messages=[], conversation_history=None, api_call_count=1,
        )
        assert action is None
        assert agent._empty_content_retries == 0


class TestDeterministicDetectionStillRunsOnMachinery:
    def test_machinery_cut_returns_deterministic_flag(self):
        """Zero budget keeps the empty-attempt history growing, so the
        deterministic-empty short-circuit still reports true after two
        signature-matched zero-output attempts."""
        agent = _agent()
        guard.record_empty_attempt(agent, finish_reason="stop", response=_empty_response())
        agent._empty_content_retries += 1
        guard.record_empty_attempt(agent, finish_reason="stop", response=_empty_response())
        agent._empty_content_retries += 1
        action, _, deterministic = _retry_empty(
            agent, _empty_response(), "stop", True,
            display_kind=MACHINERY_TURN_DISPLAY_KIND,
            messages=[], conversation_history=None, api_call_count=3,
        )
        assert action is None
        assert deterministic is True


class TestTurnScopedReset:
    def test_reset_reads_display_kind_off_the_staged_user_row(self):
        agent = _agent()
        _reset_per_turn_agent_state(agent)
        user_msg, _ = _stage_turn_user_message(
            agent, "kanban completed: task-1", None, None, None,
            MACHINERY_TURN_DISPLAY_KIND, None,
        )
        _reset_turn_scoped_state(agent, user_msg)
        assert agent._machinery_turn_display_kind == "internal_notification"
        # Cross-layer drift guard: the agent-side mirror (agent/turn_empty_response.py
        # MACHINERY_TURN_DISPLAY_KINDS) must stay equal to the set the gateway actually
        # derives its kinds from (gateway/response_filters.py MACHINERY_DISPLAY_KINDS).
        # If the gateway ever grows a second machinery kind, this mirror must follow —
        # otherwise the paid empty-retry cut silently keeps billing the new machinery
        # turns against the human retry budget.
        assert gateway_machinery_kinds == MACHINERY_TURN_DISPLAY_KINDS
        assert "internal_notification" in gateway_machinery_kinds

    def test_reset_clears_when_the_row_is_untyped(self):
        agent = _agent()
        _reset_per_turn_agent_state(agent)
        user_msg, _ = _stage_turn_user_message(agent, "hello", None, None, None, None, None)
        agent._machinery_turn_display_kind = MACHINERY_TURN_DISPLAY_KIND  # stale prior turn
        _reset_turn_scoped_state(agent, user_msg)
        assert agent._machinery_turn_display_kind is None

    def test_reset_tolerates_non_dict_row(self):
        agent = _agent()
        _reset_turn_scoped_state(agent, "not-a-dict")
        assert agent._machinery_turn_display_kind is None
