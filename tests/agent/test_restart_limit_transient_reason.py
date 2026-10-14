"""Regression tests for #133361 — a fallback chain exhausted by transient walls exits 1.

When every rung of the fallback chain is walled by a transient provider error (HTTP 429
usage limit, 503 overloaded…), the turn ends on ``rebuilt_restart_limit_exceeded`` with
``failure_reason="loop_error"`` — the per-attempt classifier verdicts are discarded — so a
Kanban worker exits 1 and the dispatcher books a counted crash, while a single-provider 429
wall exits 75 and is booked ``rate_limited`` without spending ``failure_limit``. The fix
records the classified reason of each failover activation on the agent (turn-scoped) and
re-stamps the restart-limit failure_reason from it when the whole walk was transient (or
terminal). Mixed / unattributed histories keep today's ``loop_error``.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason
from agent.turn_failure_copy import (
    TERMINAL_PROVIDER_FAILURE_REASONS,
    TRANSIENT_PROVIDER_FAILURE_REASONS,
    restart_limit_failure_reason,
)
from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent

# Dummy credential for the stub agent constructor; never sent anywhere (client is a MagicMock).
_TEST_API_KEY = os.environ.get("HERMES_TEST_API_KEY", "stub-key-not-a-secret")


class TestRestartLimitFailureReason:
    @pytest.mark.parametrize(
        "exit_reason",
        [
            "rebuilt_restart_limit_exceeded",
            "redirect_restart_limit_exceeded",
        ],
    )
    def test_all_transient_chain_picks_rate_limit_over_every_other_rung(
        self, exit_reason
    ):
        """A chain of 503s and one 429 books as rate_limited: the 429 names the cooldown the
        dispatcher should honour before re-spawning, so it wins the priority order."""
        assert (
            restart_limit_failure_reason(
                exit_reason, ["overloaded", "rate_limit", "server_error"]
            )
            == "rate_limit"
        )

    def test_transient_chain_without_rate_limit_picks_next_priority(self):
        assert (
            restart_limit_failure_reason(
                "rebuilt_restart_limit_exceeded", ["timeout", "overloaded"]
            )
            == "overloaded"
        )

    def test_all_terminal_chain_picks_auth(self):
        assert (
            restart_limit_failure_reason(
                "rebuilt_restart_limit_exceeded", ["model_not_found", "auth"]
            )
            == "auth"
        )

    def test_mixed_chain_keeps_loop_error(self):
        assert (
            restart_limit_failure_reason(
                "rebuilt_restart_limit_exceeded", ["rate_limit", "auth"]
            )
            is None
        )

    def test_unattributed_activation_keeps_loop_error(self):
        """A reasonless activation must not read as a provider verdict — fail closed to the
        historical loop_error."""
        assert (
            restart_limit_failure_reason(
                "rebuilt_restart_limit_exceeded", ["rate_limit", None]
            )
            is None
        )

    def test_empty_log_keeps_loop_error(self):
        assert (
            restart_limit_failure_reason("rebuilt_restart_limit_exceeded", []) is None
        )
        assert (
            restart_limit_failure_reason("rebuilt_restart_limit_exceeded", None) is None
        )

    def test_other_exits_are_never_rewritten(self):
        assert (
            restart_limit_failure_reason(
                "all_retries_exhausted_no_response", ["rate_limit"]
            )
            is None
        )

    def test_cli_exit_code_sets_share_the_single_source(self):
        """``_single_query_exit_code`` and the finalizer re-stamper must partition reasons the
        same way, or a stamped reason could still fall through to exit 1."""
        from hermes_cli import cli_single_query

        assert (
            cli_single_query._TRANSIENT_PROVIDER_REASONS
            == TRANSIENT_PROVIDER_FAILURE_REASONS
        )
        assert (
            cli_single_query._TERMINAL_PROVIDER_REASONS
            == TERMINAL_PROVIDER_FAILURE_REASONS
        )


class _StubBudget:
    used = 1
    max_total = 90
    remaining = 89


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from (no tool tail, no explainer)."""

    def __init__(self, activation_reasons):
        self.max_iterations = 90
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._current_streamed_assistant_text = None
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self._fallback_activation_reasons = activation_reasons

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _finalize(activation_reasons, exit_reason="rebuilt_restart_limit_exceeded"):
    agent = _StubAgent(activation_reasons)
    return agent, finalize_turn(
        agent,
        final_response=None,
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "run the task"}],
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="run the task",
        original_user_message="run the task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
    )


class TestFinalizerRestampsRestartLimit:
    def test_transient_chain_stamps_rate_limit_without_failing_the_turn(self):
        """The advisory contract holds: ``failed`` stays False (cron silence, the kanban
        breaker and gateway persistence key on it) while the descriptor gains the provider
        verdict a Kanban worker needs to exit 75."""
        agent, result = _finalize(["rate_limit", "overloaded", "rate_limit"])
        assert result["failure_reason"] == "rate_limit"
        assert result["failure_retryable"] is True
        assert result["failed"] is False

    def test_mixed_chain_keeps_loop_error(self):
        _, result = _finalize(["rate_limit", "auth"])
        assert result["failure_reason"] == "loop_error"

    def test_terminal_chain_stamps_auth(self):
        _, result = _finalize(["auth", "auth_permanent"])
        assert result["failure_reason"] == "auth"

    def test_kanban_worker_exit_code_follows_the_restamped_reason(self, monkeypatch):
        """End to end (#133361): a turn that walked a whole chain into transient walls makes
        the worker exit KANBAN_RATE_LIMIT_EXIT_CODE, exactly like a single-provider 429 wall;
        the mixed history keeps the historical counted-crash exit 1."""
        from hermes_cli.cli_single_query import _single_query_exit_code
        from hermes_cli.kanban_db import (
            KANBAN_RATE_LIMIT_EXIT_CODE,
            KANBAN_TERMINAL_PROVIDER_EXIT_CODE,
        )

        monkeypatch.setenv("HERMES_KANBAN_TASK", "1")
        _, transient = _finalize(["rate_limit", "overloaded"])
        assert _single_query_exit_code(transient) == KANBAN_RATE_LIMIT_EXIT_CODE
        _, terminal = _finalize(["auth", "model_not_found"])
        assert _single_query_exit_code(terminal) == KANBAN_TERMINAL_PROVIDER_EXIT_CODE
        _, mixed = _finalize(["rate_limit", "auth"])
        assert _single_query_exit_code(mixed) == 1


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key=_TEST_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1"):
    mock = MagicMock()
    mock.base_url = base_url
    return mock


class TestActivationReasonLog:
    def test_successful_activation_records_the_classified_reason(self):
        agent = _make_agent(
            fallback_model=[
                {"provider": "openai", "model": "gpt-4o"},
                {"provider": "zai", "model": "glm-4.7"},
            ]
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "resolved"),
        ):
            assert (
                agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
            )
            assert (
                agent._try_activate_fallback(reason=FailoverReason.overloaded) is True
            )
        assert agent._fallback_activation_reasons == ["rate_limit", "overloaded"]

    def test_unattributed_activation_is_recorded_as_none(self):
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "resolved"),
        ):
            assert agent._try_activate_fallback() is True
        assert agent._fallback_activation_reasons == [None]

    def test_restore_primary_runtime_clears_the_log_between_turns(self):
        """The log is turn-scoped: last turn's verdicts must not pollute the next turn's
        restart-limit stamp (an ``auth`` from a healed turn would flip a fresh transient
        walk to loop_error, or worse)."""
        from agent.agent_runtime_helpers import restore_primary_runtime

        agent = _make_agent()
        agent._fallback_activation_reasons = ["auth"]
        agent._fallback_activated = False
        restore_primary_runtime(agent)
        assert agent._fallback_activation_reasons == []
