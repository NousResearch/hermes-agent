"""Fatal reasons and replay-safe failover — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

import logging

import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
)
from tests.agent.claude_sdk_fakes import (
    ResultMessage,
    _make_session,
    _make_turn,
    _make_agent,
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


# ---------- fatal-reason plumbing: refusals must be machine-readable ----------
# Clean-checkout E2E finding on #65982 (jefftropeano): a fatal metered-billing
# refusal exited 0 because the runtime never sets "failed"/"failure_reason" —
# the fields the chat_completions path sets (conversation_loop) and the -Q
# exit path keys on. TurnResult.fatal_reason carries the classification out
# of run_turn (the refusal exception never propagates past it).


class TestFatalReason:
    def test_metered_refusal_sets_fatal_reason_startup(self, monkeypatch):
        # "startup", deliberately NOT "billing": the kanban -Q exit contract
        # maps failure_reason "billing" to the transient EX_TEMPFAIL requeue
        # sentinel, and a present metered key is a config error retries can't
        # fix — it must count as a real failure everywhere.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        session = ClaudeAgentSdkSession(cwd="/tmp")  # no factory → real path
        turn = session.run_turn("hi")
        assert turn.should_retire
        assert turn.fatal_reason == "startup"

    def test_auth_classified_startup_failure_sets_fatal_reason_auth(self):
        session, _ = _make_session(
            connect_exc=RuntimeError("401 unauthorized: invalid bearer token")
        )
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.should_retire
        assert turn.fatal_reason == "auth"

    def test_startup_traceback_keeps_frames_but_redacts_exception(self, caplog):
        secret = "sk-ant-api03-SUPERSECRET"
        session, _ = _make_session(
            connect_exc=RuntimeError(f"connect failed with {secret}")
        )
        with caplog.at_level(
            logging.WARNING,
            logger="agent.transports.claude_agent_sdk_session",
        ):
            try:
                turn = session.run_turn("hi")
            finally:
                session.close()

        assert turn.should_retire
        assert secret not in caplog.text
        assert "claude-agent-sdk startup failed" in caplog.text
        assert "Traceback (most recent call last)" in caplog.text
        assert "connect" in caplog.text

    def test_sdk_error_result_is_not_fatal(self):
        # An in-turn SDK error (e.g. error_max_turns) is turn-scoped, not a
        # startup/auth/billing refusal — it must stay non-fatal so one bad
        # turn can't flip an integration's exit code.
        script = [ResultMessage(subtype="error_max_turns", is_error=False)]
        session, _ = _make_session(script=script)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert turn.fatal_reason is None

    def test_runtime_glue_maps_fatal_reason_to_failed(self):
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            should_retire=True,
            error="claude-agent-sdk startup failed: refused",
            fatal_reason="startup",
            projected_messages=[],
            final_text="",
            token_usage_last=None,
        )
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        assert result["failed"] is True
        assert result["failure_reason"] == "startup"
        assert result["partial"] is True

    def test_runtime_glue_exposes_clean_transient_error_for_fallback(self):
        # A transient SDK timeout with no output or tool effects is a failed
        # provider attempt that the shared dispatcher may continue elsewhere.
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            should_retire=True,
            error="turn timed out after 600s",
            projected_messages=[],
            tool_iterations=0,
            final_text="",
            token_usage_last=None,
        )
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        assert result["failed"] is True
        assert result["failover_reason"] == "timeout"


class TestReplaySafeProviderFailureOutcome:
    def _run_failure(self, agent, *, messages=None, **turn_overrides):
        turn_kwargs = {
            "projected_messages": [],
            "final_text": "",
            "token_usage_last": None,
            "api_call_made": True,
            "tool_iterations": 0,
            **turn_overrides,
        }
        agent._claude_sdk_session.run_turn.return_value = _make_turn(**turn_kwargs)
        return run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages if messages is not None else [{"role": "user", "content": "hi"}],
            effective_task_id="task-failure",
        )

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            ("API Error: 401 OAuth access token expired", "auth"),
            ("HTTP 429 rate limit exceeded", "rate_limit"),
            ("HTTP 503 service overloaded", "overloaded"),
            ("HTTP 500 internal server error", "server_error"),
            ("request timed out while contacting provider", "timeout"),
        ],
    )
    def test_canonical_provider_failures_expose_validated_handoff(self, error, expected):
        agent = _make_agent()

        result = self._run_failure(agent, error=error, should_retire=False)

        assert result["failed"] is True
        assert result["failover_reason"] == expected
        assert agent._claude_sdk_session is None

    @pytest.mark.parametrize(
        ("error", "fatal_reason"),
        [
            ("unexpected local startup failure", "startup"),
            ("ANTHROPIC_API_KEY is present; refusing unsafe metered billing", "startup"),
            ("unrecognized SDK result failure", None),
            ("HTTP 400 invalid local configuration", None),
        ],
    )
    def test_local_unknown_and_billing_safety_failures_remain_terminal(
        self, error, fatal_reason
    ):
        agent = _make_agent()

        result = self._run_failure(
            agent,
            error=error,
            fatal_reason=fatal_reason,
            should_retire=True,
            api_call_made=False,
        )

        assert "failover_reason" not in result

    @pytest.mark.parametrize(
        "effect",
        ["tool", "projected", "streamed", "interrupted"],
    )
    def test_provider_failure_after_an_observable_effect_is_not_replayable(self, effect):
        agent = _make_agent()
        messages = [{"role": "user", "content": "hi"}]
        overrides = {
            "error": "HTTP 429 rate limit exceeded",
            "should_retire": True,
            "tool_iterations": 1 if effect == "tool" else 0,
            "projected_messages": (
                [{"role": "assistant", "content": "partial"}]
                if effect == "projected"
                else []
            ),
            "interrupted": effect == "interrupted",
        }
        if effect == "streamed":
            def streamed_failure(user_input):
                agent._current_streamed_assistant_text = "partial"
                return _make_turn(
                    final_text="",
                    token_usage_last=None,
                    api_call_made=True,
                    **overrides,
                )
            agent._claude_sdk_session.run_turn.side_effect = streamed_failure
            result = run_claude_agent_sdk_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=messages,
                effective_task_id="task-streamed",
            )
        else:
            result = self._run_failure(agent, messages=messages, **overrides)

        assert "failover_reason" not in result
        assert result["sdk_effects"][effect] is True
        if effect == "projected":
            assert messages[-1]["content"] == "partial"

    def test_stream_relay_records_delivery_before_display_callback(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._stream_callback = None

        def record(text):
            agent._current_streamed_assistant_text += text

        agent._record_streamed_assistant_text.side_effect = record

        def display(text):
            assert agent._current_streamed_assistant_text == text
            raise RuntimeError("display sink failed after delivery")

        agent.stream_delta_callback = display

        class SpySession:
            def __init__(self, **kwargs):
                self.on_stream_delta = kwargs["on_stream_delta"]

            def run_turn(self, user_input):
                self.on_stream_delta("visible partial")
                return _make_turn(
                    error="HTTP 429 rate limit exceeded",
                    should_retire=True,
                    projected_messages=[],
                    tool_iterations=0,
                    final_text="",
                    token_usage_last=None,
                )

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)

        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-stream-relay",
        )

        agent._record_streamed_assistant_text.assert_called_once_with("visible partial")
        assert result["sdk_effects"]["streamed"] is True
        assert "failover_reason" not in result

    def test_issued_tool_ledger_is_turn_local_and_precedes_progress_callback(
        self, monkeypatch
    ):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._sdk_issued_tool_effect = True
        agent._stream_callback = None
        agent.stream_delta_callback = None
        creation_count = 0

        class SpySession:
            def __init__(self, **kwargs):
                nonlocal creation_count
                creation_count += 1
                self.on_tool_started = kwargs["on_tool_started"]
                self.issue_tool = creation_count == 2

            def run_turn(self, user_input):
                if self.issue_tool:
                    self.on_tool_started("write_file", "writing", {"path": "/tmp/x"})
                return _make_turn(
                    error="HTTP 503 service overloaded",
                    should_retire=True,
                    projected_messages=[],
                    tool_iterations=0,
                    final_text="",
                    token_usage_last=None,
                )

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)

        stale_result = run_claude_agent_sdk_turn(
            agent,
            user_message="first",
            original_user_message="first",
            messages=[{"role": "user", "content": "first"}],
            effective_task_id="task-no-tool",
        )
        assert agent._sdk_issued_tool_effect is False
        assert stale_result["sdk_effects"]["tool"] is False
        assert stale_result["failover_reason"] == "overloaded"

        def progress(*_args):
            assert agent._sdk_issued_tool_effect is True
            raise RuntimeError("progress sink failed after tool issue")

        agent.tool_progress_callback = progress
        issued_result = run_claude_agent_sdk_turn(
            agent,
            user_message="second",
            original_user_message="second",
            messages=[{"role": "user", "content": "second"}],
            effective_task_id="task-issued-tool",
        )

        assert issued_result["sdk_effects"]["tool"] is True
        assert "failover_reason" not in issued_result

    def test_stream_replay_ledger_is_reset_before_each_sdk_turn(self):
        agent = _make_agent()
        agent._current_streamed_assistant_text = "completed prior turn"

        result = self._run_failure(
            agent,
            error="HTTP 429 rate limit exceeded",
            should_retire=True,
        )

        assert result["sdk_effects"]["streamed"] is False
        assert result["failover_reason"] == "rate_limit"

    def test_authoritative_startup_auth_failure_is_zero_call_and_handoff_eligible(self):
        agent = _make_agent()

        result = self._run_failure(
            agent,
            error="Not signed in to Claude; run `claude auth login`.",
            fatal_reason="auth",
            should_retire=True,
            api_call_made=False,
        )

        assert result["api_calls"] == 0
        assert result["failover_reason"] == "auth"
        assert "claude auth login" in result["error"]
