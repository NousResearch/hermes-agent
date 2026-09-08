"""Runtime glue and orchestration — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

from unittest.mock import MagicMock

import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from tests.agent.claude_sdk_fakes import (
    TextBlock,
    AssistantMessage,
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


# ---------- runtime glue ----------

class TestRuntimeGlue:
    def test_turn_contract(self):
        agent = _make_agent()
        messages = [{"role": "user", "content": "hi"}]
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages,
            effective_task_id="task-1",
        )
        assert result["final_response"] == "SDK_ASSISTANT"
        assert result["completed"] is True
        assert result["agent_persisted"] is True
        assert result["cost_status"] == "included"
        assert result["cost_source"] == "claude-subscription"
        # Projected messages spliced after the (pre-appended) user turn.
        assert messages[-1]["content"] == "SDK_ASSISTANT"
        # Skill-nudge counter parity with the codex path.
        assert agent._iters_since_skill == 2

    def test_compact_boundary_completes_once_before_turn_end(self, monkeypatch):
        """The stream boundary is primary; terminal completion is fallback only."""
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        callbacks = {}

        class SpySession:
            def __init__(self, **kwargs):
                callbacks.update(kwargs)

            def context_usage(self):
                return None

            def set_turn_visibility_callbacks(self, **kwargs):
                pass

            def run_turn(self, user_input):
                callbacks["on_compaction"]("auto")
                callbacks["on_compact_boundary"]("auto")
                return _make_turn()

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        emitted = []
        agent._emit_status = emitted.append
        agent.status_callback = lambda kind, text: emitted.append((kind, text))

        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )

        assert agent._sdk_compaction_pending is False
        completions = [item for item in emitted if isinstance(item, tuple)]
        assert len(completions) == 1
        assert completions[0][0] == "compaction"
        assert "compaction complete" in completions[0][1].lower()

    def test_empty_rich_input_rejects_before_digest_or_session_creation(self):
        agent = _make_agent()
        agent._claude_sdk_session = None
        result = run_claude_agent_sdk_turn(
            agent,
            user_message=[],
            original_user_message=[],
            messages=[
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": []},
            ],
            effective_task_id="task-1",
        )
        assert result["api_calls"] == 0
        assert result["partial"] is True
        assert result["interrupted"] is False
        assert agent._claude_sdk_session is None

    def test_empty_rejection_consumes_interrupts_and_next_turn_runs(self):
        agent = _make_agent()
        session = agent._claude_sdk_session
        agent._interrupt_requested = True
        rejected = run_claude_agent_sdk_turn(
            agent,
            user_message="   ",
            original_user_message="   ",
            messages=[{"role": "user", "content": "   "}],
            effective_task_id="task-1",
        )
        assert rejected["api_calls"] == 0
        assert agent._interrupt_requested is False
        session.consume_interrupt.assert_called_once()
        session.run_turn.assert_not_called()

        accepted = run_claude_agent_sdk_turn(
            agent,
            user_message="continue",
            original_user_message="continue",
            messages=[{"role": "user", "content": "continue"}],
            effective_task_id="task-2",
        )
        assert accepted["api_calls"] == 1
        session.run_turn.assert_called_once_with(user_input="continue")

    def test_native_image_with_continuity_digest_preserves_blocks(
        self, monkeypatch
    ):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        seen = {}

        class SpySession:
            def __init__(self, **kwargs):
                pass

            def run_turn(self, user_input):
                seen["input"] = user_input
                return _make_turn()

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        image = {
            "type": "image_url",
            "image_url": {"url": "https://example.test/photo.png"},
        }
        run_claude_agent_sdk_turn(
            agent,
            user_message=[image],
            original_user_message=[image],
            messages=[
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": [image]},
            ],
            effective_task_id="task-1",
        )
        assert seen["input"][0]["type"] == "text"
        assert "continuity" in seen["input"][0]["text"].lower()
        assert seen["input"][1] == {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.test/photo.png",
            },
        }

    def test_transport_local_result_is_visible_without_usage_count(self):
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            api_call_made=False,
            error="local rejection",
            final_text="local rejection",
            projected_messages=[],
            token_usage_last=None,
        )
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="non-empty local input",
            original_user_message="non-empty local input",
            messages=[{"role": "user", "content": "non-empty local input"}],
            effective_task_id="task-1",
        )
        assert result["final_response"] == "local rejection"
        assert result["api_calls"] == 0
        assert result["partial"] is True
        assert agent.session_api_calls == 0

    def test_retire_closes_session(self):
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            should_retire=True, error="turn timed out after 600s",
            projected_messages=[], final_text="", token_usage_last=None,
        )
        stale = agent._claude_sdk_session
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        stale.close.assert_called_once()
        assert agent._claude_sdk_session is None
        assert result["partial"] is True


# ---------- background review spawns only when routed off this runtime ----------


class TestBackgroundReviewRouting:
    @staticmethod
    def _route(monkeypatch, routed):
        import agent.background_review as br

        monkeypatch.setattr(
            br, "_resolve_review_runtime", lambda _agent: {"routed": routed}
        )

    def _run(self, agent, *, want_memory=False):
        return run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
            should_review_memory=want_memory,
        )

    def test_unrouted_memory_nudge_does_not_spawn(self, monkeypatch):
        self._route(monkeypatch, False)
        agent = _make_agent()
        self._run(agent, want_memory=True)
        agent._spawn_background_review.assert_not_called()

    def test_unrouted_skill_nudge_does_not_spawn_but_counter_still_ticks(
        self, monkeypatch
    ):
        self._route(monkeypatch, False)
        agent = _make_agent()
        agent._skill_nudge_interval = 1
        agent.valid_tool_names = set()
        self._run(agent)
        agent._spawn_background_review.assert_not_called()
        assert agent._iters_since_skill == 0

    def test_routed_memory_nudge_spawns_the_review(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        self._run(agent, want_memory=True)
        agent._spawn_background_review.assert_called_once()
        assert agent._spawn_background_review.call_args.kwargs["review_memory"]

    def test_routed_skill_nudge_spawns_the_review(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        agent._skill_nudge_interval = 1
        agent.valid_tool_names = set()
        self._run(agent)
        agent._spawn_background_review.assert_called_once()
        assert agent._spawn_background_review.call_args.kwargs["review_skills"]

    def test_skill_review_does_not_need_foreground_skill_manage(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        agent._skill_nudge_interval = 1
        agent.valid_tool_names = set()
        self._run(agent)
        agent._spawn_background_review.assert_called_once()
        assert agent._spawn_background_review.call_args.kwargs["review_skills"]

    def test_skip_background_review_blocks_routed_skill_review(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        agent.skip_background_review = True
        agent._skill_nudge_interval = 1
        agent.valid_tool_names = set()
        self._run(agent)
        agent._spawn_background_review.assert_not_called()

    def test_dead_turn_never_spawns_even_when_routed(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            interrupted=True, final_text=""
        )
        self._run(agent, want_memory=True)
        agent._spawn_background_review.assert_not_called()

    def test_broken_resolver_falls_back_to_skipping(self, monkeypatch):
        import agent.background_review as br

        def _boom(_agent):
            raise RuntimeError("no config")

        monkeypatch.setattr(br, "_resolve_review_runtime", _boom)
        agent = _make_agent()
        result = self._run(agent, want_memory=True)
        agent._spawn_background_review.assert_not_called()
        assert result["final_response"] == "SDK_ASSISTANT"

    def test_raising_spawn_does_not_take_the_turn_down(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        agent._spawn_background_review.side_effect = RuntimeError("thread fail")
        result = self._run(agent, want_memory=True)
        agent._spawn_background_review.assert_called_once()
        assert result["final_response"] == "SDK_ASSISTANT"


# ---------- provider wiring ----------


class TestProviderWiring:
    def test_profile_registered_with_aliases(self):
        from providers import get_provider_profile

        profile = get_provider_profile("claude-agent-sdk")
        assert profile is not None
        assert profile.api_mode == "claude_agent_sdk"
        assert profile.auth_type == "oauth_external"
        assert get_provider_profile("claude-sdk") is profile
        # The anthropic profile keeps its own alias namespace untouched.
        anthropic = get_provider_profile("claude")
        assert anthropic is not None and anthropic.name == "anthropic"

    def test_runtime_resolution_short_circuit(self):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="claude-agent-sdk")
        assert runtime["provider"] == "claude-agent-sdk"
        assert runtime["api_mode"] == "claude_agent_sdk"
        # No credential-pool machinery, no metered key.
        assert runtime["api_key"] == "claude-subscription-oauth"

    def test_api_mode_accepted_by_agent_init(self):
        from hermes_cli.runtime_provider import _parse_api_mode

        assert _parse_api_mode("claude_agent_sdk") == "claude_agent_sdk"


class TestModelAttribution:
    """F3 (#65982 independent verification): with model.default unset the
    usage rows carried model='unknown' while the SDK's own AssistantMessage
    knew the real id — capture it and back-fill the attribution."""

    def test_session_captures_model_last_from_assistant_message(self):
        script = [
            AssistantMessage(
                content=[TextBlock("hey")], model="claude-opus-4-8-20260115"
            ),
            ResultMessage(
                result="hey", usage={"input_tokens": 1, "output_tokens": 1}
            ),
        ]
        session, _holder = _make_session(script=script)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.model_last == "claude-opus-4-8-20260115"

    def test_usage_row_backfills_model_from_turn(self):
        from agent.claude_sdk_runtime_usage import _record_claude_sdk_usage

        agent = _make_agent()
        agent.model = ""
        db = MagicMock()
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = "sess-attr-1"
        turn = _make_turn(model_last="claude-opus-4-8-20260115")
        _record_claude_sdk_usage(agent, turn)
        kwargs = db.update_token_counts.call_args.kwargs
        assert kwargs["model"] == "claude-opus-4-8-20260115"

    def test_explicit_agent_model_still_wins(self):
        from agent.claude_sdk_runtime_usage import _record_claude_sdk_usage

        agent = _make_agent()
        agent.model = "claude-sonnet-5"
        db = MagicMock()
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = "sess-attr-2"
        turn = _make_turn(model_last="claude-opus-4-8-20260115")
        _record_claude_sdk_usage(agent, turn)
        kwargs = db.update_token_counts.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-5"

    def test_explicit_metered_turn_is_not_recorded_as_subscription_included(self):
        from agent.claude_sdk_runtime_usage import _record_claude_sdk_usage

        agent = _make_agent()
        db = MagicMock()
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = "sess-metered-1"
        turn = _make_turn(
            billing_mode="sdk_reported_metered",
            total_cost_usd=0.25,
        )
        result = _record_claude_sdk_usage(agent, turn)
        kwargs = db.update_token_counts.call_args.kwargs
        assert kwargs["billing_mode"] == "sdk_reported_metered"
        assert kwargs["actual_cost_usd"] == 0.25
        assert kwargs["cost_status"] == "reported"
        assert result["cost_status"] == "reported"
        assert result["actual_cost_usd"] == 0.25

    def test_missing_billing_evidence_is_not_recorded_as_included(self):
        from agent.claude_sdk_runtime_usage import _record_claude_sdk_usage

        agent = _make_agent()
        db = MagicMock()
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = "sess-unverified-1"
        turn = _make_turn(billing_mode=None)
        result = _record_claude_sdk_usage(agent, turn)
        kwargs = db.update_token_counts.call_args.kwargs
        assert kwargs["billing_mode"] == "unknown"
        assert kwargs["cost_status"] == "unknown"
        assert kwargs["cost_source"] == "claude-agent-sdk-unverified"
        assert result["cost_status"] == "unknown"


class TestTurnOrchestrationOrder:
    """The orderings run_claude_agent_sdk_turn's phases must keep, pinned
    beside the per-topic suites: (i) attempt effects reset BEFORE session
    startup; (iii) terminal/stop state reconciled BEFORE the provider-fallback
    decision. (ii) record-before-sink and (iv) flush-before-persist are pinned
    by test_stream_relay_records_delivery_before_display_callback and
    test_resume_id_persisted_after_flush_and_gated_on_persist_disabled."""

    def test_session_startup_sees_the_reset_attempt_ledger(self, monkeypatch):
        import agent.claude_sdk_runtime_session as session_mod

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._current_streamed_assistant_text = "prior turn output"
        agent._sdk_issued_tool_effect = True
        seen = {}

        def fake_create_session(
            agent_, *, resume_id, on_interim_assistant, on_tool_iteration
        ):
            seen["streamed"] = agent_._current_streamed_assistant_text
            seen["tool_effect"] = agent_._sdk_issued_tool_effect
            seen["callbacks"] = (on_interim_assistant, on_tool_iteration)
            agent_._claude_sdk_session = MagicMock()
            agent_._claude_sdk_session.run_turn.return_value = _make_turn()

        monkeypatch.setattr(session_mod, "_create_session", fake_create_session)

        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )

        assert seen["streamed"] == ""
        assert seen["tool_effect"] is False
        assert all(callable(cb) for cb in seen["callbacks"])
        assert result["completed"] is True

    def test_stopped_errored_turn_never_hands_off_to_another_provider(self):
        agent = _make_agent()
        stopped_turn = _make_turn(
            interrupted=True,
            error="HTTP 429 rate limit exceeded",
            should_retire=True,
            final_text="",
            projected_messages=[],
        )

        def run_turn(**_kwargs):
            # The /stop lands DURING the turn: the agent-level flag is raised
            # while the session is still running, then the turn retires.
            agent._interrupt_requested = True
            return stopped_turn

        agent._claude_sdk_session.run_turn.side_effect = run_turn

        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )

        assert "failover_reason" not in result
        assert result["interrupted"] is True
        assert result["failed"] is False
        assert result["sdk_effects"]["interrupted"] is True
        assert agent._claude_sdk_session is None
        assert agent._interrupt_requested is False
