"""CLI + gateway wiring tests for adaptive reasoning escalation.

Covers manual-override precedence plumbing (/reasoning session vs --global),
per-session reset on /new, the immediate rendering of TTL notices in the
REPL, and the gateway's session-override detector.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestReasoningCommandSetsOverrideFlag(unittest.TestCase):
    """A session-scoped /reasoning pick suppresses adaptive escalation; a
    --global pick rewrites the baseline and keeps escalation active."""

    def _make_cli(self):
        return SimpleNamespace(
            reasoning_config={"enabled": True, "effort": "medium"},
            show_reasoning=False,
            agent=MagicMock(),
            _current_reasoning_callback=lambda: None,
        )

    def test_session_scoped_pick_sets_override(self):
        from hermes_cli.cli_commands_mixin import CLICommandsMixin

        stub = self._make_cli()
        with patch("cli.save_config_value") as save_config, patch("cli._cprint"):
            CLICommandsMixin._handle_reasoning_command(stub, "/reasoning high")

        save_config.assert_not_called()
        self.assertTrue(stub._session_reasoning_override)

    def test_global_pick_does_not_set_override(self):
        from hermes_cli.cli_commands_mixin import CLICommandsMixin

        stub = self._make_cli()
        with patch("cli.save_config_value", return_value=True), patch("cli._cprint"):
            CLICommandsMixin._handle_reasoning_command(
                stub, "/reasoning high --global"
            )

        self.assertFalse(stub._session_reasoning_override)

    def test_display_toggle_does_not_touch_override(self):
        from hermes_cli.cli_commands_mixin import CLICommandsMixin

        stub = self._make_cli()
        with patch("cli.save_config_value", return_value=True), patch("cli._cprint"):
            CLICommandsMixin._handle_reasoning_command(stub, "/reasoning show")

        self.assertFalse(getattr(stub, "_session_reasoning_override", False))


class TestInitPromptSetsOverrideFlag(unittest.TestCase):
    """--reasoning is re-homed onto CLIInitMixin after the cli.py split."""

    def _call(self, reasoning):
        from hermes_cli.cli_init_mixin import CLIInitMixin

        stub = SimpleNamespace(model="test-model")
        with patch("hermes_cli.personality.available_personalities", return_value={}), patch(
            "hermes_cli.personality.resolve_ephemeral_system_prompt", return_value=""
        ), patch(
            "hermes_constants.resolve_reasoning_config",
            return_value={"enabled": True, "effort": "medium"},
        ), patch("cli._load_prefill_messages", return_value=[]), patch(
            "cli._resolve_prefill_messages_file", return_value=""
        ), patch(
            "cli._parse_reasoning_config",
            side_effect=lambda value: {"enabled": True, "effort": str(value).strip()}
            if value
            else None,
        ), patch("cli._parse_service_tier_config", return_value=None), patch(
            "cli.CLI_CONFIG", {"agent": {}, "provider_routing": {}, "openrouter": {}}
        ):
            CLIInitMixin._init_prompt_and_reasoning(stub, reasoning)
        return stub

    def test_explicit_reasoning_sets_override(self):
        stub = self._call("high")
        self.assertTrue(stub._session_reasoning_override)
        self.assertEqual(stub.reasoning_config, {"enabled": True, "effort": "high"})

    def test_missing_reasoning_keeps_override_false(self):
        stub = self._call(None)
        self.assertFalse(stub._session_reasoning_override)


class TestNewSessionResetsOverrideFlag(unittest.TestCase):
    def test_new_session_clears_session_reasoning_override(self):
        from cli import CLI_CONFIG, HermesCLI

        agent = SimpleNamespace(
            reasoning_config={"enabled": True, "effort": "high"},
            reset_session_state=MagicMock(),
        )
        stub = SimpleNamespace(
            agent=agent,
            conversation_history=[],
            session_id="old-session",
            _session_db=None,
            _pending_title=None,
            _resumed=False,
            reasoning_config={"enabled": True, "effort": "high"},
            _session_reasoning_override=True,
            _notify_session_boundary=MagicMock(),
        )

        with patch.dict(CLI_CONFIG.setdefault("agent", {}), {"reasoning_effort": "medium"}):
            HermesCLI.new_session(stub, silent=True)

        self.assertFalse(stub._session_reasoning_override)
        self.assertEqual(stub.reasoning_config, {"enabled": True, "effort": "medium"})


class TestTtlNoticeRendersImmediately(unittest.TestCase):
    """The adaptive escalation notice prints at emission time; credit notices
    (sticky, or the mid-turn TTL recovery line) keep the end-of-turn queue."""

    def _make_cli(self):
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        cli._pending_credit_notices = []
        return cli

    def test_ttl_notice_prints_and_is_not_queued(self):
        from agent.credits_tracker import AgentNotice

        cli = self._make_cli()
        notice = AgentNotice(
            text="🧠 Reasoning raised to High for this task — debugging/diagnosis.",
            level="info",
            kind="ttl",
            ttl_ms=12000,
            key="adaptive-reasoning",
        )
        with patch("cli._cprint") as mock_cprint:
            cli._on_notice(notice)

        self.assertEqual(mock_cprint.call_count, 1)
        self.assertIn("Reasoning raised to High", mock_cprint.call_args[0][0])
        self.assertEqual(cli._pending_credit_notices, [])

    def test_sticky_notice_still_queues(self):
        from agent.credits_tracker import AgentNotice

        cli = self._make_cli()
        notice = AgentNotice(text="Credits low", level="warn", kind="sticky")
        with patch("cli._cprint") as mock_cprint:
            cli._on_notice(notice)

        mock_cprint.assert_not_called()
        self.assertEqual(cli._pending_credit_notices, [("warn", "Credits low")])

    def test_mid_turn_ttl_credit_notice_still_queues(self):
        from agent.credits_tracker import AgentNotice

        cli = self._make_cli()
        notice = AgentNotice(
            text="✓ Credit access restored", level="success", kind="ttl",
            ttl_ms=8000, key="credits.restored",
        )
        with patch("cli._cprint") as mock_cprint:
            cli._on_notice(notice)

        mock_cprint.assert_not_called()
        self.assertEqual(cli._pending_credit_notices, [("success", "✓ Credit access restored")])


class TestGatewaySessionOverrideDetector(unittest.TestCase):
    """GatewayRunner._session_reasoning_override_active mirrors the session
    override the per-turn resolver honors."""

    def _stub_runner(self, override):
        state = (
            None
            if override == "__no_state__"
            else SimpleNamespace(
                conversation=SimpleNamespace(reasoning_override=override)
            )
        )
        return SimpleNamespace(_peek_session_state=lambda _key: state)

    def _call(self, runner, key):
        from gateway.run import GatewayRunner

        return GatewayRunner._session_reasoning_override_active(runner, key)

    def test_active_override_detected(self):
        runner = self._stub_runner({"enabled": True, "effort": "low"})
        self.assertTrue(self._call(runner, "session-1"))

    def test_no_override_or_state_is_inactive(self):
        self.assertFalse(self._call(self._stub_runner(None), "session-1"))
        self.assertFalse(self._call(self._stub_runner("__no_state__"), "session-1"))

    def test_missing_session_key_is_inactive(self):
        runner = self._stub_runner({"enabled": True, "effort": "low"})
        self.assertFalse(self._call(runner, ""))


class TestNewSessionClearsRetainedAgentOverride(unittest.TestCase):
    """/new on a REUSED agent: the shell flag and the live agent's provenance both reset, so the
    fresh session adapts again; a /resume-style reset_session_state alone does not touch it."""

    def _stub(self, agent):
        return SimpleNamespace(
            agent=agent, conversation_history=[], session_id="old-session", _session_db=None,
            _pending_title=None, _resumed=False, reasoning_config={"enabled": True, "effort": "high"},
            _session_reasoning_override=True, _notify_session_boundary=MagicMock(),
        )

    def test_new_session_clears_reused_agent_reasoning_user_override(self):
        from cli import CLI_CONFIG, HermesCLI
        from agent.adaptive_reasoning import adaptive_reasoning_turn

        agent = SimpleNamespace(
            reasoning_config={"enabled": True, "effort": "high"}, reasoning_user_override=True,
            adaptive_reasoning={"enabled": True, "max_effort": "xhigh"},
            _adaptive_prev_effort=None, _adaptive_last_notified_effort=None,
            notice_callback=None, platform="cli", reset_session_state=MagicMock(),
        )
        stub = self._stub(agent)
        with patch.dict(CLI_CONFIG.setdefault("agent", {}), {"reasoning_effort": "medium"}):
            HermesCLI.new_session(stub, silent=True)

        agent.reset_session_state.assert_called_once()
        self.assertIs(stub.agent, agent, "the agent is reused, not rebuilt")
        self.assertFalse(stub._session_reasoning_override)
        self.assertIs(agent.reasoning_user_override, False)
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "medium"})
        # Executable outcome: the fresh session adapts above the config baseline again.
        with adaptive_reasoning_turn(agent, "Why does the gateway keep failing after I restart it? "
                                            "error: connection refused"):
            self.assertEqual(agent.reasoning_config["effort"], "high")
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "medium"})

    def test_reset_session_state_alone_preserves_provenance(self):
        """/resume and /branch reuse reset_session_state with the shell pick still active."""
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent.reasoning_user_override = True
        agent._transition_context_engine_session = lambda **_kw: None
        AIAgent.reset_session_state(agent)
        self.assertIs(agent.reasoning_user_override, True)


class TestModelSwitchReasoningProvenance(unittest.TestCase):
    """`/model X --reasoning L` (typed or picker) is an explicit user pick with the pick's scope."""

    DEBUG = "Why does the gateway keep failing after I restart it? error: connection refused"

    def _cli(self, agent):
        import cli as cli_mod

        stub = SimpleNamespace(
            model="old", provider="nous", requested_provider="nous", _explicit_api_key="",
            _explicit_base_url="", api_key="", base_url="", api_mode="", agent=agent,
            reasoning_config={"enabled": True, "effort": "medium"}, _session_reasoning_override=False,
            _pending_one_turn_model_restore=None, _pending_model_switch_note="",
            _persist_model_switch_to_session=lambda *_a: None)
        stub._stage_and_swap_model = lambda result, old: cli_mod.HermesCLI._stage_and_swap_model(stub, result, old)
        stub._snapshot_model_runtime = lambda: cli_mod.HermesCLI._snapshot_model_runtime(stub)
        stub._restore_model_runtime_snapshot = (
            lambda snap: cli_mod.HermesCLI._restore_model_runtime_snapshot(stub, snap))
        return stub

    def _commit(self, stub, **kw):
        import cli as cli_mod
        from hermes_cli import cli_model_switch_mixin as mixin
        from hermes_cli.model_switch import ModelSwitchResult

        result = ModelSwitchResult(success=True, new_model="new", target_provider="nous")
        with patch.object(mixin, "_print_switch_summary", lambda *_a, **_k: None), patch.object(
                cli_mod.HermesCLI, "_persist_model_switch_to_session", lambda *_a: None), patch(
                "cli.save_config_value", return_value=True), patch("cli._cprint"), patch(
                "hermes_cli.model_switch.persist_model_selection", lambda *_a: None), patch(
                "hermes_constants.resolve_reasoning_config",
                return_value={"enabled": True, "effort": "medium"}):
            mixin._commit_model_switch(stub, result, **kw)

    def _live_agent(self):
        class _Agent:
            reasoning_config = {"enabled": True, "effort": "medium"}
            reasoning_user_override = False
            adaptive_reasoning = {"enabled": True, "max_effort": "xhigh", "min_effort": "low"}
            _adaptive_prev_effort = None
            _adaptive_last_notified_effort = None
            notice_callback = None
            platform = "cli"

            def switch_model(self, **_kw):
                self.reasoning_config = {"enabled": True, "effort": "medium"}
        return _Agent()

    def test_pre_first_turn_session_pick_survives_into_built_agent(self):
        from agent.adaptive_reasoning import adaptive_reasoning_turn

        stub = self._cli(None)
        self._commit(stub, persist_global=False, reasoning_effort="low")
        self.assertTrue(stub._session_reasoning_override)
        # _init_agent forwards the shell flag at build time (cli_agent_setup_mixin).
        built = self._live_agent()
        built.reasoning_config = stub.reasoning_config
        built.reasoning_user_override = bool(stub._session_reasoning_override)
        with adaptive_reasoning_turn(built, self.DEBUG):
            self.assertEqual(built.reasoning_config["effort"], "low")

    def test_live_agent_session_and_picker_picks_are_user_overrides(self):
        from agent.adaptive_reasoning import adaptive_reasoning_turn

        for picker in (False, True):
            with self.subTest(picker=picker):
                agent = self._live_agent()
                stub = self._cli(agent)
                self._commit(stub, persist_global=False, picker=picker, reasoning_effort="medium")
                self.assertTrue(stub._session_reasoning_override)
                self.assertIs(agent.reasoning_user_override, True)
                with adaptive_reasoning_turn(agent, self.DEBUG):
                    self.assertEqual(agent.reasoning_config["effort"], "medium")

    def test_global_pick_is_a_new_baseline_not_an_override(self):
        from agent.adaptive_reasoning import adaptive_reasoning_turn

        agent = self._live_agent()
        agent.reasoning_user_override = True
        stub = self._cli(agent)
        stub._session_reasoning_override = True
        self._commit(stub, persist_global=True, picker=True, reasoning_effort="medium")
        self.assertFalse(stub._session_reasoning_override)
        self.assertIs(agent.reasoning_user_override, False)
        with adaptive_reasoning_turn(agent, self.DEBUG):
            self.assertEqual(agent.reasoning_config["effort"], "high")

    def test_once_pick_pins_one_turn_then_restores_prior_provenance(self):
        agent = self._live_agent()
        stub = self._cli(agent)
        self._commit(stub, persist_global=False, one_turn=True, reasoning_effort="low")
        self.assertIs(agent.reasoning_user_override, True)
        self.assertTrue(stub._session_reasoning_override)
        stub._restore_model_runtime_snapshot(stub._pending_one_turn_model_restore)
        self.assertFalse(stub._session_reasoning_override)
        self.assertIs(agent.reasoning_user_override, False)
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "medium"})


class TestGatewayCachedAgentAdaptiveRefresh(unittest.TestCase):
    """The gateway caches agents across turns; per-turn wiring must refresh the adaptive policy
    from the CURRENT config and the override flag from the CURRENT session state."""

    DEBUG = "Why does the gateway keep failing after I restart it? error: connection refused"

    def _wire(self, agent, user_config, override):
        import types
        from gateway.run import GatewayRunner
        from gateway.run_turn_runner import TurnRunner

        state = SimpleNamespace(conversation=SimpleNamespace(reasoning_override=override))
        runner = SimpleNamespace(
            _service_tier=None, _consume_pending_turn_sidecar_notes=lambda key: [],
            _peek_session_state=lambda key: state if key == "chat-1" else None)
        runner._session_reasoning_override_active = types.MethodType(
            GatewayRunner._session_reasoning_override_active, runner)
        ctx = SimpleNamespace(
            progress_callback=None, native_tool_start_callback=None, voice_ack_callback=None,
            _voice_ack_guild=[None], voice_turn=False, _native_slack_task_cards=False,
            native_tool_complete_callback=None,
            _step_callback_sync=None, _hooks_ref=SimpleNamespace(loaded_hooks=[]),
            _status_callback_sync=None, _event_callback_sync=None, _status_adapter=None,
            session_key="chat-1", user_config=user_config, source=SimpleNamespace(platform="telegram"),
            mute_notification_reply=False, _thinking_enabled=False, agent_holder=[None],
            tools_holder=[None], process_task_id=None, process_baseline=None, run_generation=0)
        holder = SimpleNamespace(
            _ctx=ctx, _runner=runner, _make_bg_review_callbacks=lambda: (lambda m: None, lambda: None),
            _merge_turn_request_overrides=TurnRunner._merge_turn_request_overrides,
            _clarify_callback_sync=lambda *a, **k: None, _notice_callback_sync=lambda *a, **k: None,
            _attach_session_title_callback=lambda agent, ctx: None)
        TurnRunner._wire_turn_agent_callbacks(
            holder, agent, {}, override or {"enabled": True, "effort": "medium"}, None, None, False)

    def test_cached_agent_tracks_config_and_session_override_each_turn(self):
        from agent.adaptive_reasoning import adaptive_reasoning_turn

        agent = SimpleNamespace(
            reasoning_user_override=False, adaptive_reasoning=None,
            _adaptive_prev_effort=None, _adaptive_last_notified_effort=None, platform="telegram")
        on = {"agent": {"adaptive_reasoning": {"enabled": True, "max_effort": "high"}}}
        off = {"agent": {"adaptive_reasoning": {"enabled": False}}}

        # Turn 1: policy enabled, no session pick -> escalates (capped at max_effort).
        self._wire(agent, on, None)
        self.assertEqual(agent.adaptive_reasoning["max_effort"], "high")
        self.assertIs(agent.reasoning_user_override, False)
        with adaptive_reasoning_turn(agent, self.DEBUG):
            self.assertEqual(agent.reasoning_config["effort"], "high")
        # Turn 2: the session gets a /reasoning pick -> same cached agent is pinned.
        self._wire(agent, on, {"enabled": True, "effort": "low"})
        self.assertIs(agent.reasoning_user_override, True)
        with adaptive_reasoning_turn(agent, self.DEBUG):
            self.assertEqual(agent.reasoning_config["effort"], "low")
        # Turn 3: /reasoning reset clears the pick -> adapts again.
        self._wire(agent, on, None)
        self.assertIs(agent.reasoning_user_override, False)
        with adaptive_reasoning_turn(agent, self.DEBUG):
            self.assertEqual(agent.reasoning_config["effort"], "high")
        # Turn 4: config edit disables the feature -> the cached agent stops adapting.
        self._wire(agent, off, None)
        self.assertIsNone(agent.adaptive_reasoning)
        with adaptive_reasoning_turn(agent, self.DEBUG):
            self.assertEqual(agent.reasoning_config["effort"], "medium")


if __name__ == "__main__":
    unittest.main()
