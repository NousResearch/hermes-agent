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
    """TTL notices (adaptive escalation) print at emission time; sticky
    notices (credits) keep the end-of-turn queue."""

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


if __name__ == "__main__":
    unittest.main()
