"""Tests for session-switch plugin hooks in CLI new_session()."""

from unittest.mock import MagicMock, patch

import pytest


class TestSessionSwitchHooks:
    """Verify session_switch_starting / session_switched fire correctly."""

    @staticmethod
    def _make_cli(**overrides):
        from cli import HermesCLI

        cli = object.__new__(HermesCLI)
        cli.session_id = "old-session-id"
        cli.config = {}
        cli.model = "test-model"
        cli.max_turns = 10
        cli.reasoning_config = None
        cli.agent = MagicMock()
        cli.conversation_history = []
        # Mock SessionDB so session_switched fires (lives after create_session)
        mock_db = MagicMock()
        mock_db.create_session = MagicMock()
        cli._session_db = mock_db
        cli._pending_title = None
        cli._console_print = lambda text: None
        cli._active_session_lease = None
        cli._discard_session_if_empty = MagicMock(return_value=False)
        cli._notify_session_boundary = MagicMock()
        for k, v in overrides.items():
            setattr(cli, k, v)
        return cli

    @staticmethod
    def _switch_calls(mock_invoke):
        return [
            (c[0][0], c[1])
            for c in mock_invoke.call_args_list
            if c[0][0].startswith("session_switch")
        ]

    # ── tests ──

    def test_hooks_fire_on_new_session(self):
        cli = self._make_cli()

        with patch("hermes_cli.plugins.has_hook", return_value=True), \
             patch("hermes_cli.plugins.invoke_hook") as mock_invoke, \
             patch("cli._sync_process_session_id"):
            cli.new_session(silent=True)

        calls = self._switch_calls(mock_invoke)
        assert len(calls) == 2, f"expected 2, got {calls}"

        assert calls[0][0] == "session_switch_starting"
        assert calls[0][1]["old_session_id"] == "old-session-id"
        assert calls[0][1]["cli"] is cli

        assert calls[1][0] == "session_switched"
        assert calls[1][1]["old_session_id"] == "old-session-id"
        assert calls[1][1]["new_session_id"] != "old-session-id"
        assert calls[1][1]["cli"] is cli

    def test_has_hook_false_skips_switch_hooks(self):
        cli = self._make_cli()

        def _has_hook(name):
            return not name.startswith("session_switch")

        with patch("hermes_cli.plugins.has_hook", side_effect=_has_hook), \
             patch("hermes_cli.plugins.invoke_hook") as mock_invoke, \
             patch("cli._sync_process_session_id"):
            cli.new_session(silent=True)

        assert self._switch_calls(mock_invoke) == []

    def test_hook_exception_does_not_crash_new_session(self):
        cli = self._make_cli()

        with patch("hermes_cli.plugins.has_hook", return_value=True), \
             patch("hermes_cli.plugins.invoke_hook",
                   side_effect=RuntimeError("boom")), \
             patch("cli._sync_process_session_id"):
            cli.new_session(silent=True)

    def test_starting_hook_receives_old_session_id_only(self):
        cli = self._make_cli()

        with patch("hermes_cli.plugins.has_hook", return_value=True), \
             patch("hermes_cli.plugins.invoke_hook") as mock_invoke, \
             patch("cli._sync_process_session_id"):
            cli.new_session(silent=True)

        kwargs = self._switch_calls(mock_invoke)[0][1]
        assert "old_session_id" in kwargs
        assert "new_session_id" not in kwargs

    def test_switched_hook_receives_both_session_ids(self):
        cli = self._make_cli()

        with patch("hermes_cli.plugins.has_hook", return_value=True), \
             patch("hermes_cli.plugins.invoke_hook") as mock_invoke, \
             patch("cli._sync_process_session_id"):
            cli.new_session(silent=True)

        kwargs = self._switch_calls(mock_invoke)[1][1]
        assert "old_session_id" in kwargs
        assert "new_session_id" in kwargs
