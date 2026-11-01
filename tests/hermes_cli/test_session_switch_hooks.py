"""Tests for the CLI-owned session-switch plugin hooks."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


_PRE_HOOK = "on_session_pre_switch"
_POST_HOOK = "on_session_post_switch"
_SWITCH_HOOKS = {_PRE_HOOK, _POST_HOOK}


class _SessionDB:
    """Small durable-row fake used to prove post-switch ordering."""

    def __init__(self):
        self.sessions = {}

    def end_session(self, session_id, reason):
        self.sessions.setdefault(session_id, {})["end_reason"] = reason

    def create_session(self, **fields):
        self.sessions[fields["session_id"]] = dict(fields)

    def get_session(self, session_id):
        return self.sessions.get(session_id)


class TestSessionSwitchHooks:
    """Verify the pre/post observers bracket one successful ``new_session`` switch."""

    @staticmethod
    def _make_cli(**overrides):
        from cli import HermesCLI

        cli: Any = object.__new__(HermesCLI)
        cli.session_id = "old-session-id"
        cli.config = {}
        cli.model = "test-model"
        cli.max_turns = 10
        cli.reasoning_config = None
        cli.agent = MagicMock()
        cli.conversation_history = []
        cli._session_db = _SessionDB()
        cli._pending_title = None
        cli._console_print = lambda text: None
        cli._active_session_lease = None
        cli._discard_session_if_empty = MagicMock(return_value=False)
        cli._notify_session_boundary = MagicMock()
        for key, value in overrides.items():
            setattr(cli, key, value)
        return cli

    @staticmethod
    def _switch_calls(mock_invoke):
        return [
            (call.args[0], call.kwargs)
            for call in mock_invoke.call_args_list
            if call.args[0] in _SWITCH_HOOKS
        ]

    def test_hooks_fire_once_in_order_on_successful_switch(self):
        cli = self._make_cli()

        with (
            patch("hermes_cli.plugins.has_hook", return_value=True),
            patch("hermes_cli.plugins.invoke_hook") as mock_invoke,
            patch("cli._sync_process_session_id"),
        ):
            cli.new_session(silent=True)

        calls = self._switch_calls(mock_invoke)
        assert [name for name, _kwargs in calls] == [_PRE_HOOK, _POST_HOOK]
        assert calls[0][1]["old_session_id"] == "old-session-id"
        assert "new_session_id" not in calls[0][1]
        assert calls[0][1]["cli"] is cli
        assert calls[1][1]["old_session_id"] == "old-session-id"
        assert calls[1][1]["new_session_id"] == cli.session_id
        assert calls[1][1]["new_session_id"] != "old-session-id"
        assert calls[1][1]["cli"] is cli

    def test_has_hook_false_skips_both_switch_hooks(self):
        cli = self._make_cli()

        with (
            patch("hermes_cli.plugins.has_hook", return_value=False),
            patch("hermes_cli.plugins.invoke_hook") as mock_invoke,
            patch("cli._sync_process_session_id"),
        ):
            cli.new_session(silent=True)

        assert self._switch_calls(mock_invoke) == []

    def test_dispatch_errors_are_fail_open_for_observers(self):
        cli = self._make_cli()

        with (
            patch("hermes_cli.plugins.has_hook", return_value=True),
            patch("hermes_cli.plugins.invoke_hook", side_effect=RuntimeError("boom")),
            patch("cli._sync_process_session_id"),
        ):
            cli.new_session(silent=True)

        assert cli.session_id != "old-session-id"

    def test_post_hook_observes_durable_row_and_reset_boundary(self):
        cli = self._make_cli()
        observed = []

        def invoke(name, **kwargs):
            if name == _POST_HOOK:
                new_session_id = kwargs["new_session_id"]
                observed.append(cli._session_db.get_session(new_session_id))
                cli._notify_session_boundary.assert_called_with("on_session_reset")

        with (
            patch("hermes_cli.plugins.has_hook", return_value=True),
            patch("hermes_cli.plugins.invoke_hook", side_effect=invoke),
            patch("cli._sync_process_session_id"),
        ):
            cli.new_session(silent=True)

        assert observed == [cli._session_db.get_session(cli.session_id)]
        assert observed[0]["session_id"] == cli.session_id

    def test_post_hook_is_not_called_when_switch_raises(self):
        cli = self._make_cli()
        cli.agent.reset_session_state.side_effect = RuntimeError("switch failed")

        with (
            patch("hermes_cli.plugins.has_hook", return_value=True),
            patch("hermes_cli.plugins.invoke_hook") as mock_invoke,
            patch("cli._sync_process_session_id"),
            pytest.raises(RuntimeError, match="switch failed"),
        ):
            cli.new_session(silent=True)

        assert [name for name, _kwargs in self._switch_calls(mock_invoke)] == [_PRE_HOOK]

    def test_post_hook_fires_for_switch_without_attached_agent(self):
        cli = self._make_cli(agent=None)

        with (
            patch("hermes_cli.plugins.has_hook", return_value=True),
            patch("hermes_cli.plugins.invoke_hook") as mock_invoke,
            patch("cli._sync_process_session_id"),
        ):
            cli.new_session(silent=True)

        assert [name for name, _kwargs in self._switch_calls(mock_invoke)] == [
            _PRE_HOOK,
            _POST_HOOK,
        ]


def test_session_switch_hook_registration_handles_cleanup_callbacks():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="session-switch-test", key="session-switch-test"),
        manager,
    )
    calls = []

    def callback(**kwargs):
        calls.append(kwargs)

    handles = [context.register_hook(name, callback) for name in (_PRE_HOOK, _POST_HOOK)]
    assert all(manager.has_hook(name) for name in _SWITCH_HOOKS)

    manager.invoke_hook(_PRE_HOOK, old_session_id="old", cli=object())
    assert len(calls) == 1

    for handle in handles:
        handle.dispose()

    assert all(not manager.has_hook(name) for name in _SWITCH_HOOKS)
    assert manager.invoke_hook(_PRE_HOOK, old_session_id="old", cli=object()) == []
    assert len(calls) == 1
