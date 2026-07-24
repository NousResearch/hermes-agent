from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.codex_runtime import list_codex_plugins, run_codex_app_server_turn


def _run_with_session(session, original_user_message):
    agent = SimpleNamespace(
        _codex_session=session,
        _interrupt_requested=False,
    )
    return run_codex_app_server_turn(
        agent,
        user_message="display text",
        original_user_message=original_user_message,
        messages=[],
        effective_task_id="task",
    )


def test_runtime_forwards_transport_plugin_mentions():
    mentions = [{"name": "acme", "path": "plugin://acme/tools"}]
    session = Mock()
    session.run_turn.side_effect = RuntimeError("stop after call capture")

    _run_with_session(
        session,
        {"content": "persisted text", "plugin_mentions": mentions},
    )

    session.run_turn.assert_called_once_with(
        user_input="display text",
        plugin_mentions=mentions,
    )


@pytest.mark.parametrize(
    "original_user_message",
    [
        "plain persisted text",
        {"content": "persisted text"},
        {"content": "persisted text", "plugin_mentions": None},
        {"content": "persisted text", "plugin_mentions": []},
        {"content": "persisted text", "plugin_mentions": {"name": "acme"}},
    ],
)
def test_runtime_omits_absent_or_malformed_plugin_mentions(
    original_user_message,
):
    class LegacySession:
        def __init__(self):
            self.user_input = None
            self.closed = False

        def run_turn(self, *, user_input):
            self.user_input = user_input
            raise RuntimeError("stop after call capture")

        def close(self):
            self.closed = True

    session = LegacySession()
    _run_with_session(session, original_user_message)

    assert session.user_input == "display text"
    assert session.closed is True


def test_list_codex_plugins_reuses_live_session():
    inventory = [object()]
    session = Mock()
    session.list_plugins.return_value = inventory

    assert list_codex_plugins(SimpleNamespace(_codex_session=session)) is inventory
    session.list_plugins.assert_called_once_with()
    session.close.assert_not_called()


def test_list_codex_plugins_closes_owned_session_when_unavailable(monkeypatch):
    session = Mock()
    session.list_plugins.side_effect = OSError("codex unavailable")
    factory = Mock(return_value=session)
    monkeypatch.setattr(
        "agent.transports.codex_app_server_session.CodexAppServerSession",
        factory,
    )

    assert list_codex_plugins(cwd="/tmp/project") == []
    factory.assert_called_once_with(cwd="/tmp/project")
    session.close.assert_called_once_with()
