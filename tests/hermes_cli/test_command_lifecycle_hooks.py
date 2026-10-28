"""CLI command lifecycle hook integration tests.

``pre_command`` is shared with the gateway. ``post_command`` and ``on_quit``
are CLI-only terminal observers that use the same metadata-only envelope.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import VALID_HOOKS


_COMMAND_PAYLOAD_KEYS = {
    "surface",
    "command",
    "alias_used",
    "args_raw",
    "session_key",
    "platform",
}


def _make_cli(*, quick_commands=None):
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"quick_commands": quick_commands or {}}
    cli.session_id = "test-session"
    cli._pending_resume_sessions = None
    cli._delete_session_on_exit = False
    cli.show_help = MagicMock()
    return cli


def _capture_lifecycle(events):
    return (
        patch(
            "hermes_cli.plugins.fire_pre_command_hook",
            side_effect=lambda **kwargs: events.append(("pre_command", kwargs)),
        ),
        patch(
            "hermes_cli.plugins.fire_post_command_hook",
            side_effect=lambda **kwargs: events.append(("post_command", kwargs)),
        ),
        patch(
            "hermes_cli.plugins.fire_on_quit_hook",
            side_effect=lambda **kwargs: events.append(("on_quit", kwargs)),
        ),
    )


def test_command_lifecycle_hooks_are_registered():
    assert {"pre_command", "post_command", "on_quit"} <= VALID_HOOKS


@pytest.mark.parametrize(
    ("fire_name", "hook_name"),
    (("fire_post_command_hook", "post_command"), ("fire_on_quit_hook", "on_quit")),
)
def test_terminal_helpers_use_pre_command_envelope(monkeypatch, fire_name, hook_name):
    calls = []

    class _Manager:
        def has_hook(self, name):
            return name == hook_name

        def invoke_hook(self, name, **kwargs):
            calls.append((name, kwargs))
            return [{"ignored": True}]

    monkeypatch.setattr(plugins_mod, "get_plugin_manager", _Manager)

    getattr(plugins_mod, fire_name)(
        surface="cli",
        command="help",
        alias_used="help",
        args_raw="Some RAW args",
        session_key="test-session",
        platform="cli",
    )

    assert calls == [
        (
            hook_name,
            {
                "surface": "cli",
                "command": "help",
                "alias_used": "help",
                "args_raw": "Some RAW args",
                "session_key": "test-session",
                "platform": "cli",
            },
        )
    ]
    assert set(calls[0][1]) == _COMMAND_PAYLOAD_KEYS
    assert "cli" not in calls[0][1]


@pytest.mark.parametrize("fire_name", ("fire_post_command_hook", "fire_on_quit_hook"))
def test_terminal_helpers_never_raise(monkeypatch, fire_name):
    class _BrokenManager:
        def has_hook(self, _name):
            return True

        def invoke_hook(self, _name, **_kwargs):
            raise RuntimeError("broken plugin dispatcher")

    monkeypatch.setattr(plugins_mod, "get_plugin_manager", _BrokenManager)

    assert getattr(plugins_mod, fire_name)(
        surface="cli",
        command="help",
        alias_used="help",
        args_raw="",
        session_key="test-session",
        platform="cli",
    ) is None


def test_help_fires_pre_then_post_once_with_matching_payload():
    cli = _make_cli()
    events = []
    pre, post, on_quit = _capture_lifecycle(events)

    with pre, post, on_quit:
        result = cli.process_command("/help Some RAW args")

    assert result is True
    assert [name for name, _payload in events] == ["pre_command", "post_command"]
    assert events[0][1] == events[1][1] == {
        "surface": "cli",
        "command": "help",
        "alias_used": "help",
        "args_raw": "Some RAW args",
        "session_key": "test-session",
        "platform": "cli",
    }
    cast(MagicMock, cli.show_help).assert_called_once_with("Some RAW args")


def test_exit_fires_pre_then_on_quit_once_and_returns_false():
    cli = _make_cli()
    events = []
    pre, post, on_quit = _capture_lifecycle(events)

    with pre, post, on_quit:
        result = cli.process_command("/exit")

    assert result is False
    assert [name for name, _payload in events] == ["pre_command", "on_quit"]
    assert events[0][1] == events[1][1] == {
        "surface": "cli",
        "command": "quit",
        "alias_used": "exit",
        "args_raw": "",
        "session_key": "test-session",
        "platform": "cli",
    }


def test_invalid_exit_fires_post_not_on_quit():
    cli = _make_cli()
    events = []
    pre, post, on_quit = _capture_lifecycle(events)

    with pre, post, on_quit, patch("cli._cprint"):
        result = cli.process_command("/exit not-a-flag")

    assert result is True
    assert [name for name, _payload in events] == ["pre_command", "post_command"]
    assert events[-1][1]["command"] == "quit"
    assert events[-1][1]["alias_used"] == "exit"
    assert events[-1][1]["args_raw"] == "not-a-flag"


def test_quick_alias_redispatch_fires_each_boundary_once():
    cli = _make_cli(
        quick_commands={"zzhookalias": {"type": "alias", "target": "help"}}
    )
    events = []
    pre, post, on_quit = _capture_lifecycle(events)

    with (
        pre,
        post,
        on_quit,
        patch("cli._ensure_skill_commands", return_value={}),
        patch("cli.get_skill_bundles", return_value={}),
    ):
        result = cli.process_command("/zzhookalias Some RAW args")

    assert result is True
    assert [name for name, _payload in events] == ["pre_command", "post_command"]
    assert events[0][1] == events[1][1]
    assert events[0][1]["command"] == "help"
    assert events[0][1]["alias_used"] == "help"
    assert events[0][1]["args_raw"] == "Some RAW args"


def test_unique_prefix_redispatch_fires_each_boundary_once():
    cli = _make_cli()
    events = []
    pre, post, on_quit = _capture_lifecycle(events)

    with (
        pre,
        post,
        on_quit,
        patch("cli._ensure_skill_commands", return_value={}),
        patch("cli.get_skill_bundles", return_value={}),
    ):
        result = cli.process_command("/hel Some RAW args")

    assert result is True
    assert [name for name, _payload in events] == ["pre_command", "post_command"]
    assert events[0][1] == events[1][1]
    assert events[0][1]["command"] == "help"
    assert events[0][1]["alias_used"] == "help"
    assert events[0][1]["args_raw"] == "Some RAW args"


def test_handler_error_leaves_no_reentry_state_for_next_command():
    cli = _make_cli()
    show_help = cast(MagicMock, cli.show_help)
    show_help.side_effect = RuntimeError("handler failed")
    events = []
    pre, post, on_quit = _capture_lifecycle(events)

    with pre, post, on_quit:
        with pytest.raises(RuntimeError, match="handler failed"):
            cli.process_command("/help")

        show_help.side_effect = None
        assert cli.process_command("/help") is True

    assert [name for name, _payload in events] == [
        "pre_command",
        "pre_command",
        "post_command",
    ]
    assert not hasattr(cli, "_pre_command_fired")


def test_broken_quit_hook_cannot_prevent_exit(monkeypatch):
    class _BrokenManager:
        def has_hook(self, _name):
            return True

        def invoke_hook(self, _name, **_kwargs):
            raise RuntimeError("broken plugin dispatcher")

    monkeypatch.setattr(plugins_mod, "get_plugin_manager", _BrokenManager)

    assert _make_cli().process_command("/exit") is False
