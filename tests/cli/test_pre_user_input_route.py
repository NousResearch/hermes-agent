import queue
from types import SimpleNamespace

from cli import (
    HermesCLI,
    _SyntheticInputMessage,
    _VoiceInputMessage,
    _is_real_user_input,
)


def test_cli_input_route_rewrites_before_slash_dispatch(monkeypatch):
    cli = object.__new__(HermesCLI)
    cli.session_id = "cli-session"
    monkeypatch.setattr(
        HermesCLI,
        "_get_goal_manager",
        lambda _self: SimpleNamespace(is_active=lambda: False),
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.route_pre_user_input",
        lambda **payload: (
            ("/goal ship it", "Routed")
            if payload["text"] == "ship it"
            else (payload["text"], None)
        ),
    )
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)

    assert cli._route_pre_user_input("ship it") == "/goal ship it"


def test_cli_input_route_skips_active_goal(monkeypatch):
    cli = object.__new__(HermesCLI)
    cli.session_id = "cli-session"
    calls = []
    monkeypatch.setattr(
        HermesCLI,
        "_get_goal_manager",
        lambda _self: SimpleNamespace(is_active=lambda: True),
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.route_pre_user_input",
        lambda **payload: calls.append(payload),
    )
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)

    assert cli._route_pre_user_input("follow up") == "follow up"
    assert calls == []


def test_cli_input_route_skips_unknown_goal_state(monkeypatch):
    cli = object.__new__(HermesCLI)
    cli.session_id = "cli-session"
    calls = []
    monkeypatch.setattr(HermesCLI, "_get_goal_manager", lambda _self: None)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.route_pre_user_input",
        lambda **payload: calls.append(payload),
    )
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)

    assert cli._route_pre_user_input("follow up") == "follow up"
    assert calls == []


def test_cli_only_routes_real_user_queue_entries():
    assert _is_real_user_input("typed text") is True
    assert _is_real_user_input(_VoiceInputMessage("spoken text")) is True
    assert _is_real_user_input(_SyntheticInputMessage("heartbeat prompt")) is False


def test_cli_input_route_skips_goal_lookup_without_subscriber(monkeypatch):
    cli = object.__new__(HermesCLI)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: False)

    def fail_goal_lookup(_self):
        raise AssertionError("goal lookup must be skipped")

    monkeypatch.setattr(HermesCLI, "_get_goal_manager", fail_goal_lookup)

    assert cli._route_pre_user_input("plain input") == "plain input"


def test_generated_slash_prompts_are_synthetic():
    cli = object.__new__(HermesCLI)
    cli._pending_input = queue.Queue()

    cli._handle_learn_command("/learn these notes")
    cli._handle_init_command("/init focus on commands")

    assert isinstance(cli._pending_input.get_nowait(), _SyntheticInputMessage)
    assert isinstance(cli._pending_input.get_nowait(), _SyntheticInputMessage)


def test_browser_system_notes_are_synthetic(monkeypatch):
    cli = object.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(
        "hermes_cli.cli_commands_mixin.discover_local_cdp_url",
        lambda _port, timeout: "http://127.0.0.1:9222",
    )
    monkeypatch.setattr("tools.browser_tool.cleanup_all_browsers", lambda: None)
    monkeypatch.setattr("tools.browser_tool._ensure_cdp_supervisor", lambda _name: None)
    monkeypatch.setattr("tools.browser_tool._stop_cdp_supervisor", lambda _name: None)

    cli._handle_browser_command("/browser connect")
    cli._handle_browser_command("/browser disconnect")

    assert isinstance(cli._pending_input.get_nowait(), _SyntheticInputMessage)
    assert isinstance(cli._pending_input.get_nowait(), _SyntheticInputMessage)
