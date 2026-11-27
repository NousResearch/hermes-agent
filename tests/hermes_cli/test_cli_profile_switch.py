"""Behavior contracts for classic-CLI ``/profile`` switching."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.i18n import t
from cli import HermesCLI


def _make_cli() -> HermesCLI:
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.session_id = "source-session"
    cli._pending_relaunch = None
    cli._pending_relaunch_env = None
    cli._pending_resume_sessions = None
    cli._slash_metrics_surface = None
    cli._agent_running = False
    cli._detach_on_exit = False
    cli._should_exit = False
    cli._app = None
    return cli


@pytest.fixture
def profile_catalog(monkeypatch, tmp_path):
    """A default profile plus two named profiles, without touching the real profile tree."""
    homes = {
        "default": tmp_path / ".hermes",
        "work": tmp_path / ".hermes" / "profiles" / "work",
        "research": tmp_path / ".hermes" / "profiles" / "research",
    }
    display_names = {"default": "", "work": "Builder", "research": ""}

    monkeypatch.setattr("hermes_cli.profiles.list_profile_names", lambda: list(homes))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])
    monkeypatch.setattr(
        "hermes_cli.profiles.read_profile_meta",
        lambda path: {
            "display_name": next(display_names[name] for name, home in homes.items() if home == path)
        },
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        "hermes_cli.slash_exec.execute_command",
        lambda *_args, **_kwargs: SimpleNamespace(
            data={"profile": "default", "home": str(homes["default"])}
        ),
    )
    return homes


def test_process_command_passes_the_requested_profile_to_the_handler(monkeypatch):
    cli = _make_cli()
    handler = MagicMock(return_value=False)
    cli._handle_profile_command = handler
    monkeypatch.setattr("hermes_cli.gateway_cli_commands.run", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("hermes_cli.gateway_plugin_commands.observe", lambda *_args, **_kwargs: True)

    assert cli.process_command("/profile work") is False
    handler.assert_called_once_with("/profile work")


def test_profile_registry_dispatches_while_busy():
    from hermes_cli.commands import resolve_command

    command = resolve_command("profile")

    assert command is not None
    assert command.busy_policy == "dispatch"


def test_bare_profile_reports_home_and_other_profile_hint(profile_catalog, capsys):
    cli = _make_cli()

    assert cli._handle_profile_command("/profile") is not False

    out = capsys.readouterr().out
    assert t("cli.commands.profile.profile", profile="default") in out
    assert t("cli.commands.profile.home", home=str(profile_catalog["default"])) in out
    assert t(
        "cli.commands.profile.switch_hint",
        profiles="Builder (work), research",
    ) in out
    assert cli._pending_relaunch is None


def test_bare_profile_has_no_switch_hint_when_no_other_profile(monkeypatch, profile_catalog, capsys):
    monkeypatch.setattr("hermes_cli.profiles.list_profile_names", lambda: ["default"])
    cli = _make_cli()

    cli._handle_profile_command("/profile")

    assert "switch_hint" not in capsys.readouterr().out
    assert cli._pending_relaunch is None


@pytest.mark.parametrize("command", ["/profile missing", "/profile work extra"])
def test_invalid_or_extra_arguments_list_profiles_without_relaunch(
    command, profile_catalog, capsys
):
    cli = _make_cli()

    assert cli._handle_profile_command(command) is not False

    out = capsys.readouterr().out
    assert t(
        "cli.commands.profile.invalid",
        request=command.partition(" ")[2],
        profiles="default, Builder (work), research",
    ) in out
    assert cli._pending_relaunch is None


def test_current_profile_reports_already_on_without_relaunch(profile_catalog, capsys):
    cli = _make_cli()

    assert cli._handle_profile_command("/profile DEFAULT") is not False

    assert t("cli.commands.profile.already_on", profile="default") in capsys.readouterr().out
    assert cli._pending_relaunch is None


def test_other_profile_schedules_fresh_chat_relaunch(profile_catalog, capsys):
    cli = _make_cli()

    assert cli._handle_profile_command("/profile WORK") is False

    assert t("cli.commands.profile.switching", profile="Builder (work)") in capsys.readouterr().out
    assert cli._pending_relaunch == ["-p", "work", "chat"]


def test_installation_wide_bot_mode_reopens_bot_chat_for_an_unmanaged_target(
    profile_catalog, capsys
):
    managed = profile_catalog["work"]
    managed.mkdir(parents=True)
    (managed / "profile.yaml").write_text(
        "ui_meta:\n  hermes-bots:\n    title: Builder\n",
        encoding="utf-8",
    )
    target = profile_catalog["research"]
    target.mkdir(parents=True)
    (target / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    cli = _make_cli()

    assert cli._handle_profile_command("/profile research") is False

    assert cli._pending_relaunch == [
        "-p",
        "research",
        "chat",
        "-c",
        "Bot Chat",
        "--create-if-missing",
    ]
    assert t("cli.commands.profile.switching", profile="research") in capsys.readouterr().out


def test_switch_scopes_source_context_cleanup_to_relaunch_env(monkeypatch, profile_catalog):
    source_context = {
        "HERMES_TUI_GATEWAY_URL": "ws://source.invalid",
        "HERMES_SESSION_ID": "source-session",
        "HERMES_SESSION_PROFILE": "default",
        "HERMES_GATEWAY_SESSION": "1",
        "HERMES_SINGLE_QUERY_SESSION": "1",
        "HERMES_PROFILE": "default",
        "HERMES_PROFILE_NAME": "default",
        "HERMES_TUI_CLIENT_ENV": "source-client",
    }
    for name, value in source_context.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    cli = _make_cli()

    cli._handle_profile_command("/profile work")

    assert all(os.environ[name] == value for name, value in source_context.items())
    assert os.environ["HERMES_SAFE_MODE"] == "1"
    assert cli._pending_relaunch_env is not None
    assert all(name not in cli._pending_relaunch_env for name in source_context)
    assert cli._pending_relaunch_env["HERMES_SAFE_MODE"] == "1"
    assert cli._pending_relaunch_env["HERMES_HOME"] == str(profile_catalog["work"])


def test_busy_profile_switch_detaches_the_turn_and_exits_the_tui(
    monkeypatch, profile_catalog
):
    from hermes_cli.gateway_agent import GatewayAgent

    cli = HermesCLI()
    cli._slash_metrics_surface = None
    cli._agent_running = True
    cli.agent = GatewayAgent(connection=MagicMock(), session_id="source-session")
    cli._session_db = MagicMock()
    buffer = SimpleNamespace(reset=MagicMock())
    app = SimpleNamespace(
        current_buffer=buffer,
        invalidate=MagicMock(),
        exit=MagicMock(),
        is_running=True,
    )
    cli._app = app
    event = SimpleNamespace(app=app)
    monkeypatch.setattr("hermes_cli.gateway_setting_commands.run", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("hermes_cli.gateway_cli_commands.run", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("hermes_cli.gateway_plugin_commands.observe", lambda *_args, **_kwargs: True)

    assert cli._tui_enter_inline_command(event, "/profile work", False) is True

    assert cli._pending_relaunch == ["-p", "work", "chat"]
    assert cli._detach_on_exit is True
    assert cli._should_exit is True
    app.exit.assert_called_once_with()
    buffer.reset.assert_called_once_with(append_to_history=True)
    app.invalidate.assert_called_once_with()

    with patch("cli._run_cleanup"), patch.object(cli, "_print_exit_summary"), \
            patch.object(cli, "_release_active_session"):
        cli._tui_shutdown()

    calls = [call.args[0] for call in cli.agent.connection.call.call_args_list]
    assert "session.interrupt" not in calls
    cli._session_db.end_session.assert_called_once_with("source-session", "cli_close")
