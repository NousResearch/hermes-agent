"""First-run setup continues into chat only when a provider is configured."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def first_run(tmp_path, monkeypatch):
    home = tmp_path / "state"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TUI", raising=False)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    import hermes_cli.main as main_mod
    import hermes_cli.setup as setup_mod

    monkeypatch.setattr(setup_mod, "is_interactive_stdin", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    monkeypatch.setattr(main_mod, "cmd_setup", Mock(return_value=None))
    return main_mod


@pytest.fixture
def classic_chat(first_run, monkeypatch):
    import cli as cli_mod
    from hermes_cli._parser import build_top_level_parser
    import hermes_cli.observability.shared_metrics_process as metrics

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=first_run.cmd_chat)
    args = parser.parse_args(["chat", "--cli"])
    chat = Mock()
    monkeypatch.setattr(cli_mod, "main", chat)
    monkeypatch.setattr(first_run, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(first_run, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr(first_run, "_confirm_startup_expensive_model_override", lambda _args: None)
    monkeypatch.setattr(metrics, "begin_process", lambda _surface: None)
    return args, chat


def test_classic_chat_continues_after_setup(first_run, classic_chat, monkeypatch, capsys):
    args, chat = classic_chat
    monkeypatch.setattr(first_run, "_has_any_provider_configured", Mock(side_effect=[False, True]))

    first_run.cmd_chat(args)

    first_run.cmd_setup.assert_called_once_with(args)
    chat.assert_called_once()
    assert "Starting Hermes..." in capsys.readouterr().out


def test_classic_chat_stops_after_cancelled_setup(first_run, classic_chat, monkeypatch, capsys):
    args, chat = classic_chat
    monkeypatch.setattr(first_run, "_has_any_provider_configured", Mock(return_value=False))

    first_run.cmd_chat(args)

    first_run.cmd_setup.assert_called_once_with(args)
    chat.assert_not_called()
    assert "Starting Hermes..." not in capsys.readouterr().out


def test_gateway_launch_continues_after_setup(first_run, monkeypatch):
    import hermes_cli.free_tier_bootstrap as bootstrap
    from hermes_cli.gateway_chat_startup import ensure_launch_provider

    monkeypatch.setattr(bootstrap, "run_bootstrap", lambda announce=False: None)
    monkeypatch.setattr(first_run, "_has_any_provider_configured", Mock(side_effect=[False, True]))
    args = SimpleNamespace()

    assert ensure_launch_provider(args) is True
    first_run.cmd_setup.assert_called_once_with(args)


def test_declined_setup_exits_one(first_run, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    with pytest.raises(SystemExit) as exc:
        first_run._first_run_setup_guard(SimpleNamespace())

    assert exc.value.code == 1
    first_run.cmd_setup.assert_not_called()
