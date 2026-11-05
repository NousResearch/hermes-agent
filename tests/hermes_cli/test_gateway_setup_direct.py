"""Targeted gateway setup persists credentials only after successful pairing."""

import argparse
import io
import sys
from pathlib import Path
from unittest.mock import Mock

import hermes_yaml as yaml
import pytest
from dotenv import dotenv_values

from hermes_cli import gateway as gw
from hermes_cli import gateway_setup_direct as direct
from hermes_cli import gateway_setup_wizard as wizard
from hermes_cli.subcommands.gateway import build_gateway_parser
from hermes_cli.telegram_managed_bot import TelegramBotSetupResult, TelegramPairing

TOKEN = "123456:" + "synthetic_test_token_" * 2
LINK = "https://t.me/example_setup_bot?start=synthetic"


def _parse(*arguments):
    parser = argparse.ArgumentParser(prog="hermes")
    build_gateway_parser(parser.add_subparsers(dest="command"), cmd_gateway=gw.gateway_command,
                         cmd_proxy=Mock(), cmd_gateway_enroll=Mock())
    return parser.parse_args(["gateway", "setup", *arguments])


def _run(*arguments):
    args = _parse(*arguments)
    with pytest.raises(SystemExit) as exited:
        gw._cmd_setup(args)
    return exited.value.code


@pytest.fixture
def setup_env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_HOME_CHANNEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(gw, "is_managed", lambda: False)
    monkeypatch.setattr(gw, "_is_service_running", Mock(return_value=False))
    monkeypatch.setattr(gw, "_setup_service_action", Mock(side_effect=AssertionError("service action")))
    monkeypatch.setattr(gw, "_wizard_post_setup", Mock(side_effect=AssertionError("service offer")))
    monkeypatch.setattr("gateway.relay.link.relay_channel_view", lambda: (frozenset(), False))
    monkeypatch.setattr(direct, "create_pairing", Mock(return_value=TelegramPairing(
        "synthetic-pairing", "synthetic-poll", "example_bot", LINK, LINK)))
    monkeypatch.setattr(direct, "poll_for_setup_result", Mock(return_value=TelegramBotSetupResult(
        TOKEN, "example_bot", 123)))
    monkeypatch.setattr(direct, "print_qr_code", Mock())
    return home


def test_quick_success(setup_env, capsys):
    assert _run("telegram", "--quick", "--timeout", "12") == 0
    values = dotenv_values(setup_env / ".env")
    assert values["TELEGRAM_BOT_TOKEN"] == TOKEN
    assert values["TELEGRAM_ALLOWED_USERS"] == "123"
    assert values["TELEGRAM_HOME_CHANNEL"] == "123"
    assert yaml.safe_load((setup_env / "config.yaml").read_text())["platforms"]["telegram"]["enabled"] is True
    output = capsys.readouterr()
    assert f"Link: {LINK}" in output.out
    assert output.out.splitlines()[-1] == "@example_bot connected. Next: hermes gateway ensure"
    assert TOKEN not in output.out + output.err
    direct.poll_for_setup_result.assert_called_once_with(None, direct.create_pairing.return_value, 12, on_tick=None)
    direct.print_qr_code.assert_not_called()
    gw._setup_service_action.assert_not_called()


def test_existing_token_requires_replace(setup_env, capsys):
    before = "TELEGRAM_BOT_TOKEN=old-synthetic-token\n"
    (setup_env / ".env").write_text(before)
    assert _run("telegram", "--quick") == 1
    assert (setup_env / ".env").read_text() == before
    direct.create_pairing.assert_not_called()
    assert "--replace" in capsys.readouterr().out
    assert _run("telegram", "--quick", "--replace") == 0
    assert dotenv_values(setup_env / ".env")["TELEGRAM_BOT_TOKEN"] == TOKEN


@pytest.mark.parametrize("owner, extra, expected", [
    (None, "", None), (None, "111,222", "111,222"), (123, "111,222,123,111", "123,111,222"),
])
def test_allowlist_owner_first(setup_env, capsys, owner, extra, expected):
    direct.poll_for_setup_result.return_value = TelegramBotSetupResult(TOKEN, owner_user_id=owner)
    flags = ["--allow", extra] if extra else []
    assert _run("telegram", "--quick", *flags) == 0
    values = dotenv_values(setup_env / ".env")
    assert values["TELEGRAM_BOT_TOKEN"] == TOKEN
    assert values.get("TELEGRAM_ALLOWED_USERS") == expected
    output = capsys.readouterr().out
    assert "Telegram bot connected." in output
    if expected is None:
        assert "denies everyone" in output and "hermes config set TELEGRAM_ALLOWED_USERS" in output
        assert "TELEGRAM_HOME_CHANNEL" not in values


def test_no_owner_keeps_an_existing_allowlist(setup_env, capsys):
    (setup_env / ".env").write_text("TELEGRAM_BOT_TOKEN=old-synthetic-token\nTELEGRAM_ALLOWED_USERS=555\n")
    direct.poll_for_setup_result.return_value = TelegramBotSetupResult(TOKEN)
    assert _run("telegram", "--quick", "--replace") == 0
    assert dotenv_values(setup_env / ".env")["TELEGRAM_ALLOWED_USERS"] == "555"
    output = capsys.readouterr().out
    assert "existing allowlist stays in force" in output and "denies everyone" not in output


@pytest.mark.parametrize("pinned", ["TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS"])
def test_pinned_key_refuses_before_pairing(setup_env, monkeypatch, capsys, pinned):
    from hermes_cli import config

    monkeypatch.setattr(config.managed_scope, "is_env_managed", lambda key: key == pinned)
    assert _run("telegram", "--quick") == 1
    direct.create_pairing.assert_not_called()
    assert f"Cannot set {pinned}" in capsys.readouterr().out
    assert not (setup_env / ".env").exists()
    assert not (setup_env / "config.yaml").exists()


@pytest.mark.parametrize("ids", ["abc", "111,abc", "111,", "١١١"])
def test_invalid_allow_fails_before_network(setup_env, ids):
    assert _run("telegram", "--quick", "--allow", ids) == 2
    direct.create_pairing.assert_not_called()
    assert not (setup_env / ".env").exists()


@pytest.mark.parametrize("failure", ["create", "timeout", "invalid_token"])
def test_failed_pairing_writes_nothing(setup_env, capsys, failure):
    if failure == "create":
        direct.create_pairing.return_value = None
    elif failure == "timeout":
        direct.poll_for_setup_result.return_value = None
    else:
        direct.poll_for_setup_result.return_value = TelegramBotSetupResult("invalid-synthetic-token")
    assert _run("telegram", "--quick") == 1
    assert not (setup_env / ".env").exists()
    assert not (setup_env / "config.yaml").exists()
    output = capsys.readouterr()
    assert "invalid-synthetic-token" not in output.out + output.err
    expected = {"create": "Could not reach", "timeout": "Timed out", "invalid_token": "invalid Telegram token"}
    assert expected[failure] in output.out


def test_team_gateway_refuses(setup_env, monkeypatch, capsys):
    monkeypatch.setattr("gateway.relay.link.relay_channel_view", lambda: (frozenset({"telegram"}), True))
    assert _run("telegram", "--quick") == 1
    direct.create_pairing.assert_not_called()
    assert "Team Gateway" in capsys.readouterr().out


@pytest.mark.parametrize("arguments, message", [
    (("telegram",), "--quick"), (("unknown-platform",), "Valid keys:"),
    (("discord", "--quick"), "Telegram only"),
])
def test_invalid_invocation_guidance(setup_env, capsys, arguments, message):
    assert _run(*arguments) == 2
    output = capsys.readouterr().out
    assert message in output
    if arguments == ("unknown-platform",):
        assert "telegram" in output and "mattermost" in output
    direct.create_pairing.assert_not_called()


def test_no_platform_keeps_wizard(setup_env, monkeypatch):
    args = _parse()
    assert args.platform is None
    setup = Mock(wraps=wizard.gateway_setup)
    monkeypatch.setattr(gw, "gateway_setup", setup)
    entered = Mock()
    monkeypatch.setattr(wizard, "_wizard_service_status_block", Mock())
    monkeypatch.setattr(wizard, "_wizard_platform_loop", entered)
    monkeypatch.setattr(gw, "_all_platforms", list)
    gw._cmd_setup(args)
    setup.assert_called_once_with(args)
    entered.assert_called_once_with()
    gw.gateway_setup()  # Existing no-argument callers remain valid.
    assert entered.call_count == 2


def test_managed_refuses_before_delegation(setup_env, monkeypatch):
    monkeypatch.setattr(gw, "is_managed", lambda: True)
    refused = Mock()
    monkeypatch.setattr(gw, "managed_error", refused)
    run = Mock()
    monkeypatch.setattr(direct, "run_platform_setup", run)
    gw._cmd_setup(_parse("telegram", "--quick"))
    refused.assert_called_once_with("run gateway setup")
    run.assert_not_called()


def test_tty_uses_existing_setup_and_post_step(setup_env, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    configure, post = Mock(), Mock()
    monkeypatch.setattr(gw, "_configure_platform", configure)
    monkeypatch.setattr(gw, "_wizard_post_setup", post)
    assert _run("telegram") == 0
    assert configure.call_args.args[0]["key"] == "telegram"
    post.assert_called_once_with()
    direct.create_pairing.assert_not_called()


def test_running_gateway_prints_restart_only(setup_env, monkeypatch, capsys):
    monkeypatch.setattr(gw, "_is_service_running", lambda: True)
    (setup_env / ".env").write_text("TELEGRAM_HOME_CHANNEL=456\n")
    assert _run("telegram", "--quick") == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith("Next: hermes gateway restart")
    assert dotenv_values(setup_env / ".env")["TELEGRAM_HOME_CHANNEL"] == "456"
    gw._setup_service_action.assert_not_called()
    gw._wizard_post_setup.assert_not_called()


def test_explicit_work_home(setup_env, tmp_path, monkeypatch):
    from hermes_cli import main

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    work = tmp_path / ".hermes" / "profiles" / "work"
    work.mkdir(parents=True)
    (work / "config.yaml").write_text("{}\n")
    monkeypatch.setattr(sys, "argv", ["hermes", "-p", "work", "gateway", "setup", "telegram", "--quick"])
    monkeypatch.setattr(main, "_explicit_cli_profile", None)
    # -p resolves profiles under the default root, so drop the fixture's HERMES_HOME (as
    # test_apply_profile_override does) and let the flag set it.
    monkeypatch.delenv("HERMES_HOME", raising=False)
    main._apply_profile_override()
    with pytest.raises(SystemExit) as exited:
        gw._cmd_setup(_parse(*sys.argv[3:]))
    assert exited.value.code == 0
    assert dotenv_values(work / ".env")["TELEGRAM_BOT_TOKEN"] == TOKEN
    assert not (setup_env / ".env").exists()


def test_tty_qr_and_link_before_poll(setup_env, monkeypatch, capsys):
    stdout = sys.stdout
    monkeypatch.setattr(stdout, "isatty", lambda: True)
    flushed = Mock(wraps=stdout.flush)
    monkeypatch.setattr(stdout, "flush", flushed)

    def poll(*args, **kwargs):
        assert f"Link: {LINK}" in capsys.readouterr().out
        assert flushed.called
        return TelegramBotSetupResult(TOKEN)

    direct.poll_for_setup_result.side_effect = poll
    assert _run("telegram", "--quick") == 0
    direct.print_qr_code.assert_called_once_with(LINK, include_link=False)
