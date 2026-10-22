"""Actual parser, classic CLI dispatch and owner-private gateway command fixtures."""
import argparse

import pytest

from gateway.connected_assistance_store import AssistanceStore
from tests.gateway.test_connected_assistance_store import finding, scan


def parser():
    from hermes_cli.subcommands.connectors import build_connectors_parser
    root = argparse.ArgumentParser()
    build_connectors_parser(root.add_subparsers())
    return root


@pytest.mark.parametrize(("raw", "expected"), [
    ("", "read"), ("list --status all --limit 2", "read"), ("show item", "read"),
    ("help", "read"), ("--help", "read"), ("pause", "control"), ("resume", "control"),
    ("dismiss item", "control"), ("resolve item", "control"),
    ("erase", "refused"), ("pause extra", "refused"),
])
def test_native_findings_admission_preserves_read_control_boundary(raw, expected):
    from gateway.session_commands import command_verdict

    assert command_verdict("findings", raw) == expected


def test_connectors_findings_controls_work_after_disconnect(tmp_path, monkeypatch, capsys):
    from tools import connectors

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(connectors, "connectors_available", lambda: False)
    row = AssistanceStore(tmp_path).ingest_scan(scan(finding()))[0]
    for tokens, expected in [([], row["id"]), (["show", row["id"]], "Evidence (source data"),
                             (["dismiss", row["id"]], "dismissed"), (["pause"], "paused"),
                             (["resume"], "resumed")]:
        args = parser().parse_args(["connectors", "findings", *tokens])
        assert args.func(args) == 0
        output = capsys.readouterr().out
        assert expected in output
        print("CLI FIXTURE:", output)


def test_connection_flow_discloses_background_reads_and_costs_before_auth(monkeypatch, capsys):
    from hermes_cli import connectors_cmd
    from tools.connectors import account
    from types import SimpleNamespace

    def unavailable(*args, **kwargs):
        raise RuntimeError("fixture stopped before authorization")

    monkeypatch.setattr(account, "find_or_start_operation", unavailable)
    with pytest.raises(RuntimeError, match="fixture stopped"):
        connectors_cmd._connect(SimpleNamespace(app="gmail", reconnect=False, alias=None))
    output = capsys.readouterr().out
    assert "background" in output and "model" in output and "cost" in output
    assert "hermes connectors findings pause" in output


def test_classic_cli_findings_uses_registered_shared_dispatch(tmp_path, monkeypatch, capsys):
    from cli import HermesCLI
    from hermes_cli.commands import resolve_command

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert resolve_command("findings").name == "findings"
    cli = object.__new__(HermesCLI)
    method, pass_arg = cli._slash_handler("findings")
    assert pass_arg
    getattr(cli, method)("/findings pause")
    assert "Connected assistance paused." in capsys.readouterr().out
    assert AssistanceStore(tmp_path).is_paused()


@pytest.mark.asyncio
async def test_gateway_private_handler_uses_native_receipt_and_shared_controls(tmp_path, monkeypatch):
    from types import MethodType, SimpleNamespace
    from gateway.slash_commands_findings import GatewayFindingsCommandsMixin
    from tests.gateway.test_connected_assistance_delivery import local_owner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, authority, connection, _ref, live = await local_owner(tmp_path, monkeypatch)
    runner._handle_findings_command = MethodType(GatewayFindingsCommandsMixin._handle_findings_command, runner)
    try:
        event = SimpleNamespace(source=live.source, text="/findings pause")
        assert "paused" in await runner._handle_findings_command(event)
        assert AssistanceStore(tmp_path).is_paused()
        event.text = "/findings resume"
        assert "resumed" in await runner._handle_findings_command(event)
        assert not AssistanceStore(tmp_path).is_paused()
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["group", "channel", "dm"])
async def test_messaging_cannot_read_or_mutate_private_findings(tmp_path, monkeypatch, chat_type):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.slash_commands import GatewaySlashCommandsMixin
    from types import SimpleNamespace

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = AssistanceStore(tmp_path)
    store.ingest_scan(scan(finding(title="Private account secret")))
    handler = object.__new__(GatewaySlashCommandsMixin)
    source = SessionSource(platform=Platform.SLACK, chat_id="public", user_id="any", chat_type=chat_type)
    event = SimpleNamespace(source=source, text="/findings pause")
    output = await handler._handle_findings_command(event)
    assert "hermes connectors findings" in output
    assert "Private account secret" not in output
    assert not store.is_paused()
