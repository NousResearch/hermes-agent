"""Drive gateway debt acknowledgment through the real update CLI parser."""

from __future__ import annotations

import argparse

import pytest

from hermes_cli import main, update_host_obligation as host
from hermes_cli.subcommands.update import build_update_parser


@pytest.fixture
def cli(monkeypatch, tmp_path):
    from hermes_cli import update_lock, update_receipt

    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("hermes_cli.update_owning_install.retarget_to_owning_install", lambda *_: None)
    monkeypatch.setattr(update_lock, "update_marker_path", lambda: tmp_path / "update-marker")
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: None)
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=main.cmd_update)

    def invoke(*argv):
        args = parser.parse_args(["update", *argv])
        return args.func(args)

    return invoke


def _protected(pid):
    return {"kind": "gateway", "profile": None, "pid": pid, "protected_unmapped": True,
            "incident_id": f"incident-{pid}", "home": None, "birth": None}


def test_cli_acknowledges_one_debt_without_starting_update(cli, capsys):
    rows = [_protected(11), _protected(12)]
    assert host.write_host_obligation(expected_sha="abc", runtimes=rows)
    token = dict(host.protected_acknowledgments())[11]

    with pytest.raises(SystemExit) as exc:
        cli("--acknowledge-protected-gateway", token)

    assert exc.value.code == 0
    assert host.read_host_obligation()["inventory"]["runtimes"] == rows[1:]
    assert "unrelated or unverified obligations remain" in capsys.readouterr().out


def test_cli_rejects_stale_token_without_mutation(cli):
    rows = [_protected(11), _protected(12)]
    assert host.write_host_obligation(expected_sha="abc", runtimes=rows)
    token = dict(host.protected_acknowledgments())[11]
    host.amend_host_obligation(expected_sha="new")

    with pytest.raises(SystemExit, match="token may be stale"):
        cli("--acknowledge-protected-gateway", token)

    assert host.read_host_obligation()["inventory"]["runtimes"] == rows


def test_cli_does_not_mutate_when_update_lock_is_held(cli, monkeypatch):
    from hermes_cli import update_lock

    rows = [_protected(11)]
    assert host.write_host_obligation(expected_sha="abc", runtimes=rows)
    token = dict(host.protected_acknowledgments())[11]

    class BusyLock:
        holder = None

        def __init__(self, **kwargs):
            pass

        def acquire(self):
            return False

    monkeypatch.setattr(update_lock, "UpdateLock", BusyLock)
    with pytest.raises(SystemExit):
        cli("--acknowledge-protected-gateway", token)
    assert host.read_host_obligation()["inventory"]["runtimes"] == rows


def test_cli_rejects_both_acknowledgments_without_mutation(cli):
    rows = [_protected(11)]
    assert host.write_host_obligation(expected_sha="abc", runtimes=rows)
    token = dict(host.protected_acknowledgments())[11]
    with pytest.raises(SystemExit, match="only one gateway debt"):
        cli("--acknowledge-protected-gateway", token, "--acknowledge-gateway-receipt", "receipt-token")
    assert host.read_host_obligation()["inventory"]["runtimes"] == rows


def test_cli_acknowledges_only_selected_receipt_entry(cli, monkeypatch):
    from hermes_cli import update_receipt

    first = {"pid": 11, "profile": "default", "state": "stale"}
    second = {"pid": 12, "profile": "work", "state": "stale"}
    receipt = {"update_id": "receipt-1", "post_update": {"sha": "abc"},
               "fleet": [first, second], "plan": {"runtimes": []}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(11)])
    protected_token = dict(host.protected_acknowledgments())[11]
    with pytest.raises(SystemExit) as exc:
        cli("--acknowledge-protected-gateway", protected_token)
    assert exc.value.code == 0
    token = next(token for source, row, token in host.receipt_gateway_acknowledgments(receipt)
                 if source == "fleet" and row == first)

    with pytest.raises(SystemExit) as exc:
        cli("--acknowledge-gateway-receipt", token)

    assert exc.value.code == 0
    assert host.receipt_without_acknowledged_gateways(receipt)["fleet"] == [second]


@pytest.mark.parametrize("metadata", [("--install-id",), ("--set-channel", "main")])
def test_cli_rejects_acknowledgment_with_metadata_modes(cli, metadata):
    rows = [_protected(11)]
    assert host.write_host_obligation(expected_sha="abc", runtimes=rows)
    token = dict(host.protected_acknowledgments())[11]

    with pytest.raises(SystemExit, match="cannot be combined with update metadata"):
        cli("--acknowledge-protected-gateway", token, *metadata)

    assert host.read_host_obligation()["inventory"]["runtimes"] == rows
