"""Behaviour contracts for ``hermes connectors``."""

from __future__ import annotations

import argparse

import pytest

from hermes_cli.connectors_cmd import cmd_connectors
from hermes_cli.subcommands.connectors import build_connectors_parser


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    build_connectors_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(list(argv))


@pytest.fixture
def portal(monkeypatch):
    calls: list[tuple[str, str]] = []

    class FakePortal:
        def list_accounts(self):
            return [
                {"connectionId": "conn-work", "connector": "gmail", "status": "active", "label": "a@x.test",
                 "alias": "work", "active": True},
                {"connectionId": "conn-home", "connector": "gmail", "status": "active", "label": "b@x.test",
                 "alias": "home", "active": True},
            ]

        def rename_account(self, connection_id, alias):
            calls.append((connection_id, alias))
            return {"connectionId": connection_id, "connector": "gmail", "alias": alias, "label": "a@x.test"}

    monkeypatch.setattr("tools.connectors.portal.client.PortalConnectorClient", FakePortal)
    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True, raising=False)
    return calls


def test_rename_parses_app_name_and_new_name():
    args = _parse("connectors", "rename", "gmail", "work", "--to", "home")

    assert (args.connectors_action, args.app, args.name, args.to) == ("rename", "gmail", "work", "home")
    assert args.func is cmd_connectors


def test_rename_targets_the_named_account(portal, capsys):
    rc = cmd_connectors(_parse("connectors", "rename", "gmail", "work", "--to", "office"))

    assert rc == 0
    assert portal == [("conn-work", "office")]
    assert "office" in capsys.readouterr().out


def test_rename_to_an_invalid_name_calls_nothing(portal, capsys):
    rc = cmd_connectors(_parse("connectors", "rename", "gmail", "work", "--to", "Bad Name"))

    assert rc == 1
    assert portal == []
    assert "lowercase" in capsys.readouterr().err
