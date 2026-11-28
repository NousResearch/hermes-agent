"""Machine-readable webhook listing exposes route grants without HMAC secrets."""

import argparse
import json

import pytest

from hermes_cli.subcommands.webhook import build_webhook_parser
from hermes_cli.webhook import webhook_command


@pytest.fixture
def webhook_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.webhook._is_webhook_enabled", lambda: True)
    return tmp_path


def _run_list():
    parser = argparse.ArgumentParser()
    build_webhook_parser(parser.add_subparsers(), cmd_webhook=webhook_command)
    args = parser.parse_args(["webhook", "list", "--json"])
    return args.func(args)


def test_json_list_shows_grants_without_secrets(webhook_home, capsys):
    routes = {
        "review": {"secret": "secret-canary", "toolsets": ["terminal"], "profile": "work"},
        "default": {"secret": "other-canary"},
        "restricted": {"toolsets": []},
    }
    (webhook_home / "webhook_subscriptions.json").write_text(json.dumps(routes), encoding="utf-8")

    _run_list()
    output = capsys.readouterr().out
    data = {route["name"]: route for route in json.loads(output)}

    assert data["review"]["toolsets"] == ["terminal"]
    assert data["review"]["profile"] == "work"
    assert "toolsets" not in data["default"]
    assert data["restricted"]["toolsets"] == []
    assert "secret" not in output and "canary" not in output


def test_json_list_of_no_routes_is_empty_array(webhook_home, capsys):
    _run_list()
    assert json.loads(capsys.readouterr().out) == []
