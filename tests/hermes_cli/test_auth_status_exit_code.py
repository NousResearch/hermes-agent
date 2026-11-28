"""``hermes auth status`` exit-code + ``--json`` contract (issue #158317).

A logged-out provider must not exit 0: scripts and adapters gate readiness on this
command's status. The human rendering stays byte-identical; ``--json`` adds a
machine-readable object and follows the same exit-code rule.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from hermes_cli import auth_commands
from hermes_cli.subcommands.auth import build_auth_parser

PROVIDER = "acme-demo"


@pytest.fixture()
def status_seams(monkeypatch):
    """Install a status payload and stub the pool/heal seams ``auth_status_command`` touches."""
    monkeypatch.setattr(auth_commands.auth_mod, "SINGLE_USE_REFRESH_POOL_PROVIDERS", frozenset())
    monkeypatch.setattr(auth_commands.auth_mod, "consume_oauth_heal_notices", list)

    def _install(payload: dict) -> dict:
        monkeypatch.setattr(auth_commands.auth_mod, "get_auth_status", lambda *a, **k: payload)
        return payload

    return _install


def _run(**kwargs):
    return auth_commands.auth_status_command(SimpleNamespace(provider=PROVIDER, **kwargs))


def test_human_rendering_is_unchanged_and_logged_out_exits_1(status_seams, capsys):
    status_seams({"logged_in": False})
    with pytest.raises(SystemExit) as exc:
        _run()
    assert exc.value.code == 1
    assert capsys.readouterr().out == f"{PROVIDER}: logged out\n"


def test_human_rendering_keeps_the_reason_and_exits_1(status_seams, capsys):
    status_seams({"logged_in": False, "error": "refresh token rejected"})
    with pytest.raises(SystemExit) as exc:
        _run()
    assert exc.value.code == 1
    assert capsys.readouterr().out == f"{PROVIDER}: logged out (refresh token rejected)\n"


def test_logged_in_still_exits_0(status_seams, capsys):
    status_seams({"logged_in": True, "auth_type": "api_key"})
    _run()  # no SystemExit
    assert capsys.readouterr().out == f"{PROVIDER}: logged in\n  auth_type: api_key\n"


def test_json_logged_out_reports_and_exits_1(status_seams, capsys):
    status_seams({"logged_in": False})
    with pytest.raises(SystemExit) as exc:
        _run(json=True)
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "provider": PROVIDER, "logged_in": False, "auth_type": None, "free_tier": False}


def test_json_logged_out_carries_the_reason(status_seams, capsys):
    status_seams({"logged_in": False, "error": "no api key configured"})
    with pytest.raises(SystemExit) as exc:
        _run(json=True)
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["error"] == "no api key configured"


def test_json_logged_in_exits_0(status_seams, capsys):
    status_seams({"logged_in": True, "auth_type": "oauth", "scope": "inference"})
    _run(json=True)  # no SystemExit
    assert json.loads(capsys.readouterr().out) == {
        "provider": PROVIDER, "logged_in": True, "auth_type": "oauth", "free_tier": False}


def test_json_free_tier_counts_as_signed_in(status_seams, capsys):
    """A free-tier guest serves requests with no account login: readiness is true, flagged free_tier."""
    status_seams({"logged_in": False, "free_tier": True})
    _run(json=True)  # no SystemExit
    assert json.loads(capsys.readouterr().out) == {
        "provider": PROVIDER, "logged_in": True, "auth_type": None, "free_tier": True}


def test_json_mode_prints_nothing_but_the_object(status_seams, capsys):
    status_seams({"logged_in": True, "auth_type": "api_key", "client_id": "abc"})
    _run(json=True)
    assert capsys.readouterr().out.strip().startswith("{")


def test_parser_accepts_json_flag_and_defaults_it_off():
    parser = argparse.ArgumentParser()
    build_auth_parser(parser.add_subparsers(), cmd_auth=lambda args: None)
    assert parser.parse_args(["auth", "status", "nous", "--json"]).json is True
    assert parser.parse_args(["auth", "status", "nous"]).json is False
