"""The `hermes profile purge-identity` retry path: dispatched, and honest about failure.

`hermes profile delete` reports a pending identity settlement when it cannot purge and names this
command as the retry — so the command must actually reach a handler, and must fail loudly when the
settlement still cannot be made. Regression for the delete side of #111926.
"""
import argparse
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import profile_cmd, profiles


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def test_every_profile_subcommand_has_a_dispatch_entry():
    """A subcommand that parses but has no entry in the dispatch table silently does nothing."""
    from hermes_cli.subcommands.profile import build_profile_parser
    top = argparse.ArgumentParser()
    subparsers = top.add_subparsers(dest="command")
    build_profile_parser(subparsers, cmd_profile=lambda args: None)
    profile_parser = subparsers.choices["profile"]
    groups = [a for a in profile_parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert groups, "the profile parser exposes subcommands"
    assert set(groups[0].choices) == set(profile_cmd.PROFILE_ACTIONS) - {None}


def test_purge_identity_reports_success(profile_env, monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.profile_identity.purge_profile_identity", lambda name: True)

    profile_cmd.cmd_profile(Namespace(profile_action="purge-identity", profile_name="gone"))

    assert "identity purged: gone" in capsys.readouterr().out


def test_purge_identity_exits_nonzero_when_settlement_stays_pending(
        profile_env, monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.profile_identity.purge_profile_identity", lambda name: False)

    with pytest.raises(SystemExit) as exc:
        profile_cmd.cmd_profile(Namespace(profile_action="purge-identity", profile_name="gone"))

    assert exc.value.code not in (0, None)
    assert "hermes profile purge-identity gone" in capsys.readouterr().err


def test_purge_identity_refuses_a_same_name_profile_created_after_the_delete(profile_env, capsys):
    """The retry must not purge identity out from under a profile that exists again.

    The purge keys off the name alone, so the recovery flow — ``delete foo`` (settlement pending),
    ``create foo``, ``purge-identity foo`` — would delete the NEW incarnation's routing/heartbeat
    identity. The delete path tombstones the directory before it purges, so this refusal cannot
    block the delete it belongs to (``TestDeleteProfile`` covers that path).
    """
    import json

    from hermes_cli.profiles import create_profile
    from hermes_state import SessionDB

    create_profile("gone", no_alias=True)
    scope = str(profile_env / ".hermes" / "sessions")
    db = SessionDB(profile_env / ".hermes" / "state.db")
    db.save_gateway_routing_entry(
        "agent:gone:feishu:dm:chatA",
        json.dumps({"session_key": "agent:gone:feishu:dm:chatA"}), scope=scope)
    db.close()

    with pytest.raises(SystemExit) as exc:
        profile_cmd.cmd_profile(Namespace(profile_action="purge-identity", profile_name="gone"))

    assert exc.value.code not in (0, None)
    assert "purge-identity only settles the identity of a delete" in capsys.readouterr().out
    check = SessionDB(profile_env / ".hermes" / "state.db")
    try:
        # The recreated profile still owns its routing key.
        assert set(check.load_gateway_routing_entries(scope=scope)) == {
            "agent:gone:feishu:dm:chatA"}
    finally:
        check.close()
