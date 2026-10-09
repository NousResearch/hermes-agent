"""Refresh status must disclose model benches without resetting them (#135873)."""
import argparse
import time
from types import SimpleNamespace

import pytest

from agent.credential_pool import PooledCredential
from hermes_cli import auth_commands
from hermes_cli.subcommands.auth import build_auth_parser


def test_adopted_peer_tokens_report_model_benches_without_mutation(monkeypatch, capsys):
    entry = PooledCredential.from_dict("anthropic", {
        "id": "fixture-id", "label": "fixture", "auth_type": "oauth",
        "source": "manual", "refresh_token": "fixture-refresh",
        "last_status": "exhausted",
        "model_cooldowns": {"active-model": time.time() + 3600, "expired-model": 1},
    })
    before = entry.to_dict()
    pool = SimpleNamespace(entries=lambda: [entry], try_refresh_matching=lambda **_: entry)
    monkeypatch.setattr(auth_commands, "load_pool", lambda _: pool)
    monkeypatch.setattr(auth_commands, "dispatch_plugin_auth", lambda *_: False)
    auth_commands.auth_refresh_command(SimpleNamespace(provider="anthropic", target=None))
    out = capsys.readouterr().out
    assert "Adopted current tokens" in out and "status still: exhausted" in out
    assert "active-model" in out and "expired-model" not in out
    assert "s remaining" in out and "hermes auth reset anthropic fixture-id" in out
    assert entry.to_dict() == before


def test_auth_help_explains_model_reset(capsys):
    parser = argparse.ArgumentParser()
    build_auth_parser(parser.add_subparsers(), cmd_auth=lambda _: None)
    with pytest.raises(SystemExit) as result:
        parser.parse_args(["auth", "--help"])
    assert result.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "model cooldowns require auth reset" in out
