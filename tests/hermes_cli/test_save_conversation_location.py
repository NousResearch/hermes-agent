"""Tests for /save — the conversation snapshot slash command.

Regression: the old implementation wrote ``hermes_conversation_<ts>.json``
to the current working directory (CWD). Users who ran /save expected the
file to be discoverable via ``hermes sessions browse``, but CWD-resident
snapshots are not indexed in the state DB and are generally invisible.
The fix writes snapshots under ``~/.hermes/sessions/saved/`` and prints
the absolute path plus the resume hint for the live session.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Clear any cached hermes_home computation
    import hermes_constants
    if hasattr(hermes_constants, "_hermes_home_cache"):
        hermes_constants._hermes_home_cache = None
    return home


def _make_stub_cli(history):
    """Build a minimal object exposing just what save_conversation uses."""
    return SimpleNamespace(
        conversation_history=history,
        model="test-model",
        session_id="20260101_120000_abc123",
        session_start=datetime(2026, 1, 1, 12, 0, 0),
    )


def test_save_conversation_writes_under_hermes_home(hermes_home, tmp_path, monkeypatch, capsys):
    """Snapshot must land under ~/.hermes/sessions/saved/, not CWD."""
    # Change CWD to a different directory to prove the file does NOT go there.
    work = tmp_path / "somewhere-else"
    work.mkdir()
    monkeypatch.chdir(work)

    # Import fresh to pick up the HERMES_HOME fixture
    for mod in [m for m in sys.modules if m.startswith("cli") or m == "hermes_constants"]:
        sys.modules.pop(mod, None)

    import cli

    stub = _make_stub_cli([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])

    # Call the unbound method against our stub.
    cli.HermesCLI.save_conversation(stub, "/save json")

    # File must NOT be in CWD
    cwd_leak = list(work.glob("hermes_conversation_*.json"))
    assert not cwd_leak, f"snapshot leaked to CWD: {cwd_leak}"

    # File MUST be under ~/.hermes/sessions/saved/
    saved_dir = hermes_home / "sessions" / "saved"
    assert saved_dir.is_dir(), "expected saved/ subdirectory to be created"
    files = list(saved_dir.glob("hermes_conversation_*.json"))
    assert len(files) == 1, files

    payload = json.loads(files[0].read_text())
    assert payload["model"] == "test-model"
    # /save now emits the canonical export_session shape: the session id
    # lives under "id" (was "session_id" in the legacy snapshot format).
    assert payload["id"] == "20260101_120000_abc123"
    assert payload["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    # User-facing message must include the absolute path AND the resume hint.
    out = capsys.readouterr().out
    assert str(files[0]) in out, out
    assert "hermes --resume 20260101_120000_abc123" in out, out


def test_save_conversation_empty_history_does_nothing(hermes_home, capsys):
    for mod in [m for m in sys.modules if m.startswith("cli") or m == "hermes_constants"]:
        sys.modules.pop(mod, None)
    import cli

    stub = _make_stub_cli([])
    cli.HermesCLI.save_conversation(stub, "/save json")

    saved_dir = hermes_home / "sessions" / "saved"
    assert not saved_dir.exists() or not list(saved_dir.iterdir())
    out = capsys.readouterr().out
    assert "No conversation to save" in out




def test_save_conversation_bad_format_shows_usage(hermes_home, capsys):
    for mod in [m for m in sys.modules if m.startswith("cli") or m == "hermes_constants"]:
        sys.modules.pop(mod, None)
    import cli

    stub = _make_stub_cli([{"role": "user", "content": "hi"}])
    cli.HermesCLI.save_conversation(stub, "/save pdf")

    saved_dir = hermes_home / "sessions" / "saved"
    assert not saved_dir.exists() or not list(saved_dir.iterdir())
    out = capsys.readouterr().out
    assert "Usage:" in out


class _Unreachable:
    """A store connection whose owner never answers — what a gateway restart leaves behind."""

    final = False

    def call(self, *_args, **_kwargs):
        from hermes_cli.gateway_client import GatewayClientError
        raise GatewayClientError("Gateway disconnected; the owner did not answer.")


def _gateway_stub_cli(history, connection):
    """A CLI whose session store is the gateway's, as in a shared-gateway CLI chat."""
    from hermes_cli.gateway_session_store import GatewaySessionStore
    stub = _make_stub_cli(history)
    stub._session_db = GatewaySessionStore(connection=connection)
    return stub


def test_save_conversation_falls_back_to_local_rows_when_the_gateway_is_unreachable(hermes_home, capsys):
    """/save still writes a snapshot when the gateway cannot be reached.

    On a gateway session /save asks the owner for its canonical export. A gateway restart drops the
    CLI's connection, and the command then returned nothing at all — even though this CLI still holds
    the session's rows. It writes the snapshot from them instead, and says where it came from.
    """
    for mod in [m for m in sys.modules if m.startswith("cli") or m == "hermes_constants"]:
        sys.modules.pop(mod, None)
    import cli

    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    cli.HermesCLI.save_conversation(_gateway_stub_cli(history, _Unreachable()), "/save json")

    files = list((hermes_home / "sessions" / "saved").glob("hermes_conversation_*.json"))
    assert len(files) == 1, files
    assert json.loads(files[0].read_text())["messages"] == history
    out = capsys.readouterr().out
    assert "Gateway unreachable" in out, out
    assert str(files[0]) in out, out


def test_save_conversation_reports_a_gateway_refusal_without_writing_a_local_copy(hermes_home, capsys):
    """A reachable owner's refusal is an answer: no local snapshot is written behind its back."""
    for mod in [m for m in sys.modules if m.startswith("cli") or m == "hermes_constants"]:
        sys.modules.pop(mod, None)
    import cli
    from hermes_cli.gateway_client import GatewayRPCError

    class _Refusing:
        final = False

        def call(self, *_args, **_kwargs):
            raise GatewayRPCError("not_found")

    cli.HermesCLI.save_conversation(
        _gateway_stub_cli([{"role": "user", "content": "hi"}], _Refusing()), "/save json")

    saved_dir = hermes_home / "sessions" / "saved"
    assert not saved_dir.exists() or not list(saved_dir.iterdir())
    out = capsys.readouterr().out
    assert "No conversation to save" in out, out
