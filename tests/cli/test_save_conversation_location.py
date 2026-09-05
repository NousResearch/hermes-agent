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
import os
import stat
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


posix_only = pytest.mark.skipif(
    os.name != "posix", reason="POSIX permission bits are advisory on Windows"
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
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

    import cli  # noqa: F401  (module under test)

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

    payload = json.loads(files[0].read_text(encoding="utf-8"))
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
    import cli

    stub = _make_stub_cli([])
    cli.HermesCLI.save_conversation(stub, "/save json")

    saved_dir = hermes_home / "sessions" / "saved"
    assert not saved_dir.exists() or not list(saved_dir.iterdir())
    out = capsys.readouterr().out
    assert "No conversation to save" in out


def test_save_conversation_bare_shows_usage(hermes_home, capsys):
    """Bare /save prints the usage card and writes nothing."""
    import cli

    stub = _make_stub_cli([{"role": "user", "content": "hi"}])
    cli.HermesCLI.save_conversation(stub, "/save")

    saved_dir = hermes_home / "sessions" / "saved"
    assert not saved_dir.exists() or not list(saved_dir.iterdir())
    out = capsys.readouterr().out
    # Usage card lists every format and the redact option
    for token in ("json", "md", "html", "redact", "Usage:"):
        assert token in out, (token, out)


def test_save_conversation_bad_format_shows_usage(hermes_home, capsys):
    import cli

    stub = _make_stub_cli([{"role": "user", "content": "hi"}])
    cli.HermesCLI.save_conversation(stub, "/save pdf")

    saved_dir = hermes_home / "sessions" / "saved"
    assert not saved_dir.exists() or not list(saved_dir.iterdir())
    out = capsys.readouterr().out
    assert "Usage:" in out


@posix_only
@pytest.mark.parametrize("fmt", ["json", "md", "html"])
def test_save_conversation_fresh_artifact_is_owner_only(
    hermes_home, tmp_path, monkeypatch, fmt
):
    monkeypatch.chdir(tmp_path)
    import cli

    stub = _make_stub_cli([{"role": "user", "content": "hi"}])
    old_umask = os.umask(0o022)
    try:
        cli.HermesCLI.save_conversation(stub, f"/save {fmt}")
    finally:
        os.umask(old_umask)

    saved_dir = hermes_home / "sessions" / "saved"
    files = list(saved_dir.glob(f"hermes_conversation_*.{fmt}"))
    assert len(files) == 1
    assert not _mode(saved_dir) & 0o077
    assert _mode(files[0]) == 0o600


@posix_only
def test_save_conversation_retightens_existing_snapshot(
    hermes_home, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    import cli

    frozen = datetime(2026, 1, 1, 12, 0, 0)
    monkeypatch.setattr(cli, "datetime", SimpleNamespace(now=lambda: frozen))
    saved_dir = hermes_home / "sessions" / "saved"
    saved_dir.mkdir(parents=True)
    path = saved_dir / "hermes_conversation_20260101_120000.json"
    path.write_text("{}", encoding="utf-8")
    os.chmod(path, 0o644)

    history = [{"role": "user", "content": "hi"}]
    cli.HermesCLI.save_conversation(_make_stub_cli(history), "/save json")
    assert _mode(path) == 0o600
    assert json.loads(path.read_text(encoding="utf-8"))["messages"] == history


@posix_only
def test_explicit_save_preserves_existing_output_permissions(hermes_home, tmp_path):
    import cli

    path = tmp_path / "shared.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o644)
    history = [{"role": "user", "content": "hi"}]
    cli.HermesCLI.save_conversation(_make_stub_cli(history), f"/save json {path}")
    assert _mode(path) == 0o644
    assert json.loads(path.read_text(encoding="utf-8"))["messages"] == history
