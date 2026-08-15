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
    os.name != "posix",
    reason="POSIX permission bits are advisory on Windows",
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

    # No sys.modules surgery: save_conversation resolves the home through a
    # call-time get_hermes_home(), which re-reads HERMES_HOME on every call.
    import cli  # noqa: F401  (module under test)

    stub = _make_stub_cli([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])

    # Call the unbound method against our stub.
    cli.HermesCLI.save_conversation(stub)

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
    assert payload["session_id"] == "20260101_120000_abc123"
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
    cli.HermesCLI.save_conversation(stub)

    saved_dir = hermes_home / "sessions" / "saved"
    assert not saved_dir.exists() or not list(saved_dir.iterdir())
    out = capsys.readouterr().out
    assert "No conversation to save" in out


@posix_only
def test_save_conversation_fresh_dir_and_file_owner_only(hermes_home, tmp_path, monkeypatch):
    """Under a permissive umask, saved/ and a fresh snapshot must not leak bits.

    Regression: the handler used a bare mkdir plus open(path, "w"), so a
    transcript of the entire conversation landed at umask-derived 0o755/0o644 -
    readable by every other account on a shared box.
    """
    monkeypatch.chdir(tmp_path)
    import cli

    stub = _make_stub_cli([{"role": "user", "content": "hi"}])

    old_umask = os.umask(0o022)
    try:
        cli.HermesCLI.save_conversation(stub)
    finally:
        os.umask(old_umask)

    saved_dir = hermes_home / "sessions" / "saved"
    files = list(saved_dir.glob("hermes_conversation_*.json"))
    assert len(files) == 1, files
    assert not _mode(saved_dir) & 0o077, (
        f"saved dir mode {oct(_mode(saved_dir))} leaks to group/other"
    )
    assert not _mode(files[0]) & 0o077, (
        f"snapshot mode {oct(_mode(files[0]))} leaks to group/other"
    )


@posix_only
def test_save_conversation_retightens_preexisting_broad_snapshot(
    hermes_home, tmp_path, monkeypatch
):
    """A pre-existing 0o644 snapshot at the same path is rewritten to 0o600.

    Freezing the timestamp collides the target path with a file already on disk
    at a broad mode. That is what proves the write passes an explicit mode
    rather than merely inheriting the temp file 0o600 default: the atomic
    writer preserves the destination mode unless one is passed.
    """
    monkeypatch.chdir(tmp_path)
    import cli

    frozen = datetime(2026, 1, 1, 12, 0, 0)
    monkeypatch.setattr(cli, "datetime", SimpleNamespace(now=lambda: frozen))

    saved_dir = hermes_home / "sessions" / "saved"
    saved_dir.mkdir(parents=True)
    path = saved_dir / "hermes_conversation_20260101_120000.json"
    path.write_text("{}", encoding="utf-8")
    os.chmod(path, 0o644)

    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    stub = _make_stub_cli(history)

    old_umask = os.umask(0o022)
    try:
        cli.HermesCLI.save_conversation(stub)
    finally:
        os.umask(old_umask)

    assert _mode(path) == 0o600, oct(_mode(path))
    # The rewrite must be complete JSON, not a truncated or partial file.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["messages"] == history
    assert payload["model"] == "test-model"
    assert payload["session_id"] == "20260101_120000_abc123"
    assert payload["session_start"] == "2026-01-01T12:00:00"


@posix_only
def test_save_conversation_managed_setgid_parent_group_mode(tmp_path, monkeypatch):
    """Managed + setgid parent: saved/ stays group-writable, snapshot is 0o660."""
    home = tmp_path / ".hermes"
    home.mkdir()
    os.chmod(home, 0o2770)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    monkeypatch.chdir(tmp_path)

    probe = home / "kernel-probe"
    probe.mkdir()
    # Only the setgid-preservation assertion depends on the kernel inheriting
    # the bit (Linux does, macOS/APFS does not). The managed mode contract
    # itself does not, so it is asserted on every POSIX platform.
    setgid_inherited = bool(_mode(probe) & stat.S_ISGID)

    import cli

    stub = _make_stub_cli([{"role": "user", "content": "hi"}])

    old_umask = os.umask(0o007)
    try:
        cli.HermesCLI.save_conversation(stub)
    finally:
        os.umask(old_umask)

    saved_dir = home / "sessions" / "saved"
    files = list(saved_dir.glob("hermes_conversation_*.json"))
    assert len(files) == 1, files
    assert _mode(saved_dir) & 0o777 == 0o770, oct(_mode(saved_dir))
    assert _mode(files[0]) == 0o660, oct(_mode(files[0]))
    if setgid_inherited:
        assert _mode(saved_dir) & stat.S_ISGID, (
            "inherited setgid must survive; without it 0o660 group-write is "
            "useless to the second UID of a managed install"
        )
