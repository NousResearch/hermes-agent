"""Invariant for the non-secret atomic JSON writers that used to inline ``utils._atomic_write``.

Contract under test for ``gateway/session_persistence``, ``cron/suggestions`` and
``agent/shell_hooks``: a failed replace leaves the previous file byte-identical AND leaves no temp
file behind (the interrupt-safe cleanup only the canonical helper guarantees). A hand-rolled copy
that skips the cleanup, or writes through the target instead of a sibling temp, fails this.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _failing_replace(tmp, target):
    raise OSError("simulated disk full")


def _leftovers(directory: Path, keep: str) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name != keep)


@pytest.fixture
def broken_replace(monkeypatch):
    import utils

    monkeypatch.setattr(utils, "atomic_replace", _failing_replace)


def test_sessions_json_failed_replace_keeps_old_bytes_and_no_temp(tmp_path, monkeypatch, broken_replace):
    from gateway.session_persistence import SessionPersistenceMixin

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    target = sessions_dir / "sessions.json"
    target.write_text('{"old": true}', encoding="utf-8")
    store = SessionPersistenceMixin()
    store.sessions_dir = sessions_dir
    with pytest.raises(OSError):
        store._save_sessions_json({"k": "v"})
    assert target.read_text(encoding="utf-8") == '{"old": true}'
    assert _leftovers(sessions_dir, "sessions.json") == []


def test_suggestions_failed_replace_keeps_old_bytes_and_no_temp(tmp_path, monkeypatch, broken_replace):
    from cron import suggestions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = suggestions._current_suggestions_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"old": true}', encoding="utf-8")
    with pytest.raises(OSError):
        suggestions._save_raw([{"id": 1}])
    assert target.read_text(encoding="utf-8") == '{"old": true}'
    assert _leftovers(target.parent, target.name) == []


def test_shell_hooks_allowlist_survives_failed_replace_without_temp(tmp_path, monkeypatch, broken_replace):
    from agent import shell_hooks

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    target = shell_hooks.allowlist_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"approvals": []}), encoding="utf-8")
    shell_hooks.save_allowlist({"approvals": [{"event": "a", "command": "x"}]})  # logs, never raises
    assert json.loads(target.read_text(encoding="utf-8")) == {"approvals": []}
    assert _leftovers(target.parent, target.name) == []


def test_surrogate_escaped_strings_round_trip_through_atomic_json_write(tmp_path):
    """A non-UTF-8 cwd/argv (``os.fsdecode`` → lone surrogate) must be persisted, not raise.

    Breadcrumbs, the shell-hook allowlist and the active-sessions ledger all persist paths and
    guard only ``OSError``; a ``UnicodeEncodeError`` (a ValueError) escaping the canonical writer
    silently stopped those writes.
    """
    from utils import atomic_json_write

    payload = {"cwd": "a\udcffb", "plain": "caf\u00e9"}
    target = tmp_path / "crumbs" / "crumb.json"
    atomic_json_write(target, payload)
    assert json.loads(target.read_bytes()) == payload
    assert _leftovers(target.parent, target.name) == []


@pytest.mark.linux_only
def test_new_non_secret_file_follows_umask_while_secret_and_existing_modes_hold(tmp_path):
    """The writers this helper replaced created files at process umask; only ``mode=`` tightens."""
    import os
    import stat

    from utils import atomic_json_write

    old_umask = os.umask(0o022)
    try:
        fresh = tmp_path / "cache.json"
        atomic_json_write(fresh, {"a": 1})
        assert stat.S_IMODE(fresh.stat().st_mode) == 0o644, "new non-secret file must not inherit mkstemp's 0600"

        secret = tmp_path / "creds.json"
        atomic_json_write(secret, {"token": "x"}, mode=0o600)
        assert stat.S_IMODE(secret.stat().st_mode) == 0o600

        existing = tmp_path / "state.json"
        existing.write_text("{}", encoding="utf-8")
        os.chmod(existing, 0o640)
        atomic_json_write(existing, {"b": 2})
        assert stat.S_IMODE(existing.stat().st_mode) == 0o640
    finally:
        os.umask(old_umask)


@pytest.mark.linux_only
def test_mkstemp_heritage_writers_keep_new_files_owner_only(tmp_path):
    """Writers that published through mkstemp on main created NEW files at 0600 regardless of umask
    (bot mailboxes, turn markers); folding them into utils must not loosen that to umask."""
    import os
    import stat

    from tools.bot_relay import _atomic_write_json
    from tui_gateway.turn_marker import _store

    old_umask = os.umask(0o022)
    try:
        relay_target = tmp_path / "relay" / "inbox.json"
        _atomic_write_json(relay_target, {"k": 1})
        marker = tmp_path / "turn-marker.json"
        _store(marker, {"sess": {"started_at": 1.0}})
        for path in (relay_target, marker):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600, path
    finally:
        os.umask(old_umask)
