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

