"""Regression: _auth_lock_path() must follow symlinks so shared-auth setups
serialize correctly across processes.

Motivating incident: 3 Hermes agent processes on one host each symlinked
their ~/.hermes/auth.json to a single canonical file (a shared Codex OAuth
setup). Because the flock lived at the CALLER's local ~/.hermes/auth.lock,
each process held a different lock while writing to the same file — and a
cross-process load-modify-save race purged 3 of 5 providers from
credential_pool in one refresh cycle. See #8040 (comment) for the trace.
"""

from pathlib import Path

import hermes_cli.auth as auth
from hermes_cli.auth import _auth_lock_path


def _make_home(base: Path, name: str, shared_auth: Path) -> Path:
    home = base / name / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").symlink_to(shared_auth)
    return home


def test_lock_path_follows_symlink_to_shared_target(tmp_path, monkeypatch):
    """Two profiles sharing an auth.json via symlink must resolve to the SAME lock."""
    shared = tmp_path / "shared" / "auth.json"
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_text('{"version": 1, "providers": {}}')

    home_a = _make_home(tmp_path, "agent_a", shared)
    home_b = _make_home(tmp_path, "agent_b", shared)

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    lock_a = _auth_lock_path()

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    lock_b = _auth_lock_path()

    expected = shared.parent / "auth.lock"
    assert lock_a == expected
    assert lock_b == expected
    assert lock_a == lock_b


def test_lock_path_for_regular_file_unchanged(tmp_path, monkeypatch):
    """Non-symlinked auth.json: lock sits next to it (behavior unchanged)."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text('{"version": 1, "providers": {}}')

    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _auth_lock_path() == home / "auth.lock"


def test_lock_path_when_auth_json_missing(tmp_path, monkeypatch):
    """No auth.json yet: lock path derives from _auth_file_path() unchanged."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _auth_lock_path() == home / "auth.lock"


def test_lock_path_with_broken_symlink(tmp_path, monkeypatch):
    """Dangling symlink: realpath returns the (non-existent) target;
    lock still lives next to it so a future creator picks up the same lock."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    missing_target = tmp_path / "shared" / "auth.json"
    (home / "auth.json").symlink_to(missing_target)

    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _auth_lock_path() == missing_target.parent / "auth.lock"
