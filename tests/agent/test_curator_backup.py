"""Tests for agent/curator_backup.py — snapshot + rollback of the skills tree."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def backup_env(monkeypatch, tmp_path):
    """Isolate HERMES_HOME + reload modules so every test starts clean."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Reload so get_hermes_home picks up the env var fresh.
    import hermes_constants
    importlib.reload(hermes_constants)
    from agent import curator_backup
    importlib.reload(curator_backup)
    return {"home": home, "skills": home / "skills", "cb": curator_backup}


def _write_skill(skills_dir: Path, name: str, body: str = "body") -> Path:
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: t\nversion: 1.0\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return d


# ---------------------------------------------------------------------------
# snapshot_skills
# ---------------------------------------------------------------------------

def test_snapshot_creates_tarball_and_manifest(backup_env):
    cb = backup_env["cb"]
    _write_skill(backup_env["skills"], "alpha")
    _write_skill(backup_env["skills"], "beta")

    snap = cb.snapshot_skills(reason="test")
    assert snap is not None, "snapshot should succeed with a populated skills dir"
    assert (snap / "skills.tar.gz").exists()
    manifest = json.loads((snap / "manifest.json").read_text())
    assert manifest["reason"] == "test"
    assert manifest["skill_files"] == 2
    assert manifest["archive_bytes"] > 0


def test_snapshot_excludes_backups_dir_itself(backup_env):
    """The backup must NOT contain .curator_backups/ — that would recurse
    with every subsequent snapshot and balloon disk usage."""
    cb = backup_env["cb"]
    _write_skill(backup_env["skills"], "alpha")
    snap1 = cb.snapshot_skills(reason="first")
    assert snap1 is not None
    snap2 = cb.snapshot_skills(reason="second")
    assert snap2 is not None
    with tarfile.open(snap2 / "skills.tar.gz") as tf:
        names = tf.getnames()
    assert not any(n.startswith(".curator_backups") for n in names), (
        "second snapshot must not contain the first snapshot recursively"
    )


def test_snapshot_excludes_hub_dir(backup_env):
    """.hub/ is managed by the skills hub. Rolling it back would break
    lockfile invariants, so the snapshot omits it entirely."""
    cb = backup_env["cb"]
    hub = backup_env["skills"] / ".hub"
    hub.mkdir()
    (hub / "lock.json").write_text("{}")
    _write_skill(backup_env["skills"], "alpha")
    snap = cb.snapshot_skills(reason="t")
    assert snap is not None
    with tarfile.open(snap / "skills.tar.gz") as tf:
        names = tf.getnames()
    assert not any(n.startswith(".hub") for n in names)


def test_snapshot_disabled_returns_none(backup_env, monkeypatch):
    cb = backup_env["cb"]
    monkeypatch.setattr(cb, "is_enabled", lambda: False)
    _write_skill(backup_env["skills"], "alpha")
    assert cb.snapshot_skills() is None
    # And no backup dir should have been created
    assert not (backup_env["skills"] / ".curator_backups").exists()


def test_snapshot_uniquifies_when_same_second(backup_env, monkeypatch):
    """Two snapshots in the same wallclock second must not clobber each
    other. The module appends a counter to the second snapshot's id."""
    cb = backup_env["cb"]
    _write_skill(backup_env["skills"], "alpha")
    frozen = "2026-05-01T12-00-00Z"
    monkeypatch.setattr(cb, "_utc_id", lambda now=None: frozen)
    s1 = cb.snapshot_skills(reason="a")
    s2 = cb.snapshot_skills(reason="b")
    assert s1 is not None and s2 is not None
    assert s1.name == frozen
    assert s2.name == f"{frozen}-01"


def test_snapshot_prunes_to_keep_count(backup_env, monkeypatch):
    cb = backup_env["cb"]
    _write_skill(backup_env["skills"], "alpha")
    monkeypatch.setattr(cb, "get_keep", lambda: 3)

    # Create 5 snapshots with monotonically increasing fake ids
    ids = [f"2026-05-0{i}T00-00-00Z" for i in range(1, 6)]
    for i, fid in enumerate(ids):
        monkeypatch.setattr(cb, "_utc_id", lambda now=None, _f=fid: _f)
        cb.snapshot_skills(reason=f"n{i}")

    remaining = sorted(p.name for p in (backup_env["skills"] / ".curator_backups").iterdir())
    # Newest 3 kept (lex order == date order for this id format)
    assert remaining == ids[2:], f"expected newest 3, got {remaining}"


# ---------------------------------------------------------------------------
# list_backups / _resolve_backup
# ---------------------------------------------------------------------------

def test_list_backups_empty(backup_env):
    cb = backup_env["cb"]
    assert cb.list_backups() == []


def test_list_backups_returns_manifest_data(backup_env):
    cb = backup_env["cb"]
    _write_skill(backup_env["skills"], "alpha")
    cb.snapshot_skills(reason="m1")
    rows = cb.list_backups()
    assert len(rows) == 1
    assert rows[0]["reason"] == "m1"
    assert rows[0]["skill_files"] == 1


def test_resolve_backup_newest_when_no_id(backup_env, monkeypatch):
    cb = backup_env["cb"]
    _write_skill(backup_env["skills"], "alpha")
    ids = ["2026-05-01T00-00-00Z", "2026-05-02T00-00-00Z"]
    for fid in ids:
        monkeypatch.setattr(cb, "_utc_id", lambda now=None, _f=fid: _f)
        cb.snapshot_skills()
    resolved = cb._resolve_backup(None)
    assert resolved is not None
    assert resolved.name == "2026-05-02T00-00-00Z", (
        "resolve(None) must return newest regular snapshot"
    )


def test_resolve_backup_unknown_id_returns_none(backup_env):
    cb = backup_env["cb"]
    _write_skill(backup_env["skills"], "alpha")
    cb.snapshot_skills()
    assert cb._resolve_backup("not-an-id") is None


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------

def test_rollback_restores_deleted_skill(backup_env):
    """The whole point of this feature: user loses a skill, rollback
    brings it back."""
    cb = backup_env["cb"]
    skills = backup_env["skills"]
    user_skill = _write_skill(skills, "my-personal-workflow", body="important content")
    cb.snapshot_skills(reason="pre-simulated-curator")

    # Simulate curator archiving it out of existence
    import shutil as _sh
    _sh.rmtree(user_skill)
    assert not user_skill.exists()

    ok, msg, _ = cb.rollback()
    assert ok, f"rollback failed: {msg}"
    assert user_skill.exists(), "my-personal-workflow should be restored"
    assert "important content" in (user_skill / "SKILL.md").read_text()


def test_rollback_is_itself_undoable(backup_env):
    """A rollback creates its own safety snapshot before replacing the
    tree, so the user can undo a mistaken rollback. The safety snapshot
    is a real tarball with reason='pre-rollback to <id>' — it's
    listed by list_backups() just like any other snapshot and can be
    restored the same way."""
    cb = backup_env["cb"]
    skills = backup_env["skills"]
    _write_skill(skills, "v1")
    cb.snapshot_skills(reason="snapshot-of-v1")

    # Overwrite with a new skill state
    import shutil as _sh
    _sh.rmtree(skills / "v1")
    _write_skill(skills, "v2")

    ok, _, _ = cb.rollback()
    assert ok
    assert (skills / "v1").exists()

    # list_backups should show a safety snapshot tagged "pre-rollback to <target-id>"
    rows = cb.list_backups()
    pre_rollback_entries = [r for r in rows if "pre-rollback" in (r.get("reason") or "")]
    assert len(pre_rollback_entries) >= 1, (
        f"expected a pre-rollback safety snapshot in list_backups(), got: "
        f"{[(r.get('id'), r.get('reason')) for r in rows]}"
    )
    # And the transient staging dir must be gone (it's implementation detail)
    backups_dir = skills / ".curator_backups"
    staging_dirs = [p for p in backups_dir.iterdir() if p.name.startswith(".rollback-staging-")]
    assert staging_dirs == [], (
        f"staging dir should be cleaned up on success, got: {staging_dirs}"
    )


def test_rollback_no_snapshots_returns_error(backup_env):
    cb = backup_env["cb"]
    ok, msg, _ = cb.rollback()
    assert not ok
    assert "no matching backup" in msg.lower() or "no snapshot" in msg.lower()


def test_rollback_rejects_unsafe_tarball(backup_env, monkeypatch):
    """Tarballs with absolute paths or .. components must be refused even
    if someone crafts a malicious snapshot. Defense in depth — normal
    curator snapshots never produce these."""
    cb = backup_env["cb"]
    skills = backup_env["skills"]
    _write_skill(skills, "alpha")
    cb.snapshot_skills(reason="legit")

    # Hand-craft a malicious tarball replacing the legit one
    rows = cb.list_backups()
    snap_dir = Path(rows[0]["path"])
    mal = snap_dir / "skills.tar.gz"
    mal.unlink()
    with tarfile.open(mal, "w:gz") as tf:
        evil = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
        evil.write(b"evil")
        evil.close()
        tf.add(evil.name, arcname="../../etc/evil.md")
        os.unlink(evil.name)

    ok, msg, _ = cb.rollback()
    assert not ok
    assert "unsafe" in msg.lower() or "refus" in msg.lower() or "extract" in msg.lower()


# ---------------------------------------------------------------------------
# Integration with run_curator_review
# ---------------------------------------------------------------------------

def test_real_run_takes_pre_snapshot(backup_env, monkeypatch):
    """A real (non-dry) curator pass must snapshot the tree before calling
    apply_automatic_transitions. This is the safety net #18373 asked for."""
    cb = backup_env["cb"]
    skills = backup_env["skills"]
    _write_skill(skills, "alpha")

    # Reload curator module against the freshly-env'd hermes_constants
    from agent import curator
    importlib.reload(curator)

    # Stub out LLM review and auto transitions — we only care about the
    # snapshot side-effect.
    monkeypatch.setattr(
        curator, "_run_llm_review",
        lambda p: {"final": "", "summary": "s", "model": "", "provider": "",
                   "tool_calls": [], "error": None},
    )
    monkeypatch.setattr(
        curator, "apply_automatic_transitions",
        lambda now=None: {"checked": 1, "marked_stale": 0, "archived": 0, "reactivated": 0},
    )

    curator.run_curator_review(synchronous=True)
    # Pre-run snapshot should exist
    rows = cb.list_backups()
    assert any(r.get("reason") == "pre-curator-run" for r in rows), (
        f"expected a pre-curator-run snapshot, got {[r.get('reason') for r in rows]}"
    )


def test_dry_run_skips_snapshot(backup_env, monkeypatch):
    """Dry-run previews must not spend disk on a snapshot — they don't
    mutate anything, so there's nothing to back up."""
    cb = backup_env["cb"]
    skills = backup_env["skills"]
    _write_skill(skills, "alpha")

    from agent import curator
    importlib.reload(curator)
    monkeypatch.setattr(
        curator, "_run_llm_review",
        lambda p: {"final": "", "summary": "s", "model": "", "provider": "",
                   "tool_calls": [], "error": None},
    )

    curator.run_curator_review(synchronous=True, dry_run=True)
    rows = cb.list_backups()
    assert not any(r.get("reason") == "pre-curator-run" for r in rows), (
        "dry-run must not create a pre-run snapshot"
    )
