"""Isolated-restore verification of backup archives (#117005).

``hermes import --verify-only`` must prove a capture is restorable *without* touching the
profile it is run from, and must never report ``verified`` for a damaged archive — the
whole point of the drill is that a readable zip is not the same thing as a restore.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.backup_verify import run_verify_import, verify_backup_archive


def _make_source_home(root: Path) -> Path:
    """A minimal but real Hermes home: config, skill, auth, cron state, session store."""
    home = root / "source"
    (home / "skills" / "demo").mkdir(parents=True)
    (home / "config.yaml").write_text("model: demo\n", encoding="utf-8")
    (home / "skills" / "demo" / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    (home / "auth.json").write_text(json.dumps({"active_provider": "nous"}), encoding="utf-8")
    (home / "cron").mkdir()
    (home / "cron" / "jobs.json").write_text(
        json.dumps({"jobs": [{"id": "job-1"}]}), encoding="utf-8")
    conn = sqlite3.connect(home / "state.db")
    try:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE messages (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES ('s1')")
        conn.execute("INSERT INTO messages VALUES ('m1')")
        conn.commit()
    finally:
        conn.close()
    return home


def _zip_home(home: Path, zip_path: Path, extra: dict[str, str] | None = None) -> Path:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted(p for p in home.rglob("*") if p.is_file()):
            zf.write(path, arcname=path.relative_to(home).as_posix())
        for name, content in (extra or {}).items():
            zf.writestr(name, content)
    return zip_path


def _tree_digest(root: Path) -> dict[str, str]:
    """Every file under *root* as rel-path -> content hash (mutation detector)."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_valid_backup_verifies_without_touching_the_profile(tmp_path, monkeypatch):
    """A good archive verifies into its own home; the profile it ran from never moves."""
    live = tmp_path / "live-hermes"
    live.mkdir()
    (live / "config.yaml").write_text("model: live\n", encoding="utf-8")
    (live / "sentinel.db").write_bytes(b"live state, must not move")
    monkeypatch.setenv("HERMES_HOME", str(live))
    before = _tree_digest(live)

    archive = _zip_home(_make_source_home(tmp_path), tmp_path / "backup.zip")
    candidate = tmp_path / "candidate"

    receipt = verify_backup_archive(archive, candidate_home=candidate)

    assert receipt["status"] == "verified", receipt["errors"]
    assert receipt["missing_objects"] == []
    assert receipt["restored_objects"] == receipt["required_objects"] > 0
    assert (candidate / "state.db").is_file()
    store = receipt["stores"]["state.db"]
    assert store["status"] == "healthy"
    assert (store["sessions"], store["messages"]) == (1, 1)
    assert receipt["config"] == {"status": "valid"}
    assert receipt["auth"] == {"status": "valid"}
    assert receipt["cron"] == {"status": "valid", "jobs": 1}
    # The profile the drill ran from is byte-for-byte unchanged.
    assert _tree_digest(live) == before


def test_successful_drill_removes_its_restore_unless_kept(tmp_path):
    """The candidate home is temporary by default and only survives --keep-candidate."""
    archive = _zip_home(_make_source_home(tmp_path), tmp_path / "backup.zip")

    removed = verify_backup_archive(archive)
    assert removed["status"] == "verified", removed["errors"]
    assert removed["candidate_home"] is None
    assert removed["candidate_retained"] is False

    kept = verify_backup_archive(archive, keep_candidate=True)
    assert kept["status"] == "verified", kept["errors"]
    assert kept["candidate_retained"] is True
    assert Path(kept["candidate_home"], "config.yaml").is_file()
    shutil.rmtree(kept["candidate_home"], ignore_errors=True)


@pytest.mark.parametrize(
    "damage, blamed",
    [
        ("config", "config.yaml"),
        ("state", "state.db"),
    ],
)
def test_damaged_archive_cannot_report_verified(tmp_path, monkeypatch, damage, blamed):
    """Corrupting one required object fails the drill nonzero and names the object."""
    source = _make_source_home(tmp_path)
    if damage == "config":
        (source / "config.yaml").write_text("model: [unclosed\n", encoding="utf-8")
    else:
        (source / "state.db").write_bytes(b"not a sqlite database, just noise" * 8)
    archive = _zip_home(source, tmp_path / "backup.zip")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "live-hermes"))

    receipt = verify_backup_archive(archive)

    assert receipt["status"] == "failed", receipt
    assert any(blamed in error for error in receipt["errors"]), receipt["errors"]
    if damage == "state":
        assert receipt["stores"]["state.db"]["status"] == "unhealthy"

    # The CLI contract callers gate destructive recovery on: nonzero, never "verified".
    with pytest.raises(SystemExit) as excinfo:
        run_verify_import(SimpleNamespace(zipfile=str(archive), keep_candidate=False))
    assert excinfo.value.code == 1


def test_external_members_are_policy_exclusions(tmp_path, monkeypatch):
    """``_external/`` entries restore outside HERMES_HOME, so a drill must not publish them."""
    fake_home = tmp_path / "user-home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    archive = _zip_home(
        _make_source_home(tmp_path), tmp_path / "backup.zip",
        extra={"_external/.honcho/config.json": "{}"})
    candidate = tmp_path / "candidate"

    receipt = verify_backup_archive(archive, candidate_home=candidate)

    assert receipt["status"] == "verified", receipt["errors"]
    assert receipt["policy_exclusions"] == 1
    assert _tree_digest(fake_home) == {}  # nothing was written back into the user's home
    assert not (candidate / "_external").exists()
