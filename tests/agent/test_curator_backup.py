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









# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------



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


def test_rollback_aborts_when_safety_snapshot_fails(backup_env, monkeypatch):
    """Rollback must not replace live skills without an undo snapshot."""
    cb = backup_env["cb"]
    skills = backup_env["skills"]
    skill_file = _write_skill(skills, "alpha", body="snapshot state") / "SKILL.md"
    target = cb.snapshot_skills(reason="rollback-target")
    assert target is not None

    skill_file.write_text("current state\n", encoding="utf-8")
    current_bytes = skill_file.read_bytes()
    monkeypatch.setattr(cb, "snapshot_skills", lambda *args, **kwargs: None)

    ok, msg, restored = cb.rollback(target.name)

    assert not ok
    assert "safety snapshot failed" in msg
    assert restored is None
    assert skill_file.read_bytes() == current_bytes
    assert not list((skills / ".curator_backups").glob(".rollback-staging-*"))


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




# ---------------------------------------------------------------------------
# cron-jobs backup + rollback (the part issue #18671's follow-up adds)
# ---------------------------------------------------------------------------


def _write_cron_jobs(home: Path, jobs: list) -> Path:
    """Write a synthetic cron/jobs.json under HERMES_HOME. Returns the path.
    Mirrors cron.jobs.save_jobs() wrapper shape: `{"jobs": [...], "updated_at": ...}`.
    """
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    path = cron_dir / "jobs.json"
    path.write_text(
        json.dumps({"jobs": jobs, "updated_at": "2026-05-01T00:00:00Z"}, indent=2),
        encoding="utf-8",
    )
    return path


def _reload_cron_jobs(home: Path):
    """Reload cron.jobs so its module-level HERMES_DIR picks up the tmp HOME."""
    import hermes_constants
    importlib.reload(hermes_constants)
    if "cron.jobs" in sys.modules:
        import cron.jobs as _cj
        importlib.reload(_cj)
    else:
        import cron.jobs as _cj  # noqa: F401
    import cron.jobs as cj
    return cj








def test_snapshot_cron_jobs_utf8_bom_counted_and_backup_bomless(backup_env):
    """A UTF-8 BOM on jobs.json (Windows editors) must not break the job
    count, and the snapshot copy is written BOM-less so rollback restores a
    file cron/jobs.load_jobs can read."""
    cb = backup_env["cb"]
    _write_skill(backup_env["skills"], "alpha")
    cron_dir = backup_env["home"] / "cron"
    cron_dir.mkdir()
    payload = json.dumps({"jobs": [{"id": "job-a"}, {"id": "job-b"}]})
    (cron_dir / "jobs.json").write_bytes(b"\xef\xbb\xbf" + payload.encode())

    snap = cb.snapshot_skills(reason="test")
    assert snap is not None

    mf = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    assert mf["cron_jobs"]["backed_up"] is True
    assert mf["cron_jobs"]["jobs_count"] == 2
    assert "parse_warning" not in mf["cron_jobs"]

    # Backup copy is decoded text — no BOM survives into the snapshot.
    backup_bytes = (snap / cb.CRON_JOBS_FILENAME).read_bytes()
    assert not backup_bytes.startswith(b"\xef\xbb\xbf")
    assert json.loads(backup_bytes) == json.loads(payload)


def test_rollback_restores_cron_skill_links(backup_env):
    """End-to-end: snapshot with job [alpha,beta], curator-style in-place
    rewrite to [umbrella], then rollback → skills restored to [alpha,beta]."""
    cb = backup_env["cb"]
    home = backup_env["home"]
    _write_skill(backup_env["skills"], "alpha")
    _write_skill(backup_env["skills"], "beta")
    _write_skill(backup_env["skills"], "umbrella")

    cj = _reload_cron_jobs(home)
    cj.create_job(name="weekly", prompt="p", schedule="every 7d",
                  skills=["alpha", "beta"])

    snap = cb.snapshot_skills(reason="pre-curator-run")
    assert snap is not None

    # Simulate the curator's in-place cron rewrite after consolidation
    cj.rewrite_skill_refs(
        consolidated={"alpha": "umbrella", "beta": "umbrella"},
        pruned=[],
    )
    live_after_curator = cj.load_jobs()
    assert live_after_curator[0]["skills"] == ["umbrella"]

    # Now roll back
    ok, msg, _ = cb.rollback(backup_id=snap.name)
    assert ok, msg
    assert "cron links" in msg

    live_after_rollback = cj.load_jobs()
    # skills restored; legacy `skill` mirror follows first element
    assert live_after_rollback[0]["skills"] == ["alpha", "beta"]






def test_rollback_leaves_new_jobs_untouched(backup_env):
    """Jobs created AFTER the snapshot must pass through rollback unchanged."""
    cb = backup_env["cb"]
    home = backup_env["home"]
    _write_skill(backup_env["skills"], "alpha")
    _write_cron_jobs(home, [
        {"id": "original", "name": "o", "schedule": "every 1h", "skills": ["alpha"]},
    ])
    snap = cb.snapshot_skills(reason="pre-curator-run")

    cj = _reload_cron_jobs(home)
    jobs = cj.load_jobs()
    jobs.append({"id": "new-after-snapshot", "name": "new",
                 "schedule": "every 15m", "skills": ["brand-new-skill"]})
    cj.save_jobs(jobs)

    ok, _, _ = cb.rollback(backup_id=snap.name)
    assert ok

    live = cj.load_jobs()
    by_id = {j["id"]: j for j in live}
    assert "new-after-snapshot" in by_id
    # New job's fields completely preserved
    assert by_id["new-after-snapshot"]["skills"] == ["brand-new-skill"]
    assert by_id["new-after-snapshot"]["schedule"] == "every 15m"




def test_restore_cron_skill_links_standalone(backup_env):
    """Unit-level test on _restore_cron_skill_links without the full rollback.
    Verifies the report structure carefully."""
    cb = backup_env["cb"]
    home = backup_env["home"]

    # Prime a snapshot dir manually with cron-jobs.json
    backups_dir = home / "skills" / ".curator_backups" / "fake-id"
    backups_dir.mkdir(parents=True)
    (backups_dir / cb.CRON_JOBS_FILENAME).write_text(json.dumps([
        {"id": "job-1", "name": "one", "skills": ["narrow-a", "narrow-b"]},
        {"id": "job-2", "name": "two", "skill": "legacy-single"},
        {"id": "job-gone", "name": "deleted", "skills": ["whatever"]},
    ]), encoding="utf-8")

    # Live jobs: job-1 got rewritten, job-2 unchanged, job-gone deleted
    _write_cron_jobs(home, [
        {"id": "job-1", "name": "one", "skills": ["umbrella"], "schedule": "every 1h"},
        {"id": "job-2", "name": "two", "skill": "legacy-single", "schedule": "every 1h"},
        {"id": "job-new", "name": "new", "skills": ["x"], "schedule": "every 1h"},
    ])
    _reload_cron_jobs(home)

    report = cb._restore_cron_skill_links(backups_dir)
    assert report["attempted"] is True
    assert report["error"] is None
    assert report["unchanged"] == 1  # job-2 matched
    assert len(report["restored"]) == 1  # job-1 got restored
    assert report["restored"][0]["job_id"] == "job-1"
    assert report["restored"][0]["to"]["skills"] == ["narrow-a", "narrow-b"]
    assert len(report["skipped_missing"]) == 1
    assert report["skipped_missing"][0]["job_id"] == "job-gone"


# ---------------------------------------------------------------------------
# Rollback must not let the pre-rollback safety snapshot prune the target
# (regression: restoring the oldest snapshot at the keep limit destroyed it)
# ---------------------------------------------------------------------------

def _three_ordered_snapshots(cb, skills, monkeypatch):
    """Create snapshots 05-01 / 05-02 / 05-03 capturing growing trees, with
    keep=3 so the backups dir is exactly at the retention limit. 05-01 holds
    only 'pristine'; later snapshots add 'extra2' and 'extra3'. Leaves
    _utc_id patched to a newest id so the rollback safety snapshot sorts
    last. Returns the oldest snapshot id."""
    monkeypatch.setattr(cb, "get_keep", lambda: 3)
    plan = [
        ("2026-05-01T00-00-00Z", ["pristine"]),
        ("2026-05-02T00-00-00Z", ["pristine", "extra2"]),
        ("2026-05-03T00-00-00Z", ["pristine", "extra2", "extra3"]),
    ]
    for snap_id, names in plan:
        for n in names:
            _write_skill(skills, n)
        monkeypatch.setattr(cb, "_utc_id", lambda now=None, _i=snap_id: _i)
        assert cb.snapshot_skills(reason=snap_id) is not None
    monkeypatch.setattr(cb, "_utc_id", lambda now=None: "2026-05-09T00-00-00Z")
    return "2026-05-01T00-00-00Z"




# ---------------------------------------------------------------------------
# A failed extract must leave the skills tree exactly as it was found
# ---------------------------------------------------------------------------

def test_rollback_recovers_cleanly_from_a_partial_extract(backup_env, monkeypatch):
    """An extract that dies part-way must restore the original tree exactly.

    ``shutil.move`` moves *into* an existing directory rather than replacing
    it, so debris from a half-finished extract buried the user's own skill one
    level deeper (``skills/alpha/alpha/``) while rollback still reported
    "state restored".
    """
    cb = backup_env["cb"]
    skills = backup_env["skills"]

    _write_skill(skills, "alpha", body="snapshot copy")
    assert cb.snapshot_skills(reason="before") is not None

    # Diverge from the snapshot so a real restore would be observable.
    _write_skill(skills, "alpha", body="current copy")
    _write_skill(skills, "beta", body="current only")

    real_open = tarfile.open

    class _DiesMidExtract:
        """Writes part of the archive, then fails like a full disk would."""

        def __init__(self, inner):
            self._inner = inner

        def getmembers(self):
            return self._inner.getmembers()

        def extractall(self, path, *args, **kwargs):
            partial = Path(path) / "alpha"
            partial.mkdir(parents=True, exist_ok=True)
            (partial / "SKILL.md").write_text("half written", encoding="utf-8")
            (Path(path) / "gamma").mkdir(parents=True, exist_ok=True)
            raise OSError(28, "No space left on device")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._inner.close()
            return False

    def _open(name, mode="r", *args, **kwargs):
        # Only the rollback's read is intercepted; the pre-rollback safety
        # snapshot opens for write and must keep working.
        handle = real_open(name, mode, *args, **kwargs)
        return _DiesMidExtract(handle) if mode.startswith("r") else handle

    monkeypatch.setattr(cb.tarfile, "open", _open)

    ok, msg, _ = cb.rollback()
    assert not ok

    assert not (skills / "alpha" / "alpha").exists(), \
        "staged skill was nested, not restored"
    present = sorted(
        p.name for p in skills.iterdir() if p.name not in cb._EXCLUDE_TOP_LEVEL
    )
    assert present == ["alpha", "beta"], f"tree not restored: {present}"
    assert "current copy" in (skills / "alpha" / "SKILL.md").read_text(encoding="utf-8")
    assert "current only" in (skills / "beta" / "SKILL.md").read_text(encoding="utf-8")


def test_snapshot_excludes_git_and_curator_backups_and_hub(backup_env):
    """Tar snapshots must exclude .git, .curator_backups, and .hub (top-level and nested)."""
    cb = backup_env["cb"]
    skills = backup_env["skills"]

    # Top-level excluded structures
    (skills / ".git").mkdir()
    (skills / ".git" / "config").write_text("[core]\nrepositoryformatversion = 0\n", encoding="utf-8")
    (skills / ".hub").mkdir()
    (skills / ".hub" / "lock.json").write_text("{}", encoding="utf-8")

    # Regular skill with nested .git
    _write_skill(skills, "alpha", body="alpha body")
    nested_git = skills / "alpha" / ".git"
    nested_git.mkdir()
    (nested_git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    snap_dir = cb.snapshot_skills(reason="test-exclude-git")
    assert snap_dir is not None

    archive = snap_dir / "skills.tar.gz"
    assert archive.exists()

    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getnames()

    # Ensure no member contains .git, .curator_backups, or .hub
    for name in members:
        parts = Path(name).parts
        assert ".git" not in parts, f".git found in archive: {name}"
        assert ".curator_backups" not in parts, f".curator_backups found in archive: {name}"
        assert ".hub" not in parts, f".hub found in archive: {name}"

    assert "alpha/SKILL.md" in members


def test_rollback_preserves_top_level_git(backup_env):
    """Rollback must preserve repository .git metadata untouched in the skills root."""
    cb = backup_env["cb"]
    skills = backup_env["skills"]

    git_dir = skills / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    _write_skill(skills, "alpha", body="v1")

    snap_dir = cb.snapshot_skills(reason="snap-v1")
    assert snap_dir is not None

    # Mutate skills tree and git metadata
    _write_skill(skills, "alpha", body="v2")
    _write_skill(skills, "beta", body="new")
    (git_dir / "HEAD").write_text("ref: refs/heads/feature\n", encoding="utf-8")

    ok, msg, _ = cb.rollback(backup_id=snap_dir.name)
    assert ok, f"rollback failed: {msg}"

    assert (skills / "alpha" / "SKILL.md").read_text(encoding="utf-8").find("v1") != -1
    assert not (skills / "beta").exists()
    assert (git_dir / "HEAD").exists()
    assert (git_dir / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/feature\n"

