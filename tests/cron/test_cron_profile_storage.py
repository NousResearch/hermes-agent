"""Regression tests for #32091 — profile-scoped cron jobs orphaned.

Cron storage (CRON_DIR/JOBS_FILE) must anchor at the *default root* Hermes
home, not the active profile's home. Otherwise a job created from a
profile-scoped agent session writes to ~/.hermes/profiles/<p>/cron/jobs.json,
while the profile-less gateway reads only ~/.hermes/cron/jobs.json — the job
is silently orphaned (looks healthy in `list`, never fires).
"""
import importlib
import os
from pathlib import Path


def test_cron_storage_anchors_at_root_under_profile(tmp_path, monkeypatch):
    """Under a profile HERMES_HOME (<root>/profiles/<name>), the cron store
    resolves to <root>/cron, NOT <root>/profiles/<name>/cron."""
    root = tmp_path / "hermes_home"
    profile_home = root / "profiles" / "myprofile"
    profile_home.mkdir(parents=True)

    # Pretend the platform default root IS our tmp root, and the active
    # HERMES_HOME is a profile under it (the #32091 scenario).
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home",
                        lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    # get_default_hermes_root must return the ROOT, not the profile dir.
    assert hermes_constants.get_default_hermes_root().resolve() == root.resolve()
    # ...while get_hermes_home (used elsewhere) follows the profile override.
    assert hermes_constants.get_hermes_home().resolve() == profile_home.resolve()

    # cron/jobs.py computes HERMES_DIR from get_default_hermes_root at import,
    # so a fresh import under this env anchors the store at <root>/cron.
    import cron.jobs as jobs
    importlib.reload(jobs)
    try:
        assert jobs.HERMES_DIR.resolve() == root.resolve()
        assert jobs.JOBS_FILE.resolve() == (root / "cron" / "jobs.json").resolve()
        # The orphan path (<profile>/cron/jobs.json) must NOT be the store.
        assert jobs.JOBS_FILE.resolve() != (profile_home / "cron" / "jobs.json").resolve()
    finally:
        # Restore module state for other tests (reload under the real env).
        monkeypatch.undo()
        importlib.reload(jobs)


def test_cron_storage_unaffected_when_no_profile(tmp_path, monkeypatch):
    """With no profile (HERMES_HOME == root), behavior is unchanged: store at
    <root>/cron."""
    root = tmp_path / "hermes_home"
    root.mkdir(parents=True)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home",
                        lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(root))

    import cron.jobs as jobs
    importlib.reload(jobs)
    try:
        assert jobs.JOBS_FILE.resolve() == (root / "cron" / "jobs.json").resolve()
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)


def test_tick_lock_anchors_at_root_under_profile(tmp_path, monkeypatch):
    """The cron tick lock must live at <root>/cron/.tick.lock, NOT the profile
    dir — otherwise tickers under different profiles grab different locks and
    double-fire the (now root-anchored) jobs store (#32091)."""
    import importlib
    root = tmp_path / "hermes_home"
    profile_home = root / "profiles" / "p"
    profile_home.mkdir(parents=True)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    import cron.scheduler as sched
    importlib.reload(sched)
    try:
        # _hermes_home override is None -> uses get_default_hermes_root()
        sched._hermes_home = None
        lock_dir, lock_file = sched._get_lock_paths()
        assert lock_dir.resolve() == (root / "cron").resolve()
        assert lock_file.resolve() == (root / "cron" / ".tick.lock").resolve()
        assert lock_dir.resolve() != (profile_home / "cron").resolve()
    finally:
        monkeypatch.undo()
        importlib.reload(sched)


def test_get_default_hermes_root_docker_layouts(tmp_path, monkeypatch):
    """get_default_hermes_root resolves the root for Docker/custom HERMES_HOME
    (outside ~/.hermes), so cron storage works in containers."""
    import hermes_constants
    native = tmp_path / "native_home"
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: native)

    # Docker custom root (outside native): HERMES_HOME itself IS the root.
    monkeypatch.setenv("HERMES_HOME", "/opt/data")
    assert hermes_constants.get_default_hermes_root() == Path("/opt/data")

    # Docker profile layout: <custom>/profiles/<name> -> <custom>.
    monkeypatch.setenv("HERMES_HOME", "/opt/data/profiles/coder")
    assert hermes_constants.get_default_hermes_root() == Path("/opt/data")
