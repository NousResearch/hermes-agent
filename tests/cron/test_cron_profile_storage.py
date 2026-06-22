"""Regression tests for profile-aware cron storage.

Default cron storage remains anchored at the root Hermes home so jobs created
from ordinary profile-scoped sessions are visible to the default/root gateway
(#32091). Isolated persona gateways can opt into profile-local cron storage with
``cron.storage_scope: profile`` so they do not run the default profile's jobs and
miss their own profile-local schedule.
"""
import importlib
from pathlib import Path


def test_cron_storage_anchors_at_root_under_profile_by_default(tmp_path, monkeypatch):
    """Under a profile HERMES_HOME, default behavior still stores at root."""
    root = tmp_path / "hermes_home"
    profile_home = root / "profiles" / "myprofile"
    profile_home.mkdir(parents=True)

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    assert hermes_constants.get_default_hermes_root().resolve() == root.resolve()
    assert hermes_constants.get_hermes_home().resolve() == profile_home.resolve()

    import cron.jobs as jobs
    importlib.reload(jobs)
    try:
        assert jobs.HERMES_DIR.resolve() == root.resolve()
        assert jobs.JOBS_FILE.resolve() == (root / "cron" / "jobs.json").resolve()
        assert jobs.JOBS_FILE.resolve() != (profile_home / "cron" / "jobs.json").resolve()
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)


def test_cron_storage_can_be_profile_local_for_isolated_gateways(tmp_path, monkeypatch):
    """cron.storage_scope=profile stores jobs next to the active profile."""
    root = tmp_path / "hermes_home"
    profile_home = root / "profiles" / "therapist"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("cron:\n  storage_scope: profile\n", encoding="utf-8")

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    import cron.jobs as jobs
    importlib.reload(jobs)
    try:
        assert jobs.HERMES_DIR.resolve() == profile_home.resolve()
        assert jobs.JOBS_FILE.resolve() == (profile_home / "cron" / "jobs.json").resolve()
        assert jobs.JOBS_FILE.resolve() != (root / "cron" / "jobs.json").resolve()
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)


def test_cron_storage_unaffected_when_no_profile(tmp_path, monkeypatch):
    """With no profile (HERMES_HOME == root), store at <root>/cron."""
    root = tmp_path / "hermes_home"
    root.mkdir(parents=True)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(root))

    import cron.jobs as jobs
    importlib.reload(jobs)
    try:
        assert jobs.JOBS_FILE.resolve() == (root / "cron" / "jobs.json").resolve()
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)


def test_tick_lock_anchors_at_root_under_profile_by_default(tmp_path, monkeypatch):
    """Default/root cron uses the root-wide tick lock (#32091)."""
    root = tmp_path / "hermes_home"
    profile_home = root / "profiles" / "p"
    profile_home.mkdir(parents=True)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    import cron.jobs as jobs
    import cron.scheduler as sched
    importlib.reload(jobs)
    importlib.reload(sched)
    try:
        sched._hermes_home = None
        lock_dir, lock_file = sched._get_lock_paths()
        assert lock_dir.resolve() == (root / "cron").resolve()
        assert lock_file.resolve() == (root / "cron" / ".tick.lock").resolve()
        assert lock_dir.resolve() != (profile_home / "cron").resolve()
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)
        importlib.reload(sched)


def test_tick_lock_follows_profile_local_storage(tmp_path, monkeypatch):
    """Profile-local cron gets a profile-local tick lock beside jobs.json."""
    root = tmp_path / "hermes_home"
    profile_home = root / "profiles" / "therapist"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("cron:\n  storage_scope: profile\n", encoding="utf-8")

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    import cron.jobs as jobs
    import cron.scheduler as sched
    importlib.reload(jobs)
    importlib.reload(sched)
    try:
        sched._hermes_home = None
        lock_dir, lock_file = sched._get_lock_paths()
        assert lock_dir.resolve() == (profile_home / "cron").resolve()
        assert lock_file.resolve() == (profile_home / "cron" / ".tick.lock").resolve()
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)
        importlib.reload(sched)


def test_get_default_hermes_root_docker_layouts(tmp_path, monkeypatch):
    """get_default_hermes_root resolves Docker/custom HERMES_HOME roots."""
    import hermes_constants
    native = tmp_path / "native_home"
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: native)

    monkeypatch.setenv("HERMES_HOME", "/opt/data")
    assert hermes_constants.get_default_hermes_root() == Path("/opt/data")

    monkeypatch.setenv("HERMES_HOME", "/opt/data/profiles/coder")
    assert hermes_constants.get_default_hermes_root() == Path("/opt/data")
