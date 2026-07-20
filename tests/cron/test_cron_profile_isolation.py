"""Regression tests for #4707 — cron must be per-profile.

Design intent (Teknium, June 2026): a profile's cron jobs both LIVE in that
profile's HERMES_HOME and EXECUTE under it.

- Storage: a job created under profile ``coder`` writes to
  ``~/.hermes/profiles/coder/cron/jobs.json`` — NOT the shared default root.
- Execution: the profile-scoped gateway's in-process ticker resolves the
  active HERMES_HOME (profile home) at call time, so jobs run with that
  profile's ``.env`` / ``config.yaml`` / scripts / skills.

This is the opposite direction from the (reverted) #50112/#32091 "anchor at the
shared root" approach. Anchoring at the root funnels every profile's jobs into
one store and runs them under whatever HERMES_HOME the ticker happens to have —
leaking config/credentials/skills across profiles, the security boundary #4707
was filed for. These tests pin per-profile isolation so a stale-branch merge or
a re-anchor "fix" can't silently flip it back.
"""
import importlib
import threading
from pathlib import Path


def _set_profile_env(monkeypatch, root: Path, profile_home: Path) -> None:
    """Pretend the platform default root is ``root`` and the active
    HERMES_HOME is a profile under it (``<root>/profiles/<name>``)."""
    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "_get_platform_default_hermes_home", lambda: root
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_home))


def test_cron_storage_anchors_at_profile_home(tmp_path, monkeypatch):
    """Under a profile HERMES_HOME (<root>/profiles/<name>), the cron store
    resolves to <profile>/cron, NOT the shared <root>/cron."""
    root = tmp_path / "hermes_home"
    profile_home = root / "profiles" / "coder"
    profile_home.mkdir(parents=True)

    _set_profile_env(monkeypatch, root, profile_home)

    import hermes_constants

    # Sanity: the override is wired the way the gateway sees it.
    assert hermes_constants.get_hermes_home().resolve() == profile_home.resolve()
    assert hermes_constants.get_default_hermes_root().resolve() == root.resolve()

    # cron/jobs.py computes HERMES_DIR from get_hermes_home() at import, so a
    # fresh import under this env anchors the store at <profile>/cron.
    import cron.jobs as jobs

    importlib.reload(jobs)
    try:
        assert jobs.HERMES_DIR.resolve() == profile_home.resolve()
        assert (
            jobs.JOBS_FILE.resolve()
            == (profile_home / "cron" / "jobs.json").resolve()
        )
        # The shared-root path must NOT be the store — that would re-break
        # per-profile isolation (#4707).
        assert (
            jobs.JOBS_FILE.resolve() != (root / "cron" / "jobs.json").resolve()
        )
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)


def test_cron_lock_path_anchors_at_profile_home(tmp_path, monkeypatch):
    """The tick lock is also profile-scoped, so two profile gateways tick
    independently instead of contending on one shared lock."""
    root = tmp_path / "hermes_home"
    profile_home = root / "profiles" / "coder"
    profile_home.mkdir(parents=True)

    _set_profile_env(monkeypatch, root, profile_home)

    import cron.scheduler as scheduler

    lock_dir, lock_file = scheduler._get_lock_paths()
    assert lock_dir.resolve() == (profile_home / "cron").resolve()
    assert lock_file.resolve() == (profile_home / "cron" / ".tick.lock").resolve()
    assert lock_dir.resolve() != (root / "cron").resolve()


def test_cron_execution_home_follows_active_profile(tmp_path, monkeypatch):
    """Execution-time home resolution (.env / config.yaml / scripts) follows
    the active profile, not the shared root — so a profile gateway runs its
    jobs with that profile's runtime config."""
    root = tmp_path / "hermes_home"
    profile_home = root / "profiles" / "coder"
    profile_home.mkdir(parents=True)

    _set_profile_env(monkeypatch, root, profile_home)

    import cron.scheduler as scheduler

    # The module-level test override must be clear so the dynamic path runs.
    monkeypatch.setattr(scheduler, "_hermes_home", None, raising=False)
    assert scheduler._get_hermes_home().resolve() == profile_home.resolve()
    assert scheduler._get_hermes_home().resolve() != root.resolve()


def test_cron_storage_unaffected_when_no_profile(tmp_path, monkeypatch):
    """With no profile (HERMES_HOME == root), the store is the root's cron dir
    — unchanged behavior for single-profile installs."""
    root = tmp_path / "hermes_home"
    root.mkdir(parents=True)

    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "_get_platform_default_hermes_home", lambda: root
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    import cron.jobs as jobs

    importlib.reload(jobs)
    try:
        assert jobs.HERMES_DIR.resolve() == root.resolve()
        assert jobs.JOBS_FILE.resolve() == (root / "cron" / "jobs.json").resolve()
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)


def test_named_script_resolution_ignores_overlapping_profile_contexts(
    tmp_path, monkeypatch
):
    root = tmp_path / "hermes_home"
    alpha_home = root / "profiles" / "alpha"
    beta_home = root / "profiles" / "beta"
    (alpha_home / "scripts").mkdir(parents=True)
    (beta_home / "scripts").mkdir(parents=True)
    _set_profile_env(monkeypatch, root, root)

    import cron.scheduler as scheduler
    import hermes_constants

    monkeypatch.setattr(scheduler, "_hermes_home", None)
    alpha_entered = threading.Event()
    beta_entered = threading.Event()
    release_beta = threading.Event()
    beta_exited = threading.Event()
    paths = {}
    errors = []

    def run_alpha():
        try:
            with scheduler._cron_profile_context({"id": "alpha", "profile": "alpha"}):
                alpha_entered.set()
                if not beta_entered.wait(2):
                    raise TimeoutError("beta did not enter its profile context")
                paths["alpha"] = scheduler._scripts_dir_for_job({"profile": "alpha"})
                release_beta.set()
                if not beta_exited.wait(2):
                    raise TimeoutError("beta did not leave its profile context")
        except BaseException as exc:
            errors.append(exc)

    def run_beta():
        try:
            if not alpha_entered.wait(2):
                raise TimeoutError("alpha did not enter its profile context")
            with scheduler._cron_profile_context({"id": "beta", "profile": "beta"}):
                beta_entered.set()
                if not release_beta.wait(2):
                    raise TimeoutError("alpha did not release beta")
            beta_exited.set()
        except BaseException as exc:
            errors.append(exc)

    alpha_thread = threading.Thread(target=run_alpha)
    beta_thread = threading.Thread(target=run_beta)
    alpha_thread.start()
    beta_thread.start()
    alpha_thread.join(3)
    beta_thread.join(3)

    assert not alpha_thread.is_alive()
    assert not beta_thread.is_alive()
    assert errors == []
    assert paths["alpha"].resolve() == (alpha_home / "scripts").resolve()
    assert scheduler._get_hermes_home().resolve() == root.resolve()


def test_named_run_blocks_unprofiled_prelude_and_restores_owner_home(
    tmp_path, monkeypatch
):
    root = tmp_path / "hermes_home"
    alpha_home = root / "profiles" / "alpha"
    alpha_home.mkdir(parents=True)
    _set_profile_env(monkeypatch, root, root)

    import cron.scheduler as scheduler
    from agent import secret_scope

    monkeypatch.setattr(scheduler, "_hermes_home", None)
    fresh_lock = scheduler._ReadWriteLock()
    monkeypatch.setattr(scheduler, "_profile_home_lock", fresh_lock)
    reader_attempted = threading.Event()
    original_acquire_read = fresh_lock.acquire_read

    def acquire_read():
        reader_attempted.set()
        original_acquire_read()

    monkeypatch.setattr(fresh_lock, "acquire_read", acquire_read)
    alpha_running = threading.Event()
    release_alpha = threading.Event()
    execution_homes = {}
    scope_homes = {}
    bookkeeping_homes = []
    errors = []

    def build_scope(home):
        scope_homes[threading.current_thread().name] = Path(home).resolve()
        return {"scope_home": str(Path(home).resolve())}

    def run_unscoped(job, *, defer_agent_teardown=None):
        name = threading.current_thread().name
        execution_homes[name] = scheduler._get_hermes_home().resolve()
        assert secret_scope.current_secret_scope()["scope_home"] == str(
            execution_homes[name]
        )
        if name == "alpha-run":
            alpha_running.set()
            if not release_alpha.wait(3):
                raise TimeoutError("alpha was not released")
        return True, "output", "final", None

    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", build_scope)
    monkeypatch.setattr(scheduler, "_run_job_unscoped", run_unscoped)
    monkeypatch.setattr(scheduler, "create_execution", lambda *a, **k: {"id": "exec"})
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *a, **k: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *a, **k: None)
    monkeypatch.setattr(
        scheduler,
        "save_job_output",
        lambda *a, **k: bookkeeping_homes.append(
            (threading.current_thread().name, scheduler._get_hermes_home().resolve())
        ) or "output.txt",
    )
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda *a, **k: bookkeeping_homes.append(
            (threading.current_thread().name, scheduler._get_hermes_home().resolve())
        ),
    )
    monkeypatch.setattr(scheduler, "finish_execution", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "_is_interrupted", lambda *a, **k: False)
    monkeypatch.setattr(scheduler, "_consume_interrupted_flag", lambda *a, **k: False)

    def invoke(job):
        try:
            scheduler.run_one_job(job)
        except BaseException as exc:
            errors.append(exc)

    alpha_thread = threading.Thread(
        name="alpha-run",
        target=invoke,
        args=({"id": "alpha", "profile": "alpha"},),
    )
    default_thread = threading.Thread(
        name="default-run",
        target=invoke,
        args=({"id": "default"},),
    )
    alpha_thread.start()
    assert alpha_running.wait(2)
    default_thread.start()
    assert reader_attempted.wait(2)
    assert "default-run" not in scope_homes
    assert "default-run" not in execution_homes
    release_alpha.set()
    alpha_thread.join(3)
    default_thread.join(3)

    assert errors == []
    assert execution_homes == {
        "alpha-run": alpha_home.resolve(),
        "default-run": root.resolve(),
    }
    assert scope_homes == execution_homes
    assert bookkeeping_homes
    assert all(home == root.resolve() for _, home in bookkeeping_homes)
    assert scheduler._get_hermes_home().resolve() == root.resolve()


def test_explicit_default_execution_switches_from_active_named_profile(
    tmp_path, monkeypatch
):
    root = tmp_path / "hermes_home"
    alpha_home = root / "profiles" / "alpha"
    alpha_home.mkdir(parents=True)
    _set_profile_env(monkeypatch, root, alpha_home)

    import cron.scheduler as scheduler
    import hermes_constants

    monkeypatch.setattr(scheduler, "_hermes_home", None)
    monkeypatch.setattr(scheduler, "_profile_home_lock", scheduler._ReadWriteLock())
    execution_homes = []

    def run_unscoped(job, *, defer_agent_teardown=None):
        execution_homes.append(
            (
                scheduler._get_hermes_home().resolve(),
                hermes_constants.get_hermes_home().resolve(),
            )
        )
        return True, "output", "final", None

    monkeypatch.setattr(scheduler, "_run_job_unscoped", run_unscoped)

    ambient_token = hermes_constants.set_hermes_home_override(alpha_home)
    try:
        result = scheduler.run_job({"id": "default", "profile": "default"})
    finally:
        hermes_constants.reset_hermes_home_override(ambient_token)

    assert result == (True, "output", "final", None)
    assert execution_homes == [(root.resolve(), root.resolve())]
    assert scheduler._get_hermes_home().resolve() == alpha_home.resolve()
    assert scheduler._job_needs_sequential_tick({"profile": "default"}) is True


def test_tick_routes_named_profiles_through_sequential_pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes_home"
    alpha_home = root / "profiles" / "alpha"
    beta_home = root / "profiles" / "beta"
    (alpha_home / "scripts").mkdir(parents=True)
    (beta_home / "scripts").mkdir(parents=True)
    _set_profile_env(monkeypatch, root, root)

    import cron.scheduler as scheduler
    from agent import secret_scope

    monkeypatch.setattr(scheduler, "_hermes_home", None)
    monkeypatch.setattr(scheduler, "_profile_home_lock", scheduler._ReadWriteLock())
    jobs = [
        {"id": "alpha-tick", "profile": "alpha"},
        {"id": "beta-tick", "profile": "beta"},
    ]
    records = []

    def build_scope(home):
        records.append(("scope", threading.current_thread().name, Path(home).resolve()))
        return {}

    def run_unscoped(job, *, defer_agent_teardown=None):
        records.append(
            (
                "execute",
                threading.current_thread().name,
                scheduler._get_hermes_home().resolve(),
                scheduler._scripts_dir_for_job(job).resolve(),
            )
        )
        return True, "output", "final", None

    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", build_scope)
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: jobs)
    monkeypatch.setattr(scheduler, "advance_next_run", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler, "create_execution", lambda job_id, **k: {"id": job_id})
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *a, **k: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "_run_job_unscoped", run_unscoped)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *a, **k: "output.txt")
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "finish_execution", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "_is_interrupted", lambda *a, **k: False)
    monkeypatch.setattr(scheduler, "_consume_interrupted_flag", lambda *a, **k: False)

    assert scheduler.tick(verbose=False, sync=True) == 2

    scopes = [record for record in records if record[0] == "scope"]
    executions = [record for record in records if record[0] == "execute"]
    assert [record[2] for record in scopes] == [alpha_home.resolve(), beta_home.resolve()]
    assert [record[2] for record in executions] == [alpha_home.resolve(), beta_home.resolve()]
    assert [record[3] for record in executions] == [
        (alpha_home / "scripts").resolve(),
        (beta_home / "scripts").resolve(),
    ]
    assert all(record[1].startswith("cron-seq") for record in scopes + executions)
