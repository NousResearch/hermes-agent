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


def test_cron_session_persists_only_to_active_profile_home(tmp_path, monkeypatch):
    """The multiplex ticker persists a cron session in its owning profile."""
    import cron.scheduler as scheduler
    import cron.jobs as jobs
    from cron.scheduler_provider import InProcessCronScheduler
    import hermes_constants
    import hermes_state

    SessionDB = hermes_state.SessionDB

    user_home = tmp_path / "user"
    default_home = tmp_path / "default"
    secondary_home = tmp_path / "profiles" / "secondary"
    default_home.mkdir(parents=True)
    secondary_home.mkdir(parents=True)

    # Match the multiplex gateway: the process belongs to the default profile,
    # while the ticker scopes each dispatched run to the profile owning it.
    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("HERMES_CRON_SESSION_DB_TIMEOUT", "10")
    monkeypatch.setattr(scheduler, "_hermes_home", None)
    # The autouse test fixture deliberately pins DEFAULT_DB_PATH to its own
    # sandbox when hermes_state was imported during collection. Neutralize that
    # test-only escape hatch so this test exercises production's dynamic,
    # ContextVar-aware default resolution.
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache", lambda: None
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "test-key",
            "base_url": None,
            "provider": "",
            "requested_provider": "",
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )
    monkeypatch.setattr(
        "hermes_constants.resolve_reasoning_config", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: [])
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(
        scheduler, "_guard_job_credential_exfil", lambda _job: None
    )
    monkeypatch.setattr(
        jobs, "_compute_provider_model_snapshots", lambda **_kwargs: (None, None)
    )

    # Create the job in the secondary profile's real cron store, then make it
    # due so the provider's normal multiplex tick discovers and dispatches it.
    home_token = hermes_constants.set_hermes_home_override(secondary_home)
    try:
        with jobs.use_cron_store(secondary_home):
            job = jobs.create_job(
                prompt="Reply only: OK.",
                schedule="every 1h",
                name="Profile Session",
                model="test-model",
                deliver="local",
                enabled_toolsets=["no_mcp"],
            )
            stored_jobs = jobs.load_jobs()
            stored_jobs[0]["next_run_at"] = "2000-01-01T00:00:00+00:00"
            jobs.save_jobs(stored_jobs)
    finally:
        hermes_constants.reset_hermes_home_override(home_token)

    class _ObservableStop:
        """Expose when one complete multiplex tick reaches its wait point."""

        def __init__(self):
            self._stop = threading.Event()
            self.after_cycle = threading.Event()

        def is_set(self):
            return self._stop.is_set()

        def wait(self, timeout):
            # start() reaches this only after tick(sync=False) returns and every
            # temporary profile override for the cycle has been reset.
            self.after_cycle.set()
            return self._stop.wait(timeout)

        def set(self):
            self._stop.set()

    stop = _ObservableStop()
    agent_done = threading.Event()

    # Keep the real SessionDB implementation, but delay its construction until
    # the provider has returned from the fire-and-forget tick and reset the
    # calling thread's profile scope. The only remaining profile state is what
    # the executor chain propagated into the job worker.
    RealSessionDB = hermes_state.SessionDB

    def _session_db_after_multiplex_cycle(*args, **kwargs):
        assert stop.after_cycle.wait(10), "multiplex tick did not finish its cycle"
        return RealSessionDB(*args, **kwargs)

    monkeypatch.setattr(hermes_state, "SessionDB", _session_db_after_multiplex_cycle)

    created_session_ids: list[str] = []

    class _PersistingCronAgent:
        def __init__(self, *args, session_db, session_id, **kwargs):
            assert session_db is not None
            self.session_db = session_db
            self.session_id = session_id

        def run_conversation(self, _prompt):
            try:
                self.session_db.create_session(self.session_id, "cron")
                self.session_db.append_message(
                    self.session_id, "assistant", "profile isolation sentinel"
                )
                created_session_ids.append(self.session_id)
                return {
                    "completed": True,
                    "failed": False,
                    "final_response": "ok",
                    "turn_exit_reason": "",
                }
            finally:
                agent_done.set()

        def close(self):
            pass

    monkeypatch.setattr("run_agent.AIAgent", _PersistingCronAgent)

    provider = InProcessCronScheduler()
    monkeypatch.setattr(provider, "recover_interrupted", lambda: 0)
    monkeypatch.setattr(
        jobs, "record_ticker_heartbeat", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(jobs, "clear_ticker_error", lambda: None)
    monkeypatch.setattr(
        jobs, "record_ticker_error", lambda *_args, **_kwargs: None
    )

    # The cron pools are process-global. Start clean, and join the dispatched
    # fire-and-forget future before querying either database or leaving the test.
    scheduler._shutdown_parallel_pool()
    ticker = threading.Thread(
        target=provider.start,
        args=(stop,),
        kwargs={
            "interval": 60,
            "profile_homes": [
                ("default", default_home),
                ("secondary", secondary_home),
            ],
        },
        daemon=True,
    )
    ticker.start()
    try:
        assert stop.after_cycle.wait(10), "multiplex ticker did not finish a cycle"
        assert agent_done.wait(20), "dispatched cron job did not persist its session"
    finally:
        stop.set()
        ticker.join(timeout=10)
        scheduler._shutdown_parallel_pool()
        scheduler.release_running_job(job["id"])

    assert not ticker.is_alive(), "multiplex ticker did not stop"
    assert len(created_session_ids) == 1
    session_id = created_session_ids[0]

    def _contains_session(home: Path) -> bool:
        db_path = home / "state.db"
        if not db_path.exists():
            return False
        # Some cron setup paths may create an empty state.db placeholder. Open
        # normally so the public SessionDB interface initializes its schema
        # before we ask whether the cron session is retrievable there.
        db = SessionDB(db_path)
        try:
            return db.get_session(session_id) is not None
        finally:
            db.close()

    assert {
        "default": _contains_session(default_home),
        "secondary": _contains_session(secondary_home),
    } == {"default": False, "secondary": True}
