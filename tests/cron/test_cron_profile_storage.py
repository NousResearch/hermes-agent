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


# ---------------------------------------------------------------------------
# Per-job profile EXECUTION scoping (#32091 follow-up).
#
# The storage half of #32091 (above) moved every profile's jobs into one shared
# root store. But a job must still EXECUTE under its owning profile's
# environment (.env / config.yaml / credentials) — not whichever profile's
# ticker picks it up. These tests cover the execution-scoping half.
# ---------------------------------------------------------------------------


def _profile_env(tmp_path, monkeypatch, active="default"):
    """Set up a root home with a 'donna' profile dir and point the platform
    default at it. Returns (root, donna_home). ``active`` selects which
    HERMES_HOME the process runs under."""
    root = tmp_path / "hermes_home"
    (root / "cron").mkdir(parents=True)
    donna_home = root / "profiles" / "donna"
    (donna_home / "cron").mkdir(parents=True)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home",
                        lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(root if active == "default" else donna_home))
    return root, donna_home


def test_create_job_autocaptures_active_profile(tmp_path, monkeypatch):
    """A job created from inside a profile session is tagged with that profile,
    so the scheduler can later scope its execution back to it."""
    root, donna_home = _profile_env(tmp_path, monkeypatch, active="donna")
    import cron.jobs as jobs
    importlib.reload(jobs)
    try:
        job = jobs.create_job(prompt="audit", schedule="every 1h", name="a")
        # auto-captured from the active (donna) session
        assert job["profile"] == "donna"
        # and it landed in the SHARED ROOT store, not donna's profile-local one
        assert jobs.JOBS_FILE.resolve() == (root / "cron" / "jobs.json").resolve()
        assert jobs.JOBS_FILE.exists()
        assert not (donna_home / "cron" / "jobs.json").exists()
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)


def test_create_job_explicit_profile_override(tmp_path, monkeypatch):
    """An explicit profile= wins over the auto-captured active profile."""
    root, donna_home = _profile_env(tmp_path, monkeypatch, active="default")
    (root / "profiles" / "ops" / "cron").mkdir(parents=True)
    import cron.jobs as jobs
    importlib.reload(jobs)
    try:
        job = jobs.create_job(prompt="x", schedule="every 2h", profile="ops")
        assert job["profile"] == "ops"
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)


def test_resolve_profile_home_maps_names(tmp_path, monkeypatch):
    """resolve_profile_home maps default/named profiles to homes and returns
    None for a missing profile."""
    root, donna_home = _profile_env(tmp_path, monkeypatch, active="default")
    import cron.jobs as jobs
    importlib.reload(jobs)
    try:
        assert jobs.resolve_profile_home("default").resolve() == root.resolve()
        assert jobs.resolve_profile_home("").resolve() == root.resolve()
        assert jobs.resolve_profile_home("donna").resolve() == donna_home.resolve()
        assert jobs.resolve_profile_home("ghost") is None
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)


def test_normalize_backfills_legacy_profile_to_default(tmp_path, monkeypatch):
    """A pre-feature job with no profile field reads back as 'default'."""
    import cron.jobs as jobs
    legacy = {"id": "l1", "name": "old", "prompt": "x",
              "schedule": {"kind": "interval", "minutes": 60}}
    assert jobs._normalize_job_record(legacy)["profile"] == "default"


def test_run_job_scopes_execution_to_job_profile(tmp_path, monkeypatch):
    """The decisive test: a ticker running as the ROOT profile executes a
    job tagged profile='donna' with HERMES_HOME pointed at donna's home
    (both the env var and the in-process override), then restores the
    ticker's env afterward."""
    from unittest.mock import MagicMock, patch
    root, donna_home = _profile_env(tmp_path, monkeypatch, active="default")
    (donna_home / "config.yaml").write_text("model:\n  default: openrouter/test\n")

    import hermes_constants
    import cron.jobs as jobs
    import cron.scheduler as sched
    importlib.reload(jobs)
    importlib.reload(sched)

    captured = {}

    def fake_run_conversation(prompt, *a, **k):
        captured["env"] = os.environ.get("HERMES_HOME")
        captured["override"] = hermes_constants.get_hermes_home_override()
        captured["resolved"] = str(hermes_constants.get_hermes_home())
        return {"final_response": "done", "completed": True, "failed": False,
                "turn_exit_reason": "text_response(finish_reason=stop)"}

    job = {"id": "j-donna", "name": "donna-audit", "prompt": "audit",
           "profile": "donna", "schedule": {"kind": "interval", "minutes": 60},
           "deliver": "local", "model": "openrouter/test"}

    before = os.environ.get("HERMES_HOME")
    try:
        fake_agent = MagicMock()
        fake_agent.run_conversation.side_effect = fake_run_conversation
        with patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=MagicMock()), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"api_key": "k", "base_url": "https://x/v1",
                                 "provider": "openrouter", "api_mode": "chat_completions"}), \
             patch("run_agent.AIAgent", return_value=fake_agent):
            success, output, final, err = sched.run_job(job)

        assert success is True, (success, err)
        # During execution the job ran AS donna:
        assert captured["env"] == str(donna_home)
        assert captured["override"] == str(donna_home)
        assert captured["resolved"] == str(donna_home)
        # After the job, the ticker's HERMES_HOME is restored (no leak):
        assert os.environ.get("HERMES_HOME") == before
    finally:
        monkeypatch.undo()
        importlib.reload(jobs)
        importlib.reload(sched)
