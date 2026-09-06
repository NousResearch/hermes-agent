"""Regression tests for dashboard cron job profile routing."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest
from fastapi import HTTPException
import hermes_cli.web_models as _web_models
import hermes_cli.web_routers.cron as _rt_cron
import hermes_cli.web_server_cron as _web_server_cron


@pytest.fixture()
def isolated_profiles(tmp_path, monkeypatch):
    """Give profile discovery an isolated default home with one named profile."""
    from hermes_cli import profiles

    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_alpha"

    for home in (default_home, worker_home):
        (home / "cron").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker_alpha": worker_home}


def test_standalone_jobs_remain_discoverable_for_dashboard_management(isolated_profiles):
    home = isolated_profiles["worker_alpha"]
    (home / "config.yaml").write_text("gateway:\n  standalone: true\n")
    job = _web_server_cron._call_cron_for_profile(
        "worker_alpha", "create_job", prompt="local job", schedule="every 1h",
    )
    assert isinstance(job, dict)
    assert "worker_alpha" in {p["name"] for p in _web_server_cron._cron_profile_dicts()}
    assert _web_server_cron._find_cron_job_profile(job["id"]) == "worker_alpha"




def test_fire_cron_job_scopes_store_and_runtime_home_together(
    isolated_profiles,
    monkeypatch,
):
    """A profile fire must execute and persist under the same profile home."""
    from cron import jobs as cron_jobs
    from cron import scheduler

    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    default_home = isolated_profiles["default"]
    worker_home = isolated_profiles["worker_alpha"]
    monkeypatch.setattr(scheduler, "_hermes_home", None)
    captured = {}

    class RecordingProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None):
            captured["job_id"] = job_id
            captured["runtime_home"] = scheduler._get_hermes_home()
            captured["jobs_file"] = cron_jobs._current_cron_store().jobs_file
            return True

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: RecordingProvider(),
    )

    outer_token = set_hermes_home_override(default_home)
    try:
        assert _web_server_cron._fire_cron_job_for_profile("worker_alpha", "worker-job") is True
        assert captured == {
            "job_id": "worker-job",
            "runtime_home": worker_home,
            "jobs_file": worker_home / "cron" / "jobs.json",
        }
        assert scheduler._get_hermes_home() == default_home
    finally:
        reset_hermes_home_override(outer_token)


def test_create_registers_scheduler_inside_target_profile(
    isolated_profiles,
    monkeypatch,
):
    """Dashboard create must resolve and register under the selected profile."""
    from cron import jobs as cron_jobs
    from cron.scheduler_provider import CronScheduler
    from hermes_constants import get_hermes_home

    worker_home = isolated_profiles["worker_alpha"]
    captured = {}

    class RecordingProvider(CronScheduler):
        @property
        def name(self):
            return "recording"

        def start(self, stop_event, **kw):
            pass

        def register_job(self, job):
            captured["job"] = job
            captured["runtime_home"] = get_hermes_home()
            captured["jobs_file"] = cron_jobs._current_cron_store().jobs_file

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: RecordingProvider(),
    )

    job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="named-profile-job",
    )

    assert captured["job"]["id"] == job["id"]
    assert captured["runtime_home"] == worker_home
    assert captured["jobs_file"] == worker_home / "cron" / "jobs.json"
    assert job["profile"] == "worker_alpha"


def test_dashboard_create_reports_saved_but_unregistered(
    isolated_profiles,
    monkeypatch,
):
    """Dashboard callers can distinguish persistence from remote registration."""
    from cron.scheduler import CronSchedulerRegistrationError

    job = {"id": "saved-job", "name": "saved job"}
    failure = CronSchedulerRegistrationError(
        job,
        RuntimeError("private callback URL and token"),
    )

    def fail_create(*args, **kwargs):
        raise failure

    monkeypatch.setattr(_web_server_cron, "_call_cron_for_profile", fail_create)

    with pytest.raises(HTTPException) as exc_info:
        _web_server_cron._create_cron_job_sync(
            _web_models.CronJobCreate(
                prompt="managed by named profile",
                schedule="every 1h",
                name="named-profile-job",
            ),
            profile="worker_alpha",
        )

    assert exc_info.value.status_code == 424
    assert exc_info.value.detail == {
        "error": str(failure),
        "job_id": "saved-job",
        "job_saved": True,
        "scheduler_registered": False,
        "retry_create": False,
    }
    assert "private callback URL and token" not in str(exc_info.value.detail)


def test_notify_cron_provider_scopes_store_and_runtime_home_together(
    isolated_profiles,
    monkeypatch,
):
    """Provider reconciliation must observe the mutated profile, not default."""
    from cron import jobs as cron_jobs
    from cron import scheduler

    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    default_home = isolated_profiles["default"]
    worker_home = isolated_profiles["worker_alpha"]
    monkeypatch.setattr(scheduler, "_hermes_home", None)
    monkeypatch.setattr(
        _web_server_cron,
        "_cron_profile_dicts",
        lambda: [{"name": "worker_alpha"}],
    )
    captured = {}

    class RecordingProvider:
        def on_jobs_changed(self):
            captured["runtime_home"] = scheduler._get_hermes_home()
            captured["jobs_file"] = cron_jobs._current_cron_store().jobs_file

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: RecordingProvider(),
    )

    outer_token = set_hermes_home_override(default_home)
    try:
        _web_server_cron._notify_cron_provider_for_profile("worker_alpha")
        assert captured == {
            "runtime_home": worker_home,
            "jobs_file": worker_home / "cron" / "jobs.json",
        }
        assert scheduler._get_hermes_home() == default_home
    finally:
        reset_hermes_home_override(outer_token)


def test_notify_cron_provider_failure_is_best_effort(
    isolated_profiles,
    monkeypatch,
):

    class FailNotifyProvider:
        @property
        def name(self):
            return "fail-notify"

        def register_job(self, job):
            return None

        def on_jobs_changed(self):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: FailNotifyProvider(),
    )

    created = _web_server_cron._mutate_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="survives provider failure",
        schedule="every 1h",
        name="best-effort-notify",
    )

    assert created["profile"] == "worker_alpha"
    assert created["name"] == "best-effort-notify"


def test_external_provider_reconcile_fails_closed_with_multiple_profiles(
    isolated_profiles,
    monkeypatch,
):
    """Multi-profile dashboard + external provider: the unscoped reconcile
    must NOT run — its orphan cleanup would disarm the other profiles'
    armed one-shots in the shared NAS registry. The mutation itself still
    succeeds (fail-closed only skips the remote converge)."""
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_hermes_home", None)
    monkeypatch.setattr(
        _web_server_cron,
        "_cron_profile_dicts",
        lambda: [{"name": "default"}, {"name": "worker_alpha"}],
    )
    notified = []

    class ExternalProvider:
        @property
        def name(self):
            return "chronos"

        def register_job(self, job):
            return None

        def on_jobs_changed(self):
            notified.append(True)

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: ExternalProvider(),
    )

    created = _web_server_cron._mutate_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="must not disarm siblings",
        schedule="every 1h",
        name="multi-profile-guard",
    )

    assert created["profile"] == "worker_alpha"
    assert notified == [], (
        "external provider reconcile must stay fail-closed on a "
        "multi-profile dashboard"
    )


def test_builtin_provider_hook_still_fires_with_multiple_profiles(
    isolated_profiles,
    monkeypatch,
):
    """The built-in provider re-reads jobs.json per tick — its hook is a
    safe no-op and must NOT be blocked by the multi-profile guard."""
    from cron import scheduler
    from cron.scheduler_provider import InProcessCronScheduler

    monkeypatch.setattr(scheduler, "_hermes_home", None)
    monkeypatch.setattr(
        _web_server_cron,
        "_cron_profile_dicts",
        lambda: [{"name": "default"}, {"name": "worker_alpha"}],
    )
    notified = []

    class BuiltinProbe(InProcessCronScheduler):
        def on_jobs_changed(self):
            notified.append(True)

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: BuiltinProbe(),
    )

    created = _web_server_cron._mutate_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="builtin notify",
        schedule="every 1h",
        name="builtin-notify",
    )

    assert created["profile"] == "worker_alpha"
    assert notified == [True]


def test_profile_call_cannot_retarget_ticker_store_mid_write(
    isolated_profiles,
    monkeypatch,
):
    """A dashboard profile call must not redirect a concurrent ticker save."""
    from cron import jobs as cron_jobs

    default_cron = isolated_profiles["default"] / "cron"
    worker_cron = isolated_profiles["worker_alpha"] / "cron"
    default_file = default_cron / "jobs.json"
    worker_file = worker_cron / "jobs.json"
    default_job = {
        "id": "default-job",
        "name": "default job",
        "schedule": {"kind": "interval", "minutes": 60},
        "next_run_at": "2026-07-09T00:00:00+00:00",
    }
    worker_job = {
        "id": "worker-job",
        "name": "worker job",
        "schedule": {"kind": "interval", "minutes": 60},
        "next_run_at": "2026-07-09T00:00:00+00:00",
    }
    default_file.write_text(json.dumps({"jobs": [default_job]}), encoding="utf-8")
    worker_file.write_text(json.dumps({"jobs": [worker_job]}), encoding="utf-8")

    monkeypatch.setattr(cron_jobs, "CRON_DIR", default_cron)
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", default_file)
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", default_cron / "output")
    monkeypatch.setattr(
        cron_jobs,
        "compute_next_run",
        lambda _schedule, _last_run_at=None: "2026-07-10T00:00:00+00:00",
    )

    ticker_loaded = threading.Event()
    release_ticker = threading.Event()
    profile_entered = threading.Event()
    ticker_done = threading.Event()
    ticker_thread = threading.local()
    original_load_jobs = cron_jobs.load_jobs

    def blocking_load_jobs():
        loaded = original_load_jobs()
        if getattr(ticker_thread, "active", False):
            ticker_loaded.set()
            assert release_ticker.wait(5), "profile call did not enter in time"
        return loaded

    def hold_profile_call():
        profile_entered.set()
        assert ticker_done.wait(5), "ticker did not finish in time"
        return True

    def run_ticker_write():
        ticker_thread.active = True
        try:
            return cron_jobs.advance_next_run("default-job")
        finally:
            ticker_done.set()

    monkeypatch.setattr(cron_jobs, "load_jobs", blocking_load_jobs)
    monkeypatch.setattr(cron_jobs, "_hold_profile_call", hold_profile_call, raising=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        ticker_future = pool.submit(run_ticker_write)
        assert ticker_loaded.wait(5), "ticker did not load the default store"
        profile_future = pool.submit(
            _web_server_cron._call_cron_for_profile,
            "worker_alpha",
            "_hold_profile_call",
        )
        assert profile_entered.wait(5), "profile call did not retarget its store"
        release_ticker.set()
        assert ticker_future.result(timeout=5) is True
        assert profile_future.result(timeout=5) is True

    default_saved = json.loads(default_file.read_text(encoding="utf-8"))["jobs"]
    worker_saved = json.loads(worker_file.read_text(encoding="utf-8"))["jobs"]
    assert [job["id"] for job in worker_saved] == ["worker-job"]
    assert [job["id"] for job in default_saved] == ["default-job"]
    assert default_saved[0]["next_run_at"] == "2026-07-10T00:00:00+00:00"






@pytest.mark.asyncio
async def test_cron_mutation_without_profile_finds_named_profile_job(isolated_profiles):
    from fastapi import Request

    worker_job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="named-profile-job",
    )

    # pause_cron_job is admin-gated (_require_dashboard_admin) since the Mini
    # App tiered access control landed; a bare Request with no
    # request.state.token_principal set resolves to the cookie/session
    # caller's unrestricted scope, same as this direct call always got before
    # that gate existed.
    request = Request(scope={"type": "http", "headers": []})
    paused = await _rt_cron.pause_cron_job(request, worker_job["id"])
    assert paused["profile"] == "worker_alpha"
    assert paused["enabled"] is False

    default_jobs = await _rt_cron.list_cron_jobs(profile="default")
    worker_jobs = await _rt_cron.list_cron_jobs(profile="worker_alpha")

    assert default_jobs == []
    assert len(worker_jobs) == 1
    assert worker_jobs[0]["id"] == worker_job["id"]
    assert worker_jobs[0]["enabled"] is False


@pytest.mark.asyncio
async def test_dashboard_cron_mutations_notify_selected_profile_provider(
    isolated_profiles,
    monkeypatch,
):
    from fastapi import Request

    # pause/resume/delete are admin-gated (_require_dashboard_admin) since the
    # Mini App tiered access control landed; a bare Request with no
    # request.state.token_principal set resolves to the cookie/session
    # caller's unrestricted scope, same as these direct calls always got
    # before that gate existed.
    request = Request(scope={"type": "http", "headers": []})

    notified_profiles = []
    monkeypatch.setattr(
        _web_server_cron,
        "_notify_cron_provider_for_profile",
        notified_profiles.append,
    )

    created = await _rt_cron.create_cron_job(
        _web_models.CronJobCreate(
            prompt="managed by named profile",
            schedule="every 1h",
            name="provider-notify-job",
        ),
        profile="worker_alpha",
    )
    await _rt_cron.update_cron_job(
        created["id"],
        _web_models.CronJobUpdate(updates={"name": "provider-notify-job-updated"}),
        profile="worker_alpha",
    )
    await _rt_cron.pause_cron_job(request, created["id"], profile="worker_alpha")
    await _rt_cron.resume_cron_job(request, created["id"], profile="worker_alpha")
    await _rt_cron.delete_cron_job(request, created["id"], profile="worker_alpha")

    assert notified_profiles == ["worker_alpha"] * 5


@pytest.mark.asyncio
async def test_blueprint_instantiation_notifies_selected_profile_provider(
    isolated_profiles,
    monkeypatch,
):

    notified_profiles = []
    monkeypatch.setattr(
        _web_server_cron,
        "_notify_cron_provider_for_profile",
        notified_profiles.append,
    )

    created = await _rt_cron.instantiate_blueprint(
        _web_models.AutomationBlueprintInstantiate(
            blueprint="morning-brief",
            values={"time": "07:30", "deliver": "local"},
        ),
        profile="worker_alpha",
    )

    assert created["profile"] == "worker_alpha"
    assert notified_profiles == ["worker_alpha"]


@pytest.mark.asyncio
async def test_trigger_cron_job_fires_only_selected_job_and_returns_refreshed_state(
    isolated_profiles,
    monkeypatch,
):
    from cron import jobs as cron_jobs

    selected = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="run immediately",
        schedule="every 1h",
        name="selected-trigger-job",
    )
    sibling = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="leave scheduled",
        schedule="every 1h",
        name="sibling-job",
    )
    fired = []

    class RecordingProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None, force=False, manual=False):
            fired.append(
                {
                    "job_id": job_id,
                    "jobs_file": cron_jobs._current_cron_store().jobs_file,
                    "force": force,
                    "manual": manual,
                }
            )
            cron_jobs.mark_job_run(job_id, success=True)
            return True

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: RecordingProvider(),
    )
    monkeypatch.setattr(
        cron_jobs,
        "trigger_job",
        lambda _job_id: (_ for _ in ()).throw(
            AssertionError("manual fire must not expose the job to the ticker first")
        ),
    )

    from fastapi import Request

    request = Request(scope={"type": "http", "headers": []})
    triggered = await _rt_cron.trigger_cron_job(
        request,
        selected["id"],
        profile="worker_alpha",
    )

    assert fired == [
        {
            "job_id": selected["id"],
            "jobs_file": isolated_profiles["worker_alpha"] / "cron" / "jobs.json",
            "force": False,
            # Off-tick run-now: the claim must not stamp next_run_at as the occurrence (#104790).
            "manual": True,
        }
    ]
    assert triggered["last_status"] == "ok"
    assert triggered["last_run_at"] is not None
    untouched = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "get_job",
        sibling["id"],
    )
    assert untouched["last_run_at"] is None


@pytest.mark.asyncio
async def test_trigger_cron_job_reports_lost_claim_as_conflict(
    isolated_profiles,
    monkeypatch,
):

    job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="already running",
        schedule="every 1h",
        name="claimed-trigger-job",
    )

    class ClaimLostProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None, force=False):
            return False

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: ClaimLostProvider(),
    )

    from fastapi import Request

    request = Request(scope={"type": "http", "headers": []})
    with pytest.raises(HTTPException) as exc:
        await _rt_cron.trigger_cron_job(request, job["id"], profile="worker_alpha")

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_trigger_cron_job_forces_paused_job_atomically(
    isolated_profiles,
    monkeypatch,
):
    from cron import jobs as cron_jobs

    job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="resume me",
        schedule="every 1h",
        name="paused-trigger-job",
    )
    _web_server_cron._call_cron_for_profile("worker_alpha", "pause_job", job["id"])
    observed = {}

    class ForceProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None, force=False):
            observed["force"] = force
            assert cron_jobs.claim_job_for_fire(job_id, force=force) is True
            cron_jobs.mark_job_run(job_id, success=True)
            return True

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: ForceProvider(),
    )

    from fastapi import Request

    request = Request(scope={"type": "http", "headers": []})
    triggered = await _rt_cron.trigger_cron_job(
        request,
        job["id"],
        profile="worker_alpha",
    )

    assert observed["force"] is True
    assert triggered["enabled"] is True
    assert triggered["state"] == "scheduled"
    assert triggered["last_status"] == "ok"


@pytest.mark.asyncio
async def test_trigger_paused_job_rejects_legacy_provider_without_mutating_job(
    isolated_profiles,
    monkeypatch,
):
    from fastapi import HTTPException

    job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="stay paused",
        schedule="every 1h",
        name="legacy-paused-trigger-job",
    )
    _web_server_cron._call_cron_for_profile("worker_alpha", "pause_job", job["id"])
    calls = []

    class LegacyProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None):
            calls.append(job_id)
            return True

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: LegacyProvider(),
    )

    from fastapi import Request

    request = Request(scope={"type": "http", "headers": []})
    with pytest.raises(HTTPException) as exc:
        await _rt_cron.trigger_cron_job(request, job["id"], profile="worker_alpha")

    assert exc.value.status_code == 409
    assert calls == []
    persisted = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "get_job",
        job["id"],
    )
    assert persisted["state"] == "paused"
    assert persisted["enabled"] is False


@pytest.mark.asyncio
async def test_trigger_cron_job_returns_refreshed_execution_failure(
    isolated_profiles,
    monkeypatch,
):
    from cron import jobs as cron_jobs

    job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="fail visibly",
        schedule="every 1h",
        name="failed-trigger-job",
    )

    class FailedProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None, force=False):
            cron_jobs.mark_job_run(job_id, success=False, error="expected failure")
            return False

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: FailedProvider(),
    )

    from fastapi import Request

    request = Request(scope={"type": "http", "headers": []})
    triggered = await _rt_cron.trigger_cron_job(
        request,
        job["id"],
        profile="worker_alpha",
    )

    assert triggered["last_status"] == "error"
    assert triggered["last_error"] == "expected failure"


@pytest.mark.asyncio
async def test_trigger_cron_job_returns_completed_snapshot_for_retained_oneshot(
    isolated_profiles,
    monkeypatch,
):
    from cron import jobs as cron_jobs

    job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="run once",
        schedule="in 30m",
        name="completed-trigger-job",
    )

    class SuccessfulProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None, force=False):
            cron_jobs.mark_job_run(job_id, success=True)
            return True

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: SuccessfulProvider(),
    )

    from fastapi import Request

    request = Request(scope={"type": "http", "headers": []})
    triggered = await _rt_cron.trigger_cron_job(
        request,
        job["id"],
        profile="worker_alpha",
    )

    assert triggered["state"] == "completed"
    assert triggered["enabled"] is False
    # Completed one-shots are retained for the retention window (#80624) with
    # their terminal status inspectable — the trigger response is the real
    # record, not a synthetic pre-removal snapshot.
    assert triggered["last_status"] == "ok"
    assert triggered["last_run_at"] is not None
    retained = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "get_job",
        job["id"],
    )
    assert retained is not None
    assert retained["state"] == "completed"







@pytest.mark.asyncio
async def test_update_cron_job_normalizes_dashboard_core_fields(isolated_profiles, tmp_path):

    scripts_dir = isolated_profiles["worker_alpha"] / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "collect.py").write_text("print('ok')\n", encoding="utf-8")
    job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="normalizes-dashboard-fields",
    )

    updated = await _rt_cron.update_cron_job(
        job["id"],
        _web_models.CronJobUpdate(
            updates={
                "base_url": "https://example.invalid/v1/",
                "script": str(scripts_dir / "collect.py"),
                "context_from": "",
                "no_agent": True,
            }
        ),
        profile="worker_alpha",
    )

    assert updated["base_url"] == "https://example.invalid/v1"
    assert updated["script"] == "collect.py"
    assert updated["context_from"] is None
    assert updated["no_agent"] is True


@pytest.mark.asyncio
async def test_create_cron_job_rejects_script_outside_profile_scripts(
    isolated_profiles, tmp_path
):

    outside = tmp_path / "outside.py"
    outside.write_text("print('nope')\n", encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        await _rt_cron.create_cron_job(
            _web_models.CronJobCreate(
                schedule="every 1h",
                script=str(outside),
                no_agent=True,
            ),
            profile="worker_alpha",
        )

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_create_cron_job_rejects_empty_agent_job(isolated_profiles):

    with pytest.raises(HTTPException) as exc:
        await _rt_cron.create_cron_job(
            _web_models.CronJobCreate(schedule="every 1h"),
            profile="worker_alpha",
        )

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_update_cron_job_no_agent_reuses_existing_script(isolated_profiles):

    scripts_dir = isolated_profiles["worker_alpha"] / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "collect.py").write_text("print('ok')\n", encoding="utf-8")

    job = await _rt_cron.create_cron_job(
        _web_models.CronJobCreate(
            schedule="every 1h",
            script=str(scripts_dir / "collect.py"),
        ),
        profile="worker_alpha",
    )

    updated = await _rt_cron.update_cron_job(
        job["id"],
        _web_models.CronJobUpdate(updates={"no_agent": True}),
        profile="worker_alpha",
    )

    assert updated["no_agent"] is True
    assert updated["script"] == "collect.py"


@pytest.mark.asyncio
async def test_dashboard_cron_rejects_missing_context_from(isolated_profiles):

    with pytest.raises(HTTPException) as create_exc:
        await _rt_cron.create_cron_job(
            _web_models.CronJobCreate(
                prompt="process missing upstream",
                schedule="every 1h",
                context_from=["missing-job-id"],
            ),
            profile="worker_alpha",
        )

    assert create_exc.value.status_code == 400
    assert "missing-job-id" in create_exc.value.detail

    job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="context-update-target",
    )

    with pytest.raises(HTTPException) as update_exc:
        await _rt_cron.update_cron_job(
            job["id"],
            _web_models.CronJobUpdate(
                updates={
                    "context_from": ["missing-job-id"],
                }
            ),
            profile="worker_alpha",
        )

    assert update_exc.value.status_code == 400
    assert "missing-job-id" in update_exc.value.detail


@pytest.mark.asyncio
async def test_update_cron_job_rejects_id_mutation(isolated_profiles, monkeypatch):
    """Dashboard surfaces a 400 (not a 500 or silent rename) when an
    id-mutation attempt is rejected by cron/jobs.update_job."""

    notified_profiles = []
    monkeypatch.setattr(
        _web_server_cron,
        "_notify_cron_provider_for_profile",
        notified_profiles.append,
    )
    worker_job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="immutable-id-job",
    )

    with pytest.raises(HTTPException) as exc:
        await _rt_cron.update_cron_job(
            worker_job["id"],
            _web_models.CronJobUpdate(updates={"id": "../escape"}),
            profile="worker_alpha",
        )

    assert exc.value.status_code == 400
    assert "id" in exc.value.detail
    assert notified_profiles == []
    worker_jobs = await _rt_cron.list_cron_jobs(profile="worker_alpha")
    assert [job["id"] for job in worker_jobs] == [worker_job["id"]]


@pytest.mark.asyncio
async def test_cron_delete_with_profile_deletes_only_target_profile(isolated_profiles):

    default_job = _web_server_cron._call_cron_for_profile(
        "default",
        "create_job",
        prompt="same-ish default",
        schedule="every 1h",
        name="shared-name",
    )
    worker_job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="same-ish worker",
        schedule="every 1h",
        name="shared-name-worker",
    )

    from fastapi import Request

    request = Request(scope={"type": "http", "headers": []})
    deleted = await _rt_cron.delete_cron_job(request, worker_job["id"], profile="worker_alpha")
    assert deleted == {"ok": True}

    remaining_default = await _rt_cron.list_cron_jobs(profile="default")
    remaining_worker = await _rt_cron.list_cron_jobs(profile="worker_alpha")
    assert [job["id"] for job in remaining_default] == [default_job["id"]]
    assert remaining_worker == []


@pytest.mark.asyncio
async def test_cron_profile_validation_errors(isolated_profiles):

    with pytest.raises(HTTPException) as bad_name:
        await _rt_cron.list_cron_jobs(profile="../bad")
    assert bad_name.value.status_code == 400

    with pytest.raises(HTTPException) as missing:
        await _rt_cron.list_cron_jobs(profile="missing_profile")
    assert missing.value.status_code == 404


@pytest.mark.asyncio
async def test_create_cron_job_without_profile_uses_backend_own_profile(
    isolated_profiles, monkeypatch
):
    """A pool backend scoped to a named profile must not default creates to
    ``~/.hermes`` when the request carries no explicit ``profile`` (the
    Desktop app's pre-profileScoped clients sent none)."""

    monkeypatch.setenv(
        "HERMES_HOME", str(isolated_profiles["worker_alpha"])
    )

    job = await _rt_cron.create_cron_job(
        _web_models.CronJobCreate(
            prompt="runs in my own profile",
            schedule="every 1h",
            name="own-profile-job",
        ),
        profile=None,
    )

    assert job["profile"] == "worker_alpha"
    assert (isolated_profiles["worker_alpha"] / "cron" / "jobs.json").exists()
    assert not (isolated_profiles["default"] / "cron" / "jobs.json").exists()


@pytest.mark.asyncio
async def test_create_cron_job_without_profile_defaults_when_unscoped(
    isolated_profiles, monkeypatch
):
    """HERMES_HOME at the default home (or unrecognized) keeps the legacy
    ``default`` fallback."""

    monkeypatch.setenv("HERMES_HOME", str(isolated_profiles["default"]))

    job = await _rt_cron.create_cron_job(
        _web_models.CronJobCreate(
            prompt="runs in default",
            schedule="every 1h",
            name="default-job",
        ),
        profile=None,
    )

    assert job["profile"] == "default"
    assert (isolated_profiles["default"] / "cron" / "jobs.json").exists()


def test_list_cron_jobs_carries_each_profiles_ticker_heartbeat_age(isolated_profiles):
    """#114309 — the dashboard must be able to say "scheduler last ticked X hours ago":
    every listed job carries its own profile's ticker heartbeat age (None = never/unknown)."""
    import time

    for name, home in isolated_profiles.items():
        with _web_server_cron._cron_store_scope(home) as cron_jobs:
            cron_jobs.create_job(prompt=f"{name} hourly", schedule="every 1h")
    (isolated_profiles["worker_alpha"] / "cron" / "ticker_heartbeat").write_text(
        str(time.time() - 25 * 3600)
    )

    ages = {job["profile"]: job["scheduler_heartbeat_age_s"] for job in _rt_cron._list_cron_jobs_sync("all")}

    assert ages["default"] is None  # no heartbeat file: cannot date the last tick
    assert 25 * 3600 <= ages["worker_alpha"] < 25 * 3600 + 60


@pytest.mark.asyncio
async def test_cron_run_history_resolves_the_jobs_owner_profile(isolated_profiles, monkeypatch):
    """#115345 — the Desktop lists jobs cross-profile (``?profile=all``) while its
    per-item calls carry the ambient active profile. The run lookup treated that
    hint as the owning profile, opened the wrong profile's state.db and answered
    ``200 {"runs": []}``, so every job owned by another profile rendered
    "No runs yet" for history that existed."""

    worker_job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="owned by worker",
        schedule="every 1h",
        name="cross-profile-runs",
    )
    default_job = _web_server_cron._call_cron_for_profile(
        "default",
        "create_job",
        prompt="owned by default",
        schedule="every 1h",
        name="default-owned-runs",
    )

    # Run sessions live in the state.db of the profile that owns the job, keyed
    # by the canonical id in the session id (cron_{job_id}_{timestamp}).
    runs_by_profile = {
        ("worker_alpha", worker_job["id"]): [
            {"id": f"cron_{worker_job['id']}_1", "started_at": 1.0}
        ],
        ("default", default_job["id"]): [
            {"id": f"cron_{default_job['id']}_1", "started_at": 2.0}
        ],
    }
    opened = []

    class _FakeRunsDB:
        def __init__(self, profile):
            self.profile = profile

        def list_cron_job_runs(self, job_id, limit=20, offset=0):
            opened.append(self.profile)
            return [dict(row) for row in runs_by_profile.get((self.profile, job_id), ())]

        def close(self):
            pass

    monkeypatch.setattr(
        _rt_cron,
        "_open_session_db_for_profile",
        lambda profile, *, read_only: _FakeRunsDB(profile),
    )

    # Active profile is `default`; the listed job belongs to worker_alpha.
    ambient = await _rt_cron.list_cron_job_runs(worker_job["id"], profile="default")
    assert [run["id"] for run in ambient["runs"]] == [f"cron_{worker_job['id']}_1"]
    assert ambient["runs"][0]["profile"] == "worker_alpha"
    assert opened == ["worker_alpha"]

    # Scope is resolved, not widened: the owning-profile hint and the omitted
    # legacy lookup still read the owner's store, and profile="default" still
    # reads default's own job (and only that job) out of default's state.db.
    opened.clear()
    owner = await _rt_cron.list_cron_job_runs(worker_job["id"], profile="worker_alpha")
    omitted = await _rt_cron.list_cron_job_runs(worker_job["id"])
    own = await _rt_cron.list_cron_job_runs(default_job["id"], profile="default")

    assert [run["id"] for run in owner["runs"]] == [f"cron_{worker_job['id']}_1"]
    assert [run["id"] for run in omitted["runs"]] == [f"cron_{worker_job['id']}_1"]
    assert [run["id"] for run in own["runs"]] == [f"cron_{default_job['id']}_1"]
    assert opened == ["worker_alpha", "worker_alpha", "default"]


@pytest.mark.asyncio
async def test_cron_job_mutations_resolve_the_owner_when_the_hint_is_another_profile(isolated_profiles):
    """#115345 sibling: the per-job get/pause/resume/delete calls carry the same ambient-profile
    hint as the run lookup. A hint that does not hold the job must resolve to the owner instead of
    404ing (or acting on the wrong profile's store); a hint that does hold it still wins."""
    worker_job = _web_server_cron._call_cron_for_profile(
        "worker_alpha", "create_job", prompt="owned by worker", schedule="every 1h", name="worker-mutations",
    )
    job_id = worker_job["id"]

    got = await _rt_cron.get_cron_job(job_id, profile="default")
    assert got["id"] == job_id and got["name"] == "worker-mutations"

    paused = await _rt_cron.pause_cron_job(job_id, profile="default")
    assert paused["id"] == job_id and paused.get("enabled") is False
    owner_view = _web_server_cron._call_cron_for_profile("worker_alpha", "get_job", job_id)
    assert owner_view["enabled"] is False  # mutation landed in the owner's jobs.json
    assert _web_server_cron._call_cron_for_profile("default", "list_jobs", True) == []  # not copied into the hint profile

    await _rt_cron.delete_cron_job(job_id, profile="default")
    assert _web_server_cron._call_cron_for_profile("worker_alpha", "get_job", job_id) is None
    with pytest.raises(HTTPException) as exc:
        await _rt_cron.get_cron_job(job_id, profile="default")
    assert exc.value.status_code == 404
