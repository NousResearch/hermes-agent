"""Regression tests for dashboard cron job profile routing."""

from concurrent.futures import ThreadPoolExecutor
import json
from queue import Empty, SimpleQueue
import threading

import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient


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


def _drain_queue(q):
    values = []
    while True:
        try:
            values.append(q.get_nowait())
        except Empty:
            return values




def test_fire_cron_job_scopes_store_and_runtime_home_together(
    isolated_profiles,
    monkeypatch,
):
    """A profile fire must execute and persist under the same profile home."""
    from cron import jobs as cron_jobs
    from cron import scheduler
    from hermes_cli import web_server

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
        assert web_server._fire_cron_job_for_profile("worker_alpha", "worker-job") is True
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
    from hermes_cli import web_server
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

    job = web_server._call_cron_for_profile(
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
    from hermes_cli import web_server

    job = {"id": "saved-job", "name": "saved job"}
    failure = CronSchedulerRegistrationError(
        job,
        RuntimeError("private callback URL and token"),
    )

    def fail_create(*args, **kwargs):
        raise failure

    monkeypatch.setattr(web_server, "_call_cron_for_profile", fail_create)

    with pytest.raises(HTTPException) as exc_info:
        web_server._create_cron_job_sync(
            web_server.CronJobCreate(
                prompt="managed by named profile",
                schedule="every 1h",
                name="named-profile-job",
            ),
            profile="worker_alpha",
        )

    assert exc_info.value.status_code == 424
    assert exc_info.value.detail == {
        "error": "scheduler_registration_failed",
        "job_id": "saved-job",
        "job_saved": True,
        "scheduler_registered": False,
        "retry_create": False,
    }
    serialized = str(exc_info.value.detail)
    assert "private callback URL and token" not in serialized
    assert "RuntimeError" not in serialized


def test_dashboard_create_redacts_unexpected_runtime_error(
    isolated_profiles,
    monkeypatch,
):
    from hermes_cli import web_server

    sentinel = "RAW_CREATE_RUNTIME_SENTINEL user@example.org /private/store.json"

    def fail_create(*args, **kwargs):
        raise OSError(sentinel)

    monkeypatch.setattr(web_server, "_call_cron_for_profile", fail_create)

    with pytest.raises(HTTPException) as exc_info:
        web_server._create_cron_job_sync(
            web_server.CronJobCreate(
                prompt="managed by named profile",
                schedule="every 1h",
            ),
            profile="worker_alpha",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "cron_create_failed"
    assert sentinel not in str(exc_info.value.detail)


def test_notify_cron_provider_scopes_store_and_runtime_home_together(
    isolated_profiles,
    monkeypatch,
):
    """Provider reconciliation must observe the mutated profile, not default."""
    from cron import jobs as cron_jobs
    from cron import scheduler
    from hermes_cli import web_server

    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    default_home = isolated_profiles["default"]
    worker_home = isolated_profiles["worker_alpha"]
    monkeypatch.setattr(scheduler, "_hermes_home", None)
    monkeypatch.setattr(
        web_server,
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
        web_server._notify_cron_provider_for_profile("worker_alpha")
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
    from hermes_cli import web_server

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

    created = web_server._mutate_cron_for_profile(
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
    from hermes_cli import web_server

    monkeypatch.setattr(scheduler, "_hermes_home", None)
    monkeypatch.setattr(
        web_server,
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

    created = web_server._mutate_cron_for_profile(
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
    from hermes_cli import web_server

    monkeypatch.setattr(scheduler, "_hermes_home", None)
    monkeypatch.setattr(
        web_server,
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

    created = web_server._mutate_cron_for_profile(
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
    from hermes_cli import web_server

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
            web_server._call_cron_for_profile,
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






def test_cron_mutations_require_concrete_profile(monkeypatch):
    from fastapi.testclient import TestClient
    from hermes_cli import web_server

    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("mutation boundary reached without a concrete profile")

    monkeypatch.setattr(web_server, "_has_valid_session_token", lambda _request: True)
    monkeypatch.setattr(web_server, "_find_cron_job_profile", forbidden)
    monkeypatch.setattr(web_server, "_mutate_cron_for_profile", forbidden)
    monkeypatch.setattr(web_server, "_call_cron_for_profile", forbidden)
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile", forbidden)

    requests = (
        ("post", "/api/cron/jobs", {"json": {"prompt": "x", "schedule": "every 1h"}}),
        ("put", "/api/cron/jobs/job-1", {"json": {"updates": {"name": "x"}}}),
        ("post", "/api/cron/jobs/job-1/pause", {}),
        ("post", "/api/cron/jobs/job-1/resume", {}),
        ("post", "/api/cron/jobs/job-1/trigger", {}),
        ("delete", "/api/cron/jobs/job-1", {}),
    )

    with TestClient(web_server.app) as client:
        for method, path, kwargs in requests:
            missing = getattr(client, method)(path, **kwargs)
            aggregate = getattr(client, method)(f"{path}?profile=all", **kwargs)
            assert missing.status_code == 422
            assert aggregate.status_code == 400

    assert calls == []


def test_public_cron_profile_metadata_is_canonical_and_route_derived():
    from hermes_cli import web_server

    public = web_server._public_cron_job_for_profile(
        {
            "id": "same-id",
            "profile": "poisoned-profile",
            "profile_name": "Poisoned Profile",
            "hermes_home": "/private/home",
        },
        "Worker_Alpha",
    )

    assert public["profile"] == "worker_alpha"
    assert public["profile_name"] == "worker_alpha"
    assert public["is_default_profile"] is False
    assert "hermes_home" not in public
    assert "poisoned" not in json.dumps(public, sort_keys=True).lower()


def test_dashboard_cron_summary_and_detail_have_separate_trust_boundaries(monkeypatch):
    from hermes_cli import web_server

    raw_job = {
        "id": "redacted-job",
        "name": "bounded summary name",
        "prompt": "PRIVATE_PROMPT_SENTINEL user@example.org\nsecond line",
        "script": "private/script.py",
        "workdir": "/private/worktree",
        "model": "private-model",
        "provider": "private-provider",
        "provider_snapshot": "private-provider-snapshot",
        "model_snapshot": "private-model-snapshot",
        "base_url": "https://private-provider.example.org/v1",
        "profile": "worker-private",
        "profile_name": "Worker Private",
        "skills": ["private-skill"],
        "context_from": ["private-upstream-job"],
        "enabled_toolsets": ["private-toolset"],
        "deliver": "matrix:private-room-id",
        "no_agent": True,
        "monitor_url": "https://private-monitor.example.org/feed",
        "last_status": "error",
        "last_error": "RAW_LAST_ERROR_SENTINEL user@example.org",
        "last_delivery_error": "RAW_DELIVERY_SENTINEL /private/report.pdf",
        "last_fire_error": {
            "at": "2026-08-22T19:00:00Z",
            "detail": "RAW_FIRE_SENTINEL provider payload",
        },
        "fire_claim": {
            "by": "RAW_CLAIM_OWNER_SENTINEL user@example.org /private/owner",
            "at": "2026-08-22T19:01:00Z",
            "fire_at": "2026-08-22T19:00:00Z",
        },
        "execution_id": "RAW_EXECUTION_SENTINEL",
        "fire_identity": "RAW_FIRE_IDENTITY_SENTINEL",
        "last_output": "RAW_OUTPUT_SENTINEL private result body",
        "hermes_home": "/private/hermes/home",
        "future_runtime_field": "RAW_FUTURE_RUNTIME_SENTINEL",
    }

    def call(_profile, func_name, *_args):
        if func_name == "list_jobs":
            return [raw_job]
        if func_name == "get_job":
            return raw_job
        raise AssertionError(func_name)

    monkeypatch.setattr(web_server, "_call_cron_for_profile", call)

    listed = web_server._list_cron_jobs_sync("default")[0]
    fetched = web_server._get_cron_job_sync("redacted-job", profile="default")
    sensitive_summary_fields = {
        "prompt", "script", "workdir", "model", "provider", "base_url",
        "provider_snapshot", "model_snapshot",
        "skills", "context_from", "enabled_toolsets", "deliver", "no_agent",
        "monitor_url",
    }
    for public in (listed, fetched):
        serialized = json.dumps(public, sort_keys=True)
        assert public["name"] == "bounded summary name"
        assert public["profile"] == "default"
        assert public["profile_name"] == "default"
        assert public["is_default_profile"] is True
        assert public["delivery_kind"] == "external"
        assert public["mode"] == "monitor"
        assert public["skill_count"] == 1
        assert public["toolset_count"] == 1
        assert public["model_configured"] is True
        assert sensitive_summary_fields.isdisjoint(public)
        assert public["last_error"] == "run_failed"
        assert public["last_delivery_error"] == "delivery_failed"
        assert public["last_fire_error"] == {
            "at": "2026-08-22T19:00:00Z",
            "error_kind": "fire_forward_failed",
        }
        assert "RAW_" not in serialized
        assert "PRIVATE_" not in serialized
        assert "worker-private" not in serialized
        assert "Worker Private" not in serialized
        assert "user@example.org" not in serialized
        assert "/private/" not in serialized

    detail = web_server._get_cron_job_detail_sync(
        "redacted-job", profile="default",
    )
    assert detail["prompt"] == raw_job["prompt"]
    assert detail["script"] == raw_job["script"]
    assert detail["workdir"] == raw_job["workdir"]
    assert detail["model"] == raw_job["model"]
    assert detail["provider"] == raw_job["provider"]
    assert detail["base_url"] == raw_job["base_url"]
    assert detail["deliver"] == raw_job["deliver"]
    assert detail["no_agent"] is True
    assert detail["monitor_url"] == raw_job["monitor_url"]
    assert "profile" not in detail
    assert "profile_name" not in detail
    assert "provider_snapshot" not in detail
    assert "model_snapshot" not in detail
    assert "fire_claim" not in detail
    assert "last_output" not in detail
    assert "future_runtime_field" not in detail

    assert raw_job["last_error"].startswith("RAW_LAST_ERROR_SENTINEL")
    assert "detail" in raw_job["last_fire_error"]


def test_dashboard_cron_detail_endpoint_requires_explicit_profile(monkeypatch):
    from hermes_cli import web_server

    raw_job = {
        "id": "detail-job",
        "name": "detail job",
        "prompt": "PRIVATE_DETAIL_PROMPT",
        "script": "private/detail.py",
        "workdir": "/private/detail-worktree",
        "enabled": True,
        "state": "scheduled",
    }

    calls = []

    def call(profile, func_name, *args):
        calls.append((profile, func_name, args))
        return raw_job

    monkeypatch.setattr(web_server, "_has_valid_session_token", lambda _request: True)
    monkeypatch.setattr(web_server, "_call_cron_for_profile", call)

    with TestClient(web_server.app) as client:
        missing = client.get("/api/cron/jobs/detail-job/detail")
        aggregate = client.get("/api/cron/jobs/detail-job/detail?profile=all")
        detail = client.get(
            "/api/cron/jobs/detail-job/detail?profile=worker_alpha",
        )

    assert missing.status_code == 422
    assert aggregate.status_code == 400
    assert detail.status_code == 200
    assert calls == [("worker_alpha", "get_job", ("detail-job",))]
    assert detail.json()["prompt"] == "PRIVATE_DETAIL_PROMPT"
    assert detail.json()["workdir"] == "/private/detail-worktree"
    assert "profile" not in detail.json()


def test_dashboard_cron_job_surface_drops_malformed_fire_error(monkeypatch):
    from hermes_cli import web_server

    sentinel = "RAW_MALFORMED_FIRE_SENTINEL user@example.org /private/report.pdf"
    raw_job = {"id": "malformed-fire", "last_fire_error": sentinel}
    monkeypatch.setattr(
        web_server,
        "_call_cron_for_profile",
        lambda _profile, _func_name, *_args: [raw_job],
    )

    public = web_server._list_cron_jobs_sync("default")[0]
    assert public["last_fire_error"] is None
    assert sentinel not in json.dumps(public, sort_keys=True)
    assert raw_job["last_fire_error"] == sentinel


@pytest.mark.parametrize(
    "malformed_deliver",
    [
        ["PRIVATE_DELIVERY_LIST_SENTINEL"],
        {"private": "PRIVATE_DELIVERY_DICT_SENTINEL"},
    ],
)
def test_dashboard_cron_job_surface_rejects_unhashable_delivery(malformed_deliver):
    from hermes_cli import web_server

    public = web_server._public_cron_job({
        "id": "malformed-delivery",
        "deliver": malformed_deliver,
    })

    assert public["delivery_kind"] == "local"
    assert "PRIVATE_DELIVERY" not in json.dumps(public, sort_keys=True)


def test_dashboard_cron_projection_rejects_builtin_subclasses_before_magic_methods():
    from hermes_cli import web_server

    class HostileDict(dict):
        def get(self, *_args, **_kwargs):
            raise AssertionError("hostile mapping get was called")

        def __contains__(self, _key):
            raise AssertionError("hostile mapping membership was evaluated")

        def __getitem__(self, _key):
            raise AssertionError("hostile mapping item access was called")

    class HostileList(list):
        def __iter__(self):
            raise AssertionError("hostile list was iterated")

        def __len__(self):
            raise AssertionError("hostile list length was evaluated")

    class HostileText(str):
        def __bool__(self):
            raise AssertionError("hostile text truthiness was evaluated")

        def __len__(self):
            raise AssertionError("hostile text length was evaluated")

    with pytest.raises(TypeError, match="requires an object"):
        web_server._public_cron_job(HostileDict({"id": "hostile"}))

    public = web_server._public_cron_job({
        "id": "bounded",
        "schedule": HostileDict({"kind": "once"}),
        "repeat": HostileDict({"times": 1}),
        "last_fire_error": HostileDict({"at": "2026-08-23T20:00:00+00:00"}),
        "skills": HostileList(["private"]),
        "enabled_toolsets": HostileList(["private"]),
        "deliver": HostileText("external:private"),
        "model": HostileText("private-model"),
    })

    assert public["schedule"] is None
    assert public["repeat"] is None
    assert public["last_fire_error"] is None
    assert public["skill_count"] == 0
    assert public["toolset_count"] == 0
    assert public["delivery_kind"] == "local"
    assert public["model_configured"] is False


def test_dashboard_cron_projection_rejects_bool_class_spoof_without_descriptor_call():
    from hermes_cli import web_server

    class_property_calls = []

    class HostileBoolSpoof:
        @property
        def __class__(self):
            class_property_calls.append("called")
            return bool

    summary = web_server._public_cron_job({
        "id": "bounded",
        "enabled": HostileBoolSpoof(),
    })
    detail = web_server._public_cron_job_detail({
        "id": "bounded",
        "enabled": HostileBoolSpoof(),
        "no_agent": HostileBoolSpoof(),
        "continuity": HostileBoolSpoof(),
    })

    assert summary["enabled"] is False
    assert detail["enabled"] is False
    assert detail["no_agent"] is False
    assert detail["continuity"] is False
    assert class_property_calls == []


def test_cron_run_and_gateway_fire_projections_reject_dict_subclasses_without_get():
    from hermes_cli import web_server
    from hermes_cli.web_routers.cron import _public_gateway_fire_body

    class HostileDict(dict):
        def get(self, *_args, **_kwargs):
            raise AssertionError("hostile projection get was called")

    assert web_server._public_cron_run(HostileDict({"id": "run"}), now=1.0) is None
    assert _public_gateway_fire_body(200, HostileDict({"status": "accepted"}), "job") == {
        "error": "gateway_fire_failed",
        "error_kind": "gateway_fire_failed",
        "job_id": "job",
    }


def test_dashboard_cron_job_surface_drops_malformed_timestamps(monkeypatch):
    from hermes_cli import web_server

    sentinel = "RAW_TIMESTAMP_SENTINEL user@example.org /private/runtime"
    raw_job = {
        "id": "malformed-timestamps",
        "last_run_at": {"raw": sentinel},
        "next_run_at": sentinel,
        "schedule": {
            "kind": "once",
            "run_at": sentinel,
            "future_nested_runtime": sentinel,
        },
    }
    monkeypatch.setattr(
        web_server,
        "_call_cron_for_profile",
        lambda _profile, _func_name, *_args: [raw_job],
    )

    public = web_server._list_cron_jobs_sync("default")[0]

    assert public["last_run_at"] is None
    assert public["next_run_at"] is None
    assert public["schedule"]["run_at"] is None
    assert "future_nested_runtime" not in public["schedule"]
    assert sentinel not in json.dumps(public, sort_keys=True)

    aware = "2026-08-23T08:00:00+00:00"
    valid = web_server._public_cron_job({
        "id": "valid-timestamps",
        "last_run_at": aware,
        "next_run_at": aware,
        "schedule": {"kind": "once", "run_at": aware},
    })
    assert valid["last_run_at"] == aware
    assert valid["next_run_at"] == aware
    assert valid["schedule"]["run_at"] == aware


def test_dashboard_cron_mutation_surfaces_redact_raw_error_details(monkeypatch, tmp_path):
    from hermes_cli import web_server

    raw_job = {
        "id": "mutation-redaction-job",
        "name": "mutation redaction",
        "prompt": "safe mutation prompt",
        "enabled": True,
        "state": "scheduled",
        "last_run_at": "2026-08-22T22:00:00+00:00",
        "last_status": "error",
        "last_error": "RAW_MUTATION_RUNTIME_SENTINEL provider says secret",
        "last_delivery_error": "RAW_MUTATION_DELIVERY_SENTINEL /private/media.pdf",
        "last_fire_error": {
            "at": "2026-08-22T22:01:00+00:00",
            "detail": "RAW_MUTATION_FIRE_SENTINEL user@example.org",
        },
        "fire_claim": {
            "by": "RAW_MUTATION_CLAIM_SENTINEL user@example.org /private/owner",
            "at": "2026-08-22T22:01:00+00:00",
            "fire_at": "2026-08-22T22:00:00+00:00",
        },
        "execution_id": "RAW_MUTATION_EXECUTION_SENTINEL",
        "fire_identity": "RAW_MUTATION_IDENTITY_SENTINEL",
        "last_output": "RAW_MUTATION_OUTPUT_SENTINEL private body",
        "future_runtime_field": "RAW_MUTATION_FUTURE_SENTINEL",
    }

    monkeypatch.setattr(
        web_server,
        "_cron_profile_home",
        lambda profile: (profile, tmp_path),
    )
    monkeypatch.setattr(
        web_server,
        "_call_cron_for_profile",
        lambda _profile, _func_name, *_args: raw_job,
    )
    monkeypatch.setattr(
        web_server,
        "_mutate_cron_for_profile",
        lambda _profile, _func_name, *_args, **_kwargs: raw_job,
    )
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile", lambda *_args, **_kwargs: True)

    responses = [
        web_server._create_cron_job_sync(
            web_server.CronJobCreate(
                prompt="safe mutation prompt",
                schedule="every 1h",
                name="mutation redaction",
            ),
            profile="default",
        ),
        web_server._update_cron_job_sync(
            raw_job["id"],
            web_server.CronJobUpdate(updates={"name": "updated"}),
            profile="default",
        ),
        web_server._pause_cron_job_sync(raw_job["id"], profile="default"),
        web_server._resume_cron_job_sync(raw_job["id"], profile="default"),
        web_server._trigger_cron_job_sync(raw_job["id"], profile="default"),
    ]

    for public in responses:
        assert "prompt" not in public
        assert public["last_error"] == "run_failed"
        assert public["last_delivery_error"] == "delivery_failed"
        assert public["last_fire_error"] == {
            "at": "2026-08-22T22:01:00+00:00",
            "error_kind": "fire_forward_failed",
        }
        serialized = json.dumps(public, sort_keys=True)
        assert "RAW_MUTATION" not in serialized
        assert "user@example.org" not in serialized
        assert "/private/media.pdf" not in serialized
        for private_field in (
            "fire_claim", "execution_id", "fire_identity", "last_output",
            "future_runtime_field",
        ):
            assert private_field not in public

    assert raw_job["last_error"].startswith("RAW_MUTATION_RUNTIME_SENTINEL")
    assert "detail" in raw_job["last_fire_error"]


def test_dashboard_update_delete_redact_unexpected_value_errors(monkeypatch, tmp_path):
    from hermes_cli import web_server

    sentinel = "RAW_UPDATE_DELETE user@example.org /private/cron/jobs.json"
    existing = {
        "id": "bounded-mutation-errors",
        "prompt": "safe prompt",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
    }

    monkeypatch.setattr(web_server, "_has_valid_session_token", lambda _request: True)
    monkeypatch.setattr(
        web_server,
        "_cron_profile_home",
        lambda profile: (profile, tmp_path),
    )
    monkeypatch.setattr(
        web_server,
        "_call_cron_for_profile",
        lambda _profile, function, *_args: existing
        if function == "get_job"
        else None,
    )

    def explode(*_args, **_kwargs):
        raise ValueError(sentinel)

    monkeypatch.setattr(web_server, "_mutate_cron_for_profile", explode)

    with TestClient(web_server.app) as client:
        updated = client.put(
            f"/api/cron/jobs/{existing['id']}?profile=default",
            json={"updates": {"name": "updated"}},
        )
        deleted = client.delete(
            f"/api/cron/jobs/{existing['id']}?profile=default",
        )

    assert updated.status_code == 400
    assert updated.json() == {"detail": "cron_update_failed"}
    assert deleted.status_code == 400
    assert deleted.json() == {"detail": "cron_delete_failed"}
    assert sentinel not in updated.text
    assert sentinel not in deleted.text


def test_dashboard_cron_run_history_uses_bounded_projection(monkeypatch):
    from hermes_cli import web_server

    sentinels = {
        "system_prompt": "RAW_SYSTEM_PROMPT user@example.org",
        "preview": "RAW_USER_MESSAGE /private/transcript.txt",
        "cwd": "/private/worktree",
        "model": "private-provider/private-model",
        "profile": "private-profile",
        "future_runtime": object(),
    }
    raw_run = {
        "id": "cron_bounded-history_20260823_100000",
        "source": "cron",
        "started_at": 1_700_000_000.0,
        "ended_at": None,
        "last_active": 1_700_000_001.0,
        "end_reason": None,
        "archived": 0,
        **sentinels,
    }
    malformed_run = {
        "id": "cron_bounded-history_20260823_100001",
        "started_at": True,
        "ended_at": "RAW_END_TIME /private/time",
        "last_active": float("nan"),
        "end_reason": {"private": "RAW_END_REASON user@example.org"},
        "archived": "1",
    }
    unsafe_id_run = {
        "id": "cron_user@example.org/private",
        "started_at": 1_700_000_002.0,
        "ended_at": None,
        "last_active": 1_700_000_002.0,
    }

    class FakeSessionDB:
        def list_cron_job_runs(self, job_id, *, limit, offset):
            assert job_id == "bounded-history"
            assert (limit, offset) == (20, 0)
            return [dict(raw_run), malformed_run, unsafe_id_run]

        def close(self):
            pass

    monkeypatch.setattr(web_server, "_has_valid_session_token", lambda _request: True)
    monkeypatch.setattr(
        web_server,
        "_call_cron_for_profile",
        lambda _profile, function, *_args: {"id": "bounded-history"}
        if function == "get_job"
        else None,
    )
    monkeypatch.setattr(
        web_server,
        "_open_session_db_for_profile",
        lambda _profile, read_only: FakeSessionDB(),
    )
    monkeypatch.setattr(web_server.time, "time", lambda: 1_700_000_100.0)

    with TestClient(web_server.app) as client:
        response = client.get(
            "/api/cron/jobs/bounded-history/runs?profile=default",
        )

    assert response.status_code == 200
    assert response.json() == {
        "runs": [
            {
                "id": raw_run["id"],
                "status": "running",
                "started_at": 1_700_000_000.0,
                "ended_at": None,
                "last_active": 1_700_000_001.0,
                "is_active": True,
                "archived": False,
            },
            {
                "id": malformed_run["id"],
                "status": "ended",
                "started_at": None,
                "ended_at": None,
                "last_active": None,
                "is_active": False,
                "archived": False,
            },
        ],
        "limit": 20,
    }
    serialized = response.text
    for sentinel in sentinels.values():
        if isinstance(sentinel, str):
            assert sentinel not in serialized
    assert "RAW_END_TIME" not in serialized
    assert "RAW_END_REASON" not in serialized
    assert "user@example.org/private" not in serialized


def test_dashboard_trigger_completed_fallback_redacts_raw_error_details(monkeypatch):
    from hermes_cli import web_server

    raw_job = {
        "id": "completed-trigger-redaction-job",
        "name": "completed trigger redaction",
        "enabled": True,
        "state": "scheduled",
        "last_run_at": None,
        "last_status": "error",
        "last_error": "RAW_COMPLETED_RUNTIME_SENTINEL provider says secret",
        "last_delivery_error": "RAW_COMPLETED_DELIVERY_SENTINEL /private/media.pdf",
        "last_fire_error": {
            "at": "2026-08-22T22:02:00+00:00",
            "detail": "RAW_COMPLETED_FIRE_SENTINEL user@example.org",
        },
    }

    def read_job(_profile, function, *_args):
        if function == "resolve_job_ref":
            return raw_job
        if function == "get_job":
            return None
        raise AssertionError(function)

    monkeypatch.setattr(web_server, "_call_cron_for_profile", read_job)
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile", lambda *_args, **_kwargs: True)

    public = web_server._trigger_cron_job_sync(raw_job["id"], profile="default")

    assert public["enabled"] is False
    assert public["state"] == "completed"
    assert public["last_error"] == "run_failed"
    assert public["last_delivery_error"] == "delivery_failed"
    assert public["last_fire_error"] == {
        "at": "2026-08-22T22:02:00+00:00",
        "error_kind": "fire_forward_failed",
    }
    serialized = json.dumps(public, sort_keys=True)
    assert "RAW_COMPLETED" not in serialized
    assert "user@example.org" not in serialized
    assert "/private/media.pdf" not in serialized


@pytest.mark.asyncio
async def test_dashboard_cron_mutations_notify_selected_profile_provider(
    isolated_profiles,
    monkeypatch,
):
    from hermes_cli import web_server

    notified_profiles = []
    monkeypatch.setattr(
        web_server,
        "_notify_cron_provider_for_profile",
        notified_profiles.append,
    )

    created = await web_server.create_cron_job(
        web_server.CronJobCreate(
            prompt="managed by named profile",
            schedule="every 1h",
            name="provider-notify-job",
        ),
        profile="worker_alpha",
    )
    await web_server.update_cron_job(
        created["id"],
        web_server.CronJobUpdate(updates={"name": "provider-notify-job-updated"}),
        profile="worker_alpha",
    )
    await web_server.pause_cron_job(created["id"], profile="worker_alpha")
    await web_server.resume_cron_job(created["id"], profile="worker_alpha")
    await web_server.delete_cron_job(created["id"], profile="worker_alpha")

    assert notified_profiles == ["worker_alpha"] * 5


@pytest.mark.asyncio
async def test_blueprint_instantiation_notifies_selected_profile_provider(
    isolated_profiles,
    monkeypatch,
):
    from hermes_cli import web_server

    notified_profiles = []
    monkeypatch.setattr(
        web_server,
        "_notify_cron_provider_for_profile",
        notified_profiles.append,
    )

    created = await web_server.instantiate_blueprint(
        web_server.AutomationBlueprintInstantiate(
            blueprint="morning-brief",
            values={"time": "07:30", "deliver": "local"},
        ),
        profile="worker_alpha",
    )

    assert "profile" not in created
    persisted = web_server._call_cron_for_profile(
        "worker_alpha", "get_job", created["id"],
    )
    assert persisted["profile"] == "worker_alpha"
    assert notified_profiles == ["worker_alpha"]


@pytest.mark.asyncio
async def test_trigger_cron_job_fires_only_selected_job_and_returns_refreshed_state(
    isolated_profiles,
    monkeypatch,
):
    from cron import jobs as cron_jobs
    from hermes_cli import web_server

    selected = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="run immediately",
        schedule="every 1h",
        name="selected-trigger-job",
    )
    sibling = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="leave scheduled",
        schedule="every 1h",
        name="sibling-job",
    )
    fired = []

    class RecordingProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None, force=False):
            fired.append(
                {
                    "job_id": job_id,
                    "jobs_file": cron_jobs._current_cron_store().jobs_file,
                    "force": force,
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

    triggered = await web_server.trigger_cron_job(
        selected["id"],
        profile="worker_alpha",
    )

    assert fired == [
        {
            "job_id": selected["id"],
            "jobs_file": isolated_profiles["worker_alpha"] / "cron" / "jobs.json",
            "force": False,
        }
    ]
    assert triggered["last_status"] == "ok"
    assert triggered["last_run_at"] is not None
    untouched = web_server._call_cron_for_profile(
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
    from hermes_cli import web_server

    job = web_server._call_cron_for_profile(
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

    with pytest.raises(HTTPException) as exc:
        await web_server.trigger_cron_job(job["id"], profile="worker_alpha")

    assert exc.value.status_code == 409
    assert "already running" in exc.value.detail


@pytest.mark.asyncio
async def test_trigger_cron_job_forces_paused_job_atomically(
    isolated_profiles,
    monkeypatch,
):
    from cron import jobs as cron_jobs
    from hermes_cli import web_server

    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="resume me",
        schedule="every 1h",
        name="paused-trigger-job",
    )
    web_server._call_cron_for_profile("worker_alpha", "pause_job", job["id"])
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

    triggered = await web_server.trigger_cron_job(
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
    from hermes_cli import web_server

    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="stay paused",
        schedule="every 1h",
        name="legacy-paused-trigger-job",
    )
    web_server._call_cron_for_profile("worker_alpha", "pause_job", job["id"])
    calls = []

    class LegacyProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None):
            calls.append(job_id)
            return True

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: LegacyProvider(),
    )

    with pytest.raises(HTTPException) as exc:
        await web_server.trigger_cron_job(job["id"], profile="worker_alpha")

    assert exc.value.status_code == 409
    assert "forced" in exc.value.detail.lower()
    assert calls == []
    persisted = web_server._call_cron_for_profile(
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
    from hermes_cli import web_server

    job = web_server._call_cron_for_profile(
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

    triggered = await web_server.trigger_cron_job(
        job["id"],
        profile="worker_alpha",
    )

    assert triggered["last_status"] == "error"
    assert triggered["last_error"] == "run_failed"
    assert "expected failure" not in json.dumps(triggered, sort_keys=True)


@pytest.mark.asyncio
async def test_trigger_cron_job_returns_completed_snapshot_for_retained_oneshot(
    isolated_profiles,
    monkeypatch,
):
    from cron import jobs as cron_jobs
    from hermes_cli import web_server

    job = web_server._call_cron_for_profile(
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

    triggered = await web_server.trigger_cron_job(
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
    retained = web_server._call_cron_for_profile(
        "worker_alpha",
        "get_job",
        job["id"],
    )
    assert retained is not None
    assert retained["state"] == "completed"


@pytest.mark.asyncio
async def test_cron_profile_scan_runs_off_event_loop(isolated_profiles, monkeypatch):
    from hermes_cli import web_server

    worker_job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="thread-offload-job",
    )

    event_loop_thread = threading.get_ident()
    profile_scan_threads = SimpleQueue()
    worker_threads = SimpleQueue()
    original_profile_dicts = web_server._cron_profile_dicts
    original_mutate = web_server._mutate_cron_for_profile

    def tracking_profile_dicts():
        profile_scan_threads.put(threading.get_ident())
        return original_profile_dicts()

    def tracking_mutate(profile, func_name, *args, **kwargs):
        worker_threads.put(threading.get_ident())
        return original_mutate(profile, func_name, *args, **kwargs)

    monkeypatch.setattr(web_server, "_cron_profile_dicts", tracking_profile_dicts)
    monkeypatch.setattr(web_server, "_mutate_cron_for_profile", tracking_mutate)

    jobs = await web_server.list_cron_jobs(profile="all")
    paused = await web_server.pause_cron_job(
        worker_job["id"], profile="worker_alpha",
    )

    listed_worker = next(job for job in jobs if job["id"] == worker_job["id"])
    assert listed_worker["profile"] == "worker_alpha"
    assert listed_worker["profile_name"] == "worker_alpha"
    assert listed_worker["is_default_profile"] is False
    assert "hermes_home" not in listed_worker
    assert "profile" not in paused
    profile_scan_thread_ids = _drain_queue(profile_scan_threads)
    worker_thread_ids = _drain_queue(worker_threads)
    assert profile_scan_thread_ids
    assert worker_thread_ids
    assert all(thread_id != event_loop_thread for thread_id in profile_scan_thread_ids)
    assert all(thread_id != event_loop_thread for thread_id in worker_thread_ids)


@pytest.mark.asyncio
async def test_cron_dashboard_io_rejects_async_callables():
    from hermes_cli import web_server

    async def async_callable():
        return "nope"

    with pytest.raises(TypeError, match="only accepts sync callables"):
        await web_server._run_cron_dashboard_io(async_callable)



@pytest.mark.asyncio
async def test_update_cron_job_normalizes_dashboard_core_fields(isolated_profiles, tmp_path):
    from hermes_cli import web_server

    scripts_dir = isolated_profiles["worker_alpha"] / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "collect.py").write_text("print('ok')\n", encoding="utf-8")
    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="normalizes-dashboard-fields",
    )

    updated = await web_server.update_cron_job(
        job["id"],
        web_server.CronJobUpdate(
            updates={
                "base_url": "https://example.invalid/v1/",
                "script": str(scripts_dir / "collect.py"),
                "context_from": "",
                "no_agent": True,
            }
        ),
        profile="worker_alpha",
    )

    assert updated["mode"] == "script"
    for field in ("base_url", "script", "context_from", "no_agent"):
        assert field not in updated
    detail = web_server._get_cron_job_detail_sync(
        job["id"], profile="worker_alpha",
    )
    assert detail["base_url"] == "https://example.invalid/v1"
    assert detail["script"] == "collect.py"
    assert detail["context_from"] == []
    assert detail["no_agent"] is True


@pytest.mark.asyncio
async def test_create_cron_job_rejects_script_outside_profile_scripts(
    isolated_profiles, tmp_path
):
    from hermes_cli import web_server

    outside = tmp_path / "outside.py"
    outside.write_text("print('nope')\n", encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        await web_server.create_cron_job(
            web_server.CronJobCreate(
                schedule="every 1h",
                script=str(outside),
                no_agent=True,
            ),
            profile="worker_alpha",
        )

    assert exc.value.status_code == 400
    assert "inside" in exc.value.detail


@pytest.mark.asyncio
async def test_create_cron_job_rejects_empty_agent_job(isolated_profiles):
    from hermes_cli import web_server

    with pytest.raises(HTTPException) as exc:
        await web_server.create_cron_job(
            web_server.CronJobCreate(schedule="every 1h"),
            profile="worker_alpha",
        )

    assert exc.value.status_code == 400
    assert "prompt, skill, or script" in exc.value.detail


@pytest.mark.asyncio
async def test_update_cron_job_no_agent_reuses_existing_script(isolated_profiles):
    from hermes_cli import web_server

    scripts_dir = isolated_profiles["worker_alpha"] / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "collect.py").write_text("print('ok')\n", encoding="utf-8")

    job = await web_server.create_cron_job(
        web_server.CronJobCreate(
            schedule="every 1h",
            script=str(scripts_dir / "collect.py"),
        ),
        profile="worker_alpha",
    )

    updated = await web_server.update_cron_job(
        job["id"],
        web_server.CronJobUpdate(updates={"no_agent": True}),
        profile="worker_alpha",
    )

    assert updated["mode"] == "script"
    assert "no_agent" not in updated
    assert "script" not in updated
    detail = web_server._get_cron_job_detail_sync(
        job["id"], profile="worker_alpha",
    )
    assert detail["no_agent"] is True
    assert detail["script"] == "collect.py"


@pytest.mark.asyncio
async def test_dashboard_cron_rejects_missing_context_from(isolated_profiles):
    from hermes_cli import web_server

    with pytest.raises(HTTPException) as create_exc:
        await web_server.create_cron_job(
            web_server.CronJobCreate(
                prompt="process missing upstream",
                schedule="every 1h",
                context_from=["missing-job-id"],
            ),
            profile="worker_alpha",
        )

    assert create_exc.value.status_code == 400
    assert "missing-job-id" in create_exc.value.detail

    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="context-update-target",
    )

    with pytest.raises(HTTPException) as update_exc:
        await web_server.update_cron_job(
            job["id"],
            web_server.CronJobUpdate(
                updates={
                    "context_from": ["missing-job-id"],
                }
            ),
            profile="worker_alpha",
        )

    assert update_exc.value.status_code == 400
    assert "missing-job-id" in update_exc.value.detail






@pytest.mark.asyncio
async def test_dashboard_cron_noop_inference_fields_keep_existing_snapshots(
    isolated_profiles,
    monkeypatch,
):
    from hermes_cli import runtime_provider, web_server

    current_provider = {"name": "initial-provider"}
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {"provider": current_provider["name"]},
    )

    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="dashboard-edit-job",
    )

    assert job["provider_snapshot"] == "initial-provider"
    assert job["model_snapshot"] == "test-model"

    current_provider["name"] = "changed-provider"
    (isolated_profiles["worker_alpha"] / "config.yaml").write_text(
        "model: changed-model\n",
        encoding="utf-8",
    )

    updated = await web_server.update_cron_job(
        job["id"],
        web_server.CronJobUpdate(
            updates={
                "name": "dashboard-edit-job-renamed",
                "provider": None,
                "model": None,
                "base_url": None,
                "no_agent": False,
            }
        ),
        profile="worker_alpha",
    )

    assert updated["name"] == "dashboard-edit-job-renamed"
    assert "provider_snapshot" not in updated
    assert "model_snapshot" not in updated
    persisted = web_server._call_cron_for_profile(
        "worker_alpha", "get_job", job["id"],
    )
    assert persisted["provider_snapshot"] == "initial-provider"
    assert persisted["model_snapshot"] == "test-model"


@pytest.mark.asyncio
async def test_update_cron_job_clears_snapshots_for_no_agent(
    isolated_profiles,
    monkeypatch,
):
    from hermes_cli import runtime_provider, web_server

    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {"provider": "worker-provider"},
    )
    scripts_dir = isolated_profiles["worker_alpha"] / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "collect.py").write_text("print('ok')\n", encoding="utf-8")

    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="agent-to-script-job",
    )

    assert job["provider_snapshot"] == "worker-provider"
    assert job["model_snapshot"] == "test-model"

    updated = await web_server.update_cron_job(
        job["id"],
        web_server.CronJobUpdate(
            updates={
                "script": str(scripts_dir / "collect.py"),
                "no_agent": True,
            }
        ),
        profile="worker_alpha",
    )

    assert "provider_snapshot" not in updated
    assert "model_snapshot" not in updated
    persisted = web_server._call_cron_for_profile(
        "worker_alpha", "get_job", job["id"],
    )
    assert persisted["provider_snapshot"] is None
    assert persisted["model_snapshot"] is None


@pytest.mark.asyncio
async def test_update_cron_job_rejects_id_mutation(isolated_profiles, monkeypatch):
    """Dashboard surfaces a 400 (not a 500 or silent rename) when an
    id-mutation attempt is rejected by cron/jobs.update_job."""
    from hermes_cli import web_server

    notified_profiles = []
    monkeypatch.setattr(
        web_server,
        "_notify_cron_provider_for_profile",
        notified_profiles.append,
    )
    worker_job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="immutable-id-job",
    )

    with pytest.raises(HTTPException) as exc:
        await web_server.update_cron_job(
            worker_job["id"],
            web_server.CronJobUpdate(updates={"id": "../escape"}),
            profile="worker_alpha",
        )

    assert exc.value.status_code == 400
    assert "id" in exc.value.detail
    assert notified_profiles == []
    worker_jobs = await web_server.list_cron_jobs(profile="worker_alpha")
    assert [job["id"] for job in worker_jobs] == [worker_job["id"]]


@pytest.mark.asyncio
async def test_cron_delete_with_profile_deletes_only_target_profile(isolated_profiles):
    from hermes_cli import web_server

    default_job = web_server._call_cron_for_profile(
        "default",
        "create_job",
        prompt="same-ish default",
        schedule="every 1h",
        name="shared-name",
    )
    worker_job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="same-ish worker",
        schedule="every 1h",
        name="shared-name-worker",
    )

    deleted = await web_server.delete_cron_job(worker_job["id"], profile="worker_alpha")
    assert deleted == {"ok": True}

    remaining_default = await web_server.list_cron_jobs(profile="default")
    remaining_worker = await web_server.list_cron_jobs(profile="worker_alpha")
    assert [job["id"] for job in remaining_default] == [default_job["id"]]
    assert remaining_worker == []


@pytest.mark.asyncio
async def test_cron_profile_validation_errors(isolated_profiles):
    from hermes_cli import web_server

    with pytest.raises(HTTPException) as bad_name:
        await web_server.list_cron_jobs(profile="../bad")
    assert bad_name.value.status_code == 400

    with pytest.raises(HTTPException) as missing:
        await web_server.list_cron_jobs(profile="missing_profile")
    assert missing.value.status_code == 404


@pytest.mark.asyncio
async def test_create_cron_job_with_explicit_worker_profile_uses_worker_store(
    isolated_profiles, monkeypatch
):
    """An explicit named profile must write only that profile's store."""
    from hermes_cli import web_server

    monkeypatch.setenv(
        "HERMES_HOME", str(isolated_profiles["worker_alpha"])
    )

    job = await web_server.create_cron_job(
        web_server.CronJobCreate(
            prompt="runs in my own profile",
            schedule="every 1h",
            name="own-profile-job",
        ),
        profile="worker_alpha",
    )

    assert "profile" not in job
    persisted = web_server._call_cron_for_profile(
        "worker_alpha", "get_job", job["id"],
    )
    assert persisted["profile"] == "worker_alpha"
    assert (isolated_profiles["worker_alpha"] / "cron" / "jobs.json").exists()
    assert not (isolated_profiles["default"] / "cron" / "jobs.json").exists()


@pytest.mark.asyncio
async def test_create_cron_job_with_explicit_default_uses_default_store(
    isolated_profiles, monkeypatch
):
    """An explicit default profile writes the default store."""
    from hermes_cli import web_server

    monkeypatch.setenv("HERMES_HOME", str(isolated_profiles["default"]))

    job = await web_server.create_cron_job(
        web_server.CronJobCreate(
            prompt="runs in default",
            schedule="every 1h",
            name="default-job",
        ),
        profile="default",
    )

    assert "profile" not in job
    persisted = web_server._call_cron_for_profile(
        "default", "get_job", job["id"],
    )
    assert persisted["profile"] == "default"
    assert (isolated_profiles["default"] / "cron" / "jobs.json").exists()
