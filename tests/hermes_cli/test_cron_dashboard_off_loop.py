"""Regression tests: cron dashboard handlers must not run profile I/O on the event loop.

Guards the residual sites missed by the 49fa04a23/346e5673d threadpool
migration: POST /api/cron/fire (_find_cron_job_profile) and
POST /api/cron/blueprints/instantiate (_call_cron_for_profile create_job).
Each stub asserts it is running OFF the event loop thread by checking that
no running asyncio loop is present in its thread.
"""

import asyncio

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture()
def loop_probe():
    """Collect (tag, on_loop) proof from stubbed profile-I/O helpers."""
    seen = []

    def probe(tag):
        try:
            asyncio.get_running_loop()
            seen.append((tag, True))
        except RuntimeError:
            seen.append((tag, False))

    return seen, probe


def test_cron_fire_profile_lookup_off_loop(monkeypatch, loop_probe):
    seen, probe = loop_probe

    def fake_find(job_id):
        probe("find")
        return None

    monkeypatch.setattr(web_server, "_find_cron_job_profile", fake_find)

    import plugins.cron_providers.chronos.verify as chv
    monkeypatch.setattr(chv, "get_fire_verifier", lambda: (lambda **kw: {"sub": "t"}))

    client = TestClient(web_server.app)
    resp = client.post(
        "/api/cron/fire",
        json={"job_id": "missing-job"},
        headers={"Authorization": "Bearer x"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "gone"
    assert ("find", False) in seen, (
        f"_find_cron_job_profile must run off the event loop; proof: {seen}"
    )


def test_blueprint_instantiate_create_job_off_loop(monkeypatch, loop_probe):
    seen, probe = loop_probe

    def fake_call(profile, fn, *args, **kwargs):
        probe("call")
        return {"id": "bp-job-1", "kwargs_seen": sorted(kwargs.keys())}

    monkeypatch.setattr(web_server, "_call_cron_for_profile", fake_call)
    monkeypatch.setattr(web_server, "_has_valid_session_token", lambda req: True)

    import cron.blueprint_catalog as bc
    monkeypatch.setattr(bc, "get_blueprint", lambda key: object())
    monkeypatch.setattr(
        bc,
        "fill_blueprint",
        lambda bp, vals: {"name": "t", "schedule": "0 9 * * *", "prompt": "hi"},
    )

    client = TestClient(web_server.app)
    resp = client.post(
        "/api/cron/blueprints/instantiate",
        json={"blueprint": "morning-brief", "values": {}},
    )
    assert resp.status_code == 200
    body = resp.json()
    # **spec kwargs must arrive at create_job intact through the partial.
    assert body["kwargs_seen"] == ["name", "prompt", "schedule"]
    assert ("call", False) in seen, (
        f"_call_cron_for_profile must run off the event loop; proof: {seen}"
    )


def test_blueprint_instantiate_reports_saved_but_unregistered(monkeypatch):
    """The instantiate endpoint maps a registration partial-failure to 424.

    Endpoint-level guard for the shared ``_raise_if_cron_registration_error``
    seam — the unit tests cover ``_create_cron_job_sync``; this proves the
    blueprint route surfaces the same structured envelope through FastAPI.
    """
    from cron.scheduler import CronSchedulerRegistrationError

    failure = CronSchedulerRegistrationError(
        {"id": "bp-saved-job", "name": "bp job"},
        RuntimeError("private callback URL and token"),
    )

    def fail_call(profile, fn, *args, **kwargs):
        raise failure

    monkeypatch.setattr(web_server, "_call_cron_for_profile", fail_call)
    monkeypatch.setattr(web_server, "_has_valid_session_token", lambda req: True)

    import cron.blueprint_catalog as bc
    monkeypatch.setattr(bc, "get_blueprint", lambda key: object())
    monkeypatch.setattr(
        bc,
        "fill_blueprint",
        lambda bp, vals: {"name": "t", "schedule": "0 9 * * *", "prompt": "hi"},
    )

    client = TestClient(web_server.app)
    resp = client.post(
        "/api/cron/blueprints/instantiate",
        json={"blueprint": "morning-brief", "values": {}},
    )
    assert resp.status_code == 424
    detail = resp.json()["detail"]
    assert detail["job_id"] == "bp-saved-job"
    assert detail["job_saved"] is True
    assert detail["scheduler_registered"] is False
    assert detail["retry_create"] is False
    assert "private callback URL and token" not in detail["error"]
