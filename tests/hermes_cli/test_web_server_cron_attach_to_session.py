"""Regression tests for per-job session attachment through the dashboard API."""

import json

import pytest


@pytest.fixture()
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    hermes_home = get_hermes_home()
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_home / "state.db")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: hermes_home / "profiles")

    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


@pytest.mark.parametrize("attach_to_session", [True, False])
def test_create_cron_job_persists_explicit_attach_to_session(
    client,
    attach_to_session,
):
    from hermes_constants import get_hermes_home

    response = client.post(
        "/api/cron/jobs",
        json={
            "prompt": "follow up in the originating session",
            "schedule": "every 1h",
            "name": f"attach-{str(attach_to_session).lower()}",
            "attach_to_session": attach_to_session,
        },
    )

    assert response.status_code == 200
    job = response.json()
    assert job["attach_to_session"] is attach_to_session

    stored_jobs = json.loads(
        (get_hermes_home() / "cron" / "jobs.json").read_text(encoding="utf-8")
    )["jobs"]
    stored_job = next(item for item in stored_jobs if item["id"] == job["id"])
    assert stored_job["attach_to_session"] is attach_to_session


def test_create_cron_job_omits_unset_attach_to_session(client):
    from hermes_constants import get_hermes_home

    response = client.post(
        "/api/cron/jobs",
        json={
            "prompt": "inherit the global mirror setting",
            "schedule": "every 1h",
            "name": "attach-unset",
        },
    )

    assert response.status_code == 200
    job = response.json()
    assert "attach_to_session" not in job

    stored_jobs = json.loads(
        (get_hermes_home() / "cron" / "jobs.json").read_text(encoding="utf-8")
    )["jobs"]
    stored_job = next(item for item in stored_jobs if item["id"] == job["id"])
    assert "attach_to_session" not in stored_job
