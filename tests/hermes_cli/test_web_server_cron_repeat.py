"""Regression coverage for finite repeat counts in the dashboard cron API."""

import pytest


@pytest.fixture()
def isolated_profiles(tmp_path, monkeypatch, _isolate_hermes_home):
    """Route dashboard cron storage to an isolated default profile."""
    from hermes_cli import profiles

    default_home = tmp_path / ".hermes"
    (default_home / "cron").mkdir(parents=True)
    (default_home / "config.yaml").write_text(
        "model: test-model\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: default_home / "profiles")
    return default_home


@pytest.fixture()
def client(monkeypatch, isolated_profiles):
    from starlette.testclient import TestClient

    import hermes_state
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(
        hermes_state,
        "DEFAULT_DB_PATH",
        isolated_profiles / "state.db",
    )
    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def test_create_preserves_finite_repeat_count(client):
    response = client.post(
        "/api/cron/jobs",
        json={
            "prompt": "run the check",
            "schedule": "every 1h",
            "name": "two checks",
            "repeat": 2,
        },
    )

    assert response.status_code == 200
    assert response.json()["repeat"] == {"times": 2, "completed": 0}


def test_create_without_repeat_remains_unlimited(client):
    response = client.post(
        "/api/cron/jobs",
        json={
            "prompt": "run forever",
            "schedule": "every 1h",
        },
    )

    assert response.status_code == 200
    assert response.json()["repeat"] == {"times": None, "completed": 0}


@pytest.mark.parametrize("repeat", [0, -1, True, 1.5, "2", "not-a-count"])
def test_create_rejects_invalid_repeat_count(client, repeat):
    response = client.post(
        "/api/cron/jobs",
        json={
            "prompt": "must not run forever",
            "schedule": "every 1h",
            "repeat": repeat,
        },
    )

    assert response.status_code == 422
    assert any(
        error["loc"][-1] == "repeat"
        for error in response.json()["detail"]
    )
    assert client.get(
        "/api/cron/jobs",
        params={"profile": "default"},
    ).json() == []
