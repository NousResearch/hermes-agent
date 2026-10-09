"""Regression coverage for finite repeat counts in the dashboard cron create API (#68012)."""

import pytest


@pytest.fixture()
def isolated_profiles(tmp_path, monkeypatch, _isolate_hermes_home):
    """Route dashboard cron storage to an isolated default profile."""
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    (default_home / "cron").mkdir(parents=True, exist_ok=True)
    (default_home / "config.yaml").write_text(
        "model: test-model\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("profiles.paths._get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr("profiles.paths._get_profiles_root", lambda: default_home / "profiles")
    return default_home


@pytest.fixture()
def client(monkeypatch, isolated_profiles):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(
        hermes_state,
        "DEFAULT_DB_PATH",
        get_hermes_home() / "state.db",
    )
    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def test_create_preserves_finite_repeat_count(client):
    """A finite repeat must survive request parsing and storage (route-level persistence)."""
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

    listed = client.get("/api/cron/jobs", params={"profile": "default"}).json()
    assert [job["repeat"] for job in listed if job["name"] == "two checks"] == [
        {"times": 2, "completed": 0}
    ]


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


@pytest.mark.parametrize("repeat", [0, -1, 1.5, "not-a-count"])
def test_create_rejects_invalid_repeat_count(client, repeat):
    """Invalid counts must 4xx instead of silently becoming unlimited jobs."""
    response = client.post(
        "/api/cron/jobs",
        json={
            "prompt": "must not run forever",
            "schedule": "every 1h",
            "repeat": repeat,
        },
    )

    assert response.status_code in (400, 422)
    detail = response.json()["detail"]
    if isinstance(detail, str):
        assert "repeat" in detail.lower()
    else:
        assert any(
            error.get("loc", [None])[-1] == "repeat" or "repeat" in str(error).lower()
            for error in detail
        )
    assert client.get(
        "/api/cron/jobs",
        params={"profile": "default"},
    ).json() == []


@pytest.mark.parametrize("repeat,expected_times", [("2", 2), ("once", 1), ("forever", None)])
def test_create_accepts_cli_string_repeat_forms(client, repeat, expected_times):
    """The dashboard create path validates repeat through the same normalize_repeat_value
    the CLI uses: numeric strings coerce, 'once' → 1, 'forever' → unlimited."""
    response = client.post(
        "/api/cron/jobs",
        json={
            "prompt": "string repeat form",
            "schedule": "every 1h",
            "repeat": repeat,
        },
    )

    assert response.status_code == 200
    assert response.json()["repeat"] == {"times": expected_times, "completed": 0}
