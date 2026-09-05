"""Behavior contract for provenance on ``GET /api/profiles``."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def test_profiles_list_reports_canonical_provenance(client, monkeypatch, tmp_path):
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(
        profiles_mod,
        "list_profiles",
        lambda: [
            SimpleNamespace(
                name="default",
                path=tmp_path,
                is_default=True,
            )
        ],
    )

    response = client.get("/api/profiles")

    assert response.status_code == 200
    assert [profile["name"] for profile in response.json()["profiles"]] == ["default"]
    assert response.json()["provenance"] == {
        "source": "canonical",
        "degraded": False,
    }


def test_profiles_list_marks_filesystem_fallback_degraded(client, monkeypatch, tmp_path):
    from hermes_cli import profiles as profiles_mod

    default_home = tmp_path / "default"
    profiles_root = tmp_path / "profiles"
    named_home = profiles_root / "worker"
    default_home.mkdir()
    named_home.mkdir(parents=True)

    monkeypatch.setattr(profiles_mod, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles_mod, "_get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr(
        profiles_mod,
        "list_profiles",
        lambda: (_ for _ in ()).throw(RuntimeError("profile listing failed")),
    )

    response = client.get("/api/profiles")

    assert response.status_code == 200
    assert [profile["name"] for profile in response.json()["profiles"]] == [
        "default",
        "worker",
    ]
    assert response.json()["provenance"] == {
        "source": "filesystem_fallback",
        "degraded": True,
    }
