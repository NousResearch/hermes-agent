"""Behavior contract for provenance on ``GET /api/profiles``."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def client(_isolate_hermes_home):
    TestClient = pytest.importorskip("starlette.testclient").TestClient

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    test_client = TestClient(app, raise_server_exceptions=False)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def test_profiles_list_reports_canonical_provenance(client, monkeypatch, tmp_path):
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_routers.profiles import _profile_to_dict

    info = SimpleNamespace(
        name="default", path=tmp_path, is_default=True,
        model="example/model", provider="example", skill_count=7,
        description="Profile metadata", display_name="Example",
        ui_meta={"extension": {"nested": ["preserved"]}},
    )

    def list_profiles(*, lazy_skill_count):
        assert lazy_skill_count is True
        return [info]

    monkeypatch.setattr(profiles_mod, "list_profiles", list_profiles)

    response = client.get("/api/profiles")

    assert response.status_code == 200
    assert response.json()["profiles"] == [_profile_to_dict(info)]
    assert response.json()["provenance"] == {
        "source": "canonical",
        "degraded": False,
    }


@pytest.mark.parametrize("failure", ["listing", "conversion"])
def test_profiles_list_marks_filesystem_fallback_degraded(client, monkeypatch, tmp_path, failure):
    from hermes_cli import profiles as profiles_mod

    default_home = tmp_path / "default"
    profiles_root = tmp_path / "profiles"
    named_home = profiles_root / "worker"
    default_home.mkdir()
    named_home.mkdir(parents=True)

    monkeypatch.setattr(profiles_mod, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles_mod, "_get_profiles_root", lambda: profiles_root)

    def list_profiles(**_kwargs):
        if failure == "listing":
            raise RuntimeError("profile listing failed")
        return [SimpleNamespace(skill_count="not-an-integer")]

    monkeypatch.setattr(profiles_mod, "list_profiles", list_profiles)

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


def test_profiles_list_fallback_failure_returns_stable_error(client, monkeypatch, tmp_path):
    from hermes_cli import profiles as profiles_mod
    from hermes_cli import web_server_profiles

    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir()
    monkeypatch.setattr(profiles_mod, "_get_default_hermes_home", lambda: tmp_path / "missing")
    monkeypatch.setattr(profiles_mod, "_get_profiles_root", lambda: profiles_root)

    def failed_listing(**_kwargs):
        raise RuntimeError("synthetic-canonical-secret")

    original_scan = web_server_profiles.os.scandir

    def failed_scan(path):
        if str(path) == str(profiles_root):
            raise PermissionError("/synthetic/private/profile synthetic-scan-secret")
        return original_scan(path)

    monkeypatch.setattr(profiles_mod, "list_profiles", failed_listing)
    monkeypatch.setattr(web_server_profiles.os, "scandir", failed_scan)

    response = client.get("/api/profiles")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Unable to list profiles."}
    assert "synthetic" not in response.text
