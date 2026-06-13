"""Tests for the LOCAL GSSAI profile-manager dashboard plugin."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "profile-manager" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_profile_manager_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_plugin_router():
    return _load_plugin_module().router


@pytest.fixture
def client(tmp_path, monkeypatch):
    hermes_root = tmp_path / ".hermes"
    profiles_root = hermes_root / "profiles"
    ivan = profiles_root / "ivan_bb"
    aria = profiles_root / "aria-vps"
    archived = profiles_root / "_archived" / "business"
    unrelated_archived = profiles_root / "_archived" / "client-x"

    (ivan / "logs").mkdir(parents=True)
    (aria / "logs").mkdir(parents=True)
    archived.mkdir(parents=True)
    unrelated_archived.mkdir(parents=True)

    (ivan / ".env").write_text("TELEGRAM_BOT_TOKEN=123456:SECRET_SHOULD_NOT_LEAK\n")
    (ivan / "config.yaml").write_text("model:\n  default: test\n")
    (ivan / "SOUL.md").write_text("# Ivan\n")
    (ivan / "state.db").write_bytes(b"sqlite-ish")
    (ivan / "logs" / "gateway.log").write_text("Traceback: should not leak full log lines\n")

    (aria / ".env").write_text("ARIA_SECRET=must-not-appear\n")
    (archived / ".env").write_text("ARCHIVED_SECRET=must-not-appear\n")
    (unrelated_archived / ".env").write_text("CLIENT_SECRET=must-not-appear\n")

    monkeypatch.setenv("PROFILE_MANAGER_HERMES_ROOT", str(hermes_root))

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/profile-manager")
    return TestClient(app)


def test_inventory_is_secret_free_and_file_metadata_only(client):
    response = client.get("/api/plugins/profile-manager/inventory")
    assert response.status_code == 200
    data = response.json()
    rendered = response.text

    assert data["scope"] == "LOCAL GSSAI PROJECT ONLY"
    assert "SECRET_SHOULD_NOT_LEAK" not in rendered
    assert "ARIA_SECRET" not in rendered
    assert "ARCHIVED_SECRET" not in rendered
    assert "CLIENT_SECRET" not in rendered
    assert "Traceback: should not leak" not in rendered

    profiles = {profile["name"]: profile for profile in data["profiles"]}
    assert "ivan_bb" in profiles
    assert profiles["ivan_bb"]["env"]["exists"] is True
    assert profiles["ivan_bb"]["env"]["size"] > 0
    assert profiles["ivan_bb"]["boundary_label"].startswith("LOCAL GSSAI PROJECT")


def test_inventory_includes_allowlisted_archived_profiles_only(client):
    response = client.get("/api/plugins/profile-manager/inventory")
    assert response.status_code == 200
    profiles = {profile["name"]: profile for profile in response.json()["profiles"]}
    assert profiles["business"]["lifecycle"] == "archived"
    assert "client-x" not in profiles


def test_inventory_excludes_side_project_profile_fragments(client):
    response = client.get("/api/plugins/profile-manager/inventory")
    assert response.status_code == 200
    names = {profile["name"] for profile in response.json()["profiles"]}
    assert "aria-vps" not in names


def test_exclusion_does_not_match_parent_path_fragments(tmp_path, monkeypatch):
    hermes_root = tmp_path / "vps-parent-name" / ".hermes"
    profile_path = hermes_root / "profiles" / "ivan_bb"
    profile_path.mkdir(parents=True)
    monkeypatch.setenv("PROFILE_MANAGER_HERMES_ROOT", str(hermes_root))

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/profile-manager")
    response = TestClient(app).get("/api/plugins/profile-manager/inventory")

    assert response.status_code == 200
    names = {profile["name"] for profile in response.json()["profiles"]}
    assert "ivan_bb" in names


def test_guardrails_are_read_only(client):
    response = client.get("/api/plugins/profile-manager/guardrails")
    assert response.status_code == 200
    data = response.json()
    assert data["read_only"] is True
    assert data["secret_values_returned"] is False
    assert ".env" in data["secret_files_metadata_only"]


def test_registered_routes_are_get_only():
    router = _load_plugin_router()
    methods = set()
    for route in router.routes:
        methods.update(getattr(route, "methods", set()))
    assert methods <= {"GET", "HEAD"}
