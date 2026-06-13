"""Tests for the LOCAL GSSAI vault-sync dashboard plugin."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "vault-sync" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_vault_sync_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin_module():
    return _load_plugin_module()


@pytest.fixture
def client(tmp_path, monkeypatch, plugin_module):
    vault = tmp_path / "obsidian-vault"
    tmp = tmp_path / "ivan-tmp"
    canonical = vault / "06-sops" / "phase-2-1-execution-checklist-approved-cleanup-with-controls.md"
    temp_copy = tmp / "phase2_1_execution_checklist_approved_cleanup_with_controls.md"
    canonical.parent.mkdir(parents=True)
    tmp.mkdir(parents=True)
    for rel in ("09-logs", "13-runtime-state", "agent-ecosystem"):
        (vault / rel).mkdir(parents=True)
    content = "LOCAL GSSAI PROJECT ONLY\nNo secrets here.\n"
    canonical.write_text(content)
    temp_copy.write_text(content)

    monkeypatch.setenv("VAULT_SYNC_VAULT_ROOT", str(vault))
    monkeypatch.setenv("VAULT_SYNC_TMP_ROOT", str(tmp))

    app = FastAPI()
    app.include_router(plugin_module.router, prefix="/api/plugins/vault-sync")
    return TestClient(app)


def test_status_reports_roots_read_only(client):
    response = client.get("/api/plugins/vault-sync/status")
    assert response.status_code == 200
    data = response.json()
    assert data["scope"] == "LOCAL GSSAI PROJECT ONLY"
    assert data["vault_root_exists"] is True
    assert "no sync/move/delete" in data["policy"]


def test_artefacts_are_metadata_only_and_duplicate_is_flagged(client):
    response = client.get("/api/plugins/vault-sync/artefacts")
    assert response.status_code == 200
    data = response.json()
    rendered = response.text
    assert data["temporary_copy_duplicates_canonical"] is True
    assert data["canonical"]["exists"] is True
    assert data["temporary_copy"]["exists"] is True
    assert "LOCAL GSSAI PROJECT ONLY\\nNo secrets here" not in rendered
    assert "sha256" in data["canonical"]


def test_duplicates_endpoint_reports_review_only(client):
    response = client.get("/api/plugins/vault-sync/duplicates")
    assert response.status_code == 200
    duplicate = response.json()["duplicates"][0]
    assert duplicate["same_sha256"] is True
    assert duplicate["action_required"] == "review before any move/delete"


def test_artefacts_non_matching_temp_copy_is_not_duplicate(tmp_path, monkeypatch, plugin_module):
    vault = tmp_path / "obsidian-vault"
    tmp = tmp_path / "ivan-tmp"
    canonical = vault / plugin_module.CANONICAL_REL
    temp_copy = tmp / plugin_module.TMP_REL
    canonical.parent.mkdir(parents=True)
    tmp.mkdir(parents=True)
    canonical.write_text("canonical\n")
    temp_copy.write_text("different\n")
    monkeypatch.setenv("VAULT_SYNC_VAULT_ROOT", str(vault))
    monkeypatch.setenv("VAULT_SYNC_TMP_ROOT", str(tmp))

    app = FastAPI()
    app.include_router(plugin_module.router, prefix="/api/plugins/vault-sync")
    response = TestClient(app).get("/api/plugins/vault-sync/artefacts")

    assert response.status_code == 200
    assert response.json()["temporary_copy_duplicates_canonical"] is False


def test_metadata_rejects_symlink_and_skips_hash(tmp_path, plugin_module):
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.md"
    target.write_text("safe but symlinked\n")
    link = root / "linked.md"
    link.symlink_to(target)

    meta = plugin_module._metadata(link, (root,))

    assert meta["exists"] is False
    assert meta["allowed"] is False
    assert meta["error"] == "symlink not permitted"
    assert "sha256" not in meta


def test_metadata_rejects_paths_outside_allowlist(tmp_path, plugin_module):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    target = outside / "file.md"
    target.write_text("outside\n")

    meta = plugin_module._metadata(target, (allowed,))

    assert meta["exists"] is False
    assert meta["allowed"] is False
    assert meta["error"] == "path outside allowlist"


def test_sha256_skips_large_files(tmp_path, monkeypatch, plugin_module):
    target = tmp_path / "large.md"
    target.write_bytes(b"x" * 32)
    monkeypatch.setattr(plugin_module, "MAX_HASH_BYTES", 16)

    assert plugin_module._sha256(target) is None


def test_expected_dirs_are_reported(client):
    response = client.get("/api/plugins/vault-sync/expected-dirs")
    assert response.status_code == 200
    dirs = {item["name"]: item["exists"] for item in response.json()["directories"]}
    assert dirs["06-sops"] is True
    assert dirs["09-logs"] is True


def test_registered_routes_are_get_only(plugin_module):
    methods = set()
    for route in plugin_module.router.routes:
        methods.update(getattr(route, "methods", set()))
    assert methods <= {"GET", "HEAD"}
