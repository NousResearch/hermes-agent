"""Tests for the dashboard plugin-catalog surface in hermes_cli.web_server.

Covers:
- GET /api/dashboard/plugins/catalog — entry serialization, installed-state
  merge (via the ``.hermes-catalog.json`` sidecar), removed list exposure.
- POST /api/dashboard/agent-plugins/install — removed-blocklist refusal for
  raw identifiers AND catalog names, catalog_name resolution to a pinned-ref
  install, sidecar write.
- /api/dashboard/plugins/hub — ``removed_reason`` annotation on rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

VALID_SHA = "38fe0fb53eff98d477f807432e965429e665ca33"
OTHER_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _write_entry(catalog_dir: Path, name: str, **overrides) -> dict:
    data = {
        "name": name,
        "repo": f"https://github.com/example/{name}",
        "sha": VALID_SHA,
        "description": f"Test entry {name}.",
        "maintainer": "Example",
        "tier": "official",
        "docs_url": f"https://example.com/docs/{name}",
        "capabilities": {
            "provides_tools": ["tool_a"],
            "provides_hooks": ["hook_b"],
            "provides_middleware": [],
            "requires_env": ["EXAMPLE_API_KEY"],
        },
    }
    data.update(overrides)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / f"{name}.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8"
    )
    return data


def _write_removed(catalog_dir: Path, removed: list) -> None:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "removed.yaml").write_text(
        yaml.safe_dump({"removed": removed}), encoding="utf-8"
    )


def _make_installed_plugin(name: str, sidecar: dict | None = None) -> Path:
    """Drop a minimal plugin dir under the isolated HERMES_HOME."""
    from hermes_constants import get_hermes_home

    plugin_dir = get_hermes_home() / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "1.0", "description": "x"}),
        encoding="utf-8",
    )
    if sidecar is not None:
        (plugin_dir / ".hermes-catalog.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
    return plugin_dir


class TestDashboardPluginCatalog:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, tmp_path, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db"
        )

        self.catalog_dir = tmp_path / "catalog"
        self.catalog_dir.mkdir()
        monkeypatch.setenv("HERMES_PLUGIN_CATALOG_DIR", str(self.catalog_dir))

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    # ── GET /api/dashboard/plugins/catalog ──────────────────────────────

    def test_catalog_endpoint_requires_token(self):
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app

        unauth = TestClient(app)
        resp = unauth.get("/api/dashboard/plugins/catalog")
        assert resp.status_code == 401

    def test_catalog_endpoint_shape(self):
        _write_entry(self.catalog_dir, "alpha-plugin")
        _write_removed(
            self.catalog_dir,
            [{"name": "bad-plugin", "repo": "https://github.com/evil/bad-plugin",
              "reason": "exfiltrated env vars", "date": "2026-07-02"}],
        )

        resp = self.client.get("/api/dashboard/plugins/catalog")
        assert resp.status_code == 200
        data = resp.json()

        assert "generated_at" in data
        assert isinstance(data["entries"], list) and len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["name"] == "alpha-plugin"
        assert entry["repo"] == "https://github.com/example/alpha-plugin"
        assert entry["sha"] == VALID_SHA
        assert entry["sha_short"] == VALID_SHA[:7]
        assert entry["tier"] == "official"
        assert entry["maintainer"] == "Example"
        assert entry["docs_url"] == "https://example.com/docs/alpha-plugin"
        assert entry["capabilities"]["provides_tools"] == ["tool_a"]
        assert entry["capabilities"]["requires_env"] == ["EXAMPLE_API_KEY"]
        assert "tool_a" in entry["capability_summary"]
        # Not installed → degraded install state.
        assert entry["installed"] is False
        assert entry["installed_sha"] is None
        assert entry["update_available"] is False
        assert entry["runtime_status"] is None

        assert len(data["removed"]) == 1
        removed = data["removed"][0]
        assert removed["name"] == "bad-plugin"
        assert removed["reason"] == "exfiltrated env vars"

    def test_catalog_installed_state_merge_with_sidecar(self):
        _write_entry(self.catalog_dir, "alpha-plugin")
        _make_installed_plugin(
            "alpha-plugin",
            sidecar={
                "catalog_name": "alpha-plugin",
                "repo": "https://github.com/example/alpha-plugin",
                "sha": OTHER_SHA,
                "installed_at": "2026-07-01T00:00:00Z",
                "tier": "official",
            },
        )

        resp = self.client.get("/api/dashboard/plugins/catalog")
        assert resp.status_code == 200
        entry = resp.json()["entries"][0]
        assert entry["installed"] is True
        assert entry["installed_sha"] == OTHER_SHA
        assert entry["update_available"] is True
        assert entry["runtime_status"] == "inactive"

    def test_catalog_installed_no_sidecar_degrades_to_null_sha(self):
        _write_entry(self.catalog_dir, "alpha-plugin")
        _make_installed_plugin("alpha-plugin", sidecar=None)

        resp = self.client.get("/api/dashboard/plugins/catalog")
        entry = resp.json()["entries"][0]
        assert entry["installed"] is True
        assert entry["installed_sha"] is None
        assert entry["update_available"] is False

    def test_catalog_installed_same_sha_no_update(self):
        _write_entry(self.catalog_dir, "alpha-plugin")
        _make_installed_plugin(
            "alpha-plugin",
            sidecar={
                "catalog_name": "alpha-plugin",
                "repo": "https://github.com/example/alpha-plugin",
                "sha": VALID_SHA,
                "installed_at": "2026-07-01T00:00:00Z",
                "tier": "official",
            },
        )

        entry = self.client.get("/api/dashboard/plugins/catalog").json()["entries"][0]
        assert entry["installed"] is True
        assert entry["installed_sha"] == VALID_SHA
        assert entry["update_available"] is False

    # ── POST /api/dashboard/agent-plugins/install ───────────────────────

    def test_install_refuses_removed_raw_identifier(self):
        _write_removed(
            self.catalog_dir,
            [{"name": "bad-plugin", "repo": "https://github.com/evil/bad-plugin",
              "reason": "exfiltrated env vars", "date": "2026-07-02"}],
        )
        resp = self.client.post(
            "/api/dashboard/agent-plugins/install",
            json={"identifier": "https://github.com/evil/bad-plugin"},
        )
        assert resp.status_code == 400
        assert "exfiltrated env vars" in resp.json()["detail"]

    def test_install_refuses_removed_catalog_name(self):
        _write_removed(
            self.catalog_dir,
            [{"name": "bad-plugin", "reason": "policy violation",
              "date": "2026-07-02"}],
        )
        resp = self.client.post(
            "/api/dashboard/agent-plugins/install",
            json={"identifier": "", "catalog_name": "bad-plugin"},
        )
        assert resp.status_code == 400
        assert "policy violation" in resp.json()["detail"]

    def test_install_unknown_catalog_name_is_400(self):
        resp = self.client.post(
            "/api/dashboard/agent-plugins/install",
            json={"identifier": "", "catalog_name": "does-not-exist"},
        )
        assert resp.status_code == 400
        assert "does-not-exist" in resp.json()["detail"]

    def test_install_missing_identifier_and_catalog_name_is_400(self):
        resp = self.client.post(
            "/api/dashboard/agent-plugins/install",
            json={"identifier": ""},
        )
        assert resp.status_code == 400

    def test_catalog_name_install_resolves_pinned_ref_and_writes_sidecar(
        self, monkeypatch, tmp_path
    ):
        from hermes_constants import get_hermes_home
        import hermes_cli.plugins_cmd as plugins_cmd

        _write_entry(self.catalog_dir, "alpha-plugin")

        captured = {}

        def fake_core(identifier, *, force, ref=None, skip_removed_check=False):
            captured["identifier"] = identifier
            captured["ref"] = ref
            target = get_hermes_home() / "plugins" / "alpha-plugin"
            target.mkdir(parents=True, exist_ok=True)
            (target / "plugin.yaml").write_text(
                yaml.safe_dump({"name": "alpha-plugin"}), encoding="utf-8"
            )
            return target, {"name": "alpha-plugin"}, "alpha-plugin"

        monkeypatch.setattr(plugins_cmd, "_install_plugin_core", fake_core)

        resp = self.client.post(
            "/api/dashboard/agent-plugins/install",
            json={"identifier": "", "catalog_name": "alpha-plugin",
                  "enable": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["plugin_name"] == "alpha-plugin"

        assert captured["ref"] == VALID_SHA
        assert captured["identifier"].startswith(
            "https://github.com/example/alpha-plugin"
        )

        sidecar_path = (
            get_hermes_home() / "plugins" / "alpha-plugin" / ".hermes-catalog.json"
        )
        assert sidecar_path.is_file()
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["catalog_name"] == "alpha-plugin"
        assert sidecar["repo"] == "https://github.com/example/alpha-plugin"
        assert sidecar["sha"] == VALID_SHA
        assert sidecar["tier"] == "official"
        assert sidecar["installed_at"]

    # ── /api/dashboard/plugins/hub removed_reason annotation ─────────────

    def test_hub_rows_annotated_with_removed_reason(self):
        _write_removed(
            self.catalog_dir,
            [{"name": "bad-plugin", "reason": "supply chain incident",
              "date": "2026-07-02"}],
        )
        _make_installed_plugin("bad-plugin")
        _make_installed_plugin("good-plugin")

        resp = self.client.get("/api/dashboard/plugins/hub")
        assert resp.status_code == 200
        rows = {r["name"]: r for r in resp.json()["plugins"]}
        assert rows["bad-plugin"]["removed_reason"] == "supply chain incident"
        assert rows["good-plugin"]["removed_reason"] is None
