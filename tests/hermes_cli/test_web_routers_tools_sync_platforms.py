"""Sync-to-all-platforms endpoint (dashboard/desktop).

``POST /api/tools/toolsets/sync-platforms`` applies the desktop/CLI toolset
selection to every other enabled platform — the non-interactive sibling of the
``hermes tools`` "Configure all platforms (global)" menu entry
(``_configure_platforms(..., all_platforms=True)``). Platform restrictions
(``discord``/``discord_admin`` are discord-only) filter the selection per
platform, and toolsets still missing provider/API-key setup are reported as
``needs_setup`` instead of interactively prompted.
"""

import pytest
import hermes_cli.web_server_gateway as _web_server_gateway


class TestSyncToolsetsToPlatforms:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, _isolate_hermes_home):
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
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        self.monkeypatch = monkeypatch

    def _set_platforms(self, platforms):
        import hermes_cli.tools_config as tools_config

        monkeypatch_targets = [
            (tools_config, "_get_enabled_platforms", lambda: platforms)]
        self.monkeypatch.setattr(*monkeypatch_targets[0])

    def test_sync_applies_cli_selection_to_other_platforms(self, monkeypatch):
        import hermes_cli.tools_config as tools_config
        from hermes_cli.config import load_config

        # cli has web enabled; telegram starts from an explicitly empty list.
        monkeypatch.setattr(tools_config, "_get_enabled_platforms", lambda: ["cli", "telegram"])
        from hermes_cli.config import save_config
        config = load_config()
        config["platform_toolsets"] = {"cli": ["web"], "telegram": []}
        save_config(config)

        resp = self.client.post("/api/tools/toolsets/sync-platforms", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["source_platform"] == "cli"
        assert data["platforms"] == ["telegram"]
        assert len(data["synced"]) == 1
        entry = data["synced"][0]
        assert entry["platform"] == "telegram"
        assert entry["changed"] is True
        assert "web" in entry["added"]

        # The saved telegram list now matches the cli selection.
        saved = load_config()["platform_toolsets"]["telegram"]
        assert "web" in saved

    def test_sync_reports_platform_restrictions(self, monkeypatch):
        import hermes_cli.tools_config as tools_config
        from hermes_cli.config import load_config, save_config

        # discord-only toolsets must not leak into the telegram list.
        monkeypatch.setattr(tools_config, "_get_enabled_platforms", lambda: ["cli", "telegram"])
        config = load_config()
        config["platform_toolsets"] = {"cli": ["web", "discord"], "telegram": []}
        save_config(config)

        resp = self.client.post("/api/tools/toolsets/sync-platforms", json={})
        assert resp.status_code == 200
        entry = resp.json()["synced"][0]
        assert entry["changed"] is True
        assert "web" in entry["added"]
        assert "discord" not in entry["enabled"]

    def test_sync_reports_needs_setup_for_unconfigured_toolsets(self, monkeypatch):
        import hermes_cli.tools_config as tools_config
        from hermes_cli.config import load_config, save_config

        monkeypatch.setattr(tools_config, "_get_enabled_platforms", lambda: ["cli", "telegram"])
        config = load_config()
        config["platform_toolsets"] = {"cli": ["web"], "telegram": []}
        save_config(config)

        # No provider configured for web → it lands in needs_setup after a sync.
        monkeypatch.setattr(
            tools_config, "_toolset_needs_configuration_prompt",
            lambda ts_key, config, force_fresh=False: ts_key == "web")

        resp = self.client.post("/api/tools/toolsets/sync-platforms", json={})
        assert resp.status_code == 200
        assert resp.json()["needs_setup"] == ["web"]

    def test_sync_without_other_platforms_is_a_noop(self, monkeypatch):
        import hermes_cli.tools_config as tools_config
        from hermes_cli.config import load_config, save_config

        monkeypatch.setattr(tools_config, "_get_enabled_platforms", lambda: ["cli"])
        config = load_config()
        config["platform_toolsets"] = {"cli": ["web"]}
        save_config(config)

        resp = self.client.post("/api/tools/toolsets/sync-platforms", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["platforms"] == []
        assert data["synced"] == []

    def test_sync_unchanged_platform_reported_without_resave(self, monkeypatch):
        import hermes_cli.tools_config as tools_config
        from hermes_cli.config import load_config, save_config

        monkeypatch.setattr(tools_config, "_get_enabled_platforms", lambda: ["cli", "telegram"])
        config = load_config()
        config["platform_toolsets"] = {"cli": ["web"], "telegram": ["web"]}
        save_config(config)

        saves = []
        real_save = tools_config._save_platform_tools

        def _counting_save(cfg, platform, enabled):
            saves.append(platform)
            return real_save(cfg, platform, enabled)

        monkeypatch.setattr(tools_config, "_save_platform_tools", _counting_save)

        resp = self.client.post("/api/tools/toolsets/sync-platforms", json={})
        assert resp.status_code == 200
        entry = resp.json()["synced"][0]
        assert entry["changed"] is False
        assert entry["added"] == []
        assert saves == []
