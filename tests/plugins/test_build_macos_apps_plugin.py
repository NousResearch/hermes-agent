"""Bundled-plugin discovery tests for build-macos-apps."""

import yaml


class TestBundledBuildMacosAppsDiscovery:
    def _write_enabled_config(self, hermes_home, names):
        cfg_path = hermes_home / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"plugins": {"enabled": list(names)}}))

    def test_discovered_but_not_loaded_by_default(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        from hermes_cli import plugins as pmod

        mgr = pmod.PluginManager()
        mgr.discover_and_load()

        assert "build-macos-apps" in mgr._plugins
        loaded = mgr._plugins["build-macos-apps"]
        assert loaded.manifest.source == "bundled"
        assert loaded.manifest.kind == "standalone"
        assert not loaded.enabled
        assert loaded.error and "not enabled" in loaded.error

    def test_loads_when_enabled_and_registers_tools(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        self._write_enabled_config(hermes_home, ["build-macos-apps"])

        from hermes_cli import plugins as pmod

        mgr = pmod.PluginManager()
        mgr.discover_and_load()

        loaded = mgr._plugins["build-macos-apps"]
        assert loaded.enabled
        assert sorted(loaded.tools_registered) == [
            "macos_build_project",
            "macos_collect_crash_reports",
            "macos_find_app_bundle",
            "macos_inspect_project",
            "macos_list_schemes",
            "macos_read_recent_logs",
            "macos_run_app",
            "macos_show_build_settings",
            "macos_stop_app",
            "macos_test_project",
        ]
        assert mgr.list_plugin_skills("build-macos-apps") == [
            "diagnose-build-failure",
            "local-run-loop-check",
        ]

    def test_toolset_is_visible_when_enabled(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        self._write_enabled_config(hermes_home, ["build-macos-apps"])

        from hermes_cli import plugins as plugins_mod
        from model_tools import get_tool_definitions

        mgr = plugins_mod.PluginManager()
        mgr.discover_and_load()
        monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
        import sys

        plugin_tools = sys.modules["hermes_plugins.build_macos_apps.tools"]
        monkeypatch.setattr(plugin_tools, "check_macos_dev_requirements", lambda: True)

        defs = get_tool_definitions(enabled_toolsets=["macos-dev"], quiet_mode=True)
        tool_names = sorted(item["function"]["name"] for item in defs)
        assert tool_names == [
            "macos_build_project",
            "macos_collect_crash_reports",
            "macos_find_app_bundle",
            "macos_inspect_project",
            "macos_list_schemes",
            "macos_read_recent_logs",
            "macos_run_app",
            "macos_show_build_settings",
            "macos_stop_app",
            "macos_test_project",
        ]

    def test_plugin_skill_view_resolves_registered_skill(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        self._write_enabled_config(hermes_home, ["build-macos-apps"])

        from hermes_cli import plugins as plugins_mod
        from tools.skills_tool import skill_view

        mgr = plugins_mod.PluginManager()
        mgr.discover_and_load()
        monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)

        result = yaml.safe_load(skill_view("build-macos-apps:diagnose-build-failure"))

        assert result["success"] is True
        assert result["name"] == "build-macos-apps:diagnose-build-failure"
        assert "macos_show_build_settings" in result["content"]
