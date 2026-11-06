"""Gateway source-context consent tests for plugin tool registration."""

import json

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class TestPluginGatewaySourceContext:
    @pytest.mark.parametrize("value,expected", [(True, True), (False, False), ("true", False), (None, False)])
    def test_gateway_source_consent_uses_profile_config_and_real_host_route(self, tmp_path, monkeypatch, value, expected):
        from tools.registry import registry
        home = tmp_path / "profiles" / "consent"
        home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        config = {"plugins": {"isolation": "host", "entries": {"consent-plugin": {}}}}
        if value is not None:
            config["plugins"]["entries"]["consent-plugin"]["allow_gateway_source_context"] = value
        (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
        mgr = PluginManager()
        manifest = PluginManifest(name="consent-plugin", source="user")
        monkeypatch.delenv("HERMES_PLUGIN_HOST_PROCESS", raising=False)
        ctx = PluginContext(manager=mgr, manifest=manifest)
        name = "_test_gateway_consent"
        try:
            ctx.register_tool(name=name, toolset="test", schema={"name": name, "description": "test", "parameters": {"type": "object", "properties": {}}}, handler=lambda args, **kw: "ok", requires_gateway_source_context=True)
            entry = registry.get_entry(name, scope=mgr.scope_key)
            assert entry is not None
            assert entry.gateway_source_consent() is expected
            assert entry.requires_gateway_source_context is True
        finally:
            mgr.unload(manifest.name)

    @pytest.mark.parametrize("route,source,expected", [("inproc", "user", False), ("host", "bundled", False)])
    def test_gateway_source_consent_denies_untrusted_route(self, tmp_path, monkeypatch, route, source, expected):
        from tools.registry import registry
        home = tmp_path / "profile"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        isolation = "host" if source == "bundled" else "in_process"
        (home / "config.yaml").write_text(json.dumps({"plugins": {"isolation": isolation, "entries": {"consent-plugin": {"allow_gateway_source_context": True}}}}), encoding="utf-8")
        monkeypatch.delenv("HERMES_PLUGIN_HOST_PROCESS", raising=False)
        mgr = PluginManager()
        manifest = PluginManifest(name="consent-plugin", source=source)
        ctx = PluginContext(manager=mgr, manifest=manifest)
        name = "_test_gateway_consent_route"
        try:
            ctx.register_tool(name=name, toolset="test", schema={"name": name, "description": "test", "parameters": {"type": "object", "properties": {}}}, handler=lambda args, **kw: "ok", requires_gateway_source_context=True)
            assert registry.get_entry(name, scope=mgr.scope_key).gateway_source_consent() is expected
        finally:
            mgr.unload(manifest.name)

    @pytest.mark.parametrize("failure", ["config", "route"])
    def test_gateway_source_consent_fails_closed_on_exceptions(self, tmp_path, monkeypatch, failure):
        from tools.registry import registry
        home = tmp_path / "profile"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        (home / "config.yaml").write_text(json.dumps({"plugins": {"isolation": "host", "entries": {"consent-plugin": {"allow_gateway_source_context": True}}}}), encoding="utf-8")
        monkeypatch.delenv("HERMES_PLUGIN_HOST_PROCESS", raising=False)
        mgr = PluginManager()
        manifest = PluginManifest(name="consent-plugin", source="user")
        if failure == "config":
            from hermes_cli import plugins as plugins_module
            monkeypatch.setattr(plugins_module, "load_config_readonly", lambda: (_ for _ in ()).throw(RuntimeError("config failed")))
        else:
            def raise_route(_manifest):
                raise RuntimeError("route failed")
            monkeypatch.setattr(mgr, "_runs_in_plugin_host", raise_route)
        ctx = PluginContext(manager=mgr, manifest=manifest)
        name = "_test_gateway_consent_exception"
        try:
            ctx.register_tool(name=name, toolset="test", schema={"name": name, "description": "test", "parameters": {"type": "object", "properties": {}}}, handler=lambda args, **kw: "ok", requires_gateway_source_context=True)
            assert registry.get_entry(name, scope=mgr.scope_key).gateway_source_consent() is False
        finally:
            mgr.unload(manifest.name)

    @pytest.mark.parametrize("flag,required", [(True, True), (False, False), ("true", False), (1, False)])
    def test_gateway_source_requirement_is_exact_boolean(self, tmp_path, monkeypatch, flag, required):
        from tools.registry import registry
        home = tmp_path / "profile"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        mgr = PluginManager()
        manifest = PluginManifest(name="consent-plugin", source="user")
        ctx = PluginContext(manager=mgr, manifest=manifest)
        name = "_test_gateway_requirement"
        try:
            ctx.register_tool(name=name, toolset="test", schema={"name": name, "description": "test", "parameters": {"type": "object", "properties": {}}}, handler=lambda args, **kw: "ok", requires_gateway_source_context=flag)
            assert registry.get_entry(name, scope=mgr.scope_key).requires_gateway_source_context is required
        finally:
            mgr.unload(manifest.name)
