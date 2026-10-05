"""Config-load plugin-materialization gates: behavior contracts.

Regression tests for the three gates that keep ``load_gateway_config()`` from
materializing every installed platform plugin on every CLI config read:

1. ``shared_loop_targets`` must answer names WITHOUT triggering deferred loads.
2. ``apply_plugin_yaml_hooks`` must resolve only sectioned platforms; a hook
   still fires for a sectioned platform (behavior preserved), and a platform
   with no section never materializes.
3. ``_enable_plugin_platforms_from_env`` must skip a platform whose declared
   env vars (``requires_env`` ∪ ``optional_env``, threaded through
   ``declare_env_keys``) are all unset and which has no config row — without
   importing it — while still enabling the platform when a declared credential
   IS present. An undeclared manifest keeps the legacy full iteration.
"""
import contextlib
import sys

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, PlatformRegistry


def _entry(name, source="plugin", **kw):
    return PlatformEntry(
        name=name, label=name.title(), adapter_factory=lambda cfg: None,
        check_fn=lambda: True, source=source, **kw
    )


class TestRegisteredNamesGate:
    def test_shared_loop_targets_never_triggers_deferred_loads(self):
        from gateway.config_loader import shared_loop_targets

        reg = PlatformRegistry()
        loads = []

        def loader():
            loads.append(1)

        # A name Platform() can resolve (bundled-scan path) so the target survives the
        # same contextlib.suppress(ValueError, KeyError) contract as production names.
        reg.register_deferred("a2a", loader)
        targets = shared_loop_targets(reg)
        assert loads == [], "shared_loop_targets must resolve names without loading adapters"
        assert any(getattr(p, "value", None) == "a2a" for p in targets), \
            "a deferred platform still counts as a shared-loop target"

    def test_registered_names_includes_deferred_without_loading(self):
        reg = PlatformRegistry()
        loads = []
        reg.register_deferred("deferredplat", lambda: loads.append(1))
        assert "deferredplat" in reg.registered_names()
        assert loads == []


class TestEnvKeysMetadata:
    def test_declare_and_lookup_roundtrip(self):
        reg = PlatformRegistry()
        reg.register_deferred("plat", lambda: None)
        reg.declare_env_keys("plat", ["A_TOKEN", "A_URL"])
        assert reg.env_keys("plat") == frozenset({"A_TOKEN", "A_URL"})

    def test_undeclared_platform_yields_empty_keys(self):
        reg = PlatformRegistry()
        assert reg.env_keys("nope") == frozenset()

    def test_manifest_env_keys_reads_requires_and_optional(self):
        from hermes_cli.plugins_loader import _manifest_env_keys
        from hermes_cli.plugins_manifest import PluginManifest

        manifest = PluginManifest(
            name="x-platform",
            requires_env=[{"name": "X_TOKEN", "prompt": "t"}, "X_PLAIN"],
            optional_env=[{"name": "X_URL"}, {"description": "no name key"}],
        )
        assert _manifest_env_keys(manifest) == ["X_TOKEN", "X_PLAIN", "X_URL"]

    def test_optional_env_parsed_from_manifest_data(self, tmp_path):
        from hermes_cli.plugins_manifest import parse_manifest_file

        plugin_dir = tmp_path / "y-platform"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: y-platform\nkind: platform\nrequires_env:\n  - name: Y_TOKEN\n"
            "optional_env:\n  - name: Y_URL\n  - name: Y_HOME_CHANNEL\n"
        )
        manifest = parse_manifest_file(plugin_dir / "plugin.yaml", plugin_dir, "bundled", "")
        assert manifest is not None
        assert [e["name"] for e in manifest.requires_env] == ["Y_TOKEN"]
        assert [e["name"] for e in manifest.optional_env] == ["Y_URL", "Y_HOME_CHANNEL"]


class TestEnablementSkip:
    @pytest.fixture
    def registry(self):
        return PlatformRegistry()

    def test_skip_without_declared_credentials(self, registry, monkeypatch):
        loads = []
        registry.register_deferred("plat", lambda: loads.append(1) or registry.register(
            _entry("plat", is_connected=lambda cfg: True, required_env=["PLAT_TOKEN"])
        ))
        registry.declare_env_keys("plat", ["PLAT_TOKEN"])

        from gateway.config import GatewayConfig

        config = GatewayConfig()

        import gateway.platform_registry as pr

        saved = pr.platform_registry
        pr.platform_registry = registry
        monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
        try:
            from gateway.config_env import _enable_plugin_platforms_from_env

            _enable_plugin_platforms_from_env(config)
        finally:
            pr.platform_registry = saved

        assert loads == [], "platform with unset declared creds must not be imported"
        assert Platform("plat") not in config.platforms

    def test_enable_when_declared_credential_present(self, registry, monkeypatch):
        loads = []
        entry = _entry("plat", is_connected=lambda cfg: True, required_env=["PLAT_TOKEN"])
        registry.register_deferred("plat", lambda: loads.append(1) or registry.register(entry))
        registry.declare_env_keys("plat", ["PLAT_TOKEN"])

        from gateway.config import GatewayConfig

        config = GatewayConfig()
        monkeypatch.setenv("PLAT_TOKEN", "tok")

        import gateway.platform_registry as pr

        saved = pr.platform_registry
        pr.platform_registry = registry
        monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
        try:
            from gateway.config_env import _enable_plugin_platforms_from_env

            _enable_plugin_platforms_from_env(config)
        finally:
            pr.platform_registry = saved

        assert loads, "platform with a set declared credential must be imported and probed"
        pcfg = config.platforms.get(Platform("plat"))
        assert pcfg is not None and pcfg.enabled

    def test_undeclared_manifest_keeps_legacy_iteration(self, registry, monkeypatch):
        loads = []
        entry = _entry("plat", is_connected=lambda cfg: False)
        registry.register_deferred("plat", lambda: loads.append(1) or registry.register(entry))
        # no declare_env_keys call → undeclared

        from gateway.config import GatewayConfig

        config = GatewayConfig()

        import gateway.platform_registry as pr

        saved = pr.platform_registry
        pr.platform_registry = registry
        monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
        try:
            from gateway.config_env import _enable_plugin_platforms_from_env

            _enable_plugin_platforms_from_env(config)
        finally:
            pr.platform_registry = saved

        assert loads, "undeclared platform must keep the legacy import-then-probe path"

    def test_config_row_forces_import(self, registry, monkeypatch):
        loads = []
        entry = _entry("plat", is_connected=lambda cfg: True)
        registry.register_deferred("plat", lambda: loads.append(1) or registry.register(entry))
        registry.declare_env_keys("plat", ["PLAT_TOKEN"])

        from gateway.config import GatewayConfig

        config = GatewayConfig()
        config.platforms[Platform("plat")] = PlatformConfig(extra={"token": "yaml"})

        import gateway.platform_registry as pr

        saved = pr.platform_registry
        pr.platform_registry = registry
        monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
        try:
            from gateway.config_env import _enable_plugin_platforms_from_env

            _enable_plugin_platforms_from_env(config)
        finally:
            pr.platform_registry = saved

        assert loads, "a YAML-derived config row must still materialize the platform"


class TestAuthPoolMetadata:
    """``auth_pools`` manifest metadata: the auth.json credential-pool keys an enablement
    path may read. Presence of a pool record is a credential signal — skipping a platform
    that has one is the silent-drop bug PR review found (photon installs whose ``hermes
    photon setup`` wrote auth.json but whose .env write failed or was cleared)."""

    @pytest.fixture
    def registry(self):
        return PlatformRegistry()

    def test_declare_and_lookup_roundtrip(self):
        reg = PlatformRegistry()
        reg.register_deferred("plat", lambda: None)
        reg.declare_auth_pools("plat", ["some_pool"])
        assert reg.auth_pools("plat") == frozenset({"some_pool"})

    def test_undeclared_platform_yields_empty_pools(self):
        reg = PlatformRegistry()
        assert reg.auth_pools("nope") == frozenset()

    def test_manifest_auth_pools_parsed_from_plugin_yaml(self, tmp_path):
        from hermes_cli.plugins_manifest import parse_manifest_file

        plugin_dir = tmp_path / "z-platform"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: z-platform\nkind: platform\nrequires_env:\n  - name: Z_TOKEN\n"
            "auth_pools:\n  - z_project\n  - ' spaced-pool '\n"
        )
        manifest = parse_manifest_file(plugin_dir / "plugin.yaml", plugin_dir, "bundled", "")
        assert manifest is not None
        assert manifest.auth_pools == ["z_project", "spaced-pool"]

    def test_declared_pool_without_record_still_skips(self, registry, monkeypatch, tmp_path):
        """Counter-case: declared pools but no auth.json record (and no env, no row) must
        keep the skip — the perf win survives the auth-pool extension."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # empty home: no auth.json

        loads = []
        entry = _entry("plat", is_connected=lambda cfg: True, required_env=["PLAT_TOKEN"])
        registry.register_deferred("plat", lambda: loads.append(1) or registry.register(entry))
        registry.declare_env_keys("plat", ["PLAT_TOKEN"])
        registry.declare_auth_pools("plat", ["z_project"])

        from gateway.config import GatewayConfig

        config = GatewayConfig()

        import gateway.platform_registry as pr

        saved = pr.platform_registry
        pr.platform_registry = registry
        try:
            from gateway.config_env import _plugin_could_be_enabled

            assert _plugin_could_be_enabled("plat", registry, config) is False, \
                "declared pools with no record, no env, no row must stay skipped"
        finally:
            pr.platform_registry = saved
        assert loads == []

    def test_auth_pool_record_forces_import(self, registry, monkeypatch, tmp_path):
        """The regression: declared pools + a real auth.json record must materialize the
        platform even when every declared env var is unset."""
        import json as _json

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "auth.json").write_text(_json.dumps(
            {"credential_pool": {"z_project": [{"spectrum_project_id": "x", "project_secret": "y"}]}}
        ))

        loads = []
        entry = _entry("plat", is_connected=lambda cfg: True, required_env=["PLAT_TOKEN"])
        registry.register_deferred("plat", lambda: loads.append(1) or registry.register(entry))
        registry.declare_env_keys("plat", ["PLAT_TOKEN"])
        registry.declare_auth_pools("plat", ["z_project"])

        from gateway.config import GatewayConfig

        config = GatewayConfig()

        import gateway.platform_registry as pr

        saved = pr.platform_registry
        pr.platform_registry = registry
        monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
        try:
            from gateway.config_env import _plugin_could_be_enabled

            assert _plugin_could_be_enabled("plat", registry, config) is True, \
                "a declared auth-pool record must defeat the skip"
        finally:
            pr.platform_registry = saved
        assert loads == [], "pre-check must stay import-free"


class TestSectionedHookGate:
    def test_hook_fires_only_for_sectioned_platform(self):
        from gateway.config_loader import apply_plugin_yaml_hooks

        reg = PlatformRegistry()
        fired = []

        def hook(yaml_cfg, platform_cfg):
            fired.append(platform_cfg)
            return {"bridged": True}

        entry = _entry("plat", apply_yaml_config_fn=hook)
        reg.register_deferred("plat", lambda: reg.register(entry))

        yaml_cfg = {"plat": {"some_key": "v"}}
        platforms_data = {}
        apply_plugin_yaml_hooks(yaml_cfg, None, platforms_data, reg)
        assert fired and platforms_data["plat"]["extra"]["bridged"] is True

    def test_unsectioned_platform_never_loads(self):
        from gateway.config_loader import apply_plugin_yaml_hooks

        reg = PlatformRegistry()
        loads = []

        def hook(yaml_cfg, platform_cfg):
            return {"bridged": True}

        entry = _entry("plat", apply_yaml_config_fn=hook)
        reg.register_deferred("plat", lambda: loads.append(1) or reg.register(entry))

        apply_plugin_yaml_hooks({"telegram": {"x": 1}}, None, {}, reg)
        assert loads == [], "a platform without a YAML section must not materialize"
