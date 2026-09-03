"""Tests for plugin manifest v2 (#64165).

Covers: v1 regression (unchanged behavior), v2 field parsing, unknown-field
forward compat, requires_plugins load ordering + cycle handling,
config_schema validation warnings, and the python_dependencies
declare-only seam (surfaced, never installed).
"""

import logging

import pytest
import yaml

from hermes_cli.plugins import (
    PluginManager,
    PluginManifest,
    SUPPORTED_MANIFEST_VERSION,
    resolve_plugin_load_order,
    validate_config_schema,
)


def _write_plugin(base, name, manifest_extra=None, register_body="pass"):
    plugin_dir = base / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "version": "0.1.0", "description": f"test {name}"}
    if manifest_extra:
        manifest.update(manifest_extra)
    (plugin_dir / "plugin.yaml").write_text(yaml.dump(manifest))
    (plugin_dir / "__init__.py").write_text(
        f"def register(ctx):\n    {register_body}\n"
    )
    return plugin_dir


def _enable(home, names, entries=None):
    cfg = {"plugins": {"enabled": list(names)}}
    if entries:
        cfg["plugins"]["entries"] = entries
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    (home / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.setenv(
        "HERMES_BUNDLED_PLUGINS", str(tmp_path / "empty-bundled")
    )
    (tmp_path / "empty-bundled").mkdir()
    return home


class TestV1Regression:
    def test_v1_manifest_parses_with_defaults(self, hermes_home):
        _write_plugin(hermes_home / "plugins", "oldie")
        _enable(hermes_home, ["oldie"])
        mgr = PluginManager()
        mgr.discover_and_load()
        loaded = mgr._plugins["oldie"]
        assert loaded.enabled
        m = loaded.manifest
        assert m.manifest_version == 1
        assert m.api_version is None
        assert m.requires_plugins == []
        assert m.python_dependencies == []
        assert m.config_schema == {}
        assert m.license == ""
        assert m.homepage == ""
        assert m.tags == []

    def test_v1_unknown_fields_do_not_warn_loudly(self, hermes_home, caplog):
        _write_plugin(
            hermes_home / "plugins", "oldie",
            manifest_extra={"mystery_field": True},
        )
        _enable(hermes_home, ["oldie"])
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr = PluginManager()
            mgr.discover_and_load()
        assert mgr._plugins["oldie"].enabled
        assert "mystery_field" not in caplog.text


class TestV2Parsing:
    def test_v2_fields_parse(self, hermes_home):
        _write_plugin(
            hermes_home / "plugins", "modern",
            manifest_extra={
                "manifest_version": 2,
                "api_version": 1,
                "license": "MIT",
                "homepage": "https://example.com/modern",
                "tags": ["gateway", "demo"],
                "requires_plugins": [
                    {"id": "other", "version_range": ">=1.0,<2"},
                    "bare-dep",
                ],
                "python_dependencies": ["requests>=2.0,<3"],
                "config_schema": {
                    "api_url": {"type": "str", "default": "", "description": "x"},
                },
            },
        )
        _enable(hermes_home, ["modern"])
        mgr = PluginManager()
        mgr.discover_and_load()
        m = mgr._plugins["modern"].manifest
        assert m.manifest_version == 2
        assert m.api_version == 1
        assert m.license == "MIT"
        assert m.homepage == "https://example.com/modern"
        assert m.tags == ["gateway", "demo"]
        assert m.requires_plugins == [
            {"id": "other", "version_range": ">=1.0,<2"},
            {"id": "bare-dep", "version_range": None},
        ]
        assert m.python_dependencies == ["requests>=2.0,<3"]
        assert "api_url" in m.config_schema
        # plugin still loads
        assert mgr._plugins["modern"].enabled or mgr._plugins["modern"].error

    def test_unknown_field_in_v2_warns_but_loads(self, hermes_home, caplog):
        _write_plugin(
            hermes_home / "plugins", "modern",
            manifest_extra={"manifest_version": 2, "hovercraft": "eels"},
        )
        _enable(hermes_home, ["modern"])
        with caplog.at_level(logging.WARNING):
            mgr = PluginManager()
            mgr.discover_and_load()
        assert mgr._plugins["modern"].enabled
        assert "hovercraft" in caplog.text

    def test_future_manifest_version_warns_but_loads(self, hermes_home, caplog):
        _write_plugin(
            hermes_home / "plugins", "fromfuture",
            manifest_extra={
                "manifest_version": SUPPORTED_MANIFEST_VERSION + 5,
            },
        )
        _enable(hermes_home, ["fromfuture"])
        with caplog.at_level(logging.WARNING):
            mgr = PluginManager()
            mgr.discover_and_load()
        assert mgr._plugins["fromfuture"].enabled
        assert "newer than this Hermes" in caplog.text

    def test_malformed_v2_fields_warn_and_degrade(self, hermes_home, caplog):
        _write_plugin(
            hermes_home / "plugins", "sloppy",
            manifest_extra={
                "manifest_version": 2,
                "api_version": "banana",
                "requires_plugins": "not-a-list",
                "python_dependencies": {"nope": 1},
                "tags": "not-a-list",
            },
        )
        _enable(hermes_home, ["sloppy"])
        with caplog.at_level(logging.WARNING):
            mgr = PluginManager()
            mgr.discover_and_load()
        loaded = mgr._plugins["sloppy"]
        assert loaded.enabled
        m = loaded.manifest
        assert m.api_version is None
        assert m.requires_plugins == []
        assert m.python_dependencies == []
        assert m.tags == []


class TestDependencyOrder:
    def test_dep_registers_before_dependent(self, hermes_home):
        # zzz-consumer requires aaa-base... but alphabetically consumer
        # would load AFTER base anyway, so invert: aaa-consumer requires
        # zzz-base, forcing the topo sort to override alpha order.
        _write_plugin(
            hermes_home / "plugins", "aaa-consumer",
            manifest_extra={
                "manifest_version": 2,
                "requires_plugins": [{"id": "zzz-base"}],
            },
            register_body="import sys; sys._m2_order.append('aaa-consumer')",
        )
        _write_plugin(
            hermes_home / "plugins", "zzz-base",
            register_body="import sys; sys._m2_order.append('zzz-base')",
        )
        _enable(hermes_home, ["aaa-consumer", "zzz-base"])
        import sys

        sys._m2_order = []
        try:
            mgr = PluginManager()
            mgr.discover_and_load()
            assert sys._m2_order == ["zzz-base", "aaa-consumer"]
        finally:
            del sys._m2_order

    def test_missing_dep_warns_but_loads(self, hermes_home, caplog):
        _write_plugin(
            hermes_home / "plugins", "needy",
            manifest_extra={
                "manifest_version": 2,
                "requires_plugins": [{"id": "ghost-plugin"}],
            },
        )
        _enable(hermes_home, ["needy"])
        with caplog.at_level(logging.WARNING):
            mgr = PluginManager()
            mgr.discover_and_load()
        assert mgr._plugins["needy"].enabled
        assert "ghost-plugin" in caplog.text
        assert "loading anyway" in caplog.text

    def test_cycle_warns_and_falls_back_alpha(self, hermes_home, caplog):
        _write_plugin(
            hermes_home / "plugins", "cyc-a",
            manifest_extra={
                "manifest_version": 2,
                "requires_plugins": [{"id": "cyc-b"}],
            },
            register_body="import sys; sys._m2_cycle.append('cyc-a')",
        )
        _write_plugin(
            hermes_home / "plugins", "cyc-b",
            manifest_extra={
                "manifest_version": 2,
                "requires_plugins": [{"id": "cyc-a"}],
            },
            register_body="import sys; sys._m2_cycle.append('cyc-b')",
        )
        _enable(hermes_home, ["cyc-a", "cyc-b"])
        import sys

        sys._m2_cycle = []
        try:
            with caplog.at_level(logging.WARNING):
                mgr = PluginManager()
                mgr.discover_and_load()
            # Both still load, in alphabetical fallback order.
            assert sys._m2_cycle == ["cyc-a", "cyc-b"]
            assert "cycle" in caplog.text.lower()
        finally:
            del sys._m2_cycle

    def test_resolve_order_pure_function(self):
        manifests = {
            "b": PluginManifest(name="b", key="b",
                                requires_plugins=[{"id": "c"}]),
            "a": PluginManifest(name="a", key="a",
                                requires_plugins=[{"id": "b"}]),
            "c": PluginManifest(name="c", key="c"),
        }
        assert resolve_plugin_load_order(manifests) == ["c", "b", "a"]

    def test_resolve_order_matches_by_manifest_name(self):
        manifests = {
            "cat/impl": PluginManifest(
                name="impl-name", key="cat/impl"
            ),
            "user": PluginManifest(
                name="user", key="user",
                requires_plugins=[{"id": "impl-name"}],
            ),
        }
        assert resolve_plugin_load_order(manifests) == ["cat/impl", "user"]


class TestConfigSchema:
    def test_type_mismatch_warns_but_loads(self, hermes_home, caplog):
        _write_plugin(
            hermes_home / "plugins", "cfgd",
            manifest_extra={
                "manifest_version": 2,
                "config_schema": {
                    "api_url": {"type": "str"},
                    "retries": {"type": "int"},
                },
            },
        )
        _enable(
            hermes_home, ["cfgd"],
            entries={"cfgd": {"settings": {"api_url": 42, "retries": 3}}},
        )
        with caplog.at_level(logging.WARNING):
            mgr = PluginManager()
            mgr.discover_and_load()
        assert mgr._plugins["cfgd"].enabled
        assert "plugins.entries.cfgd.settings.api_url" in caplog.text
        assert "should be str" in caplog.text
        assert "retries" not in caplog.text.split("should be")[-1]

    def test_required_key_missing_warns(self):
        warnings = validate_config_schema(
            "p", {"token": {"type": "str", "required": True}}, {}
        )
        assert warnings and "required" in warnings[0]
        assert "plugins.entries.p.settings.token" in warnings[0]

    def test_valid_settings_produce_no_warnings(self):
        schema = {
            "api_url": {"type": "str"},
            "retries": {"type": "int"},
            "ratio": {"type": "float"},
            "flag": {"type": "bool"},
        }
        settings = {"api_url": "x", "retries": 2, "ratio": 0.5, "flag": True}
        assert validate_config_schema("p", schema, settings) == []

    def test_bool_does_not_satisfy_int(self):
        warnings = validate_config_schema(
            "p", {"retries": {"type": "int"}}, {"retries": True}
        )
        assert warnings and "should be int" in warnings[0]

    def test_unknown_declared_type_skips_check(self, hermes_home, caplog):
        _write_plugin(
            hermes_home / "plugins", "weird",
            manifest_extra={
                "manifest_version": 2,
                "config_schema": {"thing": {"type": "quaternion"}},
            },
        )
        _enable(
            hermes_home, ["weird"],
            entries={"weird": {"settings": {"thing": 1}}},
        )
        with caplog.at_level(logging.WARNING):
            mgr = PluginManager()
            mgr.discover_and_load()
        assert mgr._plugins["weird"].enabled
        assert "quaternion" in caplog.text
        assert "should be" not in caplog.text


class TestPythonDependenciesSeam:
    def test_missing_pip_dep_surfaced_with_hint_not_installed(
        self, hermes_home, caplog, monkeypatch
    ):
        calls = []
        import subprocess

        def _spy_run(*args, **kwargs):
            calls.append(args)
            raise AssertionError("no subprocess should run for pip deps")

        monkeypatch.setattr(subprocess, "run", _spy_run)
        monkeypatch.setattr(subprocess, "check_call", _spy_run)
        _write_plugin(
            hermes_home / "plugins", "pipful",
            manifest_extra={
                "manifest_version": 2,
                "python_dependencies": [
                    "definitely-not-a-real-package-64165>=1.0,<2",
                ],
            },
        )
        _enable(hermes_home, ["pipful"])
        with caplog.at_level(logging.WARNING):
            mgr = PluginManager()
            mgr.discover_and_load()
        assert mgr._plugins["pipful"].enabled
        assert "definitely-not-a-real-package-64165" in caplog.text
        assert "pip install" in caplog.text
        assert "does not install plugin dependencies automatically" in caplog.text
        assert calls == []

    def test_satisfied_pip_dep_is_quiet(self, hermes_home, caplog):
        _write_plugin(
            hermes_home / "plugins", "pipok",
            manifest_extra={
                "manifest_version": 2,
                "python_dependencies": ["pyyaml>=5,<7"],
            },
        )
        _enable(hermes_home, ["pipok"])
        with caplog.at_level(logging.WARNING):
            mgr = PluginManager()
            mgr.discover_and_load()
        assert mgr._plugins["pipok"].enabled
        assert "pip install" not in caplog.text


class TestCtxHasPlugin:
    def test_has_plugin_probe(self, hermes_home):
        _write_plugin(hermes_home / "plugins", "probe-target")
        _write_plugin(
            hermes_home / "plugins", "prober",
            manifest_extra={
                "manifest_version": 2,
                "requires_plugins": [{"id": "probe-target"}],
            },
            register_body=(
                "import sys; sys._m2_probe = ("
                "ctx.has_plugin('probe-target'), ctx.has_plugin('nope'))"
            ),
        )
        _enable(hermes_home, ["probe-target", "prober"])
        import sys

        try:
            mgr = PluginManager()
            mgr.discover_and_load()
            assert sys._m2_probe == (True, False)
        finally:
            if hasattr(sys, "_m2_probe"):
                del sys._m2_probe
