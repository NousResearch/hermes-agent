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
        assert "hermes plugins enable pipful" in caplog.text
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


class TestRequiresHermes:
    def test_gate_reads_the_running_code_version_not_dist_metadata(self, monkeypatch):
        """An editable install's dist metadata is frozen at install time (0.21.0 here) while the checkout runs
        0.21.4; the gate must compare against the code that is running."""
        import importlib.metadata
        from hermes_cli import plugins_manifest
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.21.0")
        monkeypatch.setattr("hermes_cli.__version__", "0.21.4")
        assert plugins_manifest.running_hermes_version() == "0.21.4"
        assert plugins_manifest.version_satisfies(">=0.21.4", plugins_manifest.running_hermes_version())

    @pytest.mark.parametrize("spec, current, expected", [
        (">=99.0.0rc1", "0.21.4", False),   # rc target used to parse as None -> clause silently dropped
        (">=0.23.0", "0.22.0rc1", False),   # rc running version used to disable every gate
        (">=0.21.0", "0.22.0rc1", True),
        (">=1.2.3.post1", "1.2.3", True),
        ("banana", "0.21.4", True),         # documented: unparseable target stays permissive
    ])
    def test_prerelease_spellings_gate(self, spec, current, expected):
        from hermes_cli.plugins_manifest import version_satisfies
        assert version_satisfies(spec, current) is expected

    def test_unsatisfied_requires_hermes_skips_without_importing(self, hermes_home, monkeypatch):
        """A too-new ``requires_hermes`` records an error and never runs register(); a satisfied one loads."""
        import sys
        from hermes_cli import plugins_manifest
        monkeypatch.setattr(plugins_manifest, "running_hermes_version", lambda: "1.2.3")
        _write_plugin(hermes_home / "plugins", "future", manifest_extra={"requires_hermes": ">=99.0"},
                      register_body="import sys; sys._rh_future = True")
        _write_plugin(hermes_home / "plugins", "current", manifest_extra={"requires_hermes": ">=1.2,<2"},
                      register_body="import sys; sys._rh_current = True")
        _enable(hermes_home, ["future", "current"])
        try:
            mgr = PluginManager()
            mgr.discover_and_load()
            assert not hasattr(sys, "_rh_future")
            assert "requires hermes >=99.0" in (mgr._plugins["future"].error or "")
            assert getattr(sys, "_rh_current", False) is True
        finally:
            for attr in ("_rh_future", "_rh_current"):
                if hasattr(sys, attr):
                    delattr(sys, attr)


class TestLoadIsolation:
    def test_sys_exit_in_plugin_is_isolated_and_named(self, hermes_home, caplog):
        """A plugin calling ``sys.exit()`` at import used to propagate SystemExit out of discovery: the whole
        registry emptied, ``_discovered`` reset and ``hermes chat`` exited 3 with no output. It must be
        recorded as that plugin's error while later plugins still load."""
        _write_plugin(hermes_home / "plugins", "b_exit")
        (hermes_home / "plugins" / "b_exit" / "__init__.py").write_text("import sys\nsys.exit(0)\n")
        _write_plugin(hermes_home / "plugins", "c_after")
        _enable(hermes_home, ["b_exit", "c_after"])
        mgr = PluginManager()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr.discover_and_load()  # must not raise
        assert mgr._discovered is True
        assert mgr._plugins["c_after"].enabled
        assert not mgr._plugins["b_exit"].enabled
        assert "SystemExit(0)" in (mgr._plugins["b_exit"].error or "")

    def test_keyboard_interrupt_still_propagates(self, hermes_home):
        _write_plugin(hermes_home / "plugins", "ctrlc")
        (hermes_home / "plugins" / "ctrlc" / "__init__.py").write_text("raise KeyboardInterrupt\n")
        _enable(hermes_home, ["ctrlc"])
        with pytest.raises(KeyboardInterrupt):
            PluginManager().discover_and_load()

    def test_register_overrunning_load_timeout_skips_only_that_plugin(self, hermes_home, caplog):
        """A register() that never returns used to hang startup forever (#108139). Under
        ``plugins.load_timeout_seconds`` that plugin alone is recorded as failed with a named reason, its
        pre-hang registrations are disposed, later plugins still load, and anything the abandoned worker
        registers afterwards is ignored."""
        import sys
        import threading
        sys._deadline_gate, sys._deadline_done = threading.Event(), threading.Event()
        _write_plugin(hermes_home / "plugins", "b_slow", register_body=(
            "import sys; ctx.register_hook('pre_tool_call', lambda **kw: None); sys._deadline_gate.wait(5); "
            "ctx.register_hook('post_tool_call', lambda **kw: None); sys._deadline_done.set()"))
        _write_plugin(hermes_home / "plugins", "c_after")
        _enable(hermes_home, ["b_slow", "c_after"])
        (hermes_home / "config.yaml").write_text(yaml.safe_dump(
            {"plugins": {"enabled": ["b_slow", "c_after"], "load_timeout_seconds": 0.3}}))
        mgr = PluginManager()
        try:
            with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
                mgr.discover_and_load()
                assert mgr._plugins["c_after"].enabled
                assert not mgr._plugins["b_slow"].enabled
                assert "load timed out after 0.3s" in (mgr._plugins["b_slow"].error or "")
                assert mgr._hooks.get("pre_tool_call", []) == []  # registered before the hang → disposed
                sys._deadline_gate.set()  # release the abandoned worker; its late registration must bounce
                assert sys._deadline_done.wait(5)
            assert mgr._hooks.get("post_tool_call", []) == []
            assert "called register_hook() after its load timed out; ignored" in caplog.text
        finally:
            del sys._deadline_gate, sys._deadline_done

    def test_load_timeout_zero_runs_register_inline(self, hermes_home):
        """``plugins.load_timeout_seconds: 0`` disables the deadline: register() runs on the calling thread."""
        import sys
        import threading
        _write_plugin(hermes_home / "plugins", "inline",
                      register_body="import sys, threading; sys._load_thread = threading.current_thread()")
        (hermes_home / "config.yaml").write_text(yaml.safe_dump(
            {"plugins": {"enabled": ["inline"], "load_timeout_seconds": 0}}))
        try:
            mgr = PluginManager()
            mgr.discover_and_load()
            assert mgr._plugins["inline"].enabled
            assert sys._load_thread is threading.current_thread()
        finally:
            if hasattr(sys, "_load_thread"):
                del sys._load_thread


class TestBundledKeyShadowing:
    def test_impostor_dir_cannot_claim_a_bundled_key(self, tmp_path, monkeypatch, caplog):
        """``~/.hermes/plugins/impostor_dir/plugin.yaml`` with ``name: <bundled key>`` used to displace the
        bundled plugin silently, so ``hermes plugins enable <key>`` enabled unrelated code. The bundled
        manifest wins and the impostor is warned about; a same-named user copy still overrides (documented)."""
        home = tmp_path / "home"
        (home / "plugins").mkdir(parents=True)
        bundled = tmp_path / "bundled"
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
        _write_plugin(bundled, "genuine", register_body="import sys; sys._shadow_probe = 'bundled'")
        _write_plugin(bundled, "overridable", register_body="import sys; sys._override_probe = 'bundled'")
        _write_plugin(home / "plugins", "impostor_dir", manifest_extra={"name": "genuine"},
                      register_body="import sys; sys._shadow_probe = 'impostor'")
        (home / "plugins" / "impostor_dir" / "plugin.yaml").write_text(
            yaml.dump({"name": "genuine", "version": "0.1.0", "description": "impostor"}))
        _write_plugin(home / "plugins", "overridable", register_body="import sys; sys._override_probe = 'user'")
        _enable(home, ["genuine", "overridable"])
        import sys
        try:
            with caplog.at_level(logging.INFO, logger="hermes_cli.plugins"):
                mgr = PluginManager()
                mgr.discover_and_load()
            assert mgr._plugins["genuine"].manifest.source == "bundled"
            assert sys._shadow_probe == "bundled"
            assert "impostor_dir" in caplog.text and "rename the directory" in caplog.text
            assert mgr._plugins["overridable"].manifest.source == "user"
            assert sys._override_probe == "user"
            assert "shadows the bundled copy" in caplog.text
        finally:
            for attr in ("_shadow_probe", "_override_probe"):
                if hasattr(sys, attr):
                    delattr(sys, attr)


class TestManifestParsingRobustness:
    def test_list_manifest_is_rejected_with_a_clear_reason_and_hooks_alias(self, hermes_home, caplog):
        """A list-typed plugin.yaml (#14066) names the actual problem instead of an AttributeError; the
        long-standing ``hooks:`` spelling still populates ``provides_hooks`` (#108371)."""
        from hermes_cli.plugins_discovery import scan_directory
        bad = hermes_home / "plugins" / "listy"
        bad.mkdir()
        (bad / "plugin.yaml").write_text("- name: listy\n")
        good = _write_plugin(hermes_home / "plugins", "hooky", manifest_extra={"hooks": ["pre_tool_call"]})
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            manifests = {m.name: m for m in scan_directory(hermes_home / "plugins", "user")}
        assert "listy" not in manifests
        assert "top level must be a mapping" in caplog.text
        assert manifests["hooky"].provides_hooks == ["pre_tool_call"]
        assert manifests["hooky"].path == str(good)


class TestDirectoryPluginKeepsIdentityOverEntryPoint:
    """A pyproject-wrapper plugin depends on a pip package that ships a ``hermes_agent.plugins``
    entry point under the SAME name. The installed directory must stay the plugin's identity (it
    carries catalog provenance and is what update/remove act on); the entry point must not displace it."""

    def test_loader_and_listing_prefer_the_installed_directory(self, hermes_home, monkeypatch):
        from hermes_cli.plugins_manifest import PluginManifest
        _write_plugin(hermes_home / "plugins", "twin")
        _enable(hermes_home, ["twin"])
        twin_ep = PluginManifest(name="twin", version="9.9.9", description="pip twin",
                                 source="entrypoint", path="twin_pkg:register", key="twin")
        monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [twin_ep])
        monkeypatch.setattr("hermes_cli.plugins_cmd.discover_entrypoint_manifests", lambda: [twin_ep], raising=False)
        monkeypatch.setattr("hermes_cli.plugins.discover_entrypoint_manifests", lambda: [twin_ep])

        mgr = PluginManager()
        mgr.discover_and_load()
        assert mgr._plugins["twin"].manifest.source == "user"

        from hermes_cli.plugins_cmd import _discover_all_plugins
        rows = [r for r in _discover_all_plugins() if r[0] == "twin"]
        assert [r[3] for r in rows] == ["user"]
        assert str(rows[0][4]).endswith("plugins/twin")
