"""Tests for plugin manifest v2 (#64165).

Covers: v1 regression (unchanged behavior), v2 field parsing, unknown-field
forward compat, requires_plugins load ordering + cycle handling,
config_schema validation warnings, and the python_dependencies
declare-only seam (surfaced, never installed).
"""

import logging
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml

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
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(manifest))
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
        assert str(SUPPORTED_MANIFEST_VERSION + 5) in caplog.text

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
        assert "hermes pm repair" in caplog.text
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
        """Compatibility gates use the running code's base release version."""
        from hermes_cli import plugins_manifest
        monkeypatch.setattr(
            "hermes_cli.version_info.get_version_info",
            lambda: SimpleNamespace(base_version="0.21.4"),
        )
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
                assert mgr._plugins["b_slow"].error
                assert mgr._hooks.get("pre_tool_call", []) == []  # registered before the hang → disposed
                sys._deadline_gate.set()  # release the abandoned worker; its late registration must bounce
                assert sys._deadline_done.wait(5)
            assert mgr._hooks.get("post_tool_call", []) == []
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
            yaml.safe_dump({"name": "genuine", "version": "0.1.0", "description": "impostor"}))
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


class TestUserKeyCollisionFailLoud:
    """Flat user manifests take their registry key from the manifest ``name:`` field, so a backup copy of a
    plugin directory (``statusboard.bak-…``) races the live directory for the SAME key and
    last-in-discovery-order (``sorted(path.iterdir())``) wins silently. Each such race must be announced as
    a WARNING on the ``hermes_cli.plugins.collisions`` logger — one line per colliding key, every competing
    path in discovery order, the winner named — WITHOUT changing which manifest loads."""

    COLLISION_LOGGER = "hermes_cli.plugins.collisions"

    @staticmethod
    def _collision_records(caplog) -> list:
        return [r for r in caplog.records
                if r.name == TestUserKeyCollisionFailLoud.COLLISION_LOGGER
                and r.levelno == logging.WARNING]

    def test_stale_backup_copy_sorting_last_warns_and_still_wins(self, hermes_home, caplog):
        """Historical name shape (dot in the MIDDLE of the dir name): the stale copy sorts AFTER the live
        directory and used to re-activate old code with no trace in the log. The winner is unchanged; the
        race is now loudly named with both paths in discovery order."""
        live = _write_plugin(hermes_home / "plugins", "statusboard",
                             register_body="import sys; sys._collision_probe = 'live'")
        stale = _write_plugin(hermes_home / "plugins", "statusboard.bak-copy-20260930",
                              manifest_extra={"name": "statusboard"},
                              register_body="import sys; sys._collision_probe = 'stale'")
        _enable(hermes_home, ["statusboard"])
        import sys
        try:
            with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
                mgr = PluginManager()
                mgr.discover_and_load()
            records = self._collision_records(caplog)
            assert len(records) == 1  # the counter: exactly one WARNING per colliding key
            message = records[0].getMessage()
            assert "statusboard" in message
            assert str(live) in message and str(stale) in message
            assert message.index(str(live)) < message.index(str(stale))  # discovery order
            # Behavior unchanged: last-in-order (the stale copy) still wins and loads.
            assert mgr._plugins["statusboard"].manifest.path == str(stale)
            assert sys._collision_probe == "stale"
            assert message.split("loading", 1)[1].lstrip().startswith(str(stale))
        finally:
            if hasattr(sys, "_collision_probe"):
                delattr(sys, "_collision_probe")

    def test_dot_prefixed_copy_sorting_first_warns_and_live_dir_wins(self, hermes_home, caplog):
        """The dot-PREFIX name shape (sorts BEFORE the live dir): the live
        directory wins — and the race is still announced, not only the dangerous direction."""
        stale = _write_plugin(hermes_home / "plugins", ".bak-copy-20260930",
                              manifest_extra={"name": "statusboard"},
                              register_body="import sys; sys._collision_probe = 'stale'")
        live = _write_plugin(hermes_home / "plugins", "statusboard",
                             register_body="import sys; sys._collision_probe = 'live'")
        _enable(hermes_home, ["statusboard"])
        import sys
        try:
            with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
                mgr = PluginManager()
                mgr.discover_and_load()
            records = self._collision_records(caplog)
            assert len(records) == 1
            message = records[0].getMessage()
            assert message.index(str(stale)) < message.index(str(live))
            assert mgr._plugins["statusboard"].manifest.path == str(live)
            assert sys._collision_probe == "live"
        finally:
            if hasattr(sys, "_collision_probe"):
                delattr(sys, "_collision_probe")

    @pytest.mark.platforms("posix")
    def test_symlinked_live_dir_and_stale_copy_warn_with_symlink_winning(self, hermes_home, caplog):
        """Live dir as a symlink to a checked-out copy + a dot-prefixed stale
        copy: the symlink sorts last, wins, and both paths — symlink and stale copy — appear in the warning
        with the winner named."""
        target = hermes_home / "plugin-source" / "statusboard"
        target.mkdir(parents=True)
        (target / "plugin.yaml").write_text(yaml.safe_dump(
            {"name": "statusboard", "version": "2.0.0", "description": "live target"}))
        (target / "__init__.py").write_text("def register(ctx):\n    import sys; sys._collision_probe = 'current'\n")
        (hermes_home / "plugins" / "statusboard").symlink_to(target, target_is_directory=True)
        stale = _write_plugin(hermes_home / "plugins", ".bak-copy-20260930",
                              manifest_extra={"name": "statusboard"},
                              register_body="import sys; sys._collision_probe = 'stale'")
        _enable(hermes_home, ["statusboard"])
        import sys
        try:
            with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
                mgr = PluginManager()
                mgr.discover_and_load()
            records = self._collision_records(caplog)
            assert len(records) == 1
            message = records[0].getMessage()
            assert message.index(str(stale)) < message.index(str(hermes_home / "plugins" / "statusboard"))
            assert mgr._plugins["statusboard"].manifest.path == str(hermes_home / "plugins" / "statusboard")
            assert sys._collision_probe == "current"
        finally:
            if hasattr(sys, "_collision_probe"):
                delattr(sys, "_collision_probe")

    def test_three_copies_still_produce_exactly_one_counter_warning(self, hermes_home, caplog):
        """N copies of one plugin = ONE warning line naming every path, not N fragments: silence means no
        race, one warning per raced key is the effectiveness counter."""
        first = _write_plugin(hermes_home / "plugins", "alpha")
        second = _write_plugin(hermes_home / "plugins", "alpha.bak-1", manifest_extra={"name": "alpha"})
        third = _write_plugin(hermes_home / "plugins", "alpha.bak-2", manifest_extra={"name": "alpha"})
        _enable(hermes_home, ["alpha"])
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr = PluginManager()
            mgr.discover_and_load()
        records = self._collision_records(caplog)
        assert len(records) == 1
        message = records[0].getMessage()
        for path in (first, second, third):
            assert str(path) in message
        assert mgr._plugins["alpha"].manifest.path == str(third)  # last-in-order, unchanged

    def test_category_layout_same_manifest_name_is_no_collision(self, hermes_home, caplog):
        """Category manifests take path-derived keys (``<cat>/<dir>``), so the same ``name:`` field in two
        categories is two different registry keys — no collision warning, no false alarm."""
        foo = _write_plugin(hermes_home / "plugins" / "web", "alpha", manifest_extra={"name": "shared-name"})
        bar = _write_plugin(hermes_home / "plugins" / "tools", "beta", manifest_extra={"name": "shared-name"})
        _enable(hermes_home, ["web/alpha", "tools/beta"])
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr = PluginManager()
            mgr.discover_and_load()
        assert self._collision_records(caplog) == []
        assert mgr._plugins["web/alpha"].manifest.path == str(foo)
        assert mgr._plugins["tools/beta"].manifest.path == str(bar)

    def test_documented_bundled_override_does_not_trip_the_collision_warning(
            self, tmp_path, monkeypatch, caplog):
        """A user dir shadowing a bundled plugin is the documented override mechanism (INFO, existing
        behavior) — exactly one user manifest races, so no collision warning may fire."""
        home = tmp_path / "home"
        (home / "plugins").mkdir(parents=True)
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
        _write_plugin(bundled, "overridable")
        _write_plugin(home / "plugins", "overridable")
        _enable(home, ["overridable"])
        with caplog.at_level(logging.INFO, logger="hermes_cli.plugins"):
            mgr = PluginManager()
            mgr.discover_and_load()
        assert self._collision_records(caplog) == []
        assert mgr._plugins["overridable"].manifest.source == "user"
        assert "shadows the bundled copy" in caplog.text


class TestCrossSourceOverrideIsNotACollision:
    """A ``user`` and a ``project`` manifest claiming the same key is the documented precedence order
    (project > user > bundled): a deliberate project-local override of a personal plugin loads without
    the stale-copy collision WARNING. Only a same-SOURCE duplicate (user-vs-user, project-vs-project)
    at a different path is a race worth announcing."""

    COLLISION_LOGGER = "hermes_cli.plugins.collisions"

    @staticmethod
    def _collision_records(caplog) -> list:
        return [r for r in caplog.records
                if r.name == TestCrossSourceOverrideIsNotACollision.COLLISION_LOGGER
                and r.levelno == logging.WARNING]

    @staticmethod
    def _enable_project_plugins(tmp_path, monkeypatch):
        project = tmp_path / "project-root"
        (project / ".hermes" / "plugins").mkdir(parents=True)
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
        monkeypatch.chdir(project)
        return project

    def test_project_override_of_user_plugin_loads_without_collision_warning(
            self, hermes_home, monkeypatch, caplog):
        """The project copy overrides the personal plugin on purpose: the documented precedence decides
        silently — the stale-copy advice would be a false alarm here."""
        _write_plugin(hermes_home / "plugins", "statusboard",
                      register_body="import sys; sys._override_probe = 'user'")
        project = self._enable_project_plugins(hermes_home.parent, monkeypatch)
        project_copy = _write_plugin(project / ".hermes" / "plugins", "statusboard",
                                     manifest_extra={"version": "1.1.0"},
                                     register_body="import sys; sys._override_probe = 'project'")
        _enable(hermes_home, ["statusboard"])
        import sys
        try:
            with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
                mgr = PluginManager()
                mgr.discover_and_load()
            assert self._collision_records(caplog) == []
            assert mgr._plugins["statusboard"].manifest.source == "project"
            assert mgr._plugins["statusboard"].manifest.path == str(project_copy)
            assert sys._override_probe == "project"
        finally:
            if hasattr(sys, "_override_probe"):
                delattr(sys, "_override_probe")

    def test_project_local_duplicate_paths_still_warn(self, hermes_home, monkeypatch, caplog):
        """The project half of the same-source rule: two project manifests racing for one key are
        announced exactly like the user-vs-user race (one WARNING, both paths, winner named)."""
        project = self._enable_project_plugins(hermes_home.parent, monkeypatch)
        live = _write_plugin(project / ".hermes" / "plugins", "statusboard",
                             register_body="import sys; sys._override_probe = 'live'")
        stale = _write_plugin(project / ".hermes" / "plugins", "statusboard.bak-alt",
                              manifest_extra={"name": "statusboard"},
                              register_body="import sys; sys._override_probe = 'stale'")
        _enable(hermes_home, ["statusboard"])
        import sys
        try:
            with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
                mgr = PluginManager()
                mgr.discover_and_load()
            records = self._collision_records(caplog)
            assert len(records) == 1
            message = records[0].getMessage()
            assert message.index(str(live)) < message.index(str(stale))
            assert mgr._plugins["statusboard"].manifest.path == str(stale)
            assert sys._override_probe == "stale"
        finally:
            if hasattr(sys, "_override_probe"):
                delattr(sys, "_override_probe")

    def test_user_race_with_project_override_warns_about_the_user_race(
            self, hermes_home, monkeypatch, caplog):
        """A user-vs-user duplicate racing alongside a deliberate project override: exactly one WARNING
        for the user race — the override shows up in the ladder and as the winner, tagged [project]."""
        live = _write_plugin(hermes_home / "plugins", "statusboard",
                             register_body="import sys; sys._override_probe = 'user-live'")
        stale = _write_plugin(hermes_home / "plugins", "statusboard.bak-alt",
                              manifest_extra={"name": "statusboard"},
                              register_body="import sys; sys._override_probe = 'user-stale'")
        project = self._enable_project_plugins(hermes_home.parent, monkeypatch)
        project_copy = _write_plugin(project / ".hermes" / "plugins", "statusboard",
                                     manifest_extra={"version": "2.0.0"},
                                     register_body="import sys; sys._override_probe = 'project'")
        _enable(hermes_home, ["statusboard"])
        import sys
        try:
            with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
                mgr = PluginManager()
                mgr.discover_and_load()
            records = self._collision_records(caplog)
            assert len(records) == 1
            message = records[0].getMessage()
            assert message.index(str(live)) < message.index(str(stale)) < message.index(str(project_copy))
            assert "[user] -> " in message and "[project]" in message
            assert message.split("loading", 1)[1].lstrip().startswith(str(project_copy))
            assert mgr._plugins["statusboard"].manifest.path == str(project_copy)
            assert sys._override_probe == "project"
        finally:
            if hasattr(sys, "_override_probe"):
                delattr(sys, "_override_probe")
