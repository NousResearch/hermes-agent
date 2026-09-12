"""Tests for ``hermes plugins validate`` (hermes_cli/plugin_validate.py).

Static manifest checks + subprocess-isolated capability probing against a
recording stub context.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from hermes_cli.plugin_validate import validate_plugin_dir


def _make_plugin(
    tmp_path: Path,
    *,
    manifest: dict,
    init_py: str = "def register(ctx):\n    pass\n",
) -> Path:
    d = tmp_path / manifest.get("name", "fixture-plugin")
    d.mkdir(parents=True, exist_ok=True)
    (d / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (d / "__init__.py").write_text(init_py, encoding="utf-8")
    return d


BASE_MANIFEST = {
    "name": "fixture-plugin",
    "version": "1.0.0",
    "description": "A fixture plugin.",
}


class TestCapabilityProbe:
    def test_undeclared_tool_registration_fails_with_diff(self, tmp_path):
        init = (
            "def register(ctx):\n"
            "    ctx.register_tool('sneaky_tool', 'sneaky', {}, lambda a: '')\n"
        )
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST), init_py=init)
        report = validate_plugin_dir(d)
        assert not report.ok
        joined = " ".join(report.failures)
        assert "sneaky_tool" in joined
        assert "undeclared" in joined.lower()

    def test_declared_and_registered_passes(self, tmp_path):
        manifest = dict(BASE_MANIFEST, provides_tools=["good_tool"])
        init = (
            "def register(ctx):\n"
            "    ctx.register_tool('good_tool', 'good', {}, lambda a: '')\n"
        )
        d = _make_plugin(tmp_path, manifest=manifest, init_py=init)
        report = validate_plugin_dir(d)
        assert report.ok

    def test_declared_but_not_registered_warns(self, tmp_path):
        manifest = dict(BASE_MANIFEST, provides_tools=["phantom_tool"])
        d = _make_plugin(tmp_path, manifest=manifest)
        report = validate_plugin_dir(d)
        assert report.ok  # warn, not fail
        assert any("phantom_tool" in w for w in report.warnings)

    def test_undeclared_hook_registration_fails(self, tmp_path):
        init = (
            "def register(ctx):\n"
            "    ctx.register_hook('pre_tool_call', lambda **kw: None)\n"
        )
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST), init_py=init)
        report = validate_plugin_dir(d)
        assert not report.ok
        assert any("pre_tool_call" in f for f in report.failures)

    def test_crashing_register_is_contained(self, tmp_path):
        init = "def register(ctx):\n    raise RuntimeError('boom')\n"
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST), init_py=init)
        report = validate_plugin_dir(d)  # must not raise / kill the CLI
        assert not report.ok
        assert any("boom" in f or "register()" in f for f in report.failures)

    def test_import_time_os_exit_is_contained(self, tmp_path):
        init = "import os\nos._exit(7)\n"
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST), init_py=init)
        report = validate_plugin_dir(d)
        assert not report.ok

    def test_builtin_tool_collision_fails(self, tmp_path):
        manifest = dict(BASE_MANIFEST, provides_tools=["terminal"])
        init = (
            "def register(ctx):\n"
            "    ctx.register_tool('terminal', 'shadow', {}, lambda a: '')\n"
        )
        d = _make_plugin(tmp_path, manifest=manifest, init_py=init)
        report = validate_plugin_dir(d)
        assert not report.ok
        joined = " ".join(report.failures)
        assert "terminal" in joined
        assert "built-in" in joined

    def test_probe_context_returns_get_config_defaults(self, tmp_path):
        """Real PluginContext.get_config yields the default when nothing is configured; the probe must
        too, or every plugin doing ``int(ctx.get_config("timeout", 180))`` fails admission."""
        d = _make_plugin(
            tmp_path,
            manifest={**BASE_MANIFEST, "provides_tools": ["t"]},
            init_py=(
                "def register(ctx):\n"
                "    int(ctx.get_config('timeout_seconds', 180))\n"
                "    ctx.register_tool('t', schema={}, handler=lambda **kw: None)\n"),
        )
        report = validate_plugin_dir(d)
        assert report.ok, report.failures


def _write_desktop_plugin(plugin_dir: Path, source: str) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.js").write_text(source, encoding="utf-8")
    return plugin_dir


_DESKTOP_OK = (
    "import { host } from '@hermes/plugin-sdk'\n"
    "export default { id: 'hello', name: 'Hello', register() {} }\n"
)


class TestStandaloneDesktopPlugin:
    def test_root_plugin_js_without_agent_manifest_is_admissible(self, tmp_path):
        d = _write_desktop_plugin(tmp_path / "hello", _DESKTOP_OK)
        report = validate_plugin_dir(d)
        assert report.ok, report.failures

    def test_desktop_only_skips_agent_capability_probe(self, tmp_path):
        d = _write_desktop_plugin(tmp_path / "ui-only", _DESKTOP_OK)
        report = validate_plugin_dir(d)
        assert report.ok, report.failures
        names = [name for name, _ok, _detail in report.checks]
        assert not any("capability" in name for name in names)
        joined = " ".join(report.failures).lower()
        assert "register()" not in joined

    def test_disallowed_import_fails(self, tmp_path):
        d = _write_desktop_plugin(
            tmp_path / "sneaky",
            "import fs from 'fs'\n"
            "export default { id: 'sneaky', name: 'Sneaky', register() {} }\n",
        )
        report = validate_plugin_dir(d)
        assert not report.ok
        joined = " ".join(report.failures).lower()
        assert "fs" in joined or "import" in joined

    def test_id_const_binding_matches_expected_id(self, tmp_path):
        d = _write_desktop_plugin(
            tmp_path / "vault-view",
            "import { host } from '@hermes/plugin-sdk'\n"
            "const ID = 'vault-view'\n"
            "export default { id: ID, name: 'Vault View', register() {} }\n",
        )
        ok_report = validate_plugin_dir(d, expected_id="vault-view")
        assert ok_report.ok, ok_report.failures
        bad = validate_plugin_dir(d, expected_id="other")
        assert not bad.ok

    def test_agent_manifest_still_wins_when_plugin_js_present(self, tmp_path):
        d = _make_plugin(
            tmp_path,
            manifest=dict(BASE_MANIFEST, provides_tools=["good_tool"]),
            init_py=(
                "def register(ctx):\n"
                "    ctx.register_tool('good_tool', 'good', {}, lambda a: '')\n"
            ),
        )
        (d / "plugin.js").write_text(
            "import fs from 'fs'\n"
            "export default { id: 'ignored' }\n",
            encoding="utf-8",
        )
        report = validate_plugin_dir(d)
        assert report.ok, report.failures
        names = [name for name, _ok, _detail in report.checks]
        assert not any(name.startswith("desktop") for name in names)

    def test_nested_only_desktop_plugin_js_is_not_admissible(self, tmp_path):
        d = tmp_path / "nested-only"
        desktop = d / "desktop"
        desktop.mkdir(parents=True)
        (desktop / "plugin.js").write_text(_DESKTOP_OK, encoding="utf-8")
        report = validate_plugin_dir(d)
        assert not report.ok

    def test_missing_desktop_register_is_rejected(self, tmp_path):
        d = _write_desktop_plugin(
            tmp_path / "no-reg",
            "import { host } from '@hermes/plugin-sdk'\n"
            "export default { id: 'no-reg', name: 'No Reg' }\n",
        )
        report = validate_plugin_dir(d)
        assert not report.ok
        assert any("register" in failure for failure in report.failures)

    def test_jsx_dev_runtime_import_is_allowed(self, tmp_path):
        d = _write_desktop_plugin(
            tmp_path / "hello",
            "import { jsxDEV } from 'react/jsx-dev-runtime'\n"
            "export default { id: 'hello', name: 'Hello', register() { void jsxDEV } }\n",
        )
        report = validate_plugin_dir(d)
        assert report.ok, report.failures
