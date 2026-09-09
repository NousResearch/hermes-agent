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
