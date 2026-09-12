"""Tests for ``hermes plugins validate`` (hermes_cli/plugin_validate.py).

Static manifest checks + subprocess-isolated capability probing against a
recording stub context.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_cli.plugin_validate import validate_plugin_dir


def _make_plugin(
    tmp_path: Path,
    *,
    manifest: dict,
    init_py: str | None = "def register(ctx):\n    pass\n",
) -> Path:
    d = tmp_path / manifest.get("name", "fixture-plugin")
    d.mkdir(parents=True, exist_ok=True)
    (d / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    if init_py is not None:
        (d / "__init__.py").write_text(init_py, encoding="utf-8")
    return d


BASE_MANIFEST = {
    "name": "fixture-plugin",
    "version": "1.0.0",
    "description": "A fixture plugin.",
}


def test_requires_hermes_spec_is_validated(tmp_path):
    manifest = dict(BASE_MANIFEST, requires_hermes=">=0.21")
    d = _make_plugin(tmp_path, manifest=manifest)

    report = validate_plugin_dir(d)

    assert report.ok, report.failures
    assert ("requires_hermes", True, "spec '>=0.21' parses") in report.checks


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


_MP_MANIFEST = {
    "name": "fixture-model-provider",
    "kind": "model-provider",
    "version": "1.0.0",
    "description": "A fixture model provider.",
}

_MP_REGISTERING_INIT = """\
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="fixture-mp",
    env_vars=("FIXTURE_MP_KEY",),
    base_url="https://example.invalid/v1",
    auth_type="api_key",
    api_mode="chat_completions",
))
"""


class TestModelProviderProbe:
    """kind: model-provider must prove import-time register_provider(), not register(ctx)."""

    @pytest.mark.parametrize(
        "manifest, init_py, expect_ok, fail_contains, fail_excludes, probe_ok_contains",
        [
            pytest.param(
                _MP_MANIFEST, _MP_REGISTERING_INIT, True, None, None, "register_provider",
                id="import_register_provider",
            ),
            pytest.param(
                _MP_MANIFEST, "def register(ctx):\n    pass\n", False,
                "register_provider", "no register() function", None,
                id="shell_register_ctx",
            ),
            pytest.param(
                _MP_MANIFEST, None, False, "__init__.py", None, None,
                id="missing_init",
            ),
            pytest.param(
                {**_MP_MANIFEST, "kind": "Model-Provider"}, _MP_REGISTERING_INIT, False,
                "no register() function", None, None,
                id="wrong_case_kind",
            ),
        ],
    )
    def test_model_provider_admission(
        self, tmp_path, manifest, init_py, expect_ok, fail_contains, fail_excludes,
        probe_ok_contains,
    ):
        report = validate_plugin_dir(
            _make_plugin(tmp_path, manifest=dict(manifest), init_py=init_py)
        )
        if expect_ok:
            assert report.ok, report.failures
        else:
            assert not report.ok
        joined = " ".join(report.failures)
        if fail_contains:
            assert fail_contains in joined, report.failures
        if fail_excludes:
            assert fail_excludes not in joined, report.failures
        if probe_ok_contains:
            assert any(
                name == "capability probe" and ok and probe_ok_contains in detail
                for name, ok, detail in report.checks
            ), report.checks
        else:
            assert not any(
                name == "capability probe" and ok and "register_provider" in detail
                for name, ok, detail in report.checks
            ), report.checks

    def test_generic_plugin_still_requires_register_ctx(self, tmp_path):
        report = validate_plugin_dir(
            _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST), init_py="# no register()\n")
        )
        assert not report.ok
        assert any("no register() function" in f for f in report.failures), report.failures

