"""Tests for ``hermes plugins validate`` (hermes_cli/plugin_validate.py).

Static manifest checks + subprocess-isolated capability probing against a
recording stub context.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import hermes_cli.plugins_cmd as plugins_cmd
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


class TestStaticChecks:
    def test_valid_plugin_passes(self, tmp_path):
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST))
        report = validate_plugin_dir(d)
        assert report.ok
        assert report.exit_code == 0

    def test_missing_manifest_fails(self, tmp_path):
        d = tmp_path / "empty-plugin"
        d.mkdir()
        report = validate_plugin_dir(d)
        assert not report.ok
        assert report.exit_code == 1
        assert any("plugin.yaml" in f for f in report.failures)

    def test_missing_required_fields_fail(self, tmp_path):
        d = _make_plugin(tmp_path, manifest={"name": "fixture-plugin"})
        report = validate_plugin_dir(d)
        assert not report.ok
        joined = " ".join(report.failures)
        assert "version" in joined
        assert "description" in joined

    def test_bad_requires_hermes_spec_fails(self, tmp_path):
        manifest = dict(BASE_MANIFEST, requires_hermes=">=not.a.version")
        d = _make_plugin(tmp_path, manifest=manifest)
        report = validate_plugin_dir(d)
        assert not report.ok
        assert any("requires_hermes" in f for f in report.failures)

    def test_good_requires_hermes_spec_passes(self, tmp_path):
        manifest = dict(BASE_MANIFEST, requires_hermes=">=0.1, <99")
        d = _make_plugin(tmp_path, manifest=manifest)
        report = validate_plugin_dir(d)
        assert report.ok

    def test_invalid_config_section_fails(self, tmp_path):
        manifest = dict(BASE_MANIFEST, config=[{"prompt": "no key here"}])
        d = _make_plugin(tmp_path, manifest=manifest)
        report = validate_plugin_dir(d)
        assert not report.ok
        assert any("config" in f for f in report.failures)

    def test_valid_config_section_passes(self, tmp_path):
        manifest = dict(
            BASE_MANIFEST,
            config=[
                {"key": "endpoint", "prompt": "Endpoint?", "type": "str"},
                {"key": "token", "secret": True, "type": "str"},
            ],
        )
        d = _make_plugin(tmp_path, manifest=manifest)
        report = validate_plugin_dir(d)
        assert report.ok

    def test_lower_snake_requires_env_fails(self, tmp_path):
        manifest = dict(BASE_MANIFEST, requires_env=["lower_case_bad"])
        d = _make_plugin(tmp_path, manifest=manifest)
        report = validate_plugin_dir(d)
        assert not report.ok
        assert any("requires_env" in f for f in report.failures)

    def test_upper_snake_requires_env_passes(self, tmp_path):
        manifest = dict(BASE_MANIFEST, requires_env=["MY_API_KEY_2"])
        d = _make_plugin(tmp_path, manifest=manifest)
        report = validate_plugin_dir(d)
        assert report.ok

    def test_rich_requires_env_dict_entries_accepted(self, tmp_path):
        manifest = dict(
            BASE_MANIFEST,
            requires_env=[{"name": "MY_KEY", "description": "key"}],
        )
        d = _make_plugin(tmp_path, manifest=manifest)
        report = validate_plugin_dir(d)
        assert report.ok


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


class TestCmdValidate:
    def test_cmd_validate_exit_zero_on_pass(self, tmp_path, capsys):
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST))
        with pytest.raises(SystemExit) as e:
            plugins_cmd.cmd_validate(str(d))
        assert e.value.code == 0
        out = capsys.readouterr().out
        assert "✓" in out

    def test_cmd_validate_exit_one_on_fail(self, tmp_path, capsys):
        d = tmp_path / "not-a-plugin"
        d.mkdir()
        with pytest.raises(SystemExit) as e:
            plugins_cmd.cmd_validate(str(d))
        assert e.value.code == 1
        out = capsys.readouterr().out
        assert "✗" in out

    def test_cmd_validate_json_output(self, tmp_path, capsys):
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST))
        with pytest.raises(SystemExit) as e:
            plugins_cmd.cmd_validate(str(d), as_json=True)
        assert e.value.code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert "checks" in payload

    def test_cmd_validate_missing_dir_fails(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as e:
            plugins_cmd.cmd_validate(str(tmp_path / "ghost"))
        assert e.value.code == 1
