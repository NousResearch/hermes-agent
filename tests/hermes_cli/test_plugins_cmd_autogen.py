"""Tests for install-time autogen shim generation and update-time refresh."""

import json
from pathlib import Path

import pytest


class TestShimTemplates:
    def test_init_template_is_valid_python(self):
        """The shim __init__.py template must be parseable Python."""
        import ast
        from hermes_cli.plugins_cmd import _SHIM_INIT_PY_TEMPLATE_V1

        # Should parse without error
        ast.parse(_SHIM_INIT_PY_TEMPLATE_V1)

    def test_init_template_imports_internal_helper(self):
        from hermes_cli.plugins_cmd import _SHIM_INIT_PY_TEMPLATE_V1

        assert "_auto_register_skills_from_dir_v1" in _SHIM_INIT_PY_TEMPLATE_V1
        assert "from hermes_cli.plugins import" in _SHIM_INIT_PY_TEMPLATE_V1

    def test_init_template_defines_register(self):
        from hermes_cli.plugins_cmd import _SHIM_INIT_PY_TEMPLATE_V1

        assert "def register(ctx):" in _SHIM_INIT_PY_TEMPLATE_V1

    def test_init_template_mentions_ownership_path(self):
        from hermes_cli.plugins_cmd import _SHIM_INIT_PY_TEMPLATE_V1

        assert ".hermes-autogen" in _SHIM_INIT_PY_TEMPLATE_V1


class TestGenerateMinimalManifest:
    def test_uses_directory_name(self, tmp_path):
        from hermes_cli.plugins_cmd import _generate_minimal_manifest

        plugin_dir = tmp_path / "superpowers"
        plugin_dir.mkdir()

        yaml_text = _generate_minimal_manifest(plugin_dir)

        assert "name: superpowers" in yaml_text
        assert "HERMES_AUTOGEN_MANIFEST_V1" in yaml_text

    def test_extracts_description_from_readme(self, tmp_path):
        from hermes_cli.plugins_cmd import _generate_minimal_manifest

        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "README.md").write_text(
            "# My Plugin\n\nThis is a cool plugin for doing things.\n"
        )

        yaml_text = _generate_minimal_manifest(plugin_dir)

        assert "cool plugin for doing things" in yaml_text

    def test_fallback_description_when_no_readme(self, tmp_path):
        from hermes_cli.plugins_cmd import _generate_minimal_manifest

        plugin_dir = tmp_path / "plain-plugin"
        plugin_dir.mkdir()

        yaml_text = _generate_minimal_manifest(plugin_dir)

        assert "description:" in yaml_text

    def test_rejects_invalid_dir_name(self, tmp_path):
        from hermes_cli.plugins_cmd import _generate_minimal_manifest

        plugin_dir = tmp_path / "bad.name"
        plugin_dir.mkdir()

        with pytest.raises(ValueError, match="invalid characters"):
            _generate_minimal_manifest(plugin_dir)
