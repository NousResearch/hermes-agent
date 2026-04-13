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


from unittest.mock import MagicMock


class TestAutogenPluginShimHappyPath:
    def _make_fake_bundle(self, tmp_path, name="testbundle", with_skills=True):
        """Create a fake skill-only bundle directory (no plugin.yaml, no __init__.py)."""
        plugin_dir = tmp_path / name
        plugin_dir.mkdir()
        if with_skills:
            skills_dir = plugin_dir / "skills"
            skills_dir.mkdir()
            for skill_name in ("alpha", "beta", "gamma"):
                skill_dir = skills_dir / skill_name
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text(
                    f"---\nname: {skill_name}\n---\n\nBody.\n"
                )
        return plugin_dir

    def test_generates_init_py_from_static_template(self, tmp_path):
        from hermes_cli.plugins_cmd import _autogen_plugin_shim, _SHIM_INIT_PY_TEMPLATE_V1

        plugin_dir = self._make_fake_bundle(tmp_path)
        console = MagicMock()

        _autogen_plugin_shim(plugin_dir, console)

        init_py = plugin_dir / "__init__.py"
        assert init_py.exists()
        assert init_py.read_text(encoding="utf-8") == _SHIM_INIT_PY_TEMPLATE_V1

    def test_generates_plugin_yaml(self, tmp_path):
        from hermes_cli.plugins_cmd import _autogen_plugin_shim

        plugin_dir = self._make_fake_bundle(tmp_path)
        console = MagicMock()

        _autogen_plugin_shim(plugin_dir, console)

        plugin_yaml = plugin_dir / "plugin.yaml"
        assert plugin_yaml.exists()
        content = plugin_yaml.read_text(encoding="utf-8")
        assert "HERMES_AUTOGEN_MANIFEST_V1" in content
        assert f"name: {plugin_dir.name}" in content

    def test_creates_sidecar_lock_file(self, tmp_path):
        import json
        from hermes_cli.plugins_cmd import _autogen_plugin_shim

        plugin_dir = self._make_fake_bundle(tmp_path)
        console = MagicMock()

        _autogen_plugin_shim(plugin_dir, console)

        lock_file = plugin_dir / ".hermes-autogen" / "shim.lock"
        assert lock_file.exists()
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
        assert lock["schema_version"] == 1
        generated_paths = {entry["path"] for entry in lock["generated_files"]}
        assert "__init__.py" in generated_paths
        assert "plugin.yaml" in generated_paths

    def test_noop_if_no_skills_dir(self, tmp_path):
        from hermes_cli.plugins_cmd import _autogen_plugin_shim

        plugin_dir = self._make_fake_bundle(tmp_path, with_skills=False)
        console = MagicMock()

        _autogen_plugin_shim(plugin_dir, console)

        assert not (plugin_dir / "__init__.py").exists()
        assert not (plugin_dir / "plugin.yaml").exists()
        assert not (plugin_dir / ".hermes-autogen").exists()

    def test_noop_if_init_py_already_exists(self, tmp_path):
        from hermes_cli.plugins_cmd import _autogen_plugin_shim

        plugin_dir = self._make_fake_bundle(tmp_path)
        existing_init = plugin_dir / "__init__.py"
        existing_init.write_text("# upstream's own init\n")
        console = MagicMock()

        _autogen_plugin_shim(plugin_dir, console)

        assert existing_init.read_text(encoding="utf-8") == "# upstream's own init\n"
        assert not (plugin_dir / ".hermes-autogen").exists()

    def test_prints_skill_count(self, tmp_path):
        from hermes_cli.plugins_cmd import _autogen_plugin_shim

        plugin_dir = self._make_fake_bundle(tmp_path)
        console = MagicMock()

        _autogen_plugin_shim(plugin_dir, console)

        all_calls = " ".join(
            str(call.args[0]) if call.args else ""
            for call in console.print.call_args_list
        )
        assert "3 skills" in all_calls or "3 skill" in all_calls


class TestAutogenPluginShimRollback:
    def test_write_failure_rolls_back_all_files(self, tmp_path, monkeypatch):
        from hermes_cli.plugins_cmd import _autogen_plugin_shim

        plugin_dir = tmp_path / "testbundle"
        plugin_dir.mkdir()
        skills_dir = plugin_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "foo").mkdir()
        (skills_dir / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n")

        # Patch Path.write_text on plugin.yaml only, after init_py already writes
        original_write = Path.write_text

        def flaky_write(self, *args, **kwargs):
            if self.name == "plugin.yaml":
                raise PermissionError("disk full simulation")
            return original_write(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", flaky_write)

        with pytest.raises(RuntimeError, match="Failed to generate plugin shim"):
            _autogen_plugin_shim(plugin_dir, MagicMock())

        # Verify rollback: init_py was unlinked, no lock file left behind
        assert not (plugin_dir / "__init__.py").exists()
        assert not (plugin_dir / ".hermes-autogen").exists()


class TestAutogenPluginNameValidation:
    def test_rejects_name_with_colon(self, tmp_path):
        from hermes_cli.plugins_cmd import _autogen_plugin_shim

        plugin_dir = tmp_path / "bad:name"
        plugin_dir.mkdir()
        (plugin_dir / "skills").mkdir()
        (plugin_dir / "skills" / "x").mkdir()
        (plugin_dir / "skills" / "x" / "SKILL.md").write_text("---\nname: x\n---\n")

        with pytest.raises(ValueError, match="contains ':'"):
            _autogen_plugin_shim(plugin_dir, MagicMock())


class TestAutogenEmptySkillsDir:
    def test_empty_skills_dir_still_generates_with_warning(self, tmp_path):
        from hermes_cli.plugins_cmd import _autogen_plugin_shim

        plugin_dir = tmp_path / "emptybundle"
        plugin_dir.mkdir()
        (plugin_dir / "skills").mkdir()  # exists but empty

        console = MagicMock()
        _autogen_plugin_shim(plugin_dir, console)

        # Shim is generated (skills/ exists, so the condition is met)
        assert (plugin_dir / "__init__.py").exists()
        # Warning was printed
        all_calls = " ".join(
            str(call.args[0]) if call.args else ""
            for call in console.print.call_args_list
        )
        assert "empty" in all_calls.lower() or "Warning" in all_calls


class TestAutogenIntegratedWithInstall:
    """Integration-ish test: verify cmd_install calls _autogen_plugin_shim."""

    def test_cmd_install_triggers_autogen_for_skill_only_bundle(
        self, tmp_path, monkeypatch
    ):
        import subprocess
        import shutil

        from hermes_cli import plugins_cmd

        # Stage a fake "cloned" repo layout in a temp location
        fake_repo = tmp_path / "fake-clone"
        fake_repo.mkdir()
        skills_dir = fake_repo / "skills"
        skills_dir.mkdir()
        (skills_dir / "alpha").mkdir()
        (skills_dir / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\n---\n\nBody.\n"
        )

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: plugins_dir)

        def fake_git_clone(cmd, **kwargs):
            # Simulate git clone by copying fake_repo into cmd's destination
            dest = Path(cmd[-1])
            shutil.copytree(fake_repo, dest)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr(subprocess, "run", fake_git_clone)
        # Avoid interactive env-var prompt
        monkeypatch.setattr(
            plugins_cmd, "_prompt_plugin_env_vars", lambda *a, **k: None
        )

        plugins_cmd.cmd_install("myauthor/mybundle")

        target = plugins_dir / "mybundle"
        assert (target / "__init__.py").exists()
        assert (target / "plugin.yaml").exists()
        assert (target / ".hermes-autogen" / "shim.lock").exists()


class TestReadAutogenLock:
    def test_reads_valid_lock(self, tmp_path):
        import json
        from hermes_cli.plugins_cmd import _read_autogen_lock

        plugin_dir = tmp_path / "testplug"
        plugin_dir.mkdir()
        lock_dir = plugin_dir / ".hermes-autogen"
        lock_dir.mkdir()
        lock_data = {"schema_version": 1, "generated_files": []}
        (lock_dir / "shim.lock").write_text(json.dumps(lock_data))

        result = _read_autogen_lock(plugin_dir)
        assert result == lock_data

    def test_missing_lock_returns_none(self, tmp_path):
        from hermes_cli.plugins_cmd import _read_autogen_lock

        plugin_dir = tmp_path / "testplug"
        plugin_dir.mkdir()

        assert _read_autogen_lock(plugin_dir) is None

    def test_corrupt_json_returns_none_with_warning(self, tmp_path, caplog):
        import logging
        from hermes_cli.plugins_cmd import _read_autogen_lock

        plugin_dir = tmp_path / "testplug"
        plugin_dir.mkdir()
        lock_dir = plugin_dir / ".hermes-autogen"
        lock_dir.mkdir()
        (lock_dir / "shim.lock").write_text("{not valid json")

        with caplog.at_level(logging.WARNING):
            result = _read_autogen_lock(plugin_dir)

        assert result is None


class TestCleanupAutogenFiles:
    def test_removes_all_generated_files_and_sidecar(self, tmp_path):
        import json
        from hermes_cli.plugins_cmd import _cleanup_autogen_files

        plugin_dir = tmp_path / "testplug"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("shim code")
        (plugin_dir / "plugin.yaml").write_text("name: foo")
        lock_dir = plugin_dir / ".hermes-autogen"
        lock_dir.mkdir()
        (lock_dir / "shim.lock").write_text(json.dumps({
            "schema_version": 1,
            "generated_files": [
                {"path": "__init__.py", "shim_version": "v1", "generated_at": "x"},
                {"path": "plugin.yaml", "shim_version": "v1", "generated_at": "x"},
            ],
        }))

        lock = {
            "schema_version": 1,
            "generated_files": [
                {"path": "__init__.py"},
                {"path": "plugin.yaml"},
            ],
        }
        _cleanup_autogen_files(plugin_dir, lock)

        assert not (plugin_dir / "__init__.py").exists()
        assert not (plugin_dir / "plugin.yaml").exists()
        assert not (plugin_dir / ".hermes-autogen").exists()

    def test_missing_file_is_tolerated(self, tmp_path):
        from hermes_cli.plugins_cmd import _cleanup_autogen_files

        plugin_dir = tmp_path / "testplug"
        plugin_dir.mkdir()
        lock = {"generated_files": [{"path": "__init__.py"}]}

        # Should not raise
        _cleanup_autogen_files(plugin_dir, lock)
