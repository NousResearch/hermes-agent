"""Tool configuration is not a plugin packaging declaration (#122148)."""
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

from pm.plugin_declarations import read_python_declaration
from pm.workspace import _generate_pyproject, _workspace_member


@pytest.mark.parametrize("tooling", ["", "[tool.ruff]\nline-length = 100\n"])
def test_tooling_only_plugin_leaves_core_workspace_unchanged(tmp_path, tooling):
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    project = plugin / "pyproject.toml"
    project.write_text(tooling, encoding="utf-8")
    declaration = read_python_declaration(plugin)
    assert not declaration.is_member
    assert declaration.pyproject is None
    assert project in declaration.files
    core = tmp_path / "core"
    core.mkdir()
    core_text = ('[project]\nname="fixture-core"\nversion="1"\n'
                 'requires-python=">=3.11"\n[tool.uv]\npackage=false\n')
    (core / "pyproject.toml").write_text(core_text, encoding="utf-8")
    root = tmp_path / "workspace"
    _generate_pyproject([plugin], root, source=core)
    assert (root / "pyproject.toml").read_text(encoding="utf-8") == core_text
    _lock_offline(root)
    assert project.read_text(encoding="utf-8") == tooling


@pytest.mark.parametrize("alias", ["python_dependencies", "pip_dependencies"])
def test_tooling_only_plugin_preserves_manifest_dependencies(tmp_path, alias):
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    project = plugin / "pyproject.toml"
    text = "[tool.ruff]\nline-length = 100\n"
    project.write_text(text, encoding="utf-8")
    # The false target marker permits a real offline resolver, without downloads/install.
    spec = 'fixturedep>=1,<2; python_version < "0"'
    (plugin / "plugin.yaml").write_text(f"{alias}:\n  - '{spec}'\n", encoding="utf-8")
    declaration = read_python_declaration(plugin)
    assert declaration.pyproject is None
    assert declaration.requirements == (spec,)
    assert declaration.is_member
    member = _workspace_member(plugin, tmp_path / "workspace", identity=plugin)
    metadata = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["dependencies"] == [spec]
    assert metadata["tool"]["uv"]["package"] is False
    _lock_offline(member)
    assert project.read_text(encoding="utf-8") == text


def _lock_offline(root: Path):
    uv = shutil.which("uv")
    assert uv is not None, "real offline resolver required"
    result = subprocess.run(
        [uv, "lock", "--offline", "--no-config", "--python", sys.executable],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("metadata", [
    '[project]\nname="fixture-plugin"\nversion="1"\ndependencies=[]\n',
    '[build-system]\nrequires=[]\nbuild-backend="fixture_backend"\n',
])
def test_packaging_declaration_keeps_precedence(tmp_path, metadata):
    project = tmp_path / "pyproject.toml"
    project.write_text(metadata, encoding="utf-8")
    (tmp_path / "plugin.yaml").write_text("python_dependencies: [ignored==1]\n", encoding="utf-8")
    declaration = read_python_declaration(tmp_path)
    assert declaration.pyproject == project
    assert declaration.requirements == ()
    assert declaration.is_member


def test_external_runtime_stays_out_with_tooling_and_manifest_dependencies(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    (tmp_path / "plugin.yaml").write_text(
        "python_runtime: external\npython_dependencies: [fixturedep==1]\n", encoding="utf-8",
    )
    declaration = read_python_declaration(tmp_path)
    assert declaration.external
    assert not declaration.is_member
    assert declaration.install_requirements == ()


@pytest.mark.parametrize("manifest", ['python_dependencies: wrong\n', 'pip_dependencies: [123]\n'])
def test_tooling_does_not_hide_invalid_manifest_dependencies(tmp_path, manifest):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    (tmp_path / "plugin.yaml").write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid .*dependencies"):
        read_python_declaration(tmp_path)


def test_malformed_toml_still_fails_closed(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff\n", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        read_python_declaration(tmp_path)
