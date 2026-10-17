"""Tool-only pyprojects must not suppress plugin dependency manifests."""
from pathlib import Path

import pytest

from pm.plugin_declarations import read_python_declaration


@pytest.mark.parametrize('manifest_name', ['plugin.yaml', 'plugin.yml'])
@pytest.mark.parametrize('dependency_key', ['python_dependencies', 'pip_dependencies'])
def test_tool_only_preserves_manifest_dependencies(tmp_path: Path, manifest_name: str, dependency_key: str) -> None:
    manifest = tmp_path / manifest_name
    manifest.write_text(f'name: fixture\n{dependency_key}: ["requests>=2.32,<3"]\n')
    project = tmp_path / 'pyproject.toml'
    project.write_text('[tool.ruff]\ntarget-version="py311"\n')
    before = {p: p.read_bytes() for p in (manifest, project)}
    declaration = read_python_declaration(tmp_path)
    assert declaration.pyproject is None
    assert declaration.install_requirements == ('requests>=2.32,<3',)
    assert declaration.is_member
    assert set(declaration.files) == {manifest, project}
    assert all(p.read_bytes() == value for p, value in before.items())


def test_tool_only_without_dependencies_is_not_a_package(tmp_path: Path) -> None:
    (tmp_path / 'plugin.yaml').write_text('name: fixture\n')
    project = tmp_path / 'pyproject.toml'
    project.write_text('[tool.ruff]\ntarget-version="py311"\n')
    declaration = read_python_declaration(tmp_path)
    assert not declaration.is_member
    assert project in declaration.files


@pytest.mark.parametrize('metadata', [
    '[project]\nname="fixture"\nversion="1"\ndependencies=["httpx>=0.28,<1"]\n',
    '[build-system]\nrequires=["setuptools>=70,<81"]\nbuild-backend="setuptools.build_meta"\n',
])
def test_real_packaging_keeps_precedence(tmp_path: Path, metadata: str) -> None:
    (tmp_path / 'plugin.yaml').write_text('name: fixture\npython_dependencies: ["ignored>=1,<2"]\n')
    project = tmp_path / 'pyproject.toml'
    project.write_text(metadata)
    declaration = read_python_declaration(tmp_path)
    assert declaration.pyproject == project
    assert declaration.is_member
    assert 'ignored>=1,<2' not in declaration.requirements


@pytest.mark.parametrize('legacy', ['setup.py', 'setup.cfg'])
def test_legacy_packaging_with_tool_settings_still_is_a_package(tmp_path: Path, legacy: str) -> None:
    project = tmp_path / 'pyproject.toml'
    project.write_text('[tool.ruff]\ntarget-version="py311"\n')
    (tmp_path / legacy).write_text('# packaging input\n')
    declaration = read_python_declaration(tmp_path)
    assert declaration.pyproject == project
    assert declaration.is_member


def test_external_runtime_stays_excluded(tmp_path: Path) -> None:
    (tmp_path / 'plugin.yaml').write_text('name: fixture\npython_runtime: external\npython_dependencies: ["requests>=2.32,<3"]\n')
    (tmp_path / 'pyproject.toml').write_text('[tool.ruff]\ntarget-version="py311"\n')
    declaration = read_python_declaration(tmp_path)
    assert declaration.external
    assert not declaration.is_member


def test_malformed_tool_toml_is_not_ignored(tmp_path: Path) -> None:
    (tmp_path / 'pyproject.toml').write_text('[tool.ruff\n')
    with pytest.raises(ValueError):
        read_python_declaration(tmp_path)
