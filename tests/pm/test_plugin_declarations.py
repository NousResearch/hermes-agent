"""Plugin declaration safeguards from #113851, through PM's real resolver."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

from pm.environment import PythonEnvironment
from pm.plugin_declarations import read_python_declaration, unsupported_requirements
from pm.workspace import _generate_pyproject, enabled_member_dirs, lock_and_sync
from tests.pm import _fixtures

@pytest.mark.parametrize("manifest_deps", [False, True])
def test_tooling_only_pyproject_falls_back_to_manifest_during_workspace_lock(
        tmp_path, manifest_deps):
    """Regression for #122148: a lint config is not a uv workspace project."""
    uv = shutil.which("uv")
    assert uv is not None, "real resolver is required"
    core = tmp_path / "core"
    core.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "fixturedep", "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="core"\nversion="1"\nrequires-python=">=3.11"\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "pyproject.toml").write_text('[tool.ruff]\nline-length=88\n', encoding="utf-8")
    if manifest_deps:
        (plugin / "plugin.yaml").write_text(
            'name: plugin\npython_dependencies: ["fixturedep>=1,<2"]\n', encoding="utf-8",
        )
    declaration = read_python_declaration(plugin)
    assert declaration.pyproject is None
    assert plugin / "pyproject.toml" in declaration.files
    assert declaration.is_member == manifest_deps
    assert declaration.requirements == (("fixturedep>=1,<2",) if manifest_deps else ())

    root = tmp_path / "workspace"
    _generate_pyproject([plugin], root, source=core)
    result = subprocess.run([uv, "lock", "--offline", "--python", sys.executable],
                            cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    root_doc = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    members = root_doc.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
    assert bool(members) == manifest_deps
    if manifest_deps:
        member = tomllib.loads((root / members[0] / "pyproject.toml").read_text(encoding="utf-8"))
        assert member["project"]["dependencies"] == ["fixturedep>=1,<2"]
    assert (plugin / "pyproject.toml").read_text(encoding="utf-8") == '[tool.ruff]\nline-length=88\n'


@pytest.mark.parametrize("modern", [False, True])
def test_declaration_policy_survives_cross_profile_resolution(tmp_path, monkeypatch, modern):
    home = tmp_path / "custom-home"
    sibling = home / "profiles" / "work"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(sibling))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    core = tmp_path / "core"
    core.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "fixturedep", "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="core"\nversion="1"\nrequires-python=">=3.11"\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    plugin = home / "plugins" / "member"
    external = sibling / "plugins" / "sidecar"
    plugin.mkdir(parents=True)
    external.mkdir(parents=True)
    (home / "config.yaml").write_text("plugins:\n  enabled: [member]\n", encoding="utf-8")
    (sibling / "config.yaml").write_text("memory:\n  provider: sidecar\n", encoding="utf-8")
    (external / "plugin.yaml").write_text(
        'name: sidecar\npython_runtime: external\npython_dependencies: ["impossible==99"]\n',
        encoding="utf-8",
    )
    (external / "pyproject.toml").write_text(
        '[project]\nname="sidecar"\nversion="1"\ndependencies=["impossible==99"]\n',
        encoding="utf-8",
    )
    specs = ['fixturedep>=1,<2', 'hermes-agent>=0.1,<1',
             'remote @ https://example.invalid/unreviewed.whl',
             'missing-other-python; python_version < "3.0"']
    (plugin / "plugin.yaml").write_text(
        'name: member\npython_dependencies: ' + json.dumps(["ignored==99"] if modern else specs) + '\n',
        encoding="utf-8",
    )
    if modern:
        (plugin / "pyproject.toml").write_text(
            '[project]\nname="member"\nversion="1"\nrequires-python=">=3.11"\n'
            f'dependencies={json.dumps(specs)}\n[tool.uv]\npackage=false\n', encoding="utf-8",
        )
    before = {p: p.read_bytes() for directory in (plugin, external) for p in directory.iterdir()}
    assert enabled_member_dirs() == [plugin]
    assert read_python_declaration(external).external
    declaration = read_python_declaration(plugin)
    assert declaration.requirements == tuple(specs)
    assert unsupported_requirements(declaration.requirements) == (specs[2],)
    uv = shutil.which("uv")
    assert uv is not None, "real resolver is required"
    environment = PythonEnvironment(
        uv=Path(uv), python=Path(sys.executable), destination=tmp_path / "env",
        cache=tmp_path / "cache", env=dict(os.environ), offline=True,
    )
    root = tmp_path / "snapshot"
    lock_and_sync(enabled_member_dirs(), [], root=root, source=core, seed_lock=None,
                  environment=environment)
    probe = subprocess.run(
        [str(environment.executable), "-I", "-c", "import fixturedep; print(fixturedep.__version__)"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    assert probe.stdout.strip() == "1.0"
    [relative] = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]["members"]
    generated = tomllib.loads((root / relative / "pyproject.toml").read_text())["project"]["dependencies"]
    assert generated == [specs[0], specs[3]], "target markers must reach uv, unsafe/self requirements must not"
    assert all(p.read_bytes() == body for p, body in before.items())


@pytest.mark.parametrize("modern", [False, True])
def test_invalid_requirement_refused_at_declaration_boundary(tmp_path, modern):
    if modern:
        path = tmp_path / "pyproject.toml"
        path.write_text('[project]\ndependencies=["not a requirement >="]\n', encoding="utf-8")
    else:
        path = tmp_path / "plugin.yaml"
        path.write_text('python_dependencies: ["not a requirement >="]\n', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        read_python_declaration(tmp_path)
    assert path.read_bytes() == before
