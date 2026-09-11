"""Workspace generation carries the source inputs of a buildable core."""
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pm import workspace
from pm.packages import uv_env


@pytest.fixture(autouse=True)
def isolated_machine_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


def test_real_build_inputs_stay_in_generated_root(tmp_path, monkeypatch):
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project]\nname="buildable-core"\nversion="1.0"\nreadme="README.md"\n'
        'requires-python=">=3.11"\nlicense="MIT"\nlicense-files=["LICENSE"]\n'
        '[build-system]\nrequires=["setuptools==83.0.0","wheel==0.46.3"]\n'
        'build-backend="setuptools.build_meta"\n'
        '[tool.setuptools.packages.find]\ninclude=["buildable_core", "buildable_core.*"]\n', encoding="utf-8",
    )
    (core / "README.md").write_text("# Real core\n")
    (core / "LICENSE").write_text("MIT\n")
    package = core / "buildable_core"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'from actual source'\n")
    (core / ".env").write_text("must not copy")
    monkeypatch.setattr(workspace.paths, "repo_root", lambda: core)
    root, venv = tmp_path / "staging", tmp_path / "venv"
    uv = shutil.which("uv")
    assert uv is not None
    monkeypatch.setattr(importlib.import_module("pm.ensure"), "uv",
                        lambda **kwargs: (uv, {**uv_env(kwargs.get("base_env")), "UV_PYTHON": sys.executable}))
    workspace.lock_and_sync([], [], root=root, venv_dir=venv)
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    probe = subprocess.run([str(python), "-c", "import buildable_core; print(buildable_core.VALUE)"],
                           cwd=tmp_path, text=True, capture_output=True, check=True, timeout=30)
    assert probe.stdout.strip() == "from actual source"
    assert not (root / ".env").exists()
    assert not list(core.glob("*.egg-info")), "build must not write into the original core"
    assert not (core / "uv.lock").exists()


def test_source_refresh_does_not_need_metadata_change_and_refuses_live_root(tmp_path, monkeypatch):
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text('[project]\nname="x"\nversion="1"\n')
    (core / "code.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(workspace.paths, "repo_root", lambda: core)
    staged = tmp_path / "staged"
    workspace.build_root([], root=staged)
    (core / "code.py").write_text("VALUE = 2\n")
    workspace.build_root([], root=staged)
    assert (staged / "code.py").read_text() == "VALUE = 2\n"
    before = (core / "code.py").read_bytes()
    with pytest.raises(workspace.InstallError, match="source"):
        workspace.build_root([], root=core)
    assert (core / "code.py").read_bytes() == before


def test_legacy_member_is_generated_only_inside_workspace(tmp_path, monkeypatch):
    import tomllib
    core, plugin = tmp_path / "core", tmp_path / "readonly-plugin"
    core.mkdir(); plugin.mkdir()
    (core / "pyproject.toml").write_text('[project]\nname="core"\nversion="1"\n')
    manifest = plugin / "plugin.yaml"
    manifest.write_text('name: legacy\npython_dependencies: ["example>=1,<2"]\n')
    monkeypatch.setattr(workspace.paths, "repo_root", lambda: core)
    stamp = workspace.members_stamp([plugin])
    generated = workspace.build_root([plugin], tmp_path / "stage")
    metadata = tomllib.loads((generated / "pyproject.toml").read_text())
    member = (generated / metadata["tool"]["uv"]["workspace"]["members"][0]).resolve()
    assert member.is_relative_to(generated)
    assert not (plugin / "pyproject.toml").exists()
    assert tomllib.loads((member / "pyproject.toml").read_text())["project"]["dependencies"] == ["example>=1,<2"]
    manifest.write_text('name: legacy\npython_dependencies: ["example>=2,<3"]\n')
    assert workspace.members_stamp([plugin]) != stamp


def _wheel(directory, name, version, requirements=()):
    import zipfile

    metadata = f"{name}-{version}.dist-info"
    entries = {
        f"{name}/__init__.py": f"__version__ = {version!r}\n",
        f"{metadata}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements),
        f"{metadata}/WHEEL": "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    entries[f"{metadata}/RECORD"] = "".join(f"{path},,\n" for path in entries)
    with zipfile.ZipFile(directory / f"{name}-{version}-py3-none-any.whl", "w") as archive:
        for path, body in entries.items():
            archive.writestr(path, body)


@pytest.mark.parametrize("exact", [False, True])
def test_plugin_can_move_compatible_transitive_but_not_exact_requirement(tmp_path, monkeypatch, exact):
    import os
    import tomllib

    core = tmp_path / "core"
    core.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "pkga", "1.0", ["pkgb>=1.2,<2"])
    _wheel(wheels, "pkgb", "1.2")
    core_requirement = '["pkga==1.0", "pkgb==1.2"]' if exact else '["pkga==1.0"]'
    (core / "pyproject.toml").write_text(
        '[project]\nname="core-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        f'dependencies={core_requirement}\n[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    uv = shutil.which("uv")
    assert uv
    monkeypatch.setattr(workspace.paths, "repo_root", lambda: core)
    monkeypatch.setattr(importlib.import_module("pm.ensure"), "uv",
                        lambda **kwargs: (uv, {**uv_env(kwargs.get("base_env")), "UV_PYTHON": sys.executable}))
    baseline, first_env = tmp_path / "baseline", tmp_path / "first-env"
    workspace.lock_and_sync([], [], root=baseline, venv_dir=first_env)
    first_lock = (baseline / "uv.lock").read_bytes()
    assert next(p["version"] for p in tomllib.loads(first_lock.decode())["package"] if p["name"] == "pkgb") == "1.2"
    _wheel(wheels, "pkgb", "1.3")
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "pyproject.toml").write_text(
        '[project]\nname="plugin-proof"\nversion="1"\nrequires-python=">=3.11"\ndependencies=["pkgb==1.3"]\n',
        encoding="utf-8",
    )
    extended, candidate = tmp_path / "extended", tmp_path / "candidate"
    if exact:
        with pytest.raises(workspace.ResolutionConflict):
            workspace.lock_and_sync([plugin], [], root=extended, venv_dir=candidate, seed_lock=baseline / "uv.lock")
    else:
        workspace.lock_and_sync([plugin], [], root=extended, venv_dir=candidate, seed_lock=baseline / "uv.lock")
        installed = tomllib.loads((extended / "uv.lock").read_text(encoding="utf-8"))["package"]
        assert next(p["version"] for p in installed if p["name"] == "pkgb") == "1.3"
        python = candidate / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        result = subprocess.run([str(python), "-c", "import pkga, pkgb; print(pkga.__version__, pkgb.__version__)"],
                                capture_output=True, text=True, check=True, timeout=30)
        assert result.stdout.strip() == "1.0 1.3"
    assert (baseline / "uv.lock").read_bytes() == first_lock
