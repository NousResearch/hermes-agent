"""Workspace generation carries the source inputs of a buildable core."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pm import workspace
from tests.pm import _fixtures
from pm.environment import managed_environment




def _buildable_source(plugin):
    (plugin / "build").mkdir(parents=True, exist_ok=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname="replay-plugin"\nversion="1.0"\nrequires-python=">=3.11"\nreadme="README.md"\nlicense="MIT"\nlicense-files=["LICENSE"]\n'
        '[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["build"]\n',
        encoding="utf-8",
    )
    (plugin / "plugin.yaml").write_text("name: replay-plugin\n", encoding="utf-8")
    (plugin / "replay_plugin").mkdir()
    (plugin / "replay_plugin/__init__.py").write_text("from .values import VALUE\n", encoding="utf-8")
    (plugin / "replay_plugin/values.py").write_text("VALUE = 'recorded plugin bytes'\n", encoding="utf-8")
    (plugin / "README.md").write_text("# fixture readme")
    (plugin / "LICENSE").write_text("MIT")
    # The backend must consume the copied metadata inputs too.
    (plugin / "build/backend.py").write_text('''
from pathlib import Path
from zipfile import ZipFile

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    assert Path("README.md").read_text() == "# fixture readme"
    assert Path("LICENSE").read_text() == "MIT"
    name = "replay_plugin-1.0-py3-none-any.whl"
    dist = "replay_plugin-1.0.dist-info"
    entries = {
        "replay_plugin/__init__.py": Path("replay_plugin/__init__.py").read_bytes(),
        "replay_plugin/values.py": Path("replay_plugin/values.py").read_bytes(),
        dist + "/METADATA": "Metadata-Version: 2.1\\nName: replay-plugin\\nVersion: 1.0\\n",
        dist + "/WHEEL": "Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",
    }
    entries[dist + "/RECORD"] = "".join(path + ",,\\n" for path in entries)
    with ZipFile(Path(wheel_directory) / name, "w") as wheel:
        for path, body in entries.items():
            wheel.writestr(path, body)
    return name

build_editable = build_wheel
''', encoding="utf-8")


def test_real_build_inputs_stay_in_generated_root(tmp_path, monkeypatch):
    core = tmp_path / "core"
    core.mkdir()
    _buildable_source(core)
    # Core snapshots exclude build/ output; the member replay below deliberately
    # retains that backend-path. Keep the core backend at the source root.
    (core / "build/backend.py").rename(core / "backend.py")
    metadata = core / "pyproject.toml"
    metadata.write_text(metadata.read_text().replace('backend-path=["build"]', 'backend-path=["."]'))
    (core / ".env").write_text("must not copy")
    # Nested artifact dirs inside a copied package root are build inputs, not
    # output: PM hashes ``pm/uv.lock`` for its own runtime identity, and bundled
    # dashboard plugins ship a tracked ``dist/`` the dashboard serves. Only the
    # root copies (which the workspace regenerates) are dropped.
    nested = core / "pm"
    nested.mkdir()
    (nested / "pyproject.toml").write_text('[project]\nname="nested"\nversion="1"\n')
    (nested / "uv.lock").write_text("NESTED LOCK\n")
    (nested / "dist").mkdir()
    (nested / "dist/asset.js").write_text("ASSET\n")
    monkeypatch.setattr(workspace.paths, "repo_root", lambda: core)
    root, venv = tmp_path / "staging", tmp_path / "venv"
    uv = shutil.which("uv")
    assert uv is not None
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    workspace.lock_and_sync([], [], root=root, source=core, seed_lock=None,
                            environment=managed_environment(venv))
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    probe = subprocess.run([str(python), "-c", "import replay_plugin; print(replay_plugin.VALUE)"],
                           cwd=tmp_path, text=True, capture_output=True, check=True, timeout=30)
    assert probe.stdout.strip() == "recorded plugin bytes"
    assert not (root / ".env").exists()
    assert (root / "pm" / "uv.lock").read_text() == "NESTED LOCK\n"
    assert (root / "pm" / "dist" / "asset.js").read_text() == "ASSET\n"
    assert not (root / "dist").exists(), "the root dist stays excluded"
    assert not list(core.glob("*.egg-info")), "build must not write into the original core"
    assert not (core / "uv.lock").exists()


def test_repair_replays_saved_in_tree_build_backend(tmp_path):
    import os
    import tomllib

    from pm.environment import PythonEnvironment

    core, plugin = tmp_path / "core", tmp_path / "plugin"
    core.mkdir()
    (plugin / "build").mkdir(parents=True)
    (core / "pyproject.toml").write_text(
        '[project]\nname="replay-core"\nversion="1"\nrequires-python=">=3.11"\n'
        '[tool.uv]\npackage=false\nno-index=true\n', encoding="utf-8",
    )
    _buildable_source(plugin)
    inputs = {p.relative_to(plugin): p.read_bytes() for p in plugin.rglob("*") if p.is_file()}
    uv = shutil.which("uv")
    assert uv, "saved backend replay test requires real uv"
    saved, repaired = tmp_path / "saved", tmp_path / "repaired"
    initial = PythonEnvironment(
        uv=Path(uv), python=Path(sys.executable), destination=tmp_path / "initial-env",
        cache=tmp_path / "initial-cache", env=dict(os.environ), offline=True,
    )
    workspace.lock_and_sync([plugin], [], root=saved, source=core, seed_lock=None,
                            environment=initial)
    [relative] = tomllib.loads((saved / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]["members"]
    assert all((saved / relative / path).read_bytes() == data for path, data in inputs.items())
    saved_lock = (saved / "uv.lock").read_bytes()

    # Neither live manifests nor the live backend can provide repair's build inputs.
    (core / "pyproject.toml").write_text("damaged [", encoding="utf-8")
    (plugin / "pyproject.toml").write_text("damaged [", encoding="utf-8")
    (plugin / "plugin.yaml").write_text("damaged [", encoding="utf-8")
    (plugin / "build/backend.py").unlink()
    (plugin / "replay_plugin/values.py").write_text("raise RuntimeError('damaged live source')\n", encoding="utf-8")
    repair = PythonEnvironment(
        uv=Path(uv), python=Path(sys.executable), destination=tmp_path / "repair-env",
        # A fresh cache forces uv to invoke the saved backend again, not reuse a wheel.
        cache=tmp_path / "repair-cache", env=dict(os.environ), offline=True,
    )
    workspace.lock_and_sync([plugin], [], root=repaired, source=core, seed_lock=None,
                            replay=saved, environment=repair)
    assert (repaired / "uv.lock").read_bytes() == saved_lock
    assert (saved / "uv.lock").read_bytes() == saved_lock
    for root in (saved, repaired):
        assert all((root / relative / path).read_bytes() == data for path, data in inputs.items())
    for environment in (initial, repair):
        probe = subprocess.run(
            [str(environment.executable), "-I", "-c", "import replay_plugin; print(replay_plugin.VALUE)"],
            cwd=tmp_path, text=True, capture_output=True, check=True, timeout=30,
        )
        assert probe.stdout.strip() == "recorded plugin bytes"
    assert (core / "pyproject.toml").read_text() == "damaged ["
    assert (plugin / "pyproject.toml").read_text() == "damaged ["
    assert (plugin / "plugin.yaml").read_text() == "damaged ["
    assert not (plugin / "build/backend.py").exists()


def test_source_refresh_does_not_need_metadata_change_and_refuses_live_root(tmp_path, monkeypatch):
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text('[project]\nname="x"\nversion="1"\nrequires-python=">=3.14"\n[tool.uv]\npackage=false\nno-index=true\n')
    (core / "code.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(workspace.paths, "repo_root", lambda: core)
    staged = tmp_path / "staged"
    uv = shutil.which("uv")
    assert uv
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    workspace.lock_and_sync([], [], root=staged, source=core, seed_lock=None,
                            environment=managed_environment(tmp_path / "env"))
    (core / "code.py").write_text("VALUE = 2\n")
    fresh = tmp_path / "fresh"
    workspace.lock_and_sync([], [], root=fresh, source=core, seed_lock=staged / "uv.lock",
                            environment=managed_environment(tmp_path / "fresh-env"))
    assert (staged / "code.py").read_text() == "VALUE = 1\n"
    staged = fresh
    assert (staged / "code.py").read_text() == "VALUE = 2\n"
    before = (core / "code.py").read_bytes()
    with pytest.raises(workspace.InstallError, match="fresh"):
        workspace.lock_and_sync([], [], root=core, source=core, seed_lock=None,
                                environment=managed_environment(tmp_path / "env"))
    assert (core / "code.py").read_bytes() == before


def test_legacy_member_is_generated_only_inside_workspace(tmp_path, monkeypatch):
    import tomllib
    core, plugin = tmp_path / "core", tmp_path / "readonly-plugin"
    core.mkdir(); plugin.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "example", "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="core"\nversion="1"\nrequires-python=">=3.14"\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n')
    manifest = plugin / "plugin.yaml"
    manifest.write_text('name: legacy\npython_dependencies: ["example>=1,<2"]\n')
    monkeypatch.setattr(workspace.paths, "repo_root", lambda: core)
    stamp = workspace.members_stamp([plugin])
    uv = shutil.which("uv")
    assert uv
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    generated = tmp_path / "stage"
    workspace.lock_and_sync([plugin], [], root=generated, source=core, seed_lock=None,
                            environment=managed_environment(tmp_path / "env"))
    metadata = tomllib.loads((generated / "pyproject.toml").read_text())
    member = (generated / metadata["tool"]["uv"]["workspace"]["members"][0]).resolve()
    assert member.is_relative_to(generated)
    assert not (plugin / "pyproject.toml").exists()
    assert tomllib.loads((member / "pyproject.toml").read_text())["project"]["dependencies"] == ["example>=1,<2"]
    manifest.write_text('name: legacy\npython_dependencies: ["example>=2,<3"]\n')
    assert workspace.members_stamp([plugin]) != stamp


@pytest.mark.parametrize("exact", [False, True])
def test_plugin_can_move_compatible_transitive_but_not_exact_requirement(tmp_path, monkeypatch, exact):
    import os
    import tomllib

    core = tmp_path / "core"
    core.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "pkga", "1.0", ["pkgb>=1.2,<2"])
    _fixtures._wheel(wheels, "pkgb", "1.2")
    core_requirement = '["pkga==1.0", "pkgb==1.2"]' if exact else '["pkga==1.0"]'
    (core / "pyproject.toml").write_text(
        '[project]\nname="core-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        f'dependencies={core_requirement}\n[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    uv = shutil.which("uv")
    assert uv
    monkeypatch.setattr(workspace.paths, "repo_root", lambda: core)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    baseline, first_env = tmp_path / "baseline", tmp_path / "first-env"
    workspace.lock_and_sync([], [], root=baseline, source=core, seed_lock=None,
                            environment=managed_environment(first_env))
    first_lock = (baseline / "uv.lock").read_bytes()
    assert next(p["version"] for p in tomllib.loads(first_lock.decode())["package"] if p["name"] == "pkgb") == "1.2"
    _fixtures._wheel(wheels, "pkgb", "1.3")
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "pyproject.toml").write_text(
        '[project]\nname="plugin-proof"\nversion="1"\nrequires-python=">=3.11"\ndependencies=["pkgb==1.3"]\n',
        encoding="utf-8",
    )
    extended, candidate = tmp_path / "extended", tmp_path / "candidate"
    if exact:
        with pytest.raises(workspace.ResolutionConflict):
            workspace.lock_and_sync([plugin], [], root=extended, source=core, seed_lock=baseline / "uv.lock",
                                    environment=managed_environment(candidate))
    else:
        workspace.lock_and_sync([plugin], [], root=extended, source=core, seed_lock=baseline / "uv.lock",
                                    environment=managed_environment(candidate))
        installed = tomllib.loads((extended / "uv.lock").read_text(encoding="utf-8"))["package"]
        assert next(p["version"] for p in installed if p["name"] == "pkgb") == "1.3"
        python = candidate / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        result = subprocess.run([str(python), "-c", "import pkga, pkgb; print(pkga.__version__, pkgb.__version__)"],
                                capture_output=True, text=True, check=True, timeout=30)
        assert result.stdout.strip() == "1.0 1.3"
    assert (baseline / "uv.lock").read_bytes() == first_lock
