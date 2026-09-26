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


def test_copy_core_inputs_scopes_exclusions_to_root_and_carries_metadata(tmp_path):
    """Only the root lock/dotfiles are dropped; nested ones are build inputs.

    Regression test for #122425: the snapshot excluded ``uv.lock`` and every
    dotfile at every depth (so ``pm/uv.lock`` vanished and ``pm doctor``
    crashed), and it never carried ``install-stamp.json`` / ``.install_method``
    (so the managed venv reported ``vunknown``).
    """
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "pkg").mkdir(parents=True)
    (src / "pyproject.toml").write_text(
        '[project]\nname="scoped-core"\nversion="1"\nrequires-python=">=3.11"\n', encoding="utf-8")
    (src / "pkg" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (src / "uv.lock").write_text("ROOT LOCK\n", encoding="utf-8")
    (src / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (src / "install-stamp.json").write_text('{"baseVersion": "0.0.0"}\n', encoding="utf-8")
    (src / ".install_method").write_text("git\n", encoding="utf-8")
    (src / "pkg" / "uv.lock").write_text("NESTED LOCK\n", encoding="utf-8")
    (src / "pkg" / ".keep").write_text("nested dotfile is a build input\n", encoding="utf-8")
    (src / "pm").mkdir()
    (src / "pm" / "uv.lock").write_text("PM LOCK\n", encoding="utf-8")
    (src / ".git").mkdir()
    (src / ".git" / "config").write_text("gitdir\n", encoding="utf-8")

    workspace._copy_core_inputs(src, dst)

    assert (dst / "pkg" / "__init__.py").is_file()
    assert (dst / "pkg" / "uv.lock").read_text() == "NESTED LOCK\n"
    assert (dst / "pkg" / ".keep").is_file()
    assert (dst / "pm" / "uv.lock").read_text() == "PM LOCK\n"
    assert (dst / "install-stamp.json").is_file()
    assert (dst / ".install_method").read_text() == "git\n"
    assert not (dst / "uv.lock").exists()
    assert not (dst / ".env").exists()
    assert not (dst / ".git").exists()


def test_sync_sources_refreshes_code_but_keeps_lock(tmp_path, monkeypatch):
    """``sync_sources`` re-snapshots a committed generation without re-resolving.

    Regression test for #122425: after an update the committed workspace keeps
    running old code; the refresh carries the new sources (and the install
    metadata) while the committed ``uv.lock`` stays byte-identical.
    """
    import os

    from pm.lock import Facts

    project = tmp_path / "project"
    project.mkdir()
    core = tmp_path / "core"
    core.mkdir()
    _buildable_source(core)
    (core / "build/backend.py").rename(core / "backend.py")
    metadata = core / "pyproject.toml"
    metadata.write_text(metadata.read_text().replace('backend-path=["build"]', 'backend-path=["."]'))
    (core / "install-stamp.json").write_text('{"baseVersion": "9.9.9"}\n', encoding="utf-8")
    (core / ".install_method").write_text("git\n", encoding="utf-8")
    (core / "replay_plugin" / "doomed.py").write_text("OLD = True\n", encoding="utf-8")
    monkeypatch.setattr("pm.environments.install_state_dir", lambda _root: tmp_path / "state")
    monkeypatch.setattr("pm.environments.runtime_facts_path",
                        lambda _root: tmp_path / "state" / "facts.json")
    monkeypatch.setattr(workspace.paths, "repo_root", lambda: core)
    generation = tmp_path / "state" / "environments" / "gen0"
    workspace_root, venv = generation / "workspace", generation / "venv"
    uv = shutil.which("uv")
    assert uv is not None
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    workspace.lock_and_sync([], [], root=workspace_root, source=core, seed_lock=None,
                            environment=managed_environment(venv))
    assert (workspace_root / "replay_plugin" / "doomed.py").is_file()
    before_lock = (workspace_root / "uv.lock").read_bytes()
    Facts(tmp_path / "state" / "facts.json").record_state(
        "venv", "test-stamp", [], environment=venv, resolved_lock=workspace_root / "uv.lock")

    (core / "replay_plugin" / "values.py").write_text("VALUE = 'refreshed bytes'\n", encoding="utf-8")
    (core / "replay_plugin" / "doomed.py").unlink()
    (core / "replay_plugin" / "added.py").write_text("NEW = True\n", encoding="utf-8")

    synced = workspace.sync_sources(project, source=core, plugin_dirs=[])

    assert synced == workspace_root
    assert (workspace_root / "uv.lock").read_bytes() == before_lock
    assert (workspace_root / "replay_plugin" / "values.py").read_text() == "VALUE = 'refreshed bytes'\n"
    assert (workspace_root / "replay_plugin" / "added.py").is_file()
    assert not (workspace_root / "replay_plugin" / "doomed.py").exists()
    assert (workspace_root / "install-stamp.json").is_file()
    assert (workspace_root / ".install_method").is_file()
    assert not list(generation.glob("workspace.refresh-*"))
    assert not list(generation.glob("workspace.superseded-*"))
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    probe = subprocess.run([str(python), "-c", "import replay_plugin; print(replay_plugin.VALUE)"],
                           cwd=tmp_path, text=True, capture_output=True, check=True, timeout=120)
    assert probe.stdout.strip() == "refreshed bytes"
