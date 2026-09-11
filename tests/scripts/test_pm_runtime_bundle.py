"""A payload carries PM's independent dependency graph, not the app's imports."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _exercise_relocated_pm_runtime(tmp_path, monkeypatch):
    from scripts.bundles import native

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    uv = shutil.which("uv")
    assert uv, "the packaging test requires uv"
    root = tmp_path / "build"
    repo = root / "hermes-agent"
    source = Path(__file__).resolve().parents[2]
    shutil.copytree(source / "pm", repo / "pm", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "hermes_cli").mkdir()
    for name in ("__init__.py", "runtime_paths.py", "runtime_state.py"):
        shutil.copy2(source / "hermes_cli" / name, repo / "hermes_cli" / name)
    shutil.copy2(source / "hermes_constants.py", repo / "hermes_constants.py")
    # Copy the base executable, not a venv's launcher. The test host provides
    # its stdlib; production's package stage provides the complete distribution.
    python = root / "tools" / "python" / ("python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    if os.name == "nt":
        shutil.copytree(Path(sys.base_prefix), python.parent, dirs_exist_ok=True)
    else:
        shutil.copy2(Path(sys._base_executable).resolve(), python)
    stage = getattr(native, "stage_pm_runtime", None)
    assert callable(stage), "native payload has no isolated PM runtime stage"
    stage(root, Path(uv), python, repo)
    stage(root, Path(uv), python, repo, offline=True)
    (root / "manifest.json").write_text(json.dumps({"repo": "hermes-agent"}))
    assert not (repo / ".venv").exists()
    moved = tmp_path / "installed elsewhere"
    root.rename(moved)
    runtime = moved / "pm-runtime"
    manifest = json.loads((runtime / "pm-runtime.json").read_text())
    base = runtime / manifest["python"]
    site = runtime / manifest["sitePackages"]
    assert base.is_file() and site.is_dir()
    assert not Path(manifest["python"]).is_absolute()
    probe = """
import importlib.util, json, sys
sys.path.insert(0, sys.argv[1])
from ruamel.yaml import YAML
import packaging, tomli_w
assert importlib.util.find_spec('openai') is None
assert importlib.util.find_spec('yaml') is None
print(json.dumps(YAML(typ='safe').load('isolated: true')))
"""
    checked = subprocess.run([str(base), "-I", "-S", "-B", "-c", probe, str(site)],
                             cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == {"isolated": True}
    from pm import runtime as runtime_api, paths
    monkeypatch.setattr(paths, "repo_root", lambda: moved / "hermes-agent")
    command = runtime_api.runtime_command(moved / "hermes-agent/pm/launch.py", ["status"])
    checked = subprocess.run(command, cwd=tmp_path, env=runtime_api.runtime_environment(),
                             capture_output=True, text=True, timeout=30)
    assert checked.returncode == 0, checked.stderr
    assert "no pm sync receipt" in checked.stdout
    assert str(root) not in (runtime / "pyvenv.cfg").read_text()
    for link in (runtime / "bin").glob("python*"):
        if link.is_symlink():
            assert not os.path.isabs(os.readlink(link))
            assert link.exists()


@pytest.mark.platforms("posix")
def test_native_pm_runtime_survives_payload_move(tmp_path, monkeypatch):
    _exercise_relocated_pm_runtime(tmp_path, monkeypatch)


@pytest.mark.platforms("windows")
def test_resident_pm_bypasses_windows_redirector_after_move(tmp_path, monkeypatch):
    _exercise_relocated_pm_runtime(tmp_path, monkeypatch)


@pytest.mark.parametrize("poison", ["cwd", "global"])
def test_pm_builder_ignores_ambient_uv_configuration(tmp_path, monkeypatch, poison):
    from pm.runtime_stage import stage_runtime

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "uv.toml" if poison == "cwd" else tmp_path / "config/uv/uv.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('required-version = "<0.1"\n', encoding="utf-8")
    uv = shutil.which("uv")
    assert uv
    executable = stage_runtime(Path(uv), Path(sys.executable), tmp_path / "runtime")
    assert executable.is_file()


def test_native_stage_builds_pm_before_application_environment(tmp_path, monkeypatch):
    from scripts.bundles import native, payload
    from types import SimpleNamespace

    class StopAfterPM(Exception):
        pass

    class Facts:
        def __init__(self, *args, **kwargs):
            pass

        def retain(self, names):
            pass

        def entries_in_use(self):
            return set()

        def get(self, name):
            return {"entry": "python"}

    calls = []
    lock = tmp_path / "hermes-agent/pm/lock.json"
    lock.parent.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve().parents[2] / "pm/lock.json", lock)
    monkeypatch.setattr(payload, "snapshot", lambda *args: None)
    monkeypatch.setattr(native, "_bundle_package_names", lambda: [])
    monkeypatch.setattr(native, "_install_names", lambda names: 0)
    monkeypatch.setattr(native, "Facts", Facts)
    monkeypatch.setattr(native, "_facts", Facts)
    monkeypatch.setattr(native, "pm_uv", lambda: ("uv", {}))
    monkeypatch.setattr(native, "_store", lambda: SimpleNamespace(entry=lambda name: tmp_path / "tools" / name))
    monkeypatch.setattr(native, "get_package", lambda name: SimpleNamespace(binary=lambda path, target: path / "python"))

    def staged(root, uv, python, repo):
        calls.append((root, uv, python, repo))
        raise StopAfterPM

    monkeypatch.setattr(native, "stage_pm_runtime", staged)
    with pytest.raises(StopAfterPM):
        native._stage_native(SimpleNamespace(out=str(tmp_path), ref="HEAD"))
    assert calls == [(tmp_path, Path("uv"), tmp_path / "tools/python/python", tmp_path / "hermes-agent")]
