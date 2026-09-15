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
    shutil.copytree(source / "pm", repo / "pm", ignore=shutil.ignore_patterns("__pycache__", ".hermes-tmp.*"))
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
        # Relocatable PBS does not retain the host's stdlib prefix. Supply the
        # fixture's promised host library tree, including libpython on macOS.
        (python.parent.parent / "lib").symlink_to(Path(sys.base_prefix) / "lib", target_is_directory=True)
    stage = getattr(native, "stage_pm_runtime", None)
    assert callable(stage), "native payload has no isolated PM runtime stage"
    cache = tmp_path / "build-cache"
    monkeypatch.setattr("pm.client._request", lambda *args, **kwargs: pytest.fail("runtime staging must not bootstrap a worker"))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    stage(root, python, repo, cache=cache)
    assert cache.is_dir()
    assert not (tmp_path / "home/cache/uv").exists()
    stage(root, python, repo, offline=True, cache=cache)
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
    from pm import stage_manager_runtime

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
    monkeypatch.setattr("pm.client._request", lambda *args, **kwargs: pytest.fail("runtime staging must not bootstrap a worker"))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    executable = stage_manager_runtime(python=Path(sys.executable), destination=tmp_path / "runtime")
    assert executable.is_file()
