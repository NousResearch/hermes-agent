"""PM's resolver must not depend on the application it is repairing."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_pm_runtime_discovers_plugins_without_application_dependencies(tmp_path, monkeypatch):
    from pm.runtime import prepare_runtime

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the real dependency-runtime test")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(home / "tools"))
    (home / "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    python = prepare_runtime(Path(uv), Path(sys.executable), tmp_path / "runtime")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "UV_"))}
    env.update(HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(home / "tools"))
    # Import real PM, including its production plugin-discovery chain.
    code = f"""
import importlib.util, json, sys
sys.path.insert(0, {str(repo)!r})
from pm.workspace import enabled_member_dirs
from pm.plugins_state import _read_home_config
from pathlib import Path
assert _read_home_config(Path({str(home)!r}))["plugins"]["enabled"] == []
assert enabled_member_dirs() == []
assert importlib.util.find_spec("openai") is None
assert importlib.util.find_spec("yaml") is None
print(json.dumps({{"prefix": sys.prefix, "yaml": importlib.util.find_spec("ruamel.yaml").origin}}))
"""
    result = subprocess.run([str(python), "-I", "-B", "-c", code], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert Path(report["yaml"]).is_relative_to(Path(report["prefix"]))
    assert prepare_runtime(Path(uv), Path(sys.executable), tmp_path / "runtime", offline=True) == python
    # Repair PM itself from its own lock, without trusting an existing marker.
    (Path(report["yaml"]).parent / "main.py").unlink()
    repaired = prepare_runtime(Path(uv), Path(sys.executable), tmp_path / "runtime", offline=True)
    assert repaired != python
    checked = subprocess.run([str(repaired), "-I", "-B", "-c", code], env=env,
                             capture_output=True, text=True, timeout=30)
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_cold_worker_bootstrap_reuses_the_requests_cache(tmp_path, monkeypatch):
    import pm
    from hermes_constants import get_default_hermes_root
    from pm import client, runtime
    from pm.runtime_stage import stage_runtime

    uv = shutil.which("uv")
    assert uv, "the bootstrap cache contract requires real uv"
    tools = Path(uv), Path(sys.executable)
    cache = tmp_path / "shared-cache"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home/.hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setattr("pm.paths.repo_root", lambda: tmp_path / "project")
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: tools)
    monkeypatch.setattr(client, "is_runtime", lambda: False)
    stage_runtime(*tools, tmp_path / "warmup", cache=cache)
    shutil.rmtree(tmp_path / "warmup")

    def offline_stage(*args, **kwargs):
        # A fresh manager must use the populated explicit cache, not download
        # its dependencies again under the bundle's isolated HOME.
        kwargs["offline"] = True
        return stage_runtime(*args, **kwargs)

    monkeypatch.setattr("pm.runtime_stage.stage_runtime", offline_stage)
    worker = Path(client.__file__).with_name("worker.py")
    script = (
        "import runpy, sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(worker.parent.parent)!r}); import pm._uv; "
        f"pm._uv._toolchain = lambda **kwargs: (Path({uv!r}), Path({sys.executable!r})); "
        f"runpy.run_path({str(worker)!r}, run_name='__main__')"
    )

    def command(*args, **kwargs):
        prepared = runtime.runtime_command(*args, **kwargs)
        return [*prepared[:3], "-c", script]

    monkeypatch.setattr(client, "runtime_command", command)
    before = dict(os.environ)
    pm.prune_cache(cache)
    assert cache.is_dir()
    assert not (get_default_hermes_root() / "cache/uv").exists(), "bootstrap created an unshared private cache"
    assert dict(os.environ) == before


def test_sealed_worker_command_uses_only_its_recorded_site(tmp_path, monkeypatch):
    from pm import paths
    from pm.runtime import runtime_command

    repo = tmp_path / "payload" / "hermes-agent"
    repo.mkdir(parents=True)
    (repo.parent / "manifest.json").write_text('{"repo":"hermes-agent"}')
    runtime = repo.parent / "pm-runtime"
    site = runtime / "site"
    site.mkdir(parents=True)
    base = repo.parent / "python"
    if os.name == "nt":
        shutil.copytree(Path(sys.base_prefix), base)
        python = base / "python.exe"
    else:
        base.mkdir()
        python = base / "python"
        shutil.copy2(Path(sys._base_executable).resolve(), python)
    (runtime / "pm-runtime.json").write_text(json.dumps({
        "python": "../python/" + python.name, "sitePackages": "site",
    }))
    script = repo / "probe.py"
    script.write_text("import sys,json; print(json.dumps(sys.path))")
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    command = runtime_command(script)
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    entries = json.loads(result.stdout)
    assert str(site) in entries
    assert not any(Path(entry).name in {"site-packages", "dist-packages"} for entry in entries)
