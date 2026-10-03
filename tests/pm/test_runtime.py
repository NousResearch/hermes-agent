"""PM's resolver must not depend on the application it is repairing."""
import hashlib
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
from pm.plugins_state import read_home_selection
from pathlib import Path
assert read_home_selection(Path({str(home)!r}))["plugins"]["enabled"] == []
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


def test_runtime_bootstrap_uses_pinned_python_when_facts_are_unavailable(tmp_path, monkeypatch):
    from pm import runtime

    project = tmp_path / "project"
    project.mkdir()
    managed_uv = tmp_path / "store" / "uv"
    managed_python = tmp_path / "store" / "python"
    managed_uv.parent.mkdir()
    managed_uv.touch()
    managed_python.touch()
    captured = {}
    toolchain_calls = []

    class Lockfile:
        def __init__(self, _path):
            pass

        def version(self, name):
            return f"{name}-version"

    class Package:
        def __init__(self, name):
            self.name = name

        def store_entry(self, version, target):
            return f"{self.name}-{version}-{target}"

        def binary(self, _root, _target):
            return managed_uv if self.name == "uv" else managed_python

    def prepare(uv, python, root, **kwargs):
        captured.update(uv=uv, python=python, root=root, kwargs=kwargs)
        return python

    monkeypatch.setattr(runtime, "is_runtime", lambda: False)
    monkeypatch.setattr(runtime, "_resident_runtime", lambda: None)
    monkeypatch.setattr(runtime, "prepare_runtime", prepare)
    monkeypatch.setattr("pm.paths.repo_root", lambda: project)
    monkeypatch.setattr("pm.environments.install_state_dir", lambda _project: tmp_path / "state")
    def toolchain(**kwargs):
        toolchain_calls.append(kwargs)
        return None

    monkeypatch.setattr("pm._uv._toolchain", toolchain)
    monkeypatch.setattr("pm.lock.Lockfile", Lockfile)
    monkeypatch.setattr("pm.paths.lockfile_path", lambda: tmp_path / "uv.lock")
    monkeypatch.setattr("pm.paths.store_root", lambda: tmp_path / "store")
    monkeypatch.setattr("pm.store.current_target", lambda: "test-target")
    monkeypatch.setattr("pm.registry.get_package", lambda name: Package(name))

    assert runtime.runtime_python() == managed_python
    assert captured["uv"] == managed_uv
    assert captured["python"] == managed_python
    assert toolchain_calls == [{"realize": False}]


def test_runtime_bootstrap_fails_once_without_managed_python(tmp_path, monkeypatch):
    from pm import runtime

    project = tmp_path / "project"
    project.mkdir()
    calls = []

    monkeypatch.setattr(runtime, "is_runtime", lambda: False)
    monkeypatch.setattr(runtime, "_resident_runtime", lambda: None)
    monkeypatch.setattr("pm.paths.repo_root", lambda: project)
    monkeypatch.setattr("pm.environments.install_state_dir", lambda _project: tmp_path / "state")

    def toolchain(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr("pm._uv._toolchain", toolchain)
    monkeypatch.setattr("pm.lock.Lockfile", lambda _path: type(
        "Lockfile", (), {"version": lambda _self, _name: "3.14"})())
    monkeypatch.setattr("pm.paths.lockfile_path", lambda: tmp_path / "uv.lock")
    monkeypatch.setattr("pm.paths.store_root", lambda: tmp_path / "store")
    monkeypatch.setattr("pm.store.current_target", lambda: "test-target")
    monkeypatch.setattr("pm.registry.get_package", lambda _name: type(
        "Package", (), {
            "store_entry": lambda _self, version, target: f"{version}-{target}",
            "binary": lambda _self, _root, _target: None,
        })())

    with pytest.raises(runtime.InstallError, match="managed Python.*hermes pm install"):
        runtime.runtime_python()
    assert calls == [{"realize": False}]


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


@pytest.mark.platforms("macos", "windows")
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
        # The guard resolves symlinks (a python linked outside the payload IS an escape), so the
        # interpreter is a real copy inside the payload. A relocatable/framework build locates its
        # stdlib beside the executable: supply the host's library tree the way the bundle test does.
        (base / "bin").mkdir(parents=True)
        python = base / "bin" / "python"
        shutil.copy2(Path(sys._base_executable).resolve(), python)
        (base / "lib").symlink_to(Path(sys.base_prefix) / "lib", target_is_directory=True)
    (runtime / "pm-runtime.json").write_text(json.dumps({
        "python": os.path.relpath(python, runtime), "sitePackages": "site",
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


def test_one_store_through_symlinked_homes_keeps_one_pm_runtime(tmp_path, monkeypatch):
    """#123798: a store reached through symlinks (a per-task HERMES_HOME whose tools/ links back,
    a home under /home -> /var/home) is ONE PM runtime: every spelling reuses the selected
    generation, and a record written before canonicalization is not re-staged."""
    from pm import runtime

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    (project / "uv.lock").write_text("version = 1\n")
    real = tmp_path / "volume" / "tools"
    (real / "python" / "bin").mkdir(parents=True)
    (real / "python" / "bin" / "python3.14").touch()
    home, task = tmp_path / "home", tmp_path / "task"
    try:
        (real / "python" / "bin" / "python3").symlink_to("python3.14")  # as the pinned CPython ships
        for linked in (home, task):
            linked.mkdir()
            (linked / "tools").symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    def pre_fix_identity(python: Path) -> str:  # the selected.json format earlier releases wrote
        digest = hashlib.sha256()
        for name in ("pyproject.toml", "uv.lock"):
            digest.update((project / name).read_bytes() + b"\0")
        digest.update(str(python).encode())
        return digest.hexdigest()

    staged = []

    def stage(uv, python, environment, **_):
        staged.append(python)
        environment.mkdir(parents=True)
        return runtime._python(environment)

    monkeypatch.setattr("pm.runtime_stage.stage_runtime", stage)
    monkeypatch.setattr(runtime, "_validate", lambda python, env: "")
    monkeypatch.setattr(runtime, "_hold_for_children", lambda environment: None)
    root = tmp_path / "pm-runtime"
    old = root / "generations" / "old"
    old.mkdir(parents=True)
    (old / "pm-runtime.json").write_text("{}")
    (root / "selected.json").write_text(json.dumps(
        {"inputs": pre_fix_identity(home / "tools" / "python" / "bin" / "python3"), "generation": "generations/old"}))

    def launch(tools: Path, bootstrap: bool = False) -> Path:
        return runtime.prepare_runtime(tools / "uv", tools / "python" / "bin" / "python3", root,
                                       project=project, bootstrap=bootstrap)

    assert launch(home / "tools") == runtime._python(old)
    current = launch(task / "tools", bootstrap=True)  # a per-task home that was already churning re-stages once
    for tools in (home / "tools", task / "tools", real, home / "tools"):
        assert launch(tools) == current
    assert len(staged) == 1
