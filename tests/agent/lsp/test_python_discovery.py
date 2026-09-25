"""Pyright uses the project environment before the Hermes runtime fallback."""

import os
import subprocess
import sys
from pathlib import Path

from pm import paths
from pm.lock import Facts, Lockfile
from pm.registry import get_package
from pm.store import current_target


def _seed_pm_python(tmp_path, monkeypatch):
    """Stage a pm bundled-install layout: HERMES_RUNTIME_DIR -> store with a
    manifest sibling (bundled), a python entry, and facts recording it."""
    payload = tmp_path / "payload"
    store = payload / "tools"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    lock = Lockfile(paths.lockfile_path())
    package = get_package("python")
    target = current_target()
    version = lock.version("python")
    assert version is not None
    python_entry = store / package.store_entry(version, target)
    python_entry.mkdir(parents=True)
    exe = package.binary(python_entry, target)
    assert exe is not None
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("", encoding="utf-8")
    (payload / "manifest.json").write_text("{}", encoding="utf-8")
    Facts(paths.facts_path()).record(
        "python", version, python_entry.name, package.env(python_entry, target), store,
        target=target, artifacts=[a["sha256"] for a in lock.artifacts("python", target)],
    )
    return exe


def test_pyright_uses_the_project_interpreter_before_hermes(tmp_path, monkeypatch):
    from agent.lsp import servers

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    pm_python = _seed_pm_python(tmp_path, monkeypatch)
    context = servers.ServerContext(
        workspace_root=str(project), install_strategy="off",
        binary_overrides={"pyright": [sys.executable]},
    )
    server = servers.find_server_for_file(str(project / "app.py"))
    assert server is not None
    spec = server.build_spawn(str(project), context)
    assert spec is not None
    assert spec.initialization_options["python"]["pythonPath"] == str(pm_python)

    pm_python.unlink()
    spec = server.build_spawn(str(project), context)
    assert spec is not None
    assert "pythonPath" not in spec.initialization_options.get("python", {})

    for environment in (project / ".venv", tmp_path / "explicit-environment"):
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(environment)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if environment.name == "explicit-environment":
            monkeypatch.setenv("VIRTUAL_ENV", str(environment))
        spec = server.build_spawn(str(project), context)
        assert spec is not None
        selected = spec.initialization_options["python"]["pythonPath"]
        assert Path(selected) == python
        child = subprocess.run(
            [selected, "-I", "-c", "import sys; print(sys.prefix)"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        assert Path(child.stdout.strip()) == environment
