"""Pyright uses the project environment before the Hermes runtime fallback."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pm
from pm import paths
from pm.lock import Facts, Lockfile
from pm.registry import get_package
from pm.store import current_target

from agent.lsp.servers import _detect_python


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


def test_pyright_respects_pm_writable_selection_over_old_bundle(tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    bundled = _seed_pm_python(tmp_path, monkeypatch)
    bundled_facts = Facts(paths.facts_path())
    old = bundled_facts.get("python")
    assert old is not None
    store = paths.writable_store_root()
    package = get_package("python")
    target = current_target()
    entry = store / old["entry"]
    binary = package.binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="utf-8")
    Facts(store / "facts.json").record(
        "python", old["version"], entry.name, package.env(entry, target), store,
        target=target, artifacts=old["artifacts"],
    )
    bundled_facts.record(
        "python", "old-bundle", old["entry"], {}, paths.store_root(),
        target=target, artifacts=old["artifacts"],
    )
    before = paths.facts_path().read_bytes(), (store / "facts.json").read_bytes()
    selected = pm.installed_package("python")
    assert selected is not None and selected.binary == binary
    assert _detect_python(str(tmp_path / "workspace")) == str(binary)
    assert (paths.facts_path().read_bytes(), (store / "facts.json").read_bytes()) == before
    assert bundled.is_file()


def test_pm_store_python_resolved_without_virtual_env(tmp_path, monkeypatch):
    """No VIRTUAL_ENV anywhere — the pm store interpreter is the answer."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    exe = _seed_pm_python(tmp_path, monkeypatch)

    assert _detect_python(str(tmp_path / "some-workspace")) == str(exe)


def test_no_pm_python_falls_back_to_project_layouts(tmp_path, monkeypatch):
    """Without a pm python fact, project .venv candidates still resolve."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "no-such-store"))

    dot_venv_bin = tmp_path / "proj" / ".venv" / "bin"
    dot_venv_bin.mkdir(parents=True)
    python = dot_venv_bin / "python"
    python.write_text("", encoding="utf-8")

    assert Path(_detect_python(str(tmp_path / "proj"))) == python


def test_missing_store_entry_is_not_an_answer(tmp_path, monkeypatch):
    """A python fact whose store entry is gone resolves to None (pm only
    vouches for what is on disk)."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    payload = tmp_path / "payload"
    store = payload / "tools"
    store.mkdir(parents=True)
    (payload / "manifest.json").write_text("{}", encoding="utf-8")
    (store / "facts.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "packages": {"python": {"entry": "python-3.11.13", "version": "3.11.13"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))

    assert _detect_python(str(tmp_path / "some-workspace")) is None


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
