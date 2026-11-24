"""A launcher can name the folder a process keeps its install state in (``HERMES_INSTALL_STATE_DIR``).

It does so when it starts Hermes on an interpreter it prepared, somewhere the install's own state cannot be
used: a sandbox that may read the selected environment and nothing else of ``installs/<key>``. There the state
folder is still seen, cannot be read or locked, and start-up ends on it.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Every OS lane collects this file: the folder is named for a sandbox on Windows.
pytestmark = pytest.mark.platforms("any")


def _site_of(venv: Path) -> Path:
    return venv / ("Lib/site-packages" if os.name == "nt"
                   else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")


def _committed_environment(state: Path, name: str, version: str) -> Path:
    """A state folder that records a committed environment whose ``probe_package`` says *version*."""
    venv = state / "environments" / name / "venv"
    _site_of(venv).mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = test", encoding="utf-8")
    (_site_of(venv) / "probe_package.py").write_text(f"version = {version!r}", encoding="utf-8")
    (state / "facts.json").write_text(json.dumps({"schema": 1, "packages": {"venv": {"environment": str(venv)}}}),
                                      encoding="utf-8")
    return venv


def _entries(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))


def test_a_process_with_a_named_state_folder_starts_on_the_interpreter_it_was_given(tmp_path, monkeypatch):
    """It selects no environment, from the install's own state or from the named folder, and writes nothing into
    the install's own state: its lock is taken in the named folder."""
    from pm import environments

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_INSTALL_STATE_DIR", raising=False)
    own = environments.install_state_dir(root)
    _committed_environment(own, "committed", "the install's")
    named = tmp_path / "named"
    _committed_environment(named, "placed", "placed in the named folder")
    before = _entries(own)

    code = ("import importlib.util, sys; from pathlib import Path; from pm.environments import activate_dependencies; "
            "activate_dependencies(Path(sys.argv[1])); print(importlib.util.find_spec('probe_package') is None)")
    process = subprocess.run([sys.executable, "-c", code, str(root)], text=True, capture_output=True, timeout=60,
                             env={**os.environ, "HERMES_INSTALL_STATE_DIR": str(named)}, check=False)

    assert process.returncode == 0, process.stderr
    assert process.stdout.splitlines() == ["True"]
    assert _entries(own) == before
    assert (named / "locks").is_dir()


def test_a_process_with_a_named_state_folder_does_not_repair_the_install(tmp_path, monkeypatch):
    """The install's own state records an environment that is gone, which a process of that install repairs at
    start. A process given another state folder leaves the repair, and the install's state, alone."""
    from hermes_cli import _early_recovery
    from pm import environments, recovery

    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_INSTALL_STATE_DIR", raising=False)
    own = environments.install_state_dir(root)
    own.mkdir(parents=True)
    (own / "facts.json").write_text(json.dumps({"schema": 1, "packages": {
        "venv": {"environment": str(own / "environments" / "gone" / "venv")}}}), encoding="utf-8")
    before = _entries(own)
    repaired = []
    monkeypatch.setattr(recovery, "repair_dependencies", repaired.append)
    named = tmp_path / "named"
    named.mkdir()
    monkeypatch.setenv("HERMES_INSTALL_STATE_DIR", str(named))

    assert _early_recovery.recover_if_needed(root, ["chat"]) is False
    assert repaired == []
    assert _entries(own) == before


def test_a_process_with_a_named_state_folder_is_not_restarted_and_completes_no_source_update(tmp_path, monkeypatch):
    """A self-managed install whose dependencies are out of date is synced at start and the process restarted on
    the install's interpreter. A process given another state folder keeps the interpreter it was started with."""
    import hermes_constants
    import pm
    import pm.client
    from hermes_cli import _launchers, update_custody, venv_sync

    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: tmp_path / ".hermes")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    root = tmp_path / ".hermes" / "hermes-agent"
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")
    (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": "self", "commit": "abc"}), encoding="utf-8")
    synced, spawned = [], []
    monkeypatch.setattr(pm.client, "ensure_tools_for_sync", lambda: None)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: bool(synced))
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: synced.append(a))
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: Path(sys.executable))
    monkeypatch.setattr(venv_sync, "publish_launchers", lambda r, **kw: None)
    monkeypatch.setattr(update_custody, "run", lambda command, **kw: spawned.append(command)
                        or subprocess.CompletedProcess(command, 0))
    named = tmp_path / "named"
    named.mkdir()
    monkeypatch.setenv("HERMES_INSTALL_STATE_DIR", str(named))

    assert venv_sync.prepare_launch(root, []) is None
    assert synced == [] and spawned == []
    assert list(named.iterdir()) == []


def test_a_kanban_worker_keeps_the_named_state_folder(tmp_path):
    """The worker client is started from an allowlisted environment, inside the same sandbox as its parent."""
    from hermes_cli.kanban_worker_env import worker_client_env

    env = {"PATH": "/bin", "HERMES_INSTALL_STATE_DIR": str(tmp_path / "named")}
    assert worker_client_env(env, tmp_path / "assignee") == env
