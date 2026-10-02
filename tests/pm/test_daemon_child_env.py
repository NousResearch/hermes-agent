"""Daemon-child interpreter + environment lane (#129239 bug 2).

Bug 1 (the PM selection leaking onto os.environ["PYTHONPATH"] in
activate_dependencies) belongs to the strip tracks
(#128910/#128965/#122954) and is untouched here. These tests pin the DAEMON
lane only: daemon_python resolves the venv interpreter a daemon child
must spawn with, and daemon_child_env drops PM-generation site-packages
entries built for a different interpreter minor, with UTF-8 mode applied as
setdefault. Platform layouts ride the windows argument as data (per the
cross-platform contract), so no host marker is needed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

import hermes_constants
from pm import environments as env_mod


@pytest.fixture(autouse=True)
def _fresh_hermes_root_memo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)


def _write_pyvenv_cfg(venv: Path, version: str) -> Path:
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text(f"home = fixture\nversion = {version}\n", encoding="utf-8")
    return venv


def _pm_generation(state_dir: Path, name: str, version: str, *, windows: bool = False) -> Path:
    """Fake PM generation venv: <state_dir>/environments/<name>/venv."""
    venv = state_dir / "environments" / name / "venv"
    _write_pyvenv_cfg(venv, version)
    if windows:
        (venv / "Lib" / "site-packages").mkdir(parents=True)
    else:
        major, _, rest = version.partition(".")
        minor, _, _ = rest.partition(".")
        (venv / f"lib/python{major}.{minor}" / "site-packages").mkdir(parents=True)
    return venv


def _commit(root: Path, venv: Path) -> None:
    facts = env_mod.runtime_facts_path(root)
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text(
        json.dumps({"packages": {"venv": {"environment": str(venv)}}}), encoding="utf-8")


def test_daemon_child_env_drops_mismatched_pm_site_packages(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    gen314 = _pm_generation(tmp_path / "state", "gen314", "3.14.7")
    entry314 = str(env_mod.site_packages(gen314))
    child = tmp_path / "child311" / "bin" / "python3.11"
    child.parent.mkdir(parents=True)  # the exe name alone carries the minor
    env = {"PYTHONPATH": os.pathsep.join([str(root), entry314, "/user/libs"])}
    out = env_mod.daemon_child_env(env, python_exe=str(child))
    assert out["PYTHONPATH"] == os.pathsep.join([str(root), "/user/libs"])
    assert out["PYTHONUTF8"] == "1"
    assert out["PYTHONIOENCODING"] == "utf-8"
    assert entry314 in env["PYTHONPATH"]  # input mapping is never mutated


def test_daemon_child_env_keeps_matching_minor(tmp_path: Path):
    gen314 = _pm_generation(tmp_path / "state", "gen314", "3.14.7")
    entry314 = str(env_mod.site_packages(gen314))
    child = tmp_path / "child314" / "bin" / "python3.14"
    child.parent.mkdir(parents=True)
    before = os.pathsep.join(["/proj", entry314])
    out = env_mod.daemon_child_env({"PYTHONPATH": before}, python_exe=str(child))
    assert out["PYTHONPATH"] == before


def test_daemon_child_env_fail_open_when_child_minor_unknown(tmp_path: Path):
    gen314 = _pm_generation(tmp_path / "state", "gen314", "3.14.7")
    entry314 = str(env_mod.site_packages(gen314))
    child = tmp_path / "child" / "bin" / "python"  # no version, no pyvenv.cfg nearby
    child.parent.mkdir(parents=True)
    before = os.pathsep.join(["/proj", entry314])
    out = env_mod.daemon_child_env({"PYTHONPATH": before}, python_exe=str(child))
    assert out["PYTHONPATH"] == before


def test_daemon_child_env_windows_layout(tmp_path: Path):
    gen = _pm_generation(tmp_path / "state", "genA", "3.14.7", windows=True)
    entry = str(gen / "Lib" / "site-packages")
    child_venv = _write_pyvenv_cfg(tmp_path / "child311", "3.11.16")
    child_exe = child_venv / "Scripts" / "python.exe"  # versionless name: minor comes from pyvenv.cfg
    env = {"PYTHONPATH": ";".join([r"C:\proj", entry, r"D:\user\libs"])}
    out = env_mod.daemon_child_env(env, python_exe=str(child_exe), windows=True)
    assert out["PYTHONPATH"] == ";".join([r"C:\proj", r"D:\user\libs"])
    assert out["PYTHONUTF8"] == "1"
    assert out["PYTHONIOENCODING"] == "utf-8"


def test_daemon_child_env_keeps_foreign_text_it_cannot_date(tmp_path: Path):
    # A Windows-spelled entry on a POSIX host has no readable pyvenv.cfg, so
    # its minor is unknown and the entry is kept (fail-open).
    entry = r"C:\h\installs\i\environments\g\venv\Lib\site-packages"
    child = tmp_path / "child311" / "bin" / "python3.11"
    child.parent.mkdir(parents=True)
    out = env_mod.daemon_child_env({"PYTHONPATH": entry}, python_exe=str(child))
    assert out["PYTHONPATH"] == entry


def test_daemon_child_env_utf8_defaults_never_override_user(tmp_path: Path):
    out = env_mod.daemon_child_env({}, python_exe=sys.executable)
    assert out == {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    out = env_mod.daemon_child_env(
        {"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp950"}, python_exe=sys.executable)
    assert out == {"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp950"}


def test_daemon_python_prefers_committed_venv_over_caller_interpreter(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    venv = _pm_generation(env_mod.runtime_facts_path(root).parent, "gen311", "3.11.16")
    interpreter = venv / "bin" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.touch()
    _commit(root, venv)
    resolved = env_mod.daemon_python(root)
    assert resolved == interpreter
    assert str(resolved) != sys.executable


def test_daemon_python_missing_interpreter_raises_honest_error(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    venv = _pm_generation(env_mod.runtime_facts_path(root).parent, "gen311", "3.11.16")
    _commit(root, venv)  # committed, but the interpreter file itself is absent
    with pytest.raises(RuntimeError) as excinfo:
        env_mod.daemon_python(root)
    message = str(excinfo.value)
    assert "daemon interpreter is missing" in message
    assert str(venv / "bin" / "python") in message
    assert "hermes update" in message
    assert "#129239" in message


def test_daemon_python_without_committed_env_raises(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()  # no facts.json: the in-tree base venv has no interpreter either
    with pytest.raises(RuntimeError, match="daemon interpreter is missing"):
        env_mod.daemon_python(root)


def test_daemon_python_windows_layout(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    venv = _pm_generation(
        env_mod.runtime_facts_path(root).parent, "gen311", "3.11.16", windows=True)
    interpreter = venv / "Scripts" / "python.exe"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.touch()
    _commit(root, venv)
    assert env_mod.daemon_python(root, windows=True) == interpreter
