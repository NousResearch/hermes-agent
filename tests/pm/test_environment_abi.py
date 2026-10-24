"""``activate_dependencies`` must never graft a generation built for another Python.

A source update can commit a dependency environment with a different CPython than the
interpreter that later boots the install (the app's launcher is one such path). The graft
itself "works" — ``sys.path`` points at the tree — but every compiled extension is
unimportable, and the call sites tolerate the missing imports: tool search is skipped,
MCP servers park, Group Chat dies. The failure has no single traceback to chase, so the
guard fails at activation with the remedy instead.
"""

from __future__ import annotations

import sys

import pytest

from pm.environments import ensure_environment_matches_interpreter, venv_python_version


def _venv(tmp_path, *, cfg_version: str | None, lib_dir: str | None = None):
    venv = tmp_path / "venv"
    (venv / "lib").mkdir(parents=True)
    if lib_dir:
        (venv / "lib" / lib_dir / "site-packages").mkdir(parents=True)
    if cfg_version is not None:
        (venv / "pyvenv.cfg").write_text(
            f"home = /store/python\nversion = {cfg_version}\n", encoding="utf-8")
    return venv


def test_venv_python_version_reads_pyvenv_cfg(tmp_path):
    assert venv_python_version(_venv(tmp_path, cfg_version="3.11.9")) == (3, 11)


def test_venv_python_version_falls_back_to_the_lib_dir(tmp_path):
    assert venv_python_version(_venv(tmp_path, cfg_version=None, lib_dir="python3.14")) == (3, 14)


def test_matching_interpreter_is_allowed(tmp_path):
    ensure_environment_matches_interpreter(_venv(tmp_path, cfg_version="3.11.9"), running=(3, 11))


def test_foreign_interpreter_is_refused_with_the_remedy(tmp_path):
    with pytest.raises(RuntimeError, match=r"refusing to mix ABIs.*hermes pm repair"):
        ensure_environment_matches_interpreter(_venv(tmp_path, cfg_version="3.14.7"), running=(3, 11))


def test_unknown_version_keeps_todays_behaviour(tmp_path):
    """Trees predating pyvenv.cfg version pinning must not start failing at activation."""
    ensure_environment_matches_interpreter(_venv(tmp_path, cfg_version=None), running=(3, 11))


def test_default_running_version_is_this_interpreter(tmp_path):
    venv = _venv(tmp_path, cfg_version=f"{sys.version_info.major}.{sys.version_info.minor}.0")
    ensure_environment_matches_interpreter(venv)
