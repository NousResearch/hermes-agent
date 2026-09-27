"""Managed-runtime interpreter resolution for cron children (#123400, #123440).

Under the PM runtime the gateway runs the bare store Python with the committed
environment's site-packages activated in-process. Children spawned with
``sys.executable`` see neither tree: the subprocess sanitizer strips PYTHONPATH,
and the store ships no third-party packages — the cron external worker died
importing ``ruamel`` (#123400) and cron ``.py`` scripts died importing Hermes
modules (#123440). The committed environment's own interpreter carries its
site-packages via ``pyvenv.cfg``.

The resolution is a pure function over the platform (``windows=`` data), so the
host-independent parts test anywhere; the POSIX argv branch is asserted on POSIX.
"""
import os
import sys

import pytest


def _commit_generation(tmp_path, monkeypatch):
    """A PM-committed generation whose interpreter exists (a stub file)."""
    from pm.environments import install_state_dir, runtime_facts_path, venv_bin_dir

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    venv = install_state_dir(repo) / "environments" / "gen1" / "venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("version = 3.14\n", encoding="utf-8")
    python = venv_bin_dir(venv, windows=False) / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime_facts_path(repo).write_text(
        __import__("json").dumps({"packages": {"venv": {"environment": str(venv)}}}), encoding="utf-8")
    return repo, python


def test_resolves_committed_environments_interpreter(tmp_path, monkeypatch):
    from cron.scheduler_worker_env import managed_runtime_python

    repo, python = _commit_generation(tmp_path, monkeypatch)
    assert managed_runtime_python(repo, windows=False) == python


def test_windows_data_returns_none(tmp_path, monkeypatch):
    from cron.scheduler_worker_env import managed_runtime_python

    repo, _python = _commit_generation(tmp_path, monkeypatch)
    assert managed_runtime_python(repo, windows=True) is None


def test_no_committed_environment_falls_back_to_none(tmp_path, monkeypatch):
    from cron.scheduler_worker_env import managed_runtime_python

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    assert managed_runtime_python(repo, windows=False) is None


def test_missing_interpreter_falls_back_to_none(tmp_path, monkeypatch):
    from cron.scheduler_worker_env import managed_runtime_python

    repo, python = _commit_generation(tmp_path, monkeypatch)
    python.unlink()
    assert managed_runtime_python(repo, windows=False) is None


def test_unreadable_record_falls_back_to_none(tmp_path, monkeypatch):
    from cron.scheduler_worker_env import managed_runtime_python

    repo, _python = _commit_generation(tmp_path, monkeypatch)
    from pm.environments import runtime_facts_path

    runtime_facts_path(repo).write_text("not json", encoding="utf-8")
    assert managed_runtime_python(repo, windows=False) is None


def test_running_interpreter_returns_none(tmp_path, monkeypatch):
    """A generation whose interpreter IS this process keeps the launch contract."""
    from cron.scheduler_worker_env import managed_runtime_python

    repo, python = _commit_generation(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "executable", str(python))
    assert managed_runtime_python(repo, windows=False) is None


@pytest.mark.platforms("posix")
def test_posix_script_argv_uses_managed_interpreter(tmp_path, monkeypatch):
    """#123440: on a POSIX managed install, a .py script runs on the committed
    environment's interpreter with the repo pinned on PYTHONPATH."""
    import cron.scheduler_worker_env as worker_env
    from cron.scheduler_script import _script_argv

    script = tmp_path / "scripts" / "probe.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n")
    stub = tmp_path / "venv" / "bin" / "python"
    stub.parent.mkdir(parents=True)
    stub.write_text("#!/bin/sh\n", encoding="utf-8")

    repo = tmp_path / "repo"
    monkeypatch.setattr(worker_env, "managed_runtime_python", lambda repo_root, **kw: stub)

    argv, overlay, err = _script_argv(script)
    assert err is None
    assert argv == [str(stub), str(script)]
    assert overlay == {"PYTHONPATH": str(repo)}


@pytest.mark.platforms("posix")
def test_posix_script_argv_unmanaged_keeps_sys_executable(tmp_path, monkeypatch):
    """Without a committed environment, a POSIX script keeps ``sys.executable``."""
    import cron.scheduler_worker_env as worker_env
    from cron.scheduler_script import _script_argv

    script = tmp_path / "scripts" / "probe.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n")
    monkeypatch.setattr(worker_env, "managed_runtime_python", lambda repo_root, **kw: None)

    argv, overlay, err = _script_argv(script)
    assert err is None
    assert argv == [sys.executable, str(script)]
    assert overlay == {}
