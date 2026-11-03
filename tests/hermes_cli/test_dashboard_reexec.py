"""Test the dashboard process handoff with mocked process APIs."""

import sys
from unittest.mock import Mock

import pytest

from hermes_cli import main_dashboard


@pytest.mark.platforms("windows")
def test_windows_dashboard_reexec_keeps_child_exit_status(monkeypatch):
    argv = [sys.executable, "-m", "hermes_cli.main", "dashboard"]
    env = {"HERMES_HOME": "test-machine-root"}
    child = Mock()
    child.wait.return_value = 17
    popen = Mock(return_value=child)
    execvpe = Mock(side_effect=AssertionError("Windows must use Popen"))
    monkeypatch.setattr(main_dashboard.subprocess, "Popen", popen)
    monkeypatch.setattr(main_dashboard.os, "execvpe", execvpe)

    with pytest.raises(SystemExit) as exc:
        main_dashboard._reexec_dashboard(argv, env)

    assert exc.value.code == 17
    popen.assert_called_once_with(argv, env=env)
    child.wait.assert_called_once_with()
    execvpe.assert_not_called()


@pytest.mark.platforms("posix")
def test_posix_dashboard_reexec_keeps_argv_and_env(monkeypatch):
    argv = [sys.executable, "-m", "hermes_cli.main", "dashboard"]
    env = {"HERMES_HOME": "test-machine-root"}
    execvpe = Mock(side_effect=SystemExit(23))
    popen = Mock(side_effect=AssertionError("POSIX must use execvpe"))
    monkeypatch.setattr(main_dashboard.os, "execvpe", execvpe)
    monkeypatch.setattr(main_dashboard.subprocess, "Popen", popen)

    with pytest.raises(SystemExit) as exc:
        main_dashboard._reexec_dashboard(argv, env)

    assert exc.value.code == 23
    execvpe.assert_called_once_with(sys.executable, argv, env)
    assert execvpe.call_args.args[1] is argv
    assert execvpe.call_args.args[2] is env
    popen.assert_not_called()
