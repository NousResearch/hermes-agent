"""Regression for #120169: updater probes must not strand venv holders."""

import subprocess
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli import update_cmd_deps, update_cmd_windows


def test_pip_probe_and_bootstrap_are_bounded_and_fail_closed(tmp_path):
    pip = [str(tmp_path / "venv" / "Scripts" / "pip.exe")]
    python = str(tmp_path / "venv" / "Scripts" / "python.exe")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[-1] == "--version":
            return subprocess.CompletedProcess(argv, 1, "", "pip missing")
        return None

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        update_cmd_deps, "bounded_probe_run", side_effect=run
    ):
        with pytest.raises(subprocess.TimeoutExpired):
            update_cmd_deps._ensure_venv_pip(pip, python)
    assert [argv for argv, _ in calls] == [
        pip + ["--version"], [python, "-m", "ensurepip", "--upgrade", "--default-pip"]
    ]
    assert all(kwargs["timeout"] > 0 and kwargs["raise_on_spawn_failure"] for _, kwargs in calls)

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        update_cmd_deps, "bounded_probe_run", return_value=None
    ) as bounded:
        with pytest.raises(subprocess.TimeoutExpired):
            update_cmd_deps._ensure_venv_pip(pip, python)
    bounded.assert_called_once()


def test_refusal_identifies_only_exact_pip_probe_from_this_venv(tmp_path):
    pip = tmp_path / "venv" / "Scripts" / "pip.exe"
    python = tmp_path / "venv" / "Scripts" / "python.exe"
    own = f'"{pip}" --version'
    module = f'"{python}" -m pip --version'
    bootstrap = f'"{python}" -m ensurepip --upgrade --default-pip'
    foreign = f'"{tmp_path / "other" / "Scripts" / "pip.exe"}" --version'
    foreign_module = f'"{tmp_path / "other" / "Scripts" / "python.exe"}" -m pip --version'
    extra = module + " --other"
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path):
        message = update_cmd_windows._format_venv_python_holders_message([
            (100, "pip.exe", own), (101, "python.exe", module),
            (102, "python.exe", bootstrap), (103, "pip.exe", foreign),
            (104, "python.exe", foreign_module), (105, "python.exe", extra),
        ])
    lines = message.splitlines()
    for pid in (100, 101, 102):
        assert "leftover updater pip probe" in next(line for line in lines if f"PID {pid}" in line)
    for pid in (103, 104, 105):
        assert "leftover updater pip probe" not in next(line for line in lines if f"PID {pid}" in line)
