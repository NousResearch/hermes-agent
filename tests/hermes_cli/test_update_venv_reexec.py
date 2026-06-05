import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main


def _make_venv(root: Path, name: str = "venv") -> tuple[Path, Path]:
    venv = root / name
    scripts = venv / ("Scripts" if sys.platform == "win32" else "bin")
    scripts.mkdir(parents=True)
    python = scripts / ("python.exe" if sys.platform == "win32" else "python")
    python.write_text("", encoding="utf-8")
    return venv, python


def test_select_update_venv_prefers_dotvenv(tmp_path):
    dotvenv, dotvenv_python = _make_venv(tmp_path, ".venv")
    _make_venv(tmp_path, "venv")

    selected = cli_main._select_update_venv(tmp_path)

    assert selected == (dotvenv, dotvenv_python)


def test_select_update_venv_uses_venv_when_dotvenv_missing(tmp_path):
    venv, python = _make_venv(tmp_path, "venv")

    selected = cli_main._select_update_venv(tmp_path)

    assert selected == (venv, python)


def test_update_target_venv_follows_dotvenv_preference(tmp_path):
    dotvenv, python = _make_venv(tmp_path, ".venv")
    _make_venv(tmp_path, "venv")

    with patch.object(sys, "executable", str(python)):
        assert cli_main._update_target_venv_dir(tmp_path) == dotvenv


def test_venv_scripts_dir_uses_selected_dotvenv_on_windows(tmp_path):
    dotvenv = tmp_path / ".venv"
    scripts = dotvenv / "Scripts"
    scripts.mkdir(parents=True)
    python = scripts / "python.exe"
    python.write_text("", encoding="utf-8")
    _make_venv(tmp_path, "venv")

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        cli_main, "_is_windows", return_value=True
    ), patch.object(sys, "executable", str(python)):
        assert cli_main._venv_scripts_dir() == scripts


def test_select_update_venv_returns_none_without_usable_python(tmp_path):
    (tmp_path / "venv" / "bin").mkdir(parents=True)

    assert cli_main._select_update_venv(tmp_path) is None


def test_candidate_venvs_skip_hermes_home_fallback_for_non_git_project(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    fallback_venv, _fallback_python = _make_venv(hermes_home / "hermes-agent", "venv")
    project_root = tmp_path / "site-packages" / "hermes-agent"
    project_root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with patch.object(cli_main, "PROJECT_ROOT", project_root):
        candidates = cli_main._candidate_update_venvs(project_root)

    assert fallback_venv not in candidates


def test_python_belongs_to_expected_venv(tmp_path):
    venv, python = _make_venv(tmp_path, "venv")
    other_python = tmp_path / "outside" / "python"
    other_python.parent.mkdir()
    other_python.write_text("", encoding="utf-8")

    assert cli_main._python_belongs_to_venv(python, venv)
    assert not cli_main._python_belongs_to_venv(other_python, venv)


def test_python_belongs_to_venv_does_not_follow_posix_symlink_out(tmp_path):
    venv = tmp_path / "venv"
    scripts = venv / "bin"
    scripts.mkdir(parents=True)
    real_python = tmp_path / "system" / "python"
    real_python.parent.mkdir()
    real_python.write_text("", encoding="utf-8")
    symlinked_python = scripts / "python"
    try:
        symlinked_python.symlink_to(real_python)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not available on this filesystem")

    assert symlinked_python.resolve() == real_python
    assert cli_main._python_belongs_to_venv(symlinked_python, venv)


def test_maybe_reexec_skips_when_already_in_expected_venv(tmp_path):
    venv, python = _make_venv(tmp_path, "venv")

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        sys, "executable", str(python)
    ), patch.object(os, "execvpe") as execvpe:
        assert cli_main._maybe_reexec_update_in_managed_venv(SimpleNamespace(check=False)) is False

    execvpe.assert_not_called()


def test_venv_python_can_import_hermes_cli_uses_safe_env(tmp_path, monkeypatch):
    _venv, python = _make_venv(tmp_path, "venv")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "shadow"))

    with patch.object(
        cli_main.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0),
    ) as run:
        assert cli_main._venv_python_can_import_hermes_cli(python) is True

    assert run.call_args.args[0] == [
        str(python),
        "-P",
        "-c",
        "import hermes_cli.main",
    ]
    assert "PYTHONPATH" not in run.call_args.kwargs["env"]


def test_venv_python_can_import_hermes_cli_returns_false_on_probe_error(tmp_path):
    _venv, python = _make_venv(tmp_path, "venv")

    with patch.object(cli_main.subprocess, "run", side_effect=OSError("nope")):
        assert cli_main._venv_python_can_import_hermes_cli(python) is False


def test_maybe_reexec_uses_expected_venv_python(tmp_path, monkeypatch, capsys):
    venv, python = _make_venv(tmp_path, "venv")
    system_python = tmp_path / "system" / "python"
    system_python.parent.mkdir()
    system_python.write_text("", encoding="utf-8")
    monkeypatch.delenv(cli_main._UPDATE_VENV_REEXEC_ENV, raising=False)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "shadow"))

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        sys, "executable", str(system_python)
    ), patch.object(sys, "argv", ["hermes", "update", "--yes", "--branch", "main"]), patch.object(
        cli_main, "_venv_python_can_import_hermes_cli", return_value=True
    ), patch.object(os, "execvpe") as execvpe:
        execvpe.side_effect = RuntimeError("stop exec")
        with pytest.raises(RuntimeError, match="stop exec"):
            cli_main._maybe_reexec_update_in_managed_venv(SimpleNamespace(check=False))

    exe, cmd, env = execvpe.call_args.args
    assert exe == str(python)
    assert cmd == [
        str(python),
        "-P",
        "-m",
        "hermes_cli.main",
        "update",
        "--yes",
        "--branch",
        "main",
    ]
    assert env[cli_main._UPDATE_VENV_REEXEC_ENV] == "1"
    assert env["VIRTUAL_ENV"] == str(venv)
    assert "PYTHONPATH" not in env
    scripts_name = "Scripts" if cli_main._is_windows() else "bin"
    assert env["PATH"].split(os.pathsep)[0] == str(venv / scripts_name)
    assert str(venv) in capsys.readouterr().out


def test_maybe_reexec_skips_venv_that_cannot_import_hermes_cli(
    tmp_path, monkeypatch, capsys
):
    _venv, _python = _make_venv(tmp_path, "venv")
    system_python = tmp_path / "system" / "python"
    system_python.parent.mkdir()
    system_python.write_text("", encoding="utf-8")
    monkeypatch.delenv(cli_main._UPDATE_VENV_REEXEC_ENV, raising=False)

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        sys, "executable", str(system_python)
    ), patch.object(
        cli_main, "_venv_python_can_import_hermes_cli", return_value=False
    ) as can_import, patch.object(os, "execvpe") as execvpe:
        assert cli_main._maybe_reexec_update_in_managed_venv(SimpleNamespace(check=False)) is False

    can_import.assert_called_once()
    execvpe.assert_not_called()
    assert "cannot import hermes_cli" in capsys.readouterr().out


def test_maybe_reexec_on_windows_waits_for_child_and_exits(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    python = scripts / "python.exe"
    python.write_text("", encoding="utf-8")
    system_python = tmp_path / "system" / "python.exe"
    system_python.parent.mkdir()
    system_python.write_text("", encoding="utf-8")
    monkeypatch.delenv(cli_main._UPDATE_VENV_REEXEC_ENV, raising=False)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "shadow"))

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        cli_main, "_is_windows", return_value=True
    ), patch.object(sys, "executable", str(system_python)), patch.object(
        sys, "argv", ["hermes", "update", "--yes"]
    ), patch.object(
        cli_main, "_venv_python_can_import_hermes_cli", return_value=True
    ), patch.object(
        cli_main.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 7),
    ) as run, patch.object(os, "execvpe") as execvpe:
        with pytest.raises(SystemExit) as excinfo:
            cli_main._maybe_reexec_update_in_managed_venv(SimpleNamespace(check=False))

    assert excinfo.value.code == 7
    execvpe.assert_not_called()
    cmd = run.call_args.args[0]
    env = run.call_args.kwargs["env"]
    assert cmd == [str(python), "-P", "-m", "hermes_cli.main", "update", "--yes"]
    assert env["VIRTUAL_ENV"] == str(venv)
    assert "PYTHONPATH" not in env
    assert env["PATH"].split(os.pathsep)[0] == str(scripts)


def test_maybe_reexec_on_windows_reports_launch_failure(tmp_path, monkeypatch, capsys):
    venv = tmp_path / "venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    python = scripts / "python.exe"
    python.write_text("", encoding="utf-8")
    system_python = tmp_path / "system" / "python.exe"
    system_python.parent.mkdir()
    system_python.write_text("", encoding="utf-8")
    monkeypatch.delenv(cli_main._UPDATE_VENV_REEXEC_ENV, raising=False)

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        cli_main, "_is_windows", return_value=True
    ), patch.object(sys, "executable", str(system_python)), patch.object(
        sys, "argv", ["hermes", "update"]
    ), patch.object(
        cli_main, "_venv_python_can_import_hermes_cli", return_value=True
    ), patch.object(
        cli_main.subprocess, "run", side_effect=OSError("boom")
    ), patch.object(os, "execvpe") as execvpe:
        with pytest.raises(SystemExit) as excinfo:
            cli_main._maybe_reexec_update_in_managed_venv(SimpleNamespace(check=False))

    assert excinfo.value.code == 1
    execvpe.assert_not_called()
    assert "Failed to re-run update" in capsys.readouterr().out


def test_maybe_reexec_guard_fails_instead_of_looping(tmp_path, monkeypatch, capsys):
    _venv, _python = _make_venv(tmp_path, "venv")
    system_python = tmp_path / "system" / "python"
    system_python.parent.mkdir()
    system_python.write_text("", encoding="utf-8")
    monkeypatch.setenv(cli_main._UPDATE_VENV_REEXEC_ENV, "1")

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        sys, "executable", str(system_python)
    ), patch.object(os, "execvpe") as execvpe:
        with pytest.raises(SystemExit) as excinfo:
            cli_main._maybe_reexec_update_in_managed_venv(SimpleNamespace(check=False))

    assert excinfo.value.code == 1
    execvpe.assert_not_called()
    assert "still not running inside" in capsys.readouterr().out


def test_maybe_reexec_skips_update_check(tmp_path):
    _make_venv(tmp_path, "venv")

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(os, "execvpe") as execvpe:
        assert cli_main._maybe_reexec_update_in_managed_venv(SimpleNamespace(check=True)) is False

    execvpe.assert_not_called()
