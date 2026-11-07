from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path


def test_live_runtime_paths_reports_managed_interpreter_and_dependency_generation(
    tmp_path: Path, monkeypatch,
) -> None:
    from pm.environments import live_runtime_paths

    interpreter = tmp_path / "tools/python-3.14/bin/python3.14"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    ordinary = tmp_path / "tools/python-3.14/lib/python3.14/site-packages"
    dependency = (
        tmp_path
        / "installs/install-key/environments/generation/venv/lib/python3.14/site-packages"
    )
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setattr(sys, "path", [str(ordinary), str(dependency)])

    assert live_runtime_paths() == (str(interpreter.resolve()), str(dependency.resolve()))


def test_live_runtime_report_uses_stable_audit_line(monkeypatch, capsys) -> None:
    from pm import environments

    monkeypatch.setattr(
        environments,
        "live_runtime_paths",
        lambda: ("/managed/python3.14", "/installs/generation/site-packages"),
    )

    environments.report_live_runtime()

    assert capsys.readouterr().err == (
        "hermes: live interpreter=/managed/python3.14 "
        "deps=/installs/generation/site-packages\n"
    )


def test_pid_runtime_paths_reads_proc_exe_and_pm_dependency_generation(tmp_path: Path) -> None:
    from pm.environments import pid_runtime_paths

    pid = 4321
    proc = tmp_path / str(pid)
    proc.mkdir()
    interpreter = tmp_path / "tools/python-3.14/bin/python3.14"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    (proc / "exe").symlink_to(interpreter)
    dependency = (
        tmp_path
        / "installs/install-key/environments/generation/venv/lib/python3.14/site-packages"
    )
    mapped_file = dependency / "pydantic_core/_pydantic_core.so"
    (proc / "maps").write_text(
        f"7f000000-7f001000 r--p 00000000 00:00 0 {mapped_file}\n",
        encoding="utf-8",
    )

    assert pid_runtime_paths(pid, proc_root=tmp_path) == (
        str(interpreter.resolve()),
        str(dependency),
    )


def test_pid_runtime_paths_rejects_unrelated_site_packages(tmp_path: Path) -> None:
    from pm.environments import pid_runtime_paths

    pid = 4321
    proc = tmp_path / str(pid)
    proc.mkdir()
    interpreter = tmp_path / "usr/bin/python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    (proc / "exe").symlink_to(interpreter)
    (proc / "maps").write_text(
        "/tmp/lib/python3.14/site-packages/example/native.so\n",
        encoding="utf-8",
    )

    assert pid_runtime_paths(pid, proc_root=tmp_path) == (str(interpreter.resolve()), None)


def test_cmd_env_pid_prints_only_runtime_paths(monkeypatch, capsys) -> None:
    from pm import cli

    monkeypatch.setattr(
        "pm.environments.pid_runtime_paths",
        lambda pid: ("/managed/python3.14", "/installs/generation/site-packages"),
    )

    assert cli.cmd_env(Namespace(names=[], pid=99)) == 0
    assert json.loads(capsys.readouterr().out) == {
        "pid": 99,
        "interpreter": "/managed/python3.14",
        "deps": "/installs/generation/site-packages",
    }


def test_cmd_env_pid_rejects_non_hermes_process(monkeypatch, capsys) -> None:
    from pm import cli

    monkeypatch.setattr(
        "pm.environments.pid_runtime_paths",
        lambda pid: ("/usr/bin/python3", None),
    )

    assert cli.cmd_env(Namespace(names=[], pid=99)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "PID 99" in captured.err
    assert "Hermes PM dependency generation" in captured.err


def test_cmd_env_rejects_pid_with_package_names(capsys) -> None:
    from pm import cli

    assert cli.cmd_env(Namespace(names=["python"], pid=99)) == 2
    assert "cannot be combined" in capsys.readouterr().err
