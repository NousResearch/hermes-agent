from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch


class _FetchProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        timeout: bool = False,
    ) -> None:
        self.pid = 4321
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._timeout = timeout

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        if self._timeout:
            raise subprocess.TimeoutExpired(["git", "fetch"], timeout=timeout)
        return self._stdout, self._stderr


def test_update_check_fetch_is_bounded_and_sweeps_timeout_artifacts(tmp_path: Path) -> None:
    from hermes_cli import update_cmd

    proc = _FetchProcess(timeout=True)
    with (
        patch.object(update_cmd.subprocess, "Popen", return_value=proc) as popen,
        patch.object(
            update_cmd, "_terminate_update_check_fetch", return_value=True
        ) as terminate,
        patch("hermes_cli.gitlock.clear_stale_tmp_packs") as clear,
    ):
        result = update_cmd._run_update_check_fetch(
            ["git"], [], "origin", "main", tmp_path
        )

    assert result.returncode != 0
    assert "timed out" in result.stderr.lower()
    if sys.platform == "win32":
        assert popen.call_args.kwargs["creationflags"] & getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        assert popen.call_args.kwargs["start_new_session"] is True
    terminate.assert_called_once_with(proc)
    clear.assert_called_once_with(tmp_path, min_age_seconds=0)


def test_update_check_fetch_does_not_sweep_if_tree_cannot_be_reaped(tmp_path: Path) -> None:
    from hermes_cli import update_cmd

    proc = _FetchProcess(timeout=True)
    with (
        patch.object(update_cmd.subprocess, "Popen", return_value=proc),
        patch.object(update_cmd, "_terminate_update_check_fetch", return_value=False),
        patch("hermes_cli.gitlock.clear_stale_tmp_packs") as clear,
    ):
        result = update_cmd._run_update_check_fetch(
            ["git"], [], "origin", "main", tmp_path
        )

    assert result.returncode == 124
    clear.assert_not_called()


def test_update_check_fetch_sweeps_artifacts_after_git_failure(tmp_path: Path) -> None:
    from hermes_cli import update_cmd

    failed = _FetchProcess(returncode=128, stderr="fatal: transfer aborted")
    with (
        patch.object(update_cmd.subprocess, "Popen", return_value=failed),
        patch("hermes_cli.gitlock.clear_stale_tmp_packs") as clear,
    ):
        result = update_cmd._run_update_check_fetch(
            ["git"], ["--depth", "1"], "origin", "main", tmp_path
        )

    assert result.returncode == 128
    assert result.stderr == "fatal: transfer aborted"
    clear.assert_called_once_with(tmp_path, min_age_seconds=0)


def test_update_check_fetch_keeps_success_path_cleanup_free(tmp_path: Path) -> None:
    from hermes_cli import update_cmd

    succeeded = _FetchProcess()
    with (
        patch.object(update_cmd.subprocess, "Popen", return_value=succeeded),
        patch("hermes_cli.gitlock.clear_stale_tmp_packs") as clear,
    ):
        result = update_cmd._run_update_check_fetch(
            ["git"], [], "origin", "main", tmp_path
        )

    assert result.returncode == 0
    clear.assert_not_called()


def test_update_check_fetch_timeout_stops_spawned_child_before_sweep(
    tmp_path: Path,
) -> None:
    from hermes_cli import update_cmd

    script = tmp_path / "fetch_with_child.py"
    child_pid_path = tmp_path / "child.pid"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "pathlib.Path('child.pid').write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    def assert_child_stopped(*_args, **_kwargs) -> list[str]:
        import psutil

        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        try:
            child = psutil.Process(child_pid)
        except psutil.NoSuchProcess:
            return []
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        return []

    with (
        patch.object(update_cmd, "UPDATE_CHECK_FETCH_TIMEOUT_SECONDS", 0.5),
        patch(
            "hermes_cli.gitlock.clear_stale_tmp_packs",
            side_effect=assert_child_stopped,
        ),
    ):
        result = update_cmd._run_update_check_fetch(
            [sys.executable, str(script)], [], "origin", "main", tmp_path
        )

    assert result.returncode == 124
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            import psutil

            child = psutil.Process(child_pid)
            if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"fetch transport child {child_pid} survived timeout")


def test_update_check_fetch_reports_effective_timeout(tmp_path: Path) -> None:
    from hermes_cli import update_cmd

    proc = _FetchProcess(timeout=True)
    with (
        patch.object(update_cmd.subprocess, "Popen", return_value=proc),
        patch.object(update_cmd, "_terminate_update_check_fetch", return_value=False),
    ):
        result = update_cmd._run_update_check_fetch(
            ["git"], [], "origin", "main", tmp_path, timeout_seconds=10
        )

    assert result.stderr == "git fetch timed out after 10 seconds"
