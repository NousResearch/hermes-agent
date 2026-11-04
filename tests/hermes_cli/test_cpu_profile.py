from __future__ import annotations

import os
import pstats
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(args: list[str], hermes_home: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_")}
    env.update({"HERMES_HOME": str(hermes_home), "PYTHONPATH": str(REPO)})
    return subprocess.run(
        [sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=120, check=False)


def test_relative_cpu_profile_path_lands_in_the_profile_logs(tmp_path: Path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()

    result = _run(["-m", "hermes_cli.main", "--cpu-profile", "run.prof", "completion", "bash"], hermes_home)

    assert result.returncode == 0, result.stderr[-1000:]
    target = hermes_home / "logs" / "profiles" / "run.prof"
    assert f"CPU profiling enabled: {target.resolve()}" in result.stderr
    assert pstats.Stats(str(target)).total_calls > 0
    assert "cumulative" in (target.parent / "run.prof.txt").read_text(encoding="utf-8")


def test_oneshot_hard_exit_still_writes_the_profile(tmp_path: Path):
    """``-z`` leaves through ``os._exit`` (atexit skipped); the profile must still be written."""
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    target = tmp_path / "oneshot.prof"
    code = (
        "from hermes_cli.cpu_profile import enable_cpu_profile\n"
        f"enable_cpu_profile({str(target)!r})\n"
        "from hermes_cli.main import _exit_after_oneshot\n"
        "_exit_after_oneshot(0)\n"
    )

    result = _run(["-c", code], hermes_home)

    assert result.returncode == 0, result.stderr[-1000:]
    assert pstats.Stats(str(target)).total_calls > 0


def test_gateway_graceful_hard_exit_still_writes_the_profile(tmp_path: Path):
    """Every gateway exit funnels through ``os._exit`` (atexit skipped); the profile must still be written."""
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    target = tmp_path / "gateway.prof"
    code = (
        "from hermes_cli.cpu_profile import enable_cpu_profile\n"
        f"enable_cpu_profile({str(target)!r})\n"
        "from gateway.run import _exit_after_graceful_shutdown\n"
        "_exit_after_graceful_shutdown(0)\n"
    )

    result = _run(["-c", code], hermes_home)

    assert result.returncode == 0, result.stderr[-1000:]
    assert pstats.Stats(str(target)).total_calls > 0
