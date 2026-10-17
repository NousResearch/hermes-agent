"""Unit tests for docker/entrypoint.sh (#107263).

The deprecated docker/entrypoint.sh shim used to run stage2 bootstrap and
then stop, so a hard-coded ``ENTRYPOINT ["docker/entrypoint.sh"]`` (old
wrapper scripts, NAS app-store packages that carried the override across
an upgrade) bootstrapped the container and then exited without ever
running the CMD. These tests run without Docker: the shim's
HERMES_ENTRYPOINT_SHIM_STAGE2 / HERMES_ENTRYPOINT_SHIM_WRAPPER hooks let
us record what would be invoked, using fake scripts in place of the real
stage2-hook.sh and main-wrapper.sh.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM = REPO_ROOT / "docker" / "entrypoint.sh"


@pytest.fixture
def recorder(tmp_path: Path) -> tuple[Path, Path]:
    """Fake stage2-hook.sh + main-wrapper.sh that log their own invocation."""
    stage2 = tmp_path / "fake-stage2.sh"
    wrapper = tmp_path / "fake-wrapper.sh"
    log = tmp_path / "calls.log"
    stage2.write_text(f"#!/bin/sh\necho stage2 >> {log}\necho \"$PATH\" > {log}.path\n")
    wrapper.write_text(f"#!/bin/sh\necho \"wrapper $*\" >> {log}\n")
    stage2.chmod(0o755)
    wrapper.chmod(0o755)
    return stage2, wrapper


def _run_shim(
    recorder: tuple[Path, Path],
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    stage2, wrapper = recorder
    env = os.environ.copy()
    env["HERMES_ENTRYPOINT_SHIM_STAGE2"] = str(stage2)
    env["HERMES_ENTRYPOINT_SHIM_WRAPPER"] = str(wrapper)
    return subprocess.run(
        ["sh", str(SHIM), *args],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
        check=False,
    )


def _log(recorder: tuple[Path, Path]) -> str:
    log = recorder[0].parent / "calls.log"
    return log.read_text() if log.exists() else ""


def test_runs_stage2_then_execs_cmd(recorder: tuple[Path, Path]) -> None:
    """A hard-coded entrypoint.sh must still run the CMD after bootstrap."""
    r = _run_shim(recorder, ["gateway", "run"])
    assert r.returncode == 0, r.stderr
    log = _log(recorder)
    lines = [ln for ln in log.splitlines() if ln]
    assert lines == ["stage2", "wrapper gateway run"]


def test_runs_stage2_before_wrapper(recorder: tuple[Path, Path]) -> None:
    """Bootstrap must complete before the requested command starts."""
    r = _run_shim(recorder, ["--help"])
    assert r.returncode == 0, r.stderr
    log = _log(recorder)
    assert log.index("stage2") < log.index("wrapper --help")


def test_warns_about_deprecation(recorder: tuple[Path, Path]) -> None:
    """Callers still on this path must see the migration notice."""
    r = _run_shim(recorder, [])
    assert r.returncode == 0, r.stderr
    assert "deprecated shim" in r.stderr
    assert "entrypoint-dispatch.sh" in r.stderr


def test_no_args_still_execs_wrapper(recorder: tuple[Path, Path]) -> None:
    """`ENTRYPOINT ["docker/entrypoint.sh"]` with no CMD must still run."""
    r = _run_shim(recorder, [])
    assert r.returncode == 0, r.stderr
    log = _log(recorder)
    lines = [ln for ln in log.splitlines() if ln]
    assert lines == ["stage2", "wrapper "]


def test_stage2_can_find_s6_helpers(recorder: tuple[Path, Path]) -> None:
    """stage2 calls s6-setuidgid bare; /init would have put it on PATH."""
    r = _run_shim(recorder, [])
    assert r.returncode == 0, r.stderr
    path = (recorder[0].parent / "calls.log.path").read_text().strip().split(":")
    assert "/command" in path
    assert os.environ["PATH"].split(":")[0] in path


def test_failed_bootstrap_never_runs_cmd(recorder: tuple[Path, Path]) -> None:
    """A failing stage2 bootstrap must not silently start the CMD anyway."""
    recorder[0].write_text("#!/bin/sh\nexit 1\n")
    r = _run_shim(recorder, [])
    assert r.returncode != 0
    assert _log(recorder) == ""
