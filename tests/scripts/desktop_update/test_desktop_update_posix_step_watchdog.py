"""The posix hand-off bounds `hermes update` by the absence of progress, never by wall time.

A step whose whole tree is silent on its pipe and in update.log, starts no process and burns no
CPU for the idle bound is stopped and reported (#138672: a pipe deadlock parked the updater at 0%
CPU forever). A slow step that keeps moving is never stopped, and an update that printed its
completion banner is never re-reported as failed by a late stall (the #96205 rule windows.ps1
already applies). Every case drives the real `posix.sh` against a fake `hermes`.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SHIM = REPO_ROOT / "scripts" / "desktop-update" / "posix.sh"
IDLE_SECONDS = 3

pytestmark = [
    pytest.mark.platforms("posix"),
    pytest.mark.skipif(
        not (os.path.exists("/bin/bash") and os.path.exists("/usr/bin/python3")),
        reason="posix.sh detaches through /bin/bash and /usr/bin/python3",
    ),
]

# Stands in for `hermes`: answers the --help probe, then runs HERMES_TEST_UPDATE_BODY as the update.
FAKE_HERMES = """#!/usr/bin/env bash
case "$*" in *--help*) echo "--keep-stash"; exit 0 ;; esac
eval "$HERMES_TEST_UPDATE_BODY"
"""


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # windows-footgun: ok -- posix-only module (platforms("posix"))
    except ProcessLookupError:
        return False
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], check=False, capture_output=True, text=True, encoding="utf-8").stdout
    return bool(stat.strip()) and not stat.strip().startswith("Z")


def _run_handoff(tmp_path: Path, body: str) -> dict:
    install_root = tmp_path / "hermes-agent"
    (install_root / "venv" / "bin").mkdir(parents=True)
    hermes = install_root / "venv" / "bin" / "hermes"
    hermes.write_text(FAKE_HERMES, encoding="utf-8")
    hermes.chmod(0o755)
    (tmp_path / "logs").mkdir()
    env = {**os.environ, "TMPDIR": str(tmp_path), "HERMES_HOME": str(tmp_path),
           "HERMES_TEST_UPDATE_BODY": body, "HERMES_TEST_DIR": str(tmp_path)}
    subprocess.run(
        ["/bin/bash", str(SHIM), "--install-root", str(install_root), "--no-ui",
         "--self-test-step-idle", str(IDLE_SECONDS)],
        env=env, timeout=60, check=True,
    )
    result = tmp_path / ".hermes-update-result.json"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and not result.exists():
        time.sleep(0.2)
    assert result.exists(), (tmp_path / "logs" / "desktop-update-handoff.log").read_text(encoding="utf-8-sig")
    return json.loads(result.read_text(encoding="utf-8-sig"))


def test_a_stuck_update_is_stopped_with_its_whole_tree_and_reported(tmp_path):
    """The #138672 shape: the updater and its child wait on each other at 0% CPU, forever."""
    body = 'sleep 1000 & echo "$!" > "$HERMES_TEST_DIR/grandchild.pid"; echo "$$" > "$HERMES_TEST_DIR/child.pid"; ' \
           'echo "Fetching updates..."; wait'
    started = time.monotonic()
    result = _run_handoff(tmp_path, body)

    assert result["ok"] is False
    assert result["exit_code"] == 124
    assert "made no progress" in result["message"]
    assert time.monotonic() - started < 45
    for name in ("child.pid", "grandchild.pid"):
        pid = int((tmp_path / name).read_text(encoding="utf-8-sig"))
        assert not _process_alive(pid), f"{name} {pid} survived the stall"


def test_a_descendant_that_ignores_term_is_killed_after_the_root_dies(tmp_path):
    """TERM ends the update's root, but a child that ignores TERM is still stopped by the watchdog's KILL."""
    body = '( trap "" TERM; exec sleep 1000 ) & echo "$!" > "$HERMES_TEST_DIR/stubborn.pid"; ' \
           'echo "Fetching updates..."; wait'
    result = _run_handoff(tmp_path, body)

    assert result["exit_code"] == 124
    pid = int((tmp_path / "stubborn.pid").read_text(encoding="utf-8-sig"))
    assert not _process_alive(pid), f"TERM-ignoring child {pid} survived the stall"


def test_a_slow_update_that_keeps_moving_is_never_stopped(tmp_path):
    """Three times the idle bound in total, but never idle for it: a slow link, not a stall."""
    body = 'for i in 1 2 3 4 5 6 7 8 9; do echo "receiving objects $i"; sleep 1; done; echo "✓ Update complete!"'
    result = _run_handoff(tmp_path, body)

    assert result["exit_code"] == 0, result
    assert result["ok"] is True


def test_update_log_growth_alone_counts_as_progress(tmp_path):
    """The update tees its long build into update.log, not into the hand-off's pipe."""
    body = 'for i in 1 2 3 4 5 6 7 8 9; do echo "vite chunk $i" >> "$HERMES_TEST_DIR/logs/update.log"; sleep 1; done'
    result = _run_handoff(tmp_path, body)

    assert result["exit_code"] == 0, result


def test_a_finished_update_is_never_reported_failed_by_a_late_stall(tmp_path):
    """#96205 on posix: the banner printed, then the post-update tail parked past the bound."""
    result = _run_handoff(tmp_path, 'echo "✓ Update complete!"; sleep 1000')

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert "post-update steps" in result["message"]


def test_a_failure_after_the_banner_keeps_the_stall_a_failure(tmp_path):
    """The veto never covers a run whose restart/verify phase reported a failure."""
    result = _run_handoff(tmp_path, 'echo "✓ Update complete!"; echo "✗ Update not complete"; sleep 1000')

    assert result["ok"] is False
    assert result["exit_code"] == 124
