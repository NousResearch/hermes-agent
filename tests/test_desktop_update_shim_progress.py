"""The shim's /progress contract, exercised against the real posix hand-off.

`ui.html` is one page for three operating systems, so the stage-and-elapsed
line it renders is only as good as the weakest orchestrator behind it. These
drive the real `posix.sh` and the real `serve-ui.py` -- no mocks, no source
reading -- because the posix half is the one that had no coverage.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIM_DIR = REPO_ROOT / "scripts" / "desktop-update"

# posix.sh re-execs itself through these to detach from Electron's process
# group; without them the hand-off never reaches its own main flow. The
# interpreter is resolved, not hardcoded, so any python3 on PATH satisfies it.
requires_posix_handoff = pytest.mark.skipif(
    not (
        os.path.exists("/bin/bash")
        and (shutil.which("python3") or os.path.exists("/usr/bin/python3"))
    ),
    reason="posix.sh detaches through /bin/bash and a python3 interpreter",
)


# ── serve-ui.py: what the page actually receives ───────────────────────────


@pytest.fixture
def progress(tmp_path):
    """The real loopback server, over a status file the test drives."""
    status = tmp_path / "hermes-update-status"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(SHIM_DIR / "serve-ui.py"),
            str(SHIM_DIR / "ui.html"),
            str(status),
            str(time.time()),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        port = int(proc.stdout.readline().strip())

        class Progress:
            def publish(self, state: str, message: str) -> None:
                status.write_text(json.dumps({"status": state, "message": message}))

            def corrupt(self) -> None:
                status.write_text("}not json{")

            def poll(self) -> dict:
                with urlopen(f"http://127.0.0.1:{port}/progress", timeout=5) as r:
                    return json.loads(r.read())

        yield Progress()
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_elapsed_advances_between_publishes(progress):
    """The clock is stamped per request, not frozen at the last publish.

    Stages are minutes apart during a real update, so a value written into
    the status file would sit still through exactly the long waits this line
    exists to disprove.
    """
    progress.publish("running", "Updating code and dependencies")
    first = progress.poll()
    time.sleep(1.2)
    second = progress.poll()

    assert first["status"] == "running"
    assert second["message"] == first["message"] == "Updating code and dependencies"
    assert second["elapsed_seconds"] > first["elapsed_seconds"]


def test_every_state_carries_a_clock(progress):
    """Including the stage-less running state posix publishes at window open."""
    for state, message in [
        ("running", ""),
        ("running", "Installing the new app"),
        ("done", ""),
        ("manual", "Reopen Hermes to finish."),
        ("error", "Update failed."),
    ]:
        progress.publish(state, message)
        served = progress.poll()

        assert served["status"] == state
        assert served["message"] == message
        assert served["elapsed_seconds"] >= 0


def test_unreadable_status_still_serves_a_running_state(progress):
    """A window frozen on its last frame is the bug being fixed here."""
    progress.corrupt()
    served = progress.poll()

    assert served["status"] == "running"
    assert served["elapsed_seconds"] >= 0


# ── posix.sh: the stages it publishes at its own gates ─────────────────────

# Stands in for `hermes update`, and reports the stage that was on screen
# while it ran -- the update child is the only thing that can observe the
# window's state at the exact moment of the longest wait in the hand-off.
FAKE_HERMES = """#!/bin/bash
# The hand-off probes `update --help` for --keep-stash support before the
# real update call; answer it without consuming a counted call so the
# exits.N mapping below still refers to actual update attempts.
case "$*" in *--help*) echo "--keep-stash"; exit 0 ;; esac
n="$(cat "$HERMES_TEST_CALLS" 2>/dev/null || echo 0)"; n=$((n + 1))
printf '%s' "$n" > "$HERMES_TEST_CALLS"
for f in "$TMPDIR"/hermes-update-status.[0-9]*; do
  case "$f" in *.tmp) continue ;; esac
  cp "$f" "$HERMES_TEST_CAPTURE.$n" 2>/dev/null
done
exit "$(cat "$HERMES_TEST_EXITS.$n" 2>/dev/null || echo 0)"
"""


def _run_handoff(tmp_path, exits: dict[int, int]) -> list[dict]:
    """Run the real hand-off end to end; return the stage seen at each call."""
    install_root = tmp_path / "hermes-agent"
    (install_root / "venv" / "bin").mkdir(parents=True)
    hermes = install_root / "venv" / "bin" / "hermes"
    hermes.write_text(FAKE_HERMES)
    hermes.chmod(0o755)

    capture = tmp_path / "seen"
    calls = tmp_path / "calls"
    for call, code in exits.items():
        (tmp_path / f"exits.{call}").write_text(str(code))

    env = {
        **os.environ,
        "TMPDIR": str(tmp_path),
        "HERMES_TEST_CAPTURE": str(capture),
        "HERMES_TEST_CALLS": str(calls),
        "HERMES_TEST_EXITS": str(tmp_path / "exits"),
    }
    # The hand-off daemonizes and the launcher exits immediately; the result
    # file is the orchestrator's own completion signal.
    subprocess.run(
        [
            "/bin/bash",
            str(SHIM_DIR / "posix.sh"),
            "--install-root",
            str(install_root),
            "--no-ui",
        ],
        env=env,
        timeout=60,
        check=True,
    )

    result = tmp_path / ".hermes-update-result.json"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and not result.exists():
        time.sleep(0.1)
    assert result.exists(), "hand-off never wrote its result file"

    seen = sorted(tmp_path.glob("seen.*"), key=lambda p: int(p.suffix[1:]))

    return [json.loads(p.read_text()) for p in seen]


@requires_posix_handoff
def test_update_gate_publishes_its_stage_before_running(tmp_path):
    """`hermes update` is the longest wait in the hand-off and the one the
    Discord report sat through; the window must name it while it happens."""
    stages = _run_handoff(tmp_path, {1: 0})

    assert len(stages) == 1
    assert stages[0]["status"] == "running"
    assert stages[0]["message"] == "Updating code and dependencies"


@requires_posix_handoff
def test_retry_gate_publishes_a_distinct_stage(tmp_path):
    """A failed first attempt doubles the wait -- the second pass must not
    look like the first one hanging."""
    stages = _run_handoff(tmp_path, {1: 1, 2: 0})

    assert [s["message"] for s in stages] == [
        "Updating code and dependencies",
        "Retrying update",
    ]
