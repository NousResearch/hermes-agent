"""The POSIX hand-off streams `hermes update` output live (#130460).

The orchestrator used to capture the whole update run in a ``$(...)``
substitution and append it to ``desktop-update-handoff.log`` only after the
child exited — so the log, and the update window watching it, stayed silent
for the entire (minutes-long) run while the update was actually progressing.
These drive the real ``posix.sh`` with a fake ``hermes`` binary (same shape
as ``test_desktop_update_shim_progress.py``) and prove output lands in the
hand-off log *while the child is still running*, and that the retained copy
still drives the parked-branch skip detection after the refactor.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SHIM_DIR = REPO_ROOT / "scripts" / "desktop-update"

requires_posix_handoff = pytest.mark.skipif(
    not (os.path.exists("/bin/bash") and os.path.exists("/usr/bin/python3")),
    reason="posix.sh detaches through /bin/bash and /usr/bin/python3",
)

# Emits one progress line, then parks until the test creates $HERMES_TEST_GATE
# — the only way to observe the hand-off log mid-run. Answers the
# --keep-stash probe the same way the shim-progress fake does.
FAKE_STREAMING_HERMES = """#!/usr/bin/env bash
case "$*" in *--help*) echo "--keep-stash"; exit 0 ;; esac
echo "FAKE-HERMES-LIVE-PROGRESS-LINE"
for _ in $(seq 1 300); do
  [ -f "$HERMES_TEST_GATE" ] && break
  sleep 0.1
done
echo "FAKE-HERMES-DONE"
exit 0
"""

FAKE_SKIPPED_HERMES = """#!/usr/bin/env bash
case "$*" in *--help*) echo "--keep-stash"; exit 0 ;; esac
echo "CODE UPDATE SKIPPED: parked on a feature branch"
exit 1
"""

FAKE_UNTERMINATED_HERMES = """#!/usr/bin/env bash
case "$*" in *--help*) echo "--keep-stash"; exit 0 ;; esac
printf '%s' "PROGRESS 42% (no newline)"
exit 0
"""


def _install_fake_hermes(tmp_path: Path, body: str) -> Path:
    install_root = tmp_path / "hermes-agent"
    (install_root / "venv" / "bin").mkdir(parents=True)
    hermes = install_root / "venv" / "bin" / "hermes"
    hermes.write_text(body)
    hermes.chmod(0o755)
    return install_root


def _launch_handoff(tmp_path: Path, install_root: Path, extra_env: dict[str, str]) -> None:
    env = {**os.environ, "TMPDIR": str(tmp_path), **extra_env}
    env.pop("HERMES_HOME", None)  # Exercise the legacy install-parent fallback.
    # The launcher daemonizes and exits immediately; the orchestrator signals
    # completion through the result file.
    subprocess.run(
        ["/bin/bash", str(SHIM_DIR / "posix.sh"), "--install-root", str(install_root), "--no-ui"],
        env=env,
        timeout=60,
        check=True,
    )


def _wait_for(path: Path, needle: str, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if needle in text:
            return text
        time.sleep(0.1)
    raise AssertionError(f"{needle!r} never appeared in {path} within {timeout_s}s")


@requires_posix_handoff
def test_update_output_streams_while_child_runs(tmp_path):
    """The progress line must be in the hand-off log BEFORE the child exits.

    The test only creates the gate file after observing the line, so a
    buffering hand-off (log written after exit) can never satisfy this: the
    child would still be parked waiting for the gate.
    """
    install_root = _install_fake_hermes(tmp_path, FAKE_STREAMING_HERMES)
    gate = tmp_path / "gate"
    _launch_handoff(tmp_path, install_root, {"HERMES_TEST_GATE": str(gate)})

    handoff_log = tmp_path / "logs" / "desktop-update-handoff.log"
    _wait_for(handoff_log, "FAKE-HERMES-LIVE-PROGRESS-LINE", timeout_s=30)

    gate.touch()
    result_path = tmp_path / ".hermes-update-result.json"
    _wait_for(result_path, '"ok":true', timeout_s=45)

    text = handoff_log.read_text(encoding="utf-8", errors="replace")
    assert "hermes update exit code: 0" in text
    assert "FAKE-HERMES-DONE" in text
    # The per-run capture file is an implementation detail, never litter.
    assert list(tmp_path.glob("hermes-update-out-*")) == []


@requires_posix_handoff
def test_retained_output_still_drives_skip_detection(tmp_path):
    """Streaming must keep the post-run copy the skip/failure greps read."""
    install_root = _install_fake_hermes(tmp_path, FAKE_SKIPPED_HERMES)
    _launch_handoff(tmp_path, install_root, {})

    result_path = tmp_path / ".hermes-update-result.json"
    raw = _wait_for(result_path, '"exit_code":8', timeout_s=45)
    assert json.loads(raw)["ok"] is False

    handoff_log = tmp_path / "logs" / "desktop-update-handoff.log"
    assert "CODE UPDATE SKIPPED" in handoff_log.read_text(encoding="utf-8", errors="replace")


@requires_posix_handoff
def test_unterminated_child_output_preserves_log_line_discipline(tmp_path):
    """Child output lacking a trailing newline must not swallow the next log timestamp."""
    install_root = _install_fake_hermes(tmp_path, FAKE_UNTERMINATED_HERMES)
    _launch_handoff(tmp_path, install_root, {})

    result_path = tmp_path / ".hermes-update-result.json"
    _wait_for(result_path, '"ok":true', timeout_s=45)

    handoff_log = tmp_path / "logs" / "desktop-update-handoff.log"
    content = handoff_log.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    assert "PROGRESS 42% (no newline)" in lines
    exit_lines = [line for line in lines if "hermes update exit code: 0" in line]
    assert len(exit_lines) == 1
    # Exit line starts with ISO-8601 timestamp, not appended to the partial output
    assert not any("PROGRESS 42% (no newline)" in line and "hermes update exit code" in line for line in lines)

