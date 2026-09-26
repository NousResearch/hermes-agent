"""The hand-off's `--gateway` flag, exercised against the real posix script.

`hermes update --gateway` (re)starts the local messaging gateway after the
update. A Desktop served by a remote gateway (#117529) must not ask for that:
the restarted local gateway shares the remote host's channel credentials and
becomes a competing long-poll consumer — Telegram answers the conflict by
rejecting one of the two `getUpdates` callers, taking the production bot
offline. The Desktop therefore passes `--no-gateway` whenever its active
connection is remote-shaped; these tests pin both sides of that contract
against the real `posix.sh`, so neither the default (local ownership) nor the
opt-out can silently regress.
"""

from __future__ import annotations

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

# Stands in for `hermes`: answers the `update --help` probe (so --keep-stash
# is kept), and appends every non-help invocation's argv as one JSON line so
# the tests can inspect exactly what the update was invoked with.
FAKE_HERMES = """#!/usr/bin/env bash
case "$*" in *--help*) echo "--keep-stash"; exit 0 ;; esac
printf '%s\\n' "$*" >> "$HERMES_TEST_ARGV"
exit 0
"""

def _run_handoff(tmp_path: Path, extra_args: list[str]) -> list[str]:
    """Run the real hand-off end to end; return the argv of each hermes call."""
    install_root = tmp_path / "hermes-agent"
    (install_root / "venv" / "bin").mkdir(parents=True)
    hermes = install_root / "venv" / "bin" / "hermes"
    hermes.write_text(FAKE_HERMES)
    hermes.chmod(0o755)

    argv_log = tmp_path / "argv.jsonl"
    # posix.sh honours an ambient HERMES_HOME; the result file must land where this test looks.
    env = {**os.environ, "TMPDIR": str(tmp_path), "HERMES_TEST_ARGV": str(argv_log), "HERMES_HOME": str(tmp_path)}
    subprocess.run(
        ["/bin/bash", str(SHIM_DIR / "posix.sh"), "--install-root", str(install_root), "--no-ui", *extra_args],
        env=env,
        timeout=60,
        check=True,
    )

    result = tmp_path / ".hermes-update-result.json"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and not result.exists():
        time.sleep(0.1)
    assert result.exists(), "hand-off never wrote its result file"

    return argv_log.read_text().splitlines()

@requires_posix_handoff
def test_default_handoff_asks_for_the_local_gateway(tmp_path):
    """A locally-served Desktop owns its gateway: the update must restart it."""
    calls = _run_handoff(tmp_path, [])

    update_calls = [c for c in calls if " update " in f" {c} "]
    assert update_calls, "hand-off never ran hermes update"
    assert "--gateway" in update_calls[0].split()

@requires_posix_handoff
def test_no_gateway_flag_omits_gateway_from_update(tmp_path):
    """A remote-served Desktop (#117529): no local gateway may be (re)started.

    The competing long-poll consumer is silent channel outage risk, and the
    flag must survive the hand-off's own retry path — every update invocation
    is checked, not just the first.
    """
    calls = _run_handoff(tmp_path, ["--no-gateway"])

    update_calls = [c for c in calls if " update " in f" {c} "]
    assert update_calls, "hand-off never ran hermes update"
    for call in update_calls:
        argv = call.split()
        assert "--gateway" not in argv, f"--gateway reappeared in update argv: {call}"
        assert "--keep-stash" in argv, "--no-gateway must not disturb --keep-stash"
        assert "--yes" in argv
