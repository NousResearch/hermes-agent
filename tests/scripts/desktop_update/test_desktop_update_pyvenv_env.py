"""The Desktop update hand-off must not pass a poisoned interpreter env to the orchestrator.

The hand-off daemonizes by re-exec'ing through ``/usr/bin/nohup /usr/bin/python3``. macOS's
``/usr/bin/python3`` SETS ``__PYVENV_LAUNCHER__`` to its own path, and exec'd children inherit it.
A venv interpreter that inherits ``__PYVENV_LAUNCHER__`` resolves its base prefix to ``/install``
and dies at interpreter init::

    Fatal Python error: init_fs_encoding: failed to get the Python codec of the filesystem encoding
    ModuleNotFoundError: No module named 'encodings'

``venv/bin/hermes`` has a ``venv/bin/python3`` shebang, so the update command and its retry both
die while a terminal ``hermes update`` works and the venv is perfectly healthy.

The failure is invisible from inside the launcher — the launcher is the process that CREATES the
variable — so these tests assert on what the re-exec'd child actually inherits, driving the real
hand-off entry point rather than a reimplementation of it. The daemonizer sends that child's stdout
to /dev/null, so the child reports through a file.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
POSIX_SH = REPO_ROOT / "scripts" / "desktop-update" / "posix.sh"

requires_bash = pytest.mark.skipif(
    not os.path.exists("/bin/bash"), reason="posix.sh needs /bin/bash"
)


def _run_handoff(env_overrides: dict, tmp_path: Path) -> dict:
    """Drive the real hand-off entry point; return the env the daemonized child saw."""
    install_root = tmp_path / "hermes-agent"
    (install_root / "venv" / "bin").mkdir(parents=True, exist_ok=True)
    report = tmp_path / "pyvenv-report.txt"

    env = dict(os.environ)
    env["HERMES_SELFTEST_PYVENV_ENV"] = str(report)
    env.update(env_overrides)

    subprocess.run(
        ["/bin/bash", str(POSIX_SH), "--install-root", str(install_root),
         "--branch", "main", "--no-ui"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60, env=env, cwd=str(tmp_path),
    )

    # The orchestrator is re-exec'd detached; wait briefly for it to write its report.
    for _ in range(50):
        if report.exists():
            break
        time.sleep(0.1)
    assert report.exists(), (
        "the daemonized orchestrator never reached the self-test hook — the hand-off did not "
        "survive its own re-exec"
    )
    parsed = {}
    for line in report.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            parsed[key.strip()] = value.strip()
    return parsed


@requires_bash
class TestHandoffScrubsInterpreterEnv:
    def test_launcher_and_path_vars_are_scrubbed_from_the_daemonized_child(self, tmp_path):
        """A pre-set __PYVENV_LAUNCHER__/PYTHONPATH must not reach the orchestrator.

        Set to the real macOS system-Python path rather than relying on that host's behaviour, so
        the contract holds on any host and on Linux CI.

        PYTHONHOME is deliberately absent: set before the hand-off it bricks the daemonizer's own
        ``/usr/bin/python3`` with the same ``No module named 'encodings'`` error, so the launcher
        never runs and nothing could scrub anything. That is a real limitation of any in-process
        repair, not a gap in this one — the variable is still popped for the case where a healthy
        process hands it down.
        """
        seen = _run_handoff(
            {
                "__PYVENV_LAUNCHER__": "/Library/Developer/CommandLineTools/usr/bin/python3",
                "PYTHONPATH": "/install/lib/python3.11",
            },
            tmp_path,
        )
        assert seen.get("launcher") == "<unset>", (
            "the daemonized orchestrator inherited __PYVENV_LAUNCHER__; the venv interpreter "
            f"would die with 'No module named encodings'. Child saw: {seen!r}"
        )
        assert seen.get("pythonpath") == "<unset>"

    def test_a_clean_environment_still_runs_the_handoff(self, tmp_path):
        """No poisoning → nothing to scrub; the hand-off must still complete its re-exec."""
        seen = _run_handoff({}, tmp_path)
        assert seen.get("launcher") == "<unset>"
