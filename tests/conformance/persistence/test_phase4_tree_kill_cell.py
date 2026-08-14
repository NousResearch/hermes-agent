"""Phase 4 acceptance cell — zero orphaned descendants after a script timeout.

Red-until-landed per the #85125 acceptance conditions: this cell asserts the
contract the Phase 4 implementation must satisfy (a timeout-killed cron script
leaves no living descendants, including setsid'd grandchildren), and it
currently FAILS on main — `_run_job_script` uses ``subprocess.run(timeout=...)``
which kills only the direct child. Marked ``xfail(strict=False)`` so it cannot
break CI; it flips XPASS the moment the delegated fix (``kill_process_tree`` on
the script-timeout handler, #59379 salvage + #85147 Phase 1) merges, and is
promoted to hard-green in that same PR.

This is the end-to-end version of the Phase 1 primitive probe
(``test_kills_descendant_in_its_own_session``); it drives the real cron
script path instead of the primitive.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


@pytest.mark.xfail(
    strict=False,
    reason="#85125 Phase 4: _run_job_script timeout kills only the direct "
    "child; setsid'd grandchildren survive until kill_process_tree lands",
)
def test_script_timeout_leaves_no_living_descendants(tmp_path, monkeypatch):
    """Kill a script at its timeout and assert zero surviving descendants."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from cron import scheduler as sched

    home_token = set_hermes_home_override(str(tmp_path))
    try:
        # _run_job_script validates scripts live under HERMES_HOME/scripts/.
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        pid_file = tmp_path / "grandchild.pid"
        final_script = scripts_dir / "spawner.py"
        # Python-based detached spawn: start_new_session works on macOS and
        # Linux (the setsid(1) binary is Linux-only, which silently skipped
        # the spawn on macOS). DEVNULL stdio so the grandchild doesn't hold
        # the runner's output pipe open past the parent kill.
        final_script.write_text(
            "import subprocess, sys, time\n"
            "p = subprocess.Popen(\n"
            "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
            "    start_new_session=True,\n"
            "    stdin=subprocess.DEVNULL,\n"
            "    stdout=subprocess.DEVNULL,\n"
            "    stderr=subprocess.DEVNULL,\n"
            ")\n"
            "open('grandchild.pid', 'w').write(str(p.pid))\n"
            "time.sleep(30)\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "1")
        monkeypatch.setattr(sched, "_SCRIPT_TIMEOUT", sched._DEFAULT_SCRIPT_TIMEOUT)

        # Drive the REAL cron script path with a 1s script timeout.
        started = time.monotonic()
        ok, out = sched._run_job_script(str(final_script), workdir=str(tmp_path))
        assert not ok, f"script should have timed out, got {out!r}"
        assert time.monotonic() - started < 20, "script did not time out promptly"

        # Reap the grandchild pid (written before the timeout kill).
        deadline = time.monotonic() + 5
        gpid = None
        while time.monotonic() < deadline and gpid is None:
            try:
                gpid = int(pid_file.read_text().strip())
            except (FileNotFoundError, ValueError):
                time.sleep(0.05)

        assert gpid is not None, "spawner never wrote the grandchild pid"

        try:
            alive = True
            for _ in range(10):
                try:
                    os.kill(gpid, 0)
                except ProcessLookupError:
                    alive = False
                    break
                time.sleep(0.2)
            assert not alive, (
                f"grandchild pid {gpid} survived the script timeout — the "
                "timeout path orphaned a setsid'd descendant"
            )
        finally:
            # Never leave the probe's own grandchild behind.
            try:
                os.kill(gpid, 9)
            except ProcessLookupError:
                pass
    finally:
        reset_hermes_home_override(home_token)
