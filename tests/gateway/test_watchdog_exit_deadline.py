"""A wedged diagnostic or foreground cleanup cannot defeat watchdog exit (#148853)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("blocked", ["stderr", "status", "logging", "foreground"])
def test_liveness_watchdog_exits_past_blocked_side_effects(tmp_path, blocked):
    child = textwrap.dedent(
        """
        import asyncio
        import logging
        import os
        import sys
        import threading

        from gateway import shutdown_watchdog as watchdog

        blocked = sys.argv[1]
        if blocked == "stderr":
            read_fd, write_fd = os.pipe()
            os.set_blocking(write_fd, False)
            try:
                while True:
                    os.write(write_fd, b"x" * 512)
            except BlockingIOError:
                pass
            # Leave room for the header but not the all-thread traceback. A C-level
            # faulthandler write then blocks with the GIL held, defeating thread timers.
            os.read(read_fd, 512)
            os.set_blocking(write_fd, True)
            os.dup2(write_fd, 2)
            logging.getLogger("gateway.shutdown_watchdog").addHandler(logging.NullHandler())
        elif blocked == "status":
            import gateway.status as status
            status._runtime_status_state_lock.acquire()
        elif blocked == "logging":
            class BlockedHandler(logging.Handler):
                def emit(self, record):
                    threading.Event().wait()
            watchdog.logger.addHandler(BlockedHandler())
        elif blocked == "foreground":
            import tools.environments.base as base
            base._live_foreground_cond.acquire()

        # No event-loop callbacks run, so the first probe must trigger a real exit.
        loop = asyncio.new_event_loop()
        watchdog.start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1,
        )
        print("armed", flush=True)
        threading.Event().wait()
        """
    )
    env = {**os.environ, "HERMES_HOME": str(tmp_path)}
    with subprocess.Popen(
        [sys.executable, "-c", child, blocked],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    ) as process:
        try:
            # Deliberately do not drain stderr: the watchdog must not rely on its reader.
            try:
                code = process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                pytest.fail(f"watchdog remained alive with blocked {blocked}")
            assert process.stdout is not None
            assert process.stdout.read().strip() == "armed"
            assert code == 75
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
