"""A wedged kill-path side effect cannot keep the liveness watchdog from exiting (#148853),
and a blocked diagnostic write cannot hold the shutdown watchdog's exit either (#160012)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading

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
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pytest.fail(f"watchdog remained alive with blocked {blocked}")
            assert process.stdout is not None
            assert process.stdout.read().strip() == "armed"
            assert code == 75
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("blocked", ["stderr", "dumpfile", "dumpwrite"])
def test_shutdown_watchdog_exits_past_blocked_diagnostics(tmp_path, blocked):
    child = textwrap.dedent(
        """
        import os
        import sys
        import threading
        from pathlib import Path

        from gateway import shutdown_watchdog as watchdog

        blocked = sys.argv[1]
        dump_path = None
        if blocked == "stderr":
            read_fd, write_fd = os.pipe()
            os.set_blocking(write_fd, False)
            try:
                while True:
                    os.write(write_fd, b"x" * 512)
            except BlockingIOError:
                pass
            # Leave room for the header line but not the all-thread stacks: the dump's
            # stderr write then blocks partway through.
            os.read(read_fd, 512)
            os.set_blocking(write_fd, True)
            os.dup2(write_fd, 2)
        elif blocked == "dumpfile":
            # A FIFO with no reader: opening the dump file blocks until a reader appears,
            # a dump destination that has stopped answering.
            dump_path = Path(sys.argv[2])
            os.mkfifo(dump_path)
        elif blocked == "dumpwrite":
            # A FIFO whose reader stops after the first bytes (the parent reads 1 KiB, then
            # never again): the stacks write blocks partway through, once the pipe fills.
            # Enough deep-stacked threads that the rendered dump outgrows the pipe buffer.
            # The parent created the FIFO before spawning, so its reader cannot race creation.
            dump_path = Path(sys.argv[2])
            release = threading.Event()

            def spin(depth):
                if depth:
                    return spin(depth - 1)
                release.wait(timeout=30)

            for _ in range(105):
                threading.Thread(target=spin, args=(40,), daemon=True).start()

        watchdog.arm_shutdown_watchdog(0.5, dump_path=dump_path)
        print("armed", flush=True)
        threading.Event().wait()
        """
    )
    env = {**os.environ, "HERMES_HOME": str(tmp_path)}
    fifo_path = tmp_path / "watchdog-dump.fifo"
    reader_fd = None
    reader_outcome: dict[str, object] = {}
    reader = None
    if blocked == "dumpwrite":
        os.mkfifo(fifo_path)  # before spawning: the reader below can never race its creation

        def read_a_little():
            try:
                nonlocal reader_fd
                reader_fd = os.open(fifo_path, os.O_RDONLY)
                reader_outcome["bytes"] = len(os.read(reader_fd, 1024))
            except Exception as exc:  # surfaced by the assert after the child exits
                reader_outcome["error"] = exc

        reader = threading.Thread(target=read_a_little, daemon=True)
        reader.start()
    with subprocess.Popen(
        [sys.executable, "-c", child, blocked, str(fifo_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    ) as process:
        try:
            # Deliberately do not drain stderr: the watchdog must not rely on its reader.
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pytest.fail(f"shutdown watchdog remained alive with blocked {blocked}")
            assert process.stdout is not None
            assert process.stdout.read().strip() == "armed"
            assert code == 1
            if blocked == "dumpwrite":
                reader.join(timeout=5.0)
                assert reader_outcome.get("bytes", 0) > 0, (
                    f"dump reader never read: {reader_outcome.get('error')!r}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            if reader_fd is not None:
                os.close(reader_fd)
