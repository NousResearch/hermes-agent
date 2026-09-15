"""Keep Windows console backpressure off the packaged desktop startup path."""

import logging
import os
import sys
import threading
from contextlib import contextmanager

logger = logging.getLogger("hermes_cli.desktop")


def desktop_launch_notice(message: str, *, source_mode: bool = False) -> None:
    if sys.platform == "win32" and not source_mode:
        # setup_logging() uses an async queue and rotating, redacted files.
        logger.info(message)
    else:
        print(message)


@contextmanager
def desktop_console_output(*, source_mode: bool):
    """Drain packaged Windows stdout/stderr into the existing rotating logs.

    Inheriting the console lets an unresponsive console host/profiler block
    Electron's synchronous startup messages before it can create a window.
    Keep source launches interactive and preserve subprocess.run's lifecycle
    and exit-code handling. Read bounded lines so output cannot grow a buffer
    without limit, and never wait indefinitely on a descendant's open pipe.
    """
    if sys.platform != "win32" or source_mode:
        yield {}
        return

    import subprocess

    read_fd, write_fd = os.pipe()

    def drain():
        with os.fdopen(read_fd, "rb") as stream:
            while data := stream.readline(8192):
                logger.info("[desktop] %s", data.decode("utf-8", errors="replace").rstrip())

    reader = threading.Thread(target=drain, name="desktop-console", daemon=True)
    reader.start()
    try:
        yield {"stdout": write_fd, "stderr": subprocess.STDOUT}
    finally:
        os.close(write_fd)
        reader.join(timeout=1)
