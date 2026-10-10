"""Keep Windows console backpressure off the packaged desktop startup path."""

import logging
import os
import sys
import threading
import time
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

    def drain(read_fd, level):
        with os.fdopen(read_fd, "rb") as stream:
            # Decode/redact complete records, not arbitrary pipe-read fragments.
            # Discard an oversized line in full: even its prefix may be a secret.
            limit = 128 * 1024
            while data := stream.readline(limit + 1):
                if len(data) > limit:
                    while data and not data.endswith(b"\n"):
                        data = stream.readline(8192)
                    logger.log(level, "[desktop] [oversized line omitted]")
                else:
                    logger.log(level, "[desktop] %s", data.decode("utf-8", errors="replace").rstrip())

    streams = {}
    readers = []
    try:
        # Keep diagnostics visible even when ordinary INFO output is disabled.
        for name, level in (("stdout", logging.INFO), ("stderr", logging.ERROR)):
            read_fd, write_fd = os.pipe()
            streams[name] = write_fd
            reader = threading.Thread(
                target=drain, args=(read_fd, level), name=f"desktop-console-{name}", daemon=True,
            )
            try:
                reader.start()
            except BaseException:
                os.close(read_fd)
                raise
            readers.append(reader)
        yield streams
    finally:
        for write_fd in streams.values():
            os.close(write_fd)
        deadline = time.monotonic() + 1
        for reader in readers:
            reader.join(timeout=max(0, deadline - time.monotonic()))
