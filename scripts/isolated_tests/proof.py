"""Human-only confirmation through the controlling terminal.

Reads /dev/tty directly (never stdin) and requires the operator to retype a
fresh random code within CONFIRM_TIMEOUT_SECONDS (expiry = HOLD); there is no
flag, env var or CI mode that skips it.

"Human-only" is an operating rule, not a technical control: a process that owns
a PTY (an agent terminal included) can read the code and type it. Agents must
never run prove-human.
"""

from __future__ import annotations

import hmac
import io
import os
import select
import time

from isolated_tests.attestation import new_code


CONFIRM_TIMEOUT_SECONDS = 120


class ProofRefused(Exception):
    pass


def _read_line(tty, timeout: float, clock=time.monotonic) -> str:
    """One line within a single deadline: a partially typed line cannot outlive it."""
    deadline = clock() + timeout
    fd = tty.fileno()
    buf = bytearray()
    while b"\n" not in buf:
        remaining = deadline - clock()
        ready = select.select([fd], [], [], remaining)[0] if remaining > 0 else []
        if not ready:
            raise ProofRefused(f"no confirmation within {timeout:.0f}s; HOLD (nothing recorded)")
        chunk = os.read(fd, 4096)  # select said readable: returns what is there, never waits
        if not chunk:
            raise ProofRefused("terminal closed before confirmation; HOLD (nothing recorded)")
        buf += chunk
    return buf.split(b"\n", 1)[0].decode("utf-8", "replace")


def confirm(tty, prompt: str, timeout: float = CONFIRM_TIMEOUT_SECONDS) -> str:
    """`tty` is a text file opened on /dev/tty (read+write)."""
    if tty is None or not tty.isatty():
        raise ProofRefused("prove-human needs an interactive terminal (/dev/tty)")
    code = new_code()
    tty.write(f"{prompt}\nType this code to confirm: {code}\n> ")
    tty.flush()
    answer = _read_line(tty, timeout).strip()
    if not hmac.compare_digest(answer.encode(), code.encode()):
        raise ProofRefused("confirmation code mismatch; nothing recorded")
    return code


def open_tty(path: str = "/dev/tty"):
    """Text layer straight on the raw FileIO: open(path, "r+") builds a BufferedRandom, whose seek
    check raises UnsupportedOperation (an OSError) on a terminal, so prove-human always refused."""
    try:
        fd = os.open(path, os.O_RDWR | getattr(os, "O_NOCTTY", 0))
    except OSError:
        return None
    raw = None
    try:
        raw = io.FileIO(fd, "r+", closefd=True)
        return io.TextIOWrapper(raw, encoding="utf-8", line_buffering=True, write_through=True)
    except BaseException:
        if raw is None:
            os.close(fd)
        else:
            raw.close()
        raise
