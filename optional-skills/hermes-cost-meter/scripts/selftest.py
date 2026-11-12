#!/usr/bin/env python3
"""End-to-end self-test: launch ./hermes-cost in a pty, run one turn, assert the meter.

No TTY needed — this allocates its own pty, so it works from inside another agent
session, CI, or a plain shell. Costs one cheap model call.

    python3 tool/selftest.py            # exit 0 = meter rendered and /cost worked
    python3 tool/selftest.py --keep     # also dump the captured screen
"""

from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hermes-cost")
ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[()][A-Za-z0-9]")


def run(prompt: str = "Reply with exactly: hello", boot_s: int = 20, turn_s: int = 50) -> str:
    env = dict(os.environ, TERM="xterm-256color")
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 140, 0, 0))
    proc = subprocess.Popen([TOOL], stdin=slave, stdout=slave, stderr=slave,
                            env=env, cwd=os.path.expanduser("~"), close_fds=True)
    os.close(slave)
    buf = b""

    def pump(seconds: float) -> None:
        nonlocal buf
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([master], [], [], 0.3)
            if not ready:
                continue
            try:
                data = os.read(master, 65536)
            except OSError:
                return
            if not data:
                return
            buf += data

    try:
        pump(boot_s)
        os.write(master, prompt.encode() + b"\r")
        pump(turn_s)
        os.write(master, b"/cost\r")
        pump(8)
        os.write(master, b"/quit\r")
        pump(6)
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
    return ANSI.sub("", buf.decode("utf-8", "replace"))


def main() -> int:
    text = run()
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    # The meter must sit ON the status bar line (the one carrying the ☤ model glyph),
    # not on a row of its own — that was the whole point of the exercise.
    bar = [l for l in lines if "☤" in l]
    meter = [l for l in bar if re.search(r"~\$\d|cost: ", l)]
    breakdown = [l for l in lines if "Session cost (estimate" in l]
    priced = [l for l in lines if re.search(r"session cost\s+~\$", l)]
    crashed = [l for l in lines if "Traceback" in l or "hermes-cost:" in l]

    ok = True
    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")

    print("hermes-cost self-test")
    check("no crash", not crashed, crashed[0][:120] if crashed else "")
    check("meter on the status bar", bool(meter), meter[-1].strip()[:110] if meter else "not found")
    check("/cost breakdown printed", bool(breakdown))
    check("session priced (not 'no pricing')", bool(priced), priced[-1].strip()[:80] if priced else "")
    if "--keep" in sys.argv:
        print("\n--- captured screen (tail) ---")
        for line in lines[-25:]:
            print(line[:180])
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
