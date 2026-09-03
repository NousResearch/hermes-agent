#!/usr/bin/env python3
"""Native narrow-terminal startup smoke for the Hermes TUI.

Runs the installed ``hermes`` launcher inside a real PTY, at a phone-like
terminal size, long enough to catch import/gateway/layout startup failures.
The smoke intentionally does not need model credentials: an onboarding/setup
screen is a valid interactive state. It fails only when the process crashes,
produces a known fatal startup signature, or never paints anything.
"""

from __future__ import annotations

import argparse
import os
import select
import shutil
import signal
import struct
import subprocess
import sys
import time

if os.name != "posix":
    raise SystemExit("smoke-termux-tui.py requires a POSIX/Termux runtime")

import fcntl  # noqa: E402
import pty  # noqa: E402
import termios  # noqa: E402

FATAL_MARKERS = (
    b"Traceback (most recent call last)",
    b"Cannot find module",
    b"ERR_MODULE_NOT_FOUND",
    b"SyntaxError:",
    # This smoke creates the child on pty.openpty() and wires stdin/out/err to
    # the slave. ENOTTY therefore means startup code tried a terminal ioctl on
    # a non-TTY fd; it is not an expected property of a healthy Termux PTY.
    b"ENOTTY",
    b"gateway.start_timeout",
)
FIRST_RUN_SETUP_MARKERS = (
    b"No inference provider is configured yet",
    b"Set up a provider now?",
)


def _resize(fd: int, cols: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _signal_process_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    kill_group = getattr(os, "killpg", None)
    if callable(kill_group):
        kill_group(proc.pid, sig)
    else:  # Defensive fallback if the script is ever imported on a non-POSIX host.
        proc.terminate()


def _drain(master: int, proc: subprocess.Popen[bytes], deadline: float) -> bytes:
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], 0.1)
        if ready:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        if proc.poll() is not None and not ready:
            break
    return b"".join(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", default="hermes")
    parser.add_argument("--cols", type=int, default=48)
    parser.add_argument("--rows", type=int, default=18)
    parser.add_argument("--observe-seconds", type=float, default=6.0)
    args = parser.parse_args(argv)

    executable = shutil.which(args.command)
    if not executable:
        raise SystemExit(f"TUI smoke command not found: {args.command}")
    if args.cols < 20 or args.rows < 8:
        raise SystemExit("TUI smoke dimensions are unrealistically small")

    master, slave = pty.openpty()
    _resize(slave, args.cols, args.rows)
    env = os.environ.copy()
    env.update(
        {
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "HERMES_TUI_DISABLE_MOUSE": "1",
            "HERMES_TUI_STARTUP_TIMEOUT_MS": "12000",
        }
    )
    proc = subprocess.Popen(
        [executable],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        start_new_session=True,
    )
    os.close(slave)
    output = b""
    try:
        output += _drain(master, proc, time.monotonic() + args.observe_seconds)
        early_rc = proc.poll()
        if early_rc is not None:
            raise RuntimeError(f"Hermes TUI exited during startup with code {early_rc}")
        if len(output) < 32:
            raise RuntimeError("Hermes TUI did not paint meaningful terminal output")
        for marker in FATAL_MARKERS:
            if marker in output:
                raise RuntimeError(f"Hermes TUI emitted fatal startup marker: {marker.decode(errors='replace')}")

        # A fresh install without credentials deliberately hands off from the
        # painted TUI into an interactive provider-onboarding prompt.  That
        # prompt catches KeyboardInterrupt and treats Ctrl-C as "skip setup",
        # so using Ctrl-C immediately would test onboarding semantics instead
        # of the TUI exit path. Decline onboarding explicitly, then exercise
        # Ctrl-C once the normal interactive surface owns the terminal again.
        if any(marker in output for marker in FIRST_RUN_SETUP_MARKERS):
            os.write(master, b"n\n")
            output += _drain(master, proc, time.monotonic() + 2.0)
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Hermes exited after declining first-run setup with code {proc.returncode}"
                )

        # Ctrl-C on an idle non-dashboard TUI is the normal exit hotkey.  Give
        # the Node parent enough time to reap its gateway child; gracefulExit's
        # own failsafe is four seconds, so the smoke grace must be longer than
        # that rather than racing it at the same deadline.
        os.write(master, b"\x03")
        output += _drain(master, proc, time.monotonic() + 5.5)
        try:
            rc = proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired as exc:
            _signal_process_group(proc, signal.SIGTERM)
            proc.wait(timeout=3.0)
            raise RuntimeError("Hermes TUI did not exit after onboarding was dismissed and Ctrl-C was sent") from exc
        if rc not in (0, 130, -signal.SIGINT):
            raise RuntimeError(f"Hermes TUI did not shut down cleanly after Ctrl-C (code {rc})")
    except Exception as exc:
        preview = output[-8000:].decode("utf-8", errors="replace")
        print(preview, file=sys.stderr)
        print(f"termux-tui-smoke failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if proc.poll() is None:
            try:
                _signal_process_group(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
            except OSError:
                pass
        os.close(master)

    print(
        f"termux-tui-smoke-ok cols={args.cols} rows={args.rows} bytes={len(output)} rc={rc}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
