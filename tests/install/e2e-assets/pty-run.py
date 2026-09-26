"""Run a command under a Windows pseudoconsole, capturing its output.

Why this exists: the install-e2e windows legs run the installed CLI's one-shot
chat with its stdout piped into the evidence log. A released tag prints through
prompt_toolkit, whose ``create_output()`` on Windows returns ``Win32Output``, and
that constructor needs a console screen buffer -- a pipe has none, so the step
died with::

    prompt_toolkit.output.win32.NoConsoleScreenBufferError: No Windows console found.

HEAD guards its own print path, but the tags under test cannot be changed, so the
console has to come from the rig. pywinpty is the ConPTY spawner the product
itself uses on Windows (``hermes_cli/win_pty_bridge.py``); this wraps it so a
piped-stdout child still gets a real console. Without pywinpty the helper falls
back to a plain subprocess and says so in the log, so a missing dependency
degrades the evidence instead of silently changing what the leg proves.

Usage:
    pty-run.py --out <log> [--cwd DIR] [--cols N] [--rows N]
               [--timeout SECONDS] -- <exe> [args...]

Exits with the child's exit code (124 if the timeout killed it).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _run_under_pty(argv: list[str], out_path: str, cwd: str | None, cols: int, rows: int,
                   timeout: float | None) -> int | None:
    """Return the child's exit code, or None when pywinpty is unavailable."""
    try:
        from winpty import PtyProcess
    except ImportError:
        return None

    proc = PtyProcess.spawn(argv, cwd=cwd, dimensions=(rows, cols))
    deadline = None if timeout is None else time.monotonic() + timeout
    timed_out = False
    with open(out_path, "w", encoding="utf-8", errors="replace") as log:
        while True:
            if deadline is not None and time.monotonic() > deadline:
                timed_out = True
                proc.terminate(force=True)
                break
            try:
                chunk = proc.read(4096)
            except EOFError:
                break
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            sys.stdout.write(chunk)
            sys.stdout.flush()
            if not proc.isalive():
                break
    proc.wait()
    if timed_out:
        return 124
    return int(proc.exitstatus or 0)


def _run_piped(argv: list[str], out_path: str, cwd: str | None, timeout: float | None) -> int:
    """Fallback: a pipe, which is exactly what breaks a released tag's print path."""
    with open(out_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("pty-run: pywinpty is unavailable, so the child got a pipe instead of a "
                  "console; a released tag's print path will fail here.\n")
        log.flush()
        proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", bufsize=1)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 124
        return int(proc.returncode or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command under a Windows pseudoconsole.")
    parser.add_argument("--out", required=True, help="file to capture the child's output into")
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--cols", type=int, default=120)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=None, help="seconds; kills the child and exits 124")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = [a for a in args.command if a != "--"]
    if not argv:
        parser.error("no command given")
    if not sys.platform.startswith("win"):  # pragma: no cover - rig is windows-only here
        raise SystemExit("pty-run.py is Windows-only")

    code = _run_under_pty(argv, args.out, args.cwd, args.cols, args.rows, args.timeout)
    if code is None:
        code = _run_piped(argv, args.out, args.cwd, args.timeout)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
