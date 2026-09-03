#!/usr/bin/env python3
"""E2E harness for OMP-style Hermes TUI first-paint (PR #99776)."""

from __future__ import annotations

import argparse
import os
import pty
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MONITOR_PID = Path("/tmp/hermes-omp-monitor.pid")
MONITOR_LOG = Path("/tmp/hermes-omp-monitor.log")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
COMPOSER_BORDER_RE = re.compile(r"[╭╮╯╰┌┐└┘│─]")
READY_HINTS = ("❯", "ready", "/help", "tools ·")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def find_hermes() -> str:
    hermes = os.environ.get("HERMES_BIN") or "hermes"
    which = subprocess.run(["which", hermes], capture_output=True, text=True)
    if which.returncode == 0 and which.stdout.strip():
        return which.stdout.strip()
    local = _PROJECT_ROOT / ".venv" / "bin" / "hermes"
    if local.is_file():
        return str(local)
    return hermes


def read_available(fd: int, timeout: float) -> bytes:
    chunks: list[bytes] = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        r, _, _ = select.select([fd], [], [], min(0.25, remaining))
        if not r:
            continue
        try:
            data = os.read(fd, 8192)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


def terminate_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                return
            if done == pid:
                return
            time.sleep(0.1)
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    except ProcessLookupError:
        pass


def tui_argv(hermes: str, dev: bool = False) -> list[str]:
    argv = [hermes, "--tui"]
    if dev:
        argv.append("--dev")
    return argv


def build_env(cols: int, rows: int) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TERM", "xterm-256color")
    env["COLUMNS"] = str(cols)
    env["LINES"] = str(rows)
    env["FORCE_COLOR"] = "3"
    env["COLORTERM"] = "truecolor"
    env.pop("HERMES_TUI_RESUME", None)
    return env


def launch_pty(argv: list[str], env: dict[str, str], cols: int, rows: int) -> tuple[int, int]:
    import fcntl
    import struct
    import termios

    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe(argv[0], argv, env)
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)
    return pid, fd


def run_component_acceptance() -> int:
    ui_tui = _PROJECT_ROOT / "ui-tui"
    script = ui_tui / "scripts" / "omp-acceptance.mjs"
    return subprocess.run(["node", str(script)], cwd=str(ui_tui)).returncode


def assert_live_acceptance(text: str) -> list[str]:
    failures: list[str] = []
    plain = strip_ansi(text)

    if not any(hint in plain for hint in READY_HINTS):
        failures.append("TUI never reached ready state (missing prompt / help hints)")

    if not COMPOSER_BORDER_RE.search(plain):
        failures.append("framed composer missing box-drawing border characters")

    if "Available Tools" in plain and "file:" in plain.lower():
        failures.append("tools accordion appears expanded on fresh session")

    return failures


def cmd_test(args: argparse.Namespace) -> int:
    component_rc = run_component_acceptance()
    if component_rc != 0:
        print("component acceptance failed", file=sys.stderr)
        return component_rc

    if not args.live:
        print("OMP component acceptance passed (live PTY skipped; pass --live)")
        return 0

    hermes = find_hermes()
    ui_dist = _PROJECT_ROOT / "ui-tui" / "dist" / "entry.js"
    use_dev = args.dev or not ui_dist.exists()
    argv = tui_argv(hermes, dev=use_dev)
    env = build_env(args.cols, args.rows)
    print(f"live PTY: {' '.join(argv)}")

    pid, fd = launch_pty(argv, env, args.cols, args.rows)
    captured = b""
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            chunk = read_available(fd, min(2.0, deadline - time.monotonic()))
            if chunk:
                captured += chunk
                plain = strip_ansi(captured.decode("utf-8", errors="replace"))
                if any(h in plain for h in READY_HINTS) and COMPOSER_BORDER_RE.search(plain):
                    break
            else:
                time.sleep(0.1)
    finally:
        terminate_pid(pid)
        try:
            os.close(fd)
        except OSError:
            pass

    text = captured.decode("utf-8", errors="replace")
    failures = assert_live_acceptance(text)
    if failures:
        for item in failures:
            print(f"FAIL: {item}", file=sys.stderr)
        print(strip_ansi(text)[-2000:], file=sys.stderr)
        return 1

    print("component + live PTY OMP acceptance passed")
    return 0


def _monitor_worker(argv: list[str], env: dict[str, str], cols: int, rows: int) -> None:
    import fcntl
    import struct
    import termios

    master, slave = pty.openpty()
    child = os.fork()
    if child == 0:
        os.setsid()
        os.close(master)
        os.dup2(slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        if slave > 2:
            os.close(slave)
        os.chdir(str(_PROJECT_ROOT))
        os.execvpe(argv[0], argv, env)

    os.close(slave)
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(master, termios.TIOCSWINSZ, winsize)

    with MONITOR_LOG.open("ab", buffering=0) as log_fp:
        while True:
            r, _, _ = select.select([master], [], [], 1.0)
            if not r:
                try:
                    os.kill(child, 0)
                except OSError:
                    break
                continue
            try:
                data = os.read(master, 8192)
            except OSError:
                break
            if not data:
                break
            log_fp.write(data)
            log_fp.flush()

    try:
        os.close(master)
    except OSError:
        pass


def cmd_monitor(args: argparse.Namespace) -> int:
    if MONITOR_PID.exists():
        old = int(MONITOR_PID.read_text().strip())
        try:
            os.kill(old, 0)
            print(f"monitor already running (pid {old})")
            print(f"tail -f {MONITOR_LOG}")
            return 0
        except OSError:
            MONITOR_PID.unlink(missing_ok=True)

    hermes = find_hermes()
    argv = tui_argv(hermes, dev=args.dev)
    env = build_env(args.cols, args.rows)

    MONITOR_LOG.write_bytes(b"")

    worker = os.fork()
    if worker == 0:
        os.setsid()
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
        _monitor_worker(argv, env, args.cols, args.rows)
        raise SystemExit(0)

    MONITOR_PID.write_text(str(worker), encoding="utf-8")
    print(f"detached monitor pid {worker}")
    print(f"log: tail -f {MONITOR_LOG}")
    print(f"stop: python {Path(__file__).name} stop")
    return 0


def cmd_stop(_: argparse.Namespace) -> int:
    if not MONITOR_PID.exists():
        print("no monitor pid file")
        return 0
    pid = int(MONITOR_PID.read_text().strip())
    terminate_pid(pid)
    MONITOR_PID.unlink(missing_ok=True)
    print(f"stopped monitor pid {pid}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="OMP TUI E2E harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("test", help="run component + live PTY acceptance")
    t.add_argument("--cols", type=int, default=100)
    t.add_argument("--rows", type=int, default=32)
    t.add_argument("--timeout", type=float, default=120.0)
    t.add_argument("--live", action="store_true", help="also spawn hermes --tui in a PTY smoke test")
    t.add_argument("--dev", action="store_true", help="force tsx dev mode for --live (default when dist/ is missing)")

    m = sub.add_parser("monitor", help="launch detached TUI for external monitoring")
    m.add_argument("--cols", type=int, default=100)
    m.add_argument("--rows", type=int, default=32)
    m.add_argument("--dev", action="store_true", help="run ui-tui via tsx (slower; optional)")

    sub.add_parser("stop", help="stop detached monitor")

    args = p.parse_args()
    if args.cmd == "test":
        return cmd_test(args)
    if args.cmd == "monitor":
        return cmd_monitor(args)
    if args.cmd == "stop":
        return cmd_stop(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
