#!/usr/bin/env python3
"""E2E harness for OMP-style Hermes TUI first-paint (PR #99776)."""

from __future__ import annotations

import argparse
import os
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_TUI = _PROJECT_ROOT / "ui-tui"
UI_DIST = UI_TUI / "dist" / "entry.js"
MONITOR_PID = Path("/tmp/hermes-omp-monitor.pid")
MONITOR_LOG = Path("/tmp/hermes-omp-monitor.log")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
COMPOSER_BORDER_RE = re.compile(r"[╭╮╯╰┌┐└┘│─]")
READY_HINTS = ("❯", "ready", "/help", "tools ·")
# hermes_cli.main._suppress_mouse_residue_early() — often the only PTY bytes
# while npm install/build runs with captured stdout before Ink starts.
MOUSE_CSI = (
    b"\x1b[?1003l\x1b[?1002l\x1b[?1001l\x1b[?1000l\x1b[?9l"
    b"\x1b[?1006l\x1b[?1005l\x1b[?1015l\x1b[?1016l\x1b[?2029l"
)


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
    """Spawn *argv* on a session-leading PTY (openpty + setsid).

    ``pty.fork()`` only surfaced early mouse-mode CSI bytes here; Ink needs the
    same slave-side setup the detached monitor uses.
    """
    import fcntl
    import struct
    import termios

    master, slave = pty.openpty()
    pid = os.fork()
    if pid == 0:
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
    return pid, master


def run_component_acceptance() -> int:
    script = UI_TUI / "scripts" / "omp-acceptance.mjs"
    return subprocess.run(["node", str(script)], cwd=str(UI_TUI)).returncode


def bootstrap_tui_workspace(*, quiet: bool = False) -> int:
    """Install ui-tui deps and build dist/ so ``hermes --tui`` can reach Ink.

    ``hermes --tui`` runs npm install + ``npm run build`` with captured stdout
    before exec'ing Node. On a cold checkout that phase is silent on the PTY
    (only the early mouse-mode CSI bytes appear) and can take minutes if npm is
    slow or wedged — this preflight avoids that for E2E and local dev.
    """
    npm = shutil.which("npm") or "npm"
    env = os.environ.copy()
    env["CI"] = "1"

    if not quiet:
        print("bootstrap: npm install --workspace ui-tui …", flush=True)
    rc = subprocess.run(
        [
            npm,
            "install",
            "--workspace",
            "ui-tui",
            "--include=dev",
            "--no-fund",
            "--no-audit",
            "--progress=false",
        ],
        cwd=str(_PROJECT_ROOT),
        env=env,
    ).returncode
    if rc != 0:
        print("bootstrap: npm install failed", file=sys.stderr)
        return rc

    for step in ("build:ink", "build"):
        if not quiet:
            print(f"bootstrap: npm run {step} …", flush=True)
        rc = subprocess.run([npm, "run", step], cwd=str(UI_TUI), env=env).returncode
        if rc != 0:
            print(f"bootstrap: npm run {step} failed", file=sys.stderr)
            return rc

    if not UI_DIST.is_file():
        print("bootstrap: dist/entry.js still missing after build", file=sys.stderr)
        return 1

    if not quiet:
        kb = UI_DIST.stat().st_size // 1024
        print(f"bootstrap: ready ({kb}K ui-tui/dist/entry.js)", flush=True)
    return 0


def cmd_bootstrap(_: argparse.Namespace) -> int:
    return bootstrap_tui_workspace()


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


def _print_launch_hint(captured: bytes, text: str) -> None:
    if len(captured) <= len(MOUSE_CSI) + 20 or captured.startswith(MOUSE_CSI[:20]):
        print(
            "\nHINT: PTY only saw Hermes mouse-mode CSI — Ink never started.\n"
            "      hermes --tui blocks on silent npm install/build first.\n"
            "      Run:  python3 scripts/e2e-omp-tui.py bootstrap\n"
            "      Then: hermes --tui   (in kitty/foot, not the IDE panel)\n"
            "      Kill stuck npm:  pkill -f 'npm install'",
            file=sys.stderr,
        )
    elif strip_ansi(text):
        print(strip_ansi(text)[-2000:], file=sys.stderr)
    else:
        print(repr(captured[:500]), file=sys.stderr)


def cmd_test(args: argparse.Namespace) -> int:
    component_rc = run_component_acceptance()
    if component_rc != 0:
        print("component acceptance failed", file=sys.stderr)
        return component_rc

    if not args.live:
        print("OMP component acceptance passed (live PTY skipped; pass --live)")
        return 0

    if not args.no_bootstrap:
        boot_rc = bootstrap_tui_workspace(quiet=args.quiet_bootstrap)
        if boot_rc != 0:
            return boot_rc

    hermes = find_hermes()
    use_dev = args.dev or not UI_DIST.exists()
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
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
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
        _print_launch_hint(captured, text)
        return 1

    print("component + live PTY OMP acceptance passed")
    return 0


def _monitor_worker(argv: list[str], env: dict[str, str], cols: int, rows: int) -> None:
    child, master = launch_pty(argv, env, cols, rows)

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

    if not UI_DIST.is_file():
        print("dist/ missing — running bootstrap first …", flush=True)
        if bootstrap_tui_workspace() != 0:
            return 1

    hermes = find_hermes()
    use_dev = args.dev or not UI_DIST.exists()
    argv = tui_argv(hermes, dev=use_dev)
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

    sub.add_parser("bootstrap", help="npm install + build ui-tui (run once before hermes --tui)")

    t = sub.add_parser("test", help="run component + live PTY acceptance")
    t.add_argument("--cols", type=int, default=100)
    t.add_argument("--rows", type=int, default=32)
    t.add_argument("--timeout", type=float, default=180.0)
    t.add_argument("--live", action="store_true", help="also spawn hermes --tui in a PTY smoke test")
    t.add_argument("--dev", action="store_true", help="force tsx dev mode for --live (default when dist/ is missing)")
    t.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="skip npm install/build preflight before --live (not recommended)",
    )
    t.add_argument("--quiet-bootstrap", action="store_true", help="less bootstrap logging")

    m = sub.add_parser("monitor", help="launch detached TUI for external monitoring")
    m.add_argument("--cols", type=int, default=100)
    m.add_argument("--rows", type=int, default=32)
    m.add_argument("--dev", action="store_true", help="run ui-tui via tsx (slower; optional)")

    sub.add_parser("stop", help="stop detached monitor")

    args = p.parse_args()
    if args.cmd == "bootstrap":
        return cmd_bootstrap(args)
    if args.cmd == "test":
        return cmd_test(args)
    if args.cmd == "monitor":
        return cmd_monitor(args)
    if args.cmd == "stop":
        return cmd_stop(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
