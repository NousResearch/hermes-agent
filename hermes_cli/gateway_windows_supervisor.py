"""Single-owner bounded Windows gateway supervisor.

This module is intentionally stdlib-only. The generated hidden VBS launcher starts
one instance; an OS byte-range lock fences duplicate launchers before any child is
spawned. Stop requests carry a nonce and are acknowledged by the lock owner.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import subprocess
import sys
import time
from pathlib import Path

GATEWAY_FATAL_CONFIG_EXIT_CODE = 78
DEFAULT_RESTART_DELAY_MS = 5000
DEFAULT_FAILURE_WINDOW_S = 300
DEFAULT_MAX_CONSECUTIVE_EXITS = 3


def _paths(home: Path) -> tuple[Path, Path, Path]:
    root = home / "gateway-service"
    return root / "supervisor.stop", root / "supervisor.stop.ack", root / "supervisor.owner"


def _read_stop_token(marker: Path) -> str | None:
    try:
        token = marker.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None
    return token or None


def _ack_stop(marker: Path, ack: Path) -> bool:
    token = _read_stop_token(marker)
    if token is None:
        return False
    try:
        ack.parent.mkdir(parents=True, exist_ok=True)
        tmp = ack.with_suffix(ack.suffix + ".tmp")
        tmp.write_text(token, encoding="utf-8")
        tmp.replace(ack)
        marker.unlink(missing_ok=True)
        return True
    except OSError:
        return False


class _OwnerLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        if sys.platform != "win32":
            raise RuntimeError("Windows gateway supervisor is Windows-only")
        import msvcrt

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b", buffering=0)
        try:
            if fh.seek(0, os.SEEK_END) == 0:
                fh.write(b"0")
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        fh = self._fh
        self._fh = None
        if fh is None:
            return
        try:
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        finally:
            fh.close()


def _sleep_with_stop(seconds: float, marker: Path, ack: Path) -> bool:
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        if _ack_stop(marker, ack):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.1, remaining))


def supervise(
    child_argv: list[str],
    *,
    home: Path,
    restart_delay_ms: int = DEFAULT_RESTART_DELAY_MS,
    failure_window_s: int = DEFAULT_FAILURE_WINDOW_S,
    max_failures: int = DEFAULT_MAX_CONSECUTIVE_EXITS,
) -> int:
    if not child_argv:
        return 64
    stop_marker, stop_ack, owner_path = _paths(home)
    owner = _OwnerLock(owner_path)
    if not owner.acquire():
        return 0
    try:
        failures = 0
        while True:
            if _ack_stop(stop_marker, stop_ack):
                return 0
            started = time.monotonic()
            try:
                proc = subprocess.Popen(child_argv)
                rc = int(proc.wait())
            except (OSError, ValueError):
                rc = 70
            if _ack_stop(stop_marker, stop_ack):
                return 0
            if rc in (0, GATEWAY_FATAL_CONFIG_EXIT_CODE):
                return rc
            if time.monotonic() - started >= max(0, failure_window_s):
                failures = 0
            failures += 1
            if failures >= max(1, max_failures):
                return rc
            if _sleep_with_stop(max(0, restart_delay_ms) / 1000.0, stop_marker, stop_ack):
                return 0
    finally:
        owner.release()


def python_supervisor_home(argv: list[str]) -> str | None:
    """Read the owner's home only when Python actually enters this supervisor module."""
    index = 1
    while index < len(argv):
        arg = argv[index]
        if not arg.startswith("-") or arg in {"-", "--", "-m"}:
            break
        # CPython consumes short flags in clusters; W/X consume the rest of
        # their cluster as a value, or the next argv element when empty.
        flags = arg[1:]
        for offset, flag in enumerate(flags):
            if flag in "WX":
                if offset == len(flags) - 1:
                    index += 1
                break
            if flag not in "bBdEiIOPqsSuv":
                return None
        index += 1
    if argv[index:index + 2] != ["-m", "hermes_cli.gateway_windows_supervisor"]:
        return None
    options = argv[index + 2:]
    options = options[:options.index("--")] if "--" in options else options
    # Reuse argparse's option/value grammar, retaining every home occurrence
    # so duplicate ownership claims cannot be hidden by last-value-wins.
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ns = _parser(home_action="append").parse_args(options)
    except SystemExit:
        return None
    return ns.home[0] if len(ns.home) == 1 and not ns.child else None


def _parser(*, home_action: str = "store") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--home", required=True, action=home_action)
    parser.add_argument("--restart-delay-ms", type=int, default=DEFAULT_RESTART_DELAY_MS)
    parser.add_argument("--failure-window-s", type=int, default=DEFAULT_FAILURE_WINDOW_S)
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_CONSECUTIVE_EXITS)
    parser.add_argument("child", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    ns = _parser().parse_args(argv)
    child = list(ns.child)
    if child[:1] == ["--"]:
        child = child[1:]
    return supervise(
        child,
        home=Path(ns.home),
        restart_delay_ms=ns.restart_delay_ms,
        failure_window_s=ns.failure_window_s,
        max_failures=ns.max_failures,
    )


if __name__ == "__main__":
    raise SystemExit(main())
