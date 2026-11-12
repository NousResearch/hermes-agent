"""Name the programs holding ``apps/desktop/release`` open by working directory (#101789).

Windows refuses to rename a directory that a live process uses as its current working
directory. A resident helper that some Hermes window spawned (an IME cloud service such as
Sogou's ``SogouCloud.exe``, an AV or assistant component) inherits that window's cwd inside
``release/win-unpacked`` and keeps it after the app exits, so the stage-and-swap promotion
fails on every retry although the build succeeded. Its exe lives elsewhere, so the
exe-under-release stop never sees it.

These holders belong to the user, not to Hermes: they are NAMED in the failure and closed
only with the user's consent, given either at an interactive terminal prompt or through the
Desktop's "close it and retry" choice. That choice leaves a consent file in ``release/`` bound
to the hand-off run it starts and to the exact processes (pid + start time) the user saw; the
next swap attempt consumes it whatever its source, and only that run's swap may act on it.
A non-interactive run with no matching consent never closes anything.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# windows.ps1 parses this exact prefix out of the update output (one line per holder).
HOLDER_LINE_PREFIX = "Desktop app folder held by: "

# Written by the Desktop when the user chooses "Close <program> and retry"; removed at the start
# of the next promotion swap attempt, so a consent covers exactly one retry.
CONSENT_FILE_NAME = ".close-holders-consent.json"
# psutil create_time is a float rebuilt from the OS on each read: equal within this tolerance.
_START_TIME_TOLERANCE_S = 0.5

_CLOSE_WAIT_S = 5


@dataclass(frozen=True)
class ReleaseHolder:
    pid: int
    name: str
    exe: str
    create_time: float

    def describe(self) -> str:
        return f"{self.name} (pid {self.pid}, {self.exe or 'path unreadable'})"

    def holder_line(self) -> str:
        """The machine-read line windows.ps1 hands to the Desktop (pid + start time identify it)."""
        return (f"{HOLDER_LINE_PREFIX}{self.name} (pid {self.pid}, started {self.create_time:.3f}, "
                f"{self.exe or 'path unreadable'})")


@dataclass(frozen=True)
class HolderConsent:
    """The processes the user agreed to close, as (pid, start time, normalized exe)."""
    holders: frozenset[tuple[int, float, str]] = frozenset()

    def covers(self, holder: ReleaseHolder) -> bool:
        return bool(holder.exe) and any(
            pid == holder.pid and abs(start - holder.create_time) <= _START_TIME_TOLERANCE_S
            and exe == _norm(holder.exe)
            for pid, start, exe in self.holders
        )


NO_CONSENT = HolderConsent()


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("\\/") + os.sep)


def _release_root(release_dir: Path) -> Optional[str]:
    try:
        return _norm(str(release_dir.resolve()))
    except OSError:
        return None


def _exe_or_empty(proc) -> str:
    """Read only for a matched holder: another process's exe is none of our business."""
    try:
        return proc.exe() or ""
    except Exception:  # health: allow BLE001 -- access denied for a protected process: named without a path
        return ""


def find_release_cwd_holders(folder: Path) -> list[ReleaseHolder]:
    """Every other process whose working directory is ``folder`` (the app folder being renamed,
    e.g. ``release/win-unpacked``) or below it.

    A process whose cwd cannot be read (another user's, access denied) is skipped: it cannot
    be shown to hold the tree. Never raises.
    """
    root = _release_root(folder)
    try:
        import psutil
        procs = psutil.process_iter(["pid", "name"])
    except Exception:  # health: allow BLE001 -- diagnostics on a failure path; no psutil = no names
        return []
    if root is None:
        return []
    me = os.getpid()
    holders: list[ReleaseHolder] = []
    for proc in procs:
        try:
            if proc.pid == me:
                continue
            cwd = proc.cwd()
            if not cwd or not _inside(_norm(cwd), root):
                continue
            holders.append(ReleaseHolder(
                pid=int(proc.pid), name=proc.info.get("name") or "unknown program",
                exe=_exe_or_empty(proc), create_time=float(proc.create_time()),
            ))
        except Exception:  # health: allow BLE001 -- psutil raises per-process (gone, denied, zombie)
            continue
    return holders


def take_consent(release_dir: Path, run_id: Optional[str]) -> HolderConsent:
    """Consume the Desktop's consent at the start of a swap attempt: the file is removed whether
    or not it applies, and it applies only to the hand-off run that the Desktop started for it."""
    path = release_dir / CONSENT_FILE_NAME
    try:
        raw = path.read_bytes()
    except OSError:
        return NO_CONSENT
    with contextlib.suppress(OSError):
        path.unlink()
    try:
        data = json.loads(raw.decode("utf-8-sig"))
        if not run_id or data.get("run_id") != run_id:
            return NO_CONSENT
        holders = frozenset(
            (int(h["pid"]), float(h["start_time"]), _norm(h["exe"]))
            for h in data["holders"] if isinstance(h.get("exe"), str) and h["exe"]
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        return NO_CONSENT
    return HolderConsent(holders)


def _ask_at_terminal(holders: list[ReleaseHolder], input_fn: Callable[[str], str]) -> bool:
    """Only a person at a terminal can answer; a hidden hand-off or a pipe never consents."""
    if not holders or not (sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty()):
        return False
    names = ", ".join(h.describe() for h in holders)
    try:
        answer = input_fn(f"  Close {names} and retry? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in ("y", "yes")


def close_holders(holders: list[ReleaseHolder], folder: Path) -> list[int]:
    """Terminate exactly these holders, each re-checked right before: same pid AND creation
    time (no recycled pid), cwd still inside ``folder``, and not one of our ancestors."""
    root = _release_root(folder)
    try:
        import psutil
        ancestors = {p.pid for p in psutil.Process(os.getpid()).parents()}
    except Exception:  # health: allow BLE001 -- no process table means nothing can be verified
        return []
    targets = []
    for holder in holders:
        try:
            proc = psutil.Process(holder.pid)
            if proc.pid in ancestors or abs(proc.create_time() - holder.create_time) > _START_TIME_TOLERANCE_S:
                continue
            if root is None or not _inside(_norm(proc.cwd()), root):
                continue
            proc.terminate()
            targets.append(proc)
        except Exception:  # health: allow BLE001 -- gone or denied: it is not closed by us
            continue
    if targets:
        with contextlib.suppress(Exception):
            _, alive = psutil.wait_procs(targets, timeout=_CLOSE_WAIT_S)
            for proc in alive:
                with contextlib.suppress(Exception):
                    proc.kill()
            if alive:
                psutil.wait_procs(alive, timeout=_CLOSE_WAIT_S)
    return [int(p.pid) for p in targets if not p.is_running()]


def release_after_refused_rename(
    folder: Path, *, consent: HolderConsent = NO_CONSENT, input_fn: Callable[[str], str] = input,
) -> tuple[list[ReleaseHolder], list[int]]:
    """After Windows refused to rename ``folder``: name every cwd holder inside it, then close
    the ones covered by ``consent`` (taken at the start of this swap attempt) or agreed to at the
    terminal. Returns (holders found, pids closed)."""
    holders = find_release_cwd_holders(folder)
    if not holders:
        return [], []
    # Human lines only: the machine lines go out once, for the holders still in the way when the
    # swap is given up (describe_block), so a holder closed here is not offered to the Desktop again.
    for holder in holders:
        print(f"  {holder.describe()}")
    chosen = [h for h in holders if consent.covers(h)]
    # A holder whose program path cannot be read is usually protected (another user, elevated):
    # it cannot be closed from here either, so the user closes it themselves.
    closable = [h for h in holders if h.exe]
    for holder in holders:
        if not holder.exe:
            print(f"  Close {holder.name} (pid {holder.pid}) yourself: Hermes cannot close it.")
    if not chosen and _ask_at_terminal(closable, input_fn):
        chosen = closable
    if not chosen:
        return holders, []
    closed = close_holders(chosen, folder)
    if closed:
        print(f"  ✓ Closed with your consent: {', '.join(h.describe() for h in chosen if h.pid in closed)}")
    return holders, closed


def describe_block(holders: list[ReleaseHolder], release_dir: Path) -> str:
    """The final report of a refused swap; ``holders`` are the ones still holding the folder."""
    names = ", ".join(h.describe() for h in holders)
    pids = " ".join(f"/PID {h.pid}" for h in holders)
    return (
        "".join(f"  {h.holder_line()}\n" for h in holders)
        + f"✗ Windows would not replace the desktop app: {names} is using a folder inside {release_dir}.\n"
        f"    It is not part of Hermes, so it was left running. Close it (Task Manager, or `taskkill {pids}`)\n"
        "    and run `hermes desktop --force-build`, or run Update now in Hermes Desktop and choose to close it."
    )


def holder_names(holders: list[ReleaseHolder]) -> str:
    return ", ".join(f"{h.name} (pid {h.pid})" for h in holders)
