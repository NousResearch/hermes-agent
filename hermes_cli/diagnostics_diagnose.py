"""``hermes debug diagnose`` — host-side companion to the desktop hitch capture.

The desktop app already exports a *sanitized* bundle of its own (renderer /
main / gateway rings, see ``apps/desktop/electron/diagnostics-export.ts``).
This module adds the two things that bundle cannot contain because they live
outside the app process (U4 / KTD6):

1. **A Hermes process tree.** Which Hermes-ish processes exist, and who is
   whose parent — the question "is a second gateway running?" is unanswerable
   from inside one Electron app. It follows the SAME sanitization rule as the
   desktop manifest: pid, ppid and executable BASENAME only. Command lines are
   never read, let alone written; on this machine they routinely carry tokens,
   worktree paths and session ids.

2. **An optional, time-bounded WPR (Windows Performance Recorder) trace.**
   WPR answers questions no in-process ring can — kernel scheduling, GPU work,
   other processes stealing the CPU. It is also, for exactly that reason, a
   SYSTEM-WIDE capture: the ETL contains other applications' activity, full
   image paths and command lines.

Therefore the ETL is **not** part of any sanitized bundle. It is written to a
separate ``unsafe-to-share/`` subdirectory with a README saying so, and only
when the invocation explicitly opts in with ``--wpr``. Everything degrades: no
``wpr.exe`` (non-Windows, N-edition, unelevated) means the trace is skipped and
the process tree is still written.

Nothing here uploads anything. ``hermes debug share`` is the upload path and it
does not know about these files.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# Executables whose basename marks a process as part of the Hermes tree. The
# match is on the BASENAME only — never the command line — so the selection
# itself cannot read a secret.
_NAME_HINTS = ("hermes",)

# WPR's stock profile. `GeneralProfile` is the broad "why is this machine
# slow" recording; a caller who knows what they want can pass another.
DEFAULT_WPR_PROFILE = "GeneralProfile"

# How long to record by default. Long enough to contain a reproduced hitch,
# short enough that the ETL stays a few hundred MB rather than a few GB.
DEFAULT_WPR_SECONDS = 30

# Hard ceiling on ANY single wpr invocation. `wpr -stop` on a large trace is
# genuinely slow (it merges), so this is generous — but it is a ceiling, not a
# guideline: a wpr that blows through it is killed, because a diagnostics tool
# that itself hangs is worse than no diagnostics tool.
DEFAULT_WPR_TIMEOUT_S = 300

_UNSAFE_DIR_NAME = "unsafe-to-share"

_UNSAFE_README = """\
UNSANITIZED — DO NOT SHARE
==========================

This directory holds a Windows Performance Recorder (WPR) ETL trace.

A WPR trace is a SYSTEM-WIDE kernel capture. It records every process on this
machine for the duration of the recording, including:

  * full executable paths and COMMAND LINES (which routinely carry API keys,
    tokens and file paths)
  * file and registry activity of unrelated applications
  * window titles and, depending on the profile, network endpoints

Nothing in here has been sanitized, redacted or reviewed.

The sanitized diagnostics bundle exported by the desktop app (sizes, counts and
durations only) is the artifact that is safe to attach to a bug report. This
directory is NOT. Open the ETL locally in Windows Performance Analyzer, or
delete it.
"""


# ── process tree ─────────────────────────────────────────────────────


def basename_only(value: Any) -> str:
    """Reduce an executable path or name to its bare basename.

    Mirrors ``executableBasename`` in the desktop exporter, including the
    whitespace rule: anything past the first token is argv that arrived
    disguised as a path, and is dropped.
    """
    text = str(value or "").strip()
    if not text:
        return "unknown"
    tail = text.replace("\\", "/").rsplit("/", 1)[-1]
    return (tail.split()[0] if tail.split() else "unknown")[:128]


def _iter_psutil_processes() -> Iterable[dict[str, Any]]:
    """Default process source: pid/ppid/name/exe, and deliberately nothing else.

    ``cmdline`` is not requested. That is the whole point.
    """
    try:
        import psutil
    except Exception:
        return []

    found: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "ppid", "name", "exe"]):
        try:
            info = proc.info
        except Exception:
            continue
        found.append(
            {
                "pid": info.get("pid"),
                "ppid": info.get("ppid"),
                "name": info.get("name"),
                "exe": info.get("exe"),
            }
        )
    return found


def collect_process_tree(
    iter_processes: Callable[[], Iterable[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Snapshot the Hermes process tree, sanitized to {pid, ppid, name}.

    Selection is "any process whose executable basename looks like Hermes",
    plus each match's ancestors (so a launcher/shell parent is visible) and its
    direct children (so a spawned gateway or Electron helper is visible).
    Entries are REBUILT rather than filtered, so a psutil field this function
    never asked for cannot leak through.
    """
    source = iter_processes or _iter_psutil_processes
    try:
        raw = list(source())
    except Exception:
        return []

    by_pid: dict[int, dict[str, Any]] = {}
    children: dict[int, list[int]] = {}
    for entry in raw:
        try:
            pid = int(entry.get("pid"))
        except (TypeError, ValueError):
            continue
        ppid = entry.get("ppid")
        try:
            ppid = int(ppid)
        except (TypeError, ValueError):
            ppid = 0
        by_pid[pid] = {
            "pid": pid,
            "ppid": ppid,
            "name": basename_only(entry.get("exe") or entry.get("name")),
        }
        children.setdefault(ppid, []).append(pid)

    seeds = {pid for pid, entry in by_pid.items() if any(hint in entry["name"].lower() for hint in _NAME_HINTS)}

    selected: set[int] = set()
    for pid in seeds:
        selected.add(pid)
        # Ancestors, with a visited guard: a pid/ppid map read from a live
        # machine is a snapshot of a moving target and can contain a cycle.
        cursor = by_pid[pid]["ppid"]
        while cursor in by_pid and cursor not in selected:
            selected.add(cursor)
            cursor = by_pid[cursor]["ppid"]
        selected.update(children.get(pid, []))

    return [by_pid[pid] for pid in sorted(selected)]


# ── WPR ──────────────────────────────────────────────────────────────


def find_wpr(which: Callable[[str], str | None] | None = None) -> str | None:
    """Locate ``wpr.exe``, or None when this host cannot trace.

    PATH first, then the canonical System32 location — a non-elevated shell on
    a stock Windows install has wpr.exe on PATH but may still fail to start a
    session, which is handled as a run-time degradation, not a lookup failure.
    """
    lookup = which or shutil.which
    found = lookup("wpr")
    if found:
        return found
    system_root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT")
    if system_root:
        candidate = Path(system_root) / "system32" / "wpr.exe"
        if candidate.exists():
            return str(candidate)
    return None


class CommandResult:
    """Outcome of one bounded child process."""

    __slots__ = ("returncode", "output", "timed_out")

    def __init__(self, returncode: int | None, output: str, timed_out: bool) -> None:
        self.returncode = returncode
        self.output = output
        self.timed_out = timed_out

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run_bounded(
    cmd: list[str],
    timeout_s: float,
    popen: Callable[..., Any] | None = None,
) -> CommandResult:
    """Run *cmd* with a HARD timeout, killing it rather than waiting forever.

    ``subprocess.run(timeout=...)`` would do the killing itself, but doing it
    here keeps the kill observable (and testable) and lets the caller see that
    the timeout — not a non-zero exit — is what happened.
    """
    spawn = popen or subprocess.Popen
    proc = spawn(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = proc.communicate(timeout=timeout_s)
        return CommandResult(proc.returncode, output or "", False)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            output, _ = proc.communicate(timeout=10)
        except Exception:
            output = ""
        return CommandResult(None, output or "", True)


def run_wpr_trace(
    out_dir: Path,
    *,
    profile: str = DEFAULT_WPR_PROFILE,
    seconds: int = DEFAULT_WPR_SECONDS,
    timeout_s: float = DEFAULT_WPR_TIMEOUT_S,
    wpr_path: str | None = None,
    popen: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Record a bounded WPR trace into *out_dir*.

    Returns a status dict — never raises for an unavailable or failing wpr.
    ``status`` is one of ``recorded`` / ``skipped`` / ``failed`` / ``timeout``.

    On any failure after ``-start`` the session is cancelled, because a WPR
    session left running keeps buffering system-wide until the machine reboots.
    """
    exe = wpr_path or find_wpr()
    if not exe:
        return {
            "status": "skipped",
            "reason": "wpr.exe not found (non-Windows host, or WPT not installed)",
        }

    pause = sleep or time.sleep
    out_dir.mkdir(parents=True, exist_ok=True)
    etl = out_dir / "trace.etl"

    start = run_bounded([exe, "-start", profile, "-filemode"], timeout_s, popen=popen)
    if not start.ok:
        # The overwhelmingly common cause is "not elevated" (wpr requires an
        # administrator session); the second is a session already running.
        return {
            "status": "timeout" if start.timed_out else "failed",
            "phase": "start",
            "reason": _first_line(start.output) or "wpr -start failed (elevation required?)",
        }

    try:
        pause(max(0, int(seconds)))
    except KeyboardInterrupt:
        run_bounded([exe, "-cancel"], timeout_s, popen=popen)
        raise

    stop = run_bounded([exe, "-stop", str(etl)], timeout_s, popen=popen)
    if not stop.ok:
        run_bounded([exe, "-cancel"], timeout_s, popen=popen)
        return {
            "status": "timeout" if stop.timed_out else "failed",
            "phase": "stop",
            "reason": _first_line(stop.output) or "wpr -stop failed",
        }

    return {
        "status": "recorded",
        "profile": profile,
        "seconds": int(seconds),
        # Basename only, consistent with everything else this module writes;
        # the caller already knows the directory it asked for.
        "file": etl.name,
    }


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:400]
    return ""


# ── command ──────────────────────────────────────────────────────────


def default_output_dir() -> Path:
    from hermes_constants import get_hermes_home

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return get_hermes_home() / "diagnostics" / f"diagnose-{stamp}"


def run_diagnose(args) -> int:
    """``hermes debug diagnose`` entry point. Returns a process exit code."""
    out_dir = Path(getattr(args, "out", None) or default_output_dir()).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    tree = collect_process_tree()
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system().lower(),
        # pid/ppid/basename only — same rule as the desktop bundle manifest.
        "process_tree": tree,
        "wpr": {"status": "skipped", "reason": "not requested (pass --wpr)"},
    }

    if getattr(args, "wpr", False):
        report["wpr"] = run_wpr_trace(
            out_dir / _UNSAFE_DIR_NAME,
            profile=getattr(args, "wpr_profile", None) or DEFAULT_WPR_PROFILE,
            seconds=getattr(args, "wpr_seconds", None) or DEFAULT_WPR_SECONDS,
            timeout_s=getattr(args, "timeout", None) or DEFAULT_WPR_TIMEOUT_S,
        )
        if report["wpr"]["status"] == "recorded":
            unsafe = out_dir / _UNSAFE_DIR_NAME
            unsafe.mkdir(parents=True, exist_ok=True)
            (unsafe / "README.txt").write_text(_UNSAFE_README, encoding="utf-8")

    report_path = out_dir / "diagnose.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"✓ Process tree: {len(tree)} Hermes-related process(es)")
    for entry in tree:
        print(f"    {entry['pid']:>7}  parent {entry['ppid']:>7}  {entry['name']}")

    wpr = report["wpr"]
    if wpr["status"] == "recorded":
        print(f"✓ WPR trace: {wpr['seconds']}s of {wpr['profile']}")
        print(f"  ⚠ UNSANITIZED — {out_dir / _UNSAFE_DIR_NAME}")
        print("    A WPR ETL is a system-wide kernel capture (other processes'")
        print("    command lines and paths included). Do NOT attach it to a bug")
        print("    report. See README.txt in that directory.")
    elif wpr["status"] == "skipped":
        print(f"⏱ WPR trace skipped — {wpr.get('reason', 'unavailable')}")
    else:
        print(f"✗ WPR trace {wpr['status']} during {wpr.get('phase', '?')} — {wpr.get('reason', '')}", file=sys.stderr)

    print(f"\nWrote {report_path}")
    print("The desktop app's Settings → Diagnostics section exports the sanitized")
    print("capture bundle that pairs with this report.")

    # A failed optional trace is not a failed diagnose: the process tree, which
    # is the part that always works, was written.
    return 0
