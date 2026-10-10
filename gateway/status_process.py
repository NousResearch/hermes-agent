"""Process identity helpers: start-time fingerprints, wall-clock start epochs, and local CLI detection.

Start-time fingerprints (boot-relative, from :func:`gateway.status.get_process_start_time`) tell two
incarnations of a PID apart on the same host; the wall-clock epoch is for anything shown to a person
or compared against real timestamps.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def _start_times_agree(current: Any, *recorded: Any) -> bool:
    """Same process object: all fingerprints > 0 and within 1ms of ``current``; raises on junk."""
    cur = float(current)
    return cur > 0 and all(r > 0 and abs(r - cur) <= 0.001 for r in map(float, recorded))


# Same-host start-time readings can drift by ~1 s between the claim-time and a later liveness read
# (macOS ``kern.boottime`` adjustment, #117505). Both fingerprint scales are ×100 (Linux /proc ticks,
# psutil centiseconds), so 200 means 2 s on either platform — a recycled PID is essentially never
# that close to the original's start time.
START_TIME_DRIFT_TOLERANCE = 200


def start_time_fingerprints_match(recorded: Any, current: Any, tolerance: int = START_TIME_DRIFT_TOLERANCE) -> bool:
    """Liveness-reconciliation comparator for ``gateway.status.get_process_start_time`` fingerprints: the
    recorded owner and the current reading are the same incarnation when they agree within
    ``tolerance``. Raises on junk; callers decide what an unreadable (``None``) side means."""
    return abs(int(current) - int(recorded)) <= tolerance


def detect_active_cli_process(proc_dir: Path = Path("/proc")) -> bool:
    """True if a bare/interactive ``hermes`` CLI (``hermes`` or ``hermes chat``,
    NOT a subcommand like ``gateway``/``dashboard``/``setup``) is currently
    running on THIS host.

    Linux-only, best-effort, and explicitly local-host: it scans
    ``/proc/*/cmdline`` for a process whose argv invokes the installed
    ``hermes`` script (``.../bin/hermes``, matched by basename so it doesn't
    also catch ``hermes-agent``/``hermes-acp``), then reuses
    ``hermes_cli.main``'s OWN subcommand-detection logic
    (``_first_positional_argv`` + ``_BUILTIN_SUBCOMMANDS``) against the
    rest of that process's argv, so "is this a chat session" is defined in
    exactly one place rather than reimplemented here and left to drift.
    A subcommand of ``None`` or ``"chat"`` counts as active; any other
    recognized subcommand (``gateway``, ``dashboard``, ``setup``, ...) does
    not. The gateway/dashboard services themselves are invoked via
    ``python -m hermes_cli.main <subcommand>`` (no ``.../bin/hermes`` in
    their argv at all), so they never match this scan by construction.

    Unlike :func:`_get_process_start_time`, has no non-Linux fallback: this
    is a nice-to-have status indicator, not a correctness-load-bearing
    liveness check, so silently returning ``False`` off Linux (or on any
    read error) is the right amount of effort here. And because it only
    sees processes on the SAME host as the dashboard process, it reports
    nothing for a CLI session running on a different machine or in a
    different container than wherever this code executes — a real
    limitation, not a bug, given what ``/proc`` can see.
    """
    if not proc_dir.is_dir():
        return False

    try:
        from hermes_cli.main import _BUILTIN_SUBCOMMANDS, _first_positional_argv
    except ImportError:
        return False

    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue  # process gone, or not ours to read — skip, not fatal
        if not raw:
            continue
        argv = raw.decode("utf-8", errors="replace").split("\x00")
        if argv and argv[-1] == "":
            argv.pop()  # cmdline is NUL-terminated, trailing empty split artifact

        # Find the "hermes" script token specifically (not hermes-agent /
        # hermes-acp — those are different entry points with their own
        # process shapes, not the interactive CLI).
        hermes_idx = next(
            (i for i, tok in enumerate(argv) if Path(tok).name == "hermes"), None
        )
        if hermes_idx is None:
            continue

        subcommand = _first_positional_argv(argv[hermes_idx + 1 :])
        if subcommand is None or subcommand == "chat":
            return True
        if subcommand not in _BUILTIN_SUBCOMMANDS:
            # Unrecognized token in the position argparse would treat as the
            # subcommand -- most likely a chat message passed positionally
            # (e.g. `hermes "summarize this"`), which is still a chat
            # invocation, not a real subcommand we should ignore.
            return True

    return False


def _get_process_start_epoch(pid: int) -> Optional[int]:
    """Return the process's creation time as a Unix epoch (seconds), or None.

    DISTINCT from ``gateway.status.get_process_start_time``, which returns a
    boot-relative fingerprint (``/proc`` clock ticks since boot) usable only
    for same-host equality comparison — NOT a wall-clock time. Anything that
    needs an actual timestamp (e.g. the Mini App dashboard's gateway-uptime
    display and its ``env_mtime > gateway_start`` restart-needed banner, both
    of which compare against real epoch values) must use THIS, or it ends up
    subtracting clock-ticks-since-boot from a Unix epoch and getting a
    decades-long "uptime" / an always-true banner.

    Linux: ``btime`` (boot epoch, from ``/proc/stat``) plus the process's
    start-ticks-since-boot divided by ``CLK_TCK``. Elsewhere: psutil's
    ``create_time()``, already a float epoch. Returns an int (whole seconds)
    on both paths.
    """
    try:
        start_ticks = int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])
        clk_tck = os.sysconf("SC_CLK_TCK")
        btime = None
        with open("/proc/stat", encoding="utf-8") as f:
            for line in f:
                if line.startswith("btime "):
                    btime = int(line.split()[1])
                    break
        if btime is not None and clk_tck:
            return int(btime + start_ticks / clk_tck)
    except (FileNotFoundError, IndexError, PermissionError, ValueError, OSError):
        pass

    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    try:
        return int(psutil.Process(pid).create_time())
    except (psutil.Error, OSError, ValueError):
        return None


def get_process_start_epoch(pid: int) -> Optional[int]:
    """Public wrapper for the wall-clock process creation epoch."""
    return _get_process_start_epoch(pid)
