"""Shutdown forensics — capture context when the gateway receives SIGTERM/SIGINT.

``shutdown_signal_handler`` runs synchronously inside the asyncio loop, so
:func:`snapshot_shutdown_context` is a fast (<10ms) non-blocking probe and
:func:`spawn_async_diagnostic` is a fire-and-forget ``ps`` walk in a detached
subprocess. Anything that waits belongs in the async helper, never in the probe.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.restart import DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT, resolve_systemd_timeout_stop_sec
import contextlib

_SIGNAL_NAME_BY_NUM: Dict[int, str] = {
    int(getattr(signal, _name)): _name
    for _name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2")
    if getattr(signal, _name, None) is not None
}


def _signal_name(sig: Any) -> str:
    """Human-readable signal name (``str(sig)`` as fallback)."""
    if sig is None:
        return "UNKNOWN"
    try:
        sig_int = int(sig)
    except (TypeError, ValueError):
        return str(sig)
    return _SIGNAL_NAME_BY_NUM.get(sig_int, f"signal#{sig_int}")


def _read_proc_field(pid: int, key: str) -> Optional[str]:
    """Read a single field from /proc/<pid>/status.  Linux only; None elsewhere."""
    with contextlib.suppress(OSError), open(f"/proc/{pid}/status", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip()
    return None


def _proc_summary_psutil(pid: int, summary: Dict[str, Any]) -> None:
    """Fill ``summary`` from psutil (no /proc — macOS, BSD, Windows). Mutates in place.

    Kept to attribute lookups on an already-constructed ``Process``: measured ~0.1ms on macOS,
    well inside the signal handler's <10ms budget. Never raises — a dead or unreadable parent
    leaves the fields absent, exactly as the /proc path does.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        return
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            for key, getter in (("name", proc.name), ("ppid", proc.ppid)):
                with contextlib.suppress(Exception):  # noqa: BLE001 — signal-handler path
                    summary[key] = getter()
            with contextlib.suppress(Exception):  # noqa: BLE001
                summary["uid"] = str(proc.uids().real)
            with contextlib.suppress(Exception):  # noqa: BLE001
                # truncate aggressively — these can be 4KB
                summary["cmdline"] = " ".join(proc.cmdline()).strip()[:300]
    except Exception:  # noqa: BLE001 — NoSuchProcess/AccessDenied/anything: never raise
        return


def _proc_summary(pid: int) -> Dict[str, Any]:
    """Compact process snapshot (pid, ppid, state, uid, cmdline); missing fields omitted.

    Reads /proc on Linux and falls back to psutil elsewhere. Without the fallback the parent
    is ``{"pid": N}`` on macOS/Windows, so ``format_context_for_log`` prints
    ``parent_name=? parent_cmdline='(unknown)'`` — the single most useful field for
    "who killed my gateway" is blank on exactly the hosts where the question gets asked.
    """
    summary: Dict[str, Any] = {"pid": pid}
    if pid <= 0:
        return summary
    for out_key, proc_key in (("name", "Name"), ("state", "State")):
        if (value := _read_proc_field(pid, proc_key)) is not None:
            summary[out_key] = value
    if (ppid := _read_proc_field(pid, "PPid")) is not None:
        with contextlib.suppress(ValueError):
            summary["ppid"] = int(ppid)
    if (uid := _read_proc_field(pid, "Uid")) is not None:
        summary["uid"] = uid.split()[0] if uid else uid  # "real effective saved fs"
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        data = b""
    if data:  # truncate aggressively — these can be 4KB
        summary["cmdline"] = data.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()[:300]
    if "cmdline" not in summary:  # no /proc (macOS/BSD/Windows), or an unreadable entry
        _proc_summary_psutil(pid, summary)
    return summary


def _read_marker(path: Path) -> Optional[str]:
    """Return the marker file's text, or None if absent/unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def snapshot_shutdown_context(received_signal: Any = None) -> Dict[str, Any]:
    """Fast (<10ms) snapshot of who/what is asking us to shut down: signal name/number, own + parent
    /proc summaries, systemd parentage, takeover/planned-stop markers, TracerPid, 1-min load,
    timestamps. Pure stdlib, never raises, never blocks."""
    pid, ppid = os.getpid(), os.getppid()
    ctx: Dict[str, Any] = {
        "ts": time.time(), "ts_monotonic": time.monotonic(),
        "signal": _signal_name(received_signal),
        "signal_num": int(received_signal) if received_signal is not None else None,
        "pid": pid, "ppid": ppid, "parent": _proc_summary(ppid), "self": _proc_summary(pid),
    }
    # INVOCATION_ID is set by systemd units; ppid==1 also suggests systemd forwarded the SIGTERM.
    for ctx_key, env_key in (("systemd_invocation_id", "INVOCATION_ID"),
                             ("systemd_journal_stream", "JOURNAL_STREAM")):
        if os.environ.get(env_key):
            ctx[ctx_key] = os.environ[env_key]
    ctx["under_systemd"] = bool(os.environ.get("INVOCATION_ID")) or ppid == 1
    # High load points at "something crushing the box" rather than an external killer.
    with contextlib.suppress(OSError, AttributeError):
        ctx["loadavg_1m"] = os.getloadavg()[0]
    # Nonzero TracerPid means a debugger/strace is attached.
    with contextlib.suppress(TypeError, ValueError):
        if (tracer := _read_proc_field(pid, "TracerPid")) is not None and tracer != "0":
            ctx["tracer_pid"] = int(tracer) if tracer.isdigit() else tracer
            ctx["tracer"] = _proc_summary(int(tracer)) if tracer.isdigit() else None
    # Race hint: a takeover marker on disk that does NOT name us is a smoking gun for "another
    # --replace instance is killing us". Filenames mirror gateway.status; literals keep the signal-
    # handler path import-light.
    with contextlib.suppress(Exception):  # noqa: BLE001 — never raise from a signal handler
        hermes_home_str = os.environ.get("HERMES_HOME")
        if hermes_home_str:
            raw = _read_marker(Path(hermes_home_str) / ".gateway-takeover.json")
            if raw is not None:
                ctx["takeover_marker"] = raw[:300]
                ctx["takeover_marker_for_self"] = (f'"target_pid": {pid}' in raw
                                                   or f"'target_pid': {pid}" in raw)
            raw = _read_marker(Path(hermes_home_str) / ".gateway-planned-stop.json")
            if raw is not None:
                ctx["planned_stop_marker"] = raw[:300]
    return ctx


def resolve_ancestor_chain(pid: int, *, max_depth: int = 12) -> List[Dict[str, Any]]:
    """Ancestor summaries from ``pid`` up to init, nearest first (``pid`` itself included).

    Resolved EAGERLY in the signal handler, not by the detached diagnostic: the gateway exits
    the instant the handler returns, so a subprocess that walks ``/proc/<gateway pid>`` or runs
    ``ps -p <gateway pid>`` finds a dead process and prints nothing. Bounded by ``max_depth``
    and driven by the already-fast ``_proc_summary`` (~0.1ms/level).
    """
    chain: List[Dict[str, Any]] = []
    seen: set[int] = set()
    current = pid
    for _ in range(max_depth):
        if current <= 1 or current in seen:
            break
        seen.add(current)
        summary = _proc_summary(current)
        chain.append(summary)
        parent = summary.get("ppid")
        if not isinstance(parent, int):
            break
        current = parent
    return chain


def _format_ancestor_chain(chain: List[Dict[str, Any]]) -> str:
    """One line per ancestor, shell-safe for embedding in the diagnostic script.

    Control characters are collapsed to spaces: a ``python -c '<script>'`` parent carries literal
    newlines in its cmdline, which would otherwise split one ancestor across many lines and make
    the section unparseable (and its line count meaningless).
    """
    lines = []
    for entry in chain:
        raw = str(entry.get("cmdline") or entry.get("name") or "?")
        cmd = " ".join(raw.split())[:200]
        lines.append(f"pid={entry.get('pid')} ppid={entry.get('ppid', '?')} {cmd}")
    return "\n".join(lines) or "(chain unavailable)"


def _diagnostic_timeout_argv(timeout_seconds: float) -> list[str]:
    """``timeout``-command prefix for the diagnostic, or ``[]`` when none exists.

    GNU coreutils ``timeout`` is absent on stock macOS (it ships as ``gtimeout`` only with
    Homebrew coreutils). Hardcoding ``timeout`` made ``Popen`` raise ``FileNotFoundError`` →
    the whole diagnostic silently returned None, leaving a 0-byte log. Falling back to no
    prefix keeps the snapshot on such hosts; the script's own per-command guards bound it.
    """
    for candidate in ("timeout", "gtimeout"):
        if shutil.which(candidate):
            return [candidate, f"{timeout_seconds:.0f}"]
    return []


def spawn_async_diagnostic(log_path: Path, signal_name: str, *,
                           timeout_seconds: float = 5.0) -> Optional[int]:
    """Fire-and-forget ``ps``-style snapshot appended to ``log_path``: a detached subprocess (own
    ``timeout`` when available so a wedged ``ps`` self-cleans) rather than a blocking ``ps aux`` in
    the signal handler, which can freeze the loop >2s on a busy host. Returns the subprocess PID,
    or ``None`` on failure / Windows (bash -c is available on every POSIX target; Windows has no ps).

    Every probe is platform-guarded with a fallback: BSD ``ps`` (macOS) rejects the GNU ``auxf
    --sort=-pcpu`` spelling outright, and ``pstree``/``/proc``/``dmesg`` do not exist there. A
    diagnostic that emits nothing on the developer's own machine is worse than no diagnostic —
    it looks like the hook never ran.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    if sys.platform == "win32":
        return None
    # Every probe is a `command -v` branch, never `cmd | head -N || fallback`: a pipeline's exit
    # status is the LAST command's, so `pstree ... | head -40 || fallback` reports head's success
    # (0) even when pstree does not exist and the fallback never runs — the section renders empty.
    #
    # The ancestor chain is resolved HERE, in the still-live process, and passed as literal text:
    # by the time the detached child runs, the gateway has exited and any `ps -p <us>` walk finds
    # nothing. This is the field that answers "who killed the gateway", so it must not race.
    chain_text = shlex.quote(_format_ancestor_chain(resolve_ancestor_chain(os.getpid())))
    script = (
        f"echo '=== shutdown diagnostic @ {signal_name} ==='; "
        "echo '--- date ---'; date -u +%Y-%m-%dT%H:%M:%SZ; "
        # GNU `ps auxf --sort=-pcpu` is a hard usage error on BSD/macOS; `ps aux -r` is its
        # sort-by-cpu equivalent. Probe support instead of relying on pipeline exit status.
        "echo '--- ps (top 60 by cpu) ---'; "
        "if ps auxf --sort=-pcpu >/dev/null 2>&1; then ps auxf --sort=-pcpu 2>/dev/null | head -60; "
        "else ps aux -r 2>/dev/null | head -60; fi; "
        f"echo '--- parent chain of self ---'; printf '%s\\n' {chain_text}; "
        "echo '--- loadavg ---'; "
        "if [ -r /proc/loadavg ]; then cat /proc/loadavg; else uptime 2>/dev/null || true; fi; "
        "echo '--- recent kernel/service log (oom/killed) ---'; "
        "if dmesg -T >/dev/null 2>&1; then dmesg -T 2>/dev/null | tail -20; "
        "elif command -v journalctl >/dev/null 2>&1; then journalctl --user -n 20 --no-pager 2>/dev/null | tail -20; "
        "elif command -v log >/dev/null 2>&1; then log show --last 2m --predicate "
        "'eventMessage CONTAINS \"jetsam\" OR eventMessage CONTAINS \"memorystatus\"' 2>/dev/null | tail -20; "
        "fi; "
        "echo '=== end ==='"
    )
    try:  # O_APPEND so concurrent diagnostics from rapid signals don't trample each other
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    except OSError:
        return None
    try:  # start_new_session: outlive systemd killing our cgroup (KillMode=control-group) to flush
        return subprocess.Popen(
            [*_diagnostic_timeout_argv(timeout_seconds), "bash", "-c", script], stdout=fd,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
            close_fds=True).pid
    except OSError:
        return None
    finally:
        with contextlib.suppress(OSError):  # subprocess inherited the fd; drop our handle
            os.close(fd)


def format_context_for_log(ctx: Dict[str, Any]) -> str:
    """Render a shutdown context dict as one scannable log line (parent cmdline is key)."""
    parent = ctx.get("parent") or {}
    load_str = f"{load:.2f}" if isinstance(load := ctx.get("loadavg_1m"), (int, float)) else "?"
    extras: List[str] = []
    if ctx.get("takeover_marker") is not None:
        who = 'self' if ctx.get('takeover_marker_for_self') else 'other'
        extras.append(f"takeover_marker_present={who}")
    if ctx.get("planned_stop_marker") is not None:
        extras.append("planned_stop_marker_present=yes")
    if ctx.get("tracer_pid"):
        extras.append(f"tracer_pid={ctx['tracer_pid']}")
    extras_str = (" " + " ".join(extras)) if extras else ""
    return (
        f"signal={ctx.get('signal', '?')} under_systemd={'yes' if ctx.get('under_systemd') else 'no'} "
        f"parent_pid={parent.get('pid') or '?'} parent_name={parent.get('name') or '?'} "
        f"loadavg_1m={load_str}{extras_str} parent_cmdline={parent.get('cmdline', '(unknown)')!r}"
    )


def context_as_json(ctx: Dict[str, Any]) -> str:
    """JSON-serialise a context dict for structured ingestion.  Never raises."""
    try:
        return json.dumps(ctx, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return "{}"


def check_systemd_timing_alignment(
    drain_timeout: float, cron_drain_timeout: float = DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
) -> Optional[Dict[str, Any]]:
    """At startup, sanity-check that systemd's TimeoutStopSec covers stop. A stale unit file
    (upgraded without re-running ``hermes setup``) can have ``TimeoutStopSec`` below the stop
    budget, so systemd SIGKILLs the cgroup mid-drain (a phantom ``code=killed status=9`` in the
    journal). ``None`` when aligned OR undeterminable (not under systemd, no ``systemctl``);
    otherwise a dict with ``timeout_stop_sec``/``drain_timeout``/``expected_min``/``mismatch``.
    """
    if not os.environ.get("INVOCATION_ID"):
        return None  # Not running under systemd (or at least not directly)
    # /proc/self/cgroup: "0::/user.slice/.../hermes-gateway.service"
    unit_name: Optional[str] = None
    with contextlib.suppress(OSError), open("/proc/self/cgroup", encoding="utf-8") as fh:
        for line in fh:
            parts = reversed(line.strip().split("/"))
            unit_name = next((p for p in parts if p.endswith(".service")), None)
            if unit_name:
                break
    if (timeout_us := _systemd_timeout_stop_us(unit_name) if unit_name else None) is None:
        return None
    timeout_stop_sec = timeout_us / 1_000_000.0
    expected = float(resolve_systemd_timeout_stop_sec(drain_timeout, cron_drain_timeout))
    return {"unit": unit_name, "timeout_stop_sec": timeout_stop_sec, "drain_timeout": drain_timeout,
            "cron_drain_timeout": cron_drain_timeout, "expected_min": expected,
            "mismatch": timeout_stop_sec < expected}


def _systemd_timeout_stop_us(unit_name: str) -> Optional[int]:
    """``TimeoutStopUSec`` of ``unit_name`` in microseconds; ``--user`` first (hermes' usual)."""
    for flag in (["--user"], []):
        try:
            result = subprocess.run(
                ["systemctl", *flag, "show", unit_name, "--property=TimeoutStopUSec"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2.0,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        # Output: "TimeoutStopUSec=1min 30s" or "TimeoutStopUSec=90000000"
        for line in result.stdout.splitlines() if result.returncode == 0 else ():
            if line.startswith("TimeoutStopUSec="):
                value = line.split("=", 1)[1].strip()
                timeout_us = int(value) if value.isdigit() else parse_systemd_duration_to_us(value)
                if timeout_us is not None:
                    return timeout_us
    return None


def parse_systemd_duration_to_us(raw: str) -> Optional[int]:
    """Parse 'TimeoutStopUSec=1min 30s' / '90s' style values to microseconds. Covers us, ms, s, min,
    h, d, w, month, y; a bare number is seconds. None on anything unexpected; never raises. Public: also consumed by
    hermes_cli.gateway's restart-wait sizing.
    """
    if not raw:
        return None
    units = {"us": 1, "ms": 1_000, "s": 1_000_000, "sec": 1_000_000,
             "min": 60_000_000, "h": 3_600_000_000, "hr": 3_600_000_000,
             # Fixed systemd time-util.h constants, not variable calendar months/years.
             "d": 86_400_000_000, "w": 604_800_000_000,
             "month": 2_629_800_000_000, "y": 31_557_600_000_000}
    total_us, token, digits = 0, "", ""

    def _flush() -> bool:  # fold the pending digits/token pair into total_us
        nonlocal total_us, token, digits
        multiplier = units.get(token.lower()) if token else 1_000_000
        if multiplier is None or not digits:
            return False
        try:
            total_us += int(float(digits) * multiplier)
        except (ValueError, OverflowError):
            return False
        digits = token = ""
        return True
    for ch in raw + " ":
        if ch.isdigit() or ch == ".":
            if token and not _flush():  # a digit after a unit ends the previous number
                return None
            digits += ch
        elif ch.isalpha():
            token += ch
        elif digits and not _flush():
            return None
    return total_us if total_us > 0 else None

