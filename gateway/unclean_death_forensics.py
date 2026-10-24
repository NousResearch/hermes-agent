"""Forensics for an unclean gateway death — host-OOM vs external SIGKILL.

Kanban t_1915c9a3, finding C1(a).

The gateway lifecycle ledger detects that the previous life "exited UNCLEANLY" — no exit
path ran. That statement is true but not actionable: it cannot tell anyone WHY. On
2026-10-23 the gateway pid 11533 stopped heartbeating at 22:18:34 and never came back.
The ledger could only say "exited UNCLEANLY, suspected_oom=False". The strongest working
hypothesis at the time was a host OOM under exhausted swap.

Reconstructing that from the box after the fact showed the hypothesis was WRONG: the
kernel journal for the whole incident window (21:00–01:00) contains no OOM record at
all, while the box DID produce cgroup OOM kills at 01:53 and 02:08 — so the kernel was
perfectly capable of writing those records, and simply did not kill anything at 22:18.
The 22:18 death was external. Nothing in the gateway could have told us that at the
time; this module exists so the NEXT one can be classified.

The signals, in order of decisiveness:

1. **cgroup ``memory.events`` ``oom_kill`` counter** — the kernel's own count of
   processes it OOM-killed in this cgroup. A non-zero counter means the kernel killed
   something here. The gateway's own unit cgroup is the decisive one; the scope cgroups
   of sibling processes are recorded too, because on a busy box the victim is often a
   sibling whose death starved the gateway.
2. **The kernel OOM log** (``journalctl -k`` / ``dmesg``) filtered to the window around
   the last heartbeat. A record naming the gateway's pid is conclusive; a record naming
   a DIFFERENT pid is evidence of a memory-pressure environment even when the gateway
   was not the victim.
3. **Swap and memory state** at the last heartbeat — the context the death happened in.
   Exhausted swap with plentiful ``mem_available`` is the signature of a memory-pressure
   environment where the OOM killer, not the gateway, made the decision.
4. **The last N heartbeats**, not just the last one. One heartbeat is a snapshot; a
   trend shows whether memory was climbing (leak) or flat (spike), which separates
   "the gateway was consuming the host's memory" from "something else killed the gateway".

Everything here is best-effort and never raises: forensics that break the next boot are
worse than no forensics.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# How many heartbeat records to keep. The heartbeat file holds ONE record (the latest),
# so the history has to come from the append-only journal or the exit-diag log.
_DEFAULT_HEARTBEAT_HISTORY = 10

# How far back to look for kernel OOM records around the last heartbeat. The kernel
# writes its OOM record at kill time, so a window that brackets the last heartbeat is
# enough — but a generous window catches a kill that landed slightly before the
# heartbeat went stale (the heartbeat can lag the kill by up to its own interval).
_KERNEL_LOG_WINDOW_SECONDS = 1800

# The kernel's OOM line looks like:
#   Memory cgroup out of memory: Killed process 7731 (hermes) total-vm:2929268kB, ...
#   oom-kill:constraint=CONSTRAINT_MEMCG,...oom_memcg=/user.slice/...scope,task=hermes,pid=7731
_OOM_LINE_RE = re.compile(r"(?:Killed process|OOM-Killed|oom-kill:)", re.IGNORECASE)
_OOM_PID_RE = re.compile(r"(?:Killed process\s+|task=)\s*(\d+)", re.IGNORECASE)
_OOM_MEMCG_RE = re.compile(r"oom_memcg=([^,\s]+)")
_OOM_CONSTRAINT_RE = re.compile(r"constraint=([^,\s]+)")


def _read_text(path: Path, limit: int = 64 * 1024) -> str:
    """Read at most ``limit`` bytes from the END of a file (the tail is what matters)."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - limit))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _run(cmd: List[str], timeout: float = 10.0) -> str:
    """Run a command and return its stdout (empty on any failure)."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout or ""


def read_cgroup_memory_events(cgroup_path: Optional[str] = None) -> Dict[str, int]:
    """The kernel's own OOM counters for a cgroup, as ``{"oom_kill": 0, ...}``.

    ``memory.events`` is cgroup-v2. Its ``oom_kill`` counter increments exactly when the
    kernel OOM-killed a process in that cgroup. A non-zero value is the strongest single
    piece of evidence that the death was a host OOM; a zero value (with the file present)
    is equally strong evidence that it was NOT.

    The gateway's own unit cgroup is the default. ``None`` when cgroup-v2 is unavailable
    (cgroup v1, a non-Linux host, or no permission) — which is NOT the same as "zero
    kills", so the caller must not read it as such.
    """
    if cgroup_path is None:
        cgroup_path = _gateway_control_group()
    if not cgroup_path:
        return {}
    events: Dict[str, int] = {}
    for line in _read_text(Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "memory.events").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                events[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return events


def _gateway_control_group() -> Optional[str]:
    """The gateway unit's cgroup path (as systemd reports it), or ``None``.

    Read from systemd rather than assumed: the unit name is configurable, and a wrong
    guess would silently report the ROOT cgroup's counters — which are zero when the
    gateway's own cgroup was the one that OOMed, exactly inverting the conclusion.

    The unit name is PROFILE-dependent (``hermes-gateway`` for the default home,
    ``hermes-gateway-<profile>`` otherwise), so this probes the plausible names and
    takes the first that exists. A forensics reader running under one profile must not
    conclude "no OOM" merely because the live gateway runs under another profile's unit.
    """
    candidates: List[str] = []
    try:
        from hermes_cli.gateway import get_service_name

        candidates.append(get_service_name())
    except Exception:
        pass
    candidates.extend(["hermes-gateway.service", "hermes-gateway-friday.service"])
    seen = set()
    for unit in candidates:
        if not unit or unit in seen:
            continue
        seen.add(unit)
        out = _run(["systemctl", "--user", "show", unit, "-p", "ControlGroup", "--value"])
        cgroup = out.strip()
        if cgroup and cgroup != "/":
            return cgroup
    return None


def read_kernel_oom_records(since_iso: Optional[str] = None, pid: Optional[int] = None) -> List[Dict[str, Any]]:
    """Kernel OOM records around the death, newest last.

    Two sources are tried: the kernel journal (``journalctl -k``, systemd hosts) and
    ``dmesg`` (everything else). Each matching line is parsed for the victim pid, the
    cgroup and the constraint, so a reader can tell "the kernel killed THIS process"
    from "the kernel killed something else while this host was short on memory".

    Empty list means "no OOM records found" — which, on a host whose journal is intact
    and where OOM kills demonstrably DO get recorded, is itself evidence against the
    host-OOM hypothesis.
    """
    records: List[Dict[str, Any]] = []
    for text in _kernel_oom_sources(since_iso):
        for line in text.splitlines():
            if not _OOM_LINE_RE.search(line):
                continue
            record: Dict[str, Any] = {"line": line.strip()[:500]}
            match = _OOM_PID_RE.search(line)
            if match:
                try:
                    record["victim_pid"] = int(match.group(1))
                except ValueError:
                    pass
            memcg = _OOM_MEMCG_RE.search(line)
            if memcg:
                record["oom_memcg"] = memcg.group(1)
            constraint = _OOM_CONSTRAINT_RE.search(line)
            if constraint:
                record["constraint"] = constraint.group(1)
            if pid is not None and record.get("victim_pid") == pid:
                record["names_gateway"] = True
            records.append(record)
    return records


def _kernel_oom_sources(since_iso: Optional[str]) -> List[str]:
    """Text of the kernel OOM lines from whichever kernel-log source exists."""
    texts: List[str] = []
    if since_iso:
        texts.append(_run([
            "journalctl", "-k", "--since", since_iso, "--no-pager", "--output", "short-iso",
        ]))
        # dmesg has no time filter; the tail is the recent window.
    texts.append(_run(["dmesg", "-T"]))
    return [t for t in texts if t]


def read_heartbeat_history(limit: int = _DEFAULT_HEARTBEAT_HISTORY) -> List[Dict[str, Any]]:
    """The last N heartbeat payloads, oldest first.

    The heartbeat FILE holds only the newest record, so the history comes from the
    append-only history file the heartbeat maintains alongside it
    (``state/gateway.heartbeat.history``). That file survives an unclean death, which is
    the whole point: after the death there is one heartbeat file (the last beat) and no
    journal record of the beats before it.

    The journal is consulted as a fallback for a gateway that ran before the history file
    existed.

    Reading them back gives the memory TREND — climbing memory points at the gateway as
    the cause of host pressure, flat memory points elsewhere.

    Returns an empty list when the history is unavailable; never raises.
    """
    records: List[Dict[str, Any]] = []
    try:
        from gateway.shutdown_watchdog import get_loop_heartbeat_history_path

        history_path = get_loop_heartbeat_history_path()
    except Exception:
        history_path = None
    if history_path is not None:
        try:
            for line in history_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if isinstance(payload, dict) and isinstance(payload.get("mem"), dict):
                    records.append(payload)
        except OSError:
            pass
    if records:
        return records[-limit:]
    for unit in _candidate_gateway_units():
        out = _run([
            "journalctl", "--user", "-u", unit, "--no-pager", "--output", "cat",
            "-n", str(max(50, limit * 20)),
        ])
        for line in out.splitlines():
            line = line.strip()
            if not line.startswith("{") or '"mem"' not in line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("mem"), dict):
                records.append(payload)
    return records[-limit:]


def _candidate_gateway_units() -> List[str]:
    """Plausible gateway unit names, most specific first.

    The unit name is profile-dependent, and the gateway that died may have run under a
    DIFFERENT profile's unit than the reader. Probing all of them is the difference
    between a memory trend and no trend at all.
    """
    candidates: List[str] = []
    try:
        from hermes_cli.gateway import get_service_name

        candidates.append(get_service_name())
    except Exception:
        pass
    candidates.extend(["hermes-gateway.service", "hermes-gateway-friday.service"])
    seen, out = set(), []
    for unit in candidates:
        if unit and unit not in seen:
            seen.add(unit)
            out.append(unit)
    return out


def _memory_trend(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise the memory trend across the heartbeat history."""
    rss = [
        rec.get("mem", {}).get("rss_kib")
        for rec in records
        if isinstance(rec.get("mem"), dict) and isinstance(rec.get("mem", {}).get("rss_kib"), int)
    ]
    if len(rss) < 2:
        return {}
    first, last = rss[0], rss[-1]
    return {
        "rss_kib_first": first,
        "rss_kib_last": last,
        "rss_kib_delta": last - first,
        "rss_kib_max": max(rss),
        "samples": len(rss),
        "trend": "rising" if last - first > 0 else "flat_or_falling",
    }


def classify_unclean_death(
    *, prior_pid: Optional[int] = None, last_heartbeat_at: Optional[str] = None,
    last_heartbeat_mem: Optional[Dict[str, Any]] = None, home: Optional[Path] = None,
) -> Dict[str, Any]:
    """Gather the forensic evidence and give a best-effort VERDICT.

    The verdict is a reading, not a proof, and says so:

    * ``"host_oom"``        — the kernel's own cgroup counters or kernel log name this
                              process as the victim. Conclusive.
    * ``"host_memory_pressure"`` — the host was under memory pressure (swap exhausted, or
                              the kernel OOM-killed a SIBLING) but nothing says this
                              process was the victim. The environment was hostile; the
                              killer is still unidentified.
    * ``"external_kill"``   — the host was healthy: no OOM record for this process, no
                              OOM kill in the unit's cgroup, memory available. Something
                              outside the gateway sent the signal.
    * ``"indeterminate"``   — the evidence needed to decide is not available (no cgroup
                              v2, no kernel log, no heartbeat). Never guess.

    The default is ``indeterminate``. Claiming a cause we cannot support is worse than
    admitting we do not know — the whole point of this module is that the old ledger's
    silence was mistaken for a verdict.
    """
    evidence: Dict[str, Any] = {
        "captured_at": time.time(),
        "prior_pid": prior_pid,
    }

    cgroup_events = read_cgroup_memory_events()
    if cgroup_events:
        evidence["cgroup_memory_events"] = cgroup_events

    kernel_records = read_kernel_oom_records(since_iso=last_heartbeat_at, pid=prior_pid)
    if kernel_records:
        evidence["kernel_oom_records"] = kernel_records[:20]

    heartbeat_history = read_heartbeat_history()
    if heartbeat_history:
        evidence["heartbeat_history"] = heartbeat_history
        trend = _memory_trend(heartbeat_history)
        if trend:
            evidence["memory_trend"] = trend

    if isinstance(last_heartbeat_mem, dict):
        evidence["last_heartbeat_mem"] = last_heartbeat_mem

    evidence["verdict"] = _verdict(
        evidence, prior_pid=prior_pid, kernel_records=kernel_records,
        cgroup_events=cgroup_events, last_heartbeat_mem=last_heartbeat_mem,
    )
    return evidence


def _verdict(
    evidence: Dict[str, Any], *, prior_pid: Optional[int], kernel_records: List[Dict[str, Any]],
    cgroup_events: Dict[str, int], last_heartbeat_mem: Optional[Dict[str, Any]],
) -> str:
    """Pick the verdict from the collected evidence."""
    # 1. The kernel named THIS process as its OOM victim.
    if any(rec.get("names_gateway") for rec in kernel_records):
        return "host_oom"
    # 2. The kernel OOM-killed something in the gateway's own cgroup.
    if int(cgroup_events.get("oom_kill", 0) or 0) > 0:
        return "host_oom"

    # The host was under pressure but nothing identifies the gateway as the victim.
    swap_used = (last_heartbeat_mem or {}).get("swap_used_kib")
    swap_total = _swap_total_kib()
    swap_exhausted = (
        isinstance(swap_used, int) and isinstance(swap_total, int) and swap_total > 0
        and swap_used / swap_total > 0.9
    )
    if kernel_records or swap_exhausted:
        return "host_memory_pressure"

    # No OOM evidence at all.
    if kernel_records is not None and cgroup_events:
        # We had a cgroup counter to read (zero kills) and a kernel log to read (no
        # matching records) — that is positive evidence of an external kill, not just
        # absence of evidence.
        if not kernel_records and int(cgroup_events.get("oom_kill", 0) or 0) == 0:
            return "external_kill"
    return "indeterminate"


def _swap_total_kib() -> Optional[int]:
    """Total swap in KiB, from ``/proc/meminfo`` (``None`` when unavailable)."""
    try:
        with open("/proc/meminfo", encoding="utf-8-sig") as fh:
            for line in fh:
                if line.startswith("SwapTotal:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def format_verdict(evidence: Dict[str, Any]) -> str:
    """A one-line human summary of the verdict and its evidence."""
    verdict = evidence.get("verdict", "indeterminate")
    if verdict == "host_oom":
        return ("The kernel OOM-killed this gateway (cgroup oom_kill counter and/or a kernel "
                "OOM record naming it). Host out-of-memory.")
    if verdict == "host_memory_pressure":
        return ("The host was under memory pressure (swap exhausted and/or the kernel "
                "OOM-killed a sibling process), but nothing identifies the gateway itself as "
                "the victim. The killer is still unidentified.")
    if verdict == "external_kill":
        return ("No host OOM evidence: the unit's cgroup reports zero OOM kills and the "
                "kernel log holds no OOM record for this process. The gateway was killed "
                "from outside (SIGKILL by another process or the host).")
    return ("The evidence needed to classify this death is not available (no cgroup-v2 "
            "counters, no kernel OOM record, no heartbeat). The cause remains unknown.")


def append_forensics(evidence: Dict[str, Any], home: Optional[Path] = None) -> None:
    """Append the forensic record to the gateway's exit-diag log (best-effort)."""
    try:
        from gateway.lifecycle_ledger import _append_exit_diag

        _append_exit_diag({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tag": "gateway.unclean_death_forensics",
            "pid": os.getpid(),
            **evidence,
        }, home)
    except Exception:
        logger.debug("Failed to append unclean-death forensics", exc_info=True)
