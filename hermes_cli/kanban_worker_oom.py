"""Was this Kanban worker OOM-killed, or did it fail on the work?

Kanban t_4d12c8eb (2026-10-24).

Two good implementation cards were killed mid-flight on hermes:

    Oct 24 05:56:09 hermes-worker-kanban-t_000bf106-run-33301.scope: Failed with result 'oom-kill'.
    Oct 24 06:01:23 hermes-worker-kanban-t_38551eb5-run-33303.scope: Failed with result 'oom-kill'.

The dispatcher then recorded ``last_failure_error = "pid <n> not alive"`` and tripped
the crash-loop breaker with the event ``gave_up``. That reason string is the
problem: it reads like the worker's CODE was broken, when the implementation was
already substantial and on disk. The next reader (a human or a retry worker) draws
the wrong conclusion and re-does work that was never broken.

Why "not alive" and not "OOM-killed": the dispatcher's reap registry
(``_recent_worker_exits``) only knows children of the dispatcher's OWN process. A
per-tick dispatcher — a fresh ``hermes kanban dispatch`` process — never reaped
the worker, so the registry is empty, the kind stays ``unknown``, and the generic
"not alive" string is used. The worker's own exit trailer is also absent: an
OOM-killed process never reaches its exit epilogue.

The kernel's record is durable and unambiguous, and it names the scope:

    Oct 24 05:56:08 kernel: memory: usage 1048576kB, limit 1048576kB, failcnt 1240786
    Oct 24 05:56:08 kernel: oom-kill:constraint=CONSTRAINT_MEMCG,...,oom_memcg=/user.slice/
        user-1000.slice/user@1000.service/app.slice/hermes-worker-kanban-t_000bf106-run-33301.scope,
        task=hermes,pid=759473,uid=1000
    Oct 24 05:56:08 kernel: Memory cgroup out of memory: Killed process 759473 (hermes)
        total-vm:2712396kB, anon-rss:21732kB, file-rss:29568kB, ...

Two facts fall out of that record and both matter:

1. ``constraint=CONSTRAINT_MEMCG`` means the kill was **cgroup-local**: the worker's
   OWN scope hit ITS OWN ``MemoryMax``. It was not a host-wide OOM and it was not
   the host running out of memory. The distinction decides the fix — a cgroup-local
   kill means the limit was wrong, a host-wide kill means the host was starved.
2. ``oom_memcg=...hermes-worker-kanban-<task>-run-<n>.scope`` names the scope, and
   the scope name carries the task id. That is how a specific card is attributed to
   a specific OOM instead of a host-wide guess.

The scope unit itself is not a reliable source: it is created with ``--collect``, so
systemd removes it as soon as the worker exits. ``systemctl show`` on a collected
scope returns an empty unit. The kernel journal is what survives.

Everything here is best-effort and never raises: forensics that break the next
dispatch tick are worse than no forensics.
"""
from __future__ import annotations

import logging
import re
import subprocess
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)

# The scope unit name carries the task id: ``hermes-worker-kanban-<task_id>-run-<n>``.
# The task id is what links an OOM record back to a specific card.
_SCOPE_TASK_RE = re.compile(r"hermes-worker-kanban-(?P<task_id>[A-Za-z0-9_]+)-run-")

# Kernel OOM lines. The two forms that carry the useful payload:
#   "Memory cgroup out of memory: Killed process 759473 (hermes) total-vm:..., anon-rss:..."
#   "oom-kill:constraint=CONSTRAINT_MEMCG,...,oom_memcg=<path>,task=hermes,pid=759473,..."
_KILLED_PROCESS_RE = re.compile(
    r"Memory cgroup out of memory:\s+Killed process\s+(?P<pid>\d+)\s+\((?P<comm>[^)]*)\)"
    r"(?P<rest>.*)",
    re.IGNORECASE,
)
_OOM_KILL_LINE_RE = re.compile(r"oom-kill:constraint=(?P<constraint>[^,\s]+)", re.IGNORECASE)
_OOM_MEMCG_RE = re.compile(r"oom_memcg=(?P<memcg>[^,\s]+)")
_OOM_TASK_RE = re.compile(r"task=(?P<comm>[^,\s]+),pid=(?P<pid>\d+)")
_MEMCG_LIMIT_RE = re.compile(
    r"memory:\s+usage\s+(?P<usage>\d+)kB,\s+limit\s+(?P<limit>\d+)kB", re.IGNORECASE
)
_ANON_RSS_RE = re.compile(r"anon-rss:(?P<anon>\d+)kB", re.IGNORECASE)
_FILE_RSS_RE = re.compile(r"file-rss:(?P<file>\d+)kB", re.IGNORECASE)
_TOTAL_VM_RE = re.compile(r"total-vm:(?P<vm>\d+)kB", re.IGNORECASE)

# How far back to look for kernel OOM records. The kill lands before the
# dispatcher notices the dead pid, and the dispatcher's crash grace period plus
# the tick interval can be a couple of minutes, so the window is generous.
_KERNEL_LOG_WINDOW_SECONDS = 3600

# Reading the kernel journal is not free (it can be large); cache the answer for
# a short TTL so a burst of deaths in one tick does not re-read it per worker.
_OOM_LOOKUP_TTL_SECONDS = 30.0
_oom_cache: "tuple[float, List[Dict[str, Any]]]" = (0.0, [])


def _run(cmd: List[str], timeout: float = 10.0) -> str:
    """Run a command and return its stdout (empty on any failure)."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout or ""


def _kernel_oom_text(since_epoch: Optional[float]) -> str:
    """Kernel log text containing OOM records, from whichever source exists.

    ``journalctl -k`` is the systemd host's source; ``dmesg`` is the fallback.
    Both are read, because a host can have one without the other and a record in
    either is decisive.
    """
    texts: List[str] = []
    if since_epoch is not None:
        since_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(since_epoch))
        texts.append(_run([
            "journalctl", "-k", "--since", since_iso, "--no-pager", "--output", "short-iso",
        ]))
    # dmesg has no time filter; the tail is the recent window.
    texts.append(_run(["dmesg", "-T"]))
    return "\n".join(t for t in texts if t)


def _parse_oom_records(text: str) -> List[Dict[str, Any]]:
    """Parse kernel OOM lines into structured records.

    A record is emitted per OOM EVENT (not per line): the "Killed process" line
    names the victim and its RSS, and the ``oom-kill:`` line that precedes it
    carries the cgroup and the constraint. They are paired by adjacency, because
    the kernel prints the constraint line first and the kill line immediately
    after.
    """
    records: List[Dict[str, Any]] = []
    pending: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        limit_match = _MEMCG_LIMIT_RE.search(line)
        if limit_match:
            pending = {
                "usage_kb": int(limit_match.group("usage")),
                "limit_kb": int(limit_match.group("limit")),
            }
            continue

        constraint_match = _OOM_KILL_LINE_RE.search(line)
        if constraint_match:
            pending["constraint"] = constraint_match.group("constraint")
            memcg = _OOM_MEMCG_RE.search(line)
            if memcg:
                pending["oom_memcg"] = memcg.group("memcg")
                task_match = _SCOPE_TASK_RE.search(memcg.group("memcg"))
                if task_match:
                    pending["task_id"] = task_match.group("task_id")
            victim = _OOM_TASK_RE.search(line)
            if victim:
                pending["victim_comm"] = victim.group("comm")
                try:
                    pending["victim_pid"] = int(victim.group("pid"))
                except ValueError:
                    pass
            continue

        killed = _KILLED_PROCESS_RE.search(line)
        if killed:
            record: Dict[str, Any] = dict(pending)
            record.setdefault("victim_comm", killed.group("comm"))
            try:
                record["victim_pid"] = int(killed.group("pid"))
            except ValueError:
                pass
            rest = killed.group("rest") or ""
            anon = _ANON_RSS_RE.search(rest)
            if anon:
                record["anon_rss_kb"] = int(anon.group("anon"))
            file_rss = _FILE_RSS_RE.search(rest)
            if file_rss:
                record["file_rss_kb"] = int(file_rss.group("file"))
            vm = _TOTAL_VM_RE.search(rest)
            if vm:
                record["total_vm_kb"] = int(vm.group("vm"))
            records.append(record)
            pending = {}
            continue

        if _OOM_MEMCG_RE.search(line):
            # An oom-kill: line that did not match the constraint pattern (e.g. a
            # different kernel wording) still names the scope — keep what we can.
            memcg = _OOM_MEMCG_RE.search(line)
            if memcg:
                pending["oom_memcg"] = memcg.group("memcg")
                task_match = _SCOPE_TASK_RE.search(memcg.group("memcg"))
                if task_match:
                    pending["task_id"] = task_match.group("task_id")
    return records


def read_oom_records(
    since_epoch: Optional[float] = None,
    *,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Kernel OOM records within the window, oldest first.

    Cached for a short TTL: a tick that reaps several workers reads the journal
    once. The cache is keyed on the window's START, so a moving window does not
    serve a stale answer for a longer lookback than the caller asked for.
    """
    global _oom_cache
    now = time.time()
    window_start = since_epoch if since_epoch is not None else now - _KERNEL_LOG_WINDOW_SECONDS
    if use_cache:
        cached_at, cached_records = _oom_cache
        if cached_records and now - cached_at < _OOM_LOOKUP_TTL_SECONDS:
            if window_start >= cached_at - _KERNEL_LOG_WINDOW_SECONDS:
                return cached_records
    records = _parse_oom_records(_kernel_oom_text(window_start))
    _oom_cache = (now, records)
    return records


def find_worker_oom(
    task_id: Optional[str] = None,
    *,
    worker_pid: Optional[int] = None,
    since_epoch: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """The OOM record that killed this worker, or ``None``.

    Matched on the task id (the scope name carries it) OR on the victim pid.
    The task id is the stronger signal: the pid can be recycled, and the scope
    name is minted fresh for every run.

    ``None`` means "no OOM record found for this worker" — which, on a host whose
    journal is intact and where OOM kills demonstrably DO get recorded, is itself
    evidence that the death was NOT an OOM.
    """
    records = read_oom_records(since_epoch)
    if not records:
        return None
    matches: List[Dict[str, Any]] = []
    for record in records:
        if task_id and record.get("task_id") == task_id:
            matches.append(record)
        elif worker_pid and record.get("victim_pid") == worker_pid:
            matches.append(record)
    if not matches:
        return None
    # The most recent match is the one that killed THIS worker.
    return matches[-1]


def describe_oom(record: Dict[str, Any]) -> str:
    """A human-readable one-liner for a worker OOM.

    Deliberately says what the operator needs to act on: the cgroup limit that was
    hit, and the fact that the kill was cgroup-local (so the FIX is the limit,
    not the host).
    """
    constraint = record.get("constraint") or "UNKNOWN"
    usage_kb = record.get("usage_kb")
    limit_kb = record.get("limit_kb")
    anon_kb = record.get("anon_rss_kb")
    comm = record.get("victim_comm") or "worker"

    parts: List[str] = [
        "worker was OOM-killed by its own systemd scope",
        f"constraint={constraint}",
    ]
    if limit_kb:
        limit_mb = limit_kb // 1024
        if usage_kb:
            parts.append(f"cgroup usage {usage_kb // 1024}MiB of limit {limit_mb}MiB")
        else:
            parts.append(f"cgroup limit {limit_mb}MiB")
    if anon_kb:
        parts.append(f"victim {comm} anon-rss {anon_kb // 1024}MiB")
    parts.append("this is an infrastructure failure, not a failure of the work")
    return "; ".join(parts)
