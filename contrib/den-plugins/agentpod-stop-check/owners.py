"""External owner evidence for the stop-check.

The fleet's supervisor workers are bounded **external processes** (``pi``,
``hermes``, scripts) spawned through the runtime's own background-process tool.
They are represented on disk in the runtime's process registry checkpoint
(``$HERMES_HOME/processes.json``, written by ``tools/process_registry.py``),
NOT as kanban claims — which is why a kanban-only attendance rule reported a
provably running worker as idle.

This module reads that registry **read-only** and converts an entry into
evidence only when the identity actually verifies:

* the pid still exists, AND
* its kernel start time still equals the start time recorded at spawn
  (``host_start_time``) — so a recycled pid can never masquerade as the owner.

Two properties are reported separately and never conflated:

* **liveness** — the owner process exists right now. This is NOT progress: we
  do not observe tool calls, output, or board movement.
* **deadline** — every bounded worker carries one (``gtimeout N`` / ``timeout N``
  in its command line, else the configured ceiling). A live owner past its
  deadline is a finding, not attendance.

A **completion handle** (``notify_on_complete`` / a watcher interval) is what
makes the owner's exit re-enter the supervisor's conversation. An owner with no
completion handle is still live, but that fact is stated in the evidence so a
"someone will tell me when it lands" assumption is never implicit.
"""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# ``gtimeout 2700 pi --print ...`` / ``timeout 30m hermes ...``
_TIMEOUT_RE = re.compile(r"^(?:g?timeout)$")
_DURATION_RE = re.compile(r"^(\d+)([smhd]?)$")
_UNIT_SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


@dataclass
class OwnerEvidence:
    """One verified external owner of a board card."""

    handle: str                     # process-registry session id (the poll/wait handle)
    pid: int
    command: str
    cwd: str
    started_at: float
    alive: bool                     # pid exists
    identity_verified: bool         # start time still matches the spawn record
    deadline_at: Optional[float]    # wall-clock deadline of the bounded run
    completion_handle: str          # how the exit re-enters the conversation ("" = none)

    @property
    def usable(self) -> bool:
        """Evidence at all: the process we recorded is the process running now."""
        return bool(self.alive and self.identity_verified)

    def overdue_by(self, now: float) -> Optional[int]:
        if self.deadline_at is None:
            return None
        late = int(now - self.deadline_at)
        return late if late > 0 else None

    def describe(self, now: float) -> str:
        age = int(now - (self.started_at or now))
        comp = self.completion_handle or "NO completion handle"
        return (
            f"external owner {self.handle} pid {self.pid} "
            f"(start-time verified, running {age}s, {comp})"
        )


def parse_deadline(command: str, started_at: float, default_max_runtime: int) -> Optional[float]:
    """Wall-clock deadline for a bounded external run.

    Prefers the run's own ``timeout``/``gtimeout`` bound; falls back to the
    configured ceiling so an unbounded owner cannot be attended forever.
    """
    try:
        parts = shlex.split(command or "")
    except ValueError:
        parts = (command or "").split()
    for idx, tok in enumerate(parts[:-1]):
        base = tok.rsplit("/", 1)[-1]
        if _TIMEOUT_RE.match(base):
            m = _DURATION_RE.match(parts[idx + 1])
            if m:
                secs = int(m.group(1)) * _UNIT_SECONDS.get(m.group(2), 1)
                return float(started_at or 0) + secs
    if default_max_runtime and started_at:
        return float(started_at) + int(default_max_runtime)
    return None


def _verify(pid: Optional[int], expected_start: Any, tolerance: int = 200) -> tuple[bool, bool]:
    """(alive, identity_verified) using the runtime's own pid helpers.

    ``tolerance`` absorbs the small, observed drift between the start-time
    fingerprint recorded at spawn and the one read back later (measured on
    macOS/psutil: a consistent 1.00s offset on live, non-recycled workers —
    exact equality would have declared a running owner dead). It is ~2s in the
    same unit both backends use (centiseconds / clock ticks), which is far
    below the gap a genuinely recycled pid produces.
    """
    if not pid:
        return (False, False)
    try:
        from gateway.status import _pid_exists, get_process_start_time
    except Exception:  # pragma: no cover - import guard
        return (False, False)
    try:
        alive = bool(_pid_exists(int(pid)))
    except Exception:
        return (False, False)
    if not alive:
        return (False, False)
    if expected_start in (None, ""):
        # Registry row predates start-time recording: liveness only, identity
        # unverified -> not usable as evidence (a recycled pid must not pass).
        return (True, False)
    try:
        live = get_process_start_time(int(pid))
    except Exception:
        live = None
    if live is None:
        return (True, False)
    return (True, abs(int(live) - int(expected_start)) <= int(tolerance))


def load_registry(path: Optional[Path] = None) -> tuple[list[dict], Optional[str]]:
    """Read the process-registry checkpoint. Returns (entries, error)."""
    if path is None:
        try:
            from hermes_constants import get_hermes_home

            path = Path(get_hermes_home()) / "processes.json"
        except Exception:  # pragma: no cover - import guard
            try:
                from tools.process_registry import CHECKPOINT_PATH as path  # type: ignore
            except Exception as exc:
                return ([], f"process registry unavailable: {exc}")
    p = Path(path)
    if not p.exists():
        # An absent checkpoint is a real state (no external workers recorded),
        # not an error — but it is reported so "no owner" is never silent.
        return ([], None)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return ([], f"process registry unreadable: {type(exc).__name__}: {exc}")
    return ([e for e in data if isinstance(e, dict)], None)


def _mentions(entry: dict, task_id: str) -> bool:
    """Does this registry row belong to ``task_id``?

    Derived from the board row's own id (no hardcoded task list): the id must
    appear as a delimited token in the worker's cwd (worktree path), its command
    (prompt/receipt path), or the registry's own task/session key.
    """
    if not task_id:
        return False
    needle = re.compile(rf"(?<![0-9A-Za-z_]){re.escape(task_id)}(?![0-9A-Za-z_])")
    for field in ("cwd", "command", "task_id", "session_key"):
        if needle.search(str(entry.get(field) or "")):
            return True
    return False


def evidence_from_entry(
    entry: dict, *, default_max_runtime: int = 3600, start_time_tolerance: int = 200
) -> OwnerEvidence:
    """Verified owner evidence for one registry row (identity checked here)."""
    pid = entry.get("pid")
    alive, verified = _verify(pid, entry.get("host_start_time"), start_time_tolerance)
    started_at = float(entry.get("started_at") or 0)
    completion = []
    if entry.get("notify_on_complete"):
        completion.append("notify_on_complete")
    if entry.get("watcher_interval"):
        completion.append(f"watcher={entry.get('watcher_interval')}s")
    return OwnerEvidence(
        handle=str(entry.get("session_id") or "?"),
        pid=int(pid or 0),
        command=str(entry.get("command") or ""),
        cwd=str(entry.get("cwd") or ""),
        started_at=started_at,
        alive=alive,
        identity_verified=verified,
        deadline_at=parse_deadline(
            str(entry.get("command") or ""), started_at, default_max_runtime
        ),
        completion_handle="+".join(completion),
    )


def owners_for_task(
    task_id: str,
    entries: list[dict],
    *,
    default_max_runtime: int = 3600,
    start_time_tolerance: int = 200,
) -> list[OwnerEvidence]:
    """Verified owner evidence for one card (may be empty)."""
    return [
        evidence_from_entry(
            e,
            default_max_runtime=default_max_runtime,
            start_time_tolerance=start_time_tolerance,
        )
        for e in entries
        if _mentions(e, task_id)
    ]
