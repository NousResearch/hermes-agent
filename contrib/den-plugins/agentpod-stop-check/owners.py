"""External owner evidence for the stop-check.

The fleet's supervisor workers are bounded **external processes** (``pi``,
``hermes``, scripts) spawned through the runtime's own background-process tool.
They are represented on disk in the runtime's process registry checkpoint
(``$HERMES_HOME/processes.json``, written by ``tools/process_registry.py``),
NOT as kanban claims — which is why a kanban-only attendance rule reported a
provably running worker as idle.

This module reads that registry **read-only** and converts an entry into
evidence only when BOTH of the following hold:

1. **The row is bound to the card by structure, never by prose.** Either the
   spawn recorded the card the process runs under (``kanban_task_id`` — the
   child's own ``HERMES_KANBAN_TASK`` pin, which the runtime's kanban tools
   already read as that worker's task scope), or the
   row's ``cwd`` is inside the workspace path the *board row itself* carries —
   and only when that workspace path belongs to exactly one card in the sweep.
   A free-text ``command`` is never a binding: supervisor and reviewer prompts
   routinely name cards they must NOT touch ("preserve t_X", "don't touch
   t_Y"), and a shared checkout is not ownership of every card that ever
   pointed at it. With no structured binding the card is ``owner_unknown`` —
   actionable, never quiet.

2. **The identity verifies exactly.** The pid still exists AND its kernel
   start time still equals the value recorded at spawn, decided by the
   runtime's own PID-reuse guard (``ProcessRegistry._host_pid_is_ours``) so the
   plugin can never accept a ``(pid, start)`` pair the runtime rejects. A row
   with no recorded start time is liveness-only and is NOT usable evidence.

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
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

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
    identity_verified: bool         # start time EXACTLY matches the spawn record
    deadline_at: Optional[float]    # wall-clock deadline of the bounded run
    completion_handle: str          # how the exit re-enters the conversation ("" = none)
    binding: str = ""               # WHY this row owns the card (structured only)

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
            f"(bound by {self.binding or 'nothing'}, start-time exact-verified, "
            f"running {age}s, {comp})"
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


def _verify(pid: Optional[int], expected_start: Any) -> tuple[bool, bool]:
    """(alive, identity_verified) — EXACT equality, delegated to the runtime.

    There is no tolerance window. The previous ±200 fudge was justified by a
    record/live gap on one live registry row; controlled fixtures on this
    platform show ``get_process_start_time`` is bit-stable and equal to the
    fork time for both the direct-spawn and the ``sh -c 'gtimeout …'`` shapes,
    read from t+0.00 to t+4.0s and across separate interpreters. The gap was
    therefore never a derivation artifact, so a window could only launder a
    genuinely mismatched record into "verified".

    The decision is made by ``ProcessRegistry._host_pid_is_ours`` — the same
    guard the runtime uses before it will signal a pid — so this plugin can
    never call an identity verified that the runtime itself rejects. One place
    is deliberately STRICTER: the runtime degrades to bare liveness when no
    start time was recorded, but evidence that silences a board must not, so a
    missing baseline is unverified (``owner_unknown`` -> a bounded
    qualification step, never quiet).
    """
    if not pid:
        return (False, False)
    try:
        from gateway.status import _pid_exists
    except Exception:  # pragma: no cover - import guard
        return (False, False)
    try:
        alive = bool(_pid_exists(int(pid)))
    except Exception:
        return (False, False)
    if not alive:
        return (False, False)
    if expected_start in (None, ""):
        # No baseline recorded: liveness only. A recycled pid must not pass.
        return (True, False)
    try:
        from tools.process_registry import ProcessRegistry

        return (
            True,
            bool(ProcessRegistry._host_pid_is_ours(int(pid), int(expected_start))),
        )
    except Exception:
        pass
    try:  # runtime guard unavailable: same primitive, same exact comparison
        from gateway.status import get_process_start_time

        live = get_process_start_time(int(pid))
    except Exception:
        live = None
    if live is None:
        return (True, False)
    return (True, int(live) == int(expected_start))


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


# Workspace kinds whose recorded path is a real per-card checkout the
# dispatcher/kernel resolved. Anything else (or an unset path) is not a
# workspace relationship we will treat as ownership.
CANONICAL_WORKSPACE_KINDS = frozenset({"worktree", "dir", "scratch"})


def _norm(path: Any) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    return os.path.normpath(os.path.expanduser(raw)).rstrip(os.sep) or os.sep


def _within(child: str, parent: str) -> bool:
    """True when ``child`` is ``parent`` or a directory inside it."""
    if not child or not parent:
        return False
    if child == parent:
        return True
    return child.startswith(parent.rstrip(os.sep) + os.sep)


def ambiguous_workspaces(tasks: Iterable[Any]) -> frozenset:
    """Workspace paths claimed by more than one card in the sweep.

    A path several cards point at (observed on the real board: five unfinished
    cards sharing one workspace directory, and every shared repo checkout) says
    nothing about which card a process in it is working — so it is not a
    binding for ANY of them.
    """
    seen: dict[str, int] = {}
    for t in tasks or []:
        p = _norm(getattr(t, "workspace_path", None))
        if p:
            seen[p] = seen.get(p, 0) + 1
    return frozenset(p for p, n in seen.items() if n > 1)


def binding_for(entry: dict, task: Any, *, ambiguous: frozenset = frozenset()) -> str:
    """Why this registry row owns ``task`` — "" when nothing structured says so.

    Only two authoritative relationships, both recorded by the runtime/board
    rather than written into a prompt:

    * ``registry kanban pin`` — the spawn recorded the card the process runs
      under, read from the child's own ``HERMES_KANBAN_TASK`` env pin and
      persisted by ``tools/process_registry.py`` as ``kanban_task_id``. That is
      the same value the runtime's kanban tools use to scope a worker and to
      refuse mutations of any other card, so it is a real binding, not a label
      this plugin invented. Exact match; the rollout/sandbox ``task_id`` on the
      same row is NOT a card and is never consulted.
    * ``canonical workspace`` — the process cwd is inside the workspace path
      the BOARD row carries, and that path belongs to this card alone.

    ``command`` is never consulted.
    """
    tid = str(getattr(task, "id", "") or "")
    if not tid:
        return ""
    if str(entry.get("kanban_task_id") or "").strip() == tid:
        return "registry kanban pin"
    kind = str(getattr(task, "workspace_kind", "") or "").strip().lower()
    ws = _norm(getattr(task, "workspace_path", None))
    if ws and kind in CANONICAL_WORKSPACE_KINDS and ws not in ambiguous:
        if _within(_norm(entry.get("cwd")), ws):
            return "canonical workspace"
    return ""


def evidence_from_entry(
    entry: dict,
    *,
    default_max_runtime: int = 3600,
    binding: str = "",
) -> OwnerEvidence:
    """Verified owner evidence for one registry row (identity checked here)."""
    pid = entry.get("pid")
    alive, verified = _verify(pid, entry.get("host_start_time"))
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
        binding=binding,
    )


def owners_for_task(
    task: Any,
    entries: list[dict],
    *,
    default_max_runtime: int = 3600,
    ambiguous: frozenset = frozenset(),
) -> list[OwnerEvidence]:
    """Structurally-bound owner evidence for one card (may be empty)."""
    out = []
    for e in entries or []:
        bound = binding_for(e, task, ambiguous=ambiguous)
        if not bound:
            continue
        out.append(
            evidence_from_entry(
                e, default_max_runtime=default_max_runtime, binding=bound
            )
        )
    return out
