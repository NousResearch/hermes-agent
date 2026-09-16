"""agentpod-stop-check — board reconciliation used by the supervisor stop gate.

Pure-ish evaluation layer: reads the kanban board through the *installed*
``hermes_cli.kanban_db`` interface (no raw SQL lifecycle writes, read-only) and
decides whether a supervision turn is allowed to conclude "no material change".

Design constraints (from the canonical card):

* Derive ALL current unfinished cards from the actual board, not from the one
  card the supervisor happened to look at.
* A blocked/idle card is *attended* only with observable structured evidence:
  a live canonical executor (active run + fresh heartbeat + live pid + unexpired
  claim), a future checkpoint marker that names a real wake path, or a specific
  recorded human/external gate (typed block kind, or an explicit gate marker).
* Comments are NOT execution proof. A repeated "still working on it" comment
  can never make a card attended.
* A refused/empty board read is an explicit error, never "no work".
* Bounded: findings are capped, no loops, no dispatch, no writes.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Statuses that still owe the project an outcome.
UNFINISHED_STATUSES = frozenset(
    {"triage", "todo", "scheduled", "ready", "running", "blocked", "review"}
)
# Typed block kinds that mean a real human/external gate (kanban_db.VALID_BLOCK_KINDS).
DEFAULT_HUMAN_GATE_KINDS = ("needs_input", "capability")

# Structured evidence markers a worker/supervisor records as a task comment.
#   STOP-CHECK-CHECKPOINT: 2026-09-17T09:00:00Z wake=kanban-wake owner=software-engineer
#   STOP-CHECK-GATE: user must authorise the $X spend
CHECKPOINT_RE = re.compile(
    r"STOP-CHECK-CHECKPOINT:\s*(?P<when>\S+)(?P<rest>[^\n]*)", re.IGNORECASE
)
GATE_RE = re.compile(r"STOP-CHECK-GATE:\s*(?P<what>[^\n]+)", re.IGNORECASE)
WAKE_RE = re.compile(r"wake=(?P<wake>[A-Za-z0-9_.:-]+)")

# Wake mechanisms that actually exist in this deployment. A checkpoint that
# names nothing here has no real wake path and is therefore not attended.
DEFAULT_REAL_WAKES = ("kanban-wake", "dispatcher", "cron", "event", "webhook")

KIND_STALE_CLAIM = "stale_claim"
KIND_STALE_HOLD = "stale_hold"
KIND_OVERDUE_CHECKPOINT = "overdue_checkpoint"
KIND_NO_WAKE = "checkpoint_without_wake"
KIND_UNOWNED = "unowned_blocker"
KIND_IDLE = "idle_card"
KIND_OWNER_STOPPED = "owner_stopped"


@dataclass
class Finding:
    """One unattended unfinished card plus the concrete continuation."""

    task_id: str
    title: str
    status: str
    assignee: Optional[str]
    kind: str
    detail: str
    next_action: str

    def line(self) -> str:
        who = self.assignee or "UNASSIGNED"
        return (
            f"- {self.task_id} [{self.status}/{who}] {self.kind}: {self.detail}\n"
            f"    -> next: {self.next_action}"
        )


@dataclass
class Attended:
    task_id: str
    status: str
    reason: str
    detail: str


@dataclass
class Verdict:
    ok: bool
    error: Optional[str] = None
    board: str = ""
    unfinished: int = 0
    findings: list[Finding] = field(default_factory=list)
    attended: list[Attended] = field(default_factory=list)
    truncated: int = 0

    @property
    def quiet_allowed(self) -> bool:
        """True only when the board read succeeded AND nothing is unattended."""
        return bool(self.ok) and not self.findings

    def fingerprint(self) -> str:
        if not self.ok:
            return f"error:{self.error}"
        return "|".join(f"{f.task_id}:{f.kind}" for f in self.findings) or "clear"


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        # Exists but owned by another uid.
        return True
    except Exception:
        return False


def _parse_ts(raw: str) -> Optional[int]:
    raw = (raw or "").strip().rstrip(",")
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    text = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


@dataclass
class _Markers:
    checkpoint_at: Optional[int] = None
    checkpoint_wake: Optional[str] = None
    gate: Optional[str] = None


def _scan_markers(comments) -> _Markers:
    """Latest structured markers. Free-form comment prose is ignored."""
    out = _Markers()
    for c in comments or []:
        body = getattr(c, "body", "") or ""
        m = CHECKPOINT_RE.search(body)
        if m:
            when = _parse_ts(m.group("when"))
            if when is not None:
                out.checkpoint_at = when
                w = WAKE_RE.search(m.group("rest") or "")
                out.checkpoint_wake = w.group("wake") if w else None
        g = GATE_RE.search(body)
        if g:
            out.gate = g.group("what").strip()
    return out


def _classify(
    task,
    *,
    run,
    markers: _Markers,
    now: int,
    cfg: dict,
    parents_unfinished: bool,
) -> tuple[Optional[Finding], Optional[Attended]]:
    heartbeat_stale = int(cfg.get("heartbeat_stale_seconds", 900))
    human_kinds = tuple(cfg.get("human_gate_kinds", DEFAULT_HUMAN_GATE_KINDS))
    real_wakes = tuple(cfg.get("real_wakes", DEFAULT_REAL_WAKES))
    owner = task.assignee
    tid = task.id
    title = (task.title or "")[:100]

    def find(kind, detail, action):
        return (
            Finding(tid, title, task.status, owner, kind, detail, action),
            None,
        )

    def ok(reason, detail):
        return (None, Attended(tid, task.status, reason, detail))

    # 1. A specific recorded human/external gate stays gated — never actionable.
    if markers.gate:
        return ok("human_gate", f"recorded gate: {markers.gate[:120]}")
    # ``block_kind`` survives an unblock, so only trust it in the two parked
    # statuses it actually describes.
    if task.status in ("blocked", "scheduled") and (task.block_kind or "") in human_kinds:
        return ok("human_gate", f"typed block kind '{task.block_kind}'")

    # 2. A verifiably progressing canonical executor.
    active = run is not None and getattr(run, "ended_at", None) is None
    if active:
        hb = getattr(run, "last_heartbeat_at", None) or getattr(run, "started_at", 0)
        age = now - int(hb or 0)
        pid = getattr(run, "worker_pid", None)
        expires = getattr(run, "claim_expires", None)
        alive = _pid_alive(pid)
        expired = expires is not None and int(expires) < now
        if age <= heartbeat_stale and alive and not expired:
            return ok(
                "progressing_executor",
                f"run {run.id} pid {pid} heartbeat {age}s ago",
            )
        why = []
        if age > heartbeat_stale:
            why.append(f"heartbeat {age}s stale")
        if not alive:
            why.append(f"worker pid {pid} not alive")
        if expired:
            why.append("claim expired")
        return find(
            KIND_STALE_CLAIM,
            "claim looks live but is not: " + ", ".join(why),
            f"verify/reclaim {tid} for its existing owner "
            f"({owner or 'unassigned'}); do not spawn a second worker",
        )

    # 3. Owner's run ended while the card is still unfinished -> handoff.
    #    ``blocked``/``scheduled`` are excluded: block_task/schedule_task close
    #    the run too, and those cards are classified by their park evidence
    #    (rules 3b-6), not as a stopped worker.
    if (
        task.status not in ("blocked", "scheduled")
        and run is not None
        and getattr(run, "ended_at", None) is not None
    ):
        outcome = getattr(run, "outcome", None) or getattr(run, "status", "ended")
        if not markers.checkpoint_at:
            return find(
                KIND_OWNER_STOPPED,
                f"last run {run.id} ended ({outcome}) but card is still {task.status}",
                f"hand {tid} back to the SAME owner ({owner or 'assign one'}) "
                f"with the run summary; no duplicate worker",
            )

    # 3b. ``scheduled`` is an explicit time-park that the dispatcher will NOT
    #     pick up (kanban_db.schedule_task: "intentionally not dispatchable").
    #     Without a recorded checkpoint+wake or a gate it is an unattended
    #     hold, not a wait.
    if task.status == "scheduled" and not markers.checkpoint_at:
        return find(
            KIND_STALE_HOLD,
            "parked in 'scheduled' (not dispatchable) with no checkpoint, "
            "no wake and no recorded gate",
            f"give {tid} a checkpoint with a supported wake, resume it for its "
            f"owner ({owner or 'assign one'}), or record an explicit blocker",
        )

    # 4. Checkpoint evidence.
    if markers.checkpoint_at:
        if markers.checkpoint_at <= now:
            return find(
                KIND_OVERDUE_CHECKPOINT,
                f"checkpoint {markers.checkpoint_at} is overdue by "
                f"{now - markers.checkpoint_at}s with no newer run",
                f"diagnose {tid} now (logs/executor/dependency); "
                f"do not simply extend the deadline",
            )
        if (markers.checkpoint_wake or "") not in real_wakes:
            return find(
                KIND_NO_WAKE,
                f"future checkpoint has no real wake path "
                f"(wake={markers.checkpoint_wake!r})",
                f"attach a supported wake (one of {', '.join(real_wakes)}) to {tid} "
                f"or act on it now",
            )
        return ok(
            "future_checkpoint",
            f"checkpoint in {markers.checkpoint_at - now}s via {markers.checkpoint_wake}",
        )

    # 5. Dependency block behind a still-unfinished parent that is itself
    #    covered elsewhere in this same sweep.
    if task.status == "blocked" and (task.block_kind or "") == "dependency" and parents_unfinished:
        return ok("dependency", "waiting on an unfinished parent card in this sweep")

    # 6. Everything else is unattended work for the supervisor.
    if task.status == "blocked":
        kind = KIND_STALE_HOLD if task.block_kind else KIND_UNOWNED
        return find(
            kind,
            f"blocked (kind={task.block_kind or 'untyped'}) with no live executor, "
            "no future checkpoint and no recorded human gate",
            f"resolve or re-dispatch {tid} on its existing card "
            f"(owner {owner or 'needs one'}), or record an explicit blocker",
        )
    return find(
        KIND_IDLE,
        f"status {task.status} with no live executor, checkpoint or gate",
        f"route {tid} to its owner ({owner or 'assign one'}) or record "
        f"an explicit blocker/checkpoint",
    )


def evaluate_board(
    *,
    board: Optional[str] = None,
    db_path: Optional[str] = None,
    now: Optional[int] = None,
    cfg: Optional[dict] = None,
    kb: Any = None,
) -> Verdict:
    """Read the board and decide whether a quiet conclusion is permitted.

    Read-only. Any failure (missing board, locked DB, import failure, zero rows)
    yields ``ok=False`` — a refused read must never be reported as "no work".
    """
    cfg = dict(cfg or {})
    now = int(now if now is not None else time.time())
    max_findings = int(cfg.get("max_findings", 10))
    if kb is None:
        try:
            from hermes_cli import kanban_db as kb  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            return Verdict(ok=False, error=f"kanban_db import failed: {exc}")

    conn = None
    try:
        from pathlib import Path

        conn = kb.connect(Path(db_path)) if db_path else kb.connect(board=board)
        tasks = kb.list_tasks(conn, include_archived=False)
        if not tasks:
            return Verdict(
                ok=False,
                board=str(board or db_path or ""),
                error="board read returned zero cards — treat as an unreadable "
                "board, not as an empty backlog",
            )
        unfinished = [t for t in tasks if t.status in UNFINISHED_STATUSES]
        unfinished_ids = {t.id for t in unfinished}

        findings: list[Finding] = []
        attended: list[Attended] = []
        for task in unfinished:
            run = kb.latest_run(conn, task.id)
            markers = _scan_markers(kb.list_comments(conn, task.id))
            try:
                parents = kb.parent_ids(conn, task.id)
            except Exception:
                parents = []
            parents_unfinished = any(p in unfinished_ids for p in parents)
            f, a = _classify(
                task,
                run=run,
                markers=markers,
                now=now,
                cfg=cfg,
                parents_unfinished=parents_unfinished,
            )
            if f is not None:
                findings.append(f)
            if a is not None:
                attended.append(a)
    except Exception as exc:
        return Verdict(
            ok=False,
            board=str(board or db_path or ""),
            error=f"board read failed: {type(exc).__name__}: {exc}",
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    truncated = max(0, len(findings) - max_findings)
    return Verdict(
        ok=True,
        board=str(board or db_path or ""),
        unfinished=len(unfinished),
        findings=findings[:max_findings],
        attended=attended,
        truncated=truncated,
    )


def render_report(verdict: Verdict, *, continuation: bool = False) -> str:
    """Explicit, fail-loud text. Never claims an action was taken."""
    if not verdict.ok:
        return (
            "SUPERVISOR STOP-CHECK ERROR — the board could not be reconciled, so "
            "this turn may NOT conclude 'no material change' or 'done'.\n"
            f"board: {verdict.board or '(unresolved)'}\n"
            f"error: {verdict.error}\n"
            "Required: fix/repeat the board read, then re-assess. A refused read "
            "is not evidence of no work."
        )
    head = (
        "SUPERVISOR STOP-CHECK — whole-board reconciliation says this turn may NOT "
        "conclude 'no material change'.\n"
        f"board: {verdict.board} | unfinished cards: {verdict.unfinished} | "
        f"unattended: {len(verdict.findings) + verdict.truncated} | "
        f"attended: {len(verdict.attended)}"
    )
    body = "\n".join(f.line() for f in verdict.findings)
    tail = ""
    if verdict.truncated:
        tail += f"\n(+{verdict.truncated} more unattended card(s) not shown)"
    if verdict.attended:
        shown = "; ".join(f"{a.task_id}={a.reason}" for a in verdict.attended[:8])
        tail += f"\nattended (no action): {shown}"
    if continuation:
        tail += (
            "\nAct on the first item now on its existing canonical card: no new "
            "duplicate card, no duplicate worker, no gate bypass. If it truly "
            "cannot be advanced, record the specific blocker or a checkpoint with "
            "a supported wake."
        )
    else:
        tail += (
            "\nThis is a runtime OUTPUT gate: it proves the quiet conclusion was "
            "blocked, NOT that any of the above was executed."
        )
    return f"{head}\n{body}{tail}"
