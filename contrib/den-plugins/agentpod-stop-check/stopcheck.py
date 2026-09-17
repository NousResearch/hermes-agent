"""agentpod-stop-check — board reconciliation used by the supervisor stop gate.

Read-only evaluation layer. Reads the kanban board through the *installed*
``hermes_cli.kanban_db`` interface and decides whether a supervision turn is
allowed to conclude "no material change".

Evidence rules (each one exists because its absence produced a false verdict):

* **Owners are processes, not prose.** Attendance needs a verified owner: a
  kanban run whose pid is alive with a fresh heartbeat and unexpired claim, or
  a process-registry row that is **structurally bound** to the card (registry
  ``task_id``, or a cwd inside the workspace the board row itself records and
  that no other card shares) AND whose pid + kernel start time match exactly,
  decided by the runtime's own PID-reuse guard (``owners.py``). A card id
  appearing in a command line is never a binding; comments are never execution
  proof, and a recent comment is never liveness.
* **Liveness is not progress, and liveness alone is not silence.** Verified
  owners are reported as *live*, with that word. A live owner buys quiet only
  when its exit will re-enter the conversation — a completion handle on the
  process, or a verified wake recorded on the card. Otherwise it is a bounded
  ``owner_without_wake`` finding.
* **A dead owner cannot be hidden by a marker.** If a recorded owner is gone,
  the card is `owner_stopped` regardless of any future checkpoint someone wrote.
* **A wake must name a target that exists.** ``wake=cron`` as a bare word is not
  a wake; ``wake=cron:<job_id>`` is verified against the real job store, and a
  dispatcher wake is verified against the card actually being dispatchable.
* **A gate must be authorised and current.** A hold written by the card's own
  worker does not authorise a human wait; an expired or aged-out gate needs
  requalification, not permanent immunity — and requalification never means
  performing the gated action.
* **Unknown is explicit, never quiet.** No evidence either way yields
  `owner_unknown`, whose action is a bounded qualification step.
* **A refused or empty board read is an error**, never "no work" — and so is a
  configured scope matching zero of N cards, which would otherwise disable the
  sweep silently.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

try:  # package load (PluginManager) / direct-file load (tests)
    from . import escalation as escalation_mod
    from . import owners as owners_mod
except ImportError:  # pragma: no cover - direct-file load
    import escalation as escalation_mod  # type: ignore
    import owners as owners_mod  # type: ignore

# Statuses that still owe the project an outcome.
UNFINISHED_STATUSES = frozenset(
    {"triage", "todo", "scheduled", "ready", "running", "blocked", "review"}
)
# Typed block kinds that describe a real human/external gate.
DEFAULT_HUMAN_GATE_KINDS = ("needs_input", "capability")
# Statuses the dispatcher will actually pick up (kanban_db: 'scheduled' is
# "intentionally not dispatchable").
DISPATCHABLE_STATUSES = frozenset({"triage", "todo", "ready"})

#   STOP-CHECK-CHECKPOINT: 2026-09-17T09:00:00Z wake=cron:board-sweep
#   STOP-CHECK-GATE: user must authorise the $X spend until=2026-09-20T00:00:00Z
CHECKPOINT_RE = re.compile(
    r"STOP-CHECK-CHECKPOINT:\s*(?P<when>\S+)(?P<rest>[^\n]*)", re.IGNORECASE
)
GATE_RE = re.compile(r"STOP-CHECK-GATE:\s*(?P<what>[^\n]+)", re.IGNORECASE)
GATE_RESOLVED_RE = re.compile(r"STOP-CHECK-GATE-RESOLVED\b", re.IGNORECASE)
WAKE_RE = re.compile(r"wake=(?P<wake>[A-Za-z0-9_.:/-]+)")
UNTIL_RE = re.compile(r"until=(?P<until>\S+)")

KIND_STALE_CLAIM = "stale_claim"
KIND_STALE_HOLD = "stale_hold"
KIND_OVERDUE_CHECKPOINT = "overdue_checkpoint"
KIND_NO_WAKE = "unverified_wake"
KIND_UNOWNED = "unowned_blocker"
KIND_IDLE = "idle_card"
KIND_OWNER_STOPPED = "owner_stopped"
KIND_OWNER_UNKNOWN = "owner_unknown"
KIND_OWNER_OVERDUE = "owner_overdue"
KIND_OWNER_NO_WAKE = "owner_without_wake"
KIND_UNQUALIFIED_GATE = "unqualified_gate"
KIND_ESCALATION_ROUTABLE = "escalation_routable"
KIND_ESCALATION_STALE_REVIEW = "escalation_stale_review"
KIND_ESCALATION_ACCESS = "escalation_access_blocker"

DEFAULT_MAX_GATE_AGE = 3 * 86400        # a human gate must be re-confirmed
DEFAULT_MAX_HOLD_AGE = 3 * 86400        # a typed hold must be re-qualified
DEFAULT_MAX_OWNER_RUNTIME = 3600        # ceiling for an unbounded external owner


@dataclass
class Finding:
    """One unattended unfinished card plus the concrete next step."""

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

    def short(self, width: int = 110) -> str:
        line = f"- {self.task_id} {self.kind}: {self.next_action}"
        return line if len(line) <= width else line[: width - 1].rstrip() + "…"


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
    scope: str = ""
    unfinished: int = 0
    findings: list[Finding] = field(default_factory=list)
    attended: list[Attended] = field(default_factory=list)
    truncated: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def quiet_allowed(self) -> bool:
        """True only when the board read succeeded AND nothing is unattended."""
        return bool(self.ok) and not self.findings

    def fingerprint(self) -> str:
        if not self.ok:
            return f"error:{self.error}"
        return "|".join(f"{f.task_id}:{f.kind}" for f in self.findings) or "clear"


# --------------------------------------------------------------- helpers ---

def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True  # exists, owned by another uid
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
class _Gate:
    what: str
    author: str
    written_at: int
    until: Optional[int]


@dataclass
class _Checkpoint:
    at: int
    wake: Optional[str]
    author: str
    written_at: int


@dataclass
class _Markers:
    """The LATEST marker of each type, with its author and write time.

    Order matters: a gate recorded before a later checkpoint (or before an
    explicit resolution) no longer governs the card. Nothing here is trusted on
    content alone — the author and the timestamps are part of the evidence.
    """

    gate: Optional[_Gate] = None
    checkpoint: Optional[_Checkpoint] = None
    gate_resolved_at: Optional[int] = None
    escalation: Optional[Any] = None

    @property
    def checkpoint_at(self) -> Optional[int]:  # back-compat for readers/tests
        return self.checkpoint.at if self.checkpoint else None


def _scan_markers(comments) -> _Markers:
    out = _Markers()
    for c in comments or []:
        body = getattr(c, "body", "") or ""
        author = str(getattr(c, "author", "") or "")
        created = int(getattr(c, "created_at", 0) or 0)
        m = CHECKPOINT_RE.search(body)
        if m:
            when = _parse_ts(m.group("when"))
            if when is not None:
                w = WAKE_RE.search(m.group("rest") or "")
                out.checkpoint = _Checkpoint(
                    at=when,
                    wake=w.group("wake") if w else None,
                    author=author,
                    written_at=created,
                )
        esc = escalation_mod.parse_escalation(body, author=author, written_at=created)
        if esc is not None:
            out.escalation = esc
        if GATE_RESOLVED_RE.search(body):
            out.gate_resolved_at = created
            out.gate = None
            continue
        g = GATE_RE.search(body)
        if g:
            raw = g.group("what").strip()
            until = UNTIL_RE.search(raw)
            out.gate = _Gate(
                what=raw,
                author=author,
                written_at=created,
                until=_parse_ts(until.group("until")) if until else None,
            )
    return out


def _gate_authorities(cfg: dict) -> set[str]:
    vals = cfg.get("gate_authorities") or []
    if isinstance(vals, str):
        vals = [vals]
    return {str(v).strip().lower() for v in vals if str(v).strip()}


def verify_wake(
    wake: Optional[str],
    *,
    task,
    now: int,
    deadline: Optional[int],
    registry: list[dict],
    cfg: dict,
    ambiguous: frozenset = frozenset(),
) -> tuple[bool, str]:
    """Does this wake name a target that actually exists? (ok, detail).

    A bare mechanism word is never a wake — the target has to be resolvable.
    """
    if not wake:
        return (False, "no wake recorded")
    kind, _, target = wake.partition(":")
    kind = kind.lower()

    if kind in ("cron", "cronjob"):
        if not target:
            return (False, "wake=cron names no job id (bare word is not a wake)")
        try:
            from cron.jobs import load_jobs
        except Exception as exc:
            return (False, f"cron job store unavailable: {exc}")
        try:
            jobs = load_jobs() or []
        except Exception as exc:
            return (False, f"cron job store unreadable: {exc}")
        job = next((j for j in jobs if str(j.get("id")) == target), None)
        if job is None:
            return (False, f"cron job '{target}' does not exist ({len(jobs)} job(s) present)")
        if job.get("enabled") is False or job.get("paused"):
            return (False, f"cron job '{target}' exists but is disabled/paused")
        nxt = _parse_ts(str(job.get("next_run_at") or ""))
        if nxt is None:
            return (False, f"cron job '{target}' has no next run armed")
        if nxt <= now:
            return (False, f"cron job '{target}' next run is overdue ({now - nxt}s)")
        if deadline is not None and nxt > deadline:
            return (
                False,
                f"cron job '{target}' next run is after the checkpoint deadline",
            )
        return (True, f"cron job '{target}' fires in {nxt - now}s")

    if kind in ("process", "proc"):
        entry = next(
            (e for e in registry if str(e.get("session_id") or "") == target), None
        )
        if entry is None:
            return (False, f"process handle '{target}' is not in the process registry")
        bound = owners_mod.binding_for(entry, task, ambiguous=ambiguous)
        if not bound:
            # "A live process exists somewhere" is not a wake for THIS card.
            return (
                False,
                f"process handle '{target}' is live but is not bound to {task.id} "
                f"(no registry task_id and no canonical workspace relationship)",
            )
        ev = owners_mod.evidence_from_entry(
            entry,
            default_max_runtime=int(
                cfg.get("max_owner_runtime_seconds", DEFAULT_MAX_OWNER_RUNTIME)
            ),
            binding=bound,
        )
        if not ev.usable:
            return (False, f"process handle '{target}' is not a live verified process")
        if not ev.completion_handle:
            # A wake has to re-enter the conversation. A live process with no
            # completion handle is something to poll, not something that wakes.
            return (
                False,
                f"process handle '{target}' is live but has no completion handle "
                f"(no notify_on_complete, no watcher) — nothing would re-enter "
                f"the conversation when it exits",
            )
        return (
            True,
            f"process {target} pid {ev.pid} live, bound by {bound}, "
            f"completion via {ev.completion_handle}",
        )

    if kind in ("dispatcher", "kanban-wake", "kanban"):
        if task.status not in DISPATCHABLE_STATUSES:
            return (
                False,
                f"dispatcher wake but status '{task.status}' is not dispatchable",
            )
        if not task.assignee:
            return (False, "dispatcher wake but the card has no assignee to spawn")
        return (True, f"dispatcher can claim a '{task.status}' card for {task.assignee}")

    return (False, f"wake '{wake}' names no target this deployment can verify")


# ------------------------------------------------------------ classifier ---

def _current_head(task, cfg: dict, esc) -> Optional[str]:
    """The head the supervisor can observe RIGHT NOW, best source first.

    Order matters and is the whole point of the staleness check: a live
    resolver (or an operator-supplied observation) always beats the SHA the
    escalation marker recorded when it was written, because the marker is
    exactly the thing that can be out of date. The marker's ``head_sha`` is the
    last resort, and it is still only ever compared against ``review_sha`` —
    never trusted as proof that the review is current.
    """
    resolver = cfg.get("head_resolver")
    if callable(resolver):
        try:
            got = resolver(task)
        except Exception:
            got = None
        if got:
            return str(got)
    heads = cfg.get("current_heads")
    if isinstance(heads, dict):
        got = heads.get(getattr(task, "id", "")) or heads.get("*")
        if got:
            return str(got)
    return (getattr(esc, "head_sha", "") or "").strip() or None


# ------------------------------------------------------------ classifier ---

def _classify(
    task,
    *,
    run,
    markers: _Markers,
    owner_evidence: list,
    registry: list[dict],
    now: int,
    cfg: dict,
    parents_unfinished: bool,
    last_activity_at: int = 0,
    ambiguous: frozenset = frozenset(),
) -> tuple[Optional[Finding], Optional[Attended]]:
    heartbeat_stale = int(cfg.get("heartbeat_stale_seconds", 900))
    human_kinds = tuple(cfg.get("human_gate_kinds", DEFAULT_HUMAN_GATE_KINDS))
    max_gate_age = int(cfg.get("max_gate_age_seconds", DEFAULT_MAX_GATE_AGE))
    max_hold_age = int(cfg.get("max_hold_age_seconds", DEFAULT_MAX_HOLD_AGE))
    owner = task.assignee
    tid = task.id
    title = (task.title or "")[:100]

    def find(kind, detail, action):
        return (Finding(tid, title, task.status, owner, kind, detail, action), None)

    def ok(reason, detail):
        return (None, Attended(tid, task.status, reason, detail))

    # 1. A structurally-bound, identity-verified LIVE external owner. Liveness
    #    only — never claimed as progress, and never silence on its own.
    live = [e for e in owner_evidence if e.usable]
    for ev in live:
        late = ev.overdue_by(now)
        if late:
            return find(
                KIND_OWNER_OVERDUE,
                f"{ev.describe(now)} is past its own deadline by {late}s",
                f"poll {ev.handle} for {tid} and decide: extend with a recorded "
                f"deadline, or stop it and hand the card back to {owner or 'an owner'}",
            )
    if live:
        # A live owner buys silence ONLY if something will actually re-enter the
        # conversation when it exits or when its deadline lands. Either the
        # process itself carries a completion handle, or the CARD carries a
        # recorded, verifiable wake. With neither, "someone will tell me when it
        # lands" is an assumption, so the card stays a bounded finding.
        wakeable = [e for e in live if e.completion_handle]
        if not wakeable:
            ev = live[0]
            cp = markers.checkpoint
            okw, why = (False, "no checkpoint wake recorded on the card")
            if cp is not None and cp.at > now:
                okw, why = verify_wake(
                    cp.wake,
                    task=task,
                    now=now,
                    deadline=cp.at,
                    registry=registry,
                    cfg=cfg,
                    ambiguous=ambiguous,
                )
            if okw:
                return ok(
                    "live_external_owner",
                    f"{ev.describe(now)} — liveness, not progress; exit covered by "
                    f"recorded wake {cp.wake} ({why})",
                )
            due = ev.deadline_at
            when = f"in {int(due - now)}s" if due else "at an unrecorded time"
            return find(
                KIND_OWNER_NO_WAKE,
                f"{ev.describe(now)} — live, but NOTHING will re-enter this "
                f"conversation when it exits: no notify_on_complete, no watcher, "
                f"and {why}. Its deadline lands {when}",
                f"register a real wake for {tid}: re-poll the handle {ev.handle} at "
                f"its deadline, or record a checkpoint with a verifiable wake "
                f"(cron:<job_id> / process:<handle> / dispatcher). Do not treat "
                f"'it is still running' as an answer",
            )
        ev = wakeable[0]
        return ok(
            "live_external_owner",
            f"{ev.describe(now)} — liveness, not progress",
        )

    # 2. A recorded owner that is provably GONE beats every marker: a dead
    #    owner cannot be hidden behind a future checkpoint someone wrote.
    #    An owner whose pid is alive but whose identity will not confirm is a
    #    THIRD state — unknown — and is qualified, not declared dead.
    gone = [e for e in owner_evidence if not e.alive]
    if gone:
        ev = gone[0]
        return find(
            KIND_OWNER_STOPPED,
            f"recorded external owner {ev.handle} pid {ev.pid} is gone (pid not "
            f"alive); any checkpoint on this card is unbacked",
            f"hand {tid} back to the SAME owner ({owner or 'assign one'}) with the "
            f"last run output; do not start a second worker",
        )
    unconfirmed = [e for e in owner_evidence if e.alive and not e.identity_verified]
    if unconfirmed:
        ev = unconfirmed[0]
        return find(
            KIND_OWNER_UNKNOWN,
            f"recorded owner {ev.handle} pid {ev.pid} is alive but its identity does "
            f"not confirm (start-time mismatch or unreadable) — it may be a recycled "
            f"pid, so this is NOT evidence of work",
            f"qualify {tid} in one bounded step: poll the owner handle {ev.handle} (or "
            f"ask the owner), then record a live handle, a verified wake, or the real "
            f"blocker",
        )

    # 3. A kanban claim that is genuinely live (liveness + claim freshness).
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
                "live_executor",
                f"run {run.id} pid {pid} alive, heartbeat {age}s ago, claim unexpired "
                f"— liveness, not progress",
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

    # 3b. A recorded escalation. Pulling the human in is expensive and is only
    #     correct when the decision is genuinely theirs. A review gate that a
    #     documented, authorised OPAQUE identity can discharge, against an
    #     independent review pinned to the CURRENT head, is routable work — and
    #     escalating it anyway is the t_de12518a/#4952 defect. Every other
    #     shape (stale review, undocumented/credential-revealing identity,
    #     non-review class) keeps the human prompt or emits an access blocker.
    #     This runs BEFORE the human-gate branch so a routable escalation is
    #     never laundered into "attended by a human gate" and left quiet.
    esc = markers.escalation
    if esc is not None:
        v = escalation_mod.classify_escalation(
            esc,
            current_head=_current_head(task, cfg, esc),
            cfg=cfg,
            task_id=tid,
        )
        if v.decision == escalation_mod.DECISION_ROUTE:
            return find(KIND_ESCALATION_ROUTABLE, v.detail, v.next_action)
        if v.decision == escalation_mod.DECISION_FRESH_REVIEW:
            return find(KIND_ESCALATION_STALE_REVIEW, v.detail, v.next_action)
        if v.decision == escalation_mod.DECISION_ACCESS_BLOCKER:
            return find(KIND_ESCALATION_ACCESS, v.detail, v.next_action)
        # DECISION_HUMAN falls through: the existing gate/hold branches below
        # own the human prompt, unchanged.

    # 4. An authorised, current human/external gate.
    gate = markers.gate
    if gate:
        authorities = _gate_authorities(cfg)
        problems = []
        if str(gate.author or "").strip().lower() == str(owner or "").strip().lower():
            problems.append(
                f"written by the card's own worker ({gate.author or 'unknown'})"
            )
        elif authorities and str(gate.author or "").strip().lower() not in authorities:
            problems.append(f"author '{gate.author or 'unknown'}' is not an authority")
        elif not authorities:
            problems.append("no gate authority configured to authorise it")
        if gate.until is not None and gate.until <= now:
            problems.append(f"expired {now - gate.until}s ago")
        elif gate.until is None and gate.written_at and now - gate.written_at > max_gate_age:
            problems.append(f"no expiry and {int((now - gate.written_at) / 86400)}d old")
        if problems:
            return find(
                KIND_UNQUALIFIED_GATE,
                f"hold '{gate.what[:70]}' is not a qualified human gate: "
                + "; ".join(problems),
                f"re-confirm {tid} with the named human and record a fresh gate "
                f"(author + until=), or record the real blocker — do NOT perform "
                f"the gated action and do NOT bypass the restriction",
            )
        return ok(
            "human_gate",
            f"gate by {gate.author} until "
            f"{gate.until or 'unset'}: {gate.what[:80]}",
        )

    # 5. A typed hold, bounded by age: a forgotten park must requalify, and
    #    requalification is never permission to do the restricted thing.
    if task.status in ("blocked", "scheduled") and (task.block_kind or "") in human_kinds:
        parked_at = int(
            last_activity_at
            or getattr(task, "started_at", 0)
            or getattr(task, "created_at", 0)
            or 0
        )
        age = now - parked_at if parked_at else None
        if age is not None and age > max_hold_age:
            return find(
                KIND_STALE_HOLD,
                f"typed hold '{task.block_kind}' untouched for {int(age / 86400)}d "
                f"— stale, not permanently exempt",
                f"requalify {tid}: ask the named human/owner whether the hold still "
                f"stands and record the answer. Requalifying is NOT permission to "
                f"perform the held action",
            )
        return ok("human_gate", f"typed block kind '{task.block_kind}' ({age}s old)")

    # 6. Owner's run ended while the card is still unfinished -> handoff.
    if (
        task.status not in ("blocked", "scheduled")
        and run is not None
        and getattr(run, "ended_at", None) is not None
    ):
        outcome = getattr(run, "outcome", None) or getattr(run, "status", "ended")
        return find(
            KIND_OWNER_STOPPED,
            f"last run {run.id} ended ({outcome}) but card is still {task.status}",
            f"hand {tid} back to the SAME owner ({owner or 'assign one'}) with the "
            f"run summary; no duplicate worker",
        )

    # 7. Checkpoint evidence — deadline AND a wake target that resolves.
    cp = markers.checkpoint
    if cp:
        if cp.at <= now:
            return find(
                KIND_OVERDUE_CHECKPOINT,
                f"checkpoint overdue by {now - cp.at}s with no newer run",
                f"diagnose {tid} now (owner process, logs, dependency); "
                f"do not simply extend the deadline",
            )
        okw, detail = verify_wake(
            cp.wake,
            task=task,
            now=now,
            deadline=cp.at,
            registry=registry,
            cfg=cfg,
            ambiguous=ambiguous,
        )
        if not okw:
            return find(
                KIND_NO_WAKE,
                f"future checkpoint is not backed by a real wake: {detail}",
                f"attach a verifiable wake to {tid} (cron:<job_id>, process:<handle>, "
                f"or make it dispatcher-claimable) or act on it now",
            )
        return ok("future_checkpoint", f"in {cp.at - now}s via {cp.wake} ({detail})")

    # 8. 'scheduled' is an explicit park the dispatcher will not pick up.
    if task.status == "scheduled":
        return find(
            KIND_STALE_HOLD,
            "parked in 'scheduled' (not dispatchable) with no checkpoint, "
            "no verified wake and no qualified gate",
            f"give {tid} a checkpoint with a verifiable wake, resume it for its "
            f"owner ({owner or 'assign one'}), or record an explicit blocker",
        )

    # 9. Dependency block behind a still-unfinished parent covered in this sweep.
    if task.status == "blocked" and (task.block_kind or "") == "dependency" and parents_unfinished:
        return ok("dependency", "waiting on an unfinished parent card in this sweep")

    # 10. No evidence either way on a card whose status claims it is in flight.
    #     Explicit, bounded qualification — never quiet, never "idle".
    if task.status in ("running", "review"):
        return find(
            KIND_OWNER_UNKNOWN,
            f"status {task.status} with an assignee ({task.assignee or 'none'}) but no "
            f"verifiable owner: no live kanban claim and no matching process-registry "
            f"entry (comments are not evidence)",
            f"qualify {tid} in one bounded step: check the owner's process handle / "
            f"ask the owner, then record either a live handle, a verified wake, or the "
            f"real blocker",
        )

    if task.status == "blocked":
        kind = KIND_STALE_HOLD if task.block_kind else KIND_UNOWNED
        return find(
            kind,
            f"blocked (kind={task.block_kind or 'untyped'}) with no live owner, "
            "no verified wake and no qualified gate",
            f"resolve or re-dispatch {tid} on its existing card "
            f"(owner {owner or 'needs one'}), or record an explicit blocker",
        )
    return find(
        KIND_IDLE,
        f"status {task.status} with no owner, wake or gate",
        f"route {tid} to its owner ({owner or 'assign one'}) or record an "
        f"explicit blocker/checkpoint",
    )


# -------------------------------------------------------------- evaluate ---

class ScopeMatchedNothing(Exception):
    """A configured scope selected zero cards from a board that HAS cards."""


def _scope_tasks(tasks, cfg) -> list:
    """Restrict the sweep to the opted-in project. No other project is read.

    A configured scope that matches **zero of N** rows is a configuration
    error, not a clean board: it would otherwise produce ``ok=True`` with no
    findings, i.e. the gate silently disables itself while the config reads as
    if it were on. (Observed on the real board: every card carries
    ``project_id=NULL``, so the previously documented ``project_id: agentpod``
    example selected nothing.)
    """
    project_id = cfg.get("project_id")
    tenant = cfg.get("tenant")
    out = list(tasks)
    if project_id is not None:
        out = [t for t in out if (getattr(t, "project_id", None) or None) == project_id]
    if tenant is not None:
        out = [t for t in out if (getattr(t, "tenant", None) or None) == tenant]
    if (project_id is not None or tenant is not None) and tasks and not out:
        raise ScopeMatchedNothing(
            f"scope matched 0 of {len(tasks)} cards "
            f"(project_id={project_id!r}, tenant={tenant!r}). The board's cards "
            f"do not carry these values, so the sweep would cover nothing and "
            f"report a clean board. Fix or remove the scope keys."
        )
    return out


def evaluate_board(
    *,
    board: Optional[str] = None,
    db_path: Optional[str] = None,
    now: Optional[int] = None,
    cfg: Optional[dict] = None,
    kb: Any = None,
    registry: Optional[list[dict]] = None,
    registry_path: Optional[str] = None,
) -> Verdict:
    """Read the board and decide whether a quiet conclusion is permitted.

    Read-only. Any failure (missing board, locked DB, import failure, zero rows)
    yields ``ok=False`` — a refused read must never be reported as "no work".
    """
    cfg = dict(cfg or {})
    now = int(now if now is not None else time.time())
    max_findings = int(cfg.get("max_findings", 5))
    notes: list[str] = []

    if registry is None:
        registry, reg_err = owners_mod.load_registry(
            registry_path or cfg.get("process_registry_path") or None
        )
        if reg_err:
            # Owner evidence is unreadable: say so, and never silently downgrade
            # a live worker to "idle" — every no-owner card becomes unknown.
            notes.append(f"owner evidence degraded: {reg_err}")
    scope_desc = "project_id={} tenant={}".format(
        cfg.get("project_id", "*"), cfg.get("tenant", "*")
    )

    if kb is None:
        try:
            kb = kanban_db_module()
        except Exception as exc:
            return Verdict(
                ok=False,
                board=str(board or db_path or ""),
                scope=scope_desc,
                error=f"kanban_db import failed: {exc}",
            )

    conn = None
    try:
        from pathlib import Path

        conn = kb.connect(Path(db_path)) if db_path else kb.connect(board=board)
        # Read the board unscoped so "scope matched nothing" can be told apart
        # from "board is empty". Only SCOPED cards are ever evaluated below.
        tasks = kb.list_tasks(conn, include_archived=False)
        if not tasks:
            return Verdict(
                ok=False,
                board=str(board or db_path or ""),
                scope=scope_desc,
                error="board read returned zero cards — treat as an unreadable "
                "board, not as an empty backlog",
            )
        try:
            tasks = _scope_tasks(tasks, cfg)
        except ScopeMatchedNothing as exc:
            return Verdict(
                ok=False,
                board=str(board or db_path or ""),
                scope=scope_desc,
                error=str(exc),
            )
        ambiguous = owners_mod.ambiguous_workspaces(tasks)
        if ambiguous:
            notes.append(
                f"{len(ambiguous)} workspace path(s) shared by more than one card "
                f"— not treated as ownership of any of them"
            )
        unfinished = [t for t in tasks if t.status in UNFINISHED_STATUSES]
        unfinished_ids = {t.id for t in unfinished}

        findings: list[Finding] = []
        attended: list[Attended] = []
        for task in unfinished:
            run = kb.latest_run(conn, task.id)
            comments = kb.list_comments(conn, task.id)
            markers = _scan_markers(comments)
            # Latest *structured* activity timestamp. Used only to age typed
            # holds — never as evidence that work is happening.
            last_activity_at = max(
                [int(getattr(c, "created_at", 0) or 0) for c in (comments or [])]
                + [
                    int(getattr(run, "ended_at", 0) or 0) if run else 0,
                    int(getattr(run, "started_at", 0) or 0) if run else 0,
                    int(getattr(task, "started_at", 0) or 0),
                    int(getattr(task, "created_at", 0) or 0),
                ]
            )
            try:
                parents = kb.parent_ids(conn, task.id)
            except Exception:
                parents = []
            evidence = owners_mod.owners_for_task(
                task,
                registry or [],
                default_max_runtime=int(
                    cfg.get("max_owner_runtime_seconds", DEFAULT_MAX_OWNER_RUNTIME)
                ),
                ambiguous=ambiguous,
            )
            f, a = _classify(
                task,
                run=run,
                markers=markers,
                owner_evidence=evidence,
                registry=registry or [],
                now=now,
                cfg=cfg,
                parents_unfinished=any(p in unfinished_ids for p in parents),
                last_activity_at=last_activity_at,
                ambiguous=ambiguous,
            )
            if f is not None:
                findings.append(f)
            if a is not None:
                attended.append(a)
    except Exception as exc:
        return Verdict(
            ok=False,
            board=str(board or db_path or ""),
            scope=scope_desc,
            error=f"board read failed: {type(exc).__name__}: {exc}",
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    findings.sort(key=lambda f: (f.kind != KIND_OWNER_STOPPED, f.task_id))
    truncated = max(0, len(findings) - max_findings)
    return Verdict(
        ok=True,
        board=str(board or db_path or ""),
        scope=scope_desc,
        unfinished=len(unfinished),
        findings=findings[:max_findings],
        attended=attended,
        truncated=truncated,
        notes=notes,
    )


def kanban_db_module():
    """Import hook kept separate so callers can fail explicitly."""
    from hermes_cli import kanban_db  # type: ignore

    return kanban_db


def render_report(
    verdict: Verdict,
    *,
    continuation: bool = False,
    terminal: bool = False,
    max_chars: int = 0,
) -> str:
    """Explicit, fail-loud text. Never claims an action was taken."""
    if not verdict.ok:
        text = (
            "STOP-CHECK ERROR — the board could not be reconciled, so this turn may "
            "NOT be concluded as done or quiet.\n"
            f"board: {_short_board(verdict.board)} | scope: {verdict.scope}\n"
            f"error: {verdict.error}\n"
            "Required: repeat the board read, then re-assess. A refused read is not "
            "evidence of no work."
        )
        return _clamp(text, max_chars)

    head = (
        f"STOP-CHECK — {len(verdict.findings) + verdict.truncated} unattended of "
        f"{verdict.unfinished} unfinished card(s) on {_short_board(verdict.board)} "
        f"({verdict.scope}); {len(verdict.attended)} attended. This turn may not be "
        f"concluded as done or quiet."
    )
    lines = [f.short() if max_chars else f.line() for f in verdict.findings]
    tail = []
    if verdict.truncated:
        tail.append(f"(+{verdict.truncated} more unattended)")
    if verdict.notes:
        tail.append("; ".join(verdict.notes))
    if verdict.attended and not max_chars:
        tail.append(
            "attended (live/gated, not progress): "
            + "; ".join(f"{a.task_id}={a.reason}" for a in verdict.attended[:6])
        )
    if continuation:
        tail.append(
            "Act on the first item now on its existing card: no duplicate card, no "
            "duplicate worker, no gate bypass, no spending or production change. If "
            "it cannot be advanced, record the specific blocker or a checkpoint with "
            "a verifiable wake. (The stop-check itself started nothing and wrote "
            "nothing — it only blocked a quiet ending.)"
        )
        if terminal:
            # Last grant in this window: the gate will not ask again, so the
            # fail-explicit demand is delivered HERE, through pre_verify, which
            # runs before any transform and is unaffected by transform ordering.
            tail.append(
                "FINAL supervision continuation for this board state — the "
                "continuation budget is now spent and you will not be asked "
                "again. Your next answer MUST state the unattended cards above "
                "and their blocker explicitly. Concluding 'no material change' "
                "or any equivalent quiet ending is not permitted."
            )
    else:
        tail.append(
            "Blocked quiet ending — nothing above was executed or dispatched."
            if max_chars
            else (
                "This text is a blocked quiet conclusion, not an executed action: the "
                "stop-check dispatches nothing and writes nothing."
            )
        )
    if not max_chars:
        return "\n".join([head, *lines, *tail])

    # Budget mode: the head and the "nothing was executed" disclosure are
    # load-bearing, so findings yield first and the drop is reported honestly.
    fixed = len(head) + sum(len(t) + 1 for t in tail) + 1
    room = max_chars - fixed
    kept: list[str] = []
    dropped = verdict.truncated
    for ln in lines:
        if room - (len(ln) + 1) < 0:
            dropped += 1
            continue
        room -= len(ln) + 1
        kept.append(ln)
    if dropped:
        note = f"(+{dropped} more unattended not shown)"
        if verdict.truncated:
            tail[0] = note
        else:
            tail.insert(0, note)
    return _clamp("\n".join([head, *kept, *tail]), max_chars)


def _short_board(board: str) -> str:
    """Board identity without burning the platform budget on a long path."""
    raw = str(board or "").strip()
    if not raw:
        return "(unresolved)"
    return raw.rsplit("/", 1)[-1] or raw


def _clamp(text: str, max_chars: int) -> str:
    if not max_chars or len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 24)
    return text[:keep].rstrip() + "\n…[stop-check truncated]"
