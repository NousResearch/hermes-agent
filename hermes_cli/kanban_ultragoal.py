"""Kanban-Ultragoal durable controller.

This module implements the BO-203/RALPLAN-v2 runtime core: Kanban is the
authority, a canonical ``.hermes/goal-runs/<run_id>/`` root stores resumable
execution evidence, and controller transitions prevent PR-ready accounting until
worker, verifier, reviewer, PR, and CI evidence are all present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_cli import kanban_db as kb

_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
VALID_STATES = {
    "admitted",
    "running",
    "worker_done",
    "verification_failed",
    "verification_passed",
    "review_failed",
    "review_passed",
    "pr_created",
    "ci_pending",
    "ci_failed",
    "ci_passed",
    "review_ready",
    "blocked",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_run_id(run_id: str) -> str:
    if not _SAFE_RUN_ID_RE.match(run_id):
        raise ValueError("run_id must be 1-128 chars of letters, digits, dot, underscore, or dash; path separators are forbidden")
    return run_id


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(data: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(data).encode()).hexdigest()


def _json_loads_maybe(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _extract_done_criteria(body: str | None, snapshot: dict[str, Any] | None = None) -> list[str]:
    if snapshot:
        for key in ("doneCriteria", "done_criteria", "doneCriteriaItems"):
            val = snapshot.get(key)
            if isinstance(val, list):
                return [str(item).strip() for item in val if str(item).strip()]
    if not body:
        return []
    out: list[str] = []
    in_section = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.lower() == "done criteria:":
            in_section = True
            continue
        if in_section and stripped and not stripped.startswith("-") and stripped.endswith(":"):
            break
        if in_section and stripped.startswith("-"):
            out.append(stripped[1:].strip())
    return out


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    return row[key] if key in row.keys() else default


def build_authority_snapshot(task_ref: str) -> dict[str, Any]:
    """Read a canonical Kanban authority snapshot for ``task_ref``.

    The read path is strict read-only: a missing board DB fails closed instead
    of initializing Kanban or creating sidecar files.
    """
    db_path = kb.kanban_db_path()
    if not db_path.exists():
        raise ValueError(f"Kanban DB does not exist: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM tasks WHERE id = ? OR public_id = ?", (task_ref, task_ref)).fetchall()
        unique: dict[str, Any] = {row["id"]: row for row in rows}
        if not unique:
            raise ValueError(f"no such Kanban task: {task_ref}")
        if len(unique) != 1:
            raise ValueError(f"ambiguous Kanban task reference: {task_ref}")
        row = next(iter(unique.values()))
        raw_snapshot = _json_loads_maybe(_row_get(row, "admission_snapshot"))
        admission_snapshot = raw_snapshot if isinstance(raw_snapshot, dict) else {}
        done_criteria = _extract_done_criteria(_row_get(row, "body"), admission_snapshot)
        children = [
            {"id": r["child_id"], "relationType": r["relation_type"]}
            for r in conn.execute(
                "SELECT child_id, relation_type FROM task_links WHERE parent_id = ? ORDER BY child_id",
                (row["id"],),
            )
        ]
        parents = [
            {"id": r["parent_id"], "relationType": r["relation_type"]}
            for r in conn.execute(
                "SELECT parent_id, relation_type FROM task_links WHERE child_id = ? ORDER BY parent_id",
                (row["id"],),
            )
        ]
        base = {
            "authority": "kanban",
            "taskId": row["id"],
            "runId": row["id"],
            "publicId": _row_get(row, "public_id") or row["id"],
            "title": row["title"],
            "status": row["status"],
            "assignee": row["assignee"],
            "routingVerdict": _row_get(row, "routing_verdict"),
            "executionApproved": admission_snapshot.get("execution_approved") is True,
            "reviewPhase": _row_get(row, "review_phase"),
            "currentRunId": _row_get(row, "current_run_id"),
            "goalMode": bool(_row_get(row, "goal_mode", 0)),
            "children": children,
            "parents": parents,
            "doneCriteria": done_criteria,
            "missingDoneCriteriaContract": not bool(done_criteria),
        }
        base["doneCriteriaHash"] = _sha256(done_criteria)
        base["snapshotHash"] = _sha256({k: base[k] for k in sorted(base) if k != "snapshotHash"})
        return base
    finally:
        conn.close()


def pilot_check(task_ref: str) -> dict[str, Any]:
    """Strict read-only pilot eligibility check over live Kanban authority."""
    snapshot = build_authority_snapshot(task_ref)
    blockers: list[str] = []
    if snapshot.get("routingVerdict") != "direct-kanban":
        blockers.append("routingVerdict")
    if snapshot.get("executionApproved") is not True:
        blockers.append("executionApproved")
    if not snapshot.get("doneCriteria"):
        blockers.append("doneCriteria")
    if snapshot.get("currentRunId") is not None:
        blockers.append("currentRunId")
    return {"eligible": not blockers, "blockers": blockers, "authority": snapshot}


def _load_json_arg(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    stripped = value.lstrip()
    if stripped.startswith("{"):
        return json.loads(value)
    p = Path(value)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(value)


@dataclass
class KanbanUltragoalRun:
    run_id: str
    parent_card: str
    root_objective: str
    state: str = "admitted"
    tick: int = 0
    current_goal_id: str | None = None
    authority: dict[str, Any] = field(default_factory=dict)
    pending_action: dict[str, Any] | None = None
    resumable: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    last_terminal_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "runId": self.run_id,
            "parentCard": self.parent_card,
            "rootObjective": self.root_objective,
            "state": self.state,
            "tick": self.tick,
            "currentGoalId": self.current_goal_id,
            "authority": self.authority,
            "pendingAction": self.pending_action,
            "resumable": self.resumable,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "lastTerminalReport": self.last_terminal_report,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KanbanUltragoalRun":
        return cls(
            run_id=data["runId"],
            parent_card=data.get("parentCard") or data["runId"],
            root_objective=data.get("rootObjective") or "",
            state=data.get("state", "admitted"),
            tick=int(data.get("tick") or 0),
            current_goal_id=data.get("currentGoalId"),
            authority=data.get("authority") or {},
            pending_action=data.get("pendingAction"),
            resumable=bool(data.get("resumable")),
            created_at=data.get("createdAt") or _now(),
            updated_at=data.get("updatedAt") or _now(),
            last_terminal_report=data.get("lastTerminalReport"),
        )


class KanbanUltragoalStore:
    def __init__(self, workdir: str | Path = ".") -> None:
        self.workdir = Path(workdir)

    def root(self, run_id: str) -> Path:
        return self.workdir / ".hermes" / "goal-runs" / _validate_run_id(run_id)

    def run_path(self, run_id: str) -> Path:
        return self.root(run_id) / "run.json"

    def authority_path(self, run_id: str) -> Path:
        return self.root(run_id) / "authority.json"

    def ledger_path(self, run_id: str) -> Path:
        return self.root(run_id) / "ledger.jsonl"

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        tmp.replace(path)

    def _append_ledger(self, run_id: str, event: str, payload: dict[str, Any] | None = None) -> None:
        path = self.ledger_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"at": _now(), "event": event, "payload": payload or {}}
        with path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    def save_run(self, run: KanbanUltragoalRun) -> KanbanUltragoalRun:
        run.updated_at = _now()
        self._write_json(self.run_path(run.run_id), run.to_dict())
        return run

    def load_run(self, run_id: str) -> KanbanUltragoalRun:
        return KanbanUltragoalRun.from_dict(json.loads(self.run_path(run_id).read_text()))

    def _normalize_authority(self, authority: dict[str, Any] | None, run_id: str) -> dict[str, Any]:
        if not isinstance(authority, dict):
            raise ValueError("Kanban authority snapshot is required")
        if authority.get("authority") != "kanban":
            raise ValueError("Kanban authority snapshot must explicitly declare authority=kanban")
        task_id = authority.get("taskId") or authority.get("task_id")
        if task_id != run_id:
            raise ValueError(f"Kanban authority taskId mismatch: expected {run_id}, got {task_id}")
        routing = authority.get("routingVerdict") or authority.get("routing_verdict")
        if routing != "direct-kanban":
            raise ValueError(f"Kanban authority routing mismatch: {routing}")
        if authority.get("executionApproved") is not True:
            raise ValueError("Kanban authority executionApproved=true is required")
        if not authority.get("snapshotHash"):
            raise ValueError("Kanban authority snapshotHash is required")
        if not authority.get("doneCriteriaHash"):
            raise ValueError("Kanban authority doneCriteriaHash is required")
        return dict(authority)

    def _require_current_authority(self, run: KanbanUltragoalRun, authority: dict[str, Any] | None) -> dict[str, Any]:
        current = self._normalize_authority(authority, run.run_id)
        stored = run.authority or {}
        for key in ("taskId", "snapshotHash", "doneCriteriaHash"):
            if current.get(key) != stored.get(key):
                raise ValueError(f"Kanban authority {key} mismatch: expected {stored.get(key)}, got {current.get(key)}")
        return current

    def start(self, run_id: str, *, authority: dict[str, Any], root_objective: str, force: bool = False) -> KanbanUltragoalRun:
        run_id = _validate_run_id(run_id)
        normalized = self._normalize_authority(authority, run_id)
        root = self.root(run_id)
        if root.exists():
            if not force:
                raise FileExistsError(f"Kanban-Ultragoal run root already exists: {run_id}")
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "ultragoal").mkdir(parents=True, exist_ok=True)
        run = KanbanUltragoalRun(run_id=run_id, parent_card=run_id, root_objective=root_objective, authority=normalized)
        self._write_json(self.authority_path(run_id), normalized)
        self._write_json(root / "goals.json", {"version": 1, "goals": [], "currentGoalId": None})
        self._write_json(root / "ultragoal" / "goals.json", {"version": 1, "codexGoalMode": "aggregate", "goals": []})
        self.save_run(run)
        self._append_ledger(run_id, "run_started", {"state": run.state, "authorityHash": _sha256(normalized)})
        return run

    def _prepare_pending_action(self, run: KanbanUltragoalRun, action: str) -> dict[str, Any]:
        step_id = f"{run.run_id}:tick-{run.tick + 1}:{action}"
        return {
            "stepId": step_id,
            "fenceToken": _sha256({"authority": run.authority, "state": run.state, "action": action}),
            "phase": "prepared",
            "idempotencyKey": step_id,
            "sideEffects": {
                "branchCreated": False,
                "commitSha": None,
                "pushedRef": None,
                "prUrl": None,
                "kanbanEventId": None,
            },
        }

    def tick(self, run_id: str, *, authority: dict[str, Any], budget_remaining: int = 20) -> KanbanUltragoalRun:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        if budget_remaining <= 0:
            if run.state == "admitted":
                run.state = "running"
            run.pending_action = self._prepare_pending_action(run, "resume-controller")
            run.resumable = True
            self._append_ledger(run_id, "checkpoint_budget_near_limit", {"pendingAction": run.pending_action})
            return self.save_run(run)
        run.tick += 1
        run.resumable = False
        run.pending_action = None
        if run.state == "admitted":
            run.state = "running"
            self._append_ledger(run_id, "transition", {"from": "admitted", "to": "running"})
        else:
            self._append_ledger(run_id, "tick", {"state": run.state})
        return self.save_run(run)

    def record_worker_done(self, run_id: str, *, authority: dict[str, Any], evidence: dict[str, Any]) -> KanbanUltragoalRun:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        if run.state not in {"admitted", "running", "verification_failed", "review_failed", "ci_failed"}:
            raise ValueError(f"worker_done transition is not allowed from {run.state}")
        run.state = "worker_done"
        run.resumable = False
        self._write_json(self.root(run_id) / "evidence" / "worker.json", evidence)
        self._append_ledger(run_id, "worker_done", evidence)
        return self.save_run(run)

    def record_verifier_result(self, run_id: str, *, authority: dict[str, Any], result: dict[str, Any]) -> KanbanUltragoalRun:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        if run.state != "worker_done":
            raise ValueError(f"Verifier requires worker_done state, got {run.state}")
        self._write_json(self.root(run_id) / "verifier" / "result.json", result)
        if result.get("passed") is True:
            run.state = "verification_passed"
            run.current_goal_id = None
            event = "verification_passed"
        else:
            run.state = "verification_failed"
            run.current_goal_id = f"repair-{run.tick + 1}"
            event = "verification_failed"
        self._append_ledger(run_id, event, result)
        return self.save_run(run)

    def record_reviewer_result(self, run_id: str, *, authority: dict[str, Any], result: dict[str, Any]) -> KanbanUltragoalRun:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        if run.state != "verification_passed":
            raise ValueError(f"Reviewer requires verification_passed state, got {run.state}")
        security = result.get("securityConcerns")
        logic = result.get("logicErrors")
        if not isinstance(security, list):
            raise ValueError("Reviewer result must include explicit securityConcerns list")
        if not isinstance(logic, list):
            raise ValueError("Reviewer result must include explicit logicErrors list")
        self._write_json(self.root(run_id) / "reviews" / "final.json", result)
        if result.get("recommendation") == "APPROVE" and not security and not logic:
            run.state = "review_passed"
            run.current_goal_id = None
            event = "review_passed"
        else:
            run.state = "review_failed"
            run.current_goal_id = f"review-blocker-{run.tick + 1}"
            event = "review_failed"
        self._append_ledger(run_id, event, result)
        return self.save_run(run)

    def record_pr_created(self, run_id: str, *, authority: dict[str, Any], pr: dict[str, Any]) -> KanbanUltragoalRun:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        if run.state != "review_passed":
            raise ValueError("reviewed PR gate requires verifier pass and reviewer approve before PR creation")
        required = {"url", "number", "headSha"}
        missing = sorted(required - pr.keys())
        if missing:
            raise ValueError(f"PR evidence missing fields: {missing}")
        self._write_json(self.root(run_id) / "pr.json", pr)
        run.state = "pr_created"
        self._append_ledger(run_id, "pr_created", pr)
        return self.save_run(run)

    def record_ci_result(self, run_id: str, *, authority: dict[str, Any], ci: dict[str, Any]) -> KanbanUltragoalRun:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        if run.state not in {"pr_created", "ci_pending", "ci_failed"}:
            raise ValueError(f"CI result requires PR state, got {run.state}")
        if ci.get("state") == "success":
            pr = json.loads((self.root(run_id) / "pr.json").read_text(encoding="utf-8"))
            if not ci.get("headSha") or ci.get("headSha") != pr.get("headSha"):
                raise ValueError("CI headSha must match PR headSha before ci_passed")
            self._write_json(self.root(run_id) / "evidence" / "ci.json", ci)
            run.state = "ci_passed"
            run.current_goal_id = None
            event = "ci_passed"
        else:
            self._write_json(self.root(run_id) / "evidence" / "ci.json", ci)
            run.state = "ci_failed"
            run.current_goal_id = f"ci-repair-{run.tick + 1}"
            event = "ci_failed"
        self._append_ledger(run_id, event, ci)
        return self.save_run(run)

    def mark_review_ready(self, run_id: str, *, authority: dict[str, Any]) -> KanbanUltragoalRun:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        if run.state != "ci_passed":
            raise ValueError("CI success is required before review_ready")
        root = self.root(run_id)
        for rel, label in (("pr.json", "PR"), ("verifier/result.json", "verifier"), ("reviews/final.json", "reviewer"), ("evidence/ci.json", "CI")):
            if not (root / rel).exists():
                raise ValueError(f"{label} evidence is required before review_ready")
        run.state = "review_ready"
        run.last_terminal_report = {"kind": "review_ready", "at": _now()}
        self._append_ledger(run_id, "review_ready", run.last_terminal_report)
        return self.save_run(run)


def build_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("kanban-ultragoal", help="Durable Kanban-authority Ultragoal controller")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="kanban_ultragoal_cmd", required=True)

    start = sub.add_parser("start")
    start.add_argument("run_id")
    start.add_argument("--authority-json", required=True)
    start.add_argument("--root-objective", required=True)
    start.add_argument("--force", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("run_id")

    tick = sub.add_parser("tick")
    tick.add_argument("run_id")
    tick.add_argument("--authority-json", required=True)
    tick.add_argument("--budget-remaining", type=int, default=20)

    authority = sub.add_parser("authority-snapshot")
    authority.add_argument("task_ref")

    pilot = sub.add_parser("pilot-check")
    pilot.add_argument("task_ref")

    worker = sub.add_parser("record-worker-done")
    worker.add_argument("run_id")
    worker.add_argument("--authority-json", required=True)
    worker.add_argument("--evidence-json", required=True)

    verifier = sub.add_parser("record-verifier-result")
    verifier.add_argument("run_id")
    verifier.add_argument("--authority-json", required=True)
    verifier.add_argument("--result-json", required=True)

    reviewer = sub.add_parser("record-reviewer-result")
    reviewer.add_argument("run_id")
    reviewer.add_argument("--authority-json", required=True)
    reviewer.add_argument("--result-json", required=True)

    pr = sub.add_parser("record-pr-created")
    pr.add_argument("run_id")
    pr.add_argument("--authority-json", required=True)
    pr.add_argument("--pr-json", required=True)

    ci = sub.add_parser("record-ci-result")
    ci.add_argument("run_id")
    ci.add_argument("--authority-json", required=True)
    ci.add_argument("--ci-json", required=True)

    review_ready = sub.add_parser("mark-review-ready")
    review_ready.add_argument("run_id")
    review_ready.add_argument("--authority-json", required=True)
    return parser


def _emit(run: KanbanUltragoalRun, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(run.to_dict(), sort_keys=True, ensure_ascii=False))
    else:
        print(f"{run.run_id}: {run.state}")


def _emit_data(data: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(data, sort_keys=True, ensure_ascii=False))
    else:
        print(_canonical_json(data))


def kanban_ultragoal_command(args: argparse.Namespace) -> int:
    store = KanbanUltragoalStore(Path(args.workdir))
    cmd = args.kanban_ultragoal_cmd
    if cmd == "start":
        run = store.start(args.run_id, authority=_load_json_arg(args.authority_json) or {}, root_objective=args.root_objective, force=args.force)
    elif cmd == "status":
        run = store.load_run(args.run_id)
    elif cmd == "tick":
        run = store.tick(args.run_id, authority=_load_json_arg(args.authority_json) or {}, budget_remaining=args.budget_remaining)
    elif cmd == "authority-snapshot":
        _emit_data(build_authority_snapshot(args.task_ref), json_output=bool(args.json))
        return 0
    elif cmd == "pilot-check":
        data = pilot_check(args.task_ref)
        _emit_data(data, json_output=bool(args.json))
        return 0 if data["eligible"] else 2
    elif cmd == "record-worker-done":
        run = store.record_worker_done(args.run_id, authority=_load_json_arg(args.authority_json) or {}, evidence=_load_json_arg(args.evidence_json) or {})
    elif cmd == "record-verifier-result":
        run = store.record_verifier_result(args.run_id, authority=_load_json_arg(args.authority_json) or {}, result=_load_json_arg(args.result_json) or {})
    elif cmd == "record-reviewer-result":
        run = store.record_reviewer_result(args.run_id, authority=_load_json_arg(args.authority_json) or {}, result=_load_json_arg(args.result_json) or {})
    elif cmd == "record-pr-created":
        run = store.record_pr_created(args.run_id, authority=_load_json_arg(args.authority_json) or {}, pr=_load_json_arg(args.pr_json) or {})
    elif cmd == "record-ci-result":
        run = store.record_ci_result(args.run_id, authority=_load_json_arg(args.authority_json) or {}, ci=_load_json_arg(args.ci_json) or {})
    elif cmd == "mark-review-ready":
        run = store.mark_review_ready(args.run_id, authority=_load_json_arg(args.authority_json) or {})
    else:  # pragma: no cover
        raise SystemExit(f"unknown kanban-ultragoal command: {cmd}")
    _emit(run, json_output=bool(args.json))
    return 0


__all__ = [
    "KanbanUltragoalRun",
    "KanbanUltragoalStore",
    "build_authority_snapshot",
    "pilot_check",
    "build_parser",
    "kanban_ultragoal_command",
]
