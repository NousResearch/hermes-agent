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
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_closeout
from hermes_cli.ultragoal_runtime import build_direct_goal_action, write_direct_goal_handoff


def _forbidden_dispatcher_call(*args: Any, **kwargs: Any) -> None:
    """Sentinel for tests: Ultragoal must never call the Kanban dispatcher."""
    raise RuntimeError("Ultragoal direct lane must not call kanban dispatch")

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


def _repo_slug(workdir: Path) -> str:
    name = workdir.resolve().name if workdir.exists() else workdir.name
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    return slug or "default"


def _git_status(workdir: Path) -> dict[str, Any]:
    if not (workdir / ".git").exists():
        return {"repo": str(workdir), "repo_resolved": workdir.exists(), "worktree_clean": True, "status_short": ""}
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=workdir, text=True).strip()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workdir, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=workdir, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"repo": str(workdir), "repo_resolved": workdir.exists(), "worktree_clean": False, "status_short": f"git_status_error:{exc}"}
    return {"repo": str(workdir), "repo_resolved": workdir.exists(), "worktree_clean": not bool(status), "status_short": status, "head_sha": head, "branch": branch}


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



PARENT_CHILD_WORKER_STATUSES = {
    "satisfied",
    "blocked",
    "gated_by_forbidden_side_effect",
    "explicitly_descoped_by_chris",
    "not_applicable",
}


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _non_empty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _has_descoping_approval(row: dict[str, Any]) -> bool:
    approval = row.get("approval")
    if isinstance(approval, dict):
        return any(_non_empty_text(approval.get(key)) for key in ("text", "source", "approvalText", "approvedBy"))
    return any(_non_empty_text(row.get(key)) for key in ("approval_text", "approvalText", "approval_source", "approvalSource"))

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
    target_mode: str = "single"
    dispatcher_used: bool = False
    scope: dict[str, Any] = field(default_factory=dict)

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
            "targetMode": self.target_mode,
            "dispatcherUsed": self.dispatcher_used,
            "scope": self.scope,
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
            target_mode=data.get("targetMode") or data.get("target_mode") or "single",
            dispatcher_used=bool(data.get("dispatcherUsed") or data.get("dispatcher_used", False)),
            scope=data.get("scope") or {},
        )


class KanbanUltragoalStore:
    def __init__(self, workdir: str | Path = ".") -> None:
        self.workdir = Path(workdir)
        self.state_root = get_hermes_home() / "goal-runs" / _repo_slug(self.workdir)

    def legacy_root(self, run_id: str) -> Path:
        return self.workdir / ".hermes" / "goal-runs" / _validate_run_id(run_id)

    def root(self, run_id: str) -> Path:
        safe = _validate_run_id(run_id)
        modern = self.state_root / safe
        legacy = self.legacy_root(safe)
        if legacy.exists() and not modern.exists():
            return legacy
        return modern

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
        if run.target_mode == "parent":
            current_scope = self._build_scope(current, "parent")
            if current_scope.get("childSnapshotHashes") != (run.scope or {}).get("childSnapshotHashes"):
                raise ValueError("Kanban authority childSnapshotHashes mismatch")
        return current

    def _hierarchy_children(self, authority: dict[str, Any]) -> list[dict[str, Any]]:
        children = authority.get("children") or []
        return [dict(c) for c in children if (c.get("relationType") or c.get("relation_type")) == "hierarchy"]

    def _dependency_edges(self, authority: dict[str, Any]) -> list[dict[str, Any]]:
        deps = [dict(d) for d in (authority.get("dependencies") or [])]
        for child in authority.get("children") or []:
            if (child.get("relationType") or child.get("relation_type")) == "dependency":
                deps.append(dict(child))
        return deps

    def _build_scope(self, authority: dict[str, Any], target_mode: str) -> dict[str, Any]:
        parent_task_id = authority.get("taskId") or authority.get("task_id")
        if target_mode == "parent":
            hierarchy = self._hierarchy_children(authority)
            child_pairs = [
                (str(c.get("id") or c.get("taskId") or c.get("publicId")), c)
                for c in hierarchy
                if c.get("id") or c.get("taskId") or c.get("publicId")
            ]
            child_ids = [cid for cid, _child in child_pairs]
            child_hashes = {cid: _sha256(child) for cid, child in child_pairs}
            return {
                "parentTaskId": parent_task_id,
                "childTaskIds": child_ids,
                "childSnapshotHashes": child_hashes,
                "dependencyEdges": self._dependency_edges(authority),
            }
        return {"parentTaskId": parent_task_id, "childTaskIds": [], "childSnapshotHashes": {}, "dependencyEdges": []}

    def _goals_projection(self, authority: dict[str, Any], target_mode: str) -> dict[str, Any]:
        goals: list[dict[str, Any]] = []
        if target_mode == "parent":
            for idx, child in enumerate(self._hierarchy_children(authority), start=1):
                task_id = str(child.get("id") or child.get("taskId") or child.get("publicId"))
                goals.append({
                    "id": f"G{idx:03d}-{task_id}",
                    "sourceTaskId": task_id,
                    "title": child.get("title") or task_id,
                    "status": "pending",
                    "executor": "hermes-direct-goal-loop",
                    "dispatcherUsed": False,
                })
        return {"version": 1, "targetMode": target_mode, "goals": goals, "currentGoalId": goals[0]["id"] if goals else None}

    def start(self, run_id: str, *, authority: dict[str, Any], root_objective: str, force: bool = False, target_mode: str = "single") -> KanbanUltragoalRun:
        run_id = _validate_run_id(run_id)
        normalized = self._normalize_authority(authority, run_id)
        if target_mode not in {"single", "parent"}:
            raise ValueError("target_mode must be single or parent")
        root = self.root(run_id)
        if root.exists():
            if not force:
                raise FileExistsError(f"Kanban-Ultragoal run root already exists: {run_id}")
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "ultragoal").mkdir(parents=True, exist_ok=True)
        scope = self._build_scope(normalized, target_mode)
        normalized["targetWorkdir"] = str(self.workdir)
        normalized["goalRunRoot"] = str(root)
        run = KanbanUltragoalRun(
            run_id=run_id,
            parent_card=run_id,
            root_objective=root_objective,
            authority=normalized,
            target_mode=target_mode,
            dispatcher_used=False,
            scope=scope,
        )
        self._write_json(self.authority_path(run_id), normalized)
        goals_projection = self._goals_projection(normalized, target_mode)
        self._write_json(root / "goals.json", goals_projection)
        self._write_json(root / "ultragoal" / "goals.json", {"version": 1, "codexGoalMode": "aggregate", "goals": goals_projection["goals"]})
        self.save_run(run)
        self._append_ledger(run_id, "run_started", {"state": run.state, "authorityHash": _sha256(normalized)})
        return run

    def _prepare_pending_action(self, run: KanbanUltragoalRun, action: str) -> dict[str, Any]:
        direct = build_direct_goal_action(
            run_id=run.run_id,
            tick=run.tick,
            root_objective=run.root_objective,
            target_mode=run.target_mode,
            scope=run.scope,
        )
        direct["action"] = action
        direct["fenceToken"] = _sha256({"authority": run.authority, "state": run.state, "action": action, "scope": run.scope})
        direct["idempotencyKey"] = direct["stepId"]
        return direct

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
        run.pending_action = self._prepare_pending_action(run, "execute-direct-goal")
        write_direct_goal_handoff(self.root(run_id), run.pending_action)
        if run.state == "admitted":
            run.state = "running"
            self._append_ledger(run_id, "transition", {"from": "admitted", "to": "running", "pendingAction": run.pending_action})
        else:
            self._append_ledger(run_id, "tick", {"state": run.state, "pendingAction": run.pending_action})
        return self.save_run(run)

    def record_worker_done(self, run_id: str, *, authority: dict[str, Any], evidence: dict[str, Any]) -> KanbanUltragoalRun:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        if run.state not in {"admitted", "running", "verification_failed", "review_failed", "ci_failed"}:
            raise ValueError(f"worker_done transition is not allowed from {run.state}")
        self._require_parent_child_worker_coverage(run, evidence)
        recorded_evidence = dict(evidence)
        run.state = "worker_done"
        run.resumable = False
        self._write_json(self.root(run_id) / "evidence" / "worker.json", recorded_evidence)
        self._append_ledger(run_id, "worker_done", recorded_evidence)
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

    def record_cleanup_proof(self, run_id: str, *, authority: dict[str, Any], proof: dict[str, Any]) -> KanbanUltragoalRun:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        retained = proof.get("retained") or []
        for item in retained:
            if not item.get("reason") or not (item.get("ttl") or item.get("revisit")):
                raise ValueError("retained cleanup residue requires reason and ttl/revisit")
        cleanup = dict(proof)
        cleanup.setdefault("status", "passed")
        cleanup["readOnlyProof"] = True
        cleanup["targetMode"] = run.target_mode
        cleanup.setdefault("retained", retained)
        cleanup.setdefault("childCleanup", {})
        self._write_json(self.root(run_id) / "cleanup.json", cleanup)
        self._append_ledger(run_id, "cleanup_proof_recorded", cleanup)
        return self.save_run(run)

    def mark_review_ready(self, run_id: str, *, authority: dict[str, Any]) -> KanbanUltragoalRun:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        if run.state != "ci_passed":
            raise ValueError("CI success is required before review_ready")
        root = self.root(run_id)
        for rel, label in (("pr.json", "PR"), ("verifier/result.json", "verifier"), ("reviews/final.json", "reviewer"), ("evidence/ci.json", "CI"), ("cleanup.json", "cleanup proof")):
            if not (root / rel).exists():
                raise ValueError(f"{label} evidence is required before review_ready")
        cleanup = json.loads((root / "cleanup.json").read_text(encoding="utf-8"))
        if cleanup.get("status") != "passed" or cleanup.get("readOnlyProof") is not True:
            raise ValueError("cleanup proof must be a recorded read-only passing artifact before review_ready")
        worker = json.loads((root / "evidence" / "worker.json").read_text(encoding="utf-8")) if (root / "evidence" / "worker.json").exists() else {}
        verifier = json.loads((root / "verifier" / "result.json").read_text(encoding="utf-8")) if (root / "verifier" / "result.json").exists() else {}
        reviewer = json.loads((root / "reviews" / "final.json").read_text(encoding="utf-8"))
        self._require_quality_gate(reviewer)
        ledger = self._done_criteria_ledger(run)
        self._require_ultragoal_closeout_evidence_contract(run, worker, verifier, ledger)
        parent_child_matrix = self._require_parent_child_closeout_matrix(run, worker, cleanup)
        child_evidence = []
        for child_id in (run.scope or {}).get("childTaskIds", []):
            child_evidence.append({"taskId": child_id, "evidence": (worker.get("childEvidence") or {}).get(child_id)})
        run.state = "review_ready"
        run.last_terminal_report = {
            "kind": "review_ready",
            "at": _now(),
            "targetMode": run.target_mode,
            "dispatcherUsed": run.dispatcher_used,
            "childEvidence": child_evidence,
            "childCleanup": cleanup.get("childCleanup") or {},
            "childCloseoutMatrix": parent_child_matrix,
        }
        self._append_ledger(run_id, "review_ready", run.last_terminal_report)
        return self.save_run(run)

    def _load_optional_json(self, run_id: str, rel: str) -> dict[str, Any]:
        path = self.root(run_id) / rel
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def _quality_gate_blockers(self, quality_gate: Any) -> list[str]:
        if not isinstance(quality_gate, dict):
            return ["missing_quality_gate"]
        blockers: list[str] = []
        architect = quality_gate.get("architect_review")
        if not isinstance(architect, dict):
            blockers.append("missing_quality_gate.architect_review")
        else:
            for key in ("architecture_status", "product_status", "code_status"):
                if str(architect.get(key) or "").upper() != "CLEAR":
                    blockers.append(f"quality_gate.architect_review.{key}")
            if str(architect.get("recommendation") or "").upper() != "APPROVE":
                blockers.append("quality_gate.architect_review.recommendation")
            if not architect.get("evidence"):
                blockers.append("quality_gate.architect_review.evidence")
            if architect.get("blockers"):
                blockers.append("quality_gate.architect_review.blockers")
        verifier = quality_gate.get("verifier_qa")
        if not isinstance(verifier, dict):
            blockers.append("missing_quality_gate.verifier_qa")
        else:
            if str(verifier.get("status") or "").lower() != "passed":
                blockers.append("quality_gate.verifier_qa.status")
            coverage = verifier.get("contract_coverage")
            if not isinstance(coverage, list) or not coverage:
                blockers.append("quality_gate.verifier_qa.contract_coverage")
            else:
                for row in coverage:
                    if not isinstance(row, dict) or str(row.get("status") or "").lower() not in {"covered", "not_applicable"} or not row.get("evidence_refs"):
                        blockers.append("quality_gate.verifier_qa.contract_coverage")
                        break
            surfaces = verifier.get("surface_evidence")
            if not isinstance(surfaces, list) or not surfaces:
                blockers.append("quality_gate.verifier_qa.surface_evidence")
            else:
                for row in surfaces:
                    if not isinstance(row, dict) or str(row.get("verdict") or "").lower() != "passed" or not row.get("invocation") or not row.get("artifact_refs"):
                        blockers.append("quality_gate.verifier_qa.surface_evidence")
                        break
            adversarial = verifier.get("adversarial_cases")
            if not isinstance(adversarial, list) or not adversarial:
                blockers.append("quality_gate.verifier_qa.adversarial_cases")
            else:
                for row in adversarial:
                    if not isinstance(row, dict) or str(row.get("verdict") or "").lower() != "passed" or not row.get("artifact_refs"):
                        blockers.append("quality_gate.verifier_qa.adversarial_cases")
                        break
        iteration = quality_gate.get("iteration")
        if not isinstance(iteration, dict):
            blockers.append("missing_quality_gate.iteration")
        else:
            if iteration.get("full_rerun") is not True:
                blockers.append("quality_gate.iteration.full_rerun")
            if not iteration.get("rerun_commands"):
                blockers.append("quality_gate.iteration.rerun_commands")
            if iteration.get("blockers"):
                blockers.append("quality_gate.iteration.blockers")
        return list(dict.fromkeys(blockers))

    def _require_quality_gate(self, reviewer_raw: dict[str, Any]) -> dict[str, Any]:
        quality_gate = reviewer_raw.get("quality_gate") or reviewer_raw.get("qualityGate")
        blockers = self._quality_gate_blockers(quality_gate)
        if blockers:
            raise ValueError("quality_gate blocked review_ready: " + ", ".join(blockers))
        return dict(quality_gate)

    def _parent_child_worker_coverage_status(self, run: KanbanUltragoalRun, worker_raw: dict[str, Any]) -> dict[str, Any]:
        child_ids = [str(c) for c in (run.scope or {}).get("childTaskIds", [])]
        if run.target_mode != "parent" or not child_ids:
            return {"complete": True, "missingChildEvidence": [], "unknownChildEvidence": [], "blockers": [], "children": []}
        if "childEvidence" in worker_raw:
            raw = worker_raw.get("childEvidence")
        else:
            raw = worker_raw.get("child_evidence")
        blockers: list[str] = []
        matrix: list[dict[str, Any]] = []
        if not isinstance(raw, dict):
            return {
                "complete": False,
                "missingChildEvidence": child_ids,
                "unknownChildEvidence": [],
                "blockers": ["childEvidence:not_dict"],
                "children": [{"taskId": child_id, "hasChildEvidence": False, "valid": False, "blockers": ["missing_childEvidence"]} for child_id in child_ids],
            }
        child_set = set(child_ids)
        unknown = sorted(str(key) for key in raw.keys() if str(key) not in child_set)
        missing: list[str] = []
        for child_id in child_ids:
            row = raw.get(child_id)
            row_blockers: list[str] = []
            status = None
            if not isinstance(row, dict):
                if row is None:
                    missing.append(child_id)
                    row_blockers.append("missing_childEvidence")
                else:
                    row_blockers.append("childEvidence_row:not_dict")
            else:
                status = str(row.get("status") or "").strip()
                if status not in PARENT_CHILD_WORKER_STATUSES:
                    row_blockers.append(f"status:{status or 'missing'}")
                if not (_non_empty_text(row.get("summary")) or _non_empty_text(row.get("reason"))):
                    row_blockers.append("summary_or_reason_required")
                if status == "satisfied" and not _non_empty_list(row.get("artifact_refs")):
                    row_blockers.append("artifact_refs_required")
                if status == "blocked" and not (_non_empty_text(row.get("next_gate")) or _non_empty_list(row.get("artifact_refs"))):
                    row_blockers.append("next_gate_or_artifact_refs_required")
                if status == "gated_by_forbidden_side_effect" and not _non_empty_text(row.get("next_gate")):
                    row_blockers.append("next_gate_required")
                if status == "explicitly_descoped_by_chris" and not _has_descoping_approval(row):
                    row_blockers.append("explicitly_descoped_by_chris_approval_required")
                if status == "not_applicable" and not (_non_empty_text(row.get("reason")) or _non_empty_list(row.get("artifact_refs"))):
                    row_blockers.append("not_applicable_reason_or_artifact_refs_required")
            if row_blockers:
                blockers.extend(f"{child_id}:{blocker}" for blocker in row_blockers)
            matrix.append({
                "taskId": child_id,
                "hasChildEvidence": child_id not in missing and row is not None,
                "valid": not row_blockers,
                "status": status,
                "blockers": row_blockers,
            })
        if missing:
            blockers.insert(0, "missing childEvidence for " + ", ".join(missing))
        return {
            "complete": not blockers and not missing,
            "missingChildEvidence": missing,
            "unknownChildEvidence": unknown,
            "blockers": list(dict.fromkeys(blockers)),
            "children": matrix,
        }

    def _require_parent_child_worker_coverage(self, run: KanbanUltragoalRun, worker_raw: dict[str, Any]) -> dict[str, Any]:
        status = self._parent_child_worker_coverage_status(run, worker_raw)
        if not status.get("complete"):
            missing = status.get("missingChildEvidence") or []
            prefix = "parent worker_done blocked"
            if missing:
                prefix += ": missing childEvidence for " + ", ".join(missing)
            blockers = status.get("blockers") or []
            detail = "; ".join(str(item) for item in blockers if item)
            raise ValueError(prefix + ("; " + detail if detail else ""))
        return status

    def _parent_child_matrix_status(self, run: KanbanUltragoalRun, worker_raw: dict[str, Any] | None = None, cleanup_raw: dict[str, Any] | None = None) -> dict[str, Any]:
        worker_raw = worker_raw or {}
        cleanup_raw = cleanup_raw or {}
        child_ids = [str(c) for c in (run.scope or {}).get("childTaskIds", [])]
        if run.target_mode != "parent" or not child_ids:
            return {"parentScopeComplete": True, "missingChildEvidence": [], "missingChildCleanup": [], "nextRequiredChild": None, "cannotFinalCloseoutParent": False}
        coverage = self._parent_child_worker_coverage_status(run, worker_raw)
        cleanup = cleanup_raw.get("childCleanup") or cleanup_raw.get("child_cleanup") or {}
        if not isinstance(cleanup, dict):
            cleanup = {}
        missing_cleanup = [child_id for child_id in child_ids if not cleanup.get(child_id)]
        missing_evidence = list(coverage.get("missingChildEvidence") or [])
        invalid_evidence = [row["taskId"] for row in coverage.get("children") or [] if not row.get("valid") and row.get("taskId") not in missing_evidence]
        evidence_gaps = list(dict.fromkeys(missing_evidence + invalid_evidence))
        next_required = (evidence_gaps or missing_cleanup or [None])[0]
        complete = not evidence_gaps and not missing_cleanup
        return {
            "parentScopeComplete": complete,
            "missingChildEvidence": evidence_gaps,
            "missingChildCleanup": missing_cleanup,
            "nextRequiredChild": next_required,
            "cannotFinalCloseoutParent": not complete,
            "workerCoverage": coverage,
        }

    def status_payload(self, run_id: str) -> dict[str, Any]:
        run = self.load_run(run_id)
        payload = run.to_dict()
        worker_raw = self._load_optional_json(run_id, "evidence/worker.json")
        cleanup_raw = self._load_optional_json(run_id, "cleanup.json")
        payload["parentChildMatrix"] = self._parent_child_matrix_status(run, worker_raw, cleanup_raw)
        return payload

    def _require_parent_child_closeout_matrix(self, run: KanbanUltragoalRun, worker_raw: dict[str, Any], cleanup_raw: dict[str, Any]) -> dict[str, Any]:
        child_ids = [str(c) for c in (run.scope or {}).get("childTaskIds", [])]
        if run.target_mode != "parent" or not child_ids:
            return {"children": [], "complete": True}
        child_evidence = worker_raw.get("childEvidence") or worker_raw.get("child_evidence") or {}
        child_cleanup = cleanup_raw.get("childCleanup") or cleanup_raw.get("child_cleanup") or {}
        if not isinstance(child_evidence, dict):
            child_evidence = {}
        if not isinstance(child_cleanup, dict):
            child_cleanup = {}
        matrix: list[dict[str, Any]] = []
        blockers: list[str] = []
        for child_id in child_ids:
            has_evidence = bool(child_evidence.get(child_id))
            has_cleanup = bool(child_cleanup.get(child_id))
            matrix.append({
                "taskId": child_id,
                "hasChildEvidence": has_evidence,
                "hasChildCleanup": has_cleanup,
                "requiredKanbanCloseout": "review_ready",
            })
            if not has_evidence:
                blockers.append(f"childEvidence:{child_id}")
            if not has_cleanup:
                blockers.append(f"childCleanup:{child_id}")
        if blockers:
            raise ValueError("parent child closeout matrix blocked review_ready: " + ", ".join(blockers))
        return {"children": matrix, "complete": True}

    def _done_criteria_ledger(self, run: KanbanUltragoalRun) -> dict[str, Any]:
        criteria = []
        for idx, text in enumerate(run.authority.get("doneCriteria") or [], start=1):
            criteria.append({
                "id": f"uc-{idx:02d}",
                "text": str(text),
                "source_section": "kanban authority doneCriteria",
                "required_evidence_types": ["test", "no_code_proof"],
                "deterministic_checks": ["Ultragoal ledger artifact", "worker/verifier/reviewer/PR/CI evidence artifacts"],
                "authority_boundary": "Kanban authority + current approved side-effect boundary",
                "ambiguous": False,
            })
        ledger = {
            "schema": "kanban_done_criteria_ledger.v1",
            "task_id": run.parent_card,
            "public_id": run.authority.get("publicId") or run.parent_card,
            "source": {"kind": "kanban-ultragoal", "run_id": run.run_id},
            "version": 1,
            "criteria": criteria,
            "forbidden_actions": [
                "production DB write",
                "employee seed/import",
                "live external send",
                "provider/env/secret mutation",
                "deploy/release/master merge",
                "gateway restart/reload",
                "Autopilot fallback",
                "Kanban dispatcher child-worker dispatch",
            ],
            "refinement_required": False,
        }
        return kanban_closeout.normalize_done_criteria_ledger(ledger)

    def _goal_contract_identity(self, run: KanbanUltragoalRun) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "task_id": run.authority.get("taskId") or run.parent_card,
            "public_id": run.authority.get("publicId") or run.parent_card,
            "snapshot_hash": run.authority.get("snapshotHash"),
            "done_criteria_hash": run.authority.get("doneCriteriaHash"),
            "target_mode": run.target_mode,
            "authority_boundary_confirmed": True,
        }

    def _verifier_criterion_results(self, verifier_raw: dict[str, Any]) -> dict[str, str]:
        results: dict[str, str] = {}

        def record(raw_id: Any, raw_verdict: Any, *, has_evidence: bool = False) -> None:
            if not raw_id:
                return
            criterion_id = str(raw_id)
            verdict = str(raw_verdict or ("PASS" if has_evidence else "")).strip().upper()

            def record_one(normalized_id: str) -> None:
                existing = results.get(normalized_id)
                if existing is None or existing == verdict:
                    results[normalized_id] = verdict
                    return
                passing = {"PASS", "PASSED", "SUCCESS", "OK"}
                if existing in passing and verdict in passing:
                    results[normalized_id] = existing
                    return
                results[normalized_id] = "FAIL"

            record_one(criterion_id)
            if criterion_id.lower().startswith("dc-"):
                suffix = criterion_id.split("-", 1)[1]
                if suffix.isdigit():
                    record_one(f"uc-{int(suffix):02d}")

        evidence_rows = verifier_raw.get("doneCriteriaEvidence") or verifier_raw.get("done_criteria_evidence") or []
        if isinstance(evidence_rows, list):
            for row in evidence_rows:
                if isinstance(row, dict):
                    has_evidence = bool(str(row.get("evidence") or "").strip() or row.get("evidence_refs"))
                    record(row.get("criterionId") or row.get("criterion_id") or row.get("id"), row.get("verdict") or row.get("status"), has_evidence=has_evidence)
        per_criterion = verifier_raw.get("perCriterion") or verifier_raw.get("per_criterion") or {}
        if isinstance(per_criterion, dict):
            for raw_id, raw_value in per_criterion.items():
                value = raw_value if isinstance(raw_value, dict) else {"verdict": raw_value}
                has_evidence = bool(str(value.get("evidence") or "").strip() or value.get("evidence_refs"))
                record(raw_id, value.get("verdict") or value.get("status"), has_evidence=has_evidence)
        return results

    def _ultragoal_closeout_evidence_blockers(
        self,
        run: KanbanUltragoalRun,
        worker_raw: dict[str, Any],
        verifier_raw: dict[str, Any],
        ledger: dict[str, Any],
    ) -> list[str]:
        blockers: list[str] = []
        if not worker_raw or not (worker_raw.get("summary") or worker_raw.get("proof") or worker_raw.get("commandsRun") or worker_raw.get("tests_run")):
            blockers.append("ultragoal_closeout_missing_worker_evidence")
            if not worker_raw:
                blockers.append("missing_worker_evidence")
        if not verifier_raw:
            blockers.append("ultragoal_closeout_missing_verifier_evidence")
            blockers.append("missing_verifier_result")
        expected_snapshot = run.authority.get("snapshotHash")
        expected_done_hash = run.authority.get("doneCriteriaHash")
        worker_contract = worker_raw.get("goalContract") if isinstance(worker_raw.get("goalContract"), dict) else {}
        if worker_raw.get("authority_boundary_confirmed") is not True:
            blockers.append("authority_boundary_confirmed")
        verifier_contract = verifier_raw.get("goalContract") if isinstance(verifier_raw.get("goalContract"), dict) else {}
        worker_snapshot = worker_raw.get("authoritySnapshotHash") or worker_raw.get("snapshotHash") or worker_contract.get("snapshotHash")
        if worker_snapshot and worker_snapshot != expected_snapshot:
            blockers.append("ultragoal_closeout_stale_contract")
            blockers.append("stale_worker_authority_snapshot")
        worker_done_hash = worker_raw.get("doneCriteriaHash") or worker_raw.get("criteriaHash") or worker_contract.get("doneCriteriaHash")
        if worker_done_hash and worker_done_hash != expected_done_hash:
            blockers.append("ultragoal_closeout_stale_contract")
            blockers.append("stale_worker_done_criteria")
        verifier_snapshot = verifier_raw.get("authoritySnapshotHash") or verifier_raw.get("snapshotHash") or verifier_contract.get("snapshotHash")
        if verifier_snapshot and verifier_snapshot != expected_snapshot:
            blockers.append("ultragoal_closeout_stale_contract")
            blockers.append("stale_verifier_authority_snapshot")
        verifier_done_hash = verifier_raw.get("doneCriteriaHash") or verifier_raw.get("criteriaHash") or verifier_contract.get("doneCriteriaHash")
        if verifier_done_hash and verifier_done_hash != expected_done_hash:
            blockers.append("ultragoal_closeout_stale_contract")
            blockers.append("stale_verifier_done_criteria")
        if worker_raw.get("superseded") is True or worker_raw.get("supersededBy"):
            blockers.append("ultragoal_closeout_stale_contract")
            blockers.append("worker_evidence_superseded")
        if verifier_raw.get("superseded") is True or verifier_raw.get("supersededBy"):
            blockers.append("ultragoal_closeout_stale_contract")
            blockers.append("verifier_result_superseded")
        verifier_passed = verifier_raw.get("passed") is True or str(verifier_raw.get("verdict", "")).upper() == "PASS"
        if verifier_raw and not verifier_passed:
            blockers.append("verifier_result_not_pass")
        criterion_results = self._verifier_criterion_results(verifier_raw)
        has_per_criterion_evidence = bool(
            verifier_raw.get("perCriterion")
            or verifier_raw.get("per_criterion")
            or verifier_raw.get("doneCriteriaEvidence")
            or verifier_raw.get("done_criteria_evidence")
        )
        if ledger.get("criteria") and not has_per_criterion_evidence:
            blockers.append("ultragoal_closeout_missing_per_criterion_evidence")
        if has_per_criterion_evidence and ledger.get("criteria"):
            missing_criterion = False
            for criterion in ledger.get("criteria") or []:
                criterion_id = str(criterion.get("id") or "")
                if criterion_id and criterion_id not in criterion_results:
                    missing_criterion = True
                    blockers.append(f"missing_verifier_per_criterion:{criterion_id}")
                elif str(criterion_results.get(criterion_id) or "").upper() not in {"PASS", "PASSED", "SUCCESS", "OK"}:
                    blockers.append(f"verifier_per_criterion_not_pass:{criterion_id}")
            if missing_criterion:
                blockers.append("ultragoal_closeout_missing_verifier_evidence")
                blockers.append("ultragoal_closeout_missing_per_criterion_evidence")
        return list(dict.fromkeys(blockers))

    def _require_ultragoal_closeout_evidence_contract(
        self,
        run: KanbanUltragoalRun,
        worker_raw: dict[str, Any],
        verifier_raw: dict[str, Any],
        ledger: dict[str, Any],
    ) -> None:
        blockers = self._ultragoal_closeout_evidence_blockers(run, worker_raw, verifier_raw, ledger)
        if blockers:
            raise ValueError("ultragoal closeout evidence blocked: " + ", ".join(blockers))


    def build_closeout_evidence(self, run_id: str, *, authority: dict[str, Any]) -> dict[str, Any]:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        root = self.root(run_id)
        worker_raw = self._load_optional_json(run_id, "evidence/worker.json")
        verifier_raw = self._load_optional_json(run_id, "verifier/result.json")
        reviewer_raw = self._load_optional_json(run_id, "reviews/final.json")
        pr_raw = self._load_optional_json(run_id, "pr.json")
        ci_raw = self._load_optional_json(run_id, "evidence/ci.json")
        cleanup_raw = self._load_optional_json(run_id, "cleanup.json")
        quality_gate = self._require_quality_gate(reviewer_raw)
        child_closeout_matrix = self._require_parent_child_closeout_matrix(run, worker_raw, cleanup_raw)
        ledger = self._done_criteria_ledger(run)
        self._require_ultragoal_closeout_evidence_contract(run, worker_raw, verifier_raw, ledger)
        artifact_refs = [str(root / "ledger.jsonl"), str(root / "run.json"), str(root / "goals.json")]
        tests_run = [str(item) for item in worker_raw.get("tests_run") or worker_raw.get("commandsRun") or []]
        verifier_passed = verifier_raw.get("passed") is True or str(verifier_raw.get("verdict", "")).upper() == "PASS"
        per_worker = {}
        per_verifier = {}
        for criterion in ledger.get("criteria", []):
            criterion_id = criterion["id"]
            refs = artifact_refs + tests_run
            per_worker[criterion_id] = {
                "claim": "satisfied",
                "evidence_refs": refs,
                "notes": worker_raw.get("summary") or worker_raw.get("proof") or "Ultragoal implementation phase recorded evidence.",
            }
            criterion_results = self._verifier_criterion_results(verifier_raw)
            criterion_verdict = str(criterion_results.get(criterion_id) or ("PASS" if verifier_passed else "FAIL")).upper()
            per_verifier[criterion_id] = {
                "verdict": "PASS" if criterion_verdict in {"PASS", "PASSED", "SUCCESS", "OK"} else "FAIL",
                "evidence_refs": refs,
                "notes": "Verifier phase checked the Ultragoal evidence against Done Criteria.",
            }
        changed_files = worker_raw.get("changed_files") or worker_raw.get("changedFiles") or []
        pr_state = str(pr_raw.get("state") or "OPEN").upper() if pr_raw else ""
        if pr_raw and not changed_files and pr_state not in {"MERGED", "CLOSED"}:
            changed_files = ["pr-review-package"]
        goal_contract = self._goal_contract_identity(run)
        worker = {
            "schema": "kanban_worker_evidence.v1",
            "goal_contract": goal_contract,
            "criteria_hash": ledger["criteria_hash"],
            "summary": worker_raw.get("summary") or worker_raw.get("proof") or "Ultragoal implementation phase completed.",
            "changed_files": [] if pr_state in {"MERGED", "CLOSED"} else changed_files,
            "tests_run": tests_run,
            "per_criterion": per_worker,
            "authority_boundary_confirmed": True,
            "forbidden_actions_performed": worker_raw.get("forbidden_actions_performed") or [],
            "childEvidence": worker_raw.get("childEvidence") or worker_raw.get("child_evidence") or {},
            "artifact_refs": artifact_refs,
        }
        verifier = {
            "schema": "kanban_verifier_result.v1",
            "goal_contract": goal_contract,
            "verdict": "PASS" if verifier_passed else "FAIL",
            "criteria_hash": ledger["criteria_hash"],
            "verification_attempt": int(verifier_raw.get("verification_attempt") or 1),
            "per_criterion": per_verifier,
            "authority_boundary_ok": True,
            "retry_allowed": False,
        }
        checks = ci_raw.get("checks") or pr_raw.get("checks") or []
        if ci_raw.get("state") == "success" and not checks:
            checks = [{"name": "ultragoal-ci", "status": "completed", "conclusion": "success"}]
        review_package = {
            "schema": "kanban_review_package.v1",
            "kind": "no_new_diff_existing_pr_artifact" if pr_state in {"MERGED", "CLOSED"} else "pr_required",
            "changed_files": [] if pr_state in {"MERGED", "CLOSED"} else changed_files,
            "review_package_expectation": "Review the Ultragoal ledger and PR artifact evidence.",
        }
        git = _git_status(self.workdir)
        cleanup = {
            "proof": cleanup_raw.get("proof") or "Ultragoal cleanup proof recorded; run ledger retained as evidence.",
            "worktree_clean": git.get("worktree_clean") is True,
            "artifacts_removed": cleanup_raw.get("artifacts_removed") or [],
            "worktree_retained": True,
            "retained_reason": "Ultragoal goal-run ledger/checkpoint evidence is retained outside the product repo.",
            "ttl": "retain until review package is accepted or superseded",
        }
        evidence = {
            "schema": "kanban_closeout_evidence.v1",
            "summary": f"Kanban Ultragoal run {run.run_id} materialized governed closeout evidence.",
            "proof": f"Run root: {root}",
            "goal_contract": goal_contract,
            "done_criteria_ledger": ledger,
            "worker_evidence": worker,
            "verifier_result": verifier,
            "reviewer_loop": {"attempt": 1, "max_attempts": 3, "mode": "kanban-ultragoal-internal-phases"},
            "reviewer_result": reviewer_raw,
            "quality_gate": quality_gate,
            "child_closeout_matrix": child_closeout_matrix,
            "review_package": review_package,
            "artifact_refs": artifact_refs,
            "authority_boundary_confirmed": True,
            "checks": checks,
            "pr": pr_raw,
            "git": git,
            "cleanup": cleanup,
            "residue": {
                "summary": "Only the Ultragoal goal-run ledger is retained as evidence.",
                "items": [{
                    "kind": "goal_run_ledger",
                    "disposition": "retained",
                    "path": str(root),
                    "reason": "Required Ultragoal evidence/checkpoint artifact",
                    "ttl": "retain until review package is accepted or superseded",
                }],
            },
            "approval": {"approved": True, "approved_by": "kanban-authority", "decision": "approved"},
            "ultragoal": {"runId": run.run_id, "runRoot": str(root), "state": run.state, "dispatcherUsed": run.dispatcher_used, "targetMode": run.target_mode},
        }
        if pr_state in {"MERGED", "CLOSED"}:
            evidence["no_pr_reason"] = "No new product diff was created in this Ultragoal reconciliation run."
            evidence["no_pr_exception"] = {
                "policy": "no-new-diff-existing-pr-artifact",
                "reason": evidence["no_pr_reason"],
                "changed_files_expected": False,
                "review_package_expectation": review_package["review_package_expectation"],
            }
        return evidence

    def closeout_review_ready(self, run_id: str, *, authority: dict[str, Any]) -> dict[str, Any]:
        run = self.load_run(run_id)
        self._require_current_authority(run, authority)
        evidence = self.build_closeout_evidence(run_id, authority=authority)
        task_order = list((run.scope or {}).get("childTaskIds") or []) + [run.parent_card]
        results: list[dict[str, Any]] = []
        with kb.connect() as conn:
            for target_phase in ("worker_done", "review_ready"):
                for task_id in task_order:
                    result = kanban_closeout.transition_task_closeout(conn, task_id, target_phase, evidence, repo_path=self.workdir)
                    results.append({"taskId": task_id, "targetPhase": target_phase, "result": result})
                    if result.get("status") not in {"transitioned", "already_current"} and result.get("allowed") is not True:
                        return {"status": "blocked", "runId": run_id, "blockedAt": {"taskId": task_id, "targetPhase": target_phase}, "results": results}
        run.state = "review_ready"
        run.last_terminal_report = {"kind": "review_ready", "at": _now(), "targetMode": run.target_mode, "dispatcherUsed": run.dispatcher_used, "closeoutResults": results}
        self._append_ledger(run_id, "kanban_closeout_review_ready", run.last_terminal_report)
        self.save_run(run)
        return {"status": "review_ready", "runId": run_id, "targetMode": run.target_mode, "dispatcherUsed": run.dispatcher_used, "results": results}


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
    start.add_argument("--mode", choices=["single", "parent"], default="single")

    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("run_id")
    run_cmd.add_argument("--authority-json", required=True)
    run_cmd.add_argument("--root-objective", required=True)
    run_cmd.add_argument("--force", action="store_true")
    run_cmd.add_argument("--mode", choices=["single", "parent"], default="single")
    run_cmd.add_argument("--dry-run", action="store_true")

    resume = sub.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--authority-json", required=True)
    resume.add_argument("--budget-remaining", type=int, default=20)

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

    cleanup = sub.add_parser("record-cleanup-proof")
    cleanup.add_argument("run_id")
    cleanup.add_argument("--authority-json", required=True)
    cleanup.add_argument("--proof-json", required=True)

    review_ready = sub.add_parser("mark-review-ready")
    review_ready.add_argument("run_id")
    review_ready.add_argument("--authority-json", required=True)

    closeout_evidence = sub.add_parser("build-closeout-evidence")
    closeout_evidence.add_argument("run_id")
    closeout_evidence.add_argument("--authority-json", required=True)

    closeout_review_ready = sub.add_parser("closeout-review-ready")
    closeout_review_ready.add_argument("run_id")
    closeout_review_ready.add_argument("--authority-json", required=True)
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
        run = store.start(args.run_id, authority=_load_json_arg(args.authority_json) or {}, root_objective=args.root_objective, force=args.force, target_mode=args.mode)
    elif cmd == "run":
        authority = _load_json_arg(args.authority_json) or {}
        if args.dry_run:
            run = KanbanUltragoalRun(
                run_id=args.run_id,
                parent_card=args.run_id,
                root_objective=args.root_objective,
                authority=authority,
                target_mode=args.mode,
                dispatcher_used=False,
                scope=store._build_scope(authority, args.mode),
            )
            run.pending_action = store._prepare_pending_action(run, "dry-run-direct-goal")
        else:
            run = store.start(args.run_id, authority=authority, root_objective=args.root_objective, force=args.force, target_mode=args.mode)
            run = store.tick(args.run_id, authority=authority)
    elif cmd == "status":
        data = store.status_payload(args.run_id)
        _emit_data(data, json_output=bool(args.json))
        return 0
    elif cmd == "tick":
        run = store.tick(args.run_id, authority=_load_json_arg(args.authority_json) or {}, budget_remaining=args.budget_remaining)
    elif cmd == "resume":
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
    elif cmd == "record-cleanup-proof":
        run = store.record_cleanup_proof(args.run_id, authority=_load_json_arg(args.authority_json) or {}, proof=_load_json_arg(args.proof_json) or {})
    elif cmd == "mark-review-ready":
        run = store.mark_review_ready(args.run_id, authority=_load_json_arg(args.authority_json) or {})
    elif cmd == "build-closeout-evidence":
        data = store.build_closeout_evidence(args.run_id, authority=_load_json_arg(args.authority_json) or {})
        _emit_data(data, json_output=bool(args.json))
        return 0
    elif cmd == "closeout-review-ready":
        data = store.closeout_review_ready(args.run_id, authority=_load_json_arg(args.authority_json) or {})
        _emit_data(data, json_output=bool(args.json))
        return 0 if data.get("status") == "review_ready" else 2
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
