"""Private HTTP coordinator for a distributed Hermes Kanban board.

The coordinator is deliberately small: it owns the SQLite connection and the
claim CAS, while workers on other machines poll it and run normal Hermes task
processes locally. It is meant to bind only on a private network such as a
Tailscale interface; bearer authentication remains mandatory.
"""

from __future__ import annotations

import contextlib
import hmac
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from hermes_cli import kanban_db as kb


class MachineRegistration(BaseModel):
    machine_id: str
    hostname: Optional[str] = None
    profiles: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)


class ClaimRequest(BaseModel):
    machine_id: str


class RenewRequest(BaseModel):
    claim_lock: str


class WorkerStartedRequest(BaseModel):
    claim_lock: str
    worker_pid: int = Field(gt=0)


class CompletionRequest(BaseModel):
    result: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    created_cards: list[str] = Field(default_factory=list)
    expected_run_id: Optional[int] = None
    claim_lock: str = ""


class BlockRequest(BaseModel):
    reason: Optional[str] = None
    kind: Optional[str] = None
    expected_run_id: Optional[int] = None
    claim_lock: str = ""


class CommentRequest(BaseModel):
    author: str
    body: str


class TaskCreateRequest(BaseModel):
    title: str
    body: Optional[str] = None
    assignee: Optional[str] = None
    created_by: Optional[str] = None
    workspace_kind: str = "scratch"
    workspace_path: Optional[str] = None
    branch_name: Optional[str] = None
    tenant: Optional[str] = None
    priority: int = 0
    parents: list[str] = Field(default_factory=list)
    triage: bool = False
    idempotency_key: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    skills: Optional[list[str]] = None
    max_retries: Optional[int] = None
    goal_mode: bool = False
    goal_max_turns: Optional[int] = None
    initial_status: str = "running"
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    target_machine: Optional[str] = None
    required_capabilities: list[str] = Field(default_factory=list)


class TaskListRequest(BaseModel):
    assignee: Optional[str] = None
    status: Optional[str] = None
    tenant: Optional[str] = None
    include_archived: bool = False
    limit: Optional[int] = None


class LinkRequest(BaseModel):
    parent_id: str
    child_id: str


def _task_payload(conn, task: kb.Task) -> dict:
    payload = asdict(task)
    payload["required_capabilities"] = list(kb.task_capabilities(conn, task.id))
    return payload


def _machine_uuid(value: str) -> str:
    """Normalize an external machine identity without leaking a 500 to clients."""
    try:
        return str(uuid.UUID(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=422, detail="machine_id must be a UUID"
        ) from exc


def _claim_authorizes(task: kb.Task, claim_lock: str) -> bool:
    """Fence running mutations while permitting orchestrator edits off-lease."""
    if task.status != "running":
        return not claim_lock
    return bool(claim_lock) and task.claim_lock == claim_lock


def _maintenance_once(db_path: Path) -> dict[str, int]:
    """Refresh coordinator presence and recover expired remote leases once."""
    with contextlib.closing(kb.connect(db_path)) as conn:
        kb.register_local_machine(conn)
        reclaimed = kb.release_stale_claims(conn)
        promoted = kb.recompute_ready(conn)
    return {"reclaimed": reclaimed, "promoted": promoted}


def create_app(*, db_path: Path, token: str) -> FastAPI:
    """Create an authenticated coordinator app for one explicit board DB."""
    if not token:
        raise ValueError("coordinator token is required")
    db_path = Path(db_path).expanduser()
    app = FastAPI(title="Hermes Kanban Coordinator", version="0.1")

    def require_token(authorization: Optional[str] = Header(default=None)) -> None:
        expected = f"Bearer {token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid coordinator token")

    @app.get("/v1/health")
    def health(authorization: Optional[str] = Header(default=None)) -> dict:
        require_token(authorization)
        return {"ok": True}

    @app.post("/v1/machines/register")
    def register(
        request: MachineRegistration,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        with contextlib.closing(kb.connect(db_path)) as conn:
            machine_id = kb.register_machine(
                conn,
                _machine_uuid(request.machine_id),
                hostname=request.hostname,
                profiles=request.profiles,
                capabilities=request.capabilities,
            )
        return {"machine_id": machine_id}

    @app.get("/v1/machines")
    def machines(authorization: Optional[str] = Header(default=None)) -> dict:
        require_token(authorization)
        with contextlib.closing(kb.connect(db_path)) as conn:
            now = int(time.time())
            rows = conn.execute(
                "SELECT id, hostname, last_seen_at FROM machines ORDER BY hostname, id"
            ).fetchall()
            profiles: dict[str, list[str]] = {}
            for row in conn.execute(
                "SELECT machine_id, profile FROM machine_profiles ORDER BY machine_id, profile"
            ).fetchall():
                profiles.setdefault(row["machine_id"], []).append(row["profile"])
            capabilities: dict[str, list[str]] = {}
            for row in conn.execute(
                "SELECT machine_id, capability FROM machine_capabilities ORDER BY machine_id, capability"
            ).fetchall():
                capabilities.setdefault(row["machine_id"], []).append(row["capability"])
            active = {
                row["machine_id"]: row["count"]
                for row in conn.execute(
                    "SELECT machine_id, COUNT(*) AS count FROM tasks "
                    "WHERE status = 'running' AND machine_id IS NOT NULL GROUP BY machine_id"
                ).fetchall()
            }
        payload = [
            {
                "machine_id": row["id"],
                "hostname": row["hostname"],
                "last_seen_at": row["last_seen_at"],
                "online": now - int(row["last_seen_at"]) <= 90,
                "profiles": profiles.get(row["id"], []),
                "capabilities": capabilities.get(row["id"], []),
                "active_workers": active.get(row["id"], 0),
            }
            for row in rows
        ]
        return {"machines": payload, "count": len(payload), "checked_at": now}

    @app.post("/v1/tasks/claim-next")
    def claim_next(
        request: ClaimRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        with contextlib.closing(kb.connect(db_path)) as conn:
            machine_id = _machine_uuid(request.machine_id)
            rows = conn.execute(
                "SELECT id FROM tasks WHERE status = 'ready' AND claim_lock IS NULL "
                "ORDER BY priority DESC, created_at ASC"
            ).fetchall()
            for row in rows:
                claimed = kb.claim_task(
                    conn,
                    row["id"],
                    machine_id=machine_id,
                    enforce_machine_routing=True,
                    require_registered_profile=True,
                )
                if claimed is not None:
                    return {"task": _task_payload(conn, claimed)}
        return {"task": None}

    @app.post("/v1/tasks/{task_id}/renew")
    def renew(
        task_id: str,
        request: RenewRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        with contextlib.closing(kb.connect(db_path)) as conn:
            renewed = kb.heartbeat_claim(conn, task_id, claimer=request.claim_lock)
            if renewed:
                # Lease renewal says the process still owns the claim; the
                # worker heartbeat also records that its process remains
                # alive, preventing stale-worker recovery from reclaiming a
                # healthy long-running remote task.
                kb.heartbeat_worker(conn, task_id)
                task = kb.get_task(conn, task_id)
                if task is not None and task.machine_id:
                    conn.execute(
                        "UPDATE machines SET last_seen_at = ? WHERE id = ?",
                        (int(time.time()), task.machine_id),
                    )
        return {"renewed": renewed}

    @app.post("/v1/tasks/{task_id}/worker-started")
    def worker_started(
        task_id: str,
        request: WorkerStartedRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        with contextlib.closing(kb.connect(db_path)) as conn:
            recorded = kb.record_worker_started(
                conn,
                task_id,
                claim_lock=request.claim_lock,
                worker_pid=request.worker_pid,
            )
        return {"recorded": recorded}

    @app.post("/v1/tasks")
    def create_task(
        request: TaskCreateRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        try:
            with contextlib.closing(kb.connect(db_path)) as conn:
                task_id = kb.create_task(
                    conn,
                    title=request.title,
                    body=request.body,
                    assignee=request.assignee,
                    created_by=request.created_by,
                    workspace_kind=request.workspace_kind,
                    workspace_path=request.workspace_path,
                    branch_name=request.branch_name,
                    tenant=request.tenant,
                    priority=request.priority,
                    parents=request.parents,
                    triage=request.triage,
                    idempotency_key=request.idempotency_key,
                    max_runtime_seconds=request.max_runtime_seconds,
                    skills=request.skills,
                    max_retries=request.max_retries,
                    goal_mode=request.goal_mode,
                    goal_max_turns=request.goal_max_turns,
                    initial_status=request.initial_status,
                    session_id=request.session_id,
                    project_id=request.project_id,
                    target_machine=request.target_machine,
                    required_capabilities=request.required_capabilities,
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"task_id": task_id}

    @app.post("/v1/tasks/query")
    def list_tasks(
        request: TaskListRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        try:
            with contextlib.closing(kb.connect(db_path)) as conn:
                tasks = kb.list_tasks(
                    conn,
                    assignee=request.assignee,
                    status=request.status,
                    tenant=request.tenant,
                    include_archived=request.include_archived,
                    limit=request.limit,
                )
                payload = [_task_payload(conn, task) for task in tasks]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"tasks": payload}

    @app.post("/v1/tasks/recompute-ready")
    def recompute_ready(
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        with contextlib.closing(kb.connect(db_path)) as conn:
            promoted = kb.recompute_ready(conn)
        return {"promoted": promoted}

    @app.post("/v1/tasks/link")
    def link_tasks(
        request: LinkRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        try:
            with contextlib.closing(kb.connect(db_path)) as conn:
                kb.link_tasks(conn, request.parent_id, request.child_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"linked": True}

    @app.get("/v1/tasks/{task_id}")
    def show(task_id: str, authorization: Optional[str] = Header(default=None)) -> dict:
        require_token(authorization)
        with contextlib.closing(kb.connect(db_path)) as conn:
            task = kb.get_task(conn, task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="task not found")
            return {
                "task": _task_payload(conn, task),
                "comments": [asdict(item) for item in kb.list_comments(conn, task_id)],
                "events": [asdict(item) for item in kb.list_events(conn, task_id)],
                "runs": [asdict(item) for item in kb.list_runs(conn, task_id)],
                "parents": kb.parent_ids(conn, task_id),
                "children": kb.child_ids(conn, task_id),
                "worker_context": kb.build_worker_context(conn, task_id),
            }

    @app.post("/v1/tasks/{task_id}/complete")
    def complete(
        task_id: str,
        request: CompletionRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        with contextlib.closing(kb.connect(db_path)) as conn:
            task = kb.get_task(conn, task_id)
            if task is None or not _claim_authorizes(task, request.claim_lock):
                return {"completed": False}
            try:
                completed = kb.complete_task(
                    conn,
                    task_id,
                    result=request.result,
                    summary=request.summary,
                    metadata=request.metadata,
                    created_cards=request.created_cards,
                    expected_run_id=request.expected_run_id,
                )
            except kb.HallucinatedCardsError as exc:
                return {
                    "completed": False,
                    "error": "hallucinated_cards",
                    "phantom_cards": list(exc.phantom),
                }
        return {"completed": completed}

    @app.post("/v1/tasks/{task_id}/block")
    def block(
        task_id: str,
        request: BlockRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        with contextlib.closing(kb.connect(db_path)) as conn:
            task = kb.get_task(conn, task_id)
            if task is None or not _claim_authorizes(task, request.claim_lock):
                return {"blocked": False}
            blocked = kb.block_task(
                conn,
                task_id,
                reason=request.reason,
                kind=request.kind,
                expected_run_id=request.expected_run_id,
            )
        return {"blocked": blocked}

    @app.post("/v1/tasks/{task_id}/unblock")
    def unblock(
        task_id: str,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        with contextlib.closing(kb.connect(db_path)) as conn:
            unblocked = kb.unblock_task(conn, task_id)
        return {"unblocked": unblocked}

    @app.post("/v1/tasks/{task_id}/comments")
    def comment(
        task_id: str,
        request: CommentRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        require_token(authorization)
        with contextlib.closing(kb.connect(db_path)) as conn:
            comment_id = kb.add_comment(
                conn,
                task_id,
                author=request.author,
                body=request.body,
            )
        return {"comment_id": comment_id}

    return app


def run(*, db_path: Path, host: str, port: int, token_env: str) -> int:
    """Run one coordinator process with an explicit deployment configuration."""
    import uvicorn

    token = os.environ.get(token_env, "").strip()
    if not token:
        raise RuntimeError(f"set {token_env} to a coordinator bearer token")
    stop = threading.Event()

    def maintenance_loop() -> None:
        while not stop.is_set():
            try:
                _maintenance_once(db_path)
            except Exception:
                # The HTTP service remains authoritative; transient database
                # maintenance failures retry without taking it down.
                pass
            stop.wait(15)

    heartbeat = threading.Thread(
        target=maintenance_loop,
        name="kanban-coordinator-maintenance",
        daemon=True,
    )
    heartbeat.start()
    try:
        uvicorn.run(create_app(db_path=db_path, token=token), host=host, port=port)
    finally:
        stop.set()
        heartbeat.join(timeout=2)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a Hermes Kanban coordinator")
    parser.add_argument("--db", required=True, type=Path, help="Board SQLite path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8788, type=int)
    parser.add_argument("--token-env", default="HERMES_KANBAN_COORDINATOR_TOKEN")
    args = parser.parse_args(argv)
    try:
        return run(
            db_path=args.db,
            host=args.host,
            port=args.port,
            token_env=args.token_env,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
