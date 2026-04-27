#!/usr/bin/env python3
"""
MultiAgentCodingService — orchestrate multi-agent coding flows.

Roles: orchestrator → coder → tester → reviewer
Each flow is persisted in code_agent_flows / code_agent_flow_steps.
Integrates with CodeSession, CommandRunner, GitService, ProviderRouter, CodeIntelligence.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CodingAgentRole = Literal[
    "orchestrator",
    "coder",
    "tester",
    "reviewer",
    "researcher",
]

CodingFlowStatus = Literal[
    "created",
    "planning",
    "coding",
    "running_tests",
    "reviewing",
    "waiting_approval",
    "completed",
    "failed",
    "cancelled",
]

CodingFlowStepStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "skipped",
]

# Role → preferred preset mapping (informational, stored for future LLM routing)
ROLE_PRESET_MAP: Dict[str, str] = {
    "orchestrator": "planner",
    "coder": "strong",
    "tester": "fast",
    "reviewer": "reviewer",
    "researcher": "fast",
}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentFlowDB:
    """Persistence layer for code_agent_flows and code_agent_flow_steps."""

    import sqlite3
    import threading

    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020
    _WRITE_RETRY_MAX_S = 0.150

    def __init__(self, db_path: Optional[Path] = None):
        import sqlite3
        import threading

        from hermes_state import DEFAULT_DB_PATH

        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        import sqlite3

        cursor = self._conn.cursor()
        for ddl in [
            """CREATE TABLE IF NOT EXISTS code_agent_flows (
                id TEXT PRIMARY KEY,
                code_session_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                task_id TEXT,
                title TEXT,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'created',
                current_role TEXT,
                provider TEXT,
                model TEXT,
                preset TEXT,
                plan_json TEXT DEFAULT '{}',
                review_json TEXT,
                approval_id TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS code_agent_flow_steps (
                id TEXT PRIMARY KEY,
                flow_id TEXT NOT NULL,
                role TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                input_json TEXT DEFAULT '{}',
                output_json TEXT DEFAULT '{}',
                error TEXT,
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL
            )""",
        ]:
            try:
                cursor.execute(ddl)
            except sqlite3.OperationalError:
                pass
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_code_agent_flows_code_session_id ON code_agent_flows(code_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_agent_flows_workspace_id ON code_agent_flows(workspace_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_agent_flows_status ON code_agent_flows(status)",
            "CREATE INDEX IF NOT EXISTS idx_code_agent_flows_created_at ON code_agent_flows(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_code_agent_flow_steps_flow_id ON code_agent_flow_steps(flow_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_agent_flow_steps_status ON code_agent_flow_steps(status)",
        ]:
            try:
                cursor.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def _execute_write(self, fn):
        import random
        import sqlite3
        import time

        last_err = None
        for _ in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        self._conn.rollback()
                        raise
                    return result
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" not in str(e).lower():
                    raise
                time.sleep(
                    random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
                )
        raise last_err

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Flow CRUD
    # ------------------------------------------------------------------

    def _deserialize_flow(self, row) -> Dict[str, Any]:
        d = dict(row)
        for field in ("plan_json", "review_json"):
            raw = d.pop(field, None)
            key = field.replace("_json", "")
            if raw:
                try:
                    d[key] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    d[key] = {} if field == "plan_json" else None
            else:
                d[key] = {} if field == "plan_json" else None
        return d

    def create_flow(
        self,
        flow_id: str,
        code_session_id: str,
        workspace_id: str,
        task_id: Optional[str],
        title: Optional[str],
        description: Optional[str],
        provider: Optional[str],
        model: Optional[str],
        preset: Optional[str],
        now: str,
    ) -> Dict[str, Any]:
        def _do(conn):
            conn.execute(
                """INSERT INTO code_agent_flows
                   (id, code_session_id, workspace_id, task_id, title, description,
                    status, provider, model, preset, plan_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'created', ?, ?, ?, '{}', ?, ?)""",
                (
                    flow_id, code_session_id, workspace_id, task_id,
                    title, description, provider, model, preset, now, now,
                ),
            )

        self._execute_write(_do)
        return self.get_flow(flow_id)

    def get_flow(self, flow_id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM code_agent_flows WHERE id = ?", (flow_id,)
        )
        row = cursor.fetchone()
        return self._deserialize_flow(row) if row else None

    def list_flows(
        self,
        code_session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM code_agent_flows WHERE 1=1"
        params: List[Any] = []
        if code_session_id:
            q += " AND code_session_id = ?"
            params.append(code_session_id)
        if workspace_id:
            q += " AND workspace_id = ?"
            params.append(workspace_id)
        if status:
            q += " AND status = ?"
            params.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = self._conn.execute(q, params)
        return [self._deserialize_flow(r) for r in cursor.fetchall()]

    def update_flow(self, flow_id: str, **updates) -> Dict[str, Any]:
        now = _utc_now()
        updates["updated_at"] = now

        # Serialize dict fields
        if "plan" in updates:
            updates["plan_json"] = json.dumps(updates.pop("plan"))
        if "review" in updates:
            val = updates.pop("review")
            updates["review_json"] = json.dumps(val) if val is not None else None

        allowed = {
            "status", "current_role", "provider", "model", "preset",
            "plan_json", "review_json", "approval_id", "error",
            "completed_at", "updated_at",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed}

        if not filtered:
            return self.get_flow(flow_id)

        set_clause = ", ".join(f"{k} = ?" for k in filtered)
        values = list(filtered.values()) + [flow_id]

        def _do(conn):
            conn.execute(
                f"UPDATE code_agent_flows SET {set_clause} WHERE id = ?", values
            )

        self._execute_write(_do)
        return self.get_flow(flow_id)

    # ------------------------------------------------------------------
    # Step CRUD
    # ------------------------------------------------------------------

    def _deserialize_step(self, row) -> Dict[str, Any]:
        d = dict(row)
        for field in ("input_json", "output_json"):
            raw = d.pop(field, None)
            key = field.replace("_json", "")
            try:
                d[key] = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                d[key] = {}
        return d

    def create_step(
        self,
        step_id: str,
        flow_id: str,
        role: str,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        now: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = now or _utc_now()

        def _do(conn):
            conn.execute(
                """INSERT INTO code_agent_flow_steps
                   (id, flow_id, role, name, status, input_json, output_json, created_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, '{}', ?)""",
                (
                    step_id, flow_id, role, name,
                    json.dumps(input_data or {}), now,
                ),
            )

        self._execute_write(_do)
        return self._get_step(step_id)

    def _get_step(self, step_id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM code_agent_flow_steps WHERE id = ?", (step_id,)
        )
        row = cursor.fetchone()
        return self._deserialize_step(row) if row else None

    def list_steps(self, flow_id: str) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM code_agent_flow_steps WHERE flow_id = ? ORDER BY created_at ASC",
            (flow_id,),
        )
        return [self._deserialize_step(r) for r in cursor.fetchall()]

    def update_step(self, step_id: str, **updates) -> Dict[str, Any]:
        if "output" in updates:
            updates["output_json"] = json.dumps(updates.pop("output"))
        if "input" in updates:
            updates["input_json"] = json.dumps(updates.pop("input"))

        allowed = {
            "status", "input_json", "output_json", "error",
            "started_at", "completed_at",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return self._get_step(step_id)

        set_clause = ", ".join(f"{k} = ?" for k in filtered)
        values = list(filtered.values()) + [step_id]

        def _do(conn):
            conn.execute(
                f"UPDATE code_agent_flow_steps SET {set_clause} WHERE id = ?", values
            )

        self._execute_write(_do)
        return self._get_step(step_id)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class MultiAgentCodingService:
    """Orchestrate multi-agent coding flows.

    Each flow moves through roles (orchestrator → coder → tester → reviewer)
    and integrates with the existing Hermes Code Mode services.
    """

    def __init__(self, db_path: Optional[Path] = None, realtime_hub=None):
        self._db_path = db_path
        self._realtime_hub = realtime_hub

    def _flow_db(self) -> AgentFlowDB:
        return AgentFlowDB(db_path=self._db_path)

    def _session_db(self):
        from hermes_state import CodeSessionDB
        return CodeSessionDB(db_path=self._db_path)

    def _workspace_db(self):
        from hermes_state import WorkspaceDB
        return WorkspaceDB(db_path=self._db_path)

    def _approval_db(self):
        from hermes_state import ApprovalDB
        return ApprovalDB(db_path=self._db_path)

    def _command_runner(self):
        from hermes_cli.code.command_runner import CommandRunnerService
        return CommandRunnerService(db_path=self._db_path, realtime_hub=self._realtime_hub)

    def _git_service(self):
        from hermes_cli.code.git_service import GitService
        return GitService(db_path=self._db_path, realtime_hub=self._realtime_hub)

    def _lsp_service(self):
        from hermes_cli.code.lsp_service import CodeIntelligenceService
        return CodeIntelligenceService(db_path=self._db_path, realtime_hub=self._realtime_hub)

    async def _broadcast(self, event_type: str, payload: dict):
        if self._realtime_hub:
            try:
                await self._realtime_hub.broadcast(event_type, {"payload": payload})
            except Exception:
                pass

    def _add_session_event(
        self,
        code_session_id: str,
        event_type: str,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ):
        try:
            db = self._session_db()
            db.add_event(code_session_id, event_type, message=message, payload=payload or {})
        except Exception as exc:
            logger.warning("Failed to add session event %s: %s", event_type, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_flow(
        self,
        code_session_id: str,
        workspace_id: str,
        task_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        preset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new agent flow linked to a code session."""
        # Validate code session exists
        session_db = self._session_db()
        session = session_db.get_session(code_session_id)
        if not session:
            raise ValueError(f"CodeSession not found: {code_session_id}")

        # Validate workspace exists
        workspace_db = self._workspace_db()
        workspace = workspace_db.get_workspace(workspace_id)
        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        flow_id = str(uuid.uuid4())
        now = _utc_now()

        db = self._flow_db()
        flow = db.create_flow(
            flow_id=flow_id,
            code_session_id=code_session_id,
            workspace_id=workspace_id,
            task_id=task_id,
            title=title or "Untitled flow",
            description=description,
            provider=provider or session.get("provider"),
            model=model or session.get("model"),
            preset=preset or "planner",
            now=now,
        )

        self._add_session_event(
            code_session_id,
            "agent.started",
            message=f"Agent flow created: {flow_id}",
            payload={"flow_id": flow_id},
        )

        return flow

    def get_flow(self, flow_id: str) -> Optional[Dict[str, Any]]:
        db = self._flow_db()
        flow = db.get_flow(flow_id)
        if flow:
            flow["steps"] = db.list_steps(flow_id)
        return flow

    def list_flows(
        self,
        code_session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        db = self._flow_db()
        return db.list_flows(
            code_session_id=code_session_id,
            workspace_id=workspace_id,
            status=status,
            limit=limit,
        )

    def cancel_flow(self, flow_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        db = self._flow_db()
        flow = db.get_flow(flow_id)
        if not flow:
            raise ValueError(f"Flow not found: {flow_id}")
        if flow["status"] in ("completed", "failed", "cancelled"):
            raise ValueError(f"Flow already in terminal state: {flow['status']}")

        flow = db.update_flow(
            flow_id,
            status="cancelled",
            error=reason or "Cancelled by user",
            completed_at=_utc_now(),
        )
        self._add_session_event(
            flow["code_session_id"],
            "agent.status_changed",
            message="Flow cancelled",
            payload={"flow_id": flow_id, "status": "cancelled"},
        )
        return flow

    def run_flow(self, flow_id: str) -> Dict[str, Any]:
        """Execute the full multi-agent coding flow synchronously."""
        db = self._flow_db()
        flow = db.get_flow(flow_id)
        if not flow:
            raise ValueError(f"Flow not found: {flow_id}")
        if flow["status"] not in ("created", "planning"):
            raise ValueError(f"Flow cannot be run from status: {flow['status']}")

        try:
            flow = self._run_orchestrator(flow, db)
            if flow["status"] == "failed":
                return self.get_flow(flow_id)

            flow = self._run_coder(flow, db)
            if flow["status"] == "failed":
                return self.get_flow(flow_id)

            flow = self._run_tester(flow, db)
            if flow["status"] in ("failed", "waiting_approval"):
                return self.get_flow(flow_id)

            flow = self._run_reviewer(flow, db)
            return self.get_flow(flow_id)

        except Exception as exc:
            logger.error("Flow %s failed unexpectedly: %s", flow_id, exc, exc_info=True)
            db.update_flow(
                flow_id,
                status="failed",
                error=str(exc),
                completed_at=_utc_now(),
            )
            self._add_session_event(
                flow["code_session_id"],
                "agent.failed",
                message=str(exc),
                payload={"flow_id": flow_id},
            )
            return self.get_flow(flow_id)

    def resume_flow(self, flow_id: str) -> Dict[str, Any]:
        """Resume a flow that is waiting for approval."""
        db = self._flow_db()
        flow = db.get_flow(flow_id)
        if not flow:
            raise ValueError(f"Flow not found: {flow_id}")
        if flow["status"] != "waiting_approval":
            raise ValueError(f"Flow is not waiting_approval: {flow['status']}")

        # Resume from reviewing since approval was granted
        flow = db.update_flow(flow_id, status="reviewing", current_role="reviewer")
        self._add_session_event(
            flow["code_session_id"],
            "agent.status_changed",
            message="Flow resumed after approval",
            payload={"flow_id": flow_id, "status": "reviewing"},
        )

        try:
            flow = self._run_reviewer(flow, db)
        except Exception as exc:
            logger.error("Resume flow %s failed: %s", flow_id, exc, exc_info=True)
            db.update_flow(
                flow_id,
                status="failed",
                error=str(exc),
                completed_at=_utc_now(),
            )

        return self.get_flow(flow_id)

    # ------------------------------------------------------------------
    # Agent role implementations
    # ------------------------------------------------------------------

    def _run_orchestrator(self, flow: Dict[str, Any], db: AgentFlowDB) -> Dict[str, Any]:
        flow_id = flow["id"]
        code_session_id = flow["code_session_id"]
        workspace_id = flow["workspace_id"]

        flow = db.update_flow(flow_id, status="planning", current_role="orchestrator")
        self._add_session_event(
            code_session_id,
            "agent.status_changed",
            message="Orchestrator planning",
            payload={"flow_id": flow_id, "role": "orchestrator", "status": "planning"},
        )

        step_id = str(uuid.uuid4())
        now = _utc_now()
        db.create_step(
            step_id=step_id,
            flow_id=flow_id,
            role="orchestrator",
            name="Plan coding task",
            input_data={
                "description": flow.get("description"),
                "workspace_id": workspace_id,
                "provider": flow.get("provider"),
                "model": flow.get("model"),
            },
            now=now,
        )
        db.update_step(step_id, status="running", started_at=_utc_now())

        try:
            plan = self._build_plan(flow, workspace_id)
            db.update_step(
                step_id,
                status="completed",
                output=plan,
                completed_at=_utc_now(),
            )
            flow = db.update_flow(flow_id, plan=plan)
            return flow
        except Exception as exc:
            db.update_step(step_id, status="failed", error=str(exc), completed_at=_utc_now())
            flow = db.update_flow(
                flow_id, status="failed", error=f"Orchestrator failed: {exc}", completed_at=_utc_now()
            )
            return flow

    def _build_plan(self, flow: Dict[str, Any], workspace_id: str) -> Dict[str, Any]:
        """Build a deterministic coding plan from workspace context."""
        try:
            workspace_db = self._workspace_db()
            workspace = workspace_db.get_workspace(workspace_id)
        except Exception:
            workspace = {}

        stack = []
        if workspace:
            try:
                stack = json.loads(workspace.get("detected_stack_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                stack = []

        # Determine test commands from stack
        test_commands: List[str] = []
        if any("typescript" in s.lower() or "javascript" in s.lower() or "node" in s.lower() for s in stack):
            pm = workspace.get("package_manager") or "npm"
            test_commands.extend([f"{pm} run typecheck", f"{pm} run lint", f"{pm} run build"])
        if any("go" in s.lower() for s in stack):
            test_commands.extend(["go vet ./...", "go test ./..."])
        if any("python" in s.lower() for s in stack):
            test_commands.extend(["python -m pytest"])
        if not test_commands:
            test_commands = ["make test"]

        description = flow.get("description") or flow.get("title") or "Implement coding task"
        risks: List[str] = []
        if "migration" in description.lower() or "database" in description.lower():
            risks.append("Database migration may require backup")
        if "auth" in description.lower() or "security" in description.lower():
            risks.append("Security-sensitive change — human review required")

        steps = [
            {"role": "coder", "name": "Collect workspace context"},
            {"role": "coder", "name": "Register coding intent"},
            {"role": "tester", "name": "Run validation commands"},
            {"role": "reviewer", "name": "Review diff and diagnostics"},
        ]

        return {
            "summary": description,
            "steps": steps,
            "test_commands": test_commands,
            "risks": risks,
            "requires_approval": bool(risks),
            "stack": stack,
        }

    def _run_coder(self, flow: Dict[str, Any], db: AgentFlowDB) -> Dict[str, Any]:
        flow_id = flow["id"]
        code_session_id = flow["code_session_id"]
        workspace_id = flow["workspace_id"]

        flow = db.update_flow(flow_id, status="coding", current_role="coder")
        self._add_session_event(
            code_session_id,
            "agent.status_changed",
            message="Coder collecting context",
            payload={"flow_id": flow_id, "role": "coder", "status": "coding"},
        )

        # Step: collect workspace context
        step_id = str(uuid.uuid4())
        db.create_step(
            step_id=step_id,
            flow_id=flow_id,
            role="coder",
            name="Collect workspace context",
            now=_utc_now(),
        )
        db.update_step(step_id, status="running", started_at=_utc_now())

        context: Dict[str, Any] = {}
        try:
            git_svc = self._git_service()
            git_status = git_svc.get_status(workspace_id)
            context["git_status"] = git_status
            context["branch"] = git_status.get("branch")
            context["modified_files"] = [
                f["path"] for f in git_status.get("files", []) if f.get("staged") or f.get("modified")
            ]
        except Exception as exc:
            logger.warning("Coder: git status failed: %s", exc)
            context["git_error"] = str(exc)

        try:
            lsp_svc = self._lsp_service()
            diag_result = lsp_svc.run_diagnostics(workspace_id, code_session_id=code_session_id)
            context["initial_diagnostics"] = {
                "id": diag_result.get("id"),
                "status": diag_result.get("status"),
                "summary": diag_result.get("summary", {}),
            }
        except Exception as exc:
            logger.warning("Coder: initial diagnostics failed: %s", exc)
            context["diagnostics_error"] = str(exc)

        db.update_step(step_id, status="completed", output=context, completed_at=_utc_now())

        # Step: register coding intent
        step2_id = str(uuid.uuid4())
        db.create_step(
            step_id=step2_id,
            flow_id=flow_id,
            role="coder",
            name="Register coding intent",
            input_data={"description": flow.get("description")},
            now=_utc_now(),
        )
        db.update_step(step2_id, status="running", started_at=_utc_now())
        db.update_step(
            step2_id,
            status="completed",
            output={"intent": flow.get("description"), "workspace_context": context},
            completed_at=_utc_now(),
        )

        return db.get_flow(flow_id)

    def _run_tester(self, flow: Dict[str, Any], db: AgentFlowDB) -> Dict[str, Any]:
        flow_id = flow["id"]
        code_session_id = flow["code_session_id"]
        workspace_id = flow["workspace_id"]

        flow = db.update_flow(flow_id, status="running_tests", current_role="tester")
        self._add_session_event(
            code_session_id,
            "agent.status_changed",
            message="Tester running validation",
            payload={"flow_id": flow_id, "role": "tester", "status": "running_tests"},
        )

        plan = flow.get("plan") or {}
        test_commands = plan.get("test_commands") or []

        runner = self._command_runner()

        for cmd_str in test_commands:
            safety = runner.classify_command(cmd_str)
            step_id = str(uuid.uuid4())
            db.create_step(
                step_id=step_id,
                flow_id=flow_id,
                role="tester",
                name=f"Run: {cmd_str}",
                input_data={"command": cmd_str, "safety": safety},
                now=_utc_now(),
            )
            db.update_step(step_id, status="running", started_at=_utc_now())

            if safety == "blocked":
                err = f"Command blocked by security policy: {cmd_str}"
                db.update_step(step_id, status="failed", error=err, completed_at=_utc_now())
                logger.warning("Tester: blocked command skipped: %s", cmd_str)
                continue

            if safety == "needs_approval":
                # Create approval and pause flow
                approval_id = self._create_approval_for_command(
                    flow=flow, cmd_str=cmd_str, step_id=step_id
                )
                db.update_step(
                    step_id,
                    status="failed",
                    error=f"Needs approval (approval_id={approval_id})",
                    completed_at=_utc_now(),
                )
                flow = db.update_flow(
                    flow_id,
                    status="waiting_approval",
                    approval_id=approval_id,
                )
                self._add_session_event(
                    code_session_id,
                    "agent.waiting_approval",
                    message=f"Command needs approval: {cmd_str}",
                    payload={
                        "flow_id": flow_id,
                        "approval_id": approval_id,
                        "command": cmd_str,
                    },
                )
                return flow

            # Safe command — run it
            try:
                cmd_record = runner.create_command(
                    code_session_id=code_session_id,
                    workspace_id=workspace_id,
                    command=cmd_str,
                )
                result = runner.run_command_sync(cmd_record["id"])
                exit_code = result.get("exit_code", -1)
                output = {"exit_code": exit_code, "stdout": result.get("stdout", ""), "command_id": cmd_record["id"]}
                status = "completed" if exit_code == 0 else "failed"
                error = None if exit_code == 0 else f"Command exited with code {exit_code}"
                db.update_step(
                    step_id,
                    status=status,
                    output=output,
                    error=error,
                    completed_at=_utc_now(),
                )
            except Exception as exc:
                db.update_step(step_id, status="failed", error=str(exc), completed_at=_utc_now())
                logger.warning("Tester: command %r failed: %s", cmd_str, exc)

        return db.get_flow(flow_id)

    def _create_approval_for_command(
        self, flow: Dict[str, Any], cmd_str: str, step_id: str
    ) -> str:
        approval_id = str(uuid.uuid4())
        now = _utc_now()
        try:
            approval_db = self._approval_db()
            approval_db.create_approval(
                approval_id=approval_id,
                session_id=flow.get("code_session_id"),
                agent_id=flow["id"],
                title=f"Agent flow requests command: {cmd_str}",
                command=cmd_str,
                created_at=now,
                kind="command",
                details=json.dumps({
                    "flow_id": flow["id"],
                    "step_id": step_id,
                    "workspace_id": flow.get("workspace_id"),
                }),
            )
        except Exception as exc:
            logger.error("Failed to create approval for command %r: %s", cmd_str, exc)
        return approval_id

    def _run_reviewer(self, flow: Dict[str, Any], db: AgentFlowDB) -> Dict[str, Any]:
        flow_id = flow["id"]
        code_session_id = flow["code_session_id"]
        workspace_id = flow["workspace_id"]

        flow = db.update_flow(flow_id, status="reviewing", current_role="reviewer")
        self._add_session_event(
            code_session_id,
            "agent.status_changed",
            message="Reviewer analysing diff and diagnostics",
            payload={"flow_id": flow_id, "role": "reviewer", "status": "reviewing"},
        )

        step_id = str(uuid.uuid4())
        db.create_step(
            step_id=step_id,
            flow_id=flow_id,
            role="reviewer",
            name="Review diff and diagnostics",
            now=_utc_now(),
        )
        db.update_step(step_id, status="running", started_at=_utc_now())

        review: Dict[str, Any] = {
            "decision": "approve",
            "summary": "",
            "risks": [],
            "files_changed": [],
            "commands_run": [],
            "diagnostics_summary": {},
            "requires_human_approval": False,
        }

        # Collect git diff
        try:
            git_svc = self._git_service()
            git_status = git_svc.get_status(workspace_id)
            diff_result = git_svc.get_diff(workspace_id)
            files_changed = [f["path"] for f in git_status.get("files", [])]
            review["files_changed"] = files_changed
            review["diff_stat"] = diff_result.get("stat") or ""
        except Exception as exc:
            logger.warning("Reviewer: git info failed: %s", exc)

        # Collect final diagnostics
        try:
            lsp_svc = self._lsp_service()
            diag_result = lsp_svc.run_diagnostics(workspace_id, code_session_id=code_session_id)
            summary = diag_result.get("summary", {})
            review["diagnostics_summary"] = summary
            if summary.get("errors", 0) > 0:
                review["risks"].append(f"{summary['errors']} diagnostic error(s) detected")
                review["decision"] = "request_changes"
        except Exception as exc:
            logger.warning("Reviewer: diagnostics failed: %s", exc)

        # Collect commands run from steps
        steps = db.list_steps(flow_id)
        commands_run = [
            {
                "name": s["name"],
                "role": s["role"],
                "status": s["status"],
                "output": s.get("output", {}),
            }
            for s in steps
            if s["role"] == "tester"
        ]
        review["commands_run"] = commands_run

        # Check plan risks
        plan_risks = (flow.get("plan") or {}).get("risks", [])
        review["risks"].extend(plan_risks)

        # Determine if human approval needed
        has_errors = review.get("decision") == "request_changes"
        has_risks = bool(review["risks"])
        has_changes = bool(review["files_changed"])
        review["requires_human_approval"] = has_errors or has_risks or has_changes
        review["summary"] = (
            f"Reviewed {len(review['files_changed'])} changed file(s). "
            f"Decision: {review['decision']}. "
            f"Risks: {len(review['risks'])}."
        )

        db.update_step(step_id, status="completed", output=review, completed_at=_utc_now())

        # If human approval needed, create approval and pause
        if review["requires_human_approval"]:
            approval_id = str(uuid.uuid4())
            now = _utc_now()
            try:
                approval_db = self._approval_db()
                approval_db.create_approval(
                    approval_id=approval_id,
                    session_id=code_session_id,
                    agent_id=flow_id,
                    title="Code review requires human approval",
                    command=None,
                    created_at=now,
                    kind="code_review",
                    details=json.dumps(review),
                )
            except Exception as exc:
                logger.error("Reviewer: failed to create approval: %s", exc)
                approval_id = None

            flow = db.update_flow(
                flow_id,
                status="waiting_approval",
                review=review,
                approval_id=approval_id,
            )
            self._add_session_event(
                code_session_id,
                "agent.waiting_approval",
                message="Code review awaiting human approval",
                payload={"flow_id": flow_id, "approval_id": approval_id, "review": review},
            )
        else:
            flow = db.update_flow(
                flow_id,
                status="completed",
                review=review,
                completed_at=_utc_now(),
            )
            self._add_session_event(
                code_session_id,
                "agent.completed",
                message="Flow completed successfully",
                payload={"flow_id": flow_id, "review": review},
            )

        return flow
