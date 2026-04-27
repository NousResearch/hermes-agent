#!/usr/bin/env python3
"""
CodingSkillsService — structured, safe, reusable coding routines.

Skills wrap common dev tasks as traceable, approval-aware flows.
Each skill integrates with CodeSession, CommandRunner, GitService,
CodeIntelligence, MultiAgentCodingService, and Approvals.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

CodingSkillName = Literal[
    "fix_build",
    "fix_runtime_error",
    "refactor_react_page",
    "implement_feature",
    "review_diff",
    "stabilize_hanging_task",
    "benchmark_provider",
]

CodingSkillRunStatus = Literal[
    "created",
    "running",
    "waiting_approval",
    "completed",
    "failed",
    "cancelled",
]

SKILL_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "fix_build",
        "title": "Fix Build",
        "description": "Run diagnostics, detect build/typecheck/lint errors, prepare correction plan.",
        "safe_only": True,
        "requires_workspace": True,
    },
    {
        "name": "review_diff",
        "title": "Review Diff",
        "description": "Review current git diff, detect risks, generate review summary.",
        "safe_only": True,
        "requires_workspace": True,
    },
    {
        "name": "stabilize_hanging_task",
        "title": "Stabilize Hanging Task",
        "description": "Detect and stabilize stuck sessions, flows, or commands.",
        "safe_only": True,
        "requires_workspace": False,
    },
    {
        "name": "fix_runtime_error",
        "title": "Fix Runtime Error",
        "description": "Parse error/stack trace, localise files, prepare correction plan.",
        "safe_only": True,
        "requires_workspace": True,
    },
    {
        "name": "implement_feature",
        "title": "Implement Feature",
        "description": "Transform a feature description into a MultiAgentCodingFlow.",
        "safe_only": False,
        "requires_workspace": True,
    },
    {
        "name": "refactor_react_page",
        "title": "Refactor React Page",
        "description": "Prepare safe refactoring plan for a React page preserving design system.",
        "safe_only": True,
        "requires_workspace": True,
    },
    {
        "name": "benchmark_provider",
        "title": "Benchmark Provider",
        "description": "Compare providers/models on a task without modifying files (dry_run).",
        "safe_only": True,
        "requires_workspace": False,
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# SkillRunDB — persistence
# ---------------------------------------------------------------------------


class SkillRunDB:
    """Persistence layer for code_skill_runs."""

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
        try:
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS code_skill_runs (
                    id TEXT PRIMARY KEY,
                    skill_name TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    code_session_id TEXT,
                    task_id TEXT,
                    agent_flow_id TEXT,
                    status TEXT NOT NULL DEFAULT 'created',
                    input_json TEXT DEFAULT '{}',
                    output_json TEXT DEFAULT '{}',
                    summary TEXT,
                    diagnostics_before_json TEXT,
                    diagnostics_after_json TEXT,
                    commands_json TEXT DEFAULT '[]',
                    artifacts_json TEXT DEFAULT '[]',
                    approval_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )"""
            )
        except sqlite3.OperationalError:
            pass
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_code_skill_runs_workspace_id ON code_skill_runs(workspace_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_skill_runs_code_session_id ON code_skill_runs(code_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_skill_runs_skill_name ON code_skill_runs(skill_name)",
            "CREATE INDEX IF NOT EXISTS idx_code_skill_runs_status ON code_skill_runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_code_skill_runs_created_at ON code_skill_runs(created_at DESC)",
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

    def _deserialize(self, row) -> Dict[str, Any]:
        d = dict(row)
        for field in ("input_json", "output_json", "commands_json", "artifacts_json"):
            raw = d.pop(field, None)
            key = field.replace("_json", "")
            try:
                d[key] = json.loads(raw) if raw else ({} if "json" in field and field in ("input_json", "output_json") else [])
            except (json.JSONDecodeError, TypeError):
                d[key] = {} if field in ("input_json", "output_json") else []
        for field in ("diagnostics_before_json", "diagnostics_after_json"):
            raw = d.pop(field, None)
            key = field.replace("_json", "")
            try:
                d[key] = json.loads(raw) if raw else None
            except (json.JSONDecodeError, TypeError):
                d[key] = None
        return d

    def create_run(
        self,
        run_id: str,
        skill_name: str,
        workspace_id: str,
        code_session_id: Optional[str],
        task_id: Optional[str],
        input_data: Dict[str, Any],
        now: str,
    ) -> Dict[str, Any]:
        def _do(conn):
            conn.execute(
                """INSERT INTO code_skill_runs
                   (id, skill_name, workspace_id, code_session_id, task_id,
                    status, input_json, output_json, commands_json, artifacts_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'created', ?, '{}', '[]', '[]', ?, ?)""",
                (
                    run_id, skill_name, workspace_id, code_session_id, task_id,
                    json.dumps(input_data), now, now,
                ),
            )

        self._execute_write(_do)
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM code_skill_runs WHERE id = ?", (run_id,)
        )
        row = cursor.fetchone()
        return self._deserialize(row) if row else None

    def list_runs(
        self,
        workspace_id: Optional[str] = None,
        code_session_id: Optional[str] = None,
        skill_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM code_skill_runs WHERE 1=1"
        params: List[Any] = []
        if workspace_id:
            q += " AND workspace_id = ?"
            params.append(workspace_id)
        if code_session_id:
            q += " AND code_session_id = ?"
            params.append(code_session_id)
        if skill_name:
            q += " AND skill_name = ?"
            params.append(skill_name)
        if status:
            q += " AND status = ?"
            params.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = self._conn.execute(q, params)
        return [self._deserialize(r) for r in cursor.fetchall()]

    def update_run(self, run_id: str, **updates) -> Dict[str, Any]:
        now = _utc_now()
        updates["updated_at"] = now

        # Serialize dict/list fields
        for src, dst in [("input", "input_json"), ("output", "output_json"),
                         ("commands", "commands_json"), ("artifacts", "artifacts_json"),
                         ("diagnostics_before", "diagnostics_before_json"),
                         ("diagnostics_after", "diagnostics_after_json")]:
            if src in updates:
                val = updates.pop(src)
                updates[dst] = json.dumps(val) if val is not None else None

        allowed = {
            "status", "summary", "agent_flow_id", "approval_id", "error",
            "completed_at", "updated_at",
            "input_json", "output_json", "commands_json", "artifacts_json",
            "diagnostics_before_json", "diagnostics_after_json",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return self.get_run(run_id)

        set_clause = ", ".join(f"{k} = ?" for k in filtered)
        values = list(filtered.values()) + [run_id]

        def _do(conn):
            conn.execute(
                f"UPDATE code_skill_runs SET {set_clause} WHERE id = ?", values
            )

        self._execute_write(_do)
        return self.get_run(run_id)


# ---------------------------------------------------------------------------
# CodingSkillsService
# ---------------------------------------------------------------------------


class CodingSkillsService:
    """Orchestrate coding skill runs.

    Each skill wraps a common dev task (fix_build, review_diff, etc.)
    and integrates with CommandRunner, GitService, CodeIntelligence,
    MultiAgentCodingService, and Approvals.
    """

    def __init__(self, db_path: Optional[Path] = None, realtime_hub=None):
        self._db_path = db_path
        self._realtime_hub = realtime_hub

    # ------ internal helpers ------

    def _run_db(self) -> SkillRunDB:
        return SkillRunDB(db_path=self._db_path)

    def _workspace_db(self):
        from hermes_state import WorkspaceDB
        return WorkspaceDB(db_path=self._db_path)

    def _session_db(self):
        from hermes_state import CodeSessionDB
        return CodeSessionDB(db_path=self._db_path)

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

    def _agent_service(self):
        from hermes_cli.code.multi_agent_coding import MultiAgentCodingService
        return MultiAgentCodingService(db_path=self._db_path, realtime_hub=self._realtime_hub)

    async def _broadcast(self, event_type: str, payload: dict):
        if self._realtime_hub:
            try:
                await self._realtime_hub.broadcast(event_type, {"payload": payload})
            except Exception:
                pass

    def _add_session_event(
        self,
        code_session_id: Optional[str],
        event_type: str,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ):
        if not code_session_id:
            return
        try:
            db = self._session_db()
            db.add_event(code_session_id, event_type, message=message, payload=payload or {})
        except Exception as exc:
            logger.warning("Failed to add session event %s: %s", event_type, exc)

    def _create_approval(
        self,
        run: Dict[str, Any],
        title: str,
        kind: str = "code_review",
        command: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        approval_id = str(uuid.uuid4())
        now = _utc_now()
        try:
            approval_db = self._approval_db()
            approval_db.create_approval(
                approval_id=approval_id,
                session_id=run.get("code_session_id"),
                agent_id=run["id"],
                title=title,
                command=command,
                created_at=now,
                kind=kind,
                details=json.dumps(details or {}),
            )
        except Exception as exc:
            logger.error("Failed to create approval: %s", exc)
        return approval_id

    # ------ Public API ------

    def list_skills(self) -> List[Dict[str, Any]]:
        return list(SKILL_CATALOG)

    def create_run(
        self,
        skill_name: str,
        workspace_id: str,
        code_session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        skill_names = [s["name"] for s in SKILL_CATALOG]
        if skill_name not in skill_names:
            raise ValueError(f"Unknown skill: {skill_name}. Available: {skill_names}")

        # Validate workspace
        wdb = self._workspace_db()
        workspace = wdb.get_workspace(workspace_id)
        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        # Validate session if provided
        if code_session_id:
            sdb = self._session_db()
            session = sdb.get_session(code_session_id)
            if not session:
                raise ValueError(f"CodeSession not found: {code_session_id}")

        run_id = str(uuid.uuid4())
        now = _utc_now()

        db = self._run_db()
        run = db.create_run(
            run_id=run_id,
            skill_name=skill_name,
            workspace_id=workspace_id,
            code_session_id=code_session_id,
            task_id=task_id,
            input_data=input_data or {},
            now=now,
        )
        db.close()

        self._add_session_event(
            code_session_id,
            "skill.started",
            message=f"Skill '{skill_name}' run created",
            payload={"skill_run_id": run_id, "skill_name": skill_name},
        )

        return run

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        db = self._run_db()
        try:
            return db.get_run(run_id)
        finally:
            db.close()

    def list_runs(
        self,
        workspace_id: Optional[str] = None,
        code_session_id: Optional[str] = None,
        skill_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        db = self._run_db()
        try:
            return db.list_runs(
                workspace_id=workspace_id,
                code_session_id=code_session_id,
                skill_name=skill_name,
                status=status,
                limit=limit,
            )
        finally:
            db.close()

    def cancel_run(self, run_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        db = self._run_db()
        run = db.get_run(run_id)
        if not run:
            raise ValueError(f"Skill run not found: {run_id}")
        if run["status"] in ("completed", "failed", "cancelled"):
            raise ValueError(f"Skill run already in terminal state: {run['status']}")
        run = db.update_run(
            run_id,
            status="cancelled",
            error=reason or "Cancelled by user",
            completed_at=_utc_now(),
        )
        db.close()
        self._add_session_event(
            run.get("code_session_id"),
            "skill.cancelled",
            message=f"Skill '{run['skill_name']}' cancelled",
            payload={"skill_run_id": run_id},
        )
        return run

    def resume_run(self, run_id: str) -> Dict[str, Any]:
        db = self._run_db()
        run = db.get_run(run_id)
        if not run:
            raise ValueError(f"Skill run not found: {run_id}")
        if run["status"] != "waiting_approval":
            raise ValueError(f"Skill run is not waiting_approval: {run['status']}")
        run = db.update_run(run_id, status="running")
        db.close()

        # Re-execute from the skill's resume point
        return self.run_skill(run_id)

    def run_skill(self, run_id: str) -> Dict[str, Any]:
        db = self._run_db()
        run = db.get_run(run_id)
        db.close()

        if not run:
            raise ValueError(f"Skill run not found: {run_id}")
        if run["status"] not in ("created", "running"):
            raise ValueError(f"Skill run cannot be executed from status: {run['status']}")

        dispatch = {
            "fix_build": self._run_fix_build,
            "review_diff": self._run_review_diff,
            "stabilize_hanging_task": self._run_stabilize_hanging_task,
            "fix_runtime_error": self._run_fix_runtime_error,
            "implement_feature": self._run_implement_feature,
            "refactor_react_page": self._run_refactor_react_page,
            "benchmark_provider": self._run_benchmark_provider,
        }

        skill_fn = dispatch.get(run["skill_name"])
        if not skill_fn:
            raise ValueError(f"No executor for skill: {run['skill_name']}")

        db = self._run_db()
        run = db.update_run(run_id, status="running")
        db.close()

        self._add_session_event(
            run.get("code_session_id"),
            "skill.updated",
            message=f"Skill '{run['skill_name']}' running",
            payload={"skill_run_id": run_id, "status": "running"},
        )

        try:
            result = skill_fn(run)
            return result
        except Exception as exc:
            logger.error("Skill run %s failed: %s", run_id, exc, exc_info=True)
            db = self._run_db()
            result = db.update_run(
                run_id,
                status="failed",
                error=str(exc),
                completed_at=_utc_now(),
            )
            db.close()
            self._add_session_event(
                run.get("code_session_id"),
                "skill.failed",
                message=str(exc),
                payload={"skill_run_id": run_id},
            )
            return result

    # =========================================================================
    # Skill implementations
    # =========================================================================

    def _run_fix_build(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Run build/typecheck/lint, collect errors, produce correction plan."""
        run_id = run["id"]
        workspace_id = run["workspace_id"]
        code_session_id = run.get("code_session_id")
        input_data = run.get("input") or {}

        db = self._run_db()

        # 1. Diagnostics before
        diag_before = None
        try:
            lsp = self._lsp_service()
            diag_before = lsp.run_diagnostics(workspace_id, code_session_id=code_session_id)
        except Exception as exc:
            logger.warning("fix_build: diagnostics_before failed: %s", exc)

        db.update_run(run_id, diagnostics_before=diag_before)

        # 2. Determine commands to run
        custom_commands = input_data.get("commands") or []
        if not custom_commands:
            custom_commands = self._detect_build_commands(workspace_id)

        runner = self._command_runner()
        commands_run: List[Dict[str, Any]] = []
        errors_found: List[str] = []
        needs_approval_cmd: Optional[str] = None

        for cmd_str in custom_commands:
            safety = runner.classify_command(cmd_str)

            if safety == "blocked":
                commands_run.append({"command": cmd_str, "safety": "blocked", "skipped": True})
                logger.warning("fix_build: blocked command skipped: %s", cmd_str)
                continue

            if safety == "needs_approval":
                needs_approval_cmd = cmd_str
                approval_id = self._create_approval(
                    run,
                    title=f"fix_build requests: {cmd_str}",
                    kind="command",
                    command=cmd_str,
                    details={"run_id": run_id, "command": cmd_str},
                )
                run = db.update_run(
                    run_id,
                    status="waiting_approval",
                    approval_id=approval_id,
                    commands=commands_run,
                )
                self._add_session_event(
                    code_session_id,
                    "skill.waiting_approval",
                    message=f"fix_build waiting approval for: {cmd_str}",
                    payload={"skill_run_id": run_id, "approval_id": approval_id},
                )
                db.close()
                return run

            # Safe — run it
            try:
                if code_session_id:
                    cmd_record = runner.create_command(
                        code_session_id=code_session_id,
                        command=cmd_str,
                    )
                    result = runner.run_command_sync(cmd_record["id"])
                    exit_code = result.get("exit_code", -1)
                    stdout = result.get("stdout", "")
                    stderr = result.get("stderr", "")
                else:
                    exit_code, stdout, stderr = self._run_cmd_direct(cmd_str, workspace_id)

                entry = {
                    "command": cmd_str,
                    "safety": "safe",
                    "exit_code": exit_code,
                    "stdout": stdout[:2000],
                    "stderr": stderr[:2000] if stderr else "",
                }
                commands_run.append(entry)

                if exit_code != 0:
                    errors_found.append(f"{cmd_str}: exit {exit_code}")

            except Exception as exc:
                commands_run.append({"command": cmd_str, "safety": "safe", "error": str(exc)})
                errors_found.append(f"{cmd_str}: {exc}")

        db.update_run(run_id, commands=commands_run)

        # 3. Create agent-flow if errors found and auto_fix requested
        agent_flow_id = None
        if errors_found and input_data.get("auto_fix", False) and code_session_id:
            try:
                agent_svc = self._agent_service()
                flow = agent_svc.create_flow(
                    code_session_id=code_session_id,
                    workspace_id=workspace_id,
                    description=f"fix_build errors: {'; '.join(errors_found[:3])}",
                    title="Auto fix build errors",
                )
                agent_flow_id = flow["id"]
            except Exception as exc:
                logger.warning("fix_build: could not create agent flow: %s", exc)

        # 4. Diagnostics after (if something ran)
        diag_after = None
        if commands_run and code_session_id:
            try:
                lsp = self._lsp_service()
                diag_after = lsp.run_diagnostics(workspace_id, code_session_id=code_session_id)
            except Exception as exc:
                logger.warning("fix_build: diagnostics_after failed: %s", exc)

        # 5. Build summary
        if errors_found:
            summary = f"Build failed: {len(errors_found)} error(s). " + "; ".join(errors_found[:3])
        else:
            summary = f"Build passed. {len(commands_run)} command(s) ran successfully."

        output = {
            "errors_found": errors_found,
            "commands_count": len(commands_run),
            "needs_approval_cmd": needs_approval_cmd,
            "agent_flow_id": agent_flow_id,
        }

        run = db.update_run(
            run_id,
            status="completed",
            summary=summary,
            output=output,
            diagnostics_after=diag_after,
            agent_flow_id=agent_flow_id,
            completed_at=_utc_now(),
        )
        db.close()

        self._add_session_event(
            code_session_id,
            "skill.completed",
            message=summary,
            payload={"skill_run_id": run_id, "errors": len(errors_found)},
        )

        return run

    def _run_review_diff(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Review current git diff, detect risks, generate review summary."""
        run_id = run["id"]
        workspace_id = run["workspace_id"]
        code_session_id = run.get("code_session_id")

        db = self._run_db()

        # 1. Git status + diff
        git_status: Dict[str, Any] = {}
        git_diff: Dict[str, Any] = {}
        try:
            git_svc = self._git_service()
            git_status = git_svc.get_status(workspace_id)
            git_diff = git_svc.get_diff(workspace_id)
        except Exception as exc:
            logger.warning("review_diff: git failed: %s", exc)

        # 2. Current diagnostics
        diag_result = None
        try:
            lsp = self._lsp_service()
            diag_result = lsp.run_diagnostics(workspace_id, code_session_id=code_session_id)
        except Exception as exc:
            logger.warning("review_diff: diagnostics failed: %s", exc)

        # 3. Risk detection
        risks: List[str] = []
        files_changed = [f["path"] for f in git_status.get("files", [])]

        diff_content = git_diff.get("diff") or git_diff.get("content") or ""
        diff_str = str(diff_content)

        if any(s in diff_str.lower() for s in [".env", "secret", "password", "api_key", "token ="]):
            risks.append("Possible secret or credential in diff")
        if any(path in files_changed for path in ["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "go.mod", "go.sum", "pyproject.toml", "requirements.txt"]):
            risks.append("Dependency file changed — review carefully")
        if any(path.endswith((".sql", "migration")) for path in files_changed):
            risks.append("Database migration detected")
        if len(files_changed) > 20:
            risks.append(f"Large diff: {len(files_changed)} files changed")

        diag_summary = {}
        if diag_result:
            diag_summary = diag_result.get("summary", {})
            if diag_summary.get("errors", 0) > 0:
                risks.append(f"{diag_summary['errors']} diagnostic error(s) present")

        # 4. Decision
        if any("secret" in r.lower() or "credential" in r.lower() for r in risks):
            decision = "blocked"
        elif risks:
            decision = "request_changes"
        else:
            decision = "approve"

        requires_human = bool(files_changed) or bool(risks)

        review = {
            "decision": decision,
            "summary": (
                f"Reviewed {len(files_changed)} file(s). Decision: {decision}. "
                f"Risks: {len(risks)}."
            ),
            "risks": risks,
            "files_changed": files_changed,
            "diagnostics_summary": diag_summary,
            "requires_human_approval": requires_human,
        }

        # 5. Create approval if needed
        approval_id = None
        if requires_human:
            approval_id = self._create_approval(
                run,
                title=f"review_diff: {len(files_changed)} file(s) changed — {decision}",
                kind="code_review",
                details=review,
            )

        summary_str = review["summary"]
        run = db.update_run(
            run_id,
            status="waiting_approval" if requires_human else "completed",
            summary=summary_str,
            output=review,
            diagnostics_before=diag_result,
            approval_id=approval_id,
            completed_at=None if requires_human else _utc_now(),
        )
        db.close()

        self._add_session_event(
            code_session_id,
            "skill.waiting_approval" if requires_human else "skill.completed",
            message=summary_str,
            payload={"skill_run_id": run_id, "decision": decision},
        )

        return run

    def _run_stabilize_hanging_task(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and stabilize stuck sessions, flows, and commands."""
        run_id = run["id"]
        workspace_id = run["workspace_id"]
        code_session_id = run.get("code_session_id")
        input_data = run.get("input") or {}

        db = self._run_db()
        stabilized: List[Dict[str, Any]] = []

        # 1. Check stuck sessions
        stuck_sessions: List[Dict[str, Any]] = []
        try:
            sdb = self._session_db()
            if code_session_id:
                session = sdb.get_session(code_session_id)
                if session and session.get("status") in ("running", "planning", "coding", "running_tests", "reviewing"):
                    stuck_sessions.append(session)
            else:
                # List sessions for workspace that are in active states
                all_sessions = sdb.list_sessions(workspace_id=workspace_id)
                active_states = {"running", "planning", "coding", "running_tests", "reviewing"}
                stuck_sessions = [s for s in all_sessions if s.get("status") in active_states]
        except Exception as exc:
            logger.warning("stabilize: session check failed: %s", exc)

        # 2. Check stuck agent flows
        stuck_flows: List[Dict[str, Any]] = []
        try:
            agent_svc = self._agent_service()
            active_flow_statuses = {"planning", "coding", "running_tests", "reviewing"}
            flows = agent_svc.list_flows(
                code_session_id=code_session_id,
                workspace_id=workspace_id if not code_session_id else None,
            )
            stuck_flows = [f for f in flows if f.get("status") in active_flow_statuses]
        except Exception as exc:
            logger.warning("stabilize: flow check failed: %s", exc)

        # 3. Check running commands
        stuck_commands: List[Dict[str, Any]] = []
        target_session_id = code_session_id or input_data.get("code_session_id")
        if target_session_id:
            try:
                runner = self._command_runner()
                commands = runner.list_commands(target_session_id)
                stuck_commands = [c for c in commands if c.get("status") == "running"]
            except Exception as exc:
                logger.warning("stabilize: command check failed: %s", exc)

        # 4. Cancel safe stuck commands
        for cmd in stuck_commands:
            cmd_id = cmd["id"]
            cmd_str = cmd.get("command", "")
            safety = self._command_runner().classify_command(cmd_str)
            if safety == "safe":
                try:
                    runner = self._command_runner()
                    runner.cancel_command(cmd_id)
                    stabilized.append({"type": "command", "id": cmd_id, "action": "cancelled", "command": cmd_str})
                except Exception as exc:
                    logger.warning("stabilize: cancel command %s failed: %s", cmd_id, exc)
            else:
                approval_id = self._create_approval(
                    run,
                    title=f"stabilize: cancel command requires approval: {cmd_str[:60]}",
                    kind="command",
                    command=cmd_str,
                    details={"command_id": cmd_id, "run_id": run_id},
                )
                stabilized.append({
                    "type": "command", "id": cmd_id,
                    "action": "approval_requested", "approval_id": approval_id,
                })

        summary_parts = []
        if stuck_sessions:
            summary_parts.append(f"{len(stuck_sessions)} stuck session(s) detected")
        if stuck_flows:
            summary_parts.append(f"{len(stuck_flows)} stuck flow(s) detected")
        if stuck_commands:
            summary_parts.append(f"{len(stuck_commands)} stuck command(s) — {len([s for s in stabilized if s.get('action') == 'cancelled'])} cancelled")
        if not summary_parts:
            summary_parts.append("No stuck tasks detected")

        summary = "; ".join(summary_parts)
        output = {
            "stuck_sessions": [{"id": s["id"], "status": s.get("status")} for s in stuck_sessions],
            "stuck_flows": [{"id": f["id"], "status": f.get("status")} for f in stuck_flows],
            "stuck_commands": [{"id": c["id"], "command": c.get("command")} for c in stuck_commands],
            "stabilized": stabilized,
        }

        run = db.update_run(
            run_id,
            status="completed",
            summary=summary,
            output=output,
            completed_at=_utc_now(),
        )
        db.close()

        self._add_session_event(
            code_session_id,
            "skill.completed",
            message=summary,
            payload={"skill_run_id": run_id},
        )

        return run

    def _run_fix_runtime_error(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Parse runtime error, localise files, prepare diagnostic/correction plan."""
        run_id = run["id"]
        workspace_id = run["workspace_id"]
        code_session_id = run.get("code_session_id")
        input_data = run.get("input") or {}

        db = self._run_db()

        error_message = input_data.get("error_message", "")
        stack_trace = input_data.get("stack_trace", "")
        file_hint = input_data.get("file_hint", "")

        # Parse likely files from stack trace
        file_refs = self._extract_file_refs(stack_trace or error_message, workspace_id)
        if file_hint:
            file_refs.insert(0, file_hint)

        # Run diagnostics
        diag_before = None
        try:
            lsp = self._lsp_service()
            diag_before = lsp.run_diagnostics(workspace_id, code_session_id=code_session_id)
        except Exception as exc:
            logger.warning("fix_runtime_error: diagnostics failed: %s", exc)

        diag_summary = (diag_before or {}).get("summary", {})
        plan_steps = [
            f"Investigate: {f}" for f in file_refs[:5]
        ] + [
            "Review error message and stack trace",
            "Check diagnostics for related errors",
            "Propose targeted fix",
        ]

        output = {
            "error_message": error_message[:500],
            "file_refs": file_refs[:10],
            "plan": plan_steps,
            "diagnostics_summary": diag_summary,
        }

        summary = (
            f"Runtime error analysed. "
            f"{len(file_refs)} file reference(s) found. "
            f"Diagnostics: {diag_summary.get('errors', 0)} error(s)."
        )

        run = db.update_run(
            run_id,
            status="completed",
            summary=summary,
            output=output,
            diagnostics_before=diag_before,
            completed_at=_utc_now(),
        )
        db.close()

        self._add_session_event(
            code_session_id,
            "skill.completed",
            message=summary,
            payload={"skill_run_id": run_id},
        )

        return run

    def _run_implement_feature(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap MultiAgentCodingFlow to implement a feature."""
        run_id = run["id"]
        workspace_id = run["workspace_id"]
        code_session_id = run.get("code_session_id")
        input_data = run.get("input") or {}

        if not code_session_id:
            raise ValueError("implement_feature requires code_session_id")

        description = input_data.get("description") or input_data.get("feature") or "Implement feature"
        title = input_data.get("title") or description[:60]

        db = self._run_db()

        try:
            agent_svc = self._agent_service()
            flow = agent_svc.create_flow(
                code_session_id=code_session_id,
                workspace_id=workspace_id,
                description=description,
                title=title,
                provider=input_data.get("provider"),
                model=input_data.get("model"),
                preset=input_data.get("preset"),
            )
            flow_result = agent_svc.run_flow(flow["id"])
        except Exception as exc:
            raise RuntimeError(f"implement_feature: agent flow failed: {exc}") from exc

        agent_flow_id = flow_result["id"]
        flow_status = flow_result.get("status", "unknown")
        summary = f"Feature flow {flow_status}. Flow: {agent_flow_id}"

        run = db.update_run(
            run_id,
            status="waiting_approval" if flow_status == "waiting_approval" else "completed",
            summary=summary,
            agent_flow_id=agent_flow_id,
            output={"flow_id": agent_flow_id, "flow_status": flow_status},
            completed_at=None if flow_status == "waiting_approval" else _utc_now(),
        )
        db.close()

        self._add_session_event(
            code_session_id,
            "skill.completed",
            message=summary,
            payload={"skill_run_id": run_id, "agent_flow_id": agent_flow_id},
        )

        return run

    def _run_refactor_react_page(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare safe refactoring plan for a React page."""
        run_id = run["id"]
        workspace_id = run["workspace_id"]
        code_session_id = run.get("code_session_id")
        input_data = run.get("input") or {}

        page_path = input_data.get("page_path", "")
        goal = input_data.get("goal", "Refactor React page")
        preserve_design_system = input_data.get("preserve_design_system", True)

        db = self._run_db()

        # Diagnostics before
        diag_before = None
        try:
            lsp = self._lsp_service()
            diag_before = lsp.run_diagnostics(workspace_id, code_session_id=code_session_id)
        except Exception as exc:
            logger.warning("refactor_react_page: diagnostics failed: %s", exc)

        diag_summary = (diag_before or {}).get("summary", {})

        # Build refactor plan following HermesWeb principles
        plan_steps = [
            f"Analyse page: {page_path or 'target page'}",
            "Identify components and imports",
            "Check design system usage (no decorative animations, dark mode, clear hierarchy)",
            f"Goal: {goal}",
        ]
        if preserve_design_system:
            plan_steps.append("Preserve existing UI components and design tokens")

        risks: List[str] = []
        if diag_summary.get("errors", 0) > 0:
            risks.append(f"{diag_summary['errors']} type error(s) exist before refactor")
        if not page_path:
            risks.append("No page_path provided — target file unknown")

        output = {
            "page_path": page_path,
            "goal": goal,
            "plan_steps": plan_steps,
            "risks": risks,
            "preserve_design_system": preserve_design_system,
            "diagnostics_summary": diag_summary,
            "requires_human_approval": True,
            "hermesWeb_principles": [
                "Calm interface — no decorative animations",
                "Dark mode first",
                "Clear component hierarchy",
                "Low cognitive load",
            ],
        }

        # Create approval for the proposed refactor
        approval_id = self._create_approval(
            run,
            title=f"refactor_react_page: {page_path or 'page'} — {goal[:40]}",
            kind="code_review",
            details=output,
        )

        summary = f"Refactor plan ready for {page_path or 'page'}. {len(risks)} risk(s). Awaiting approval."

        run = db.update_run(
            run_id,
            status="waiting_approval",
            summary=summary,
            output=output,
            diagnostics_before=diag_before,
            approval_id=approval_id,
        )
        db.close()

        self._add_session_event(
            code_session_id,
            "skill.waiting_approval",
            message=summary,
            payload={"skill_run_id": run_id, "approval_id": approval_id},
        )

        return run

    def _run_benchmark_provider(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Compare providers on a task. dry_run=True by default — no file changes."""
        run_id = run["id"]
        workspace_id = run["workspace_id"]
        code_session_id = run.get("code_session_id")
        input_data = run.get("input") or {}

        db = self._run_db()

        task_prompt = input_data.get("task_prompt", "Analyse this workspace")
        providers = input_data.get("providers") or [{"provider": "anthropic", "model": "claude-sonnet-4-5"}]
        preset = input_data.get("preset", "planner")
        dry_run = input_data.get("dry_run", True)

        if not dry_run:
            approval_id = self._create_approval(
                run,
                title="benchmark_provider: dry_run=False requires approval before file changes",
                kind="code_review",
                details={"providers": providers, "task_prompt": task_prompt},
            )
            run = db.update_run(
                run_id,
                status="waiting_approval",
                summary="benchmark_provider: approval required for non-dry-run execution",
                approval_id=approval_id,
            )
            db.close()
            return run

        import time

        results: List[Dict[str, Any]] = []
        for p in providers:
            provider_name = p.get("provider", "unknown")
            model_name = p.get("model", "unknown")
            start = time.monotonic()

            # Dry-run: just build the plan without actually running anything
            entry: Dict[str, Any] = {
                "provider": provider_name,
                "model": model_name,
                "preset": preset,
                "dry_run": True,
                "task_prompt": task_prompt[:100],
            }

            try:
                if code_session_id:
                    agent_svc = self._agent_service()
                    flow = agent_svc.create_flow(
                        code_session_id=code_session_id,
                        workspace_id=workspace_id,
                        description=task_prompt,
                        provider=provider_name,
                        model=model_name,
                        preset=preset,
                    )
                    # Only run orchestrator (planning phase)
                    from hermes_cli.code.multi_agent_coding import AgentFlowDB
                    flow_db = AgentFlowDB(db_path=self._db_path)
                    plan = agent_svc._build_plan(flow, workspace_id)
                    flow_db.close()

                    elapsed = time.monotonic() - start
                    entry.update({
                        "status": "ok",
                        "flow_id": flow["id"],
                        "plan_steps": len(plan.get("steps", [])),
                        "test_commands": plan.get("test_commands", []),
                        "risks": plan.get("risks", []),
                        "elapsed_ms": int(elapsed * 1000),
                    })
                else:
                    elapsed = time.monotonic() - start
                    entry.update({
                        "status": "skipped",
                        "reason": "no code_session_id for flow creation",
                        "elapsed_ms": int(elapsed * 1000),
                    })
            except Exception as exc:
                elapsed = time.monotonic() - start
                entry.update({"status": "error", "error": str(exc), "elapsed_ms": int(elapsed * 1000)})

            results.append(entry)

        # Rank by plan quality (more steps = more comprehensive)
        ranked = sorted(results, key=lambda r: r.get("plan_steps", 0), reverse=True)

        output = {"results": ranked, "dry_run": dry_run, "task_prompt": task_prompt[:200]}
        summary = f"Benchmarked {len(providers)} provider(s). Fastest: {ranked[0]['provider']} ({ranked[0].get('elapsed_ms', 0)}ms)."

        run = db.update_run(
            run_id,
            status="completed",
            summary=summary,
            output=output,
            completed_at=_utc_now(),
        )
        db.close()

        self._add_session_event(
            code_session_id,
            "skill.completed",
            message=summary,
            payload={"skill_run_id": run_id, "providers_count": len(providers)},
        )

        return run

    # =========================================================================
    # Helpers
    # =========================================================================

    def _detect_build_commands(self, workspace_id: str) -> List[str]:
        """Detect safe build/test commands from workspace stack."""
        try:
            wdb = self._workspace_db()
            workspace = wdb.get_workspace(workspace_id)
            if not workspace:
                return []
            stack_raw = workspace.get("detected_stack_json") or workspace.get("detected_stack") or "[]"
            if isinstance(stack_raw, str):
                try:
                    stack = json.loads(stack_raw)
                except (json.JSONDecodeError, TypeError):
                    stack = []
            else:
                stack = stack_raw or []
        except Exception:
            stack = []

        commands: List[str] = []
        if any("typescript" in s.lower() or "node" in s.lower() or "javascript" in s.lower() for s in stack):
            pm = "npm"
            commands += [f"{pm} run typecheck", f"{pm} run lint", f"{pm} run build"]
        if any("go" in s.lower() for s in stack):
            commands += ["go vet ./...", "go test ./..."]
        if any("python" in s.lower() for s in stack):
            commands += ["python -m pytest"]
        if not commands:
            commands = ["make test"]
        return commands

    def _run_cmd_direct(self, cmd_str: str, workspace_id: str):
        """Run a command directly (when no code_session_id available)."""
        import subprocess

        try:
            wdb = self._workspace_db()
            workspace = wdb.get_workspace(workspace_id)
            cwd = workspace["path"] if workspace else None
        except Exception:
            cwd = None

        try:
            result = subprocess.run(
                cmd_str, shell=True, cwd=cwd,
                capture_output=True, text=True, timeout=120,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as exc:
            return -1, "", str(exc)

    def _extract_file_refs(self, text: str, workspace_id: str) -> List[str]:
        """Extract file paths from error/stack trace text."""
        import re

        workspace_path = ""
        try:
            wdb = self._workspace_db()
            ws = wdb.get_workspace(workspace_id)
            if ws:
                workspace_path = ws.get("path", "")
        except Exception:
            pass

        # Match common path patterns
        patterns = [
            r"(?:at |in |File )['\"]?([^\s'\"]+\.[a-z]{2,4}):\d+",
            r"([a-zA-Z0-9_./\-]+\.[a-z]{2,4}):\d+:\d+",
            r"([a-zA-Z0-9_./\-]+\.(ts|tsx|js|jsx|py|go|rs))",
        ]
        refs: List[str] = []
        seen: set = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                path = match.group(1)
                if path not in seen:
                    refs.append(path)
                    seen.add(path)
        return refs[:10]
