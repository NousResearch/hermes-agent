"""Local web Mission Control for Agents OS.

This module deliberately binds to 127.0.0.1 by default and exposes a thin,
local-only API/UI over the existing Agents OS SQLite control-plane. It does not
send data to external services, restart gateways, read credentials, or deploy.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sqlite3
import sys
import threading
import uuid
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from hermes_cli import agents_os

API_ROUTES = [
    "/api/status",
    "/api/dashboard",
    "/api/tasks",
    "/api/approvals",
    "/api/runs",
    "/api/events",
    "/api/agents",
    "/api/workflows",
    "/api/safety",
    "/api/artifacts",
]
STATIC_DIR = Path(__file__).with_name("agents_os_web_static")


def ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None}


def err(code: str, message: str, status: int = 400) -> dict[str, Any]:
    return {"ok": False, "data": None, "error": {"code": code, "message": message, "status": status}}


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return agents_os.row_to_dict(row) if row is not None else None


def _json_or_empty(raw: bytes | str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = raw.strip()
    if not raw:
        return {}
    return json.loads(raw)


@dataclass
class RequestResult:
    status: int
    content_type: str
    body: bytes


class MissionControlWebApp:
    def __init__(self, paths: agents_os.AgentsOSPaths | None = None, *, bind_host: str = "127.0.0.1"):
        self.paths = paths or agents_os.resolve_paths(None)
        self.bind_host = bind_host

    # ---------- query helpers ----------
    def _connect(self) -> sqlite3.Connection:
        return agents_os.connect(self.paths)

    def _doctor_payload(self) -> dict[str, Any]:
        with self._connect() as conn:
            schema_row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            schema_version = schema_row[0] if schema_row else None
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            orphan_artifacts = conn.execute("SELECT COUNT(*) FROM artifacts WHERE task_id IS NOT NULL AND task_id != '' AND task_id NOT IN (SELECT id FROM tasks)").fetchone()[0]
            orphan_events = conn.execute("SELECT COUNT(*) FROM events WHERE task_id IS NOT NULL AND task_id != '' AND task_id NOT IN (SELECT id FROM tasks)").fetchone()[0]
            orphan_runs = conn.execute("SELECT COUNT(*) FROM runs WHERE task_id IS NOT NULL AND task_id != '' AND task_id NOT IN (SELECT id FROM tasks)").fetchone()[0]
            pending_approvals = conn.execute("SELECT COUNT(*) FROM approvals WHERE status='pending'").fetchone()[0]
        required = {"meta", "tasks", "approvals", "artifacts", "events", "runs", "agents", "workflows", "routing_rules", "reviews", "state_snapshots"}
        policy_home_isolated = str(self.paths.root).startswith(str(self.paths.home)) and ".hermes-marija" not in str(self.paths.root) and ".openclaw" not in str(self.paths.root)
        checks = {
            "state_db_exists": self.paths.db.exists(),
            "schema_version": schema_version,
            "schema_current": schema_version == agents_os.SCHEMA_VERSION,
            "required_tables_present": required.issubset(set(tables)),
            "pending_approvals": pending_approvals,
            "orphan_records": orphan_artifacts + orphan_events + orphan_runs,
            "policy_home_isolated": policy_home_isolated,
            "network_side_effects": False,
            "runtime_config_changed": False,
            "gateway_restart": False,
        }
        checks["ok"] = bool(checks["state_db_exists"] and checks["schema_current"] and checks["required_tables_present"] and checks["orphan_records"] == 0 and checks["policy_home_isolated"])
        return checks

    def status_payload(self) -> dict[str, Any]:
        checks = self._doctor_payload()
        return {
            "status": "ok" if checks["ok"] else "attention",
            "bind_host": self.bind_host,
            "agents_os_home": str(self.paths.root),
            "state_db": str(self.paths.db),
            "vault_root": str(self.paths.vault_root),
            "schema_version": checks["schema_version"],
            "routes": ["/"] + API_ROUTES,
            "safety": {
                "local_only": True,
                "network_side_effects": False,
                "deploy": False,
                "gateway_restart": False,
                "credentials_touched": False,
            },
        }

    def dashboard_payload(self) -> dict[str, Any]:
        with self._connect() as conn:
            tasks = [_row(r) for r in conn.execute("""
                SELECT * FROM tasks ORDER BY CASE status
                WHEN 'blocked' THEN 0 WHEN 'needs_approval' THEN 1 WHEN 'review' THEN 2
                WHEN 'in_progress' THEN 3 WHEN 'ready' THEN 4 WHEN 'pending' THEN 5
                WHEN 'routed' THEN 6 WHEN 'new' THEN 7 WHEN 'completed' THEN 8 ELSE 9 END,
                priority ASC, created_at ASC LIMIT 100
            """).fetchall()]
            approvals = [_row(r) for r in conn.execute("SELECT * FROM approvals ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, created_at DESC LIMIT 100").fetchall()]
            runs = [_row(r) for r in conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 100").fetchall()]
            artifacts = [_row(r) for r in conn.execute("SELECT * FROM artifacts ORDER BY created_at DESC LIMIT 100").fetchall()]
            events = [_row(r) for r in conn.execute("SELECT * FROM events ORDER BY created_at DESC LIMIT 100").fetchall()]
            agents = self._agent_rows(conn)
            workflows = self._workflow_rows(conn)
        queue_summary = self._queue_summary(tasks, approvals, runs)
        next_task = next((t for t in tasks if t["status"] in {"ready", "pending", "routed"} and not t.get("approval_required")), None)
        return {
            "status": self.status_payload(),
            "queue_summary": queue_summary,
            "next_best_action": self._next_best_action(queue_summary, next_task),
            "tasks": tasks,
            "approvals": approvals,
            "runs": self._annotate_runs(runs),
            "artifacts": artifacts,
            "events": events,
            "agents": agents,
            "workflows": workflows,
            "safety": self.safety_payload(),
        }

    def _queue_summary(self, tasks: list[dict[str, Any]], approvals: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, int]:
        open_status = {"new", "pending", "routed", "ready", "in_progress", "needs_approval"}
        summary = {
            "open_tasks": sum(1 for t in tasks if t["status"] in open_status),
            "running_tasks": sum(1 for t in tasks if t["status"] == "in_progress"),
            "blocked_tasks": sum(1 for t in tasks if t["status"] == "blocked"),
            "review_tasks": sum(1 for t in tasks if t["status"] == "review"),
            "completed_tasks": sum(1 for t in tasks if t["status"] == "completed"),
            "pending_approvals": sum(1 for a in approvals if a["status"] == "pending"),
            "failed_executions": sum(1 for r in runs if r["status"] == "failed"),
        }
        summary["action_required"] = summary["blocked_tasks"] + summary["review_tasks"] + summary["pending_approvals"] + summary["failed_executions"]
        return summary

    def _next_best_action(self, summary: dict[str, int], next_task: dict[str, Any] | None) -> dict[str, Any]:
        if summary["pending_approvals"]:
            return {"kind": "approval", "label": "Review pending approvals", "priority": 1}
        if summary["failed_executions"]:
            return {"kind": "triage", "label": "Triage failed runs", "priority": 2}
        if summary["review_tasks"]:
            return {"kind": "review", "label": "Close or return review tasks", "priority": 3}
        if next_task:
            return {"kind": "task", "label": f"Route/execute {next_task['id']}: {next_task['title']}", "task_id": next_task["id"], "priority": 4}
        return {"kind": "idle", "label": "System clean — no action required", "priority": 99}

    def _annotate_runs(self, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        execution_task_ids = {r.get("task_id") for r in runs if r.get("task_id") and (r.get("completed_at") or r.get("status") in {"succeeded", "failed", "blocked", "needs_review"})}
        for r in runs:
            if r.get("completed_at") or r.get("status") in {"succeeded", "failed", "blocked", "needs_review"}:
                r["kind"] = "execution"
            elif r.get("task_id") in execution_task_ids:
                r["kind"] = "draft_superseded"
            else:
                r["kind"] = "draft"
        return runs

    def _agent_rows(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = [_row(r) for r in conn.execute("SELECT * FROM agents ORDER BY status ASC, created_at ASC").fetchall()]
        existing = {r["id"] for r in rows}
        pseudo = [
            {"id": "kodi-codex", "name": "Kodi / Codex", "kind": "external-cli", "status": "not_verified", "capabilities": '["code","review"]', "created_at": "", "home": "Codex auth/runtime separate", "memory_boundary": "No Doni memory merge"},
            {"id": "marija-profile", "name": "Marija", "kind": "separate-profile", "status": "not_verified", "capabilities": '["assistant"]', "created_at": "", "home": "/home/goran/.hermes-marija-clean", "memory_boundary": "Separate memory/session/auth"},
            {"id": "ero-openclaw", "name": "ERO / OpenClaw", "kind": "reference-runtime", "status": "not_verified", "capabilities": '["reference","runtime"]', "created_at": "", "home": "/home/goran/.openclaw/workspace", "memory_boundary": "Reference layer only"},
        ]
        return rows + [p for p in pseudo if p["id"] not in existing]

    def _workflow_rows(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = [_row(r) for r in conn.execute("SELECT * FROM workflows ORDER BY id ASC").fetchall()]
        seen = {r["id"] for r in rows}
        for wid, spec in agents_os.SAFE_WORKFLOWS.items():
            if wid not in seen:
                rows.append({"id": wid, "kind": spec["kind"], "requires_approval": int(spec["requires_approval"]), "template": spec["template"], "route": "doni:direct", "capabilities": "[]", "allowed_paths": "[]", "blocked_paths": "[]", "created_at": ""})
        extras = [
            {"id": "seo-web-task", "kind": "seo_web", "requires_approval": 0, "template": "SEO/web local analysis task", "route": "doni:direct", "capabilities": '["seo","web"]', "allowed_paths": "[]", "blocked_paths": "[]", "created_at": ""},
            {"id": "memory-curation-draft", "kind": "memory", "requires_approval": 1, "template": "Draft memory curation; approval before durable changes", "route": "approval_gate", "capabilities": '["memory"]', "allowed_paths": "[]", "blocked_paths": "[]", "created_at": ""},
            {"id": "cron-watchdog-draft", "kind": "ops", "requires_approval": 1, "template": "Draft cron/watchdog; approval before scheduling", "route": "approval_gate", "capabilities": '["ops"]', "allowed_paths": "[]", "blocked_paths": "[]", "created_at": ""},
        ]
        known = {r["id"] for r in rows}
        rows.extend([x for x in extras if x["id"] not in known])
        return rows

    def safety_payload(self) -> dict[str, Any]:
        doctor = self._doctor_payload()
        mirror_path = self.paths.vault_root / "00-command-center" / "RUNTIME-DASHBOARD.md"
        credential_like_matches = 0
        if self.paths.vault_root.exists():
            for md_path in self.paths.vault_root.rglob("*.md"):
                try:
                    credential_like_matches += agents_os.leak_scan_text(md_path.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    continue
        return {
            "doctor": doctor,
            "mirror_validate": {"status": "ok" if mirror_path.exists() and credential_like_matches == 0 else "attention", "dashboard_path": str(mirror_path), "issues": [] if mirror_path.exists() else ["missing_dashboard"], "credential_like_matches": credential_like_matches},
            "profile_home_isolation": doctor["policy_home_isolated"],
            "doni_marija_ero_separation": True,
            "network_side_effects": False,
            "runtime_config_changed": False,
            "gateway_restart": False,
            "credential_scan": {"credential_like_matches": credential_like_matches, "status": "ok" if credential_like_matches == 0 else "attention"},
        }

    # ---------- mutations ----------
    def create_task(self, data: dict[str, Any]) -> dict[str, Any]:
        title = str(data.get("title") or "").strip()
        if not title:
            return err("validation_error", "title is required")
        workflow = str(data.get("workflow") or "code-task")
        priority = int(data.get("priority") or 3)
        notes = str(data.get("notes") or "")
        task_id = str(data.get("id") or f"task-{uuid.uuid4().hex[:8]}")
        spec = agents_os.SAFE_WORKFLOWS.get(workflow, {})
        approval_required = 1 if spec.get("requires_approval") else 0
        status = "needs_approval" if approval_required else "pending"
        now = agents_os.utc_now()
        with self._connect() as conn:
            conn.execute("INSERT INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,approval_required) VALUES(?,?,?,?,?,?,?,?,?)", (task_id, title, status, workflow, priority, now, now, notes, approval_required))
            agents_os.log_event(conn, "task_created", task_id=task_id, payload={"source": "mission_control_web", "workflow": workflow, "status": status})
            conn.commit()
            task = _row(conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())
        return ok(task)

    def route_task(self, task_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if task is None:
                return err("task_not_found", f"Task not found: {task_id}", 404)
            route = agents_os._route_for_task(conn, task)
            pending = agents_os._pending_approval_count(conn, task_id)
            requires_approval = route["requires_approval"] or pending > 0 or bool(task["approval_required"])
            no_agent = not requires_approval and route.get("assigned_agent") is None
            new_status = "needs_approval" if requires_approval else ("blocked" if no_agent else "ready")
            conn.execute("UPDATE tasks SET status=?, route=?, approval_required=?, updated_at=? WHERE id=?", (new_status, route["route"], 1 if requires_approval else 0, agents_os.utc_now(), task_id))
            agents_os.log_event(conn, "routed", task_id=task_id, payload={"route": route["route"], "status": new_status, "reason": route["reason"], "assigned_agent": route.get("assigned_agent"), "source": "mission_control_web"})
            conn.commit()
        return ok({"task_id": task_id, "route": route["route"], "reason": route["reason"], "assigned_agent": route.get("assigned_agent"), "approval_required": requires_approval, "execution_allowed": not requires_approval and not no_agent, "new_status": new_status})

    def execute_task(self, task_id: str) -> dict[str, Any]:
        args = argparse.Namespace(id=task_id, dry_run=False, json=True, vault_root=str(self.paths.vault_root))
        with self._connect() as conn:
            task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if task is None:
                return err("task_not_found", f"Task not found: {task_id}", 404)
            if task["approval_required"] or agents_os._pending_approval_count(conn, task_id) > 0 or task["status"] == "needs_approval":
                agents_os.log_event(conn, "execution_blocked", task_id=task_id, payload={"reason": "approval_required", "source": "mission_control_web"})
                conn.commit()
                return err("approval_required", "Task requires approval before execution", 403)
            run_id = f"run-{uuid.uuid4().hex[:8]}"
            now = agents_os.utc_now()
            log_path = self.paths.artifacts / "runs" / f"{now.split('T',1)[0]}-{run_id}.md"
            body = f"## Execution log\n\n- task_id: {task_id}\n- workflow: {task['workflow'] or ''}\n- source: mission_control_web\n- status: succeeded\n"
            agents_os.write_markdown(log_path, f"Run {run_id}", body, {"run_id": run_id, "task_id": task_id, "status": "succeeded", "created_at": now})
            conn.execute("INSERT INTO runs(id,task_id,workflow,status,input,created_at,completed_at) VALUES(?,?,?,?,?,?,?)", (run_id, task_id, task["workflow"] or "manual", "succeeded", task["notes"] or "", now, agents_os.utc_now()))
            artifact_id = f"artifact-{uuid.uuid4().hex[:8]}"
            conn.execute("INSERT INTO artifacts(id,kind,title,path,task_id,workflow,created_at,run_id) VALUES(?,?,?,?,?,?,?,?)", (artifact_id, "run-log", f"Run log {run_id}", str(log_path), task_id, task["workflow"], now, run_id))
            conn.execute("UPDATE tasks SET status='review', updated_at=? WHERE id=?", (agents_os.utc_now(), task_id))
            agents_os.log_event(conn, "executed", task_id=task_id, run_id=run_id, payload={"status": "succeeded", "log_path": str(log_path), "source": "mission_control_web"})
            conn.commit()
        return ok({"task_id": task_id, "run_id": run_id, "status": "succeeded", "log_path": str(log_path), "artifact_id": artifact_id})

    def close_task(self, task_id: str, data: dict[str, Any]) -> dict[str, Any]:
        evidence = str(data.get("evidence") or "").strip()
        if not evidence:
            return err("validation_error", "evidence is required")
        with self._connect() as conn:
            task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if task is None:
                return err("task_not_found", f"Task not found: {task_id}", 404)
            if task["approval_required"] or agents_os._pending_approval_count(conn, task_id) > 0 or task["status"] == "needs_approval":
                agents_os.log_event(conn, "close_blocked", task_id=task_id, payload={"reason": "approval_required", "source": "mission_control_web"})
                conn.commit()
                return err("approval_required", "Task requires approval before close", 403)
            now = agents_os.utc_now()
            conn.execute("UPDATE tasks SET status='completed', updated_at=? WHERE id=?", (now, task_id))
            agents_os.log_event(conn, "task_closed", task_id=task_id, payload={"evidence": evidence, "source": "mission_control_web"})
            conn.commit()
        return ok({"task_id": task_id, "status": "completed", "evidence": evidence})

    def run_workflow(self, workflow: str, data: dict[str, Any]) -> dict[str, Any]:
        if workflow not in agents_os.SAFE_WORKFLOWS:
            return err("workflow_not_found", f"Unknown workflow: {workflow}", 404)
        input_text = str(data.get("input") or data.get("notes") or "").strip()
        if not input_text:
            return err("validation_error", "input is required")
        title = str(data.get("title") or f"{workflow}: {input_text[:60]}")
        task_id = str(data.get("task_id") or f"task-{uuid.uuid4().hex[:8]}")
        priority = int(data.get("priority") or 3)
        spec = agents_os.SAFE_WORKFLOWS[workflow]
        now = agents_os.utc_now()
        artifact_id = f"artifact-{uuid.uuid4().hex[:8]}"
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        target = self.paths.artifacts / workflow / f"{now.split('T',1)[0]}-{agents_os.slugify(title)}.md"
        body = f"## Input\n{input_text}\n\n## Workflow\n{workflow}\n\n## Template\n{spec['template']}\n\n## Execution state\n- status: draft-created\n- approval_required: {spec['requires_approval']}\n"
        agents_os.write_markdown(target, title, body, {"id": artifact_id, "type": spec["kind"], "workflow": workflow, "task_id": task_id, "status": "draft-created", "created_at": now})
        approval_id = None
        initial_status = "needs_approval" if spec["requires_approval"] else "pending"
        with self._connect() as conn:
            conn.execute("INSERT INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,approval_required) VALUES(?,?,?,?,?,?,?,?,?)", (task_id, title, initial_status, workflow, priority, now, now, input_text, 1 if spec["requires_approval"] else 0))
            conn.execute("INSERT INTO runs(id,task_id,workflow,status,input,created_at) VALUES(?,?,?,?,?,?)", (run_id, task_id, workflow, "created", input_text, now))
            conn.execute("INSERT INTO artifacts(id,kind,title,path,task_id,workflow,created_at,run_id) VALUES(?,?,?,?,?,?,?,?)", (artifact_id, spec["kind"], title, str(target), task_id, workflow, now, run_id))
            agents_os.log_event(conn, "task_created", task_id=task_id, run_id=run_id, payload={"workflow": workflow, "status": initial_status, "source": "mission_control_web"})
            agents_os.log_event(conn, "artifact_created", task_id=task_id, run_id=run_id, payload={"artifact_id": artifact_id, "path": str(target), "source": "mission_control_web"})
            if spec["requires_approval"]:
                approval_id = f"approval-{uuid.uuid4().hex[:8]}"
                conn.execute("INSERT INTO approvals(id,title,status,risk,task_id,payload,created_at) VALUES(?,?,?,?,?,?,?)", (approval_id, f"Approval required: {title}", "pending", "external-action", task_id, input_text, now))
                agents_os.log_event(conn, "approval_requested", task_id=task_id, run_id=run_id, payload={"approval_id": approval_id, "risk": "external-action", "source": "mission_control_web"})
            conn.commit()
        return ok({"task_id": task_id, "run_id": run_id, "artifact_id": artifact_id, "artifact_path": str(target), "approval_id": approval_id})

    def resolve_approval(self, approval_id: str, status: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        if status not in {"approved", "rejected", "cancelled"}:
            return err("validation_error", "invalid approval status")
        notes = str((data or {}).get("notes") or "")
        with self._connect() as conn:
            approval = conn.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
            if approval is None:
                return err("approval_not_found", f"Approval not found: {approval_id}", 404)
            now = agents_os.utc_now()
            conn.execute("UPDATE approvals SET status=?, resolved_at=? WHERE id=?", (status, now, approval_id))
            task_id = approval["task_id"]
            if task_id:
                if status == "approved":
                    pending = conn.execute("SELECT COUNT(*) FROM approvals WHERE task_id=? AND status='pending' AND id != ?", (task_id, approval_id)).fetchone()[0]
                    if pending == 0:
                        conn.execute("UPDATE tasks SET approval_required=0, status=CASE WHEN status='needs_approval' THEN 'ready' ELSE status END, updated_at=? WHERE id=?", (now, task_id))
                elif status == "rejected":
                    conn.execute("UPDATE tasks SET status='blocked', updated_at=? WHERE id=?", (now, task_id))
                agents_os.log_event(conn, "approval_resolved", task_id=task_id, payload={"approval_id": approval_id, "status": status, "notes": notes, "source": "mission_control_web"})
            conn.commit()
        return ok({"approval_id": approval_id, "status": status, "task_id": task_id, "notes": notes})

    # ---------- artifacts ----------
    def artifacts_payload(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [_row(r) for r in conn.execute("SELECT * FROM artifacts ORDER BY created_at DESC LIMIT 200").fetchall()]

    def _allowed_path(self, path: Path) -> bool:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            resolved = path.expanduser().absolute()
        allowed_roots = [self.paths.artifacts.resolve(), self.paths.vault_root.resolve()]
        forbidden = {".env", "auth.json", "credentials.json", "config.yaml"}
        if resolved.name in forbidden or any(part.lower() in {"secrets", "credentials"} for part in resolved.parts):
            return False
        return any(str(resolved).startswith(str(root)) for root in allowed_roots)

    def preview_path(self, path_value: str) -> dict[str, Any]:
        path = Path(path_value)
        if not self._allowed_path(path):
            return err("path_not_allowed", "Artifact preview is restricted to Agents OS artifact and vault roots", 403)
        if not path.exists() or not path.is_file():
            return err("file_not_found", str(path), 404)
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return ok({"path": str(path), "preview_type": "image", "content": str(path)})
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > 60000:
            text = text[:30000] + "\n\n... [truncated] ...\n\n" + text[-10000:]
        preview_type = "json" if suffix == ".json" else ("markdown" if suffix in {".md", ".markdown"} else "text")
        return ok({"path": str(path), "preview_type": preview_type, "content": text})

    def artifact_preview(self, artifact_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
            if row is None:
                return err("artifact_not_found", f"Artifact not found: {artifact_id}", 404)
            artifact = _row(row)
        preview = self.preview_path(artifact["path"])
        if preview["ok"]:
            preview["data"]["artifact"] = artifact
            if preview["data"].get("preview_type") == "image":
                preview["data"]["raw_url"] = f"/api/artifacts/{artifact_id}/raw"
        return preview

    def artifact_raw(self, artifact_id: str) -> RequestResult:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
            if row is None:
                payload = err("artifact_not_found", f"Artifact not found: {artifact_id}", 404)
                return RequestResult(404, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))
            artifact = _row(row)
        path = Path(artifact["path"])
        if not self._allowed_path(path):
            payload = err("path_not_allowed", "Artifact preview is restricted to Agents OS artifact and vault roots", 403)
            return RequestResult(403, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))
        if not path.exists() or not path.is_file():
            payload = err("file_not_found", str(path), 404)
            return RequestResult(404, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return RequestResult(200, ctype, path.read_bytes())

    # ---------- dispatch ----------
    def handle_json(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        parsed = urlparse(path)
        route = parsed.path.rstrip("/") or "/"
        try:
            if method == "GET" and route == "/api/status":
                return ok(self.status_payload())
            if method == "GET" and route == "/api/dashboard":
                return ok(self.dashboard_payload())
            if method == "GET" and route == "/api/tasks":
                return ok(self.dashboard_payload()["tasks"])
            if method == "GET" and route == "/api/approvals":
                return ok(self.dashboard_payload()["approvals"])
            if method == "GET" and route == "/api/runs":
                return ok(self.dashboard_payload()["runs"])
            if method == "GET" and route == "/api/events":
                return ok(self.dashboard_payload()["events"])
            if method == "GET" and route == "/api/agents":
                return ok(self.dashboard_payload()["agents"])
            if method == "GET" and route == "/api/workflows":
                return ok(self.dashboard_payload()["workflows"])
            if method == "GET" and route == "/api/safety":
                return ok(self.safety_payload())
            if method == "GET" and route == "/api/artifacts":
                return ok(self.artifacts_payload())
            m = re.fullmatch(r"/api/artifacts/([^/]+)", route)
            if method == "GET" and m:
                return self.artifact_preview(m.group(1))
            if method == "POST" and route == "/api/tasks":
                return self.create_task(body)
            m = re.fullmatch(r"/api/tasks/([^/]+)/route", route)
            if method == "POST" and m:
                return self.route_task(m.group(1))
            m = re.fullmatch(r"/api/tasks/([^/]+)/execute", route)
            if method == "POST" and m:
                return self.execute_task(m.group(1))
            m = re.fullmatch(r"/api/tasks/([^/]+)/close", route)
            if method == "POST" and m:
                return self.close_task(m.group(1), body)
            m = re.fullmatch(r"/api/approvals/([^/]+)/(approve|deny|reject)", route)
            if method == "POST" and m:
                status = "approved" if m.group(2) == "approve" else "rejected"
                return self.resolve_approval(m.group(1), status, body)
            m = re.fullmatch(r"/api/workflows/([^/]+)/run", route)
            if method == "POST" and m:
                return self.run_workflow(m.group(1), body)
            return err("not_found", f"No route for {method} {route}", 404)
        except Exception as exc:  # defensive: UI should receive JSON not traceback
            return err("internal_error", str(exc), 500)

    def handle_http(self, method: str, path: str, raw_body: bytes | None = None) -> RequestResult:
        parsed = urlparse(path)
        route = parsed.path
        if method == "GET" and route in {"/", "/index.html"}:
            return self._static_response("index.html")
        if method == "GET" and route.startswith("/static/"):
            return self._static_response(route.removeprefix("/static/"))
        raw_match = re.fullmatch(r"/api/artifacts/([^/]+)/raw", route)
        if method == "GET" and raw_match:
            return self.artifact_raw(raw_match.group(1))
        if route.startswith("/api/"):
            payload = self.handle_json(method, path, _json_or_empty(raw_body))
            status = payload.get("error", {}).get("status", 200) if not payload.get("ok") else 200
            return RequestResult(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        payload = err("not_found", f"No route for {method} {route}", 404)
        return RequestResult(404, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _static_response(self, relative: str) -> RequestResult:
        rel = relative.strip("/") or "index.html"
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists() or not target.is_file():
            payload = err("not_found", f"Static file not found: {rel}", 404)
            return RequestResult(404, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix in {".html", ".css", ".js"}:
            ctype += "; charset=utf-8"
        return RequestResult(200, ctype, target.read_bytes())


def create_server(app: MissionControlWebApp, host: str = "127.0.0.1", port: int = 18790) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Agents OS Mission Control web server is local-only; host must be 127.0.0.1")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, result: RequestResult) -> None:
            self.send_response(result.status)
            self.send_header("Content-Type", result.content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(result.body)

        def do_GET(self) -> None:  # noqa: N802
            self._send(app.handle_http("GET", self.path))

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b""
            self._send(app.handle_http("POST", self.path, raw))

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[agents-os-web] " + fmt % args + "\n")

    return ThreadingHTTPServer((host, port), Handler)


def web_cmd(args: argparse.Namespace) -> int:
    host = args.host or "127.0.0.1"
    if host not in {"127.0.0.1", "localhost"}:
        print("Agents OS web server is local-only; use --host 127.0.0.1", file=sys.stderr)
        return 2
    app = MissionControlWebApp(agents_os.resolve_paths(args), bind_host=host)
    routes = ["/"] + API_ROUTES
    payload = {"status": "ok", "bind_host": host, "port": args.port, "url": f"http://{host}:{args.port}", "routes": routes, "safety": app.status_payload()["safety"]}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    server = create_server(app, host=host, port=args.port)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"Agents OS Mission Control: {url}")
    print("Local-only. Press Ctrl+C to stop.")
    if args.open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Agents OS Mission Control.")
    finally:
        server.server_close()
    return 0
