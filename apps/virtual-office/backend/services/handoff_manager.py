from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException

from backend.models.handoff import Handoff
from backend.models.task import Task
from backend.services.codex_bridge import CodexBridge
from backend.services.json_store import HANDOFFS_PATH, TASKS_PATH, read_list_store, write_list_store
from backend.services.log_store import append_event
from backend.services.settings_store import get_settings


class HandoffManager:
    def __init__(self) -> None:
        self.codex_bridge = CodexBridge()

    def _model_dump(self, model: Handoff | Task) -> dict:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    def create_handoff(
        self,
        from_agent: str,
        to_agent: str,
        payload: dict,
        auto_run: bool = False,
        existing_task: dict | None = None,
    ) -> dict:
        now = datetime.now(UTC).isoformat()
        handoff = Handoff(
            id=str(uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            status="running" if auto_run and to_agent == "codex" and payload.get("prompt") else "pending",
            payload=payload,
            created_at=now,
        )
        handoffs = read_list_store(HANDOFFS_PATH)
        handoff_record = self._model_dump(handoff)
        handoffs.append(handoff_record)
        write_list_store(HANDOFFS_PATH, handoffs)

        if auto_run and to_agent == "codex" and payload.get("prompt"):
            return self._run_codex_handoff(handoff_record, existing_task=existing_task)

        if existing_task is not None:
            task_record = dict(existing_task)
            task_record["handoff_id"] = handoff_record["id"]
            task_record["status"] = "pending"
            task_record["updated_at"] = datetime.now(UTC).isoformat()
            write_list_store(TASKS_PATH, self._replace_task(task_record))

        return handoff_record

    def run_task(self, task_id: str, from_agent: str = "hermes") -> dict:
        tasks = read_list_store(TASKS_PATH)
        task = next((item for item in tasks if item.get("id") == task_id), None)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        prompt = str(task.get("goal") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Task goal is required to run")

        settings = get_settings()
        to_agent = str(task.get("agent") or "codex")
        payload = {
            "prompt": prompt,
            "context": str(task.get("context") or ""),
            "workdir": str(settings.get("codex_workdir") or r"D:\Codex"),
            "timeout": 180,
            "room": str(task.get("room") or "main-office"),
            "task_id": task_id,
        }
        return self.create_handoff(
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            auto_run=to_agent == "codex",
            existing_task=task,
        )

    def retry_task(self, task_id: str, from_agent: str = "hermes") -> dict:
        self.requeue_task(task_id)
        return self.run_task(task_id, from_agent=from_agent)

    def requeue_task(self, task_id: str) -> dict:
        tasks = read_list_store(TASKS_PATH)
        task = next((item for item in tasks if item.get("id") == task_id), None)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        task["status"] = "pending"
        task["result"] = None
        task["error"] = None
        task["handoff_id"] = None
        task["updated_at"] = datetime.now(UTC).isoformat()
        write_list_store(TASKS_PATH, self._replace_task(task))
        return task

    def run_handoff_again(self, handoff_id: str) -> dict:
        handoffs = read_list_store(HANDOFFS_PATH)
        handoff = next((item for item in handoffs if item.get("id") == handoff_id), None)
        if handoff is None:
            raise HTTPException(status_code=404, detail="Handoff not found")

        payload = handoff.get("payload") or {}
        handoff_payload = payload if isinstance(payload, dict) else {"text": str(payload)}
        auto_run = str(handoff.get("to_agent") or "") == "codex" and bool(handoff_payload.get("prompt"))
        return self.create_handoff(
            from_agent=str(handoff.get("from_agent") or "hermes"),
            to_agent=str(handoff.get("to_agent") or "codex"),
            payload=handoff_payload,
            auto_run=auto_run,
        )

    def _run_codex_handoff(self, handoff: dict, existing_task: dict | None = None) -> dict:
        now = datetime.now(UTC).isoformat()
        payload = handoff.get("payload", {}) or {}
        prompt = str(payload.get("prompt") or "").strip()
        workdir = str(payload.get("workdir") or r"D:\Codex")

        if existing_task is None:
            title = prompt[:80] or "Codex handoff"
            task = Task(
                id=str(uuid4()),
                title=title,
                goal=prompt,
                agent="codex",
                room=str(payload.get("room") or "main-office"),
                status="in_progress",
                handoff_id=handoff["id"],
                context=str(payload.get("context") or ""),
                created_at=now,
                updated_at=now,
            )
            tasks = read_list_store(TASKS_PATH)
            task_record = self._model_dump(task)
            tasks.append(task_record)
            write_list_store(TASKS_PATH, tasks)
        else:
            task_record = dict(existing_task)
            task_record["status"] = "in_progress"
            task_record["handoff_id"] = handoff["id"]
            task_record["updated_at"] = now
            write_list_store(TASKS_PATH, self._replace_task(task_record))

        start_event = append_event(
            level="INFO",
            message="Starting Codex handoff execution",
            agent="codex",
            task_id=task_record["id"],
            handoff_id=handoff["id"],
        )
        handoff["log_refs"] = [start_event["id"]]
        write_list_store(HANDOFFS_PATH, self._replace_handoff(handoff))

        try:
            result = self.codex_bridge.exec(
                prompt=prompt,
                workdir=workdir,
                timeout=int(payload.get("timeout") or 120),
            )
            task_record["status"] = "completed"
            task_record["result"] = result.get("output") or ""
            task_record["error"] = None
            task_record["updated_at"] = datetime.now(UTC).isoformat()
            write_list_store(TASKS_PATH, self._replace_task(task_record))

            success_event = append_event(
                level="INFO",
                message="Codex handoff completed",
                agent="codex",
                task_id=task_record["id"],
                handoff_id=handoff["id"],
            )
            handoff["status"] = "completed"
            handoff["result"] = {
                "task_id": task_record["id"],
                "session_id": result.get("session_id"),
                "exit_code": result.get("exit_code"),
                "output_preview": str(result.get("output") or "")[:400],
            }
            handoff["completed_at"] = datetime.now(UTC).isoformat()
            handoff["log_refs"] = [start_event["id"], success_event["id"]]
            write_list_store(HANDOFFS_PATH, self._replace_handoff(handoff))
            return handoff
        except Exception as exc:
            task_record["status"] = "failed"
            task_record["error"] = str(exc)
            task_record["updated_at"] = datetime.now(UTC).isoformat()
            write_list_store(TASKS_PATH, self._replace_task(task_record))

            failure_event = append_event(
                level="ERROR",
                message=f"Codex handoff failed: {exc}",
                agent="codex",
                task_id=task_record["id"],
                handoff_id=handoff["id"],
            )
            handoff["status"] = "failed"
            handoff["result"] = {
                "task_id": task_record["id"],
                "error": str(exc),
            }
            handoff["completed_at"] = datetime.now(UTC).isoformat()
            handoff["log_refs"] = [start_event["id"], failure_event["id"]]
            write_list_store(HANDOFFS_PATH, self._replace_handoff(handoff))
            return handoff

    def _replace_task(self, task: dict) -> list[dict]:
        tasks = read_list_store(TASKS_PATH)
        return [task if item.get("id") == task.get("id") else item for item in tasks]

    def _replace_handoff(self, handoff: dict) -> list[dict]:
        handoffs = read_list_store(HANDOFFS_PATH)
        return [handoff if item.get("id") == handoff.get("id") else item for item in handoffs]

    def route_to_codex(self, prompt: str, workdir: str | None = None) -> dict:
        return self.codex_bridge.exec(prompt=prompt, workdir=workdir)
