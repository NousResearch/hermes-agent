"""Kynver AgentOS memory/context provider for Hermes."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .agentos_bridge import (
    KynverAgentOSClient,
    KynverAgentOSConfig,
    agentos_enabled,
    redact,
)
from .contract import (
    MEMORY_WRITE_PATH,
    SESSION_OPEN_PATH,
    SKILL_LIST_PATH,
    TASK_CREATE_PATH,
    make_idempotency_key,
    memory_search_path,
    normalize_terminal_task_status,
    normalize_task_status,
    session_events_path,
    session_path,
    skill_get_path,
    task_close_path,
    task_events_path,
    task_list_path,
    task_steer_path,
    task_update_path,
    todo_to_task_record,
)
from .pre_transition import normalize_hermes_status
from .schemas import ALL_TOOL_SCHEMAS

logger = logging.getLogger(__name__)

_TERMINAL_HERMES_STATUSES = frozenset({"completed", "cancelled"})

RUNTIME = "hermes"
CALL_SIGN = "Forge"
CONTEXT_TAG = "hermes-forge"
SOURCE_ID = "hermes:forge"
DEFAULT_SEARCH_LIMIT = 5


def _todos_requiring_task_mirror(
    arg_todos: List[Dict[str, Any]],
    result_todos: List[Any],
    *,
    read_back_reconciled: bool,
) -> List[Dict[str, Any]]:
    """Skip task-plane mirrors for todos promoted to terminal only via Kynver read-back."""
    arg_by_id = {
        str(item.get("id")): item
        for item in arg_todos
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    mirrored: List[Dict[str, Any]] = []
    for todo in result_todos:
        if not isinstance(todo, dict):
            continue
        todo_id = str(todo.get("id") or "").strip()
        if not todo_id:
            continue
        arg = arg_by_id.get(todo_id)
        if not arg:
            mirrored.append(todo)
            continue
        arg_status = normalize_hermes_status(str(arg.get("status", "")))
        result_status = normalize_hermes_status(str(todo.get("status", "")))
        if (
            read_back_reconciled
            and arg_status not in _TERMINAL_HERMES_STATUSES
            and result_status in _TERMINAL_HERMES_STATUSES
            and arg_status != result_status
        ):
            continue
        mirrored.append(todo)
    return mirrored
TASK_EVENT_TYPES = {
    "created",
    "started",
    "worker_update",
    "blocked",
    "steer",
    "artifact",
    "review",
    "done",
    "failed",
}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _json_result(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _compact_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None and value != "" and value != []}


def _response_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("id", "taskId", "sessionId"):
        value = payload.get(key)
        if value:
            return str(value)
    for key in ("task", "session", "data", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            nested = _response_id(value)
            if nested:
                return nested
    return ""


def _coerce_items(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, str):
        try:
            return _coerce_items(json.loads(payload))
        except Exception:
            return [{"content": payload}]
    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {"content": str(item)} for item in payload]
    if not isinstance(payload, dict):
        return [{"content": str(payload)}]
    for key in ("structuredContent", "result", "data"):
        if key in payload:
            nested = _coerce_items(payload[key])
            if nested:
                return nested
    for key in ("memories", "results", "items", "tasks", "skills", "skill", "memory"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item if isinstance(item, dict) else {"content": str(item)} for item in value]
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            return [{"content": value}]
    if any(k in payload for k in ("id", "slug", "key", "content", "text", "title", "description")):
        return [payload]
    return []


def _item_text(item: dict[str, Any]) -> str:
    for key in ("content", "text", "memory", "summary", "description", "title"):
        value = item.get(key)
        if value:
            return str(value).strip()
    return ""


def _format_context(items: list[dict[str, Any]]) -> str:
    lines = ["## Kynver AgentOS Context", "Authoritative runtime memory for Hermes Forge."]
    count = 0
    for item in items[:DEFAULT_SEARCH_LIMIT]:
        text = _item_text(item)
        if not text:
            continue
        count += 1
        ref = item.get("key") or item.get("slug") or item.get("id") or item.get("sourceId")
        suffix = f" [{ref}]" if ref else ""
        lines.append(f"- {text}{suffix}")
    return "\n".join(lines) if count else ""


def _filter_skill_items(items: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    needle = query.strip().lower()
    if not needle:
        return items[:limit]
    matches = []
    for item in items:
        haystack = " ".join(
            str(item.get(key) or "")
            for key in ("slug", "id", "name", "description", "triggerRules", "category")
        ).lower()
        if needle in haystack:
            matches.append(item)
    return matches[:limit]


def _first_threat(content: str) -> Optional[str]:
    try:
        from tools.threat_patterns import first_threat_message

        return first_threat_message(content, scope="strict")
    except Exception:
        return None


class KynverMemoryProvider(MemoryProvider):
    """Authoritative MemoryProvider backed by Kynver AgentOS."""

    authoritative_context = False

    def __init__(self, client: Optional[Any] = None, config: Optional[KynverAgentOSConfig] = None):
        self._client = client
        self._config = config
        self._active = client is not None
        self._session_id = ""
        self._agentos_session_id = ""
        self._platform = ""
        self._model = ""
        self._agent_identity = ""
        self._agent_workspace = "hermes"
        self._user_id = ""
        self._degraded_reason = ""
        self._last_error = ""
        self._last_success = ""
        self._degraded_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "kynver"

    def is_available(self) -> bool:
        if self._client is not None:
            config = getattr(self._client, "config", self._config)
            return bool(getattr(config, "enabled", True))
        return agentos_enabled()

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        self._platform = str(kwargs.get("platform") or "cli")
        self._model = str(kwargs.get("model") or kwargs.get("model_name") or "")
        self._agent_identity = str(kwargs.get("agent_identity") or "")
        self._agent_workspace = str(kwargs.get("agent_workspace") or "hermes")
        self._user_id = str(kwargs.get("user_id") or kwargs.get("user_id_alt") or "")
        if self._client is None:
            self._client = KynverAgentOSClient(self._config)
        self._config = getattr(self._client, "config", self._config)
        self._active = bool(getattr(self._config, "enabled", True))
        if self._active and not self._sessions_disabled:
            self._open_session()

    @property
    def _observe_only(self) -> bool:
        return bool(getattr(self._config, "observe_only", False))

    @property
    def _memory_disabled(self) -> bool:
        return bool(getattr(self._config, "memory_disabled", False))

    @property
    def _tasks_disabled(self) -> bool:
        return bool(getattr(self._config, "tasks_disabled", False))

    @property
    def _skills_disabled(self) -> bool:
        return bool(getattr(self._config, "skills_disabled", False))

    @property
    def _sessions_disabled(self) -> bool:
        return bool(getattr(self._config, "session_sync_disabled", False))

    @property
    def _todo_disabled(self) -> bool:
        return bool(getattr(self._config, "todo_mirror_disabled", False))

    @property
    def _client_timeout(self) -> float:
        return float(getattr(self._config, "timeout", 3.0) or 3.0)

    @property
    def _side_effect_timeout(self) -> float:
        return float(getattr(self._config, "side_effect_timeout", self._client_timeout) or self._client_timeout)

    def _provenance(self) -> dict[str, Any]:
        data = {
            "runtime": RUNTIME,
            "callSign": CALL_SIGN,
            "contextTag": CONTEXT_TAG,
            "sourceId": SOURCE_ID,
            "hermesSessionId": self._session_id,
            "agentOSSessionId": self._agentos_session_id,
            "platform": self._platform,
            "agentWorkspace": self._agent_workspace,
            "observedAt": _now_iso(),
        }
        if self._agent_identity:
            data["agentIdentity"] = self._agent_identity
        if self._user_id:
            data["userId"] = self._user_id
        return data

    def _mark_degraded(self, where: str, exc: Exception) -> dict[str, Any]:
        reason = redact(str(exc))[:500]
        with self._degraded_lock:
            self._degraded_reason = f"{where}: {reason}"
            self._last_error = self._degraded_reason
        logger.warning("Kynver AgentOS degraded (%s): %s", where, reason)
        return {"provider": "kynver", "degraded": True, "where": where, "error": reason}

    def _mark_success(self, where: str) -> None:
        with self._degraded_lock:
            self._last_success = where
            self._degraded_reason = ""

    def is_authoritative_context(self) -> bool:
        """Kynver suppresses local MEMORY/USER only when memory is healthy."""
        if not self._active or self._observe_only or self._memory_disabled:
            return False
        with self._degraded_lock:
            return not bool(self._degraded_reason)

    def _require_client(self) -> Any:
        if not self._active or not self._client:
            raise RuntimeError("Kynver AgentOS is unavailable or disabled")
        return self._client

    def _open_session(self) -> None:
        if self._observe_only:
            return
        try:
            body = _compact_dict({
                "channel": self._platform,
                "model": self._model,
            })
            payload = self._require_client().post(
                SESSION_OPEN_PATH,
                body,
                timeout=self._side_effect_timeout,
            )
            if isinstance(payload, dict):
                self._agentos_session_id = _response_id(payload)
            self._mark_success("session.open")
        except Exception as exc:
            self._mark_degraded("session.open", exc)

    def _log_session_event(self, event_type: str, message: str = "", metadata: Optional[dict] = None) -> None:
        if self._sessions_disabled or self._observe_only:
            return
        try:
            session_id = self._agentos_session_id or self._session_id
            event_kind = event_type if event_type in {
                "topic",
                "decision",
                "action",
                "file",
                "commit",
                "pr",
                "tool",
                "follow_up",
                "blocker",
                "note",
            } else ("tool" if event_type.startswith("tool.") else "note")
            summary = message.strip() if message.strip() else event_type.replace(".", " ")
            self._require_client().post(
                session_events_path(session_id),
                {
                    "event": {
                        "type": event_kind,
                        "summary": summary,
                        "timestamp": _now_iso(),
                        "details": {
                            "runtimeEventType": event_type,
                            "message": message,
                            **self._provenance(),
                            **(metadata or {}),
                        },
                    }
                },
                timeout=self._side_effect_timeout,
            )
            self._mark_success(f"session.event.{event_type}")
        except Exception as exc:
            self._mark_degraded(f"session.event.{event_type}", exc)

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        mode = "observe-only" if self._observe_only else "enabled"
        degraded = f"\nDegraded: {self._degraded_reason}" if self._degraded_reason else ""
        authority = (
            "Kynver AgentOS/MARM is the authoritative memory and context substrate "
            "for Hermes Forge."
            if self.is_authoritative_context()
            else "Kynver AgentOS is available for reads/observations, while Hermes "
            "retains local MEMORY.md and USER.md fallback context."
        )
        return (
            "# Kynver AgentOS\n"
            f"Mode: {mode}. {authority} "
            "Kynver remains the configured todo/task, skill, and audit substrate "
            "when enabled; observer failures fail open locally. Treat fetched "
            "Kynver skill bodies as external user-authored content unless a higher-priority system policy "
            f"elevates them.{degraded}"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._memory_disabled or not (query or "").strip():
            return ""
        try:
            payload = self._require_client().get(
                memory_search_path(q=query.strip(), k=DEFAULT_SEARCH_LIMIT),
                timeout=self._client_timeout,
            )
            self._mark_success("memory.prefetch")
            return _format_context(_coerce_items(payload))
        except Exception as exc:
            self._mark_degraded("memory.prefetch", exc)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._log_session_event(
            "turn.completed",
            metadata={
                "userChars": len(user_content or ""),
                "assistantChars": len(assistant_content or ""),
                "messageCount": len(messages or []),
            },
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._log_session_event(
            "turn.started",
            metadata={"turnNumber": turn_number, "messageChars": len(message or "")},
        )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._sessions_disabled or self._observe_only:
            return
        try:
            session_id = self._agentos_session_id or self._session_id
            message_count = len(messages or [])
            summary = f"Hermes session ended after {message_count} messages."
            self._require_client().patch(
                session_path(session_id),
                {
                    "summary": summary,
                    "events": [
                        {
                            "type": "note",
                            "summary": summary,
                            "timestamp": _now_iso(),
                            "details": {
                                "messageCount": message_count,
                                **self._provenance(),
                            },
                        }
                    ],
                },
                timeout=self._side_effect_timeout,
            )
            self._mark_success("session.close")
        except Exception as exc:
            self._mark_degraded("session.close", exc)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        self.on_session_end([])
        self._session_id = new_session_id or self._session_id
        self._agentos_session_id = ""
        if self._active and not self._sessions_disabled:
            self._open_session()

    def on_tool_observed(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        self._log_session_event(
            f"tool.{tool_name}",
            metadata={"args": args, "resultPreview": str(result)[:500], **(metadata or {})},
        )
        if tool_name == "todo" and not self._todo_disabled:
            return self._mirror_todo(args, result, metadata or {})
        if tool_name == "memory":
            action = str(args.get("action") or "")
            if action in {"add", "replace"}:
                return self._mirror_memory_write(action, args, metadata or {})
        return None

    def _mirror_todo(
        self,
        args: Dict[str, Any],
        result: Any,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self._tasks_disabled or self._observe_only:
            return {"provider": "kynver", "todo_mirror": "observed", "durable": False}
        try:
            payload = json.loads(result) if isinstance(result, str) else result
            todos = payload.get("todos", []) if isinstance(payload, dict) else []
            read_back = (
                bool((payload.get("kynverReadBack") or {}).get("reconciled"))
                if isinstance(payload, dict)
                else False
            )
            arg_todos = args.get("todos") if isinstance(args.get("todos"), list) else []
            todos = _todos_requiring_task_mirror(
                [item for item in arg_todos if isinstance(item, dict)],
                todos,
                read_back_reconciled=read_back,
            )
            tasks = [
                todo_to_task_record(todo, SOURCE_ID)
                for todo in todos
                if isinstance(todo, dict)
            ]
            updated = 0
            unresolved = 0
            for task in tasks:
                create_payload = _compact_dict({
                    "title": task["title"],
                    "description": task.get("description"),
                    "idempotencyKey": task["idempotencyKey"],
                })
                created = self._require_client().post(
                    TASK_CREATE_PATH,
                    create_payload,
                    timeout=self._side_effect_timeout,
                )
                task_id = _response_id(created)
                status = normalize_task_status(task.get("status"), default="ready")
                if status == "ready":
                    continue
                if not task_id:
                    unresolved += 1
                    continue
                if status in {"done", "cancelled", "failed"}:
                    self._require_client().post(
                        task_close_path(task_id),
                        {
                            "status": normalize_terminal_task_status(status),
                            "summary": task.get("description") or task["title"],
                        },
                        timeout=self._side_effect_timeout,
                    )
                else:
                    self._require_client().patch(
                        task_update_path(task_id),
                        {"status": status},
                        timeout=self._side_effect_timeout,
                    )
                updated += 1
            self._mark_success("todo.mirror")
            result = {"provider": "kynver", "todo_mirror": "synced", "count": len(tasks), "state_updates": updated}
            if unresolved:
                result["unresolved_state_updates"] = unresolved
            return result
        except Exception as exc:
            return self._mark_degraded("todo.mirror", exc)

    def _mirror_memory_write(
        self,
        action: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self._memory_disabled or self._observe_only:
            return {"provider": "kynver", "memory_mirror": "observed", "durable": False}
        try:
            self._write_memory(
                str(args.get("content") or ""),
                memory_type="preference" if args.get("target") == "user" else "fact",
                metadata={**metadata, "action": action, "target": args.get("target") or "memory"},
                timeout=self._side_effect_timeout,
            )
            return {"provider": "kynver", "memory_mirror": "synced"}
        except Exception as exc:
            return self._mark_degraded("memory.write", exc)

    def _write_memory(
        self,
        content: str,
        *,
        key: str = "",
        memory_type: str = "fact",
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        if self._observe_only:
            raise RuntimeError("Kynver AgentOS is in observe mode; durable memory writes are disabled")
        clean = (content or "").strip()
        if not clean:
            raise ValueError("content is required")
        threat = _first_threat(clean)
        if threat:
            raise ValueError(threat)
        body = _compact_dict(
            {
                "content": clean,
                "slug": key or None,
                "memoryType": memory_type or "fact",
                "sourceId": SOURCE_ID,
                "metadata": {**self._provenance(), **(metadata or {})},
            }
        )
        payload = self._require_client().post(
            MEMORY_WRITE_PATH,
            body,
            timeout=timeout or self._client_timeout,
        )
        self._mark_success("memory.write")
        return payload

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if action in {"add", "replace"}:
            self._mirror_memory_write(action, {"content": content, "target": target}, metadata or {})

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = []
        for schema in ALL_TOOL_SCHEMAS:
            name = schema["name"]
            if name.startswith("kynver_task_") and self._tasks_disabled:
                continue
            if name.startswith("kynver_skill_") and self._skills_disabled:
                continue
            if name.startswith("kynver_memory_") and self._memory_disabled:
                continue
            schemas.append(schema)
        return schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "kynver_memory_search":
                return self._handle_memory_search(args)
            if tool_name == "kynver_memory_write":
                return self._handle_memory_write(args)
            if tool_name == "kynver_task_create":
                return self._handle_task_create(args)
            if tool_name == "kynver_task_update":
                return self._handle_task_update(args)
            if tool_name == "kynver_task_list":
                return self._handle_task_list(args)
            if tool_name == "kynver_task_close":
                return self._handle_task_close(args)
            if tool_name == "kynver_task_log_event":
                return self._handle_task_log_event(args)
            if tool_name == "kynver_task_steer":
                return self._handle_task_steer(args)
            if tool_name == "kynver_skill_list":
                return self._handle_skill_list(args)
            if tool_name == "kynver_skill_search":
                return self._handle_skill_search(args)
            if tool_name == "kynver_skill_get":
                return self._handle_skill_get(args)
        except Exception as exc:
            safe = redact(str(exc))[:500]
            return tool_error(f"{tool_name} failed: {safe}")
        return tool_error(f"Kynver provider does not handle tool '{tool_name}'")

    def _handle_memory_search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        k = max(1, min(20, int(args.get("k") or DEFAULT_SEARCH_LIMIT)))
        payload = self._require_client().get(
            memory_search_path(q=query, k=k),
            timeout=self._client_timeout,
        )
        self._mark_success("memory.search")
        memories = _coerce_items(payload)[:k]
        return _json_result({"success": True, "memories": memories, "count": len(memories), "sourceId": SOURCE_ID})

    def _handle_memory_write(self, args: Dict[str, Any]) -> str:
        result = self._write_memory(
            str(args.get("content") or ""),
            key=str(args.get("key") or "").strip(),
            memory_type=str(args.get("memoryType") or "fact").strip() or "fact",
            metadata={"tool": "kynver_memory_write"},
            timeout=self._client_timeout,
        )
        return _json_result({"success": True, "result": result, "sourceId": SOURCE_ID})

    def _handle_task_create(self, args: Dict[str, Any]) -> str:
        if self._tasks_disabled:
            return tool_error("Kynver tasks are disabled by KYNVER_TASKS_DISABLED")
        if self._observe_only:
            return tool_error("Kynver AgentOS is in observe mode; task writes are disabled")
        title = str(args.get("title") or "").strip()
        if not title:
            return tool_error("title is required")
        description = str(args.get("description") or "")
        idempotency_key = str(args.get("idempotencyKey") or "").strip() or make_idempotency_key(
            SOURCE_ID,
            "task.create",
            self._session_id,
            title,
            description,
        )
        body = _compact_dict({
            "title": title,
            "description": description,
            "priority": args.get("priority"),
            "executor": args.get("executor"),
            "executorRef": args.get("executorRef"),
            "parentTaskId": args.get("parentTaskId"),
            "goalId": args.get("goalId"),
            "projectId": args.get("projectId"),
            "personaSlug": args.get("personaSlug"),
            "scheduledFor": args.get("scheduledFor"),
            "dependsOnTaskIds": args.get("dependsOnTaskIds"),
            "idempotencyKey": idempotency_key,
            "requestId": args.get("requestId"),
        })
        payload = self._require_client().post(TASK_CREATE_PATH, body, timeout=self._client_timeout)
        self._mark_success("task.create")
        return _json_result({"success": True, "task": payload})

    def _handle_task_update(self, args: Dict[str, Any]) -> str:
        if self._observe_only:
            return tool_error("Kynver AgentOS is in observe mode; task writes are disabled")
        task_id = str(args.get("taskId") or "").strip()
        if not task_id:
            return tool_error("taskId is required")
        body: dict[str, Any] = {}
        for key in (
            "title",
            "description",
            "priority",
            "executor",
            "executorRef",
            "parentTaskId",
            "goalId",
            "projectId",
            "personaSlug",
            "scheduledFor",
            "dependsOnTaskIds",
            "lastSummary",
            "blocker",
            "branch",
            "worktreePath",
            "prUrl",
            "headCommit",
        ):
            if args.get(key) is not None:
                body[key] = args.get(key)
        if args.get("status") is not None:
            body["status"] = normalize_task_status(args.get("status"), default="running")
        body = _compact_dict(body)
        if not body:
            return tool_error("at least one supported task update field is required")
        payload = self._require_client().patch(task_update_path(task_id), body, timeout=self._client_timeout)
        self._mark_success("task.update")
        return _json_result({"success": True, "task": payload})

    def _handle_task_list(self, args: Dict[str, Any]) -> str:
        params: dict[str, Any] = {}
        if args.get("status"):
            params["status"] = normalize_task_status(args["status"], default=str(args["status"]))
        for key in ("executor", "parentTaskId", "personaSlug"):
            if args.get(key) is not None:
                params[key] = args.get(key)
        params["limit"] = max(1, min(100, int(args.get("limit") or 20)))
        payload = self._require_client().get(task_list_path(**params), timeout=self._client_timeout)
        self._mark_success("task.list")
        tasks = _coerce_items(payload)
        return _json_result({"success": True, "tasks": tasks, "count": len(tasks)})

    def _handle_task_close(self, args: Dict[str, Any]) -> str:
        if self._observe_only:
            return tool_error("Kynver AgentOS is in observe mode; task writes are disabled")
        task_id = str(args.get("taskId") or "").strip()
        if not task_id:
            return tool_error("taskId is required")
        payload = self._require_client().post(
            task_close_path(task_id),
            _compact_dict({
                "status": normalize_terminal_task_status(args.get("status")),
                "summary": str(args.get("summary") or args.get("message") or ""),
            }),
            timeout=self._client_timeout,
        )
        self._mark_success("task.close")
        return _json_result({"success": True, "task": payload})

    def _handle_task_log_event(self, args: Dict[str, Any]) -> str:
        if self._observe_only:
            return tool_error("Kynver AgentOS is in observe mode; task writes are disabled")
        task_id = str(args.get("taskId") or "").strip()
        raw_event_type = str(args.get("eventType") or "").strip()
        if not task_id or not raw_event_type:
            return tool_error("taskId and eventType are required")
        event_type = raw_event_type if raw_event_type in TASK_EVENT_TYPES else "worker_update"
        payload_body = args.get("payload") if isinstance(args.get("payload"), dict) else {}
        if raw_event_type != event_type:
            payload_body = {"runtimeEventType": raw_event_type, **payload_body}
        if args.get("message"):
            payload_body = {"message": str(args.get("message")), **payload_body}
        if isinstance(args.get("metadata"), dict):
            payload_body = {**payload_body, "metadata": args["metadata"]}
        payload = self._require_client().post(
            task_events_path(task_id),
            _compact_dict({
                "type": event_type,
                "payload": payload_body,
                "artifactVisibility": args.get("artifactVisibility"),
                "eventKey": str(args.get("eventKey") or "").strip() or make_idempotency_key(
                    SOURCE_ID,
                    "task.event",
                    self._session_id,
                    task_id,
                    event_type,
                    payload_body,
                ),
            }),
            timeout=self._client_timeout,
        )
        self._mark_success("task.log_event")
        return _json_result({"success": True, "event": payload})

    def _handle_task_steer(self, args: Dict[str, Any]) -> str:
        if self._observe_only:
            return tool_error("Kynver AgentOS is in observe mode; task writes are disabled")
        task_id = str(args.get("taskId") or "").strip()
        message = str(args.get("message") or "").strip()
        if not task_id or not message:
            return tool_error("taskId and message are required")
        detail = args.get("detail") if isinstance(args.get("detail"), dict) else {}
        if isinstance(args.get("metadata"), dict):
            detail = {**detail, **args["metadata"]}
        payload = self._require_client().post(
            task_steer_path(task_id),
            _compact_dict({
                "message": message,
                "detail": detail,
                "eventKey": str(args.get("eventKey") or "").strip() or make_idempotency_key(
                    SOURCE_ID,
                    "task.steer",
                    self._session_id,
                    task_id,
                    message,
                    detail,
                ),
            }),
            timeout=self._client_timeout,
        )
        self._mark_success("task.steer")
        return _json_result({"success": True, "steer": payload})

    def _handle_skill_list(self, args: Dict[str, Any]) -> str:
        limit = max(1, min(100, int(args.get("limit") or 50)))
        payload = self._require_client().get(SKILL_LIST_PATH, timeout=self._client_timeout)
        self._mark_success("skill.list")
        skills = _coerce_items(payload)
        if args.get("category"):
            category = str(args["category"]).lower()
            skills = [item for item in skills if str(item.get("category") or "").lower() == category]
        skills = skills[:limit]
        return _json_result({"success": True, "skills": skills, "count": len(skills), "manifest_only": True})

    def _handle_skill_search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        limit = max(1, min(100, int(args.get("limit") or 20)))
        payload = self._require_client().get(SKILL_LIST_PATH, timeout=self._client_timeout)
        self._mark_success("skill.search")
        skills = _filter_skill_items(_coerce_items(payload), query, limit)
        return _json_result({"success": True, "skills": skills, "count": len(skills), "manifest_only": True})

    def _handle_skill_get(self, args: Dict[str, Any]) -> str:
        skill_id = str(args.get("skillId") or "").strip()
        if not skill_id:
            return tool_error("skillId is required")
        payload = self._require_client().get(
            skill_get_path(skill_id, source=str(args.get("source") or "")),
            timeout=self._client_timeout,
        )
        self._mark_success("skill.get")
        return _json_result({
            "success": True,
            "skill": payload,
            "content_policy": "external_user_authored_content",
        })


def register(ctx) -> None:
    from .integration import ensure_kynver_operating_seams_registered, register_operating_providers
    from .operating_config import kynver_operating_tools_enabled

    ctx.register_memory_provider(KynverMemoryProvider())
    ensure_kynver_operating_seams_registered()
    register_operating_providers(ctx)
    if agentos_enabled() and kynver_operating_tools_enabled():
        logger.info(
            "Kynver operating tools enabled (default-on when AgentOS credentials are set; "
            "set KYNVER_OPERATING_TOOLS=false to opt out)"
        )
