"""Tool bridge for the Hermes voice channel.

The bridge is intentionally transport-neutral: Gemini Live or another realtime
audio frontend can fetch the tool declarations, then POST function calls to the
API server.  Side effects go through existing Hermes primitives so voice remains
one more channel over the same core, memory, and delivery tools.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import uuid
from typing import Any, Awaitable, Callable, Dict, List


HER_VOICE_SYSTEM_PROMPT = """\
You are Hermes vocal.

Tone:
- Warm, calm, concise.
- Never hurry the user.
- Auto-detect French or English and answer in the same language.
- Prefer short spoken replies. Ask one focused clarification only when needed.

Continuity:
- Treat this as the voice channel of the same Hermes workspace.
- Important voice turns are saved in the shared session database with channel voice.
- When recalling prior discussions, search memory before answering.
"""


def her_voice_tool_declarations() -> List[Dict[str, Any]]:
    """Return Gemini-compatible function declarations for the voice bridge."""
    return [
        {
            "name": "run_hermes_agent",
            "description": "Run a task through Hermes Core with optional context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "context": {"type": "string"},
                    "session_id": {"type": "string"},
                },
                "required": ["task"],
            },
        },
        {
            "name": "create_issue",
            "description": "Create a Multica issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "assignee": {"type": "string"},
                },
                "required": ["title"],
            },
        },
        {
            "name": "list_issues",
            "description": "List Multica issues with an optional filter object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "priority": {"type": "string"},
                            "assignee": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                },
            },
        },
        {
            "name": "send_whatsapp",
            "description": "Send a WhatsApp message through Hermes delivery.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
        {
            "name": "send_discord",
            "description": "Send a Discord message through Hermes delivery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "channel": {"type": "string"},
                },
                "required": ["message"],
            },
        },
        {
            "name": "get_memory",
            "description": "Search shared Hermes conversation memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_calendar",
            "description": "Read calendar context through configured Hermes tools.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "read_email",
            "description": "Read email context through configured Hermes tools.",
            "parameters": {
                "type": "object",
                "properties": {"filter": {"type": "string"}},
            },
        },
    ]


class HerVoiceBridge:
    """Execute voice function calls against Hermes and local platform tools."""

    def __init__(
        self,
        *,
        run_agent: Callable[..., Awaitable[tuple]],
        session_db_factory: Callable[[], Any],
        timeout_seconds: float = 120.0,
    ) -> None:
        self._run_agent = run_agent
        self._session_db_factory = session_db_factory
        self._timeout_seconds = timeout_seconds

    async def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(arguments, dict):
            arguments = {}

        handlers = {
            "run_hermes_agent": self._run_hermes_agent,
            "create_issue": self._create_issue,
            "list_issues": self._list_issues,
            "send_whatsapp": self._send_whatsapp,
            "send_discord": self._send_discord,
            "get_memory": self._get_memory,
            "get_calendar": self._get_calendar,
            "read_email": self._read_email,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown voice tool: {name}"}

        try:
            return await asyncio.wait_for(handler(arguments), timeout=self._timeout_seconds)
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"Voice tool timed out: {name}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def _run_hermes_agent(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        task = str(arguments.get("task") or "").strip()
        if not task:
            return {"ok": False, "error": "task is required"}

        context = str(arguments.get("context") or "").strip()
        session_id = str(arguments.get("session_id") or "").strip() or f"voice-{uuid.uuid4().hex[:16]}"
        user_message = task if not context else f"{task}\n\nContext:\n{context}"

        result, usage = await self._run_agent(
            user_message=user_message,
            conversation_history=[],
            ephemeral_system_prompt=HER_VOICE_SYSTEM_PROMPT,
            session_id=session_id,
            platform="voice",
        )
        return {
            "ok": True,
            "session_id": session_id,
            "response": result.get("final_response") or result.get("error") or "",
            "usage": usage,
        }

    async def _create_issue(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        title = str(arguments.get("title") or "").strip()
        if not title:
            return {"ok": False, "error": "title is required"}

        cmd = ["multica", "issue", "create", "--title", title]
        description = str(arguments.get("description") or "").strip()
        assignee = str(arguments.get("assignee") or "").strip()
        if description:
            cmd.extend(["--description", description])
        if assignee:
            cmd.extend(["--assignee", assignee])
        return await self._run_multica(cmd)

    async def _list_issues(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        filt = arguments.get("filter") or {}
        if not isinstance(filt, dict):
            filt = {}

        cmd = ["multica", "issue", "list", "--output", "json"]
        for key in ("status", "priority", "assignee"):
            value = str(filt.get(key) or "").strip()
            if value:
                cmd.extend([f"--{key}", value])
        limit = filt.get("limit")
        if isinstance(limit, int) and 0 < limit <= 100:
            cmd.extend(["--limit", str(limit)])
        return await self._run_multica(cmd)

    async def _send_whatsapp(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        message = str(arguments.get("message") or "").strip()
        if not message:
            return {"ok": False, "error": "message is required"}
        return await self._send_message("whatsapp", message)

    async def _send_discord(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        message = str(arguments.get("message") or "").strip()
        if not message:
            return {"ok": False, "error": "message is required"}
        channel = str(arguments.get("channel") or "").strip()
        target = "discord" if not channel else f"discord:{channel}"
        return await self._send_message(target, message)

    async def _get_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "query is required"}
        limit = arguments.get("limit")
        if not isinstance(limit, int) or limit <= 0:
            limit = 8
        limit = min(limit, 25)

        db = self._session_db_factory()
        if db is None:
            return {"ok": False, "error": "Session database is unavailable"}

        results = db.search_messages(query=query, limit=limit)
        memories = [
            {
                "session_id": row.get("session_id"),
                "role": row.get("role"),
                "source": row.get("source"),
                "channel": row.get("channel"),
                "snippet": row.get("snippet"),
                "content": row.get("content"),
                "timestamp": row.get("timestamp"),
            }
            for row in results
        ]
        return {"ok": True, "count": len(memories), "memories": memories}

    async def _get_calendar(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return await self._run_context_task(
            "Read my calendar and summarize the relevant upcoming events.",
            arguments,
        )

    async def _read_email(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        filter_text = str(arguments.get("filter") or "").strip()
        task = "Read my email."
        if filter_text:
            task = f"Read my email matching this filter: {filter_text}"
        return await self._run_context_task(task, arguments)

    async def _run_context_task(self, task: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(arguments)
        merged["task"] = task
        return await self._run_hermes_agent(merged)

    async def _send_message(self, target: str, message: str) -> Dict[str, Any]:
        def _send() -> str:
            from tools.send_message_tool import send_message_tool
            return send_message_tool({"action": "send", "target": target, "message": message})

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _send)
        return self._json_result(result)

    async def _run_multica(self, cmd: List[str]) -> Dict[str, Any]:
        env = os.environ.copy()
        env.setdefault("NO_COLOR", "1")

        def _run() -> Dict[str, Any]:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=30,
                env=env,
                check=False,
            )
            if proc.returncode != 0:
                return {
                    "ok": False,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                }
            payload = self._json_result(proc.stdout)
            payload.setdefault("ok", True)
            return payload

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run)

    @staticmethod
    def _json_result(raw: str) -> Dict[str, Any]:
        raw = (raw or "").strip()
        if not raw:
            return {"ok": True}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"ok": True, "output": raw}
        if isinstance(parsed, dict):
            parsed.setdefault("ok", True)
            return parsed
        return {"ok": True, "result": parsed}
