"""Chorus memory provider for Hermes Agent.

Deep integration path:
- static prompt block confirms Chorus as active substrate
- startup warms identity/resume context
- prefetch injects relevant Chorus memory/resume context before turns
- session end, compression, delegation, and built-in memory writes mirror into Chorus
- bare tools expose common Chorus operations without going through MCP name noise
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from plugins.chorus_common import ChorusClient, compact_resume, emit_signal
from tools.registry import tool_error


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _trunc(text: str, limit: int = 2000) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 20] + "…[truncated]"


def _truthy(name: str, default: str = "") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


RESUME_SCHEMA = {
    "name": "chorus_resume_context",
    "description": "Fetch project-aware Chorus resume context for the active identity. Use for wake briefings, orientation, inbox/tasks, and recent memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string", "description": "Working directory hint for project inference."},
            "project_tag": {"type": "string", "description": "Explicit project tag/ring name."},
            "compact": {"type": "boolean", "description": "Return compact signal previews. Default true."},
            "limit": {"type": "integer", "description": "Items per section. Default 8."},
            "max_tokens": {"type": "integer", "description": "Approximate max tokens. Default 12000."},
        },
        "required": [],
    },
}

QUERY_SCHEMA = {
    "name": "chorus_memory_query",
    "description": "Semantic search over Chorus memory in a namespace.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "namespace": {"type": "string", "description": "Namespace such as ring:agents-of-proto or agent:vesta."},
            "limit": {"type": "integer", "description": "Default 8."},
            "tags": {"type": "array", "items": {"type": "string"}},
            "memory_type": {"type": "string"},
            "category": {"type": "string"},
            "entity": {"type": "string"},
        },
        "required": ["query", "namespace"],
    },
}

STORE_SCHEMA = {
    "name": "chorus_memory_store",
    "description": "Store a durable structured memory in Chorus. Store why, not temporary task progress.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "memory_type": {"type": "string", "enum": ["semantic", "charter", "episodic", "emotional", "reflective", "skill", "instruction", "resource", "decision", "policy", "protocol"]},
            "namespace": {"type": "string"},
            "entity": {"type": "string"},
            "category": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "source": {"type": "string"},
        },
        "required": ["content", "memory_type", "namespace"],
    },
}

SIGNAL_SCHEMA = {
    "name": "chorus_emit_signal",
    "description": "Emit a Chorus signal. Use for pulses, sense updates, alerts, queries, proposals, and handoffs.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "signal_type": {"type": "string", "enum": ["pulse", "sense", "task", "query", "alert", "artifact", "proposal", "shift"]},
            "to_ring": {"type": "string"},
            "from_role": {"type": "string"},
            "urgency": {"type": "number"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["content"],
    },
}


class ChorusMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self.client = ChorusClient.from_env(timeout=float(os.getenv("CHORUS_TIMEOUT", "20")))
        self.session_id = ""
        self.identity_name = ""
        self.agent_identity = ""
        self.platform = ""
        self._prefetch_cache = ""
        self._prefetch_lock = threading.Lock()
        self._last_resume_ts = 0.0
        self._resume_cache = ""
        self._sync_turns = _truthy("CHORUS_SYNC_TURNS", "false")
        self._session_emit = _truthy("CHORUS_SESSION_SIGNALS", "true")
        self._default_namespace = os.getenv("CHORUS_DEFAULT_NAMESPACE", "ring:agents-of-proto")

    @property
    def name(self) -> str:
        return "chorus"

    def is_available(self) -> bool:
        return self.client.is_configured()

    def initialize(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id or ""
        self.agent_identity = str(kwargs.get("agent_identity") or "")
        self.platform = str(kwargs.get("platform") or "")
        who = self.client.safe_rpc("identity/whoami", {})
        if who.get("ok") and isinstance(who.get("result"), dict):
            self.identity_name = who["result"].get("name") or ""
        self._refresh_resume(kwargs.get("cwd") or os.getcwd())

    def system_prompt_block(self) -> str:
        who = self.identity_name or self.agent_identity or "this profile"
        return (
            "Chorus memory provider is active. Treat Chorus as ambient durable memory and coordination substrate. "
            f"Authenticated identity: {who}. Prefer ring-scoped memories for shared project facts. "
            "Use chorus_resume_context for wake orientation, chorus_memory_query/store for durable recall, and chorus_emit_signal for visible coordination."
        )

    def _refresh_resume(self, cwd: str = "") -> None:
        now = time.time()
        if now - self._last_resume_ts < 60 and self._resume_cache:
            return
        params = {"cwd": cwd or os.getcwd(), "compact": True, "limit": 8, "max_tokens": 12000}
        res = self.client.safe_rpc("workflow/resume", params)
        if res.get("ok"):
            self._resume_cache = compact_resume(res.get("result"))
            self._last_resume_ts = now

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        parts: List[str] = []
        with self._prefetch_lock:
            if self._prefetch_cache:
                parts.append(self._prefetch_cache)
                self._prefetch_cache = ""
        if self._resume_cache:
            parts.append(self._resume_cache)
        if query:
            res = self.client.safe_rpc("memory/query", {"query": query, "namespace": self._default_namespace, "limit": 5, "tags": []})
            if res.get("ok"):
                memories = (res.get("result") or {}).get("memories") or []
                if memories:
                    lines = ["Relevant Chorus memory:"]
                    for m in memories[:5]:
                        lines.append(f"- {_trunc(m.get('content',''), 260)}")
                    parts.append("\n".join(lines))
        return "\n\n".join(p for p in parts if p).strip()

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query:
            return
        def _work() -> None:
            text = self.prefetch(query, session_id=session_id)
            if text:
                with self._prefetch_lock:
                    self._prefetch_cache = text
        threading.Thread(target=_work, daemon=True, name="chorus-prefetch").start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._sync_turns:
            return
        content = f"Hermes turn sync ({self.identity_name or self.agent_identity or 'agent'}):\nUser: {_trunc(user_content, 800)}\nAssistant: {_trunc(assistant_content, 1200)}"
        threading.Thread(target=lambda: self.client.safe_rpc("memory/note", {"content": content, "namespace": self._default_namespace, "tags": ["hermes-turn", self.identity_name or "agent"]}), daemon=True).start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RESUME_SCHEMA, QUERY_SCHEMA, STORE_SCHEMA, SIGNAL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "chorus_resume_context":
                params = {"compact": args.get("compact", True), "limit": args.get("limit", 8), "max_tokens": args.get("max_tokens", 12000)}
                if args.get("cwd"): params["cwd"] = args["cwd"]
                if args.get("project_tag"): params["project_tag"] = args["project_tag"]
                return _json(self.client.rpc("workflow/resume", params))
            if tool_name == "chorus_memory_query":
                params = {k: v for k, v in args.items() if v not in (None, "")}
                params.setdefault("limit", 8); params.setdefault("tags", [])
                return _json(self.client.rpc("memory/query", params))
            if tool_name == "chorus_memory_store":
                params = dict(args); params.setdefault("tags", []); params.setdefault("confidence", 1.0); params.setdefault("source", "hermes-chorus-provider")
                return _json(self.client.rpc("memory/store", params))
            if tool_name == "chorus_emit_signal":
                return _json(emit_signal(self.client, content=args["content"], signal_type=args.get("signal_type", "sense"), to_ring=args.get("to_ring", "agents-of-proto"), from_role=args.get("from_role", "ops"), urgency=float(args.get("urgency", 0.4)), tags=args.get("tags") or ["hermes", "chorus"]))
            return tool_error(f"Unknown Chorus memory provider tool: {tool_name}")
        except Exception as exc:
            return tool_error(str(exc))

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        if turn_number == 1:
            self._refresh_resume(str(kwargs.get("cwd") or os.getcwd()))

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._session_emit:
            return
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
        content = f"Hermes session ended for {self.identity_name or self.agent_identity or 'agent'}: session={self.session_id}, user_turns={user_turns}, assistant_turns={assistant_turns}."
        self.client.safe_rpc("memory/note", {"content": content, "namespace": self._default_namespace, "tags": ["session-end", "hermes", self.identity_name or "agent"]})
        self.client.safe_rpc("signals/batch_emit", {"signals": [{"signal_type": "sense", "content": content, "from_role": "ops", "to_ring": "fleet", "urgency": 0.25, "tags": ["hermes", "session-end", self.identity_name or "agent"], "resources": [], "attachments": []}]})

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ""
        summary = f"Compression boundary for {self.identity_name or self.agent_identity or 'agent'}: preserving {len(messages)} messages in Chorus session trace pointer {self.session_id}."
        self.client.safe_rpc("memory/note", {"content": summary, "namespace": self._default_namespace, "tags": ["compression-boundary", "hermes"]})
        return summary

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        content = f"Delegation observed by {self.identity_name or self.agent_identity or 'agent'}: child_session={child_session_id}\nTask: {_trunc(task, 1000)}\nResult: {_trunc(result, 1600)}"
        self.client.safe_rpc("memory/note", {"content": content, "namespace": self._default_namespace, "tags": ["delegation", "worker-audit", "hermes"]})

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        note = f"Built-in Hermes memory {action} mirrored to Chorus. target={target}\n{_trunc(content, 1800)}"
        self.client.safe_rpc("memory/note", {"content": note, "namespace": self._default_namespace, "tags": ["builtin-memory-mirror", target, action]})

    def shutdown(self) -> None:
        return None


def register_memory_provider():
    return ChorusMemoryProvider()
