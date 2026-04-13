"""MemOS memory plugin — MemoryProvider interface.

Server-side memory extraction and semantic search via MemOS Platform.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    from hermes_constants import get_hermes_home

    config = {
        "api_key": os.environ.get("MEMOS_API_KEY", ""),
        "base_url": os.environ.get("MEMOS_BASE_URL", "https://memos.memtensor.cn/api/openmem/v1"),
        "user_id": os.environ.get("MEMOS_USER_ID", "hermes_user"),
        "knowledgebase": None,
        "allowedAgents": None,
        "multiAgentMode": False,
    }

    config_path = get_hermes_home() / "memos.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
        except Exception:
            pass

    return config

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "memos_search",
    "description": "Search user's memories using MemOS Platform.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for in memories."},
        },
        "required": ["query"],
    },
}

ADD_MESSAGE_SCHEMA = {
    "name": "memos_add_message",
    "description": "Explicitly store a fact or message into MemOS memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact or message to store."},
        },
        "required": ["content"],
    },
}

# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class MemosMemoryProvider(MemoryProvider):
    """MemOS Platform memory provider."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._base_url = ""
        self._user_id = "hermes_user"
        self._agent_id = ""
        self._session_id = ""
        self._sync_thread = None
        self._allowed_agents = None
        self._multi_agent_mode = False

    @property
    def name(self) -> str:
        return "memos"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("api_key"))

    def save_config(self, values: dict, hermes_home: str) -> None:
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "memos.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "MemOS API key",
                "secret": True,
                "required": True,
                "env_var": "MEMOS_API_KEY",
                "url": "https://memos-dashboard.openmem.net/cn/apikeys/",
            },
            {
                "key": "user_id",
                "description": "MemOS user ID",
                "default": "hermes_user",
            },
            {
                "key": "knowledgebase",
                "description": "Knowledgebase ID or list of IDs for searching. e.g., 'kb-123' or ['kb-123', 'kb-456']",
                "required": False,
            },
            {
                "key": "allowedAgents",
                "type": "list",
                "description": "List of agent IDs allowed to use memory. If empty, all agents are allowed. e.g., ['agent-1', 'agent-2']",
                "required": False,
            },
            {
                "key": "multiAgentMode",
                "description": "Enable multi-agent memory isolation. e.g., true (for isolation) or false (for shared)",
                "required": False,
                "default": False,
            }
        ]

    def _get_client(self):
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from memos.api.client import MemOSClient
                if self._base_url:
                    os.environ["MEMOS_BASE_URL"] = self._base_url
                self._client = MemOSClient(api_key=self._api_key)
                return self._client
            except ImportError:
                raise RuntimeError("MemoryOS package not installed. Run: pip install MemoryOS")

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("api_key", "")
        self._base_url = self._config.get("base_url", "")
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes_user")
        self._agent_id = kwargs.get("agent_id") or kwargs.get("agent_identity", "hermes")
        self._session_id = session_id
        def _parse_list_or_str(val):
            if not val:
                return None
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    return parsed if isinstance(parsed, list) else [val]
                except json.JSONDecodeError:
                    return [val]
            return val if isinstance(val, list) else [val]

        self._knowledgebase = _parse_list_or_str(self._config.get("knowledgebase"))

        self._allowed_agents = _parse_list_or_str(self._config.get("allowedAgents"))

        multi_agent = self._config.get("multiAgentMode", False)
        if isinstance(multi_agent, str):
            self._multi_agent_mode = multi_agent.lower() in ("true", "1", "yes", "y", "t")
        else:
            self._multi_agent_mode = bool(multi_agent)

    def _is_memory_enabled(self) -> bool:
        if not self._allowed_agents:
            return True
        return self._agent_id in self._allowed_agents

    def system_prompt_block(self) -> str:
        if not self._is_memory_enabled():
            return ""
        return (
            "# MemOS Memory\n"
            f"Active. User: {self._user_id}.\n"
            "Use memos_search to find memories, and memos_add_message to store facts."
        )

    def _build_search_filter(self) -> dict | None:
        agent_id = self._agent_id if self._multi_agent_mode else None
        
        if agent_id:
            return {"user": {"and": [{"agent_id": agent_id}]}}
            
        return None

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._is_memory_enabled():
            return ""
        if len(query) < 3:
            return ""
        try:
            client = self._get_client()
            sid = session_id or self._session_id
            
            kwargs = {
                "query": query,
                "user_id": self._user_id,
                "conversation_id": sid,
            }
            if self._knowledgebase:
                kwargs["knowledgebase_ids"] = self._knowledgebase if isinstance(self._knowledgebase, list) else [self._knowledgebase]
            search_filter = self._build_search_filter()
            if search_filter:
                kwargs["filter"] = search_filter
                
            res = client.search_memory(**kwargs)
            if res is None:
                return ""
            if hasattr(res, "model_dump"):
                res = res.model_dump()
            elif hasattr(res, "dict"):
                res = res.dict()
            
            if res.get("code") == 0 and "data" in res:
                data = res["data"]
                memories = []
                if "memory_detail_list" in data and isinstance(data["memory_detail_list"], list):
                    for m in data["memory_detail_list"]:
                        if m.get("memory_value"):
                            memories.append(m["memory_value"])
                if "preference_detail_list" in data and isinstance(data["preference_detail_list"], list):
                    prefs = [p["preference"] for p in data["preference_detail_list"] if p.get("preference")]
                    if prefs:
                        if len(prefs) == 1:
                            memories.append(f"Preference: {prefs[0]}")
                        else:
                            memories.append("Preferences:\n  - " + "\n  - ".join(prefs))
                            
                if memories:
                    result = "\n".join(f"- {m}" for m in memories)
                    return f"## MemOS Memory\n{result}"
        except Exception as e:
            logger.debug("MemOS prefetch failed: %s", e, exc_info=True)

        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._is_memory_enabled():
            return
            
        def _sync():
            try:
                client = self._get_client()
                sid = session_id or self._session_id
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                info = {
                    "source": "Hermes Agent",
                    "agent_id": self._agent_id
                }
                client.add_message(
                    messages=messages, 
                    user_id=self._user_id, 
                    conversation_id=sid, 
                    agent_id=self._agent_id,
                    source="Hermes Agent",
                    info=info
                )
            except Exception as e:
                logger.warning("MemOS sync failed: %s", e, exc_info=True)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="memos-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, ADD_MESSAGE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if not self._is_memory_enabled():
            return tool_error("Memory is disabled for this agent.")

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        sid = kwargs.get("session_id") or self._session_id

        if tool_name == "memos_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            if len(query) < 3:
                return json.dumps({"result": "No relevant memories found."}, ensure_ascii=False)
            try:
                kwargs = {
                    "query": query,
                    "user_id": self._user_id,
                    "conversation_id": sid,
                }
                if self._knowledgebase:
                    kwargs["knowledgebase_ids"] = self._knowledgebase if isinstance(self._knowledgebase, list) else [self._knowledgebase]
                search_filter = self._build_search_filter()
                if search_filter:
                    kwargs["filter"] = search_filter

                res = client.search_memory(**kwargs)
                if res is None:
                    return json.dumps({"result": "No relevant memories found or API error."}, ensure_ascii=False)
                if hasattr(res, "model_dump"):
                    res = res.model_dump()
                elif hasattr(res, "dict"):
                    res = res.dict()
                
                if res.get("code") == 0 and "data" in res:
                    data = res["data"]
                    results = []
                    if "memory_detail_list" in data and isinstance(data["memory_detail_list"], list):
                        for m in data["memory_detail_list"]:
                            if m.get("memory_value"):
                                results.append(m["memory_value"])
                    if "preference_detail_list" in data and isinstance(data["preference_detail_list"], list):
                        for p in data["preference_detail_list"]:
                            if p.get("preference"):
                                results.append(f"Preference: {p['preference']}")
                    
                    if not results:
                        return json.dumps({"result": "No relevant memories found."}, ensure_ascii=False)
                    return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)
                else:
                    return json.dumps({"result": "No relevant memories found or API error."}, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Search failed: {e}")

        elif tool_name == "memos_add_message":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            try:
                messages = [{"role": "user", "content": content}]
                info = {
                    "source": "Hermes Agent",
                    "agent_id": self._agent_id
                }
                res = client.add_message(
                    messages=messages, 
                    user_id=self._user_id, 
                    conversation_id=sid, 
                    agent_id=self._agent_id,
                    source="Hermes Agent",
                    info=info
                )
                if res is None:
                    return json.dumps({"error": "API error: No response"}, ensure_ascii=False)
                if hasattr(res, "model_dump"):
                    res = res.model_dump()
                elif hasattr(res, "dict"):
                    res = res.dict()
                    
                if res.get("code") == 0:
                    return json.dumps({"result": "Fact stored successfully."}, ensure_ascii=False)
                else:
                    return json.dumps({"error": f"API error: {res.get('message', 'Unknown error')}"}, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        with self._client_lock:
            self._client = None

def register(ctx) -> None:
    """Register MemOS as a memory provider plugin."""
    ctx.register_memory_provider(MemosMemoryProvider())
