"""天人·Anima memory plugin — MemoryProvider interface.

天人·Anima 认知记忆引擎。记忆即人格。
Supports two modes:
  - local_mode=True:  Direct Memory() from anima (SQLite-backed)
  - local_mode=False: HTTP client against a remote server

Config via $HERMES_HOME/tianren-anima.json.

Mapping:
  memory_add      → client.add()
  memory_search   → client.search()
  memory_remove   → client.delete()
  memory_profile  → client.history()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _ensure_anima_env():
    """Set env vars for anima before first import.

    Uses LM Studio bge-m3 (1024-dim) for embeddings.
    Falls back to BM25 if LM Studio is offline.
    """
    import socket
    from hermes_constants import get_hermes_home

    # Ensure anima package is importable
    anima_src = Path.home() / "dev" / "tianren-anima" / "src"
    if anima_src.exists() and str(anima_src) not in sys.path:
        sys.path.insert(0, str(anima_src))

    # DB path — must be absolute (sqlite:/// relative = CWD-relative)
    if "OM_DB_URL" not in os.environ:
        db_path = get_hermes_home() / "openmemory.db"
        os.environ["OM_DB_URL"] = f"sqlite:///{db_path}"

    # Detect LM Studio availability
    lm_online = False
    try:
        with socket.create_connection(("localhost", 11434), timeout=2):
            lm_online = True
    except (OSError, socket.timeout):
        pass

    if lm_online:
        os.environ["OM_EMBED_KIND"] = "openai"
        os.environ["OM_VEC_DIM"] = "1024"
        os.environ["OM_OPENAI_BASE_URL"] = "http://localhost:11434/v1"
        os.environ["OM_OPENAI_MODEL"] = "text-embedding-bge-m3"
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "lm-studio"
        logger.info("tianren-anima: LM Studio bge-m3 active (1024-dim)")
    else:
        os.environ["OM_EMBED_KIND"] = "synthetic"
        os.environ["OM_VEC_DIM"] = "1024"
        logger.warning("tianren-anima: LM Studio offline, using BM25 fallback")

    # Patch the already-initialized env singleton
    try:
        from anima.core.config import env as _env
        _env.emb_kind = os.environ["OM_EMBED_KIND"]
        _env.vec_dim = 1024
        if lm_online:
            _env.openai_base_url = "http://localhost:11434/v1"
            _env.openai_model = "text-embedding-bge-m3"
    except Exception:
        pass


def _load_config() -> dict:
    from hermes_constants import get_hermes_home

    config = {
        "local_mode": True,
        "remote_url": os.environ.get("ANIMA_URL", "http://localhost:8000"),
        "user_id": os.environ.get("ANIMA_USER_ID", "hermes-user"),
    }

    config_path = get_hermes_home() / "tianren-anima.json"
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

ADD_SCHEMA = {
    "name": "memory_add",
    "description": (
        "Store an explicit memory for future recall. "
        "Use for preferences, decisions, facts, and project context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "nature": {
                "type": "string",
                "enum": ["egoistic", "altruistic", "hybrid"],
                "description": (
                    "天人合一记忆天性。egoistic=利己(个人偏好,速忘,隔离), "
                    "altruistic=利他(共享知识,慢忘,跨用户可见), hybrid=混合。"
                ),
            },
        },
        "required": ["content"],
    },
}

SEARCH_SCHEMA = {
    "name": "memory_search",
    "description": "Search memories by semantic similarity. Returns relevant facts ranked by relevance.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "nature": {
                "type": "string",
                "enum": ["egoistic", "altruistic", "hybrid"],
                "description": "Filter by memory nature. altruistic=搜索所有用户的利他知识池。",
            },
            "tian_ren_ratio": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": (
                    "天人比 [0,1]。0=纯利己排序, 0.5=平衡, 1=纯利他排序。"
                    "利他记忆在 ratio>0.5 时排序权重提升。"
                ),
            },
        },
        "required": ["query"],
    },
}

REMOVE_SCHEMA = {
    "name": "memory_remove",
    "description": (
        "Delete a specific memory by its ID, or delete all memories for the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "The ID of the memory to delete."},
            "all": {"type": "boolean", "description": "Set to true to delete all memories for the user."},
        },
    },
}

PROFILE_SCHEMA = {
    "name": "memory_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Use at conversation start for full context."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ---------------------------------------------------------------------------
# Async bridge
# ---------------------------------------------------------------------------

def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=30)


# ---------------------------------------------------------------------------
# Remote HTTP client
# ---------------------------------------------------------------------------

class _RemoteClient:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _request(self, method: str, path: str, json_data: dict | None = None) -> dict:
        import urllib.request, urllib.error
        url = f"{self._base_url}{path}"
        data = json.dumps(json_data).encode("utf-8") if json_data else None
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Anima HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Anima connection error: {e.reason}") from e

    def add(self, content: str, user_id: str | None = None, **kwargs) -> Dict[str, Any]:
        payload = {"content": content}
        if user_id: payload["user_id"] = user_id
        if kwargs.get("nature"): payload["nature"] = kwargs["nature"]
        if kwargs.get("metadata"): payload["metadata"] = kwargs["metadata"]
        return self._request("POST", "/add", payload).get("data", {})

    def search(self, query: str, user_id: str | None = None, limit: int = 10, **kwargs) -> List[Dict]:
        payload = {"query": query, "limit": limit}
        if user_id: payload["user_id"] = user_id
        if kwargs: payload["filters"] = kwargs
        return self._request("POST", "/search", payload).get("results", [])

    def history(self, user_id: str | None = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        params = f"user_id={user_id or ''}&limit={limit}&offset={offset}"
        return self._request("GET", f"/history?{params}").get("history", [])

    def delete(self, memory_id: str) -> None:
        raise NotImplementedError("Remote delete not available. Use local mode.")

    def delete_all(self, user_id: str | None = None) -> None:
        raise NotImplementedError("Remote delete_all not available. Use local mode.")


# ---------------------------------------------------------------------------
# Nature Classifier (rule-based, fast)
# ---------------------------------------------------------------------------

def _classify_nature(user_text: str, assistant_text: str) -> str:
    """Classify memory nature from content signals."""
    combined = (user_text + " " + assistant_text).lower()

    altruistic_signals = [
        "代码", "实现", "架构", "算法", "bug", "fix", "commit",
        "rust", "python", "cargo", "npm", "git",
        "研究", "分析", "对比", "benchmark", "论文",
        "技术", "系统", "设计", "方案", "优化",
        "code", "implement", "architect", "algorithm", "research",
    ]
    egoistic_signals = [
        "我喜欢", "我不喜欢", "偏好", "感觉", "心情",
        "帮我", "给我", "我要", "i prefer", "i like",
        "密码", "token", "key", "secret",
    ]

    a_count = sum(1 for s in altruistic_signals if s in combined)
    e_count = sum(1 for s in egoistic_signals if s in combined)

    if a_count >= 3:
        return "altruistic"
    elif e_count >= 2:
        return "egoistic"
    return "hybrid"


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class AnimaProvider(MemoryProvider):
    """天人·Anima provider with local/remote modes."""

    def __init__(self):
        self._config = None
        self._client = None
        self._remote_client = None
        self._client_lock = threading.Lock()
        self._local_mode = True
        self._remote_url = ""
        self._user_id = "hermes-user"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "tianren-anima"

    def is_available(self) -> bool:
        try:
            _ensure_anima_env()
            __import__("anima")
            return True
        except ImportError:
            return False

    def _get_client(self):
        with self._client_lock:
            if self._client is not None:
                return self._client
            if self._remote_client is not None:
                return self._remote_client

            try:
                if self._local_mode:
                    _ensure_anima_env()
                    from anima import Memory
                    self._client = Memory(user=self._user_id, use_working_memory=True)
                    logger.info("天人·Anima initialized (user=%s, WM=ON)", self._user_id)
                else:
                    self._remote_client = _RemoteClient(self._remote_url)
                    logger.info("Anima REMOTE mode (url=%s)", self._remote_url)
                    return self._remote_client
                return self._client
            except ImportError:
                raise RuntimeError(
                    "anima package not installed. "
                    "Run: cd ~/dev/tianren-anima && pip install -e ."
                )

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Anima circuit breaker tripped (%d failures). Pausing %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._local_mode = self._config.get("local_mode", True)
        self._remote_url = self._config.get("remote_url", "")
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")

    def save_config(self, values, hermes_home):
        config_path = Path(hermes_home) / "tianren-anima.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        return [
            {"key": "local_mode", "description": "Use local SQLite (true) or remote (false)", "default": "true", "choices": ["true", "false"]},
            {"key": "remote_url", "description": "Remote server URL (when local_mode=false)", "default": "http://localhost:8000"},
            {"key": "user_id", "description": "User identifier for memory scoping", "default": "hermes-user"},
        ]

    def system_prompt_block(self) -> str:
        return (
            "# OpenMemory\n"
            f"Active. User: {self._user_id}. Mode: {'local' if self._local_mode else 'remote'}.\n"
            "Use memory_search to find memories, memory_add to store facts, "
            "memory_profile for a full overview, memory_remove to delete."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return f"## OpenMemory Recall\n{result}" if result else ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return
        user_id = self._user_id

        def _run():
            try:
                client = self._get_client()
                if self._local_mode:
                    results = _run_async(client.search(query=query, user_id=user_id, limit=5))
                else:
                    results = client.search(query=query, user_id=user_id, limit=5)
                if results:
                    lines = [f"- {r.get('content') or r.get('memory', '')}" for r in results if r.get("content") or r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Anima prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="anima-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages=None) -> None:
        if self._is_breaker_open():
            return
        user_id = self._user_id
        nature = _classify_nature(user_content, assistant_content)

        def _sync():
            try:
                client = self._get_client()
                content = f"User: {user_content}\nAssistant: {assistant_content}"
                metadata = {"session_id": session_id, "turn_type": "conversation", "auto_classified": True}
                if self._local_mode:
                    _run_async(client.add(content=content, user_id=user_id, nature=nature, metadata=metadata))
                else:
                    client.add(content=content, user_id=user_id, nature=nature)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Anima sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="anima-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [ADD_SCHEMA, SEARCH_SCHEMA, REMOVE_SCHEMA, PROFILE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({"error": "Anima temporarily unavailable (circuit breaker). Will retry."})

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        user_id = self._user_id
        is_local = self._local_mode

        # memory_add
        if tool_name == "memory_add":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            nature = args.get("nature")
            try:
                if is_local:
                    result = _run_async(client.add(content=content, user_id=user_id, nature=nature))
                else:
                    result = client.add(content=content, user_id=user_id, nature=nature)
                self._record_success()
                memory_id = result.get("id") or result.get("root_memory_id", "")
                return json.dumps({"result": "Memory stored.", "id": memory_id})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store memory: {e}")

        # memory_search
        elif tool_name == "memory_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            limit = min(int(args.get("limit", 10)), 50)
            nature = args.get("nature")
            tian_ren_ratio = args.get("tian_ren_ratio")
            try:
                if is_local:
                    results = _run_async(
                        client.search(query=query, user_id=user_id, limit=limit, nature=nature, tian_ren_ratio=tian_ren_ratio)
                    )
                else:
                    results = client.search(query=query, user_id=user_id, limit=limit, nature=nature)
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = []
                for r in results:
                    content = r.get("content") or r.get("memory", "")
                    item = {
                        "memory": content,
                        "id": r.get("id") or r.get("memory_id", ""),
                        "score": r.get("score") or r.get("similarity"),
                        "nature": r.get("nature"),
                    }
                    if "created_at" in r: item["created_at"] = r["created_at"]
                    if "type" in r: item["type"] = r["type"]
                    items.append(item)
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        # memory_remove
        elif tool_name == "memory_remove":
            memory_id = args.get("memory_id", "")
            delete_all_flag = args.get("all", False)
            if not memory_id and not delete_all_flag:
                return tool_error("Provide memory_id or all=true.")
            try:
                if is_local:
                    if delete_all_flag:
                        _run_async(client.delete_all(user_id=user_id))
                    else:
                        _run_async(client.delete(memory_id=memory_id))
                else:
                    if delete_all_flag:
                        client.delete_all(user_id=user_id)
                    else:
                        client.delete(memory_id=memory_id)
                self._record_success()
                return json.dumps({"result": f"Deleted {'all' if delete_all_flag else memory_id}."})
            except NotImplementedError as e:
                return tool_error(f"Delete not available in remote mode: {e}")
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to delete: {e}")

        # memory_profile
        elif tool_name == "memory_profile":
            try:
                if is_local:
                    from anima.memory.user_summary import gen_user_summary_async
                    memories = client.history(user_id=user_id, limit=200)
                    if not memories:
                        return json.dumps({"result": "No memories stored yet."})
                    try:
                        summary = _run_async(gen_user_summary_async(user_id))
                    except Exception:
                        summary = ""
                    lines = []
                    for m in memories:
                        content = m.get("content") or m.get("memory", "") or m.get("text", "")
                        if not content:
                            continue
                        created = m.get("created_at", "")
                        nature_tag = f" [{m.get('nature', '')}]" if m.get("nature") else ""
                        if created:
                            try:
                                ts = float(created)
                                from datetime import datetime, timezone
                                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                                lines.append(f"[{dt.strftime('%Y-%m-%d %H:%M')}]{nature_tag} {content}")
                            except Exception:
                                lines.append(f"{nature_tag} {content}")
                        else:
                            lines.append(f"{nature_tag} {content}")
                else:
                    memories = client.history(user_id=user_id, limit=200)
                    if not memories:
                        return json.dumps({"result": "No memories stored yet."})
                    summary = ""
                    lines = []
                    for m in memories:
                        content = m.get("content") or m.get("memory", "") or m.get("text", "")
                        if content:
                            nature_tag = f" [{m.get('nature', '')}]" if m.get("nature") else ""
                            lines.append(f"{nature_tag} {content}")

                self._record_success()
                result = {"memories": "\n".join(lines), "count": len(lines)}
                if summary:
                    result["summary"] = summary
                return json.dumps(result)
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        client = None
        with self._client_lock:
            client = self._client
        if client and hasattr(client, 'flush'):
            try:
                _run_async(client.flush())
                logger.info("天人·Anima: flushed WM → LTM")
            except Exception as e:
                logger.warning("天人·Anima: flush failed: %s", e)
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None
            self._remote_client = None


def register(ctx) -> None:
    """Register 天人·Anima as a memory provider plugin."""
    ctx.register_memory_provider(AnimaProvider())
