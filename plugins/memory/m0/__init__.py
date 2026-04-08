"""seekdb M0 cloud memory — HTTP API (m0.seekdb.ai), MemoryProvider implementation."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://m0.seekdb.ai"
_DEFAULT_RECALL_LIMIT = 10
_DEFAULT_API_TIMEOUT = 5.0
_MIN_CAPTURE_LENGTH = 10
_TRIVIAL_RE = re.compile(
    r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np)\.?$",
    re.IGNORECASE,
)


def _m0_api_key() -> str:
    return (os.environ.get("M0_API_KEY") or "").strip()


def _default_m0_config() -> dict:
    return {
        "auto_recall": True,
        "auto_capture": True,
        "recall_limit": _DEFAULT_RECALL_LIMIT,
        "api_timeout": _DEFAULT_API_TIMEOUT,
        "search_rewrite": False,
        "base_url": "",
    }


def _load_m0_config(hermes_home: str) -> dict:
    cfg = _default_m0_config()
    path = Path(hermes_home) / "m0.json"
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg.update({k: v for k, v in raw.items() if v is not None})
        except Exception:
            logger.debug("Failed to parse %s", path, exc_info=True)
    try:
        cfg["recall_limit"] = max(1, min(50, int(cfg.get("recall_limit", _DEFAULT_RECALL_LIMIT))))
    except Exception:
        cfg["recall_limit"] = _DEFAULT_RECALL_LIMIT
    try:
        cfg["api_timeout"] = max(0.5, min(60.0, float(cfg.get("api_timeout", _DEFAULT_API_TIMEOUT))))
    except Exception:
        cfg["api_timeout"] = _DEFAULT_API_TIMEOUT
    cfg["auto_recall"] = _as_bool(cfg.get("auto_recall"), True)
    cfg["auto_capture"] = _as_bool(cfg.get("auto_capture"), True)
    cfg["search_rewrite"] = _as_bool(cfg.get("search_rewrite"), False)
    cfg["base_url"] = str(cfg.get("base_url") or "").strip()
    return cfg


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "y", "on"):
            return True
        if lowered in ("false", "0", "no", "n", "off"):
            return False
    return default


def _save_m0_config(values: dict, hermes_home: str) -> None:
    path = Path(hermes_home) / "m0.json"
    existing: dict = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}
    existing.update(values or {})
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_text_for_capture(text: str) -> str:
    return (text or "").strip()


def _is_trivial_message(text: str) -> bool:
    return bool(_TRIVIAL_RE.match((text or "").strip()))


def _format_prefetch_memories(memories: List[dict], max_results: int) -> str:
    if not memories:
        return ""
    lines = []
    for item in memories[:max_results]:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        mid = item.get("id")
        score = item.get("score")
        prefix_bits = []
        if mid is not None:
            prefix_bits.append(f"[id={mid}]")
        if score is not None:
            try:
                prefix_bits.append(f"[score={float(score):.3f}]")
            except (TypeError, ValueError):
                pass
        prefix = " ".join(prefix_bits)
        lines.append(f"- {prefix} {content}".strip() if prefix else f"- {content}")
    if not lines:
        return ""
    intro = (
        "The following is background context from long-term memory (seekdb M0). "
        "Use it silently when relevant. Do not force memories into the conversation."
    )
    body = "\n".join(lines)
    return f"<m0-context>\n{intro}\n\n{body}\n</m0-context>"


def _normalize_memory(obj: Any) -> dict:
    if not isinstance(obj, dict):
        return {}
    return {
        "id": obj.get("id"),
        "content": obj.get("content") or "",
        "metadata": obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {},
        "score": obj.get("score"),
        "created_at": obj.get("created_at") or "",
        "updated_at": obj.get("updated_at") or "",
    }


def _parse_memory_list_payload(data: Any) -> List[dict]:
    if isinstance(data, list):
        return [_normalize_memory(x) for x in data]
    if isinstance(data, dict):
        for key in ("memories", "items", "results", "data"):
            inner = data.get(key)
            if isinstance(inner, list):
                return [_normalize_memory(x) for x in inner]
    return []


class _M0Client:
    def __init__(self, api_key: str, base_url: str, timeout: float) -> None:
        self._api_key = api_key
        self._base = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        *,
        expect_json: bool = True,
    ) -> Any:
        url = f"{self._base}{path}"
        payload: Optional[bytes] = None
        headers = {
            "X-API-Key": self._api_key,
            "Accept": "application/json",
        }
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                if not raw.strip():
                    return {} if expect_json else raw
                if not expect_json:
                    return raw
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                err_body = str(e.reason or "")
            raise RuntimeError(f"m0 HTTP {e.code}: {err_body or e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"m0 connection error: {e.reason}") from e

    def health(self) -> dict:
        url = f"{self._base}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=min(self._timeout, 5.0)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}

    def instance_status(self) -> dict:
        ak = urllib.parse.quote(self._api_key, safe="")
        return self._request("GET", f"/api/instances/{ak}/status")

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        rewrite: bool = False,
        context: Optional[List[dict]] = None,
    ) -> dict:
        body: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "rewrite": rewrite,
            "context": context or [],
        }
        return self._request("POST", "/api/memories/search", body)

    def capture(self, messages: List[dict]) -> dict:
        return self._request("POST", "/api/memories/capture", {"messages": messages})

    def store(self, content: str, metadata: Optional[dict] = None) -> dict:
        body: dict[str, Any] = {"content": content.strip(), "metadata": metadata or {}}
        return self._request("POST", "/api/memories/", body)

    def list_memories(self) -> List[dict]:
        data = self._request("GET", "/api/memories/")
        return _parse_memory_list_payload(data)

    def get_memory(self, memory_id: int) -> dict:
        return self._request("GET", f"/api/memories/{int(memory_id)}")

    def update_memory(self, memory_id: int, content: str, metadata: Optional[dict] = None) -> dict:
        body: dict[str, Any] = {"content": content.strip(), "metadata": metadata or {}}
        return self._request("PUT", f"/api/memories/{int(memory_id)}", body)

    def delete_memory(self, memory_id: int) -> None:
        self._request("DELETE", f"/api/memories/{int(memory_id)}")


# Schemas --------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "m0_search",
    "description": "Search seekdb M0 long-term memory by semantic similarity.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language search query."},
            "limit": {"type": "integer", "description": "Max results (1–50)."},
            "rewrite": {
                "type": "boolean",
                "description": "Ask server to rewrite query for better recall (slower).",
            },
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "m0_store",
    "description": "Store an explicit memory in M0 (direct write, no conversation capture pipeline).",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory text to store."},
            "metadata": {"type": "object", "description": "Optional JSON metadata."},
        },
        "required": ["content"],
    },
}

LIST_SCHEMA = {
    "name": "m0_list",
    "description": "List memories from M0 (paged list from the API).",
    "parameters": {"type": "object", "properties": {}},
}

GET_SCHEMA = {
    "name": "m0_get",
    "description": "Fetch a single M0 memory by numeric id.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "Memory id returned by search/list/store."},
        },
        "required": ["id"],
    },
}

UPDATE_SCHEMA = {
    "name": "m0_update",
    "description": "Update an existing M0 memory by id.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "Memory id."},
            "content": {"type": "string", "description": "New memory text."},
            "metadata": {"type": "object", "description": "Optional metadata (merged/replaced per server)."},
        },
        "required": ["id", "content"],
    },
}

DELETE_SCHEMA = {
    "name": "m0_delete",
    "description": "Delete one M0 memory by numeric id.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "Memory id to delete."},
        },
        "required": ["id"],
    },
}


class M0MemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._cfg = _default_m0_config()
        self._api_key = ""
        self._base_url = _DEFAULT_BASE_URL
        self._client: Optional[_M0Client] = None
        self._session_id = ""
        self._auto_recall = True
        self._auto_capture = True
        self._recall_limit = _DEFAULT_RECALL_LIMIT
        self._search_rewrite = False
        self._api_timeout = _DEFAULT_API_TIMEOUT
        self._hermes_home = ""
        self._write_enabled = True
        self._active = False
        self._sync_thread: Optional[threading.Thread] = None
        self._write_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "m0"

    def is_available(self) -> bool:
        return bool(_m0_api_key())

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "seekdb M0 Access Key (ak_...)",
                "secret": True,
                "required": True,
                "env_var": "M0_API_KEY",
                "url": "https://m0.seekdb.ai",
            },
            {
                "key": "base_url",
                "description": "M0 API base URL (empty = https://m0.seekdb.ai)",
                "required": False,
                "default": "",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        to_save: dict = {}
        bu = str((values or {}).get("base_url") or "").strip()
        if bu:
            to_save["base_url"] = bu
        if to_save:
            _save_m0_config(to_save, hermes_home)

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home

        self._hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._session_id = session_id
        self._cfg = _load_m0_config(self._hermes_home)
        self._api_key = _m0_api_key()
        env_base = (os.environ.get("M0_BASE_URL") or "").strip()
        self._base_url = env_base or self._cfg.get("base_url") or _DEFAULT_BASE_URL
        self._auto_recall = self._cfg["auto_recall"]
        self._auto_capture = self._cfg["auto_capture"]
        self._recall_limit = self._cfg["recall_limit"]
        self._search_rewrite = self._cfg["search_rewrite"]
        self._api_timeout = self._cfg["api_timeout"]

        agent_context = kwargs.get("agent_context", "")
        self._write_enabled = agent_context not in ("cron", "flush", "subagent")
        self._active = bool(self._api_key)
        self._client = None
        if self._active:
            try:
                self._client = _M0Client(self._api_key, self._base_url, self._api_timeout)
            except Exception:
                logger.warning("M0 client init failed", exc_info=True)
                self._active = False

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        return (
            "# seekdb M0\n"
            "Cloud memory is active. Use m0_search, m0_store, m0_list, m0_get, m0_update, m0_delete "
            "for explicit memory operations."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._auto_recall or not self._client or not (query or "").strip():
            return ""
        try:
            data = self._client.search(
                query.strip()[:2000],
                limit=self._recall_limit,
                rewrite=self._search_rewrite,
                context=[],
            )
            memories = _parse_memory_list_payload(data.get("memories", data))
            return _format_prefetch_memories(memories, self._recall_limit)
        except Exception:
            logger.debug("M0 prefetch failed", exc_info=True)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._active or not self._auto_capture or not self._write_enabled or not self._client:
            return
        clean_u = _clean_text_for_capture(user_content)
        clean_a = _clean_text_for_capture(assistant_content)
        if not clean_u or not clean_a:
            return
        if len(clean_u) < _MIN_CAPTURE_LENGTH or len(clean_a) < _MIN_CAPTURE_LENGTH:
            return
        if _is_trivial_message(clean_u):
            return
        messages = [
            {"role": "user", "content": clean_u},
            {"role": "assistant", "content": clean_a},
        ]

        def _run() -> None:
            try:
                self._client.capture(messages)  # type: ignore[union-attr]
            except Exception:
                logger.debug("M0 sync_turn capture failed", exc_info=True)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)
        self._sync_thread = threading.Thread(target=_run, daemon=True, name="m0-sync")
        self._sync_thread.start()

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._active or not self._write_enabled or not self._client:
            return
        if action != "add" or not (content or "").strip():
            return

        def _run() -> None:
            try:
                self._client.store(  # type: ignore[union-attr]
                    content.strip(),
                    metadata={"source": "hermes_memory", "target": target, "type": "explicit_memory"},
                )
            except Exception:
                logger.debug("M0 on_memory_write failed", exc_info=True)

        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=2.0)
        self._write_thread = threading.Thread(target=_run, daemon=False, name="m0-memory-write")
        self._write_thread.start()

    def shutdown(self) -> None:
        for attr_name in ("_sync_thread", "_write_thread"):
            thread = getattr(self, attr_name, None)
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
            setattr(self, attr_name, None)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            SEARCH_SCHEMA,
            STORE_SCHEMA,
            LIST_SCHEMA,
            GET_SCHEMA,
            UPDATE_SCHEMA,
            DELETE_SCHEMA,
        ]

    def _tool_search(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        try:
            limit = max(1, min(50, int(args.get("limit") or self._recall_limit)))
        except Exception:
            limit = self._recall_limit
        rewrite = _as_bool(args.get("rewrite"), self._search_rewrite)
        try:
            assert self._client is not None
            data = self._client.search(query, limit=limit, rewrite=rewrite, context=[])
            memories = _parse_memory_list_payload(data.get("memories", data))
            out = []
            for m in memories:
                entry = {"id": m.get("id"), "content": m.get("content", "")}
                if m.get("score") is not None:
                    entry["score"] = m.get("score")
                out.append(entry)
            resp: dict[str, Any] = {"results": out, "count": len(out)}
            rq = data.get("rewritten_queries")
            if rq:
                resp["rewritten_queries"] = rq
            return json.dumps(resp)
        except Exception as exc:
            return tool_error(f"M0 search failed: {exc}")

    def _tool_store(self, args: dict) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        md = args.get("metadata") or {}
        if not isinstance(md, dict):
            md = {}
        md.setdefault("source", "hermes_tool")
        try:
            assert self._client is not None
            result = self._client.store(content, metadata=md)
            return json.dumps({"saved": True, "response": result})
        except Exception as exc:
            return tool_error(f"M0 store failed: {exc}")

    def _tool_list(self, _args: dict) -> str:
        try:
            assert self._client is not None
            items = self._client.list_memories()
            slim = [{"id": x.get("id"), "content": (x.get("content") or "")[:200]} for x in items[:100]]
            return json.dumps({"memories": slim, "count": len(slim)})
        except Exception as exc:
            return tool_error(f"M0 list failed: {exc}")

    def _tool_get(self, args: dict) -> str:
        try:
            mid = int(args.get("id"))
        except Exception:
            return tool_error("id must be an integer")
        try:
            assert self._client is not None
            raw = self._client.get_memory(mid)
            if isinstance(raw, dict):
                return json.dumps(_normalize_memory(raw))
            return json.dumps({"raw": raw})
        except Exception as exc:
            return tool_error(f"M0 get failed: {exc}")

    def _tool_update(self, args: dict) -> str:
        try:
            mid = int(args.get("id"))
        except Exception:
            return tool_error("id must be an integer")
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        md = args.get("metadata")
        if md is not None and not isinstance(md, dict):
            return tool_error("metadata must be an object")
        try:
            assert self._client is not None
            result = self._client.update_memory(mid, content, metadata=md)
            return json.dumps({"updated": True, "response": result})
        except Exception as exc:
            return tool_error(f"M0 update failed: {exc}")

    def _tool_delete(self, args: dict) -> str:
        try:
            mid = int(args.get("id"))
        except Exception:
            return tool_error("id must be an integer")
        try:
            assert self._client is not None
            self._client.delete_memory(mid)
            return json.dumps({"deleted": True, "id": mid})
        except Exception as exc:
            return tool_error(f"M0 delete failed: {exc}")

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._active or not self._client:
            return tool_error("seekdb M0 is not configured (set M0_API_KEY)")
        if tool_name == "m0_search":
            return self._tool_search(args)
        if tool_name == "m0_store":
            return self._tool_store(args)
        if tool_name == "m0_list":
            return self._tool_list(args)
        if tool_name == "m0_get":
            return self._tool_get(args)
        if tool_name == "m0_update":
            return self._tool_update(args)
        if tool_name == "m0_delete":
            return self._tool_delete(args)
        return tool_error(f"Unknown tool: {tool_name}")


def register(ctx: Any) -> None:
    ctx.register_memory_provider(M0MemoryProvider())
