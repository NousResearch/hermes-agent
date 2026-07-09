"""JiuwenMemory memory plugin using the official memory-server API."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_SCOPE_ID = "hermes"
_DEFAULT_THRESHOLD = 0.3
_DEFAULT_TIMEOUT = 5.0
_DEFAULT_MAX_RECALL_RESULTS = 8
_DEFAULT_MAX_SUMMARY_RESULTS = 4
_DEFAULT_FALLBACK_PAGE_SIZE = 50
_DEFAULT_FALLBACK_MAX_PAGES = 4
_STATUS_PROBE_QUERY = "Hermes JiuwenMemory status probe"

_ADD_MESSAGES_PATH = "/add_messages/"
_SEARCH_MEMORY_PATH = "/search_memory/"
_SEARCH_SUMMARY_PATH = "/search_user_history_summary/"
_GET_USER_MEM_BY_PAGE_PATH = "/get_user_mem_by_page/"
_HEALTH_PATH = "/health"
_FALLBACK_FILE_NAME = "jiuwenmemory_fallback.jsonl"
_FALLBACK_MEMORY_TYPES = ("semantic_memory", "episodic_memory", "user_profile", "summary")

_ENABLE_KEYS = (
    "enable_long_term_mem",
    "enable_user_profile",
    "enable_semantic_memory",
    "enable_episodic_memory",
    "enable_summary_memory",
)


def _default_config() -> dict:
    config = {
        "base_url": _DEFAULT_BASE_URL,
        "scope_id": _DEFAULT_SCOPE_ID,
        "user_id": "",
        "threshold": _DEFAULT_THRESHOLD,
        "timeout": _DEFAULT_TIMEOUT,
        "max_recall_results": _DEFAULT_MAX_RECALL_RESULTS,
        "max_summary_results": _DEFAULT_MAX_SUMMARY_RESULTS,
        "auto_recall": True,
        "auto_capture": True,
        "local_fallback": True,
        "fallback_page_size": _DEFAULT_FALLBACK_PAGE_SIZE,
        "fallback_max_pages": _DEFAULT_FALLBACK_MAX_PAGES,
    }
    config.update({key: True for key in _ENABLE_KEYS})
    return config


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "/"
    return text if text.startswith("/") else f"/{text}"


def _coerce_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except Exception:
        return default


def _coerce_float(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        return max(minimum, min(maximum, float(value)))
    except Exception:
        return default


def _clean_optional_str(value: Any) -> str:
    return str(value or "").strip()


def _load_config(hermes_home: str) -> dict:
    config = _default_config()
    path = Path(hermes_home) / "jiuwenmemory.json"
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                if "scope_id" not in raw and raw.get("collection"):
                    raw = dict(raw)
                    raw["scope_id"] = raw["collection"]
                for key in (
                    "base_url",
                    "scope_id",
                    "user_id",
                    "threshold",
                    "timeout",
                    "max_recall_results",
                    "max_summary_results",
                    "auto_recall",
                    "auto_capture",
                    "local_fallback",
                    "fallback_page_size",
                    "fallback_max_pages",
                    *_ENABLE_KEYS,
                ):
                    if key in raw:
                        config[key] = raw[key]
        except Exception:
            logger.debug("Failed to parse %s", path, exc_info=True)

    env_base = os.environ.get("JIUWENMEMORY_BASE_URL", "").strip()
    env_scope = os.environ.get("JIUWENMEMORY_SCOPE_ID", "").strip()
    env_user = os.environ.get("JIUWENMEMORY_USER_ID", "").strip()
    if env_base:
        config["base_url"] = env_base
    if env_scope:
        config["scope_id"] = env_scope
    if env_user:
        config["user_id"] = env_user

    config["base_url"] = str(config.get("base_url") or _DEFAULT_BASE_URL).rstrip("/")
    config["scope_id"] = _clean_optional_str(config.get("scope_id")) or _DEFAULT_SCOPE_ID
    config["user_id"] = _clean_optional_str(config.get("user_id"))
    config["threshold"] = _coerce_float(
        config.get("threshold"),
        _DEFAULT_THRESHOLD,
        minimum=0.0,
        maximum=1.0,
    )
    config["timeout"] = _coerce_float(
        config.get("timeout"),
        _DEFAULT_TIMEOUT,
        minimum=0.5,
        maximum=30.0,
    )
    config["max_recall_results"] = _coerce_int(
        config.get("max_recall_results"),
        _DEFAULT_MAX_RECALL_RESULTS,
        minimum=1,
        maximum=20,
    )
    config["max_summary_results"] = _coerce_int(
        config.get("max_summary_results"),
        _DEFAULT_MAX_SUMMARY_RESULTS,
        minimum=0,
        maximum=20,
    )
    config["auto_recall"] = _coerce_bool(config.get("auto_recall"), True)
    config["auto_capture"] = _coerce_bool(config.get("auto_capture"), True)
    config["local_fallback"] = _coerce_bool(config.get("local_fallback"), True)
    config["fallback_page_size"] = _coerce_int(
        config.get("fallback_page_size"),
        _DEFAULT_FALLBACK_PAGE_SIZE,
        minimum=1,
        maximum=200,
    )
    config["fallback_max_pages"] = _coerce_int(
        config.get("fallback_max_pages"),
        _DEFAULT_FALLBACK_MAX_PAGES,
        minimum=1,
        maximum=20,
    )
    for key in _ENABLE_KEYS:
        config[key] = _coerce_bool(config.get(key), True)
    return config


def _is_embedding_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    if "timed out" in message or "timeout" in message:
        return True
    return "embedding" in message and (
        "failed to get embedding" in message
        or "embedding_request" in message
        or "embedding request" in message
        or "retrieval embedding" in message
    )


def _extract_results(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = (
            payload.get("results")
            or payload.get("memories")
            or payload.get("items")
            or payload.get("data")
            or []
        )
    else:
        raw_items = []
    if isinstance(raw_items, dict):
        raw_items = raw_items.get("results") or raw_items.get("items") or []
    if not isinstance(raw_items, list):
        return []

    results: list[dict] = []
    for item in raw_items:
        if isinstance(item, str):
            entry: dict[str, Any] = {"content": item}
        elif isinstance(item, dict):
            mem_info = item.get("mem_info") if isinstance(item.get("mem_info"), dict) else {}
            content = (
                item.get("content")
                or item.get("memory")
                or item.get("text")
                or item.get("body")
                or mem_info.get("content")
                or ""
            )
            entry = dict(item)
            entry["content"] = str(content)
        else:
            continue
        if entry.get("content"):
            results.append(entry)
    return results


def _format_item(item: dict) -> str:
    content = str(item.get("content") or "").strip()
    if not content:
        return ""
    score = item.get("score") or item.get("similarity")
    prefix = ""
    if score is not None:
        try:
            prefix = f"[{round(float(score) * 100)}%] "
        except Exception:
            prefix = ""
    return f"- {prefix}{content[:500]}"


def _format_context(
    memories: list[dict],
    summaries: list[dict],
    *,
    memory_limit: int,
    summary_limit: int,
) -> str:
    sections: list[str] = []
    memory_lines = [_format_item(item) for item in memories[:memory_limit]]
    memory_lines = [line for line in memory_lines if line]
    if memory_lines:
        sections.append("Relevant memories:\n" + "\n".join(memory_lines))

    summary_lines = [_format_item(item) for item in summaries[:summary_limit]]
    summary_lines = [line for line in summary_lines if line]
    if summary_lines:
        sections.append("Relevant conversation summaries:\n" + "\n".join(summary_lines))

    if not sections:
        return ""
    return "<jiuwenmemory-context>\n" + "\n\n".join(sections) + "\n</jiuwenmemory-context>"


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if token.strip()}


def _lexical_score(query: str, content: str) -> float:
    query_tokens = _tokenize(query)
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens)
    if overlap == 0 and query.strip().lower() not in content.lower():
        return 0.0
    score = overlap / max(1, len(query_tokens))
    if query.strip().lower() in content.lower():
        score += 0.25
    return min(score, 1.0)


def _rank_lexical(query: str, items: list[dict], limit: int) -> list[dict]:
    ranked: list[tuple[float, dict]] = []
    for item in items:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        score = _lexical_score(query, content)
        if score <= 0:
            continue
        entry = dict(item)
        entry.setdefault("source", "jiuwenmemory_fallback")
        entry["score"] = max(float(entry.get("score") or 0.0), score)
        ranked.append((score, entry))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _score, entry in ranked[:limit]]


class _LocalFallbackStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, content: str, *, source: str, metadata: Optional[dict[str, Any]] = None) -> dict:
        clean = str(content or "").strip()
        if not clean:
            return {"status": "skipped", "reason": "empty content"}
        record = {
            "content": clean[:12000],
            "source": source,
            "timestamp": time.time(),
        }
        if metadata:
            record["metadata"] = metadata
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self._ensure_parent()
        with self._lock:
            if not self.path.exists():
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(line)
            else:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        return {"status": "fallback_stored", "path": str(self.path)}

    def records(self, *, max_records: int = 1000) -> list[dict]:
        if not self.path.exists():
            return []
        items: list[dict] = []
        with self._lock:
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except Exception:
                logger.debug("Failed to read JiuwenMemory fallback store", exc_info=True)
                return []
        for line in lines[-max_records:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("content"):
                items.append(item)
        return items

    def search(self, query: str, limit: int) -> list[dict]:
        return _rank_lexical(query, self.records(), limit)

    def status(self) -> dict[str, Any]:
        records = self.records(max_records=10000)
        return {
            "enabled": True,
            "path": str(self.path),
            "records": len(records),
            "writable": os.access(self.path.parent, os.W_OK) if self.path.parent.exists() else True,
        }


class _JiuwenMemoryClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        scope_id: str,
        user_id: str,
        threshold: float,
        timeout: float,
        enable_flags: dict[str, bool],
    ):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.scope_id = scope_id
        self.user_id = user_id.strip()
        self.threshold = threshold
        self.timeout = timeout
        self.enable_flags = {key: bool(enable_flags.get(key, True)) for key in _ENABLE_KEYS}

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": "Hermes-JiuwenMemory/1.0"}
        if self.api_key:
            token = self.api_key.removeprefix("Bearer ").strip()
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _scoped_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"scope_id": self.scope_id}
        if self.user_id:
            payload["user_id"] = self.user_id
        return payload

    def request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{_normalize_path(path)}",
            data=data,
            headers=self._headers(),
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"JiuwenMemory HTTP {exc.code}: {body[:300]}") from exc
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"text": body}

    def add_messages(self, messages: list[dict[str, str]], *, session_id: str = "") -> dict:
        payload = {"messages": messages, **self._scoped_payload(), **self.enable_flags}
        if session_id:
            payload["session_id"] = session_id
        result = self.request("POST", _ADD_MESSAGES_PATH, payload)
        return result if isinstance(result, dict) else {"result": result}

    def search_memory(self, query: str, limit: int, threshold: Optional[float] = None) -> list[dict]:
        payload = {
            "query": query,
            "num": limit,
            **self._scoped_payload(),
            "threshold": self.threshold if threshold is None else threshold,
        }
        return _extract_results(self.request("POST", _SEARCH_MEMORY_PATH, payload))

    def search_summary(self, query: str, limit: int, threshold: Optional[float] = None) -> list[dict]:
        payload = {
            "query": query,
            "num": limit,
            **self._scoped_payload(),
            "threshold": self.threshold if threshold is None else threshold,
        }
        return _extract_results(self.request("POST", _SEARCH_SUMMARY_PATH, payload))

    def get_user_mem_by_page(
        self,
        *,
        page_size: int = 10,
        page_idx: int = 1,
        memory_type: str = "UNKNOWN",
    ) -> dict:
        payload = {
            **self._scoped_payload(),
            "page_size": page_size,
            "page_idx": page_idx,
            "memory_type": memory_type,
        }
        result = self.request("POST", _GET_USER_MEM_BY_PAGE_PATH, payload)
        return result if isinstance(result, dict) else {"result": result}

    def list_memories(
        self,
        *,
        page_size: int,
        max_pages: int,
        memory_types: tuple[str, ...] = _FALLBACK_MEMORY_TYPES,
    ) -> list[dict]:
        items: list[dict] = []
        seen: set[str] = set()
        for memory_type in memory_types:
            for page_idx in range(1, max_pages + 1):
                payload = self.get_user_mem_by_page(
                    page_size=page_size,
                    page_idx=page_idx,
                    memory_type=memory_type,
                )
                page_items = _extract_results(payload)
                if not page_items:
                    break
                for item in page_items:
                    key = str(item.get("mem_id") or item.get("id") or item.get("content") or "")
                    if key and key in seen:
                        continue
                    if key:
                        seen.add(key)
                    entry = dict(item)
                    entry.setdefault("source", "jiuwenmemory_page")
                    items.append(entry)
                if len(page_items) < page_size:
                    break
        return items

    def lexical_search_from_pages(self, query: str, limit: int, *, page_size: int, max_pages: int) -> list[dict]:
        return _rank_lexical(query, self.list_memories(page_size=page_size, max_pages=max_pages), limit)

    def store(self, content: str, *, session_id: str = "") -> dict:
        return self.add_messages([{"role": "user", "content": content}], session_id=session_id)

    def sync_turn(self, user: str, assistant: str, *, session_id: str = "") -> dict:
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        return self.add_messages(messages, session_id=session_id)

    def status(self) -> dict:
        result = self.request("GET", _HEALTH_PATH)
        return result if isinstance(result, dict) else {"result": result}

    def diagnostic_status(self) -> dict:
        raw_status = self.status()
        status: dict[str, Any] = dict(raw_status) if isinstance(raw_status, dict) else {"result": raw_status}

        status.setdefault("status", "unknown")
        try:
            self.search_memory(_STATUS_PROBE_QUERY, 1, threshold=0.0)
        except Exception as exc:
            status["memory_search"] = "failed"
            status["memory_search_error"] = str(exc)
            status["diagnostic"] = (
                "JiuwenMemory API health check passed, but memory search failed. "
                "Check the JiuwenMemory server embedding configuration: "
                "EMBED_MODEL_NAME, EMBED_API_KEY, and EMBED_API_BASE."
            )
            if _is_embedding_failure(exc):
                try:
                    self.get_user_mem_by_page(page_size=1, page_idx=1)
                except Exception as fallback_exc:
                    status["fallback_search"] = "failed"
                    status["fallback_search_error"] = str(fallback_exc)
                else:
                    status["fallback_search"] = "available"
                    status["diagnostic"] += (
                        " Hermes can use degraded local lexical fallback for recall, "
                        "but configure server embeddings for native vector search and writes."
                    )
        else:
            status["memory_search"] = "ok"
        return status


SEARCH_SCHEMA = {
    "name": "jiuwenmemory_search",
    "description": "Search JiuwenMemory long-term memory for relevant context.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "description": "Maximum results, 1 to 20."},
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "jiuwenmemory_store",
    "description": "Store an explicit long-term memory in JiuwenMemory.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory content to store."},
        },
        "required": ["content"],
    },
}

STATUS_SCHEMA = {
    "name": "jiuwenmemory_status",
    "description": "Check JiuwenMemory memory-server health and configured scope.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


class JiuwenMemoryProvider(MemoryProvider):
    def __init__(self):
        self._client: Optional[_JiuwenMemoryClient] = None
        self._fallback_store: Optional[_LocalFallbackStore] = None
        self._config = _default_config()
        self._session_id = ""
        self._user_id = ""
        self._sync_thread: Optional[threading.Thread] = None
        self._write_thread: Optional[threading.Thread] = None
        self._write_enabled = True

    @property
    def name(self) -> str:
        return "jiuwenmemory"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "JiuwenMemory API key (optional for local unauthenticated deployments)",
                "secret": True,
                "required": False,
                "env_var": "JIUWENMEMORY_API_KEY",
            },
            {"key": "base_url", "description": "JiuwenMemory base URL", "default": _DEFAULT_BASE_URL},
            {"key": "scope_id", "description": "JiuwenMemory scope id", "default": _DEFAULT_SCOPE_ID},
            {"key": "user_id", "description": "Optional JiuwenMemory user id", "required": False},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from utils import atomic_json_write

        path = Path(hermes_home) / "jiuwenmemory.json"
        existing = _load_config(hermes_home)
        for key in ("base_url", "scope_id", "user_id"):
            if key in values:
                existing[key] = str(values.get(key) or "").strip()
        public_keys = (
            "base_url",
            "scope_id",
            "user_id",
            "threshold",
            "max_recall_results",
            "max_summary_results",
            "auto_recall",
            "auto_capture",
            "local_fallback",
            "fallback_page_size",
            "fallback_max_pages",
            *_ENABLE_KEYS,
        )
        atomic_json_write(
            path,
            {key: existing[key] for key in public_keys},
            mode=0o600,
            sort_keys=True,
        )

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home

        hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._config = _load_config(hermes_home)
        self._session_id = session_id
        self._user_id = self._config.get("user_id") or str(kwargs.get("user_id") or "").strip()
        self._write_enabled = kwargs.get("agent_context", "") not in {"cron", "flush", "subagent"}
        self._fallback_store = (
            _LocalFallbackStore(Path(hermes_home) / _FALLBACK_FILE_NAME)
            if self._config.get("local_fallback")
            else None
        )
        self._client = _JiuwenMemoryClient(
            os.environ.get("JIUWENMEMORY_API_KEY", ""),
            self._config["base_url"],
            self._config["scope_id"],
            self._user_id,
            self._config["threshold"],
            self._config["timeout"],
            {key: self._config[key] for key in _ENABLE_KEYS},
        )

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""
        return (
            "# JiuwenMemory\n"
            f"Active. Scope: {self._client.scope_id}.\n"
            "Use jiuwenmemory_search to recall memories, jiuwenmemory_store to save durable facts, "
            "and jiuwenmemory_status to check connectivity."
        )

    def _fallback_search(self, query: str, limit: int) -> list[dict]:
        results: list[dict] = []
        if self._client:
            try:
                results.extend(
                    self._client.lexical_search_from_pages(
                        query,
                        limit,
                        page_size=self._config["fallback_page_size"],
                        max_pages=self._config["fallback_max_pages"],
                    )
                )
            except Exception:
                logger.debug("JiuwenMemory page fallback failed", exc_info=True)
        if self._fallback_store:
            results.extend(self._fallback_store.search(query, limit))
        if not results:
            return []
        deduped: list[dict] = []
        seen: set[str] = set()
        for item in sorted(results, key=lambda entry: float(entry.get("score") or 0.0), reverse=True):
            key = str(item.get("mem_id") or item.get("id") or item.get("content") or "")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _search_memory(self, query: str, limit: int) -> tuple[list[dict], dict[str, Any]]:
        if not self._client:
            return [], {"fallback": False}
        try:
            return self._client.search_memory(query, limit), {"fallback": False}
        except Exception as exc:
            if not _is_embedding_failure(exc):
                raise
            results = self._fallback_search(query, limit)
            return results, {
                "fallback": True,
                "server_search_error": str(exc),
                "warning": (
                    "JiuwenMemory native vector search failed because the server embedding backend is not working. "
                    "Returned degraded lexical fallback results; configure EMBED_MODEL_NAME, EMBED_API_KEY, "
                    "and EMBED_API_BASE on the JiuwenMemory server for full recall."
                ),
            }

    def _fallback_store_content(
        self,
        content: str,
        *,
        source: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict:
        if not self._fallback_store:
            return {"status": "fallback_disabled"}
        try:
            return self._fallback_store.append(content, source=source, metadata=metadata)
        except Exception as exc:
            logger.debug("JiuwenMemory local fallback write failed", exc_info=True)
            return {"status": "fallback_failed", "error": str(exc)}

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        del session_id
        if not self._client or not self._config.get("auto_recall") or not query.strip():
            return ""

        clean_query = query[:1000]
        memories: list[dict] = []
        summaries: list[dict] = []
        try:
            memories, _meta = self._search_memory(clean_query, self._config["max_recall_results"])
        except Exception:
            logger.debug("JiuwenMemory memory prefetch failed", exc_info=True)
        if self._config["max_summary_results"] > 0:
            try:
                summaries = self._client.search_summary(clean_query, self._config["max_summary_results"])
            except Exception:
                logger.debug("JiuwenMemory summary prefetch failed", exc_info=True)
        return _format_context(
            memories,
            summaries,
            memory_limit=self._config["max_recall_results"],
            summary_limit=self._config["max_summary_results"],
        )

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        del messages
        if not self._client or not self._write_enabled or not self._config.get("auto_capture"):
            return
        if not (user_content or "").strip() and not (assistant_content or "").strip():
            return

        client = self._client
        user = user_content[:8000]
        assistant = assistant_content[:8000]
        effective_session_id = session_id or self._session_id

        def _run() -> None:
            try:
                client.sync_turn(user, assistant, session_id=effective_session_id)
            except Exception as exc:
                if _is_embedding_failure(exc):
                    self._fallback_store_content(
                        f"User: {user}\nAssistant: {assistant}",
                        source="sync_turn_embedding_fallback",
                        metadata={"session_id": effective_session_id},
                    )
                else:
                    logger.debug("JiuwenMemory turn sync failed", exc_info=True)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)
        self._sync_thread = threading.Thread(target=_run, daemon=True, name="jiuwenmemory-sync")
        self._sync_thread.start()

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        del target, metadata
        if not self._client or not self._write_enabled or action not in {"add", "replace"} or not content.strip():
            return

        client = self._client
        memory = content.strip()
        session_id = self._session_id

        def _run() -> None:
            try:
                client.store(memory, session_id=session_id)
            except Exception as exc:
                if _is_embedding_failure(exc):
                    self._fallback_store_content(
                        memory,
                        source="memory_write_embedding_fallback",
                        metadata={"action": action, "session_id": session_id},
                    )
                else:
                    logger.debug("JiuwenMemory memory mirror failed", exc_info=True)

        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=2.0)
        self._write_thread = threading.Thread(target=_run, daemon=True, name="jiuwenmemory-write")
        self._write_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, STORE_SCHEMA, STATUS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        del kwargs
        if not self._client:
            return tool_error("JiuwenMemory is not initialized")
        try:
            if tool_name == "jiuwenmemory_search":
                query = str(args.get("query") or "").strip()
                if not query:
                    return tool_error("query is required")
                limit = max(1, min(20, int(args.get("limit", self._config["max_recall_results"]) or 8)))
                results, meta = self._search_memory(query, limit)
                return json.dumps({"results": results, **meta}, ensure_ascii=False)
            if tool_name == "jiuwenmemory_store":
                content = str(args.get("content") or "").strip()
                if not content:
                    return tool_error("content is required")
                try:
                    return json.dumps(self._client.store(content, session_id=self._session_id), ensure_ascii=False)
                except Exception as exc:
                    if not _is_embedding_failure(exc):
                        raise
                    result = self._fallback_store_content(
                        content,
                        source="store_tool_embedding_fallback",
                        metadata={"session_id": self._session_id},
                    )
                    result["server_store_error"] = str(exc)
                    result["warning"] = (
                        "JiuwenMemory server write failed because the embedding backend is not working. "
                        "Stored the memory in the local fallback journal."
                    )
                    return json.dumps(result, ensure_ascii=False)
            if tool_name == "jiuwenmemory_status":
                status = self._client.diagnostic_status()
                status.setdefault("scope_id", self._client.scope_id)
                status.setdefault("user_id", self._client.user_id or None)
                status.setdefault("base_url", self._client.base_url)
                if self._fallback_store:
                    status["local_fallback"] = self._fallback_store.status()
                return json.dumps(status, ensure_ascii=False)
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as exc:
            return tool_error(str(exc))

    def shutdown(self) -> None:
        for attr in ("_sync_thread", "_write_thread"):
            thread = getattr(self, attr, None)
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
            setattr(self, attr, None)

    def backup_paths(self) -> list[str]:
        if self._fallback_store and self._fallback_store.path.exists():
            return [str(self._fallback_store.path)]
        return []


def register(ctx) -> None:
    ctx.register_memory_provider(JiuwenMemoryProvider())
