"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search with reranking, and
automatic deduplication via the Mem0 Platform API.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Config via environment variables:
  MEM0_API_KEY       — Mem0 Platform or self-hosted API key (required)
  MEM0_BASE_URL      — Optional self-hosted REST base URL (e.g. http://127.0.0.1:8888)
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)

Or via $HERMES_HOME/mem0.json.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "base_url": os.environ.get("MEM0_BASE_URL", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "project": os.environ.get("MEM0_PROJECT", ""),
        "strict_search": os.environ.get("MEM0_STRICT_SEARCH", "false"),
        "candidate_k": os.environ.get("MEM0_CANDIDATE_K", "50"),
        "local_rerank": os.environ.get("MEM0_LOCAL_RERANK", "true"),
        "rerank": True,
        "keyword_search": False,
    }

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Set rerank=true for higher accuracy on important queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking for precision (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "project": {"type": "string", "description": "Optional project metadata filter for self-hosted shadow memory."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


# ---------------------------------------------------------------------------
# Self-hosted REST client
# ---------------------------------------------------------------------------

def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_STOPWORDS = {
    "the", "and", "or", "not", "as", "is", "are", "with", "for", "to", "of", "in", "a", "an",
    "uses", "use", "main", "then", "that", "this", "from", "into", "only", "should", "be", "by", "on", "at", "it",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_가-힣]+", text or "")
        if len(token) > 1 and token.lower() not in _STOPWORDS
    }


def _memory_text(item: Dict[str, Any]) -> str:
    return str(item.get("memory") or item.get("text") or "")


class _Mem0RestClient:
    """Small Mem0 self-hosted REST adapter with the MemoryClient-like methods we use.

    The hosted ``mem0.MemoryClient`` is still used when ``base_url`` is absent.
    Self-hosted Mem0 exposes a FastAPI REST surface instead, so this adapter keeps
    the provider dependency-light and lets Hermes talk to a local/Mac mini server.
    """

    def __init__(self, *, base_url: str, api_key: str, candidate_k: int = 50, local_rerank: bool = True):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.candidate_k = max(1, candidate_k)
        self.local_rerank = local_rerank

    def _request(self, method: str, path: str, payload: Dict[str, Any] | None = None,
                 query: Dict[str, Any] | None = None) -> Any:
        qs = ""
        if query:
            clean = {k: v for k, v in query.items() if v is not None and v != ""}
            if clean:
                qs = "?" + urllib.parse.urlencode(clean)
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            if self.api_key.startswith(("m0sk_", "m0adm")):
                headers["X-API-Key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path + qs,
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Mem0 REST {method} {path} failed: HTTP {exc.code}: {body[:500]}") from exc
        return json.loads(text) if text else None

    @staticmethod
    def _split_filters(filters: Dict[str, Any] | None) -> tuple[Dict[str, Any], Dict[str, Any]]:
        filters = filters or {}
        entity = {k: filters[k] for k in ("user_id", "agent_id", "run_id") if filters.get(k) is not None}
        metadata = {k: v for k, v in filters.items() if k not in entity}
        return entity, metadata

    @staticmethod
    @staticmethod
    def _sort_distance_results(response: Any, *, limit: int | None = None, query: str = "", local_rerank: bool = False) -> Any:
        """Self-hosted Mem0 currently returns distance-like scores.

        The REST server response contains lower-is-better scores but may return
        them in descending order. Normalize only inside the self-host adapter so
        hosted ``MemoryClient`` behavior is untouched.
        """
        if isinstance(response, dict) and isinstance(response.get("results"), list):
            results = response["results"]
            if all(isinstance(item, dict) and isinstance(item.get("score"), (int, float)) for item in results):
                response = dict(response)
                response["results"] = _Mem0RestClient._rank_results(results, query=query, local_rerank=local_rerank)
                if limit is not None:
                    response["results"] = response["results"][:limit]
        elif isinstance(response, list):
            if all(isinstance(item, dict) and isinstance(item.get("score"), (int, float)) for item in response):
                response = _Mem0RestClient._rank_results(response, query=query, local_rerank=local_rerank)
                if limit is not None:
                    response = response[:limit]
        return response

    @staticmethod
    def _rank_results(results: list[Dict[str, Any]], *, query: str, local_rerank: bool) -> list[Dict[str, Any]]:
        distance_sorted = sorted(results, key=lambda item: item["score"])
        if not local_rerank:
            return distance_sorted
        query_tokens = _tokens(query)
        if not query_tokens:
            return distance_sorted

        def score(item_with_rank: tuple[int, Dict[str, Any]]) -> tuple[float, float]:
            rank, item = item_with_rank
            text = _memory_text(item)
            header = "\n".join(text.splitlines()[:8])
            overlap = len(query_tokens & _tokens(text))
            header_overlap = len(query_tokens & _tokens(header))
            distance = float(item.get("score") or 0)
            # Higher lexical score is better; lower distance/rank break ties.
            lexical = 12 * header_overlap + 4 * overlap - 0.01 * rank
            return (-lexical, distance)

        return [item for _rank, item in sorted(enumerate(distance_sorted, 1), key=score)]

    def search(self, **kwargs) -> Any:
        filters = kwargs.pop("filters", None) or {}
        requested_top_k = kwargs.get("top_k")
        try:
            requested_limit = int(requested_top_k) if requested_top_k is not None else None
        except (TypeError, ValueError):
            requested_limit = None

        # The current self-host REST server advertises top-level user_id/agent_id
        # in OpenAPI, but the observed implementation returns 502 for that path
        # and scopes correctly through the flat filters object.
        payload = {"query": kwargs.pop("query"), "filters": dict(filters)}
        for key in ("top_k", "rerank", "keyword_search"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]
        if requested_limit is not None:
            payload["top_k"] = max(requested_limit, self.candidate_k)
        return self._sort_distance_results(
            self._request("POST", "/search", payload),
            limit=requested_limit,
            query=payload["query"],
            local_rerank=self.local_rerank,
        )

    def get_all(self, **kwargs) -> Any:
        entity, _metadata = self._split_filters(kwargs.get("filters"))
        return self._request("GET", "/memories", query=entity)

    def add(self, messages, **kwargs) -> Any:
        payload = {"messages": messages}
        filters = kwargs.pop("filters", None)
        entity, metadata = self._split_filters(filters)
        payload.update(entity)
        if metadata:
            payload.setdefault("metadata", {}).update(metadata)
        for key, value in kwargs.items():
            if key == "metadata" and isinstance(value, dict):
                payload.setdefault("metadata", {}).update(value)
            elif value is not None:
                payload[key] = value
        return self._request("POST", "/memories", payload)


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 Platform memory with server-side extraction and semantic search."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._base_url = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._project = ""
        self._strict_search = False
        self._candidate_k = 50
        self._local_rerank = True
        self._rerank = True
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def get_config_schema(self):
        return [
            {"key": "api_key", "description": "Mem0 Platform or self-hosted API key", "secret": True, "required": True, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "base_url", "description": "Optional self-hosted Mem0 REST base URL", "default": "", "env_var": "MEM0_BASE_URL"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "project", "description": "Optional project metadata filter for strict self-host search", "default": "", "env_var": "MEM0_PROJECT"},
            {"key": "strict_search", "description": "For self-host: include agent/project filters and candidate expansion", "default": "false", "choices": ["true", "false"], "env_var": "MEM0_STRICT_SEARCH"},
            {"key": "candidate_k", "description": "For self-host strict recall: search candidate count before trimming", "default": "50", "env_var": "MEM0_CANDIDATE_K"},
            {"key": "local_rerank", "description": "For self-host: local lexical rerank over expanded candidates", "default": "true", "choices": ["true", "false"], "env_var": "MEM0_LOCAL_RERANK"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
        ]

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            if self._base_url:
                self._client = _Mem0RestClient(
                    base_url=self._base_url,
                    api_key=self._api_key,
                    candidate_k=self._candidate_k,
                    local_rerank=self._local_rerank,
                )
                return self._client
            try:
                from mem0 import MemoryClient
                self._client = MemoryClient(api_key=self._api_key)
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            # Cooldown expired — reset and allow a retry
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
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("api_key", "")
        self._base_url = str(self._config.get("base_url", "") or "").rstrip("/")
        # Prefer gateway-provided user_id for per-user memory scoping;
        # fall back to config/env default for CLI (single-user) sessions.
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._project = str(self._config.get("project", "") or "")
        self._strict_search = _as_bool(self._config.get("strict_search"), default=False)
        self._candidate_k = max(1, _as_int(self._config.get("candidate_k"), default=50))
        self._local_rerank = _as_bool(self._config.get("local_rerank"), default=True)
        self._rerank = self._config.get("rerank", True)

    def _read_filters(self, *, project: str = "") -> Dict[str, Any]:
        """Filters for search/get_all.

        Hosted/default behavior remains user-scoped for cross-session recall.
        Self-host strict mode narrows to agent/project metadata because Joohyun's
        shadow gates showed native broad search is too noisy without namespace and
        project constraints.
        """
        filters: Dict[str, Any] = {"user_id": self._user_id}
        if self._strict_search:
            filters["agent_id"] = self._agent_id
            selected_project = project or self._project
            if selected_project:
                filters["project"] = selected_project
        return filters

    def _write_filters(self) -> Dict[str, Any]:
        """Filters for add — scoped to user + agent for attribution."""
        return {"user_id": self._user_id, "agent_id": self._agent_id}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize Mem0 API response — v2 wraps results in {"results": [...]}."""
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Memory\n"
            f"Active. User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=self._rerank,
                    top_k=5,
                ))
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                client.add(messages, **self._write_filters())
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        # Wait for any previous sync before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                memories = self._unwrap_results(client.get_all(filters=self._read_filters()))
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            project = str(args.get("project", "") or "")
            try:
                results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(project=project),
                    rerank=rerank,
                    top_k=top_k,
                ))
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [
                    {
                        "memory": r.get("memory", ""),
                        "score": r.get("score", 0),
                        "metadata": r.get("metadata", {}),
                    }
                    for r in results
                ]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                client.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_filters(),
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
