"""QMD memory provider — read-only retrieval against a QMD HTTP API.

QMD is a small JSON HTTP API on top of a vector index (e.g.
sqlite-vec) with optional cross-encoder reranking. This provider
sends queries to it and returns ranked snippets. Point it at a
self-hosted QMD instance via ``QMD_REMOTE_API_BASE_URL``.

This plugin connects to a remote QMD-compatible HTTP API. It does
not vendor or depend on @tobilu/qmd. The protocol is OpenClaw's
wire format (``POST /search``, ``GET /health``).

This provider is intentionally **read-only**: writes still go through
the existing built-in MEMORY.md path and the QMD ingest pipeline on
the QMD server. Hermes only queries.

Config via environment variables (preferred — token lives in
$HERMES_HOME/.env so the secret never lands in the JSON file):

  QMD_REMOTE_API_TOKEN     — bearer token for the remote API (required)
  QMD_REMOTE_API_BASE_URL  — base URL (default: http://localhost:18181)
  QMD_DEFAULT_INDEX        — default index name (default: default)
  QMD_TIMEOUT              — HTTP timeout in seconds (default: 30)

Or via $HERMES_HOME/qmd.json (non-secret keys only):

  {
    "base_url":      "http://localhost:18181",
    "default_index": "default",
    "timeout":       30,
    "prefetch_top_k": 3,
    "manual_top_k":  5,
    "snippet_max":   300
  }

Tools exposed:

  qmd_search(query, top_k=5, index=None, collection_filter=None)
      — semantic search against the QMD remote API.  Returns top hits
        with snippets, scores, source URIs, and rerank scores.

  qmd_status()
      — fetch the remote /health payload (uptime, monitored indexes,
        per-index document counts).  No params.

Lifecycle hooks implemented:

  is_available           — true iff QMD_REMOTE_API_TOKEN is set
  initialize             — loads config, primes httpx client lazily
  shutdown               — closes httpx client, joins prefetch thread
  system_prompt_block    — 3-line markdown summary
  queue_prefetch         — background thread, mem0 pattern
  prefetch               — drains the queued result
  sync_turn              — no-op (read-only)
  get_tool_schemas       — qmd_search + qmd_status
  handle_tool_call       — dispatches to _qmd_search / _qmd_status
  get_config_schema      — used by `hermes memory setup`
  save_config            — writes only $HERMES_HOME/qmd.json (never .env)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:18181"
DEFAULT_INDEX = "default"
DEFAULT_TIMEOUT = 30
PREFETCH_TOP_K = 3
MANUAL_TOP_K = 5
SNIPPET_MAX = 300
MIN_PREFETCH_QUERY_LEN = 12

# Circuit breaker — mirrors mem0's: N consecutive failures pauses calls
# for COOLDOWN seconds to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# qmd:// URI: qmd://<collection>/<relative/path>
_QMD_URI_RE = re.compile(r"^qmd://([^/]+)/(.+)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int = SNIPPET_MAX) -> str:
    """Truncate *s* to at most *n* chars, appending an ellipsis if cut."""
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)].rstrip() + "…"


def _collection_from_file(file_field: str) -> Optional[str]:
    """Extract the collection name from a ``qmd://<collection>/<path>`` URI."""
    if not file_field:
        return None
    m = _QMD_URI_RE.match(file_field)
    if m:
        return m.group(1)
    return None


def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/qmd.json overrides.

    Env vars provide defaults; qmd.json (if present) overrides individual
    keys.  ``api_token`` is intentionally never read from JSON — it must
    come from the env so secrets stay in .env.
    """
    # Lazy import to avoid circular import at module load time
    from hermes_constants import get_hermes_home

    config = {
        "api_token": os.environ.get("QMD_REMOTE_API_TOKEN", ""),
        "base_url": os.environ.get("QMD_REMOTE_API_BASE_URL", DEFAULT_BASE_URL),
        "default_index": os.environ.get("QMD_DEFAULT_INDEX", DEFAULT_INDEX),
        "timeout": int(os.environ.get("QMD_TIMEOUT", str(DEFAULT_TIMEOUT))),
        "prefetch_top_k": PREFETCH_TOP_K,
        "manual_top_k": MANUAL_TOP_K,
        "snippet_max": SNIPPET_MAX,
    }

    config_path = get_hermes_home() / "qmd.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            # Never let the JSON file inject api_token — secret stays in .env
            for k, v in file_cfg.items():
                if k == "api_token":
                    continue
                if v is None or v == "":
                    continue
                config[k] = v
        except Exception as e:
            logger.debug("Failed to read qmd.json: %s", e)

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "qmd_search",
    "description": (
        "Search a QMD memory index by meaning. Returns ranked snippets "
        "with source URIs (qmd://collection/path), vector score, and an "
        "optional rerank score. Use for cross-session recall, distilled "
        "facts, and curated long-term memory. Read-only — writes still "
        "go through MEMORY.md and the QMD ingest pipeline."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for (natural language).",
            },
            "top_k": {
                "type": "integer",
                "description": f"Max results (default: {MANUAL_TOP_K}, max: 25).",
            },
            "index": {
                "type": "string",
                "description": (
                    f"Index name to query. Defaults to '{DEFAULT_INDEX}'. "
                    "Use qmd_status to list available indexes."
                ),
            },
            "collection_filter": {
                "type": "string",
                "description": (
                    "Optional collection prefix to filter by, e.g. "
                    "'session-logs'. Matches against the qmd:// URI's "
                    "collection segment as a prefix (client-side filter)."
                ),
            },
        },
        "required": ["query"],
    },
}

STATUS_SCHEMA = {
    "name": "qmd_status",
    "description": (
        "Fetch the QMD service /health status: uptime, monitored "
        "indexes, per-index document counts and embedding dimensions, "
        "rerank URL. Use to diagnose retrieval issues."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class QMDMemoryProvider(MemoryProvider):
    """QMD remote-API memory provider (read-only retrieval)."""

    def __init__(self):
        self._config: Optional[dict] = None
        self._api_token: str = ""
        self._base_url: str = DEFAULT_BASE_URL
        self._default_index: str = DEFAULT_INDEX
        self._timeout: int = DEFAULT_TIMEOUT
        self._prefetch_top_k: int = PREFETCH_TOP_K
        self._manual_top_k: int = MANUAL_TOP_K
        self._snippet_max: int = SNIPPET_MAX

        self._client: Optional[httpx.Client] = None
        self._client_lock = threading.Lock()

        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

        # Circuit breaker state
        self._consecutive_failures: int = 0
        self._breaker_open_until: float = 0.0

    # -- Identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "qmd"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("api_token"))

    # -- Config ------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_token",
                "description": "QMD remote API bearer token",
                "secret": True,
                "required": True,
                "env_var": "QMD_REMOTE_API_TOKEN",
            },
            {
                "key": "base_url",
                "description": "QMD remote API base URL",
                "default": DEFAULT_BASE_URL,
                "env_var": "QMD_REMOTE_API_BASE_URL",
            },
            {
                "key": "default_index",
                "description": "Default index name",
                "default": DEFAULT_INDEX,
                # No fixed choices — depends on the deployment
            },
            {
                "key": "timeout",
                "description": "HTTP timeout (seconds)",
                "default": str(DEFAULT_TIMEOUT),
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to $HERMES_HOME/qmd.json.

        ``api_token`` is filtered out — it lives in .env via the wizard's
        ``secret: True`` field.  Never write secrets to qmd.json.
        """
        config_path = Path(hermes_home) / "qmd.json"
        existing: Dict[str, Any] = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

        for k, v in values.items():
            if k == "api_token":
                # Never persist the token to the JSON file
                continue
            if v is None or v == "":
                continue
            existing[k] = v

        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # -- Lifecycle ---------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        cfg = _load_config()
        self._config = cfg
        self._api_token = cfg.get("api_token", "")
        self._base_url = (cfg.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._default_index = cfg.get("default_index") or DEFAULT_INDEX
        try:
            self._timeout = int(cfg.get("timeout") or DEFAULT_TIMEOUT)
        except (TypeError, ValueError):
            self._timeout = DEFAULT_TIMEOUT
        try:
            self._prefetch_top_k = int(cfg.get("prefetch_top_k") or PREFETCH_TOP_K)
        except (TypeError, ValueError):
            self._prefetch_top_k = PREFETCH_TOP_K
        try:
            self._manual_top_k = int(cfg.get("manual_top_k") or MANUAL_TOP_K)
        except (TypeError, ValueError):
            self._manual_top_k = MANUAL_TOP_K
        try:
            self._snippet_max = int(cfg.get("snippet_max") or SNIPPET_MAX)
        except (TypeError, ValueError):
            self._snippet_max = SNIPPET_MAX

    def shutdown(self) -> None:
        # Wait briefly for any in-flight prefetch
        t = self._prefetch_thread
        if t and t.is_alive():
            try:
                t.join(timeout=3.0)
            except Exception:
                pass
        # Close the httpx client
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    # -- HTTP plumbing -----------------------------------------------------

    def _get_client(self) -> httpx.Client:
        """Return a cached httpx.Client with connection pooling.

        Lock-protected so concurrent threads (prefetch + tool call) don't
        race on creation.
        """
        with self._client_lock:
            if self._client is not None:
                return self._client
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
            return self._client

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # -- Circuit breaker ---------------------------------------------------

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "QMD circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    # -- System-prompt block ----------------------------------------------

    def system_prompt_block(self) -> str:
        return (
            "# QMD Memory\n"
            f"Active. Index: {self._default_index} (host: {self._base_url}).\n"
            "Use qmd_search to recall facts and curated memory; "
            "qmd_status to check the remote service. Read-only."
        )

    # -- Prefetch ----------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # Drain any queued background result
        t = self._prefetch_thread
        if t and t.is_alive():
            try:
                t.join(timeout=2.0)
            except Exception:
                pass
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## QMD Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._api_token:
            return
        if not query or len(query) < MIN_PREFETCH_QUERY_LEN:
            return
        if self._is_breaker_open():
            return

        # Guard against overlapping background threads (mem0 pattern)
        prev = self._prefetch_thread
        if prev and prev.is_alive():
            return

        def _run():
            try:
                hits = self._search_remote(
                    query=query,
                    top_k=self._prefetch_top_k,
                    index=self._default_index,
                    collection_filter=None,
                )
                self._record_success()
                if not hits:
                    return
                lines = []
                for h in hits:
                    title = h.get("title") or h.get("file") or h.get("docid") or ""
                    snippet = _truncate(h.get("snippet") or "", self._snippet_max)
                    src = h.get("file") or ""
                    line = f"- **{title}** ({src})\n  {snippet}" if snippet else f"- **{title}** ({src})"
                    lines.append(line)
                with self._prefetch_lock:
                    self._prefetch_result = "\n".join(lines)
            except Exception as e:
                self._record_failure()
                logger.debug("QMD prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="qmd-prefetch",
        )
        self._prefetch_thread.start()

    # -- sync_turn — no-op (read-only) ------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        return

    # -- Tools -------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, STATUS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._api_token:
            return tool_error(
                "QMD_REMOTE_API_TOKEN not set. Run `hermes memory setup qmd`."
            )
        if self._is_breaker_open():
            return tool_error(
                "QMD remote API temporarily unavailable (consecutive failures). "
                "Will retry automatically after cooldown."
            )

        if tool_name == "qmd_search":
            return self._qmd_search(args)
        if tool_name == "qmd_status":
            return self._qmd_status()
        return tool_error(f"Unknown tool: {tool_name}")

    # -- Tool implementations ---------------------------------------------

    def _qmd_search(self, args: Dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required parameter: query")
        try:
            top_k = int(args.get("top_k") or self._manual_top_k)
        except (TypeError, ValueError):
            top_k = self._manual_top_k
        top_k = max(1, min(top_k, 25))
        index = args.get("index") or self._default_index
        collection_filter = args.get("collection_filter") or None

        try:
            hits = self._search_remote(
                query=query,
                top_k=top_k,
                index=index,
                collection_filter=collection_filter,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            return tool_error(f"QMD search failed: {e}")

        if not hits:
            return json.dumps({
                "result": "No relevant QMD memories found.",
                "query": query,
                "index": index,
                "count": 0,
            })

        items = []
        for h in hits:
            items.append({
                "docid": h.get("docid"),
                "file": h.get("file"),
                "title": h.get("title"),
                "score": h.get("score"),
                "rerank_score": h.get("externalRerankScore"),
                "context": _truncate(h.get("context") or "", self._snippet_max),
                "snippet": _truncate(h.get("snippet") or "", self._snippet_max),
            })

        return json.dumps({
            "results": items,
            "count": len(items),
            "index": index,
            "query": query,
        })

    def _qmd_status(self) -> str:
        try:
            payload = self._status_remote()
            self._record_success()
        except Exception as e:
            self._record_failure()
            return tool_error(f"QMD status failed: {e}")
        return json.dumps(payload)

    # -- Remote calls ------------------------------------------------------

    def _search_remote(
        self,
        *,
        query: str,
        top_k: int,
        index: str,
        collection_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """POST /search and return the (optionally filtered) hits list."""
        client = self._get_client()
        body: Dict[str, Any] = {
            "query": query,
            "topK": int(top_k),
            "index": index,
        }
        try:
            r = client.post("/search", headers=self._auth_headers(), json=body)
        except httpx.HTTPError as e:
            raise RuntimeError(f"HTTP error: {e}") from e
        if r.status_code >= 400:
            raise RuntimeError(
                f"QMD /search returned {r.status_code}: {r.text[:200]}"
            )
        try:
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"QMD /search returned non-JSON: {e}") from e
        if not isinstance(data, dict) or not data.get("ok", True) is True and "results" not in data:
            # Some QMD responses omit "ok" — accept anything with results
            if "results" not in data:
                raise RuntimeError(f"QMD /search bad payload: {str(data)[:200]}")
        results = data.get("results") or []

        if collection_filter:
            cf = str(collection_filter).strip()
            filtered = []
            for h in results:
                col = _collection_from_file(h.get("file") or "")
                if col and (col == cf or col.startswith(cf)):
                    filtered.append(h)
            results = filtered

        return list(results)

    def _status_remote(self) -> Dict[str, Any]:
        """GET /health and return the parsed JSON dict."""
        client = self._get_client()
        try:
            r = client.get("/health", headers=self._auth_headers())
        except httpx.HTTPError as e:
            raise RuntimeError(f"HTTP error: {e}") from e
        if r.status_code >= 400:
            raise RuntimeError(
                f"QMD /health returned {r.status_code}: {r.text[:200]}"
            )
        try:
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"QMD /health returned non-JSON: {e}") from e
        return data


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register QMD as a memory provider plugin."""
    ctx.register_memory_provider(QMDMemoryProvider())
