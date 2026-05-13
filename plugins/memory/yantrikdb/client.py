"""HTTP client for yantrikdb-server.

Thin wrapper over the YantrikDB REST API (port 7438 by default). The
plugin never manages the server process — the user runs yantrikdb-server
themselves and the client connects to it.

Config resolution matches the mem0 pattern:
  1. Environment variables (YANTRIKDB_URL / YANTRIKDB_TOKEN / YANTRIKDB_NAMESPACE /
     YANTRIKDB_TOP_K / YANTRIKDB_READ_TIMEOUT / YANTRIKDB_CONNECT_TIMEOUT /
     YANTRIKDB_RETRY_TOTAL / YANTRIKDB_MAX_TEXT_LEN)
  2. $HERMES_HOME/yantrikdb.json (overrides individual keys when present)

Errors are mapped into a small taxonomy so the provider can decide which
conditions trip the circuit breaker and which are deterministic caller
mistakes.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover — urllib3 ships with requests
    Retry = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:7438"
DEFAULT_NAMESPACE = "hermes"
DEFAULT_TOP_K = 10
DEFAULT_READ_TIMEOUT = 15.0
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_RETRY_TOTAL = 3
DEFAULT_MAX_TEXT_LEN = 25000  # matches Honcho's message cap

_USER_AGENT = "hermes-yantrikdb-plugin/0.1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class YantrikDBConfig:
    # Backend selector — "embedded" (default in v0.2.0+) or "http"
    mode: str = "embedded"
    # HTTP-only fields
    url: str = DEFAULT_URL
    token: str = ""
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    read_timeout: float = DEFAULT_READ_TIMEOUT
    retry_total: int = DEFAULT_RETRY_TOTAL
    # Embedded-only fields
    db_path: str = ""               # default $HERMES_HOME/yantrikdb-memory.db
    embedder_name: str = ""         # bundled potion-2M when empty; "potion-base-8M" / "potion-base-32M" for tier 2/3
    embedder_class: str = ""        # dotted python path "pkg.module.Class" — instantiated and passed to db.set_embedder(); for custom embedders
    embedder_model2vec: str = ""    # HF model id for built-in Model2VecEmbedder loader (v0.4.2+); dim auto-probed
    embedder_huggingface: str = ""  # HF model id for built-in SentenceTransformerEmbedder loader (v0.4.2+); dim auto-probed
    embedding_dim: int = 0          # output dim; required for embedder_name and embedder_class paths, ignored for the two auto-probe paths above (bundled potion-2M is dim=64)
    # Shared fields
    namespace: str = DEFAULT_NAMESPACE
    top_k: int = DEFAULT_TOP_K
    max_text_len: int = DEFAULT_MAX_TEXT_LEN
    auto_think_on_session_end: bool = True
    sync_user_messages: bool = True
    # v0.3.0+ skills surface — opt-in. Disabled by default so adding the
    # plugin to an existing Hermes install doesn't change the tool schema
    # the model sees. Users enable explicitly when they want the agentic
    # skill loop (define / search / outcome). When disabled, the three
    # skill schemas are hidden from get_tool_schemas() and any direct
    # call to a skill tool short-circuits with a clear error.
    skills_enabled: bool = False

    @classmethod
    def from_env(cls) -> YantrikDBConfig:
        return cls(
            mode=os.environ.get("YANTRIKDB_MODE", "embedded").strip().lower(),
            url=os.environ.get("YANTRIKDB_URL", DEFAULT_URL).rstrip("/"),
            token=os.environ.get("YANTRIKDB_TOKEN", ""),
            db_path=os.environ.get("YANTRIKDB_DB_PATH", ""),
            embedder_name=os.environ.get("YANTRIKDB_EMBEDDER", ""),
            embedder_class=os.environ.get("YANTRIKDB_EMBEDDER_CLASS", ""),
            embedder_model2vec=os.environ.get("YANTRIKDB_EMBEDDER_MODEL2VEC", ""),
            embedder_huggingface=os.environ.get("YANTRIKDB_EMBEDDER_HF", ""),
            embedding_dim=_parse_int(os.environ.get("YANTRIKDB_EMBEDDING_DIM"), 0),
            skills_enabled=_parse_bool(
                os.environ.get("YANTRIKDB_SKILLS_ENABLED"), default=False,
            ),
            namespace=os.environ.get("YANTRIKDB_NAMESPACE", DEFAULT_NAMESPACE),
            top_k=_parse_int(os.environ.get("YANTRIKDB_TOP_K"), DEFAULT_TOP_K),
            connect_timeout=_parse_float(
                os.environ.get("YANTRIKDB_CONNECT_TIMEOUT"), DEFAULT_CONNECT_TIMEOUT,
            ),
            read_timeout=_parse_float(
                os.environ.get("YANTRIKDB_READ_TIMEOUT"), DEFAULT_READ_TIMEOUT,
            ),
            retry_total=_parse_int(
                os.environ.get("YANTRIKDB_RETRY_TOTAL"), DEFAULT_RETRY_TOTAL,
            ),
            max_text_len=_parse_int(
                os.environ.get("YANTRIKDB_MAX_TEXT_LEN"), DEFAULT_MAX_TEXT_LEN,
            ),
        )

    @classmethod
    def load(cls, hermes_home: Path | None = None) -> YantrikDBConfig:
        """Load config from env, then overlay with $HERMES_HOME/yantrikdb.json.

        Mirrors the mem0 pattern — env provides defaults, the JSON file is an
        optional override for individual keys.
        """
        cfg = cls.from_env()
        path = _resolve_config_path(hermes_home)
        if path is None or not path.exists():
            return cfg
        try:
            overrides = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read %s: %s — using env config", path, e)
            return cfg
        if not isinstance(overrides, dict):
            return cfg
        int_fields = {"top_k", "retry_total", "max_text_len", "embedding_dim"}
        float_fields = {"connect_timeout", "read_timeout"}
        bool_fields = {"skills_enabled", "auto_think_on_session_end", "sync_user_messages"}
        for key, val in overrides.items():
            if val in (None, ""):
                continue
            if not hasattr(cfg, key):
                continue
            if key == "url" and isinstance(val, str):
                setattr(cfg, key, val.rstrip("/"))
            elif key in int_fields:
                setattr(cfg, key, _parse_int(val, getattr(cfg, key)))
            elif key in float_fields:
                setattr(cfg, key, _parse_float(val, getattr(cfg, key)))
            elif key in bool_fields:
                setattr(cfg, key, _parse_bool(val, default=getattr(cfg, key)))
            else:
                setattr(cfg, key, val)
        return cfg


def _parse_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _parse_bool(raw: Any, *, default: bool) -> bool:
    """Parse a bool from env var or JSON string. Accepts the usual truthy/falsy spellings.

    Truthy:  "1", "true", "yes", "on", "enabled"
    Falsy:   "0", "false", "no", "off", "disabled"
    Anything else (or None) → ``default``.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on", "enabled"):
        return True
    if s in ("0", "false", "no", "off", "disabled"):
        return False
    return default


def _parse_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _resolve_config_path(hermes_home: Path | None) -> Path | None:
    if hermes_home is not None:
        return Path(hermes_home) / "yantrikdb.json"
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return get_hermes_home() / "yantrikdb.json"
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Text hygiene
# ---------------------------------------------------------------------------

def truncate_text(text: str, max_len: int) -> str:
    """Truncate at a word boundary when possible, preserving a trailing ellipsis.

    Oversize memory bodies would otherwise fail with a 400 from the server.
    We truncate client-side with a visible marker so the agent (and anyone
    reading the recall result) can see that the body was clipped.
    """
    if max_len <= 0 or len(text) <= max_len:
        return text
    marker = " …[truncated]"
    budget = max_len - len(marker)
    if budget <= 0:
        return text[:max_len]
    cut = text.rfind(" ", 0, budget)
    if cut < int(budget * 0.8):
        cut = budget
    return text[:cut].rstrip() + marker


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class YantrikDBError(Exception):
    """Base for all yantrikdb client errors."""


class YantrikDBAuthError(YantrikDBError):
    """401 / 403 — token missing, expired, or revoked."""


class YantrikDBClientError(YantrikDBError):
    """4xx — request was rejected (bad shape, quota, not found)."""


class YantrikDBServerError(YantrikDBError):
    """5xx — server failed to process the request."""


class YantrikDBTransientError(YantrikDBError):
    """Network error, timeout, 429, or 503 — retry later."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class YantrikDBClient:
    """HTTP client for yantrikdb-server.

    - Session-pooled ``requests.Session`` for keep-alive.
    - Bounded retries on 5xx and connection blips via urllib3.
    - Exceptions mapped to the taxonomy above so callers can decide what
      trips a circuit breaker (transient/server/auth) versus what does not
      (4xx client mistakes).
    - Each request is tagged with a short ``req_id`` and logged at DEBUG
      with the operation and latency, so operators can correlate slow or
      failing calls in the Hermes log stream.
    """

    def __init__(self, config: YantrikDBConfig, session: Session | None = None):
        self.config = config
        self._session = session if session is not None else self._build_session(config)

    @staticmethod
    def _build_session(config: YantrikDBConfig) -> Session:
        s = requests.Session()
        if Retry is None:
            return s
        retry = Retry(
            total=max(0, config.retry_total),
            connect=min(config.retry_total, 2),
            read=min(config.retry_total, 2),
            backoff_factor=0.5,
            status_forcelist=(500, 502, 504),
            allowed_methods=frozenset(("GET", "POST", "DELETE")),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry, pool_connections=4, pool_maxsize=8,
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": _USER_AGENT}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.url}{path}"
        req_id = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        try:
            resp = self._session.request(
                method,
                url,
                json=body,
                headers=self._headers(),
                timeout=(self.config.connect_timeout, self.config.read_timeout),
            )
        except requests.Timeout as e:
            self._log_failure(req_id, method, path, started, f"timeout: {e}")
            raise YantrikDBTransientError(f"Timeout contacting {url}: {e}") from e
        except requests.ConnectionError as e:
            self._log_failure(req_id, method, path, started, f"conn: {e}")
            raise YantrikDBTransientError(f"Connection error: {e}") from e
        except requests.RequestException as e:
            self._log_failure(req_id, method, path, started, f"req: {e}")
            raise YantrikDBError(f"Request failed: {e}") from e

        latency_ms = (time.perf_counter() - started) * 1000
        logger.debug(
            "yantrikdb req=%s %s %s status=%s duration_ms=%.1f",
            req_id, method, path, resp.status_code, latency_ms,
        )
        return _parse_response(resp)

    @staticmethod
    def _log_failure(
        req_id: str, method: str, path: str, started: float, reason: str,
    ) -> None:
        latency_ms = (time.perf_counter() - started) * 1000
        logger.debug(
            "yantrikdb req=%s %s %s FAIL duration_ms=%.1f reason=%s",
            req_id, method, path, latency_ms, reason,
        )

    # -- Endpoints --

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/health")

    def remember(
        self,
        text: str,
        *,
        namespace: str | None = None,
        importance: float = 0.6,
        domain: str | None = None,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_text = truncate_text(text, self.config.max_text_len)
        body: dict[str, Any] = {
            "text": safe_text,
            "namespace": namespace or self.config.namespace,
            "importance": float(importance),
        }
        if domain:
            body["domain"] = domain
        if memory_type:
            body["memory_type"] = memory_type
        if metadata:
            body["metadata"] = metadata
        return self._request("POST", "/v1/remember", body)

    def recall(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = None,
        memory_type: str | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query,
            "namespace": namespace or self.config.namespace,
            "top_k": int(top_k or self.config.top_k),
        }
        if memory_type:
            body["memory_type"] = memory_type
        if domain:
            body["domain"] = domain
        return self._request("POST", "/v1/recall", body)

    def forget(self, rid: str) -> dict[str, Any]:
        return self._request("POST", "/v1/forget", {"rid": rid})

    def think(
        self,
        *,
        run_consolidation: bool = True,
        run_conflict_scan: bool = True,
        run_pattern_mining: bool = False,
        run_personality: bool = False,
        consolidation_limit: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "run_consolidation": run_consolidation,
            "run_conflict_scan": run_conflict_scan,
            "run_pattern_mining": run_pattern_mining,
            "run_personality": run_personality,
        }
        if consolidation_limit is not None:
            body["consolidation_limit"] = int(consolidation_limit)
        return self._request("POST", "/v1/think", body)

    def conflicts(self) -> dict[str, Any]:
        return self._request("GET", "/v1/conflicts")

    def resolve_conflict(
        self,
        conflict_id: str,
        *,
        strategy: str,
        winner_rid: str | None = None,
        new_text: str | None = None,
        resolution_note: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"strategy": strategy}
        if winner_rid:
            body["winner_rid"] = winner_rid
        if new_text:
            body["new_text"] = new_text
        if resolution_note:
            body["resolution_note"] = resolution_note
        return self._request(
            "POST", f"/v1/conflicts/{conflict_id}/resolve", body,
        )

    def relate(
        self,
        entity: str,
        target: str,
        relationship: str,
        *,
        weight: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "entity": entity,
            "target": target,
            "relationship": relationship,
        }
        if weight is not None:
            body["weight"] = float(weight)
        return self._request("POST", "/v1/relate", body)

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/v1/stats")

    # -- Skills (v0.3.0+) ---------------------------------------------
    #
    # The HTTP path delegates to yantrikdb-server's wrapper endpoints,
    # which handle schema validation server-side. The plugin still
    # validates client-side via embedded.validate_skill_define_args
    # before calling — defense in depth, and gives users early errors
    # without a network round-trip on simple shape mistakes.

    def skill_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        applies_to: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query,
            "top_k": int(top_k or self.config.top_k),
        }
        if applies_to:
            body["applies_to"] = applies_to
        return self._request("POST", "/v1/skills/search", body)

    def skill_define(
        self,
        skill_id: str,
        body: str,
        skill_type: str,
        applies_to: list[str],
        *,
        triggers: list[str] | None = None,
        on_conflict: str = "reject",
        version: str | None = None,
        supersedes_skill_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "skill_id": skill_id,
            "body": body,
            "skill_type": skill_type,
            "applies_to": list(applies_to),
            "on_conflict": on_conflict,
        }
        if triggers:
            payload["triggers"] = list(triggers)
        if version:
            payload["version"] = version
        if supersedes_skill_id:
            payload["supersedes_skill_id"] = supersedes_skill_id
        return self._request("POST", "/v1/skills/define", payload)

    def skill_outcome(
        self,
        skill_id: str,
        succeeded: bool,
        *,
        note: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"succeeded": bool(succeeded)}
        if note:
            payload["note"] = note
        return self._request(
            "POST", f"/v1/skills/{skill_id}/outcome", payload,
        )

    def close(self) -> None:
        try:
            self._session.close()
        except Exception as e:  # pragma: no cover
            logger.debug("Session close failed: %s", e)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(resp: Response) -> dict[str, Any]:
    status = resp.status_code
    if status in (401, 403):
        raise YantrikDBAuthError(f"{status}: {_safe_error_text(resp)}")
    if status in (429, 503):
        raise YantrikDBTransientError(f"{status}: {_safe_error_text(resp)}")
    if 400 <= status < 500:
        raise YantrikDBClientError(f"{status}: {_safe_error_text(resp)}")
    if status >= 500:
        raise YantrikDBServerError(f"{status}: {_safe_error_text(resp)}")

    if not resp.content:
        return {}
    try:
        data = resp.json()
    except ValueError:
        return {"raw": resp.text}
    return data if isinstance(data, dict) else {"data": data}


def _safe_error_text(resp: Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        data = None
    if isinstance(data, dict) and "error" in data:
        return str(data["error"])[:500]
    text = (resp.text or "").strip()
    return text[:500] if text else f"HTTP {resp.status_code}"


__all__ = [
    "DEFAULT_MAX_TEXT_LEN",
    "DEFAULT_NAMESPACE",
    "DEFAULT_RETRY_TOTAL",
    "DEFAULT_TOP_K",
    "DEFAULT_URL",
    "YantrikDBAuthError",
    "YantrikDBClient",
    "YantrikDBClientError",
    "YantrikDBConfig",
    "YantrikDBError",
    "YantrikDBServerError",
    "YantrikDBTransientError",
    "truncate_text",
]
