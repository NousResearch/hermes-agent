"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search with reranking, and
automatic deduplication via the Mem0 Platform API.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Config via environment variables:
  MEM0_API_KEY       — Mem0 Platform API key (required)
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)

Or via $HERMES_HOME/mem0.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

# Local modules (Pattern 1: BM25, Pattern 3: Supersession)
from .bm25_index import BM25Index, rrf_fuse
from . import supersession

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Time-Series Awareness — Ebbinghaus Decay + Domain-Aware Half-Life
# ---------------------------------------------------------------------------
# References:
#   - CortexGraph: Ebbinghaus decay with power-law use reinforcement
#   - SuperLocalMemory V3.3: Multi-factor strength + quantization downgrade
#
# Core formula (CortexGraph):
#   score(t) = (n_use)^β · e^(-λ·Δt) · s
#   where λ = ln(2) / half_life

_EBBINGHAUS_PARAMS = {
    "volatile": {  # Market data, real-time metrics
        "half_life_days": 3,
        "beta": 0.8,
        "forget_threshold": 0.10,
        "strength_default": 1.0,
    },
    "normal": {    # Projects, tools, general facts
        "half_life_days": 7,
        "beta": 0.6,
        "forget_threshold": 0.05,
        "strength_default": 1.0,
    },
    "stable": {    # User preferences, infrastructure
        "half_life_days": 30,
        "beta": 0.4,
        "forget_threshold": 0.02,
        "strength_default": 1.3,
    },
}

_TRUST_ACCELERATION_FACTOR = 2.0  # κ: low-trust memories decay faster

_DOMAIN_KEYWORDS = {
    "volatile": [
        r"非农", r"gdp", r"利差", r"实时", r"今日", r"最新",
        r"股价", r"汇率", r"利率", r"收益率", r"spread",
        r"nonfarm", r"realtime", r"yield", r"basis",
    ],
    "stable": [
        r"偏好", r"服务器", r"毕业", r"学位", r"姓名", r"生日",
        r"配置", r"密码", r"地址", r"电话", r"结婚", r"邮箱",
        r"preference", r"server", r"graduated", r"config",
    ],
}

_LIFECYCLE_STATES = {
    "active":  {"retention": 0.8,  "bit_width": 32, "tag": ""},
    "warm":    {"retention": 0.5,  "bit_width": 8,  "tag": "⏳"},
    "cold":    {"retention": 0.2,  "bit_width": 4,  "tag": "⚠️"},
    "archive": {"retention": 0.05, "bit_width": 2,  "tag": "🔴"},
}

# Contradiction detection keywords (checked BEFORE similarity threshold)
_CONTRADICTION_KEYWORDS = {
    "zh": ["搬到", "移居", "换", "改", "变", "不再", "已经不", "取消", "删除", "放弃", "停止"],
    "en": ["moved to", "changed to", "switched to", "no longer", "cancelled", "deleted", "stopped"],
}

_EXTENSION_KEYWORDS = {
    "zh": ["还有", "另外", "补充", "增加", "扩展", "也是"],
    "en": ["also", "additionally", "further", "extended", "as well"],
}


def _detect_domain(memory_text: str) -> str:
    """Detect domain type from memory content using keyword patterns."""
    if not memory_text:
        return "normal"
    text_lower = memory_text.lower()
    for kw in _DOMAIN_KEYWORDS["volatile"]:
        if re.search(kw, text_lower):
            return "volatile"
    for kw in _DOMAIN_KEYWORDS["stable"]:
        if re.search(kw, text_lower):
            return "stable"
    return "normal"


def _calculate_age_seconds(created_at: str) -> float:
    """Calculate age in seconds from an ISO-8601 timestamp."""
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return max(0.0, age)
    except Exception:
        return 0.0


def _calculate_ebbinghaus_score(
    age_seconds: float,
    n_use: int = 1,
    domain: str = "normal",
    strength: float | None = None,
    trust_weight: float = 1.0,
) -> float:
    """Ebbinghaus forgetting-curve score.

    Formula: score(t) = (n_use)^β · e^(-λ·Δt) · s
    Enhanced: λ_eff = λ · (1 + κ·(1 - trust))
    """
    params = _EBBINGHAUS_PARAMS.get(domain, _EBBINGHAUS_PARAMS["normal"])
    half_life_secs = params["half_life_days"] * 86400
    base_lambda = math.log(2) / half_life_secs
    effective_lambda = base_lambda * (1 + _TRUST_ACCELERATION_FACTOR * (1 - trust_weight))
    use_factor = math.pow(n_use, params["beta"])
    s = strength if strength is not None else params["strength_default"]
    score = use_factor * math.exp(-effective_lambda * age_seconds) * s
    return max(0.0, min(1.0, score))


def _determine_lifecycle(score: float, domain: str = "normal") -> dict:
    """Map retention score to lifecycle state + bit-width suggestion."""
    params = _EBBINGHAUS_PARAMS.get(domain, _EBBINGHAUS_PARAMS["normal"])
    if score > 0.8:
        state = "active"
    elif score > 0.5:
        state = "warm"
    elif score > 0.2:
        state = "cold"
    elif score > params["forget_threshold"]:
        state = "archive"
    else:
        state = "forgotten"
    lc = _LIFECYCLE_STATES.get(state, _LIFECYCLE_STATES["active"])
    return {
        "lifecycle_state": state,
        "retention_score": round(score, 3),
        "suggested_bit_width": lc["bit_width"],
        "freshness_tag": lc["tag"],
    }


def _add_time_aware_fields(memory: dict) -> dict:
    """Enrich a memory dict with Ebbinghaus time-series fields."""
    if not isinstance(memory, dict):
        return memory
    enhanced = dict(memory)
    created_at = memory.get("created_at", "")
    if not created_at:
        return enhanced

    age_secs = _calculate_age_seconds(created_at)
    age_days = int(age_secs / 86400)
    enhanced["age_days"] = age_days

    memory_text = memory.get("memory", memory.get("data", ""))
    domain = _detect_domain(memory_text)
    enhanced["domain"] = domain

    n_use = memory.get("access_count", memory.get("n_use", 1))
    enhanced["n_use"] = n_use

    params = _EBBINGHAUS_PARAMS.get(domain, _EBBINGHAUS_PARAMS["normal"])
    strength = memory.get("strength", params["strength_default"])
    trust_weight = 1.0
    meta = memory.get("metadata")
    if isinstance(meta, dict):
        trust_weight = meta.get("trust", 1.0)

    retention = _calculate_ebbinghaus_score(
        age_seconds=age_secs, n_use=n_use, domain=domain,
        strength=strength, trust_weight=trust_weight,
    )
    enhanced["retention_score"] = round(retention, 3)

    lc = _determine_lifecycle(retention, domain)
    enhanced["lifecycle_state"] = lc["lifecycle_state"]
    enhanced["suggested_bit_width"] = lc["suggested_bit_width"]
    enhanced["freshness_tag"] = lc["freshness_tag"]
    enhanced["eligible_for_promotion"] = n_use >= 5 and age_days <= 14

    return enhanced


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts (Jaccard)."""
    if re.search(r'[\u4e00-\u9fff]', text1 + text2):
        chars1, chars2 = set(text1), set(text2)
        if not chars1 or not chars2:
            return 0.0
        return len(chars1 & chars2) / len(chars1 | chars2)
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def _detect_contradiction_type(new_memory: str, existing_memory: str) -> str:
    """Detect contradiction type between new and existing memory.

    Returns: SUPERSEDE / EXTEND / ADD
    
    Priority: keyword signals > similarity thresholds
    But keyword SUPERSEDE only triggers when there's SOME content overlap (sim > 0.15),
    to avoid false positives on unrelated memories that happen to share a verb.
    """
    similarity = _calculate_text_similarity(new_memory, existing_memory)
    
    # Contradiction keywords — only trigger if some content overlap exists
    # Jaccard is harsh on short Chinese text, so gate at 0.10 (not 0.15)
    if similarity > 0.10:
        for keyword in _CONTRADICTION_KEYWORDS["zh"]:
            if keyword in new_memory:
                return "SUPERSEDE"
        for keyword in _CONTRADICTION_KEYWORDS["en"]:
            if keyword in new_memory.lower():
                return "SUPERSEDE"

    if similarity < 0.4:
        return "ADD"  # Not similar → different facts

    # Preference changes
    if "偏好" in new_memory and "偏好" in existing_memory:
        new_pref = re.search(r'偏好[:：]?\s*(.+)', new_memory)
        old_pref = re.search(r'偏好[:：]?\s*(.+)', existing_memory)
        if new_pref and old_pref and new_pref.group(1).strip() != old_pref.group(1).strip() and similarity > 0.7:
            return "SUPERSEDE"

    # Extension keywords
    for keyword in _EXTENSION_KEYWORDS["zh"]:
        if keyword in new_memory:
            return "EXTEND"
    for keyword in _EXTENSION_KEYWORDS["en"]:
        if keyword in new_memory.lower():
            return "EXTEND"

    # High similarity + no keywords → SUPERSEDE
    if similarity > 0.85:
        return "SUPERSEDE"

    return "ADD"


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
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
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
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 Platform memory with server-side extraction and semantic search."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        # BM25 local index (Pattern 1: dual-path search)
        self._bm25 = BM25Index()
        self._bm25_ready = False

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
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        return [
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": True, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
        ]

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from mem0 import MemoryClient
                # SDK 2.0.2+ self-hosted compat: patch Project validation
                try:
                    from .selfhost_patch import patch as patch_selfhost
                    patch_selfhost()
                except (ImportError, Exception):
                    pass  # SDK < 2.0.2 or patch failed, proceed anyway
                # Self-hosted detection: if host in config points to
                # localhost/private IP, pass it to MemoryClient.
                host = self._config.get("host", "")
                kwargs = {"api_key": self._api_key}
                if host and self._is_self_hosted(host):
                    kwargs["host"] = host
                    # SDK 2.0.2+ removed org_id/project_id from MemoryClient constructor
                    # (moved to Project class). Only pass them for SDK < 2.0.2.
                    try:
                        import inspect
                        sig = inspect.signature(MemoryClient.__init__)
                        if "org_id" in sig.parameters:
                            for key in ("org_id", "project_id"):
                                val = self._config.get(key, "")
                                if val:
                                    kwargs[key] = val
                    except Exception:
                        pass
                    logger.info("Mem0: using self-hosted API at %s", host)
                self._client = MemoryClient(**kwargs)
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    @staticmethod
    def _is_self_hosted(host: str) -> bool:
        """Check if host points to a local/self-hosted endpoint."""
        import re
        # Strip scheme for matching
        clean = re.sub(r'^https?://', '', host).rstrip('/')
        private_patterns = re.compile(
            r'^(localhost|127\.[\d.]+|0\.0\.0\.0|'
            r'10\.[\d.]+|'
            r'172\.(1[6-9]|2[0-9]|3[01])\.[\d.]+|'
            r'192\.168\.[\d.]+|'
            r'100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.[\d.]+)'
        )
        return bool(private_patterns.match(clean))

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
        host = self._config.get("host", "")
        # Self-hosted: force config user_id (e.g. "global") to keep all memories
        # in one pool. Cloud: per-user scoping via gateway-provided user_id.
        if host and self._is_self_hosted(host):
            self._user_id = self._config.get("user_id", "global")
            logger.info("Self-hosted mem0 detected: using config user_id=%s (ignoring gateway %s)",
                        self._user_id, kwargs.get("user_id"))
        else:
            # Cloud: per-user memory scoping from gateway
            self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)
        # BM25: try loading cache (non-blocking, will rebuild if stale)
        if self._bm25.is_available:
            self._bm25_ready = self._bm25.load_cache()
            if self._bm25_ready:
                logger.info("BM25 index ready from cache (%d docs)", self._bm25.doc_count)

    def _read_filters(self) -> Dict[str, Any]:
        """Filters for search/get_all — scoped to user only for cross-session recall."""
        return {"user_id": self._user_id}

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

    @staticmethod
    def _enrich_results(results: list) -> list:
        """Add Ebbinghaus freshness tags to search results."""
        enriched = []
        for r in results:
            item = dict(r)
            if isinstance(r, dict):
                created_at = r.get("created_at", "")
                if created_at:
                    age_secs = _calculate_age_seconds(created_at)
                    age_days = int(age_secs / 86400)
                    memory_text = r.get("memory", r.get("data", ""))
                    domain = _detect_domain(memory_text)
                    retention = _calculate_ebbinghaus_score(age_seconds=age_secs, domain=domain)
                    lc = _determine_lifecycle(retention, domain)
                    item["age_days"] = age_days
                    item["domain"] = domain
                    item["freshness_tag"] = lc["freshness_tag"]
                    item["lifecycle_state"] = lc["lifecycle_state"]
                    # Add human-readable freshness hint
                    if lc["freshness_tag"]:
                        item["freshness"] = f"{lc['freshness_tag']} {age_days}天前"
                    elif age_days <= 7:
                        item["freshness"] = ""
                    elif age_days <= 30:
                        item["freshness"] = f"⏳{age_days}天前"
                    elif age_days <= 90:
                        item["freshness"] = f"⚠️{age_days}天未验证"
                    else:
                        item["freshness"] = f"🔴可能过时"
            enriched.append(item)
        return enriched

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
            try:
                # --- Vector search via API ---
                vector_results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=rerank,
                    top_k=top_k * 2,  # Fetch more for RRF fusion
                ))
                self._record_success()

                # --- BM25 local search (Pattern 1) ---
                bm25_results = []
                if self._bm25.is_available:
                    # Lazy-build BM25 index if not ready
                    if not self._bm25_ready:
                        try:
                            all_memories = self._unwrap_results(
                                client.get_all(filters=self._read_filters())
                            )
                            self._bm25.build_from_memories(all_memories)
                            self._bm25_ready = True
                        except Exception as e:
                            logger.debug("BM25 index build failed: %s", e)
                    if self._bm25_ready:
                        bm25_results = self._bm25.search(query, top_k=top_k * 2)

                # --- RRF Fusion ---
                if vector_results and bm25_results:
                    # Both paths returned: fuse with RRF
                    results = rrf_fuse(vector_results, bm25_results, top_k=top_k)
                    search_mode = "hybrid"
                elif bm25_results and not vector_results:
                    # Vector failed, BM25 fallback
                    results = bm25_results[:top_k]
                    search_mode = "bm25_only"
                else:
                    results = vector_results[:top_k]
                    search_mode = "vector_only"

                if not results:
                    return json.dumps({"result": "No relevant memories found."})

                # Enrich with Ebbinghaus freshness tags
                enriched = self._enrich_results(results)
                # Annotate with supersession info (Pattern 3)
                enriched = [supersession.annotate_result(r) for r in enriched]

                items = []
                for r in enriched:
                    item = {"memory": r.get("memory", ""), "score": r.get("score", r.get("rrf_score", 0))}
                    if r.get("freshness"):
                        item["freshness"] = r["freshness"]
                    if r.get("domain") and r["domain"] != "normal":
                        item["domain"] = r["domain"]
                    if r.get("source"):
                        item["source"] = r["source"]
                    if r.get("supersession"):
                        item["supersession"] = r["supersession"]
                    items.append(item)
                return json.dumps({
                    "results": items,
                    "count": len(items),
                    "mode": search_mode,
                    "bm25_docs": self._bm25.doc_count if self._bm25_ready else 0,
                })
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                # Contradiction detection: search for similar existing memories
                similar_memories = self._unwrap_results(client.search(
                    query=conclusion,
                    filters=self._read_filters(),
                    rerank=True,
                    top_k=5,
                ))

                # Detect contradictions
                contradictions = []
                superseded_ids = []
                for existing in similar_memories:
                    existing_text = existing.get("memory", existing.get("data", ""))
                    existing_id = existing.get("id", "")
                    if not existing_text:
                        continue
                    contradiction_type = _detect_contradiction_type(conclusion, existing_text)
                    if contradiction_type in ["SUPERSEDE", "EXTEND"]:
                        contradictions.append({
                            "existing_id": existing_id,
                            "existing_memory": existing_text[:100] + "..." if len(existing_text) > 100 else existing_text,
                            "contradiction_type": contradiction_type,
                            "similarity": round(_calculate_text_similarity(conclusion, existing_text), 2),
                        })
                        if contradiction_type == "SUPERSEDE":
                            superseded_ids.append((existing_id, existing_text))

                # Store the new fact
                add_response = client.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_filters(),
                    infer=False,
                )
                self._record_success()

                # Extract new memory ID from response
                new_mem0_id = ""
                if isinstance(add_response, dict):
                    new_mem0_id = add_response.get("id", "")
                    if not new_mem0_id:
                        results = add_response.get("results", [])
                        if results and isinstance(results[0], dict):
                            new_mem0_id = results[0].get("id", "")

                # Pattern 3: Create supersession chains for SUPERSEDE contradictions
                chain_ids = []
                for old_id, old_text in superseded_ids:
                    if new_mem0_id and old_id:
                        try:
                            chain_id = supersession.create_supersession(
                                old_mem0_id=old_id,
                                old_text=old_text,
                                new_mem0_id=new_mem0_id,
                                new_text=conclusion,
                                reason="SUPERSEDE",
                            )
                            chain_ids.append(chain_id)
                        except Exception as e:
                            logger.debug("Supersession chain creation failed: %s", e)

                # Pattern 1: Add to BM25 index incrementally
                if self._bm25_ready and self._bm25.is_available:
                    self._bm25.add_document(
                        doc_id=new_mem0_id or "pending",
                        text=conclusion,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )

                # Return result with contradiction warnings
                result = {"result": "Fact stored."}
                if chain_ids:
                    result["supersession_chains"] = chain_ids
                    result["result"] = f"Fact stored. Superseded {len(chain_ids)} previous version(s)."
                elif contradictions:
                    result["warnings"] = {
                        "contradictions_detected": contradictions,
                        "note": f"Found {len(contradictions)} similar memories. "
                                f"Types: {', '.join(set(c['contradiction_type'] for c in contradictions))}. "
                                f"Manual review recommended.",
                    }
                return json.dumps(result)
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
