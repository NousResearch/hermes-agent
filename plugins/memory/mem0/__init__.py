from __future__ import annotations

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
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Normalize text for hash-based dedup: NFKC + strip punctuation + collapse whitespace + lowercase."""
    import unicodedata
    # NFKC normalization handles full-width chars, zero-width spaces, etc.
    text = unicodedata.normalize('NFKC', text)
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text.strip())).lower()


def _text_hash(text: str) -> str:
    """MD5 hash of normalized text for O(N) exact-duplicate detection."""
    return hashlib.md5(_normalize_text(text).encode('utf-8')).hexdigest()

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
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "rerank": True,
        "keyword_search": False,
        # Local mode paths — local defaults overridden by mem0.json/env vars
        "mem0_server": os.environ.get("MEM0_SERVER", "m/media/data/mem045MEM0_DIR/mem0_server.py"),
        "mem0_python": os.environ.get("MEM0_PYTHON", "m/media/data/mem045MEM0_DIR/.venv/bin/python"),
        "llm_base_url": os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"),
        "llm_model": os.environ.get("LLM_MODEL", "qwen3"),
        "embedder_model": os.environ.get("EMBEDDER_MODEL", "m/home/herocco/bge45BGE_DIR/bge-large-zh-v1.5"),
        "embedding_dims": int(os.environ.get("EMBEDDING_DIMS", "1024")),
        "qdrant_host": os.environ.get("QDRANT_HOST", "localhost"),
        "qdrant_port": int(os.environ.get("QDRANT_PORT", "6333")),
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

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        # Support both platform mode (api_key) and local mode
        return bool(cfg.get("api_key")) or cfg.get("mode") == "local"

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
                cfg = _load_config()
                if cfg.get("mode") == "local":
                    # Local mode: use Memory.from_config()
                    # ⚡ Force offline mode — prevent transformers/hf_hub from downloading
                    # config files every time the model loads. Model files are already local.
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    from mem0 import Memory
                    self._client = Memory.from_config({
                        'llm': {
                            'provider': 'openai',
                            'config': {
                                'api_key': 'local',
                                'openai_base_url': cfg.get("llm_base_url", "http://localhost:1234/v1"),
                                'model': cfg.get("llm_model", "qwen3")
                            }
                        },
                        'embedder': {
                            'provider': 'huggingface',
                            'config': {
                                'model': cfg.get("embedder_model", "m/home/herocco/bge45BGE_DIR/bge-large-zh-v1.5")
                            }
                        },
                        'vector_store': {
                            'provider': 'qdrant',
                            'config': {
                                'collection_name': 'mem0',
                                'embedding_model_dims': cfg.get("embedding_dims", 1024),
                                'host': cfg.get("qdrant_host", "localhost"),
                                'port': cfg.get("qdrant_port", 6333)
                            }
                        },
                    })
                else:
                    # Platform mode: use MemoryClient with API key
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

    # -- Shared deduplication & formatting -----------------------------------

    def _advanced_dedup_and_format(self, results, top_k=5, mem0_python=None, mem0_server=None):
        """Advanced deduplication with dual thresholds, config check, and track freezing.

        Dual-level conflict detection:
          > 0.92: high-confidence conflict (⚠️) — only track winner
          0.75-0.92: possible related (🔗) — both track normally
          < 0.75: unrelated — normal injection
        Config-type conflicts (IP/port/version) in 0.75-0.92 are promoted to high-confidence.

        Returns (formatted_lines, kept_ids, shadow_ids).
        """
        from datetime import datetime, timezone
        import re
        import subprocess

        logger.info("[DEDUP] Entry: %d results, top_k=%d", len(results), top_k)

        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a < 1e-9 or norm_b < 1e-9:
                return 0.0
            return dot / (norm_a * norm_b)

        def is_exact_duplicate(text1, text2):
            clean1 = re.sub(r'[^\w\s]', '', text1).strip()
            clean2 = re.sub(r'[^\w\s]', '', text2).strip()
            return clean1 == clean2

        def freq_label(ac):
            if ac > 20:
                return "高频"
            if ac >= 5:
                return "中频"
            return "低频"

        def fmt_mem(mem, conflict_info=None):
            now = datetime.now(timezone.utc)
            days_ago = 0
            upd = mem.get("updated_at", "")
            if upd:
                try:
                    dt = datetime.fromisoformat(upd.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    days_ago = max(0, (now - dt).days)
                except Exception:
                    pass
            ac = (mem.get("metadata") or {}).get("access_count", 0)
            prefix = f"[Updated {days_ago} days ago | {now.strftime('%Y-%m-%d')} | {freq_label(ac)}]"
            if conflict_info:
                target_index, sim, level = conflict_info
                if level == 'high':
                    note = f" ⚠️ 可能与第{target_index}条冲突(相似度{sim:.2f})"
                else:
                    note = f" 🔗 可能与第{target_index}条相关(相似度{sim:.2f}，可能描述同一事物的不同状态)"
                return f"- {prefix}{note}：{mem.get('memory', '')}"
            return f"- {prefix} {mem.get('memory', '')}"

        has_vectors = any("_embedding" in r and r["_embedding"] for r in results)
        logger.info("[DEDUP] has_vectors=%s", has_vectors)

        kept = []
        shadow_ids = []
        conflicts = []  # (mem, conflict_with_mem, similarity, level)

        _EXCLUSIVE_KEYS = {
            'ip', 'host', 'hostname', 'address', 'url', 'endpoint',
            'port', 'version', 'path', 'directory', 'dir',
            'username', 'password', 'token', 'api_key', 'secret',
            'default_browser', 'theme', 'language', 'os',
            'model', 'backend', 'engine', 'database',
        }

        def _extract_entities_and_values(text):
            triples = []
            for m in re.finditer(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', text):
                ip = m.group(1)
                before = text[:m.start()]
                entity = _find_entity(before)
                if entity:
                    triples.append((entity.lower(), 'ip', ip))
            for m in re.finditer(r'\bport\s*(\d{2,5})\b', text, re.I):
                port = m.group(1)
                before = text[:m.start()]
                entity = _find_entity(before)
                if entity:
                    triples.append((entity.lower(), 'port', port))
            for m in re.finditer(r'(?:version|v)\s*[:=]?\s*(\d+\.\d+(?:\.\d+)?)', text, re.I):
                ver = m.group(1)
                before = text[:m.start()]
                entity = _find_entity(before)
                if entity:
                    triples.append((entity.lower(), 'version', ver))
            for m in re.finditer(r'(https?://\S+)', text):
                url = m.group(1).rstrip('.,')
                before = text[:m.start()]
                entity = _find_entity(before)
                if entity:
                    triples.append((entity.lower(), 'url', url))
            return triples

        def _find_entity(before_text):
            known_services = {
                'sglang', 'qdrant', 'python', 'node', 'nginx', 'redis',
                'postgres', 'mysql', 'mongodb', 'kafka', 'docker',
                'mem0', 'hermes', 'cloakbrowser', 'comfyui', 'vllm',
                'sillytavern', 'whisper', 'transformers', 'torch',
            }
            before_lower = before_text.lower()[-80:]
            for svc in known_services:
                if svc in before_lower:
                    return svc
            words = before_text[-60:].split()
            for w in reversed(words):
                clean = re.sub(r'[^\w]', '', w)
                if len(clean) >= 2 and (clean[0].isupper() or len(clean) >= 4):
                    return clean.lower()
            return None

        # _is_config_conflict returns (is_conflict, entity_type, reason)
        # entity_type: 'endpoint' | 'version' | 'path' | 'other' | ''
        # endpoint (IP/Port/URL) →天然全局唯一,直接 HIGH
        # version/path/hostname →允许多版本共存,需 cs≥0.85 才升 HIGH
        # NOTE: 多地址共存(如负载均衡+直连)是极低概率事件,当前按 HIGH 处理是安全取舍
        def _is_config_conflict(text_a, text_b):
            triples_a = _extract_entities_and_values(text_a)
            triples_b = _extract_entities_and_values(text_b)
            if triples_a and triples_b:
                map_a = {(e, t): v for e, t, v in triples_a}
                map_b = {(e, t): v for e, t, v in triples_b}
                for key in set(map_a.keys()) & set(map_b.keys()):
                    entity, vtype = key
                    if vtype in _EXCLUSIVE_KEYS and map_a[key] != map_b[key]:
                        etype = 'endpoint' if vtype in ('ip', 'host', 'hostname', 'address', 'url', 'endpoint', 'port') else vtype
                        return True, etype, f'{vtype} mismatch: {map_a[key]} vs {map_b[key]}'
            # IP only (no port) — port conflicts handled by separate port regex below
            ip_a = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text_a)
            ip_b = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text_b)
            if ip_a and ip_b and set(ip_a) != set(ip_b):
                return True, 'endpoint', f'IP mismatch'
            # Port detection: "port N", "端口N", "port has been changed to N" (allow words between)
            port_a = re.findall(r'(?:port|端口|端口号)(?:\s.*?|\s*[是:：]?\s*)(\d+)', text_a, re.I)
            port_b = re.findall(r'(?:port|端口|端口号)(?:\s.*?|\s*[是:：]?\s*)(\d+)', text_b, re.I)
            if port_a and port_b and set(port_a) != set(port_b):
                return True, 'endpoint', f'Port mismatch'
            ver_a = re.findall(r'(?:v|version\s*)?(\d+\.\d+(?:\.\d+)?)', text_a, re.I)
            ver_b = re.findall(r'(?:v|version\s*)?(\d+\.\d+(?:\.\d+)?)', text_b, re.I)
            if ver_a and ver_b and set(ver_a) != set(ver_b):
                return True, 'version', f'Version mismatch'
            url_a = re.findall(r'https?://\S+', text_a)
            url_b = re.findall(r'https?://\S+', text_b)
            if url_a and url_b and set(url_a) != set(url_b):
                return True, 'endpoint', f'URL mismatch'
            return False, '', ''

        if has_vectors:
            # --- Phase 1: Hash pre-scan (O(N) exact dedup) ---
            # Normalize text → MD5 hash. If hash collides, mark as shadow.
            # DeepSeek suggestion: if shadow is newer, backfill updated_at to retained entry.
            hash_map = {}  # hash → (mem, index_in_results)
            for idx, mem in enumerate(results):
                h = _text_hash(mem.get('memory', ''))
                mem_id = mem.get('id', '')
                if h in hash_map:
                    existing_mem, _ = hash_map[h]
                    shadow_ids.append(mem_id)
                    # Timestamp backfill: if shadow is newer, update retained entry
                    try:
                        shadow_ts = mem.get('updated_at', '')
                        existing_ts = existing_mem.get('updated_at', '')
                        if shadow_ts and existing_ts and shadow_ts > existing_ts:
                            existing_mem['updated_at'] = shadow_ts
                            logger.info("[DEDUP] Hash shadow backfilled updated_at: %s → %s",
                                       existing_ts[:19], shadow_ts[:19])
                    except Exception:
                        pass
                else:
                    hash_map[h] = (mem, idx)

            # Process only non-shadowed memories
            unique_results = [mem for mem, _ in hash_map.values()]

            for mem in unique_results:
                mem_id = mem.get('id', '')
                mem_text = mem.get('memory', '')
                mem_vec = mem.get('_embedding')

                is_duplicate = False
                conflict_with = None
                max_sim = 0.0
                conflict_level = None

                for kept_mem in kept:
                    kvec = kept_mem.get('_embedding')
                    if kvec and mem_vec:
                        cos_sim = cosine_similarity(mem_vec, kvec)
                    else:
                        continue

                    logger.info("SIM: %.3f vs thresholds (0.92/0.75)", cos_sim)

                    # Tier 1: cos_sim > 0.92 → high-confidence conflict
                    if cos_sim > 0.92 and cos_sim > max_sim:
                        # Guard: if entity values are identical (e.g. same version), don't upgrade to HIGH
                        # Strip IPs first to avoid matching IP fragments as version numbers
                        clean_a = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b', '', mem_text)
                        clean_b = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b', '', kept_mem.get('memory', ''))
                        ver_a = re.findall(r'(\d+\.\d+(?:\.\d+)?)', clean_a, re.I)
                        ver_b = re.findall(r'(\d+\.\d+(?:\.\d+)?)', clean_b, re.I)
                        ver_same = bool(ver_a and ver_b and set(ver_a) & set(ver_b))
                        if ver_same:
                            conflict_level = 'medium'
                        else:
                            conflict_level = 'high'
                        conflict_with = kept_mem
                        max_sim = cos_sim
                    # Tier 2: cos_sim > 0.75 → check config conflict for promotion
                    elif cos_sim > 0.75 and cos_sim > max_sim:
                        is_cc, entity_type, reason = _is_config_conflict(mem_text, kept_mem.get('memory', ''))
                        logger.info("Conflict check: sim=%.3f, entity_type=%s, reason=%s",
                                   cos_sim, entity_type, reason)
                        if is_cc:
                            # Entity type tiering: endpoint → always HIGH; version/path → needs cs≥0.85
                            if entity_type == 'endpoint':
                                conflict_level = 'high'
                            elif entity_type in ('version', 'path', 'hostname'):
                                if cos_sim >= 0.85:
                                    conflict_level = 'high'
                                else:
                                    conflict_level = 'medium'
                            else:
                                conflict_level = 'medium'
                        else:
                            conflict_level = 'medium'

                        if conflict_level == 'high' or (cos_sim > max_sim and conflict_with is None):
                            conflict_with = kept_mem
                            max_sim = cos_sim

                if not is_duplicate:
                    kept.append(mem)
                    if conflict_with is not None:
                        logger.info("[DEDUP] Conflict: '%s' vs '%s' sim=%.3f level=%s",
                                   mem_text[:30], conflict_with.get('memory', '')[:30],
                                   max_sim, conflict_level)
                        conflicts.append((mem, conflict_with, max_sim, conflict_level))
        else:
            seen_texts = set()
            for mem in results:
                mid = mem.get("id", "")
                mtext = mem.get("memory", "")
                clean = re.sub(r'[^\w\s]', '', mtext).strip().lower()
                if clean in seen_texts:
                    shadow_ids.append(mid)
                else:
                    seen_texts.add(clean)
                    kept.append(mem)

        # Track winners / freeze losers
        if mem0_python and mem0_server:
            logger.info("[DEDUP] Track/freeze mode enabled")
            def _decide_winner(mem_a, mem_b):
                def _parse_dt(s):
                    try:
                        dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt
                    except Exception:
                        return datetime.min.replace(tzinfo=timezone.utc)
                dt_a = _parse_dt(mem_a.get('updated_at', ''))
                dt_b = _parse_dt(mem_b.get('updated_at', ''))
                if dt_a > dt_b:
                    return mem_a
                elif dt_b > dt_a:
                    return mem_b
                else:
                    ac_a = (mem_a.get('metadata') or {}).get('access_count', 0)
                    ac_b = (mem_b.get('metadata') or {}).get('access_count', 0)
                    return mem_a if ac_a >= ac_b else mem_b

            def _in_grace_period(mem):
                try:
                    created = datetime.fromisoformat(mem.get('created_at', '').replace('Z', '+00:00'))
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    return (datetime.now(timezone.utc) - created).days < 14
                except Exception:
                    return False

            loser_ids = set()
            for c_mem, c_with, c_sim, c_level in conflicts:
                logger.info("[DEDUP] Processing conflict: level=%s sim=%.3f", c_level, c_sim)
                if c_level != 'high':
                    logger.info("[DEDUP] -> Skipping (medium)")
                    continue
                if _in_grace_period(c_mem) or _in_grace_period(c_with):
                    logger.info("[DEDUP] -> Skipping (grace period)")
                    continue
                winner = _decide_winner(c_mem, c_with)
                loser_id = c_with.get('id') if winner is c_mem else c_mem.get('id')
                loser_ids.add(loser_id)
                logger.info("[DEDUP] -> Loser: %s", loser_id)

            track_ids = [m.get('id', '') for m in kept if m.get('id') not in loser_ids]
            if track_ids:
                try:
                    subprocess.run(
                        [mem0_python, mem0_server, "track", json.dumps(track_ids)],
                        capture_output=True, text=True, timeout=10
                    )
                except Exception as e:
                    logger.debug("Track failed: %s", e)

            if loser_ids:
                try:
                    subprocess.run(
                        [mem0_python, mem0_server, "set_frozen", json.dumps(list(loser_ids))],
                        capture_output=True, text=True, timeout=10
                    )
                except Exception as e:
                    logger.debug("Set frozen failed: %s", e)

        final = kept[:top_k]
        lines = []
        for i, mem in enumerate(final):
            ci = None
            for cm, cw, cs, cl in conflicts:
                if cm.get("id") == mem.get("id"):
                    for j, km in enumerate(final):
                        if km.get("id") == cw.get("id"):
                            ci = (j + 1, cs, cl)
                            break
            lines.append(fmt_mem(mem, ci))

        kept_ids = [m.get("id", "") for m in kept]
        return lines, kept_ids, shadow_ids

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
        # Prefer gateway-provided user_id for per-user memory scoping;
        # fall back to config/env default for CLI (single-user) sessions.
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)

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

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Memory\n"
            f"Active. User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview.\n"
            "\n"
            "## Memory Format\n"
            "Each memory has a prefix: `[Updated N days ago | YYYY-MM-DD | 高频/中频/低频]`\n"
            "- 高频 (>20次访问): 重要事实，优先信任\n"
            "- 中频 (5-20次): 常规信息\n"
            "- 低频 (<5次): 可能已过时\n"
            "Conflicts are marked: `⚠️ 可能与第N条冲突(相似度X.XX)` (高确定性冲突)\n"\
            "Related items are marked: `🔗 可能与第N条相关(相似度X.XX，可能描述同一事物的不同状态)` (主题相关，非冲突)\n"\
            "\n"\
            "## Conflict Resolution Rules (when conflicting memories appear)\n"\
            "1. **时效优先**: 选择更新时间较近的记忆（Updated N days ago 较小）\n"\
            "2. **频次优先** (时间差 < 3天): 选择访问频次较高的记忆（高频 > 中频 > 低频）\n"\
            "3. **冲突确认** (权重相当时): 不要自行决定丢弃任何一方，在回复中委婉提及两种可能性\n"\
            "4. **上下文优先**: 如果某条冲突记忆与当前对话上下文高度吻合（如用户刚提了某个地点、版本号），则覆盖时效优先规则，优先采信上下文一致的那条\n"\
            "5. **🔗 标记说明**: 带 🔗 的记忆对属于主题相关而非冲突，可结合上下文判断二者是否并存或替代关系"
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
                import subprocess
                import re
                
                # Over-fetching: retrieve top_k=20, deduplicate, then inject top 5
                # This solves high-frequency memory occupation and improves recall
                top_k_fetch = 20
                top_k_inject = 5
                
                # Get paths from config (with local defaults)
                cfg = _load_config()
                MEM0_SERVER = cfg.get("mem0_server", "m/media/data/mem045MEM0_DIR/mem0_server.py")
                MEM0_PYTHON = cfg.get("mem0_python", "m/media/data/mem045MEM0_DIR/.venv/bin/python")

                def run_with_retry(cmd, timeout=30, max_retries=3):
                    """Run subprocess with exponential backoff retry."""
                    last_error = None
                    for attempt in range(max_retries):
                        try:
                            result = subprocess.run(
                                cmd, capture_output=True, text=True, timeout=timeout
                            )
                            if result.returncode == 0:
                                return result
                            last_error = f"Command failed (attempt {attempt + 1}): {result.stderr}"
                        except subprocess.TimeoutExpired as e:
                            last_error = f"Timeout (attempt {attempt + 1}): {e}"
                        except Exception as e:
                            last_error = f"Error (attempt {attempt + 1}): {e}"

                        # Exponential backoff: 1s, 2s, 4s
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)

                    raise RuntimeError(last_error)

                # Step 1: Search with --no-track (decouple search from tracking)
                result = run_with_retry(
                    [
                        MEM0_PYTHON, MEM0_SERVER,
                        "search", query, self._user_id, str(top_k_fetch),
                        "true" if self._rerank else "false", "--no-track"
                    ],
                    timeout=30
                )
                
                data = json.loads(result.stdout)
                results = self._unwrap_results(data)
                
                if not results:
                    return

                # Apply shared deduplication + conflict detection + track freezing
                cfg = _load_config()
                MEM0_SERVER = cfg.get("mem0_server", "m/media/data/mem045MEM0_DIR/mem0_server.py")
                MEM0_PYTHON = cfg.get("mem0_python", "m/media/data/mem045MEM0_DIR/.venv/bin/python")

                top_k_inject = 5
                lines, kept_ids, shadow_ids = self._advanced_dedup_and_format(
                    results, top_k_inject,
                    mem0_python=MEM0_PYTHON,
                    mem0_server=MEM0_SERVER,
                )

                with self._prefetch_lock:
                    self._prefetch_result = "\n".join(lines)

                
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
                # Use subprocess to mem0_server.py for consistent embedding support
                import subprocess
                
                # Get paths from config (with local defaults)
                cfg = _load_config()
                MEM0_SERVER = cfg.get("mem0_server", "m/media/data/mem045MEM0_DIR/mem0_server.py")
                MEM0_PYTHON = cfg.get("mem0_python", "m/media/data/mem045MEM0_DIR/.venv/bin/python")

                result = subprocess.run(
                    [
                        MEM0_PYTHON, MEM0_SERVER,
                        "search", query, self._user_id, str(top_k),
                        "true" if rerank else "false", "--no-track"
                    ],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr)

                data = json.loads(result.stdout)
                results = self._unwrap_results(data)
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})

                # Apply deduplication + conflict detection (shared with prefetch)
                inject_k = min(top_k, 10)
                # DEBUG: log vector status
                vec_count = sum(1 for r in results if isinstance(r.get("_embedding"), list))
                logger.info("mem0_search: %d results, %d with vectors", len(results), vec_count)
                lines, kept_ids, shadow_ids = self._advanced_dedup_and_format(
                    results, inject_k,
                    mem0_python=MEM0_PYTHON,
                    mem0_server=MEM0_SERVER,
                )
                # DEBUG: log conflict count
                conflict_count = sum(1 for l in lines if "⚠️" in l)
                logger.info("mem0_search: %d lines, %d conflicts", len(lines), conflict_count)
                if lines:
                    return json.dumps({"result": "\n".join(lines), "count": len(lines)})
                return json.dumps({"result": "No relevant memories found after deduplication.", "count": 0})
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
