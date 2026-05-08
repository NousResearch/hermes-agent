"""LanceDB memory plugin — Hybrid Retrieval (Vector + BM25) with Cross-Encoder Rerank.

Local-only, no external API required. Supports multi-scope isolation
(user_id / agent_id / session_id) and configurable embedding models.

Config via environment variables or $HERMES_HOME/lancedb.json:
  LANCEDB_EMBEDDING_MODEL   — sentence-transformers model name (default: all-MiniLM-L6-v2)
  LANCEDB_VECTOR_DIM        — embedding dimension (default: 384, auto-detected)
  LANCEDB_TABLE_NAME        — LanceDB table name (default: memories)
  LANCEDB_RERANK            — enable Cross-Encoder reranking (default: true)
  LANCEDB_RERANK_MODEL      — cross-encoder model (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
  LANCEDB_RERANK_TOP_K      — top_k for reranker (default: 20)

HuggingFace proxy config (for China/enterprise networks):
  HF_ENDPOINT              — HF mirror endpoint, e.g. https://hf-mirror.com
  HF_HOME                  — local model cache directory (default: ~/.cache/huggingface)
  HF_HTTP_PROXY            — HTTP proxy URL, e.g. http://192.168.1.188:63399
  HF_HTTPS_PROXY           — HTTPS proxy URL
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _default_hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _apply_hf_proxy() -> None:
    """Apply HuggingFace proxy/mirror settings from environment or lancedb.json config.

    This must be called BEFORE sentence-transformers loads any model, because
    the model is downloaded on first use and cached for the process lifetime.
    Settings applied here affect torch/huggingface_hub underneath the model
    loader, so subsequent calls to _get_embedding_model() and _get_reranker()
    will use the configured proxy automatically.
    """
    import json as _json

    # Load lancedb.json for HF settings
    cfg_path = _default_hermes_home() / "lancedb.json"
    hf_settings = {}
    if cfg_path.exists():
        try:
            hf_settings = _json.loads(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass

    # Priority: env vars override file settings
    hf_endpoint = os.environ.get("HF_ENDPOINT") or hf_settings.get("hf_endpoint", "")
    hf_home = os.environ.get("HF_HOME") or hf_settings.get("hf_home", "")
    hf_http_proxy = os.environ.get("HF_HTTP_PROXY") or hf_settings.get("hf_http_proxy", "")
    hf_https_proxy = os.environ.get("HF_HTTPS_PROXY") or hf_settings.get("hf_https_proxy", "")

    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["TRANSFORMERS_CACHE"] = hf_home
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = hf_home

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    if hf_http_proxy:
        os.environ["HF_HTTP_PROXY"] = hf_http_proxy
        os.environ["HTTP_PROXY"] = hf_http_proxy

    if hf_https_proxy:
        os.environ["HF_HTTPS_PROXY"] = hf_https_proxy
        os.environ["HTTPS_PROXY"] = hf_https_proxy


def _load_config() -> dict:
    # Apply HF proxy settings before any model download
    _apply_hf_proxy()

    config = {
        "embedding_model": os.environ.get("LANCEDB_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "vector_dim": int(os.environ.get("LANCEDB_VECTOR_DIM", "0")),  # 0 = auto
        "table_name": os.environ.get("LANCEDB_TABLE_NAME", "memories"),
        "rerank": os.environ.get("LANCEDB_RERANK", "true").lower() not in ("false", "0", ""),
        "rerank_model": os.environ.get("LANCEDB_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "rerank_top_k": int(os.environ.get("LANCEDB_RERANK_TOP_K", "20")),
        # HF proxy / mirror settings
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "hf_home": os.environ.get("HF_HOME", ""),
        "hf_http_proxy": os.environ.get("HF_HTTP_PROXY", ""),
        "hf_https_proxy": os.environ.get("HF_HTTPS_PROXY", ""),
    }

    config_path = _default_hermes_home() / "lancedb.json"
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

PROFILE_SCHEMA = {
    "name": "lancedb_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "lancedb_search",
    "description": (
        "Search memories by meaning (vector) or keywords (BM25). "
        "Hybrid search combines both. Use rerank=true for higher accuracy."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable Cross-Encoder reranking (default: true)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "scope": {"type": "string", "description": "Scope filter: 'user', 'agent', 'session', or 'all' (default: all)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "lancedb_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim. "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
            "scope": {"type": "string", "description": "Scope: 'user', 'agent', 'session' (default: user)."},
        },
        "required": ["conclusion"],
    },
}

REMOVE_SCHEMA = {
    "name": "lancedb_remove",
    "description": "Remove a specific memory by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "The memory ID to remove."},
        },
        "required": ["memory_id"],
    },
}


# ---------------------------------------------------------------------------
# LanceDBMemoryProvider
# ---------------------------------------------------------------------------

class LanceDBMemoryProvider(MemoryProvider):
    """Local LanceDB memory with hybrid vector + BM25 retrieval and Cross-Encoder reranking."""

    def __init__(self):
        self._config: Optional[dict] = None
        self._table = None
        self._db = None
        self._embedding_model = None
        self._reranker = None
        self._bm25_index: Optional["_BM25Index"] = None
        self._init_lock = threading.Lock()
        self._initialized = False
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._session_id = ""
        self._hermes_home = str(_default_hermes_home())

    @property
    def name(self) -> str:
        return "lancedb"

    def is_available(self) -> bool:
        try:
            import lancedb
            return True
        except ImportError:
            return False

    def get_config_schema(self):
        return [
            {
                "key": "embedding_model",
                "description": "sentence-transformers model for embeddings",
                "default": "all-MiniLM-L6-v2",
                "env_var": "LANCEDB_EMBEDDING_MODEL",
            },
            {
                "key": "rerank",
                "description": "Enable Cross-Encoder reranking",
                "default": "true",
                "choices": ["true", "false"],
                "env_var": "LANCEDB_RERANK",
            },
            {
                "key": "rerank_model",
                "description": "Cross-encoder model for reranking",
                "default": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "env_var": "LANCEDB_RERANK_MODEL",
            },
            {
                "key": "hf_home",
                "description": "Local model cache directory (e.g. /tmp/hf-models or ~/.cache/huggingface)",
                "default": "",
                "env_var": "HF_HOME",
            },
            {
                "key": "hf_endpoint",
                "description": "HuggingFace mirror endpoint (e.g. https://hf-mirror.com)",
                "default": "",
                "env_var": "HF_ENDPOINT",
            },
            {
                "key": "hf_http_proxy",
                "description": "HTTP proxy for model downloads (e.g. http://192.168.1.188:63399)",
                "default": "",
                "env_var": "HF_HTTP_PROXY",
            },
            {
                "key": "hf_https_proxy",
                "description": "HTTPS proxy for model downloads",
                "default": "",
                "env_var": "HF_HTTPS_PROXY",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        config_path = Path(hermes_home) / "lancedb.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._do_init()
            self._initialized = True

    def _do_init(self) -> None:
        import lancedb

        cfg = self._config or {}
        hermes_home = Path(self._hermes_home)
        db_path = hermes_home / "data" / "lancedb"
        db_path.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(db_path))
        table_name = cfg.get("table_name", "memories")

        # Create or open table
        vector_dim = cfg.get("vector_dim", 0)
        try:
            self._table = self._db.open_table(table_name)
            if vector_dim == 0:
                # Auto-detect from existing schema
                schema = self._table.schema
                for field in schema.fields:
                    if field.name == "vector":
                        vector_dim = field.type.num_dims if hasattr(field.type, "num_dims") else 384
                        break
        except Exception:
            pass

        if vector_dim == 0:
            vector_dim = 384  # default for all-MiniLM-L6-v2

        if self._table is None:
            import pyarrow as pa
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),
                pa.field("user_id", pa.string()),
                pa.field("agent_id", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("scope", pa.string()),
                pa.field("created_at", pa.float64()),
                pa.field("token_count", pa.int32()),
            ])
            self._table = self._db.create_table(table_name, schema=schema)

        # Lazy-load models
        self._vector_dim = vector_dim

    def _get_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        cfg = self._config or {}
        model_name = cfg.get("embedding_model", "all-MiniLM-L6-v2")
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(model_name)
            return self._embedding_model
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required for LanceDB memory. "
                "Install with: pip install sentence-transformers"
            )

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        cfg = self._config or {}
        if not cfg.get("rerank", True):
            return None
        model_name = cfg.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(model_name)
            return self._reranker
        except ImportError:
            logger.debug("Cross-Encoder not available, skipping rerank")
            return None

    def _embed(self, texts: List[str]) -> List[List[float]]:
        model = self._get_embedding_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    def _build_bm25(self) -> "_BM25Index":
        """Build BM25 index from current table contents."""
        self._ensure_initialized()
        table = self._table

        try:
            df = table.to_lance().to_pandas()
        except Exception:
            df = table.to_pandas()

        if "content" in df.columns:
            corpus = df["content"].fillna("").astype(str).tolist()
        else:
            corpus = []
        if "id" in df.columns and len(df["id"]) == len(corpus):
            doc_ids = [str(x) for x in df["id"].tolist()]
        else:
            doc_ids = [str(i) for i in range(len(corpus))]
        return _BM25Index(corpus, doc_ids)

    def _query_scope_filter(self, scope: str) -> dict:
        """Build LanceDB filter for scope."""
        filters = []
        if scope in ("user", "session"):
            filters.append(f'user_id == "{self._user_id}"')
        if scope in ("agent", "session"):
            filters.append(f'agent_id == "{self._agent_id}"')
        if scope == "session":
            filters.append(f'session_id == "{self._session_id}"')
        return " AND ".join(filters) if filters else None

    def _hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        scope: str = "all",
        use_rerank: bool = True,
    ) -> List[dict]:
        """Hybrid search: vector similarity + BM25 + optional reranking."""
        self._ensure_initialized()

        cfg = self._config or {}
        rerank_top_k = cfg.get("rerank_top_k", 20)

        # Get more candidates than final results for reranking
        candidate_k = max(top_k * 4, rerank_top_k)

        # 1. Vector search
        query_vec = self._embed([query])[0]

        vector_filter = self._query_scope_filter(scope)
        try:
            if vector_filter:
                vector_results = self._table.search(
                    query_vec, vector_column_name="vector"
                ).where(vector_filter).limit(candidate_k).to_list()
            else:
                vector_results = self._table.search(
                    query_vec, vector_column_name="vector"
                ).limit(candidate_k).to_list()
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            vector_results = []

        # 2. BM25 search (keyword-based)
        try:
            bm25 = self._build_bm25()
            bm25_scores = bm25.search(query, top_k=candidate_k)
            bm25_ids = set(m["id"] for m in bm25_scores)
        except Exception as e:
            logger.debug("BM25 search failed: %s", e)
            bm25_scores = []
            bm25_ids = set()

        # 3. Combine results (reciprocal rank fusion)
        combined: Dict[str, float] = {}
        for i, r in enumerate(vector_results):
            rid = r.get("id", str(i))
            combined[rid] = combined.get(rid, 0) + 1.0 / (60 + i)

        for i, r in enumerate(bm25_scores):
            rid = r.get("id", str(i))
            combined[rid] = combined.get(rid, 0) + 1.0 / (60 + i)

        # Sort by fused score
        ranked_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)

        # Build result map
        id_to_result = {}
        for r in vector_results:
            id_to_result[r.get("id", "")] = r
        for r in bm25_scores:
            id_to_result[r.get("id", "")] = r

        candidates = [id_to_result[rid] for rid in ranked_ids if rid in id_to_result][:candidate_k]

        # 4. Reranking
        if use_rerank and candidates:
            reranker = self._get_reranker()
            if reranker:
                pairs = [[query, r.get("content", "")] for r in candidates]
                try:
                    scores = reranker.predict(pairs)
                    scored = sorted(
                        zip(candidates, scores),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    candidates = [r for r, _ in scored[:top_k]]
                except Exception as e:
                    logger.debug("Reranking failed: %s", e)
                    candidates = candidates[:top_k]
            else:
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        return candidates

    # -----------------------------------------------------------------------
    # MemoryProvider implementation
    # -----------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._user_id = kwargs.get("user_id") or "hermes-user"
        self._agent_id = kwargs.get("agent_identity") or "hermes"
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", str(_default_hermes_home()))
        self._initialized = False  # defer actual init to first use

    def system_prompt_block(self) -> str:
        return (
            "# LanceDB Memory\n"
            f"Active. User: {self._user_id}. Agent: {self._agent_id}.\n"
            "Use lancedb_search to find memories, lancedb_conclude to store facts, "
            "lancedb_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query:
            return ""
        try:
            results = self._hybrid_search(query, top_k=5, scope="user", use_rerank=False)
            if not results:
                return ""
            lines = [r.get("content", "")[:200] for r in results if r.get("content")]
            return "## LanceDB Memory\n" + "\n".join(f"- {l}" for l in lines)
        except Exception as e:
            logger.debug("LanceDB prefetch failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass  # synchronous prefetch is sufficient for local storage

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Store conversation turn as a memory entry
        turn_text = f"User: {user_content}\nAssistant: {assistant_content}"
        try:
            self._ensure_initialized()
            import uuid
            vector = self._embed([turn_text])[0]
            self._table.add([{
                "id": str(uuid.uuid4()),
                "content": turn_text,
                "vector": vector,
                "user_id": self._user_id,
                "agent_id": self._agent_id,
                "session_id": self._session_id,
                "scope": "session",
                "created_at": time.time(),
                "token_count": len(turn_text.split()),
            }])
        except Exception as e:
            logger.debug("LanceDB sync_turn failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA, REMOVE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        self._ensure_initialized()

        if tool_name == "lancedb_profile":
            try:
                scope_filter = self._query_scope_filter("user")
                if scope_filter:
                    rows = self._table.search().where(scope_filter).limit(100).to_list()
                else:
                    rows = self._table.search().limit(100).to_list()
                if not rows:
                    return json.dumps({"result": "No memories stored yet."})
                contents = [r.get("content", "") for r in rows if r.get("content")]
                return json.dumps({"result": "\n".join(contents), "count": len(contents)})
            except Exception as e:
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "lancedb_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", True)
            top_k = min(int(args.get("top_k", 10)), 50)
            scope = args.get("scope", "all")
            try:
                results = self._hybrid_search(query, top_k=top_k, scope=scope, use_rerank=rerank)
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [
                    {"id": r.get("id", ""), "content": r.get("content", ""), "scope": r.get("scope", "")}
                    for r in results
                ]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                return tool_error(f"Search failed: {e}")

        elif tool_name == "lancedb_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            scope = args.get("scope", "user")
            try:
                import uuid
                vector = self._embed([conclusion])[0]
                self._table.add([{
                    "id": str(uuid.uuid4()),
                    "content": conclusion,
                    "vector": vector,
                    "user_id": self._user_id if scope in ("user", "agent") else "",
                    "agent_id": self._agent_id if scope in ("agent",) else "",
                    "session_id": self._session_id if scope == "session" else "",
                    "scope": scope,
                    "created_at": time.time(),
                    "token_count": len(conclusion.split()),
                }])
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                return tool_error(f"Failed to store: {e}")

        elif tool_name == "lancedb_remove":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            try:
                self._table.delete(f'id == "{memory_id}"')
                return json.dumps({"result": "Memory removed."})
            except Exception as e:
                return tool_error(f"Failed to remove: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        self._initialized = False
        self._db = None
        self._table = None


# ---------------------------------------------------------------------------
# BM25 (lightweight, no external deps beyond stdlib)
# ---------------------------------------------------------------------------

class _BM25Index:
    """Simple in-memory BM25 implementation. No external dependencies."""

    def __init__(self, documents: List[str], doc_ids: Optional[List[str]] = None):
        self.documents = documents
        if doc_ids is not None and len(doc_ids) == len(documents):
            self.doc_ids = list(doc_ids)
        else:
            self.doc_ids = [str(i) for i in range(len(documents))]
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _build_index(self):
        import math
        self.term_freqs: List[Dict[str, int]] = []
        self.doc_len: List[int] = []
        N = len(self.documents)
        self.avgdl = 0.0

        if N == 0:
            self.idf: Dict[str, float] = {}
            return

        df: Dict[str, int] = {}
        for doc in self.documents:
            tokens = self._tokenize(doc)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            self.doc_len.append(len(tokens))
            for t in tf:
                df[t] = df.get(t, 0) + 1

        self.avgdl = sum(self.doc_len) / N

        k1 = 1.5
        b = 0.75
        self.idf = {}
        for t, d in df.items():
            self.idf[t] = math.log((N - d + 0.5) / (d + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        import math
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores: Dict[int, float] = {}
        k1 = 1.5
        b = 0.75

        for i, tf in enumerate(self.term_freqs):
            score = 0.0
            for t in tokens:
                if t in tf:
                    freq = tf[t]
                    idf = self.idf.get(t, 0)
                    dl = self.doc_len[i]
                    numerator = freq * (k1 + 1)
                    denominator = freq + k1 * (1 - b + b * dl / max(self.avgdl, 1))
                    score += idf * numerator / denominator
            if score > 0:
                scores[i] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"id": self.doc_ids[i], "content": self.documents[i], "bm25_score": s}
            for i, s in ranked
        ]


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_memory_provider(LanceDBMemoryProvider())
