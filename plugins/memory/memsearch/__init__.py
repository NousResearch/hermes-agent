"""MemSearch memory plugin — MemoryProvider for semantic long-term recall via Milvus.

Provides cross-session semantic memory with hybrid search (dense vector + BM25 RRF),
progressive disclosure (recall → expand → transcript), auto-ingest of conversation
turns, and compact summarization via the MemSearch Python library.

Config in $HERMES_HOME/memsearch_config.json or config.yaml:
  plugins:
    memsearch:
      milvus_uri: ~/.memsearch/milvus.db
      embedding_provider: openai
      collection: hermes_memory
      auto_ingest: true
      auto_compact: true
      max_recall_results: 10
      context_budget_tokens: 800
      index_paths: ""

Requires:
  - pip install memsearch milvus-lite>=3.0
  - Embedding API key (OpenAI, Google, etc.) or local/onnx provider
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "milvus_uri": "~/.memsearch/milvus.db",
    "embedding_provider": "openai",
    "embedding_model": "",
    "collection": "",  # auto-derived from profile name; "" → "hermes_memory" fallback
    "auto_ingest": True,
    "auto_compact": True,
    "compact_model": "",
    "max_recall_results": 5,
    "context_budget_tokens": 800,
    "index_paths": "",
    "sync_mode": "daemon",
}

# Collection name prefix — all per-profile collections start with this
_COLLECTION_PREFIX = "hermes_memory"


def _retry(fn, max_retries: int = 3, base_delay: float = 1.0):
    """Retry a callable with exponential backoff.

    Only retries on rate-limit errors (429, "rate", "quota").
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err_str = str(e).lower()
            if "rate" not in err_str and "429" not in err_str and "quota" not in err_str:
                raise
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("Rate limited (%s), retrying in %.1fs", e, delay)
            time.sleep(delay)
    return None


def _derive_collection_name(hermes_home: str) -> str:
    """Derive a per-profile Milvus collection name from the HERMES_HOME path.

    Profile isolation strategy:
      - ``~/.hermes`` (default, no profile)  → ``hermes_memory``
      - ``~/.hermes/profiles/dtpham``         → ``hermes_memory_dtpham``
      - ``~/.hermes/profiles/dtpham-02``       → ``hermes_memory_dtpham_02``
      - ``~/.hermes/profiles/dtpham2--clone``  → ``hermes_memory_dtpham2_clone``

    Milvus collection names must be alphanumeric + underscores, so ``-`` and
    any other special chars are replaced with ``_``.
    """
    home = Path(hermes_home).resolve()

    # Walk up to find «profiles/<name>» segment
    parts = home.parts
    for i in range(len(parts) - 1, 0, -1):
        if parts[i - 1] == "profiles":
            profile_slug = parts[i]
            # Sanitise for Milvus: only [a-zA-Z0-9_]
            safe = re.sub(r"[^a-zA-Z0-9]", "_", profile_slug)
            # Collapse runs of underscores
            safe = re.sub(r"_{2,}", "_", safe).strip("_")
            return f"{_COLLECTION_PREFIX}_{safe}"

    # No «profiles/<name>» found — default profile, use base name
    return _COLLECTION_PREFIX


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "memsearch_recall",
    "description": (
        "Semantic search across indexed memory. Returns ranked excerpts from past "
        "conversations, notes, and documents. Use for finding specific facts, "
        "decisions, or context from earlier sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results (default 5, max 20).",
            },
        },
        "required": ["query"],
    },
}

EXPAND_SCHEMA = {
    "name": "memsearch_expand",
    "description": (
        "Expand a memory chunk to show full section context. Use after memsearch_recall "
        "when a search result snippet is not enough — shows the complete heading section "
        "from the original document. Progressive disclosure: recall → expand → transcript."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chunk_hash": {
                "type": "string",
                "description": "The chunk hash from a memsearch_recall result to expand.",
            },
            "lines": {
                "type": "integer",
                "description": "Number of context lines around the chunk (default: full section).",
            },
        },
        "required": ["chunk_hash"],
    },
}

INGEST_SCHEMA = {
    "name": "memsearch_ingest",
    "description": (
        "Index a file or directory into semantic memory. Markdown files are chunked, "
        "embedded, and stored in Milvus for future recall. Only new/changed content "
        "is indexed (content-hash dedup)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File or directory path to index.",
            },
            "force": {
                "type": "boolean",
                "description": "Re-index everything, even unchanged content (default: false).",
            },
        },
        "required": ["path"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_plugin_config(hermes_home: str = "") -> dict:
    """Load config from $HERMES_HOME/memsearch_config.json or config.yaml."""
    from hermes_constants import get_hermes_home
    _home = hermes_home or str(get_hermes_home())

    # 1. Try native JSON config
    config_path = Path(_home) / "memsearch_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 2. Try config.yaml under plugins.memsearch
    try:
        import yaml
        from hermes_cli.config import cfg_get
        yaml_path = Path(_home) / "config.yaml"
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8-sig") as f:
                all_config = yaml.safe_load(f) or {}
            plugin_cfg = cfg_get(all_config, "plugins", "memsearch", default={})
            if plugin_cfg:
                return plugin_cfg
    except Exception:
        pass

    return {}


def _real_home() -> str:
    """Return the real user home, even inside a sandboxed HERMES_HOME.

    When Hermes runs the gateway, HOME may be set to a profile sandbox
    (e.g., ``~/.hermes/profiles/<name>/home``).  Milvus and other data
    stores should use the *real* home so that the CLI and the agent
    share the same database.
    """
    import pwd
    for candidate in (
        os.environ.get("SUDO_HOME"),
        os.environ.get("HOME"),
    ):
        if candidate and "/profiles/" not in candidate:
            return candidate
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except Exception:
        return os.path.expanduser("~")


def _expand_paths(cfg: dict) -> dict:
    """Expand ``~`` in path values to the real (non-sandboxed) home."""
    real_home = _real_home()
    path_keys = {"milvus_uri", "index_paths"}
    for key in path_keys & cfg.keys():
        val = cfg[key]
        if isinstance(val, str) and "~" in val:
            cfg[key] = val.replace("~", real_home, 1)
    return cfg


def _default_config() -> dict:
    """Return a fresh copy of defaults merged with any saved config."""
    cfg = dict(_DEFAULTS)
    saved = _load_plugin_config()
    if saved:
        cfg.update(saved)
    cfg = _expand_paths(cfg)
    return cfg


# ---------------------------------------------------------------------------
# MemSearchMemoryProvider
# ---------------------------------------------------------------------------

class MemSearchMemoryProvider(MemoryProvider):
    """MemSearch-backed semantic memory with hybrid search and auto-ingest.

    Uses the MemSearch Python library directly (no subprocess) for low-latency
    search, expand, and index operations. Falls back to subprocess for compact
    (which is async-only in the library and runs infrequently).
    """

    def __init__(self, config: dict | None = None):
        self._config = config or _default_config()
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._is_primary: bool = True
        self._memsearch_available: bool = False
        self._ms: Any = None  # MemSearch instance (lazy-init)
        self._ms_lock = threading.Lock()
        self._ms_init_failed: bool = False
        self._daemon_thread: threading.Thread | None = None
        self._pending_turns: list[tuple[str, str]] = []
        self._turn_lock = threading.Lock()
        # Stats cache: (chunk_count, timestamp)
        self._stats_cache: tuple[int, float] = (0, 0.0)

    @property
    def name(self) -> str:
        return "memsearch"

    # -------------------------------------------------------------------
    # Lazy MemSearch instance management
    # -------------------------------------------------------------------

    def _get_ms(self) -> Any | None:
        """Return the cached MemSearch instance, lazy-initializing on first call.

        If the cached instance has a stale/closed gRPC channel, it is discarded
        and a fresh instance is created.  This handles the case where a daemon
        thread races with shutdown() or where pymilvus closes the channel after
        an idle period.
        """
        if self._ms is not None:
            # Health check: if the underlying gRPC channel is closed, discard
            # the instance so we reconnect on the next call.
            if self._is_ms_channel_closed(self._ms):
                logger.warning("MemSearch: detected closed gRPC channel, reconnecting")
                self._ms = None
                self._ms_init_failed = False
            else:
                return self._ms
        if self._ms_init_failed:
            return None
        return self._init_ms()

    @staticmethod
    def _is_ms_channel_closed(ms: Any) -> bool:
        """Check if the MemSearch instance's gRPC channel is closed/broken."""
        try:
            store = getattr(ms, '_store', None)
            if store is None:
                return False
            client = getattr(store, '_client', None)
            if client is None:
                return False
            # pymilvus MilvusClient sets _handler = None after close()
            handler = getattr(client, '_handler', None)
            if handler is None:
                return True  # Client was explicitly closed
            # GrpcHandler sets _channel = None after close()
            channel = getattr(handler, '_channel', None)
            if channel is None:
                return True  # Channel was explicitly closed
            # Try a lightweight gRPC connectivity check
            try:
                # grpc._cython.cygrpc.ChannelWrapped has check_connectivity_state
                inner = getattr(channel, '_channel', None)
                if inner is not None and hasattr(inner, 'check_connectivity_state'):
                    from grpc import ChannelConnectivity
                    state = inner.check_connectivity_state(False)
                    # TRANSIENT_FAILURE or SHUTDOWN means channel is unusable
                    if state in (ChannelConnectivity.TRANSIENT_FAILURE,
                                 ChannelConnectivity.SHUTDOWN):
                        return True
            except Exception:
                pass
            return False
        except Exception:
            return False

    def _init_ms(self) -> Any | None:
        """Initialize the MemSearch Python API instance. Thread-safe."""
        with self._ms_lock:
            # Double-check after acquiring lock
            if self._ms is not None:
                return self._ms
            if self._ms_init_failed:
                return None
            try:
                from memsearch.core import MemSearch
                ms = MemSearch(
                    milvus_uri=self._config.get("milvus_uri", "~/.memsearch/milvus.db"),
                    collection=self._config.get("collection", "hermes_memory"),
                    embedding_provider=self._config.get("embedding_provider", "openai"),
                    embedding_model=self._config.get("embedding_model", "") or None,
                    embedding_base_url=self._config.get("embedding_base_url", "") or None,
                )
                self._ms = ms
                logger.info("MemSearch Python API initialized (direct, no subprocess)")
                return ms
            except Exception as e:
                self._ms_init_failed = True
                logger.warning("MemSearch direct API init failed: %s — falling back to subprocess", e)
                return None

    # -----------------------------------------------------------------------
    # Core lifecycle
    # -----------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if memsearch CLI is installed and embedding API key is set.

        Does NOT make network calls — only checks package and env vars.
        """
        try:
            import memsearch  # noqa: F401
        except ImportError:
            return False
        provider = self._config.get("embedding_provider", "openai")
        if provider == "openai":
            return bool(os.environ.get("OPENAI_API_KEY", ""))
        elif provider == "google":
            # google-genai accepts both GOOGLE_API_KEY and GEMINI_API_KEY
            return bool(
                os.environ.get("GOOGLE_API_KEY", "")
                or os.environ.get("GEMINI_API_KEY", "")
            )
        elif provider in ("local", "onnx", "ollama"):
            return True  # no API key needed
        # Default: check for OPENAI_API_KEY as fallback
        return bool(os.environ.get("OPENAI_API_KEY", ""))

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize MemSearch for a session.

        kwargs always includes:
          - hermes_home (str): Active HERMES_HOME path
          - platform (str): "cli", "telegram", etc.
          - agent_context (str): "primary", "subagent", "cron", or "flush"
        """
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))

        # Skip writes for non-primary contexts (cron, subagent)
        self._is_primary = kwargs.get("agent_context", "primary") == "primary"

        # Reload config from hermes_home (expands ~ to real home)
        saved = _load_plugin_config(self._hermes_home)
        if saved:
            self._config.update(saved)
        self._config = _expand_paths(self._config)

        # --- Per-profile collection isolation ---
        # If the user explicitly set a collection name in config, respect it.
        # Otherwise, auto-derive from the profile name so each profile gets
        # its own isolated collection within the shared Milvus DB.
        if not self._config.get("collection"):
            derived = _derive_collection_name(self._hermes_home)
            self._config["collection"] = derived
            logger.info("MemSearch: auto-derived collection=%s from hermes_home=%s",
                        derived, self._hermes_home)
        elif self._config["collection"] == _COLLECTION_PREFIX:
            # Config says "hermes_memory" but that's the default — also auto-derive
            # so existing configs don't accidentally share a single collection.
            derived = _derive_collection_name(self._hermes_home)
            if derived != _COLLECTION_PREFIX:
                logger.info("MemSearch: overriding default collection 'hermes_memory' → '%s' "
                            "for profile isolation", derived)
                self._config["collection"] = derived

        # --- Migrate legacy data if needed ---
        # If we just derived a profile-specific collection and the old
        # "hermes_memory" collection has data but the new one is empty,
        # copy all vectors from the legacy collection.
        self._maybe_migrate_collection()

        # Reset init state so we re-create MemSearch instance with new config
        self._ms = None
        self._ms_init_failed = False

        # Verify memsearch is importable
        try:
            import memsearch  # noqa: F401
            self._memsearch_available = True
        except ImportError:
            self._memsearch_available = False
            logger.warning("MemSearch package not installed — memory features disabled")

        # Auto-index configured paths on init (only for primary sessions)
        if self._config.get("auto_ingest", True) and self._is_primary:
            index_paths = self._config.get("index_paths", "")
            if index_paths:
                for path_str in index_paths.split(","):
                    path_str = path_str.strip()
                    if path_str:
                        expanded = path_str.replace("$HERMES_HOME", self._hermes_home)
                        expanded = expanded.replace("${HERMES_HOME}", self._hermes_home)
                        expanded = os.path.expanduser(expanded)
                        if Path(expanded).exists():
                            self._index_path(expanded)

        logger.info(
            "MemSearch initialized (session=%s, primary=%s, available=%s)",
            session_id, self._is_primary, self._memsearch_available,
        )

    def shutdown(self) -> None:
        """Flush pending turns and shut down."""
        if self._pending_turns:
            self._flush_turns()
        # Close Milvus client connection — but do NOT stop the Milvus Lite
        # gRPC server. The server is a process-level singleton shared across
        # sessions; stopping it via release_server() kills it for all clients
        # and causes "Cannot invoke RPC on closed channel" for any concurrent
        # or subsequent operations (including the daemon sync thread).
        ms = self._ms
        if ms is not None:
            try:
                # Only close the pymilvus client channel, not the Milvus Lite
                # server.  MilvusStore.close() calls server_manager_instance.
                # release_server() which stops the embedded gRPC server entirely
                # — we must not do that in a long-running process.
                if hasattr(ms, '_store') and hasattr(ms._store, '_client'):
                    ms._store._client.close()
                else:
                    ms.close()
            except Exception:
                pass
            self._ms = None
        logger.info("MemSearch shutdown (session=%s)", self._session_id)

    # -----------------------------------------------------------------------
    # Per-profile collection migration
    # -----------------------------------------------------------------------

    def _maybe_migrate_collection(self) -> None:
        """One-time migration: copy data from legacy 'hermes_memory' to the
        per-profile collection if the new collection is empty but the legacy
        one has data.

        Safe to call on every init — it's a no-op after the first run.
        """
        collection = self._config.get("collection", "")
        if not collection or collection == _COLLECTION_PREFIX:
            # No profile-specific collection — nothing to migrate.
            return

        milvus_uri = self._config.get("milvus_uri", "")
        if not milvus_uri:
            return

        # Check if migration already done (flag file)
        flag_path = Path(self._hermes_home) / "memory" / f".migrated_{collection}"
        if flag_path.exists():
            return

        try:
            from pymilvus import MilvusClient
            client = MilvusClient(uri=milvus_uri)
            try:
                # Ensure legacy collection is loaded
                if client.has_collection(_COLLECTION_PREFIX):
                    client.load_collection(_COLLECTION_PREFIX)

                # Check if legacy has data
                legacy_count = 0
                try:
                    stats = client.get_collection_stats(_COLLECTION_PREFIX)
                    if isinstance(stats, dict):
                        legacy_count = int(stats.get("row_count", 0))
                    elif isinstance(stats, (int, float)):
                        legacy_count = int(stats)
                except Exception:
                    pass

                if legacy_count == 0:
                    # Nothing to migrate
                    flag_path.parent.mkdir(parents=True, exist_ok=True)
                    flag_path.write_text("no_data", encoding="utf-8")
                    return

                # Check if target collection already has data
                if client.has_collection(collection):
                    target_count = 0
                    try:
                        client.load_collection(collection)
                        stats = client.get_collection_stats(collection)
                        if isinstance(stats, dict):
                            target_count = int(stats.get("row_count", 0))
                        elif isinstance(stats, (int, float)):
                            target_count = int(stats)
                    except Exception:
                        pass
                    if target_count > 0:
                        # Already has data — skip migration
                        flag_path.parent.mkdir(parents=True, exist_ok=True)
                        flag_path.write_text(f"skipped_target_has_{target_count}", encoding="utf-8")
                        return

                # Migrate: query all from legacy, insert into target
                logger.info("MemSearch: migrating %d chunks from '%s' → '%s'",
                           legacy_count, _COLLECTION_PREFIX, collection)
                from memsearch.core import MemSearch
                ms_src = MemSearch(
                    milvus_uri=milvus_uri,
                    collection=_COLLECTION_PREFIX,
                    embedding_provider=self._config.get("embedding_provider", "google"),
                    embedding_model=self._config.get("embedding_model", "") or None,
                    embedding_base_url=self._config.get("embedding_base_url", "") or None,
                )
                ms_dst = MemSearch(
                    milvus_uri=milvus_uri,
                    collection=collection,
                    embedding_provider=self._config.get("embedding_provider", "google"),
                    embedding_model=self._config.get("embedding_model", "") or None,
                    embedding_base_url=self._config.get("embedding_base_url", "") or None,
                )
                try:
                    # Get all source paths that were indexed
                    src_store = ms_src._store
                    # Re-index all source files into the new collection
                    # by querying the source for unique source paths
                    results = _retry(lambda: self._run_async(ms_src.search("*", top_k=min(legacy_count, 200))))
                    source_paths = set()
                    for r in results:
                        sp = r.get("source", "")
                        if sp and os.path.exists(sp):
                            source_paths.add(sp)

                    indexed = 0
                    for sp in sorted(source_paths):
                        try:
                            n = _retry(lambda: self._run_async(ms_dst.index_file(sp)))
                            indexed += 1
                            logger.debug("MemSearch migration: indexed %s (%s chunks)", sp, n)
                        except Exception as e:
                            logger.warning("MemSearch migration: failed to index %s: %s", sp, e)

                    logger.info("MemSearch: migration complete — %d/%d source files re-indexed into '%s'",
                               indexed, len(source_paths), collection)
                finally:
                    ms_src.close()
                    ms_dst.close()

                # Mark migration as done
                flag_path.parent.mkdir(parents=True, exist_ok=True)
                flag_path.write_text(f"migrated_{len(source_paths)}_files", encoding="utf-8")

            finally:
                client.close()

        except Exception as e:
            # Milvus unavailable or locked — non-critical, will retry next init
            logger.debug("MemSearch migration skipped (will retry): %s", e)

    # -----------------------------------------------------------------------
    # Context injection
    # -----------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        """Static description for the system prompt. Cached for 5 minutes."""
        if not self._memsearch_available:
            return ""

        count, ts = self._stats_cache
        now = time.monotonic()
        if now - ts > 300:  # Cache for 5 minutes
            count = self._get_chunk_count()
            self._stats_cache = (count, now)

        if count == 0:
            return (
                "# MemSearch Memory\n"
                "Active. Empty index — conversations will be auto-indexed.\n"
                "Use memsearch_recall to search memory, memsearch_ingest to add files."
            )
        return (
            f"# MemSearch Memory\n"
            f"Active. {count} chunks indexed with hybrid semantic search.\n"
            f"Use memsearch_recall for semantic search, memsearch_expand for full context, "
            f"memsearch_ingest to add files."
        )

    def _get_chunk_count(self) -> int:
        """Get chunk count from Milvus. Uses direct API if available, falls back to subprocess."""
        ms = self._get_ms()
        if ms is not None:
            try:
                # Access the store directly
                from pymilvus import MilvusClient
                stats = ms._store._client.get_collection_stats(ms._store._collection)
                for line in str(stats).strip().split("\n"):
                    if "row_count" in line.lower() or "count" in line.lower():
                        try:
                            num_part = line.split(":")[-1].strip().strip('"').strip("'")
                            return int(num_part)
                        except (ValueError, IndexError):
                            pass
                # Try dict access
                if isinstance(stats, dict):
                    return int(stats.get("row_count", stats.get("count", 0)))
            except Exception:
                pass
        # Fallback to subprocess
        return self._subprocess_chunk_count()

    def _subprocess_chunk_count(self) -> int:
        """Fallback: count chunks via memsearch CLI."""
        collection = self._config.get("collection", "hermes_memory")
        try:
            result = __import__("subprocess").run(
                ["memsearch", "stats", "--collection", collection],
                capture_output=True, text=True, timeout=10,
            )
            output = (result.stdout or "") + (result.stderr or "")
            for line in output.strip().split("\n"):
                if "chunk" in line.lower():
                    try:
                        num_part = line.split(":")[-1].strip()
                        return int(num_part)
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass
        return 0

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant memory before each API call."""
        if not self._memsearch_available or not query or len(query) < 5:
            return ""
        max_results = int(self._config.get("max_recall_results", 5))
        budget = int(self._config.get("context_budget_tokens", 800))
        results = self._search(query, top_k=max_results)
        if not results:
            return ""

        # Format results within token budget (≈4 chars per token)
        max_chars = budget * 4
        lines: list[str] = []
        total_chars = 0
        for r in results:
            score = r.get("score", 0)
            source = r.get("source", "")
            heading = r.get("heading", "")
            content = r.get("content", "")[:500]
            chunk_hash = r.get("chunk_hash", "")
            line = f"- [{score:.2f}] {heading} ({source})"
            if chunk_hash:
                line += f" [ref={chunk_hash[:12]}]"
            line += f"\n  {content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        if not lines:
            return ""
        return "## MemSearch Recall\n" + "\n".join(lines)

    # -----------------------------------------------------------------------
    # Turn sync and session lifecycle
    # -----------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Queue a completed turn for background indexing.

        MUST be non-blocking per the MemoryProvider contract.
        """
        if not self._is_primary or not self._config.get("auto_ingest", True):
            return
        if not self._memsearch_available:
            return
        if not user_content or not assistant_content:
            return

        need_spawn = False
        with self._turn_lock:
            self._pending_turns.append((user_content, assistant_content))
            need_spawn = self._daemon_thread is None or not self._daemon_thread.is_alive()
            if need_spawn:
                self._daemon_thread = threading.Thread(target=self._background_sync, daemon=True)

        if need_spawn:
            self._daemon_thread.start()

    def _background_sync(self) -> None:
        """Background worker: debounce then flush pending turns."""
        time.sleep(2)  # debounce
        self._flush_turns()

    def _flush_turns(self) -> None:
        """Write pending turns to markdown, then index via MemSearch."""
        with self._turn_lock:
            turns = list(self._pending_turns)
            self._pending_turns.clear()

        if not turns:
            return

        # Write turns as markdown to hermes_home/memory/
        memory_dir = Path(self._hermes_home) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        md_path = memory_dir / f"{ts}-{self._session_id[:8]}.md"

        lines: list[str] = []
        for user, assistant in turns:
            lines.append(f"## User\n\n{user}\n")
            lines.append(f"## Assistant\n\n{assistant}\n")

        # Append (don't overwrite — multiple flushes per session)
        with open(md_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Index the file
        self._index_path(str(md_path))

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Flush all pending turns and optionally run compact."""
        self._flush_turns()
        if self._config.get("auto_compact", True) and self._is_primary:
            self._run_compact()

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Hook called at the start of each agent turn."""
        pass

    def on_session_switch(self, new_session_id: str, *,
                          parent_session_id: str = "", reset: bool = False,
                          **kwargs) -> None:
        """Called when the agent switches session_id mid-process."""
        pass

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Called on the parent agent when a subagent completes."""
        pass

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract key insights before context compression discards messages."""
        if not messages:
            return ""
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return ""
        # Summarize last few user messages for preservation
        topics: set[str] = set()
        for m in user_msgs[-5:]:
            content = m.get("content", "")
            if isinstance(content, str) and len(content) > 10:
                first_sentence = content.split(".")[0][:100]
                topics.add(first_sentence)
        if not topics:
            return ""
        return "MemSearch context from compressed messages: " + "; ".join(topics)

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror built-in memory writes to MemSearch index."""
        if not content or action == "remove":
            return
        if not self._memsearch_available:
            return
        memory_dir = Path(self._hermes_home) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        md_path = memory_dir / f"memory-{target}-{action}.md"
        heading = f"# Memory {action}: {target}"
        md_path.write_text(f"{heading}\n\n{content}\n", encoding="utf-8")
        self._index_path(str(md_path))

    # -----------------------------------------------------------------------
    # Tool schemas and dispatch
    # -----------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, EXPAND_SCHEMA, INGEST_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "memsearch_recall":
            return self._handle_recall(args)
        elif tool_name == "memsearch_expand":
            return self._handle_expand(args)
        elif tool_name == "memsearch_ingest":
            return self._handle_ingest(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def _handle_recall(self, args: dict) -> str:
        query = args.get("query", "")
        top_k = min(int(args.get("top_k", 5)), 20)
        if not query:
            return tool_error("query is required")
        results = self._search(query, top_k=top_k)
        if not results:
            return json.dumps({"results": [], "count": 0, "message": "No results found"})
        formatted = []
        for r in results:
            formatted.append({
                "content": r.get("content", "")[:500],
                "source": r.get("source", ""),
                "heading": r.get("heading", ""),
                "chunk_hash": r.get("chunk_hash", ""),
                "score": r.get("score", 0),
            })
        return json.dumps({"results": formatted, "count": len(formatted)})

    def _handle_expand(self, args: dict) -> str:
        chunk_hash = args.get("chunk_hash", "")
        if not chunk_hash:
            return tool_error("chunk_hash is required")
        lines = args.get("lines")

        # Try direct Python API first
        ms = self._get_ms()
        if ms is not None:
            try:
                result = self._expand_direct(ms, chunk_hash, lines)
                if result is not None:
                    return result
            except ValueError as e:
                if "Cannot invoke RPC on closed channel" in str(e):
                    logger.warning("MemSearch: closed channel on expand, resetting and retrying")
                    self._ms = None
                    self._ms_init_failed = False
                    ms = self._get_ms()
                    if ms is not None:
                        try:
                            result = self._expand_direct(ms, chunk_hash, lines)
                            if result is not None:
                                return result
                        except Exception:
                            pass
            except Exception as e:
                logger.debug("MemSearch direct expand failed, falling back to subprocess: %s", e)

        # Subprocess fallback
        cmd = ["memsearch", "expand", chunk_hash, "--collection",
               self._config.get("collection", "hermes_memory"), "--json-output"]
        milvus_uri = self._config.get("milvus_uri", "")
        if milvus_uri:
            cmd.extend(["--milvus-uri", milvus_uri])
        if lines:
            cmd.extend(["--lines", str(lines)])
        self._add_provider_args(cmd)
        try:
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            return tool_error(f"Expand failed: {result.stderr.strip()[:200]}")
        except Exception as e:
            return tool_error(f"Expand error: {e}")

    def _expand_direct(self, ms: Any, chunk_hash: str, lines: int | None) -> str | None:
        """Expand a chunk using the direct Python API (no subprocess).

        Queries Milvus for the chunk, then reads the source file from disk
        to extract the surrounding context (section or lines).

        Returns JSON string on success, None on failure (caller falls back to subprocess).
        """
        try:
            store = ms._store
            # Query Milvus for the chunk by hash
            chunks = store.query(filter_expr=f'chunk_hash == "{chunk_hash}"')
            if not chunks:
                return json.dumps({"error": f"Chunk not found: {chunk_hash}", "count": 0})

            chunk = chunks[0]
            source = chunk.get("source", "")
            heading = chunk.get("heading", "")
            heading_level = chunk.get("heading_level", 0)
            start_line = chunk.get("start_line", 0)
            end_line = chunk.get("end_line", 0)

            if not source:
                return None  # Can't expand without a source file

            source_path = Path(source)
            if not source_path.exists():
                return None  # Source file gone — fall back to subprocess (different error path)

            all_lines = source_path.read_text(encoding="utf-8").splitlines()

            if lines is not None:
                # Show N lines before/after the chunk
                ctx_start = max(0, start_line - 1 - lines)
                ctx_end = min(len(all_lines), end_line + lines)
                expanded = "\n".join(all_lines[ctx_start:ctx_end])
                expanded_start = ctx_start + 1
                expanded_end = ctx_end
            else:
                # Show full section under the same heading
                expanded, expanded_start, expanded_end = self._extract_section(
                    all_lines, start_line, heading_level
                )

            result = {
                "chunk_hash": chunk_hash,
                "source": source,
                "heading": heading,
                "start_line": expanded_start,
                "end_line": expanded_end,
                "content": expanded,
            }
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug("MemSearch _expand_direct failed: %s", e)
            return None

    @staticmethod
    def _extract_section(
        all_lines: list[str], start_line: int, heading_level: int,
    ) -> tuple[str, int, int]:
        """Extract the full markdown section containing the chunk.

        Walks backward to find the section heading, then forward to the next
        heading of equal or higher level (or EOF).
        """
        section_start = start_line - 1  # 0-indexed
        if heading_level > 0:
            for i in range(start_line - 2, -1, -1):
                line = all_lines[i]
                if line.startswith("#"):
                    level = len(line) - len(line.lstrip("#"))
                    if level <= heading_level:
                        section_start = i
                        break

        section_end = len(all_lines)
        if heading_level > 0:
            for i in range(start_line, len(all_lines)):
                line = all_lines[i]
                if line.startswith("#"):
                    level = len(line) - len(line.lstrip("#"))
                    if level <= heading_level:
                        section_end = i
                        break

        content = "\n".join(all_lines[section_start:section_end])
        return content, section_start + 1, section_end

    def _handle_ingest(self, args: dict) -> str:
        path = args.get("path", "")
        if not path:
            return tool_error("path is required")
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            return tool_error(f"Path not found: {path}")
        force = args.get("force", False)
        self._index_path(str(path_obj), force=force)
        return json.dumps({"status": "indexed", "path": str(path_obj), "force": force})

    # -----------------------------------------------------------------------
    # Config schema
    # -----------------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/.memsearch/milvus.db"
        return [
            {
                "key": "embedding_provider",
                "description": "Embedding provider (openai, google, voyage, jina, mistral, ollama, local, onnx)",
                "default": "openai",
                "choices": ["openai", "google", "voyage", "jina", "mistral", "ollama", "local", "onnx"],
            },
            {
                "key": "milvus_uri",
                "description": "Milvus connection URI (local file or remote server)",
                "default": _default_db,
            },
            {
                "key": "collection",
                "description": "Milvus collection name (auto-derived from profile for isolation; leave empty for auto)",
                "default": "",
            },
            {
                "key": "api_key",
                "description": "Embedding API key",
                "secret": True,
                "required": True,
                "env_var": "OPENAI_API_KEY",
                "url": "https://platform.openai.com/api-keys",
            },
            {
                "key": "auto_ingest",
                "description": "Auto-index conversation turns as they happen",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "auto_compact",
                "description": "Run compact summary at session end",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "max_recall_results",
                "description": "Max results from semantic search",
                "default": "10",
            },
            {
                "key": "index_paths",
                "description": "Comma-separated paths to auto-index on init (e.g. ~/.hermes/skills/)",
                "default": "",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to memsearch_config.json, native settings via CLI."""
        config_path = Path(hermes_home) / "memsearch_config.json"
        # Save our config (excluding secrets)
        our_config = {k: v for k, v in values.items() if k not in ("api_key",)}
        config_path.write_text(json.dumps(our_config, indent=2), encoding="utf-8")
        # Also configure memsearch CLI for native settings
        key_map = {
            "milvus_uri": ("milvus", "uri"),
            "embedding_provider": ("embedding", "provider"),
            "embedding_model": ("embedding", "model"),
            "collection": ("milvus", "collection"),
        }
        for key, val in values.items():
            if key == "api_key":
                continue
            toml_key = key_map.get(key)
            if toml_key:
                try:
                    import subprocess
                    subprocess.run(
                        ["memsearch", "config", "set", f"{toml_key[0]}.{toml_key[1]}", str(val)],
                        capture_output=True, timeout=10,
                    )
                except Exception:
                    logger.debug("memsearch config set %s failed (non-critical)", key)

    # -----------------------------------------------------------------------
    # Internal: search (direct Python API with subprocess fallback)
    # -----------------------------------------------------------------------

    def _search(self, query: str, top_k: int = 5) -> list:
        """Search memory: direct Python API first, subprocess fallback.

        Automatically reconnects if the gRPC channel was closed (e.g., after
        an idle period or a session reset that closed the client).
        """
        ms = self._get_ms()
        if ms is not None:
            try:
                results = _retry(lambda: self._run_async(ms.search(query, top_k=top_k)))
                return results
            except ValueError as e:
                if "Cannot invoke RPC on closed channel" in str(e):
                    logger.warning("MemSearch: closed channel on search, resetting and retrying")
                    # Force re-init — _get_ms will detect the closed channel next time
                    self._ms = None
                    self._ms_init_failed = False
                    ms = self._get_ms()
                    if ms is not None:
                        try:
                            results = _retry(lambda: self._run_async(ms.search(query, top_k=top_k)))
                            return results
                        except Exception:
                            pass
                logger.debug("MemSearch direct search failed, falling back to subprocess: %s", e)
                return self._search_subprocess(query, top_k)
            except Exception as e:
                logger.debug("MemSearch direct search failed, falling back to subprocess: %s", e)
                return self._search_subprocess(query, top_k)
        return self._search_subprocess(query, top_k)

    @staticmethod
    def _run_async(coro) -> Any:
        """Run an async coroutine synchronously, handling event loop edge cases."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're inside an existing event loop (e.g., gateway asyncio loop)
            # Run in a separate thread to avoid "already running" error
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)
        else:
            return asyncio.run(coro)

    def _search_subprocess(self, query: str, top_k: int = 5) -> list:
        """Fallback: search via memsearch CLI subprocess."""
        import subprocess
        cmd = [
            "memsearch", "search", query,
            "--top-k", str(top_k),
            "--collection", self._config.get("collection", "hermes_memory"),
            "--json-output",
        ]
        self._add_milvus_uri_arg(cmd)
        self._add_provider_args(cmd)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except Exception as e:
            logger.debug("MemSearch subprocess search failed: %s", e)
        return []

    def _index_path(self, path: str, force: bool = False) -> None:
        """Index a path: direct Python API first, subprocess fallback."""
        ms = self._get_ms()
        if ms is not None:
            try:
                if force:
                    # force reindex — use subprocess which supports --force
                    raise ValueError("force reindex requires subprocess")
                n = _retry(lambda: self._run_async(ms.index_file(path)))
                logger.info("MemSearch indexed %s (direct API): %s chunks", path, n)
                # Invalidate stats cache since we added data
                self._stats_cache = (0, 0.0)
                return
            except ValueError as e:
                # Force reindex — fall through to subprocess
                if "force reindex" in str(e):
                    pass
                # Closed gRPC channel — reset and retry once
                elif "Cannot invoke RPC on closed channel" in str(e):
                    logger.warning("MemSearch: closed channel on index, resetting and retrying")
                    self._ms = None
                    self._ms_init_failed = False
                    ms = self._get_ms()
                    if ms is not None:
                        try:
                            n = _retry(lambda: self._run_async(ms.index_file(path)))
                            logger.info("MemSearch indexed %s (retry): %s chunks", path, n)
                            self._stats_cache = (0, 0.0)
                            return
                        except Exception:
                            pass
                else:
                    # Other ValueError — fall through to subprocess
                    logger.debug("MemSearch direct index ValueError: %s", e)
            except Exception as e:
                logger.debug("MemSearch direct index failed, falling back: %s", e)

        # Subprocess fallback
        import subprocess
        cmd = ["memsearch", "index", path, "--collection",
               self._config.get("collection", "hermes_memory")]
        if force:
            cmd.append("--force")
        self._add_milvus_uri_arg(cmd)
        self._add_provider_args(cmd)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                logger.info("MemSearch indexed %s: %s", path, result.stdout.strip())
                self._stats_cache = (0, 0.0)  # invalidate cache
            else:
                logger.warning("MemSearch index failed for %s: %s", path, result.stderr.strip()[:200])
        except Exception as e:
            logger.warning("MemSearch index error for %s: %s", path, e)

    def _run_compact(self) -> None:
        """Run memsearch compact to summarize indexed chunks.

        Compact uses LLM summarization which is async-only in the Python API.
        Use subprocess for simplicity.
        """
        import subprocess
        cmd = ["memsearch", "compact", "--collection",
               self._config.get("collection", "hermes_memory")]
        self._add_milvus_uri_arg(cmd)
        self._add_provider_args(cmd)
        compact_model = self._config.get("compact_model", "")
        if compact_model:
            cmd.extend(["--llm-model", compact_model])
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            logger.info("MemSearch compact completed")
        except Exception as e:
            logger.debug("MemSearch compact failed (non-critical): %s", e)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _add_milvus_uri_arg(self, cmd: list) -> None:
        """Add --milvus-uri to a subprocess command if configured."""
        uri = self._config.get("milvus_uri", "")
        if uri:
            cmd.extend(["--milvus-uri", uri])

    def _add_provider_args(self, cmd: list) -> None:
        """Add --provider and --model flags to a subprocess command."""
        provider = self._config.get("embedding_provider", "openai")
        cmd.extend(["--provider", provider])
        model = self._config.get("embedding_model", "")
        if model:
            cmd.extend(["--model", model])

    


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the MemSearch memory provider with the plugin system.

    Called by the memory plugin discovery mechanism when this provider
    is selected via ``memory.provider = memsearch`` in config.yaml.
    """
    config = _default_config()
    provider = MemSearchMemoryProvider(config=config)
    ctx.register_memory_provider(provider)