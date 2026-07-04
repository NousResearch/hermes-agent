"""kv-memory — Q4-compressed semantic memory provider for Hermes Agent.

Registers as a MemoryProvider plugin, providing:
  - Automatic turn embedding storage with Q4 quantization
  - Semantic recall via cosine similarity over stored embeddings
  - kv_memory_search and kv_memory_status tools
  - Session compaction and governance

Storage: SQLite with float16 embeddings (2x compression, zero quality loss).
Retrieval: Cosine similarity + temporal decay + MMR diversity reranking.
Backend: Pluggable — sentence-transformers (default), API, or future local-inference.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    kv-memory:
      embedding_backend: auto         # auto|sentence-transformers|api
      top_k: 5                        # results per search
      min_similarity: 0.0             # cosine similarity threshold
      retention_days: 90              # auto-prune threshold
"""

from __future__ import annotations

import logging

from .config import load_config
from .provider import KVMemoryProvider

logger = logging.getLogger(__name__)


def _create_post_llm_hook(provider: KVMemoryProvider):
    """Factory: return a post_llm_call hook that captures turns.

    This ensures sync happens even when the agent runs without tools
    (e.g., -z/--oneshot without --yolo). The MemoryManager also calls
    sync_turn for tool-using turns; we guard against double-sync via
    a per-turn lock.
    """
    _last_synced = set()  # track (session_id, turn_content_hash) pairs

    def hook(session_id=None, user_message=None, assistant_response=None,
             conversation_history=None, model=None, platform=None, **kwargs):
        # Lazy-init if the MemoryManager hasn't called initialize() yet
        if not provider._initialized:
            try:
                provider.initialize(session_id or "default", platform=platform or "cli")
            except Exception:
                return  # can't init, skip silently
        if not assistant_response or not user_message:
            return
        # Avoid double-sync: if the MemoryManager already called sync_turn
        # for this user+assistant pair, skip.
        sync_key = hash((session_id or "", user_message or "", assistant_response or ""))
        if sync_key in _last_synced:
            return
        _last_synced.add(sync_key)
        # Limit set size
        if len(_last_synced) > 100:
            _last_synced.clear()

        try:
            provider.sync_turn(
                user_content=user_message,
                assistant_content=assistant_response,
                session_id=session_id or "",
            )
        except Exception:
            pass  # never let a hook failure affect the agent

    return hook


def register(ctx) -> None:
    """Register the kv-memory provider with the Hermes plugin system."""
    config = load_config()
    provider = KVMemoryProvider(config=config)
    ctx.register_memory_provider(provider)

    # Also register with the global PluginManager so the hook fires.
    # The memory plugin ctx (ProviderCollector) silently ignores hooks.
    try:
        from hermes_cli.plugins import get_plugin_manager
        pm = get_plugin_manager()
        pm._hooks.setdefault("post_llm_call", []).append(
            _create_post_llm_hook(provider)
        )
    except Exception:
        pass
