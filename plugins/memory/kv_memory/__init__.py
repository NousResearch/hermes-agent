"""kv-memory — Q4-compressed semantic memory provider for Hermes Agent.

Registers as a MemoryProvider plugin, providing:
  - Automatic turn embedding storage with Q4 quantization
  - Semantic recall via cosine similarity over stored embeddings
  - kv_memory_search and kv_memory_status tools
  - Session compaction and governance

Storage: SQLite with Q4-quantized embeddings (~3.8x compression).
Retrieval: Cosine similarity + temporal decay + MMR diversity reranking.
Backend: Pluggable — sentence-transformers (default), API, or future local-inference.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    kv-memory:
      embedding_backend: auto         # auto|sentence-transformers|api
      top_k: 5                        # results per search
      min_similarity: 0.5             # cosine similarity threshold
      retention_days: 90              # auto-prune threshold
"""

from __future__ import annotations

from .config import load_config
from .provider import KVMemoryProvider


def register(ctx) -> None:
    """Register the kv-memory provider with the Hermes plugin system."""
    config = load_config()
    provider = KVMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
