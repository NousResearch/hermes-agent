"""Configuration for the kv-memory provider.

All tunables have defaults derived from the Persistent Q4 KV Cache paper
(arXiv Feb 2026) and the True Memory paper (May 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class KVMemoryConfig:
    """Configuration for the KV-Cache / Hidden-State memory provider."""

    # ── Storage ───────────────────────────────────────────────────
    db_path: str = ""            # SQLite database path (default: $HERMES_HOME/kv_memory.db)
    storage_mode: str = "hybrid" # "hybrid" | "fp16" | "q4" — embedding precision

    # ── Model ─────────────────────────────────────────────────────
    embedding_backend: str = "auto"  # "auto" | "sentence-transformers" | "api"
    embedding_model: str = ""        # model name for the embedding backend
    embedding_dim: int = 0           # embedding dimension (0 = auto-detect)

    # ── Retrieval ─────────────────────────────────────────────────
    top_k: int = 5                   # number of results to return
    min_similarity: float = 0.5      # minimum cosine similarity threshold
    temporal_decay_half_life: float = 7.0  # days; 0 = no decay
    causal_boost: float = 0.1        # boost for same-session / linked-session results
    diversity_lambda: float = 0.3    # MMR diversity weight (0 = no diversity reranking)

    # ── Q4 Quantization ───────────────────────────────────────────
    q4_channel_size: int = 0         # 0 = auto (head_dim); override for custom grouping

    # ── Memory governance ─────────────────────────────────────────
    max_stored_turns: int = 10000    # max turns before compaction triggers
    retention_days: int = 90         # auto-prune turns older than this
    auto_compact: bool = True        # run compaction on session end

    # ── Debug ─────────────────────────────────────────────────────
    log_level: str = "INFO"
    store_fp16_fidelity_check: bool = False  # store raw FP16 for Q4 fidelity checks

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KVMemoryConfig":
        """Create config from a dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage."""
        return {
            "db_path": self.db_path,
            "storage_mode": self.storage_mode,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "temporal_decay_half_life": self.temporal_decay_half_life,
            "causal_boost": self.causal_boost,
            "diversity_lambda": self.diversity_lambda,
            "q4_channel_size": self.q4_channel_size,
            "max_stored_turns": self.max_stored_turns,
            "retention_days": self.retention_days,
            "auto_compact": self.auto_compact,
            "log_level": self.log_level,
            "store_fp16_fidelity_check": self.store_fp16_fidelity_check,
        }


def load_config() -> KVMemoryConfig:
    """Load kv-memory config from Hermes config.yaml.

    Looks under ``plugins.kv-memory`` in the profile-scoped config.
    """
    config = KVMemoryConfig()
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return config

        import yaml
        from hermes_cli.config import cfg_get
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        plugin_config = cfg_get(all_config, "plugins", "kv-memory", default={}) or {}
        if plugin_config:
            config = KVMemoryConfig.from_dict(plugin_config)

        # Resolve default db_path
        if not config.db_path:
            config.db_path = str(get_hermes_home() / "kv_memory.db")
    except Exception:
        pass

    return config
