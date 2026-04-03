"""Embedding-based skill discovery using cosine similarity.

Embeds SKILL.md content via an API (OpenAI text-embedding-3-small or equivalent)
and caches embeddings in a JSON file. With ~100 skills and 1536-dim embeddings,
the cache is ~600KB — no vector database needed.

Cache: ~/.hermes/cache/skill_embeddings.json
Format: {skill_name: {embedding: [...], content_hash: "...", updated_at: "..."}}
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = get_hermes_home() / "cache" / "skill_embeddings.json"
_DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


class SkillEmbeddingStore:
    """Manages skill embeddings with file-based cache and cosine similarity search."""

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ):
        self._cache_path = cache_path or _DEFAULT_CACHE_PATH
        self._embedding_model = embedding_model
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self._loaded:
            return
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load embedding cache: %s", e)
                self._cache = {}
        self._loaded = True

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except OSError as e:
            logger.warning("Failed to save embedding cache: %s", e)

    def _content_hash(self, text: str) -> str:
        """SHA-256 hash of skill content for cache invalidation."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Embed text via the configured embedding API.

        Returns the embedding vector, or None on failure.
        """
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.debug("No API key available for embeddings")
            return None

        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.embeddings.create(
                model=self._embedding_model,
                input=text[:8000],  # Truncate to stay within limits
            )
            return response.data[0].embedding
        except ImportError:
            logger.debug("openai package not installed — embeddings unavailable")
            return None
        except Exception as e:
            logger.warning("Embedding API call failed: %s", e)
            return None

    def embed_skill(self, skill_name: str, skill_text: str) -> Optional[List[float]]:
        """Embed a skill's content, using cache when available.

        Returns the embedding vector, or None on failure.
        """
        self._load_cache()

        content_hash = self._content_hash(skill_text)
        cached = self._cache.get(skill_name)

        if cached and cached.get("content_hash") == content_hash:
            return cached.get("embedding")

        embedding = self.embed_text(skill_text)
        if embedding is None:
            return None

        self._cache[skill_name] = {
            "embedding": embedding,
            "content_hash": content_hash,
            "updated_at": time.time(),
        }
        self._save_cache()

        return embedding

    def invalidate(self, skill_name: str) -> None:
        """Remove a skill's cached embedding (call on skill update/delete)."""
        self._load_cache()
        if skill_name in self._cache:
            del self._cache[skill_name]
            self._save_cache()

    def find_matching_skills(
        self,
        query: str,
        skill_texts: Dict[str, str],
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """Find skills matching a query by cosine similarity.

        Args:
            query: The user's message or search query.
            skill_texts: Dict of {skill_name: skill_text} for all available skills.
            top_k: Number of top results to return.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of (skill_name, similarity_score) sorted by score descending.
        """
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return []

        self._load_cache()

        # Ensure all skills are embedded
        for name, text in skill_texts.items():
            if name not in self._cache or self._cache[name].get("content_hash") != self._content_hash(text):
                self.embed_skill(name, text)

        # Compute similarities
        results = []
        for name in skill_texts:
            cached = self._cache.get(name)
            if not cached or "embedding" not in cached:
                continue
            similarity = _cosine_similarity(query_embedding, cached["embedding"])
            if similarity >= min_similarity:
                results.append((name, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def refresh_stale(self, skills_dir: Path) -> int:
        """Re-embed skills whose content has changed on disk.

        Returns the number of skills refreshed.
        """
        self._load_cache()
        refreshed = 0

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            name = skill_dir.name
            text = skill_md.read_text(encoding="utf-8")
            content_hash = self._content_hash(text)

            cached = self._cache.get(name)
            if cached and cached.get("content_hash") == content_hash:
                continue

            embedding = self.embed_text(text)
            if embedding is not None:
                self._cache[name] = {
                    "embedding": embedding,
                    "content_hash": content_hash,
                    "updated_at": time.time(),
                }
                refreshed += 1

        if refreshed > 0:
            self._save_cache()

        return refreshed


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return dot_product / (norm_a * norm_b)


# Module-level singleton (lazy init)
_store: Optional[SkillEmbeddingStore] = None


def get_skill_embedding_store() -> SkillEmbeddingStore:
    """Get the global skill embedding store singleton."""
    global _store
    if _store is None:
        _store = SkillEmbeddingStore()
    return _store
