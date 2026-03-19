"""
Semantic Deduplication System - Autonomous Improvement #4 by Claude Sonnet 4.5

PROBLEM: Memory system can store duplicate/similar conversations, wasting space
SOLUTION: Semantic similarity checking before storage

Uses:
- TF-IDF for fast similarity (no embeddings needed)
- Fuzzy matching for exact duplicates
- Configurable similarity thresholds
- Smart merging of similar entries
"""

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from collections import Counter
import math
from loguru import logger


@dataclass
class SimilarityMatch:
    """A match between content and existing memory."""
    memory_id: str
    similarity: float
    content: str
    is_duplicate: bool  # True if > duplicate threshold


class SemanticDeduplicator:
    """
    Detects and handles duplicate/similar content in memory.

    Features:
    - TF-IDF similarity (no embeddings needed, fast)
    - Exact duplicate detection via hashing
    - Fuzzy similarity for near-duplicates
    - Configurable thresholds
    - Smart merging suggestions
    """

    def __init__(
        self,
        duplicate_threshold: float = 0.95,  # 95% similar = duplicate
        similar_threshold: float = 0.75,    # 75% similar = related
        min_content_length: int = 20        # Min chars to check
    ):
        self.duplicate_threshold = duplicate_threshold
        self.similar_threshold = similar_threshold
        self.min_content_length = min_content_length

        # Cache for existing content
        self.content_cache: Dict[str, str] = {}  # id -> content
        self.hash_cache: Dict[str, str] = {}     # hash -> id

        # Statistics
        self.duplicates_prevented = 0
        self.merges_suggested = 0

    def add_to_cache(self, memory_id: str, content: str):
        """Add content to cache for future comparisons."""
        if len(content) >= self.min_content_length:
            self.content_cache[memory_id] = content
            content_hash = self._hash_content(content)
            self.hash_cache[content_hash] = memory_id

    def check_duplicate(
        self,
        new_content: str,
        check_against: Optional[List[Tuple[str, str]]] = None
    ) -> Optional[SimilarityMatch]:
        """
        Check if content is a duplicate of existing memories.

        Args:
            new_content: Content to check
            check_against: List of (id, content) tuples to check against
                          If None, uses cache

        Returns:
            SimilarityMatch if duplicate found, None otherwise
        """
        if len(new_content) < self.min_content_length:
            return None

        # Check exact hash first (fastest)
        content_hash = self._hash_content(new_content)
        if content_hash in self.hash_cache:
            existing_id = self.hash_cache[content_hash]
            self.duplicates_prevented += 1
            logger.info(f"Exact duplicate detected: {existing_id}")
            return SimilarityMatch(
                memory_id=existing_id,
                similarity=1.0,
                content=self.content_cache.get(existing_id, ""),
                is_duplicate=True
            )

        # Use provided list or cache
        if check_against is None:
            check_against = list(self.content_cache.items())

        # Find most similar existing content
        best_match = None
        best_similarity = 0.0

        for memory_id, existing_content in check_against:
            similarity = self._calculate_similarity(new_content, existing_content)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (memory_id, existing_content)

        # Return match if above threshold
        if best_match and best_similarity >= self.similar_threshold:
            is_dup = best_similarity >= self.duplicate_threshold
            if is_dup:
                self.duplicates_prevented += 1
                logger.info(f"Semantic duplicate detected: {best_match[0]} ({best_similarity:.2%})")
            else:
                self.merges_suggested += 1
                logger.debug(f"Similar content found: {best_match[0]} ({best_similarity:.2%})")

            return SimilarityMatch(
                memory_id=best_match[0],
                similarity=best_similarity,
                content=best_match[1],
                is_duplicate=is_dup
            )

        return None

    def _hash_content(self, content: str) -> str:
        """Create normalized hash of content."""
        # Normalize: lowercase, remove extra whitespace
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using TF-IDF approach.

        Fast and effective without embeddings.
        """
        # Tokenize
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        # Calculate term frequencies
        tf1 = Counter(tokens1)
        tf2 = Counter(tokens2)

        # Get all unique terms
        all_terms = set(tf1.keys()) | set(tf2.keys())

        # Simple cosine similarity
        dot_product = sum(tf1[term] * tf2[term] for term in all_terms)
        magnitude1 = math.sqrt(sum(count ** 2 for count in tf1.values()))
        magnitude2 = math.sqrt(sum(count ** 2 for count in tf2.values()))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)

        # Adjust for length difference (very different lengths = less similar)
        length_ratio = min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2))
        adjusted_similarity = similarity * (0.5 + 0.5 * length_ratio)

        return adjusted_similarity

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\w+', text)

        # Remove very common stop words for better similarity
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had'
        }
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

        return tokens

    def suggest_merge(
        self,
        new_content: str,
        existing_content: str,
        new_metadata: Optional[Dict] = None,
        existing_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Suggest how to merge similar content.

        Returns merged content and metadata.
        """
        # Prefer longer, more detailed content
        if len(new_content) > len(existing_content):
            primary = new_content
            secondary = existing_content
            primary_meta = new_metadata or {}
            secondary_meta = existing_metadata or {}
        else:
            primary = existing_content
            secondary = new_content
            primary_meta = existing_metadata or {}
            secondary_meta = new_metadata or {}

        # Build merged metadata
        merged_metadata = {**primary_meta}

        # Merge tags
        tags = set(primary_meta.get("tags", []))
        tags.update(secondary_meta.get("tags", []))
        merged_metadata["tags"] = list(tags)

        # Track that this was merged
        merged_metadata["merged_from"] = merged_metadata.get("merged_from", [])
        if isinstance(merged_metadata["merged_from"], list):
            merged_metadata["merged_from"].append({
                "timestamp": datetime.now().isoformat(),
                "content_preview": secondary[:100]
            })

        # Increase importance if both were important
        primary_importance = primary_meta.get("importance", 0.5)
        secondary_importance = secondary_meta.get("importance", 0.5)
        merged_metadata["importance"] = max(primary_importance, secondary_importance)

        return {
            "content": primary,
            "metadata": merged_metadata,
            "merge_reason": "Similar content detected",
            "similarity": self._calculate_similarity(new_content, existing_content)
        }

    def get_stats(self) -> Dict:
        """Get deduplication statistics."""
        return {
            "cache_size": len(self.content_cache),
            "duplicates_prevented": self.duplicates_prevented,
            "merges_suggested": self.merges_suggested,
            "duplicate_threshold": self.duplicate_threshold,
            "similar_threshold": self.similar_threshold
        }


# Global instance
_deduplicator: Optional[SemanticDeduplicator] = None


def get_deduplicator() -> SemanticDeduplicator:
    """Get global deduplicator instance."""
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = SemanticDeduplicator()
        logger.info("SemanticDeduplicator initialized")
    return _deduplicator


# Convenience functions
def check_for_duplicate(content: str) -> Optional[SimilarityMatch]:
    """Check if content is a duplicate."""
    dedup = get_deduplicator()
    return dedup.check_duplicate(content)


def add_to_dedup_cache(memory_id: str, content: str):
    """Add content to deduplication cache."""
    dedup = get_deduplicator()
    dedup.add_to_cache(memory_id, content)


def get_dedup_stats() -> Dict:
    """Get deduplication statistics."""
    dedup = get_deduplicator()
    return dedup.get_stats()
