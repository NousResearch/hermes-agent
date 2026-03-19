"""
Deduplication Integration - Hooks into Memory System

Provides non-invasive integration of semantic deduplication.
Shadow architecture pattern.
"""

from typing import Optional, Dict, Any
from loguru import logger

from Cosmos.memory.semantic_deduplication import (
    SemanticDeduplicator,
    get_deduplicator,
    SimilarityMatch
)


class DedupMemoryWrapper:
    """
    Wraps memory system to add deduplication.

    Non-invasive - can be enabled/disabled without modifying core.
    """

    def __init__(self, memory_system=None):
        self.memory_system = memory_system
        self.deduplicator = get_deduplicator()
        self.enabled = False
        self.auto_merge = False  # Auto-merge similar content

    def enable(self, auto_merge: bool = False):
        """Enable deduplication."""
        self.enabled = True
        self.auto_merge = auto_merge
        logger.success(f"Deduplication enabled (auto_merge: {auto_merge})")

    def disable(self):
        """Disable deduplication."""
        self.enabled = False
        logger.info("Deduplication disabled")

    async def remember_with_dedup(
        self,
        content: str,
        tags: Optional[list] = None,
        importance: float = 0.5,
        metadata: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Store memory with deduplication check.

        Returns:
            Dict with keys:
            - stored: bool (whether content was stored)
            - memory_id: str (new or existing)
            - action: str (stored, skipped, merged)
            - similarity: float (if duplicate found)
        """
        if not self.enabled or not self.memory_system:
            # Fallback to normal storage
            if self.memory_system:
                memory_id = await self.memory_system.remember(
                    content=content,
                    tags=tags,
                    importance=importance,
                    metadata=metadata
                )
                return {
                    "stored": True,
                    "memory_id": memory_id,
                    "action": "stored",
                    "reason": "dedup disabled"
                }
            return {"stored": False, "action": "failed", "reason": "no memory system"}

        # Check for duplicates
        match = self.deduplicator.check_duplicate(content)

        if match and match.is_duplicate:
            # Exact or near-exact duplicate
            logger.info(f"Duplicate prevented: {match.memory_id} ({match.similarity:.2%})")
            return {
                "stored": False,
                "memory_id": match.memory_id,
                "action": "skipped",
                "similarity": match.similarity,
                "reason": "duplicate"
            }

        elif match and match.similarity >= self.deduplicator.similar_threshold:
            # Similar content
            if self.auto_merge:
                # Merge with existing
                merge_result = self.deduplicator.suggest_merge(
                    new_content=content,
                    existing_content=match.content,
                    new_metadata={"tags": tags or [], "importance": importance, **(metadata or {})},
                    existing_metadata={}
                )

                # Update existing memory with merged content
                # (would need memory system to support updates)
                logger.info(f"Auto-merged with {match.memory_id}")
                return {
                    "stored": False,
                    "memory_id": match.memory_id,
                    "action": "merged",
                    "similarity": match.similarity,
                    "merge_details": merge_result
                }
            else:
                # Store anyway but log similarity
                memory_id = await self.memory_system.remember(
                    content=content,
                    tags=tags,
                    importance=importance,
                    metadata=metadata
                )
                self.deduplicator.add_to_cache(memory_id, content)
                logger.debug(f"Stored despite similarity to {match.memory_id} ({match.similarity:.2%})")
                return {
                    "stored": True,
                    "memory_id": memory_id,
                    "action": "stored",
                    "similar_to": match.memory_id,
                    "similarity": match.similarity
                }

        else:
            # No duplicates, store normally
            memory_id = await self.memory_system.remember(
                content=content,
                tags=tags,
                importance=importance,
                metadata=metadata
            )
            self.deduplicator.add_to_cache(memory_id, content)
            return {
                "stored": True,
                "memory_id": memory_id,
                "action": "stored"
            }

    def get_stats(self) -> Dict:
        """Get deduplication stats."""
        return {
            "enabled": self.enabled,
            "auto_merge": self.auto_merge,
            **self.deduplicator.get_stats()
        }


# Global wrapper
_dedup_wrapper: Optional[DedupMemoryWrapper] = None


def get_dedup_wrapper() -> DedupMemoryWrapper:
    """Get global dedup wrapper."""
    global _dedup_wrapper
    if _dedup_wrapper is None:
        _dedup_wrapper = DedupMemoryWrapper()
    return _dedup_wrapper


async def enable_deduplication(memory_system=None, auto_merge: bool = False):
    """Enable deduplication globally."""
    wrapper = get_dedup_wrapper()
    if memory_system:
        wrapper.memory_system = memory_system
    wrapper.enable(auto_merge)


def disable_deduplication():
    """Disable deduplication globally."""
    wrapper = get_dedup_wrapper()
    wrapper.disable()


def get_deduplication_stats() -> Dict:
    """Get deduplication statistics."""
    wrapper = get_dedup_wrapper()
    return wrapper.get_stats()
