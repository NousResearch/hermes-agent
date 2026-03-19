"""
cosmos Virtual Context - MemGPT-Style Memory Paging

Novel Approaches:
1. Attention-Weighted Importance - Score memory by semantic attention
2. Predictive Prefetching - Anticipate needed memories
3. Hierarchical Paging - Multi-level memory hierarchy
4. Compression-Aware Chunking - Optimal chunk sizes for recall

Cross-platform compatible for Windows, macOS, and Linux.
"""

import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Callable
from collections import OrderedDict
import threading

from loguru import logger


# =============================================================================
# PROACTIVE CONTEXT COMPACTION DATACLASSES
# =============================================================================

@dataclass
class CompactionResult:
    """Result of context compaction operation."""
    blocks_compacted: int
    tokens_saved: int
    original_tokens: int
    final_tokens: int
    summaries_created: int
    archival_links: list[str]  # IDs of offloaded full content

    def compression_ratio(self) -> float:
        """Calculate compression ratio achieved."""
        if self.original_tokens == 0:
            return 1.0
        return self.final_tokens / self.original_tokens


@dataclass
class AllocationPlan:
    """Plan for token-aware allocation."""
    slots_to_evict: list[str]
    needs_compaction: bool
    remaining_budget: int
    can_fit: bool

# Nexus integration for overflow warnings
try:
    from Cosmos.core.nexus import nexus, SignalType
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False
    nexus = None


class MemoryTier(Enum):
    """Memory hierarchy tiers."""
    WORKING = "working"      # In-context, immediately accessible
    HOT = "hot"              # Recently used, fast retrieval
    WARM = "warm"            # Indexed, searchable
    COLD = "cold"            # Archived, compressed


@dataclass
class MemoryBlock:
    """A block of memory with metadata."""
    id: str
    content: str
    tier: MemoryTier = MemoryTier.WARM
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Importance metrics
    importance_score: float = 0.5
    attention_weight: float = 0.0
    semantic_centrality: float = 0.0

    # Relationships
    linked_blocks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Compression
    is_compressed: bool = False
    original_size: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "importance_score": self.importance_score,
            "attention_weight": self.attention_weight,
            "semantic_centrality": self.semantic_centrality,
            "linked_blocks": self.linked_blocks,
            "tags": self.tags,
            "is_compressed": self.is_compressed,
            "original_size": self.original_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryBlock":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            tier=MemoryTier(data.get("tier", "warm")),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            accessed_at=datetime.fromisoformat(data.get("accessed_at", datetime.now().isoformat())),
            access_count=data.get("access_count", 0),
            importance_score=data.get("importance_score", 0.5),
            attention_weight=data.get("attention_weight", 0.0),
            semantic_centrality=data.get("semantic_centrality", 0.0),
            linked_blocks=data.get("linked_blocks", []),
            tags=data.get("tags", []),
            is_compressed=data.get("is_compressed", False),
            original_size=data.get("original_size", 0),
        )


class ContextWindow:
    """
    Manages the active context window with intelligent paging.

    The context window represents what's currently "in view" for the LLM.
    Implements smart eviction and prefetching strategies.
    """

    def __init__(
        self,
        max_tokens: int = 1_000_000_000,
        reserve_tokens: int = 512,  # Reserved for generation
    ):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens

        self.blocks: OrderedDict[str, MemoryBlock] = OrderedDict()
        self.system_prompt: str = ""
        self.system_tokens: int = 0

        self._lock = threading.RLock()

    def set_system_prompt(self, prompt: str):
        """Set the system prompt (always in context)."""
        self.system_prompt = prompt
        self.system_tokens = self._estimate_tokens(prompt)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Average ~4 chars per token for English
        return len(text) // 4 + 1

    def get_current_tokens(self) -> int:
        """Get current token usage."""
        content_tokens = sum(
            self._estimate_tokens(block.content)
            for block in self.blocks.values()
        )
        return self.system_tokens + content_tokens

    def get_available_tokens(self) -> int:
        """Get available tokens for new content."""
        return self.available_tokens - self.get_current_tokens()

    def add_block(self, block: MemoryBlock) -> bool:
        """
        Add a memory block to the context window.

        Returns True if added successfully, False if eviction was needed.
        """
        with self._lock:
            block_tokens = self._estimate_tokens(block.content)

            # Check if we need to evict
            while self.get_available_tokens() < block_tokens and self.blocks:
                self._evict_least_important()

            if self.get_available_tokens() >= block_tokens:
                self.blocks[block.id] = block
                block.tier = MemoryTier.WORKING
                block.accessed_at = datetime.now()
                block.access_count += 1

                # v1.4: Check for context overflow warning
                self._check_overflow_warning()

                return True

            return False

    def _check_overflow_warning(self):
        """
        v1.4: Emit warning signal when context is getting full.
        Triggers when less than 20% of context remains available.
        """
        usage_ratio = self.get_current_tokens() / self.max_tokens

        if usage_ratio > 0.8:
            # Less than 20% remaining - emit warning
            if NEXUS_AVAILABLE and nexus:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._emit_overflow_warning(usage_ratio))
                except RuntimeError:
                    # No running loop, skip
                    pass

            logger.warning(
                f"Context overflow warning: {usage_ratio*100:.1f}% used "
                f"({self.get_current_tokens()}/{self.max_tokens} tokens)"
            )

    async def _emit_overflow_warning(self, usage_ratio: float):
        """Emit overflow warning via Nexus."""
        try:
            await nexus.emit(
                type=SignalType.ANOMALY_DETECTED,
                payload={
                    "event": "context_overflow_warning",
                    "usage_ratio": usage_ratio,
                    "tokens_used": self.get_current_tokens(),
                    "tokens_max": self.max_tokens,
                    "tokens_available": self.get_available_tokens(),
                    "blocks_count": len(self.blocks),
                },
                source="virtual_context",
            )
        except Exception:
            pass

    def _evict_least_important(self) -> Optional[MemoryBlock]:
        """Evict the least important block from the context."""
        if not self.blocks:
            return None

        # Score blocks for eviction (lower score = more likely to evict)
        def eviction_score(block: MemoryBlock) -> float:
            recency = (datetime.now() - block.accessed_at).total_seconds()
            recency_factor = 1.0 / (1.0 + recency / 3600)  # Decay over hours

            return (
                block.importance_score * 0.4 +
                block.attention_weight * 0.3 +
                recency_factor * 0.2 +
                min(1.0, block.access_count / 10) * 0.1
            )

        # Find block with lowest score
        min_block = min(self.blocks.values(), key=eviction_score)
        min_block.tier = MemoryTier.HOT  # Demote to hot tier

        del self.blocks[min_block.id]
        logger.debug(f"Evicted block {min_block.id} from context window")

        return min_block

    def get_context(self) -> str:
        """Get the full context for LLM input."""
        parts = [self.system_prompt] if self.system_prompt else []

        for block in self.blocks.values():
            parts.append(block.content)

        return "\n\n".join(parts)

    def get_block_ids(self) -> list[str]:
        """Get IDs of blocks currently in context."""
        return list(self.blocks.keys())

    def remove_block(self, block_id: str) -> Optional[MemoryBlock]:
        """Remove a specific block from context."""
        with self._lock:
            return self.blocks.pop(block_id, None)

    def clear(self):
        """Clear all blocks from context."""
        with self._lock:
            self.blocks.clear()

    def to_dict(self) -> dict:
        """Serialize context window state."""
        return {
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
            "system_prompt": self.system_prompt,
            "blocks": [b.to_dict() for b in self.blocks.values()],
            "current_tokens": self.get_current_tokens(),
        }

    # =========================================================================
    # PROACTIVE CONTEXT COMPACTION (AGI Upgrade 3)
    # =========================================================================

    async def compact_context(
        self,
        memory_system=None,  # Reference to MemorySystem for archival
        llm_fn: Optional[Callable] = None,  # LLM for smart summarization
        compression_strategy: str = "smart",  # "smart", "extractive", "hybrid"
        preserve_ratio: float = 0.3,
        importance_threshold: float = 0.3,
    ) -> CompactionResult:
        """
        Compact context by summarizing low-importance blocks.

        Pipeline:
        1. Identify blocks below importance threshold
        2. Summarize using LLM (or extractive fallback)
        3. Store full content in archival with link
        4. Replace in-context with summary
        5. Update token budget

        Args:
            memory_system: Reference to MemorySystem for archival offloading
            llm_fn: Async function for LLM summarization (text -> summary)
            compression_strategy: "smart" (LLM), "extractive", or "hybrid"
            preserve_ratio: Target ratio of tokens to preserve (0.3 = 30%)
            importance_threshold: Blocks below this score are candidates

        Returns:
            CompactionResult with statistics
        """
        with self._lock:
            original_tokens = self.get_current_tokens()
            blocks_compacted = 0
            summaries_created = 0
            archival_links = []

            # Find compaction candidates (low importance blocks)
            candidates = [
                (block_id, block)
                for block_id, block in self.blocks.items()
                if block.importance_score < importance_threshold
            ]

            # Sort by importance (lowest first)
            candidates.sort(key=lambda x: x[1].importance_score)

            # Calculate how many tokens we need to free
            target_tokens = int(self.max_tokens * preserve_ratio)
            tokens_to_free = max(0, self.get_current_tokens() - target_tokens)

            tokens_freed = 0

            for block_id, block in candidates:
                if tokens_freed >= tokens_to_free:
                    break

                block_tokens = self._estimate_tokens(block.content)

                # Store full content to archival if available
                if memory_system:
                    try:
                        archival_id = asyncio.get_event_loop().run_until_complete(
                            memory_system.remember(
                                content=block.content,
                                tags=block.tags + ["compacted"],
                                importance=block.importance_score,
                                metadata={"compacted_from_context": True},
                            )
                        )
                        archival_links.append(archival_id)
                    except Exception as e:
                        logger.debug(f"Failed to archive block {block_id}: {e}")

                # Generate summary
                summary = await self._summarize_block(
                    block, llm_fn, compression_strategy
                )

                if summary and len(summary) < len(block.content):
                    # Replace with summary
                    block.content = summary
                    block.is_compressed = True
                    blocks_compacted += 1
                    summaries_created += 1

                    new_tokens = self._estimate_tokens(summary)
                    tokens_freed += block_tokens - new_tokens

            final_tokens = self.get_current_tokens()

            result = CompactionResult(
                blocks_compacted=blocks_compacted,
                tokens_saved=original_tokens - final_tokens,
                original_tokens=original_tokens,
                final_tokens=final_tokens,
                summaries_created=summaries_created,
                archival_links=archival_links,
            )

            logger.info(
                f"Context compaction: {blocks_compacted} blocks, "
                f"{result.tokens_saved} tokens saved "
                f"({100 * (1 - result.compression_ratio()):.1f}% reduction)"
            )

            return result

    async def _summarize_block(
        self,
        block: "MemoryBlock",
        llm_fn: Optional[Callable],
        strategy: str,
    ) -> str:
        """Generate summary for a block using specified strategy."""
        content = block.content

        if strategy == "smart" and llm_fn:
            # Use LLM for intelligent summarization
            try:
                prompt = f"Summarize this memory concisely, preserving key information:\n\n{content}"
                if asyncio.iscoroutinefunction(llm_fn):
                    summary = await llm_fn(prompt)
                else:
                    summary = llm_fn(prompt)
                return summary
            except Exception as e:
                logger.debug(f"LLM summarization failed: {e}, falling back to extractive")

        # Extractive fallback (or explicit extractive strategy)
        if len(content) > 500:
            # Keep first sentence + key sentences
            sentences = content.split('. ')
            if len(sentences) > 3:
                # Keep first, middle, and last
                summary_parts = [
                    sentences[0],
                    sentences[len(sentences) // 2],
                    sentences[-1],
                ]
                return '. '.join(summary_parts) + '.'

        return content

    async def predictive_summarize(
        self,
        memory_system=None,
        threshold: float = 0.7,
    ) -> bool:
        """
        Proactively summarize context before hitting overflow.

        Triggers when context usage exceeds threshold percentage.

        Args:
            memory_system: Reference to MemorySystem for archival
            threshold: Usage ratio to trigger summarization (0.7 = 70%)

        Returns:
            True if summarization was triggered, False otherwise
        """
        usage_ratio = self.get_current_tokens() / self.max_tokens

        if usage_ratio < threshold:
            return False

        logger.info(
            f"Predictive summarization triggered at {usage_ratio*100:.1f}% capacity"
        )

        await self.compact_context(
            memory_system=memory_system,
            compression_strategy="extractive",  # Fast, no LLM needed
            preserve_ratio=0.5,  # Target 50% after compaction
            importance_threshold=0.4,
        )

        return True

    def _token_aware_allocation(
        self,
        required_tokens: int,
        priority: float,
    ) -> AllocationPlan:
        """
        Plan token allocation for incoming content.

        Returns a plan indicating what needs to be evicted or compacted
        to fit the new content.

        Args:
            required_tokens: Tokens needed for new content
            priority: Importance priority of new content (0-1)

        Returns:
            AllocationPlan with eviction/compaction instructions
        """
        available = self.get_available_tokens()

        if available >= required_tokens:
            # Can fit without eviction
            return AllocationPlan(
                slots_to_evict=[],
                needs_compaction=False,
                remaining_budget=available - required_tokens,
                can_fit=True,
            )

        # Need to make room
        deficit = required_tokens - available
        slots_to_evict = []
        tokens_recovered = 0

        # Find blocks to evict (lower importance than new content)
        eviction_candidates = sorted(
            [
                (block_id, block)
                for block_id, block in self.blocks.items()
                if block.importance_score < priority
            ],
            key=lambda x: x[1].importance_score,
        )

        for block_id, block in eviction_candidates:
            if tokens_recovered >= deficit:
                break
            block_tokens = self._estimate_tokens(block.content)
            slots_to_evict.append(block_id)
            tokens_recovered += block_tokens

        can_fit = tokens_recovered >= deficit

        return AllocationPlan(
            slots_to_evict=slots_to_evict,
            needs_compaction=not can_fit,
            remaining_budget=max(0, available + tokens_recovered - required_tokens),
            can_fit=can_fit,
        )


class PageManager:
    """
    Manages memory paging between tiers with intelligent caching.

    Novel features:
    - Predictive prefetching based on access patterns
    - Attention-weighted importance scoring
    - Automatic tier promotion/demotion
    """

    def __init__(
        self,
        data_dir: str = "./data/memory",
        hot_cache_size: int = 100,
        warm_cache_size: int = 1000,
    ):
        # Cross-platform path handling
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.hot_cache_size = hot_cache_size
        self.warm_cache_size = warm_cache_size

        # In-memory caches
        self.hot_cache: OrderedDict[str, MemoryBlock] = OrderedDict()
        self.warm_cache: OrderedDict[str, MemoryBlock] = OrderedDict()

        # Access pattern tracking for prefetching
        self.access_sequences: list[list[str]] = []
        self.cooccurrence: dict[str, dict[str, int]] = {}

        self._lock = asyncio.Lock()

    async def store(self, block: MemoryBlock) -> str:
        """Store a memory block, returning its ID."""
        async with self._lock:
            if not block.id:
                block.id = self._generate_id(block.content)

            block.original_size = len(block.content)

            # Determine initial tier based on size and importance
            if block.importance_score > 0.8:
                block.tier = MemoryTier.HOT
            elif len(block.content) > 2000:
                block.tier = MemoryTier.WARM
            else:
                block.tier = MemoryTier.HOT

            # Store in appropriate cache
            if block.tier == MemoryTier.HOT:
                self._add_to_hot_cache(block)
            else:
                self._add_to_warm_cache(block)

            # Persist to disk (async-friendly)
            await self._persist_block(block)

            return block.id

    async def retrieve(self, block_id: str) -> Optional[MemoryBlock]:
        """Retrieve a memory block by ID."""
        async with self._lock:
            # Check hot cache
            if block_id in self.hot_cache:
                block = self.hot_cache[block_id]
                block.accessed_at = datetime.now()
                block.access_count += 1
                self._record_access(block_id)
                return block

            # Check warm cache
            if block_id in self.warm_cache:
                block = self.warm_cache[block_id]
                block.accessed_at = datetime.now()
                block.access_count += 1

                # Promote to hot if frequently accessed
                if block.access_count > 3:
                    self._promote_to_hot(block)

                self._record_access(block_id)
                return block

            # Load from cold storage
            block = await self._load_from_disk(block_id)
            if block:
                block.accessed_at = datetime.now()
                block.access_count += 1
                self._add_to_warm_cache(block)
                self._record_access(block_id)

            return block

    async def retrieve_many(self, block_ids: list[str]) -> list[MemoryBlock]:
        """Retrieve multiple blocks efficiently."""
        blocks = []
        for block_id in block_ids:
            block = await self.retrieve(block_id)
            if block:
                blocks.append(block)
        return blocks

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{content[:100]}{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _add_to_hot_cache(self, block: MemoryBlock):
        """Add block to hot cache with eviction."""
        while len(self.hot_cache) >= self.hot_cache_size:
            # Evict oldest
            _, evicted = self.hot_cache.popitem(last=False)
            evicted.tier = MemoryTier.WARM
            self._add_to_warm_cache(evicted)

        self.hot_cache[block.id] = block
        block.tier = MemoryTier.HOT

    def _add_to_warm_cache(self, block: MemoryBlock):
        """Add block to warm cache with eviction."""
        while len(self.warm_cache) >= self.warm_cache_size:
            # Evict to cold storage
            _, evicted = self.warm_cache.popitem(last=False)
            evicted.tier = MemoryTier.COLD

        self.warm_cache[block.id] = block
        block.tier = MemoryTier.WARM

    def _promote_to_hot(self, block: MemoryBlock):
        """Promote a block from warm to hot cache."""
        if block.id in self.warm_cache:
            del self.warm_cache[block.id]
        self._add_to_hot_cache(block)

    def _record_access(self, block_id: str):
        """Record access for prefetching patterns."""
        if not self.access_sequences or len(self.access_sequences[-1]) > 10:
            self.access_sequences.append([])

        current_seq = self.access_sequences[-1]
        current_seq.append(block_id)

        # Update co-occurrence
        for prev_id in current_seq[:-1]:
            if prev_id not in self.cooccurrence:
                self.cooccurrence[prev_id] = {}
            self.cooccurrence[prev_id][block_id] = (
                self.cooccurrence[prev_id].get(block_id, 0) + 1
            )

        # Limit history
        if len(self.access_sequences) > 100:
            self.access_sequences = self.access_sequences[-50:]

    async def prefetch(self, block_id: str) -> list[MemoryBlock]:
        """
        Novel: Predictive prefetching based on access patterns.

        Returns blocks that are likely to be accessed next.
        """
        if block_id not in self.cooccurrence:
            return []

        # Get blocks with highest co-occurrence
        related = sorted(
            self.cooccurrence[block_id].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        prefetched = []
        for related_id, _ in related:
            block = await self.retrieve(related_id)
            if block:
                prefetched.append(block)

        return prefetched

    async def _persist_block(self, block: MemoryBlock):
        """Persist block to disk."""
        block_file = self.data_dir / f"{block.id}.json"

        # Use cross-platform file writing
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: block_file.write_text(json.dumps(block.to_dict()), encoding='utf-8')
        )

    async def _load_from_disk(self, block_id: str) -> Optional[MemoryBlock]:
        """Load block from disk."""
        block_file = self.data_dir / f"{block_id}.json"

        if not block_file.exists():
            return None

        try:
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                lambda: block_file.read_text(encoding='utf-8')
            )
            data = json.loads(content)
            return MemoryBlock.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load block {block_id}: {e}")
            return None

    async def update_importance(self, block_id: str, importance: float):
        """Update importance score for a block."""
        block = await self.retrieve(block_id)
        if block:
            block.importance_score = importance

            # Promote/demote based on new importance
            if importance > 0.8 and block.tier != MemoryTier.HOT:
                self._promote_to_hot(block)
            elif importance < 0.3 and block.tier == MemoryTier.HOT:
                self.hot_cache.pop(block.id, None)
                self._add_to_warm_cache(block)

            await self._persist_block(block)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "hot_cache_size": len(self.hot_cache),
            "hot_cache_max": self.hot_cache_size,
            "warm_cache_size": len(self.warm_cache),
            "warm_cache_max": self.warm_cache_size,
            "access_sequences": len(self.access_sequences),
            "cooccurrence_pairs": sum(len(v) for v in self.cooccurrence.values()),
        }


class VirtualContext:
    """
    High-level virtual context manager combining window and paging.

    Provides a unified interface for memory management with:
    - Automatic context window management
    - Transparent tier transitions
    - Attention-based importance scoring
    """

    def __init__(
        self,
        context_window_size: int = 1_000_000_000,
        data_dir: str = "./data/memory",
    ):
        self.context_window = ContextWindow(max_tokens=context_window_size)
        self.page_manager = PageManager(data_dir=data_dir)

        # Attention scoring (would be computed by LLM in full implementation)
        self.attention_weights: dict[str, float] = {}

    async def remember(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
    ) -> str:
        """
        Store new memory content.

        Returns the memory block ID.
        """
        block = MemoryBlock(
            id="",
            content=content,
            tags=tags or [],
            importance_score=importance,
        )

        block_id = await self.page_manager.store(block)

        # Try to add to context window if important enough
        if importance > 0.7:
            self.context_window.add_block(block)

        return block_id

    async def recall(self, block_id: str, add_to_context: bool = True) -> Optional[str]:
        """
        Recall a specific memory.

        Optionally adds it to the active context window.
        """
        block = await self.page_manager.retrieve(block_id)
        if not block:
            return None

        if add_to_context:
            self.context_window.add_block(block)

        return block.content

    async def page_in(self, block_ids: list[str]) -> int:
        """
        Page multiple memories into the context window.

        Returns number of blocks successfully paged in.
        """
        blocks = await self.page_manager.retrieve_many(block_ids)
        count = 0

        for block in blocks:
            if self.context_window.add_block(block):
                count += 1

        return count

    def page_out(self, block_ids: Optional[list[str]] = None):
        """
        Page memories out of the context window.

        If block_ids is None, pages out least important blocks.
        """
        if block_ids:
            for block_id in block_ids:
                self.context_window.remove_block(block_id)
        else:
            # Page out least important
            while self.context_window.get_available_tokens() < 0:
                self.context_window._evict_least_important()

    def get_context(self) -> str:
        """Get the current context for LLM input."""
        return self.context_window.get_context()

    def set_system_prompt(self, prompt: str):
        """Set the system prompt."""
        self.context_window.set_system_prompt(prompt)

    async def update_attention(self, block_id: str, attention: float):
        """
        Update attention weight for a memory block.

        Called after LLM generation to track what memories were actually used.
        """
        self.attention_weights[block_id] = attention

        # Update block importance based on attention
        await self.page_manager.update_importance(
            block_id,
            importance=0.5 + attention * 0.5,
        )

    def get_status(self) -> dict:
        """Get comprehensive status for UI."""
        return {
            "context_window": self.context_window.to_dict(),
            "page_manager": self.page_manager.get_stats(),
            "attention_weights": dict(list(self.attention_weights.items())[:10]),
        }
