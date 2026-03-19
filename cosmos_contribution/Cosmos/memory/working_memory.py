"""
cosmos Working Memory - In-Context Scratchpad

Novel Approaches:
1. Structured Scratchpad - Type-safe slots for different data types
2. Temporal Decay - Automatic relevance decay over time
3. Cross-Reference Tracking - Automatic linking between related items
4. Snapshot/Restore - Point-in-time memory states
"""

import asyncio
import copy
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar, Generic, Callable
from collections import deque

from loguru import logger


# =============================================================================
# COST-SENSITIVE BUDGET DATACLASSES
# =============================================================================

@dataclass
class BudgetStatus:
    """Token and cost budget status for working memory."""
    total_tokens_used: int
    total_tokens_budget: int
    estimated_cost_used: float
    daily_cost_limit: float
    slots_over_budget: list[str]
    recommendations: list[str]

    def is_over_budget(self) -> bool:
        """Check if we're over token or cost budget."""
        return (
            self.total_tokens_used > self.total_tokens_budget or
            self.estimated_cost_used > self.daily_cost_limit
        )


class SlotType(Enum):
    """Types of working memory slots."""
    TEXT = "text"
    CODE = "code"
    DATA = "data"
    TASK = "task"
    REFERENCE = "reference"
    SCRATCH = "scratch"


@dataclass
class WorkingMemorySlot:
    """A typed slot in working memory."""
    name: str
    slot_type: SlotType
    content: Any
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    importance: float = 0.5
    references: list[str] = field(default_factory=list)  # Other slot names
    access_count: int = 0

    def is_expired(self) -> bool:
        """Check if slot has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def get_age_seconds(self) -> float:
        """Get age of slot in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def to_dict(self) -> dict:
        """Serialize slot."""
        return {
            "name": self.name,
            "slot_type": self.slot_type.value,
            "content": self.content if isinstance(self.content, (str, int, float, bool, list, dict)) else str(self.content),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "importance": self.importance,
            "references": self.references,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkingMemorySlot":
        """Deserialize slot."""
        return cls(
            name=data["name"],
            slot_type=SlotType(data["slot_type"]),
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            importance=data.get("importance", 0.5),
            references=data.get("references", []),
            access_count=data.get("access_count", 0),
        )


@dataclass
class MemorySnapshot:
    """Point-in-time snapshot of working memory."""
    timestamp: datetime
    slots: dict[str, dict]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "slots": self.slots,
            "metadata": self.metadata,
        }


class WorkingMemory:
    """
    In-context scratchpad for active task processing.

    Provides structured storage for:
    - Current task context
    - Intermediate results
    - Active references
    - Scratch calculations

    Features:
    - Typed slots with validation
    - Automatic decay and cleanup
    - Cross-reference tracking
    - Snapshot/restore for backtracking
    """

    def __init__(
        self,
        max_slots: int = 50,
        default_ttl_minutes: int = 30,
        decay_rate: float = 0.1,
    ):
        self.max_slots = max_slots
        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        self.decay_rate = decay_rate

        self.slots: dict[str, WorkingMemorySlot] = {}
        self.snapshots: deque[MemorySnapshot] = deque(maxlen=10)
        self.history: list[dict] = []  # Operation history

        self._lock = asyncio.Lock()

    async def set(
        self,
        name: str,
        content: Any,
        slot_type: SlotType = SlotType.SCRATCH,
        importance: float = 0.5,
        ttl_minutes: Optional[int] = None,
        references: Optional[list[str]] = None,
    ) -> WorkingMemorySlot:
        """
        Set a value in working memory.

        Args:
            name: Slot name (unique identifier)
            content: Content to store
            slot_type: Type of content
            importance: Importance score (0-1)
            ttl_minutes: Time-to-live in minutes (None for default)
            references: Names of related slots

        Returns:
            The created/updated slot
        """
        async with self._lock:
            # Check capacity
            if name not in self.slots and len(self.slots) >= self.max_slots:
                await self._evict_least_important()

            expires_at = None
            if ttl_minutes is not None:
                expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
            elif self.default_ttl:
                expires_at = datetime.now() + self.default_ttl

            if name in self.slots:
                # Update existing slot
                slot = self.slots[name]
                slot.content = content
                slot.updated_at = datetime.now()
                slot.expires_at = expires_at
                slot.importance = importance
                if references:
                    slot.references = list(set(slot.references + references))
            else:
                # Create new slot
                slot = WorkingMemorySlot(
                    name=name,
                    slot_type=slot_type,
                    content=content,
                    expires_at=expires_at,
                    importance=importance,
                    references=references or [],
                )
                self.slots[name] = slot

            # Update cross-references
            self._update_references(name, references or [])

            # Record history
            self._record_operation("set", name, slot_type.value)

            return slot

    async def get(self, name: str, default: Any = None) -> Any:
        """
        Get a value from working memory.

        Returns the content (not the slot) or default if not found/expired.
        """
        async with self._lock:
            slot = self.slots.get(name)

            if slot is None:
                return default

            if slot.is_expired():
                del self.slots[name]
                return default

            # Update access tracking
            slot.access_count += 1
            slot.updated_at = datetime.now()

            # Apply decay to other slots
            self._apply_decay(exclude=name)

            return slot.content

    async def get_slot(self, name: str) -> Optional[WorkingMemorySlot]:
        """Get the full slot object (not just content)."""
        async with self._lock:
            slot = self.slots.get(name)
            if slot and not slot.is_expired():
                return slot
            return None

    async def delete(self, name: str) -> bool:
        """Delete a slot from working memory."""
        async with self._lock:
            if name in self.slots:
                # Remove from other slots' references
                for other_slot in self.slots.values():
                    if name in other_slot.references:
                        other_slot.references.remove(name)

                del self.slots[name]
                self._record_operation("delete", name)
                return True
            return False

    async def clear(self, slot_type: Optional[SlotType] = None):
        """Clear working memory, optionally by type."""
        async with self._lock:
            if slot_type is None:
                self.slots.clear()
            else:
                self.slots = {
                    k: v for k, v in self.slots.items()
                    if v.slot_type != slot_type
                }
            self._record_operation("clear", str(slot_type) if slot_type else "all")

    async def _evict_least_important(self):
        """Evict the least important slot."""
        if not self.slots:
            return

        # Score: lower = more evictable
        def eviction_score(slot: WorkingMemorySlot) -> float:
            age_factor = 1.0 / (1.0 + slot.get_age_seconds() / 600)  # Decay over 10 min
            access_factor = min(1.0, slot.access_count / 5)
            ref_factor = min(1.0, len(slot.references) / 3)

            return (
                slot.importance * 0.4 +
                age_factor * 0.3 +
                access_factor * 0.2 +
                ref_factor * 0.1
            )

        min_slot = min(self.slots.values(), key=eviction_score)
        del self.slots[min_slot.name]
        logger.debug(f"Evicted working memory slot: {min_slot.name}")

    def _update_references(self, source_name: str, target_names: list[str]):
        """Update bidirectional references."""
        for target_name in target_names:
            if target_name in self.slots:
                if source_name not in self.slots[target_name].references:
                    self.slots[target_name].references.append(source_name)

    def _apply_decay(self, exclude: Optional[str] = None):
        """Apply temporal decay to importance scores."""
        for name, slot in self.slots.items():
            if name != exclude:
                slot.importance *= (1.0 - self.decay_rate * 0.01)

    def _record_operation(self, op_type: str, slot_name: str, extra: str = ""):
        """Record operation for history."""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "operation": op_type,
            "slot": slot_name,
            "extra": extra,
        })

        # Keep bounded history
        if len(self.history) > 100:
            self.history = self.history[-50:]

    async def snapshot(self, metadata: Optional[dict] = None) -> MemorySnapshot:
        """Create a snapshot of current state."""
        async with self._lock:
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                slots={name: slot.to_dict() for name, slot in self.slots.items()},
                metadata=metadata or {},
            )
            self.snapshots.append(snapshot)
            return snapshot

    async def restore(self, snapshot: MemorySnapshot):
        """Restore working memory from a snapshot."""
        async with self._lock:
            self.slots = {
                name: WorkingMemorySlot.from_dict(data)
                for name, data in snapshot.slots.items()
            }
            self._record_operation("restore", f"snapshot_{snapshot.timestamp.isoformat()}")

    async def get_related(self, name: str, depth: int = 1) -> list[WorkingMemorySlot]:
        """Get slots related to the given slot."""
        async with self._lock:
            if name not in self.slots:
                return []

            related = set()
            to_explore = [name]
            explored = set()

            for _ in range(depth):
                next_explore = []
                for current in to_explore:
                    if current in explored:
                        continue
                    explored.add(current)

                    if current in self.slots:
                        for ref in self.slots[current].references:
                            if ref not in explored:
                                related.add(ref)
                                next_explore.append(ref)
                to_explore = next_explore

            return [self.slots[r] for r in related if r in self.slots]

    async def cleanup_expired(self) -> int:
        """Remove all expired slots."""
        async with self._lock:
            expired = [name for name, slot in self.slots.items() if slot.is_expired()]
            for name in expired:
                del self.slots[name]
            return len(expired)

    def get_by_type(self, slot_type: SlotType) -> list[WorkingMemorySlot]:
        """Get all slots of a specific type."""
        return [slot for slot in self.slots.values() if slot.slot_type == slot_type]

    def to_context_string(self, max_length: int = 2000) -> str:
        """
        Convert working memory to a string for LLM context.

        Prioritizes important and recently accessed slots.
        """
        # Sort by importance * recency
        sorted_slots = sorted(
            self.slots.values(),
            key=lambda s: s.importance * (1.0 / (1.0 + s.get_age_seconds() / 600)),
            reverse=True,
        )

        parts = ["[Working Memory]"]
        current_length = len(parts[0])

        for slot in sorted_slots:
            content_str = str(slot.content)[:500]  # Truncate long content
            slot_str = f"\n[{slot.slot_type.value}:{slot.name}] {content_str}"

            if current_length + len(slot_str) > max_length:
                break

            parts.append(slot_str)
            current_length += len(slot_str)

        return "".join(parts)

    def get_status(self) -> dict:
        """Get working memory status for UI."""
        return {
            "slot_count": len(self.slots),
            "max_slots": self.max_slots,
            "slots_by_type": {
                st.value: len([s for s in self.slots.values() if s.slot_type == st])
                for st in SlotType
            },
            "total_importance": sum(s.importance for s in self.slots.values()),
            "snapshot_count": len(self.snapshots),
            "recent_operations": self.history[-5:] if self.history else [],
        }

    # =========================================================================
    # COST-SENSITIVE TOKEN OPTIMIZATION (AGI Upgrade 4)
    # =========================================================================

    def _estimate_tokens(self, content: Any) -> int:
        """Estimate token count for content (rough approximation: ~4 chars per token)."""
        content_str = str(content)
        return len(content_str) // 4 + 1

    def _estimate_cost(self, tokens: int, model: str = "gpt-4") -> float:
        """
        Estimate cost in USD for token usage.

        Rough estimates per 1K tokens:
        - GPT-4: $0.03 input, $0.06 output
        - Claude: $0.015 input, $0.075 output
        - Gemini: $0.00025 input
        - Local/HuggingFace: $0.00
        """
        cost_per_1k = {
            "gpt-4": 0.03,
            "claude": 0.015,
            "gemini": 0.00025,
            "huggingface": 0.0,
            "local": 0.0,
            "ollama": 0.0,
            "groq": 0.0001,
        }
        rate = cost_per_1k.get(model.lower(), 0.01)
        return (tokens / 1000) * rate

    async def add_context_with_budget(
        self,
        name: str,
        content: Any,
        slot_type: SlotType = SlotType.SCRATCH,
        token_budget: Optional[int] = None,
        cost_threshold: float = 0.01,  # Max cost per operation in USD
        model_router: Optional[Callable] = None,
    ) -> WorkingMemorySlot:
        """
        Add content to working memory with budget awareness.

        If content exceeds budget or cost threshold:
        1. Route to efficient local model for compression
        2. Store compressed version
        3. Keep link to full content in archival

        Args:
            name: Slot name
            content: Content to store
            slot_type: Type of content
            token_budget: Max tokens for this slot (None for unlimited)
            cost_threshold: Max cost per operation in USD
            model_router: Optional function to route to efficient models

        Returns:
            The created/updated slot
        """
        content_tokens = self._estimate_tokens(content)

        # Check if we need to compress based on budget
        needs_compression = False
        compression_reason = ""

        if token_budget and content_tokens > token_budget:
            needs_compression = True
            compression_reason = f"Token budget exceeded ({content_tokens} > {token_budget})"

        estimated_cost = self._estimate_cost(content_tokens)
        if estimated_cost > cost_threshold:
            needs_compression = True
            compression_reason = f"Cost threshold exceeded (${estimated_cost:.4f} > ${cost_threshold})"

        final_content = content
        model_used = "none"

        if needs_compression:
            logger.debug(f"Working memory compression needed: {compression_reason}")
            final_content, model_used = await self._route_to_efficient_model(
                operation="compress",
                content=str(content),
                max_cost=cost_threshold,
                router=model_router,
            )

        # Store with compression metadata
        slot = await self.set(
            name=name,
            content=final_content,
            slot_type=slot_type,
            importance=0.5,
        )

        # Add budget metadata
        slot.budget_metadata = {
            "original_tokens": content_tokens,
            "compressed": needs_compression,
            "compression_reason": compression_reason if needs_compression else None,
            "model_used": model_used,
            "estimated_cost": self._estimate_cost(self._estimate_tokens(final_content), model_used),
        }

        return slot

    async def _route_to_efficient_model(
        self,
        operation: str,
        content: str,
        max_cost: float,
        router: Optional[Callable] = None,
    ) -> tuple[str, str]:
        """
        Route operation to most efficient model based on cost.

        Priority: HuggingFace (free) > Ollama/Groq (cheap) > Gemini/Grok > Claude

        Args:
            operation: Type of operation ("compress", "summarize", etc.)
            content: Content to process
            max_cost: Maximum acceptable cost in USD
            router: Optional external router function

        Returns:
            Tuple of (processed_content, model_used)
        """
        # Model priority list (cheapest first)
        model_priority = [
            ("huggingface", 0.0),
            ("local", 0.0),
            ("ollama", 0.0),
            ("groq", 0.0001),
            ("gemini", 0.00025),
            ("grok", 0.001),
            ("claude", 0.015),
        ]

        # If external router provided, use it
        if router:
            try:
                result = await router(operation, content, max_cost)
                if result:
                    return result
            except Exception as e:
                logger.debug(f"External router failed: {e}")

        # Fallback: Simple extractive compression (no model needed)
        if operation == "compress":
            # Keep first and last portions, summarize middle
            if len(content) > 2000:
                # Keep 30% from start, 30% from end
                keep_chars = int(len(content) * 0.3)
                compressed = (
                    content[:keep_chars] +
                    "\n\n[... content compressed ...]\n\n" +
                    content[-keep_chars:]
                )
                return (compressed, "extractive")

        return (content, "none")

    def get_budget_status(
        self,
        token_budget: int = 50000,
        daily_cost_limit: float = 1.0,
    ) -> BudgetStatus:
        """
        Get current budget consumption across all slots.

        Args:
            token_budget: Total token budget for working memory
            daily_cost_limit: Daily cost limit in USD

        Returns:
            BudgetStatus with current consumption and recommendations
        """
        total_tokens = 0
        total_cost = 0.0
        slots_over_budget = []
        recommendations = []

        for name, slot in self.slots.items():
            slot_tokens = self._estimate_tokens(slot.content)
            total_tokens += slot_tokens

            # Check for individual slot issues
            if slot_tokens > 5000:  # Individual slot is large
                slots_over_budget.append(name)

            # Estimate cost (assume default model)
            slot_cost = self._estimate_cost(slot_tokens)
            total_cost += slot_cost

        # Generate recommendations
        if total_tokens > token_budget * 0.8:
            recommendations.append(
                f"Token usage at {total_tokens}/{token_budget} ({100*total_tokens/token_budget:.1f}%). "
                "Consider compacting old slots."
            )

        if total_cost > daily_cost_limit * 0.8:
            recommendations.append(
                f"Cost at ${total_cost:.4f}/${daily_cost_limit:.2f}. "
                "Route to local models where possible."
            )

        if slots_over_budget:
            recommendations.append(
                f"Large slots: {', '.join(slots_over_budget)}. Consider compression."
            )

        if not recommendations:
            recommendations.append("Budget healthy. No action needed.")

        return BudgetStatus(
            total_tokens_used=total_tokens,
            total_tokens_budget=token_budget,
            estimated_cost_used=total_cost,
            daily_cost_limit=daily_cost_limit,
            slots_over_budget=slots_over_budget,
            recommendations=recommendations,
        )
