"""
cosmos Context Compression - Intelligent Summarization and Token Optimization

Q1 2025 Feature: Context Compression
- Intelligent summarization of retrieved context
- Priority-based context allocation
- Token-efficient memory injection
- Hierarchical context compression
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Callable
from enum import Enum
import numpy as np

from loguru import logger


class CompressionLevel(Enum):
    """Levels of context compression."""
    NONE = 0          # Full context, no compression
    LIGHT = 1         # Remove redundancy, keep details
    MODERATE = 2      # Summarize paragraphs, keep key points
    AGGRESSIVE = 3    # Extract only essential information
    EXTREME = 4       # Single sentence summaries


class ContentPriority(Enum):
    """Priority levels for content."""
    CRITICAL = 4      # Must include
    HIGH = 3          # Include if space
    MEDIUM = 2        # Include summarized
    LOW = 1           # Include only key points
    OPTIONAL = 0      # Exclude if tight


@dataclass
class ContextBlock:
    """A block of context with metadata."""
    id: str
    content: str
    source: str
    priority: ContentPriority

    # Token info
    token_count: int = 0
    compressed_content: Optional[str] = None
    compressed_token_count: int = 0

    # Relevance
    relevance_score: float = 0.5
    recency_score: float = 0.5
    importance_score: float = 0.5

    # Metadata
    timestamp: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    @property
    def effective_score(self) -> float:
        """Combined score for ranking."""
        return (
            self.relevance_score * 0.5 +
            self.importance_score * 0.3 +
            self.recency_score * 0.2
        )


@dataclass
class CompressionResult:
    """Result of context compression."""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    blocks_included: int
    blocks_excluded: int
    context: str
    block_summaries: list[dict] = field(default_factory=list)


@dataclass
class ContextBudget:
    """Token budget for context injection."""
    total_tokens: int
    system_prompt_tokens: int = 0
    conversation_tokens: int = 0
    memory_tokens: int = 0
    reserved_tokens: int = 500  # For safety margin

    @property
    def available_tokens(self) -> int:
        return max(0, self.total_tokens - self.system_prompt_tokens -
                   self.conversation_tokens - self.reserved_tokens)


class ContextCompressor:
    """
    Intelligent context compression for token-efficient memory injection.

    Features:
    - Adaptive compression based on token budget
    - Priority-based content selection
    - Hierarchical summarization
    - Redundancy elimination
    - Semantic deduplication
    """

    def __init__(
        self,
        default_budget: int = 4000,
        compression_threshold: float = 0.8,  # Compress when > 80% budget
        chars_per_token: float = 4.0,  # Rough estimate
    ):
        self.default_budget = default_budget
        self.compression_threshold = compression_threshold
        self.chars_per_token = chars_per_token

        # LLM function for summarization
        self.summarize_fn: Optional[Callable] = None

        # Embedding function for semantic dedup
        self.embed_fn: Optional[Callable] = None

        self._lock = asyncio.Lock()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 chars per token for English
        return max(1, int(len(text) / self.chars_per_token))

    async def compress_context(
        self,
        blocks: list[ContextBlock],
        budget: ContextBudget,
        compression_level: CompressionLevel = CompressionLevel.MODERATE,
        preserve_critical: bool = True,
    ) -> CompressionResult:
        """
        Compress context blocks to fit within token budget.

        Args:
            blocks: List of context blocks to compress
            budget: Token budget constraints
            compression_level: How aggressively to compress
            preserve_critical: Always include critical priority content

        Returns:
            Compression result with optimized context
        """
        async with self._lock:
            if not blocks:
                return CompressionResult(
                    original_tokens=0,
                    compressed_tokens=0,
                    compression_ratio=1.0,
                    blocks_included=0,
                    blocks_excluded=0,
                    context="",
                )

            # Calculate token counts
            for block in blocks:
                block.token_count = self.estimate_tokens(block.content)

            total_original = sum(b.token_count for b in blocks)
            available = budget.available_tokens

            # Check if compression needed
            if total_original <= available:
                # No compression needed
                context = self._assemble_context(blocks)
                return CompressionResult(
                    original_tokens=total_original,
                    compressed_tokens=total_original,
                    compression_ratio=1.0,
                    blocks_included=len(blocks),
                    blocks_excluded=0,
                    context=context,
                )

            # Remove duplicates first
            blocks = await self._remove_semantic_duplicates(blocks)

            # Sort by effective score
            blocks.sort(key=lambda b: (b.priority.value, b.effective_score), reverse=True)

            # Separate by priority
            critical_blocks = [b for b in blocks if b.priority == ContentPriority.CRITICAL]
            other_blocks = [b for b in blocks if b.priority != ContentPriority.CRITICAL]

            # Reserve space for critical blocks
            critical_tokens = sum(b.token_count for b in critical_blocks)
            remaining_budget = available - critical_tokens

            if remaining_budget < 0 and preserve_critical:
                # Need to compress even critical blocks
                critical_blocks = await self._compress_blocks(
                    critical_blocks, available, CompressionLevel.AGGRESSIVE
                )
                remaining_budget = 0
                other_blocks = []

            # Compress other blocks to fit
            included_blocks = critical_blocks.copy()
            excluded_count = 0

            for block in other_blocks:
                if remaining_budget <= 0:
                    excluded_count += 1
                    continue

                if block.token_count <= remaining_budget:
                    # Include as-is
                    included_blocks.append(block)
                    remaining_budget -= block.token_count
                elif compression_level.value > CompressionLevel.NONE.value:
                    # Try to compress
                    compressed = await self._compress_single_block(
                        block, remaining_budget, compression_level
                    )
                    if compressed and compressed.compressed_token_count <= remaining_budget:
                        included_blocks.append(compressed)
                        remaining_budget -= compressed.compressed_token_count
                    else:
                        excluded_count += 1
                else:
                    excluded_count += 1

            # Assemble final context
            context = self._assemble_context(included_blocks)
            compressed_tokens = sum(
                b.compressed_token_count if b.compressed_content else b.token_count
                for b in included_blocks
            )

            return CompressionResult(
                original_tokens=total_original,
                compressed_tokens=compressed_tokens,
                compression_ratio=compressed_tokens / total_original if total_original > 0 else 1.0,
                blocks_included=len(included_blocks),
                blocks_excluded=excluded_count,
                context=context,
                block_summaries=[
                    {
                        "id": b.id,
                        "source": b.source,
                        "original_tokens": b.token_count,
                        "compressed": b.compressed_content is not None,
                    }
                    for b in included_blocks
                ],
            )

    async def _compress_single_block(
        self,
        block: ContextBlock,
        target_tokens: int,
        level: CompressionLevel,
    ) -> Optional[ContextBlock]:
        """Compress a single block to target token count."""
        if level == CompressionLevel.LIGHT:
            # Just remove redundancy
            compressed = self._remove_redundancy(block.content)
        elif level == CompressionLevel.MODERATE:
            # Summarize
            compressed = await self._summarize_content(
                block.content,
                target_length=target_tokens * int(self.chars_per_token),
                style="concise",
            )
        elif level == CompressionLevel.AGGRESSIVE:
            # Extract key points only
            compressed = await self._extract_key_points(block.content, max_points=3)
        elif level == CompressionLevel.EXTREME:
            # Single sentence
            compressed = await self._summarize_content(
                block.content,
                target_length=100,
                style="single_sentence",
            )
        else:
            return block

        if compressed:
            block.compressed_content = compressed
            block.compressed_token_count = self.estimate_tokens(compressed)
            return block

        return None

    async def _compress_blocks(
        self,
        blocks: list[ContextBlock],
        target_tokens: int,
        level: CompressionLevel,
    ) -> list[ContextBlock]:
        """Compress multiple blocks to fit target."""
        if not blocks:
            return []

        # Distribute budget proportionally
        total_original = sum(b.token_count for b in blocks)
        compressed = []

        for block in blocks:
            block_budget = int(target_tokens * (block.token_count / total_original))
            block_budget = max(50, block_budget)  # Minimum budget

            comp_block = await self._compress_single_block(block, block_budget, level)
            if comp_block:
                compressed.append(comp_block)

        return compressed

    async def _remove_semantic_duplicates(
        self,
        blocks: list[ContextBlock],
        similarity_threshold: float = 0.9,
    ) -> list[ContextBlock]:
        """Remove semantically similar blocks."""
        if not self.embed_fn or len(blocks) <= 1:
            return blocks

        # Get embeddings
        embeddings = []
        for block in blocks:
            try:
                if asyncio.iscoroutinefunction(self.embed_fn):
                    emb = await self.embed_fn(block.content[:500])
                else:
                    emb = self.embed_fn(block.content[:500])
                embeddings.append(emb)
            except Exception:
                embeddings.append(None)

        # Find duplicates
        unique_blocks = []
        unique_indices = []
        index_to_position: dict[int, int] = {}  # Maps original index to position in unique_blocks

        for i, block in enumerate(blocks):
            if embeddings[i] is None:
                index_to_position[i] = len(unique_blocks)
                unique_blocks.append(block)
                unique_indices.append(i)
                continue

            is_duplicate = False
            for j in unique_indices:
                if embeddings[j] is None:
                    continue

                similarity = self._cosine_similarity(
                    np.array(embeddings[i]),
                    np.array(embeddings[j])
                )

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    # Keep the one with higher score
                    if block.effective_score > blocks[j].effective_score:
                        position = index_to_position.get(j)
                        if position is not None:
                            unique_blocks[position] = block
                    break

            if not is_duplicate:
                index_to_position[i] = len(unique_blocks)
                unique_blocks.append(block)
                unique_indices.append(i)

        return unique_blocks

    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant content from text."""
        # Remove repeated phrases
        sentences = re.split(r'[.!?]+', text)
        seen = set()
        unique_sentences = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Normalize for comparison
            normalized = re.sub(r'\s+', ' ', sent.lower())
            if normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(sent)

        # Remove filler phrases
        result = '. '.join(unique_sentences)
        filler_patterns = [
            r'\b(basically|essentially|actually|literally|really|just|very)\b',
            r'\b(in terms of|with respect to|in order to)\b',
            r'\b(it is worth noting that|it should be noted that)\b',
        ]

        for pattern in filler_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        # Clean up whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    async def _summarize_content(
        self,
        content: str,
        target_length: int,
        style: str = "concise",
    ) -> Optional[str]:
        """Summarize content using LLM."""
        if not self.summarize_fn:
            # Fallback: truncate intelligently
            return self._truncate_intelligently(content, target_length)

        style_prompts = {
            "concise": "Summarize concisely, keeping key information:",
            "single_sentence": "Summarize in exactly one sentence:",
            "bullet_points": "Summarize as bullet points:",
        }

        prompt = f"""{style_prompts.get(style, style_prompts['concise'])}

{content[:2000]}

Keep the summary under {target_length // 4} words."""

        try:
            if asyncio.iscoroutinefunction(self.summarize_fn):
                result = await self.summarize_fn(prompt)
            else:
                result = self.summarize_fn(prompt)

            return result.strip() if result else None
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return self._truncate_intelligently(content, target_length)

    async def _extract_key_points(
        self,
        content: str,
        max_points: int = 5,
    ) -> Optional[str]:
        """Extract key points from content."""
        if not self.summarize_fn:
            # Fallback: extract first sentences
            sentences = re.split(r'[.!?]+', content)
            key_sentences = [s.strip() for s in sentences[:max_points] if s.strip()]
            return '. '.join(key_sentences) + '.' if key_sentences else None

        prompt = f"""Extract the {max_points} most important points from this text:

{content[:2000]}

Return only the key points, one per line."""

        try:
            if asyncio.iscoroutinefunction(self.summarize_fn):
                result = await self.summarize_fn(prompt)
            else:
                result = self.summarize_fn(prompt)

            return result.strip() if result else None
        except Exception as e:
            logger.error(f"Key point extraction failed: {e}")
            return None

    def _truncate_intelligently(self, text: str, target_length: int) -> str:
        """Truncate text at sentence boundaries."""
        if len(text) <= target_length:
            return text

        # Find sentence boundaries
        sentences = re.split(r'([.!?]+)', text)

        result = ""
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            if len(result) + len(sentence) <= target_length:
                result += sentence
            else:
                break

        if not result:
            # Fall back to word boundary
            words = text[:target_length].rsplit(' ', 1)[0]
            result = words + "..."

        return result

    def _assemble_context(self, blocks: list[ContextBlock]) -> str:
        """Assemble blocks into final context string."""
        parts = []

        for block in blocks:
            content = block.compressed_content or block.content

            # Add source attribution
            header = f"[{block.source}]"
            if block.timestamp:
                header += f" ({block.timestamp.strftime('%Y-%m-%d')})"

            parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(parts)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def optimize_for_query(
        self,
        blocks: list[ContextBlock],
        query: str,
        budget: ContextBudget,
    ) -> CompressionResult:
        """
        Optimize context specifically for a query.

        Re-ranks blocks based on query relevance before compression.
        """
        if self.embed_fn:
            # Get query embedding
            try:
                if asyncio.iscoroutinefunction(self.embed_fn):
                    query_emb = await self.embed_fn(query)
                else:
                    query_emb = self.embed_fn(query)

                query_vec = np.array(query_emb)

                # Update relevance scores
                for block in blocks:
                    try:
                        if asyncio.iscoroutinefunction(self.embed_fn):
                            block_emb = await self.embed_fn(block.content[:500])
                        else:
                            block_emb = self.embed_fn(block.content[:500])

                        if block_emb is not None:
                            block.relevance_score = self._cosine_similarity(
                                query_vec, np.array(block_emb)
                            )
                    except Exception as block_err:
                        logger.debug(f"Failed to embed block {block.id}: {block_err}")
                        # Keep default relevance_score
            except Exception as e:
                logger.error(f"Query optimization failed: {e}")

        return await self.compress_context(
            blocks, budget, CompressionLevel.MODERATE
        )

    def create_budget(
        self,
        total_tokens: int,
        system_prompt: Optional[str] = None,
        conversation: Optional[str] = None,
    ) -> ContextBudget:
        """Create a context budget from current state."""
        budget = ContextBudget(total_tokens=total_tokens)

        if system_prompt:
            budget.system_prompt_tokens = self.estimate_tokens(system_prompt)

        if conversation:
            budget.conversation_tokens = self.estimate_tokens(conversation)

        return budget

    def get_stats(self) -> dict:
        """Get compressor statistics."""
        return {
            "default_budget": self.default_budget,
            "compression_threshold": self.compression_threshold,
            "chars_per_token": self.chars_per_token,
            "has_summarize_fn": self.summarize_fn is not None,
            "has_embed_fn": self.embed_fn is not None,
        }
