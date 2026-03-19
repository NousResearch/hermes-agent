"""
cosmos Token Saver - Claude/API Token Optimization

"Entropy conservation — recycle tokens, preserve coherence."

This module provides intelligent token management to reduce API costs:
1. Context Window Compression - Summarize old context
2. Memory-Augmented Injection - Use memories instead of raw history
3. Swarm-Assisted Compression - Use local models to compress
4. Smart Truncation - Priority-based context pruning
5. Response Caching - Cache common responses
6. Token Budget Tracking - Monitor and alert on usage
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import   Optional, Callable
from pathlib import Path
from collections import OrderedDict

from loguru import logger


@dataclass
class TokenBudget:
    """Token budget configuration and tracking."""
    max_input_tokens: int = 100000  # Claude's context window
    max_output_tokens: int = 4096
    daily_budget: int = 500000  # Daily token limit
    warning_threshold: float = 0.8  # Warn at 80% usage

    # Tracking
    tokens_used_today: int = 0
    last_reset: str = field(default_factory=lambda: datetime.now().date().isoformat())

    def check_and_reset(self):
        """Reset daily counter if new day."""
        today = datetime.now().date().isoformat()
        if today != self.last_reset:
            self.tokens_used_today = 0
            self.last_reset = today

    def add_usage(self, tokens: int):
        """Track token usage."""
        self.check_and_reset()
        self.tokens_used_today += tokens

    def remaining_today(self) -> int:
        """Get remaining daily budget."""
        self.check_and_reset()
        return max(0, self.daily_budget - self.tokens_used_today)

    def is_warning(self) -> bool:
        """Check if approaching budget limit."""
        self.check_and_reset()
        return self.tokens_used_today >= self.daily_budget * self.warning_threshold


@dataclass
class CompressedContext:
    """A compressed version of conversation context."""
    original_tokens: int
    compressed_tokens: int
    summary: str
    key_points: list[str]
    entities: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


class ResponseCache:
    """LRU cache for common responses."""

    def __init__(self, max_size: int = 100, ttl_hours: float = 24.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self._cache: Ordereddict[str, tuple[str, float]] = Orderedany()

    def _make_key(self, prompt: str) -> str:
        """Create cache key from prompt."""
        return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()[:16]

    def get(self, prompt: str) -> Optional[str]:
        """Get cached response if exists and not expired."""
        key = self._make_key(prompt)
        if key in self._cache:
            response, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self.ttl_seconds:
                # Move to end (LRU)
                self._cache.move_to_end(key)
                logger.debug(f"TokenSaver: Cache HIT for {key[:8]}...")
                return response
            else:
                # Expired
                del self._cache[key]
        return None

    def set(self, prompt: str, response: str):
        """Cache a response."""
        key = self._make_key(prompt)
        self._cache[key] = (response, datetime.now().timestamp())
        self._cache.move_to_end(key)

        # Evict oldest if over limit
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self):
        """Clear all cached responses."""
        self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "max_size": self.max_size,
            "ttl_hours": self.ttl_seconds / 3600,
        }


class ContextCompressor:
    """Compress conversation context using various strategies."""

    def __init__(self, llm_fn: Optional[Callable] = None):
        self.llm_fn = llm_fn
        self._compression_history: list[CompressedContext] = []

    async def compress(
        self,
        messages: list[dict[str, str]],
        target_tokens: int = 2000,
        strategy: str = "smart",
    ) -> CompressedContext:
        """
        Compress a list of messages to fit target token count.

        Strategies:
        - smart: Use LLM to intelligently summarize
        - extractive: Extract key sentences
        - truncate: Simple truncation with priority
        """
        original_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
        )
        original_tokens = self._estimate_tokens(original_text)

        if strategy == "smart" and self.llm_fn:
            result = await self._smart_compress(messages, target_tokens)
        elif strategy == "extractive":
            result = self._extractive_compress(messages, target_tokens)
        else:
            result = self._truncate_compress(messages, target_tokens)

        result.original_tokens = original_tokens
        self._compression_history.append(result)

        logger.info(
            f"TokenSaver: Compressed {original_tokens} -> {result.compressed_tokens} tokens "
            f"({result.compression_ratio:.1%} ratio)"
        )

        return result

    async def _smart_compress(
        self,
        messages: list[dict[str, str]],
        target_tokens: int,
    ) -> CompressedContext:
        """Use LLM to intelligently summarize conversation."""
        conversation = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')[:500]}"
            for m in messages[-20:]  # Last 20 messages
        )

        prompt = f"""Summarize this conversation in {target_tokens // 4} words or less.
Extract: 1) Key points 2) Important entities 3) Current topic

Conversation:
{conversation}

Format:
SUMMARY: <concise summary>
KEY_POINTS: <bullet points>
ENTITIES: <comma-separated list>"""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            # Parse response
            summary = ""
            key_points = []
            entities = []

            for line in response.split("\n"):
                if line.startswith("SUMMARY:"):
                    summary = line[8:].strip()
                elif line.startswith("KEY_POINTS:"):
                    points_text = line[11:].strip()
                    key_points = [p.strip("- ").strip() for p in points_text.split(",")]
                elif line.startswith("ENTITIES:"):
                    entities = [e.strip() for e in line[9:].split(",")]

            if not summary:
                summary = response[:target_tokens * 4]  # Fallback

            return CompressedContext(
                original_tokens=0,  # Set by caller
                compressed_tokens=self._estimate_tokens(summary),
                summary=summary,
                key_points=key_points[:10],
                entities=entities[:20],
            )

        except Exception as e:
            logger.warning(f"Smart compression failed: {e}")
            return self._extractive_compress(messages, target_tokens)

    def _extractive_compress(
        self,
        messages: list[dict[str, str]],
        target_tokens: int,
    ) -> CompressedContext:
        """Extract most important sentences."""
        # Score sentences by various heuristics
        all_sentences = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            # Split into sentences
            sentences = content.replace("!", ".").replace("?", ".").split(".")

            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 10:
                    continue

                # Score based on:
                score = 0
                if role == "assistant":
                    score += 2  # Assistant responses are important
                if any(w in ["important", "key", "must", "should", "remember"]):
                    score += 3
                if any(w in ["error", "bug", "fix", "issue"]):
                    score += 2
                if "?" in content and role == "user":
                    score += 1  # Questions

                all_sentences.append((score, sent, role))

        # Sort by score descending
        all_sentences.sort(key=lambda x: -x[0])

        # Take top sentences until we hit target
        summary_parts = []
        current_tokens = 0
        key_points = []

        for score, sent, role in all_sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current_tokens + sent_tokens > target_tokens:
                break
            summary_parts.append(sent)
            current_tokens += sent_tokens
            if score > 1:
                key_points.append(sent[:100])

        summary = ". ".join(summary_parts)

        return CompressedContext(
            original_tokens=0,
            compressed_tokens=current_tokens,
            summary=summary,
            key_points=key_points[:10],
            entities=[],
        )

    def _truncate_compress(
        self,
        messages: list[dict[str, str]],
        target_tokens: int,
    ) -> CompressedContext:
        """Simple priority-based truncation."""
        # Priority: recent > assistant > user
        weighted_messages = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Recency weight (recent messages get higher priority)
            recency = (i + 1) / len(messages)
            role_weight = 1.5 if role == "assistant" else 1.0

            weighted_messages.append((recency * role_weight, msg))

        # Sort by weight descending
        weighted_messages.sort(key=lambda x: -x[0])

        # Take messages until target
        kept_messages = []
        current_tokens = 0

        for weight, msg in weighted_messages:
            content = msg.get("content", "")
            msg_tokens = self._estimate_tokens(content)

            if current_tokens + msg_tokens > target_tokens:
                # Truncate this message
                remaining = target_tokens - current_tokens
                if remaining > 100:
                    truncated = content[:remaining * 4]  # Rough chars to tokens
                    kept_messages.append({**msg, "content": truncated + "..."})
                    current_tokens += self._estimate_tokens(truncated)
                break

            kept_messages.append(msg)
            current_tokens += msg_tokens

        summary = "\n".join(
            f"{m.get('role')}: {m.get('content')}"
            for m in kept_messages
        )

        return CompressedContext(
            original_tokens=0,
            compressed_tokens=current_tokens,
            summary=summary,
            key_points=[],
            entities=[],
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return len(text) // 4

    def get_compression_stats(self) -> dict:
        """Get compression statistics."""
        if not self._compression_history:
            return {"compressions": 0}

        ratios = [c.compression_ratio for c in self._compression_history]
        tokens_saved = sum(
            c.original_tokens - c.compressed_tokens
            for c in self._compression_history
        )

        return {
            "compressions": len(self._compression_history),
            "avg_ratio": sum(ratios) / len(ratios),
            "best_ratio": min(ratios),
            "tokens_saved": tokens_saved,
        }


class TokenSaver:
    """
    Main token optimization manager.

    Combines caching, compression, and budget tracking for
    optimal API token usage.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        data_dir: str = "./data",
        daily_budget: int = 500000,
    ):
        self.llm_fn = llm_fn
        self.data_dir = Path(data_dir)

        self.budget = TokenBudget(daily_budget=daily_budget)
        self.cache = ResponseCache(max_size=200, ttl_hours=48.0)
        self.compressor = ContextCompressor(llm_fn=llm_fn)

        # Settings
        self.auto_compress_threshold = 50000  # Compress when context > this
        self.memory_injection_enabled = True
        self.caching_enabled = True

        self._load_state()

    def _load_state(self):
        """Load budget state from disk."""
        state_file = self.data_dir / "token_saver_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                self.budget.tokens_used_today = data.get("tokens_used_today", 0)
                self.budget.last_reset = data.get("last_reset", datetime.now().date().isoformat())
            except Exception:
                pass

    def _save_state(self):
        """Save budget state to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        state_file = self.data_dir / "token_saver_state.json"
        try:
            with open(state_file, "w") as f:
                json.dump({
                    "tokens_used_today": self.budget.tokens_used_today,
                    "last_reset": self.budget.last_reset,
                }, f)
        except Exception:
            pass

    async def optimize_request(
        self,
        prompt: str,
        messages: Optional[list[dict[str, str]]] = None,
        memory_context: Optional[str] = None,
    ) -> dict:
        """
        Optimize an API request to reduce token usage.

        Returns optimized prompt and dict cached response.
        """
        result = {
            "original_tokens": self._estimate_tokens(prompt),
            "optimized_prompt": prompt,
            "cached_response": None,
            "context_compressed": False,
            "tokens_saved": 0,
        }

        # 1. Check cache
        if self.caching_enabled:
            cached = self.cache.get(prompt)
            if cached:
                result["cached_response"] = cached
                result["tokens_saved"] = result["original_tokens"]
                return result

        # 2. Inject memory context if available
        if self.memory_injection_enabled and memory_context:
            # Replace verbose context with memory summary
            memory_injection = f"\n[Context from memory: {memory_context[:500]}]\n"
            result["optimized_prompt"] = memory_injection + prompt

        # 3. Compress conversation if too long
        if messages and len(messages) > 5:
            total_tokens = sum(
                self._estimate_tokens(m.get("content", ""))
                for m in messages
            )

            if total_tokens > self.auto_compress_threshold:
                compressed = await self.compressor.compress(
                    messages,
                    target_tokens=self.auto_compress_threshold // 2,
                    strategy="smart" if self.llm_fn else "extractive",
                )
                result["context_compressed"] = True
                result["compressed_context"] = compressed.summary
                result["tokens_saved"] += compressed.original_tokens - compressed.compressed_tokens

        result["final_tokens"] = self._estimate_tokens(result["optimized_prompt"])
        return result

    def record_response(self, prompt: str, response: str, tokens_used: int):
        """Record a response for caching and budget tracking."""
        # Update budget
        self.budget.add_usage(tokens_used)
        self._save_state()

        # Cache response
        if self.caching_enabled:
            self.cache.set(prompt, response)

        # Check budget warning
        if self.budget.is_warning():
            logger.warning(
                f"TokenSaver: Approaching daily budget! "
                f"Used: {self.budget.tokens_used_today:,} / {self.budget.daily_budget:,}"
            )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        return len(text) // 4

    def get_status(self) -> dict:
        """Get comprehensive token saver status."""
        return {
            "budget": {
                "daily_limit": self.budget.daily_budget,
                "used_today": self.budget.tokens_used_today,
                "remaining": self.budget.remaining_today(),
                "warning": self.budget.is_warning(),
            },
            "cache": self.cache.stats(),
            "compression": self.compressor.get_compression_stats(),
            "settings": {
                "auto_compress_threshold": self.auto_compress_threshold,
                "memory_injection": self.memory_injection_enabled,
                "caching": self.caching_enabled,
            },
        }

    def set_daily_budget(self, budget: int):
        """Update daily token budget."""
        self.budget.daily_budget = budget
        self._save_state()

    def clear_cache(self):
        """Clear response cache."""
        self.cache.clear()


# Global instance
token_saver = TokenSaver()


# Convenience functions
async def optimize_prompt(prompt: str, messages: list[dict] = None) -> str:
    """Quick optimization wrapper."""
    result = await token_saver.optimize_request(prompt, messages)
    return result.get("optimized_prompt", prompt)


def get_token_status() -> dict:
    """Get current token usage status."""
    return token_saver.get_status()
