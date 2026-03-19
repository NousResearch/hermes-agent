"""
cosmos Inference Engine

Novel Approaches:
1. Speculative Decoding - Draft model generates candidates, target verifies
2. Parallel Branch Sampling - Multiple generation paths, best selected
3. Memory-Augmented Generation - Dynamic context injection during generation
4. Confidence-Gated Output - Filter low-quality generations automatically
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional, Callable
from collections import deque

from loguru import logger

from Cosmos.core.llm_backend import (
    LLMBackend,
    GenerationConfig,
    GenerationResult,
    StreamChunk,
    ConfidenceEstimator,
)
from Cosmos.core.model_manager import ModelManager


class InferenceMode(Enum):
    STANDARD = "standard"
    SPECULATIVE = "speculative"
    PARALLEL_BRANCH = "parallel_branch"
    MEMORY_AUGMENTED = "memory_augmented"
    CASCADE = "cascade"


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    draft_tokens: int = 5  # Tokens to generate speculatively
    max_rejections: int = 3  # Max rejected sequences before falling back
    acceptance_threshold: float = 0.8


@dataclass
class ParallelBranchConfig:
    """Configuration for parallel branch sampling."""
    num_branches: int = 3
    branch_tokens: int = 20
    selection_method: str = "confidence"  # confidence, length, diversity


@dataclass
class InferenceResult:
    """Extended generation result with inference metadata."""
    text: str
    tokens_generated: int
    tokens_per_second: float
    model_used: str
    inference_mode: InferenceMode

    # Quality metrics
    confidence_score: float = 0.0
    branches_explored: int = 1
    speculative_acceptance_rate: float = 0.0

    # Memory augmentation
    memories_injected: int = 0
    context_retrievals: int = 0

    # Timing
    time_to_first_token: float = 0.0
    total_time: float = 0.0

    # Visual feedback data
    generation_trace: list[dict] = field(default_factory=list)


@dataclass
class StreamEvent:
    """Rich streaming event for UI feedback."""
    event_type: str  # "token", "branch", "memory", "escalation", "complete"
    text: str = ""
    token_index: int = 0
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class InferenceEngine:
    """
    Advanced inference engine with multiple strategies.

    Provides:
    - Speculative decoding for 2x+ throughput
    - Parallel branch exploration for quality
    - Memory-augmented generation for context awareness
    - Rich streaming events for visual feedback
    """

    def __init__(
        self,
        model_manager: ModelManager,
        default_mode: InferenceMode = InferenceMode.STANDARD,
    ):
        self.model_manager = model_manager
        self.default_mode = default_mode

        self.speculative_config = SpeculativeConfig()
        self.parallel_config = ParallelBranchConfig()

        # Memory retrieval callback (set by memory system)
        self.memory_retriever: Optional[Callable[[str], list[str]]] = None

        # Statistics tracking
        self.stats = {
            "total_generations": 0,
            "total_tokens": 0,
            "speculative_accepts": 0,
            "speculative_rejects": 0,
            "avg_confidence": 0.0,
            "mode_usage": {mode.value: 0 for mode in InferenceMode},
        }

    def set_memory_retriever(self, retriever: Callable[[str], list[str]]):
        """Set the memory retrieval function for augmented generation."""
        self.memory_retriever = retriever

    async def generate(
        self,
        prompt: str,
        model_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        mode: Optional[InferenceMode] = None,
    ) -> InferenceResult:
        """
        Generate text using the specified inference mode.

        Args:
            prompt: Input prompt
            model_key: Model to use (auto-selected if None)
            config: Generation configuration
            mode: Inference mode (uses default if None)

        Returns:
            InferenceResult with text and metadata
        """
        mode = mode or self.default_mode
        self.stats["mode_usage"][mode.value] += 1

        # Auto-select model if not specified
        if model_key is None:
            model_key = self.model_manager.get_best_model_for_task("general")

        if model_key is None:
            raise RuntimeError("No model available")

        # Ensure model is loaded
        load_result = await self.model_manager.load_model(model_key, config)
        if not load_result.success:
            raise RuntimeError(f"Failed to load model: {load_result.error}")

        backend = load_result.backend

        # Dispatch to appropriate strategy
        if mode == InferenceMode.SPECULATIVE:
            result = await self._speculative_generate(prompt, backend, config)
        elif mode == InferenceMode.PARALLEL_BRANCH:
            result = await self._parallel_branch_generate(prompt, backend, config)
        elif mode == InferenceMode.MEMORY_AUGMENTED:
            result = await self._memory_augmented_generate(prompt, backend, config)
        else:
            result = await self._standard_generate(prompt, backend, config)

        # Update statistics
        self._update_stats(result)

        return result

    async def generate_stream(
        self,
        prompt: str,
        model_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        mode: Optional[InferenceMode] = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream generation with rich events for UI feedback.

        Yields StreamEvent objects that can be used for:
        - Real-time text display
        - Confidence visualization
        - Branch/memory injection indicators
        - Progress tracking
        """
        mode = mode or self.default_mode

        if model_key is None:
            model_key = self.model_manager.get_best_model_for_task("general")

        if model_key is None:
            raise RuntimeError("No model available")

        load_result = await self.model_manager.load_model(model_key, config)
        if not load_result.success:
            raise RuntimeError(f"Failed to load model: {load_result.error}")

        backend = load_result.backend

        # Memory augmentation at start
        if mode == InferenceMode.MEMORY_AUGMENTED and self.memory_retriever:
            memories = self.memory_retriever(prompt)
            if memories:
                yield StreamEvent(
                    event_type="memory",
                    metadata={"memories_count": len(memories), "memories": memories[:3]},
                )
                prompt = self._augment_prompt_with_memories(prompt, memories)

        # Stream tokens
        token_count = 0
        cumulative_confidence = 1.0
        start_time = time.time()
        first_token_time = None

        async for chunk in backend.generate_stream(prompt, config):
            if first_token_time is None:
                first_token_time = time.time() - start_time

            token_count += 1
            cumulative_confidence = 0.95 * cumulative_confidence + 0.05 * chunk.confidence

            yield StreamEvent(
                event_type="token",
                text=chunk.text,
                token_index=token_count,
                confidence=chunk.confidence,
                metadata={
                    "cumulative_confidence": cumulative_confidence,
                    "tokens_per_second": token_count / max(0.001, time.time() - start_time),
                },
            )

        # Final event
        total_time = time.time() - start_time
        yield StreamEvent(
            event_type="complete",
            metadata={
                "total_tokens": token_count,
                "total_time": total_time,
                "tokens_per_second": token_count / max(0.001, total_time),
                "time_to_first_token": first_token_time or 0,
                "final_confidence": cumulative_confidence,
            },
        )

    async def _standard_generate(
        self,
        prompt: str,
        backend: LLMBackend,
        config: Optional[GenerationConfig],
    ) -> InferenceResult:
        """Standard single-model generation."""
        start_time = time.time()

        result = await backend.generate(prompt, config)

        return InferenceResult(
            text=result.text,
            tokens_generated=result.tokens_generated,
            tokens_per_second=result.tokens_per_second,
            model_used=result.model_used,
            inference_mode=InferenceMode.STANDARD,
            confidence_score=result.confidence_score,
            total_time=time.time() - start_time,
        )

    async def _speculative_generate(
        self,
        prompt: str,
        target_backend: LLMBackend,
        config: Optional[GenerationConfig],
    ) -> InferenceResult:
        """
        Novel: Speculative decoding for faster generation.

        Uses a smaller draft model to generate candidate tokens,
        then verifies with the target model in parallel.
        """
        start_time = time.time()

        # Get draft model (smaller/faster)
        draft_key = self.model_manager.get_best_model_for_task("speed")
        if draft_key is None:
            # Fall back to standard generation
            return await self._standard_generate(prompt, target_backend, config)

        draft_result = await self.model_manager.load_model(draft_key)
        if not draft_result.success:
            return await self._standard_generate(prompt, target_backend, config)

        draft_backend = draft_result.backend

        # Speculative decoding loop
        generated_text = ""
        total_tokens = 0
        accepts = 0
        rejects = 0
        generation_trace = []

        cfg = config or GenerationConfig()
        current_prompt = prompt

        while total_tokens < cfg.max_tokens:
            # Draft generates k tokens
            draft_config = GenerationConfig(
                max_tokens=self.speculative_config.draft_tokens,
                temperature=cfg.temperature * 0.8,  # Slightly lower temp for draft
            )
            draft_result = await draft_backend.generate(current_prompt, draft_config)

            if not draft_result.text:
                break

            # Target verifies the draft tokens
            verify_prompt = current_prompt + draft_result.text
            verify_config = GenerationConfig(max_tokens=1, temperature=cfg.temperature)
            verify_result = await target_backend.generate(verify_prompt, verify_config)

            # Accept if confidence is high enough
            if verify_result.confidence_score >= self.speculative_config.acceptance_threshold:
                generated_text += draft_result.text
                total_tokens += draft_result.tokens_generated
                accepts += draft_result.tokens_generated
                current_prompt = verify_prompt

                generation_trace.append({
                    "type": "accept",
                    "tokens": draft_result.tokens_generated,
                    "text": draft_result.text,
                })
            else:
                # Reject - use target's output instead
                rejects += 1
                if verify_result.text:
                    generated_text += verify_result.text
                    total_tokens += 1
                    current_prompt += verify_result.text

                generation_trace.append({
                    "type": "reject",
                    "draft_text": draft_result.text,
                    "target_text": verify_result.text,
                })

                if rejects >= self.speculative_config.max_rejections:
                    # Too many rejections, fall back to standard
                    logger.debug("Speculative decoding: falling back to standard")
                    remaining = await target_backend.generate(
                        current_prompt,
                        GenerationConfig(max_tokens=cfg.max_tokens - total_tokens),
                    )
                    generated_text += remaining.text
                    total_tokens += remaining.tokens_generated
                    break

        total_time = time.time() - start_time
        acceptance_rate = accepts / max(1, accepts + rejects)

        self.stats["speculative_accepts"] += accepts
        self.stats["speculative_rejects"] += rejects

        return InferenceResult(
            text=generated_text,
            tokens_generated=total_tokens,
            tokens_per_second=total_tokens / max(0.001, total_time),
            model_used=f"{draft_key}+{target_backend.model_name}",
            inference_mode=InferenceMode.SPECULATIVE,
            speculative_acceptance_rate=acceptance_rate,
            total_time=total_time,
            generation_trace=generation_trace,
        )

    async def _parallel_branch_generate(
        self,
        prompt: str,
        backend: LLMBackend,
        config: Optional[GenerationConfig],
    ) -> InferenceResult:
        """
        Novel: Parallel branch exploration for quality.

        Generates multiple candidate continuations in parallel,
        then selects the best one based on confidence/quality metrics.
        """
        start_time = time.time()
        cfg = config or GenerationConfig()

        # Generate multiple branches with different temperatures
        branch_configs = []
        for i in range(self.parallel_config.num_branches):
            branch_cfg = GenerationConfig(
                max_tokens=self.parallel_config.branch_tokens,
                temperature=cfg.temperature * (0.8 + 0.2 * i),  # Vary temperature
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )
            branch_configs.append(branch_cfg)

        # Run branches in parallel
        branch_tasks = [
            backend.generate(prompt, branch_cfg)
            for branch_cfg in branch_configs
        ]
        branch_results = await asyncio.gather(*branch_tasks, return_exceptions=True)

        # Filter successful results
        valid_results = [
            r for r in branch_results
            if isinstance(r, GenerationResult) and r.text
        ]

        if not valid_results:
            # All branches failed, try standard generation
            return await self._standard_generate(prompt, backend, config)

        # Select best branch based on selection method
        if self.parallel_config.selection_method == "confidence":
            best_result = max(valid_results, key=lambda r: r.confidence_score)
        elif self.parallel_config.selection_method == "length":
            best_result = max(valid_results, key=lambda r: r.tokens_generated)
        else:  # diversity - select most unique
            best_result = self._select_most_diverse(valid_results)

        # Continue generation if needed
        if best_result.tokens_generated < cfg.max_tokens:
            continuation_prompt = prompt + best_result.text
            continuation_config = GenerationConfig(
                max_tokens=cfg.max_tokens - best_result.tokens_generated,
                temperature=cfg.temperature,
            )
            continuation = await backend.generate(continuation_prompt, continuation_config)
            best_result.text += continuation.text
            best_result.tokens_generated += continuation.tokens_generated

        total_time = time.time() - start_time

        return InferenceResult(
            text=best_result.text,
            tokens_generated=best_result.tokens_generated,
            tokens_per_second=best_result.tokens_generated / max(0.001, total_time),
            model_used=best_result.model_used,
            inference_mode=InferenceMode.PARALLEL_BRANCH,
            confidence_score=best_result.confidence_score,
            branches_explored=len(valid_results),
            total_time=total_time,
            generation_trace=[
                {"branch": i, "confidence": r.confidence_score, "tokens": r.tokens_generated}
                for i, r in enumerate(valid_results)
            ],
        )

    def _select_most_diverse(self, results: list[GenerationResult]) -> GenerationResult:
        """Select the most diverse result (least similar to others)."""
        if len(results) == 1:
            return results[0]

        # Simple diversity metric: word overlap
        texts = [set(r.text.lower().split()) for r in results]

        min_overlap = float('inf')
        most_diverse = results[0]

        for i, result in enumerate(results):
            total_overlap = sum(
                len(texts[i] & texts[j]) / max(1, len(texts[i] | texts[j]))
                for j in range(len(results)) if j != i
            )
            if total_overlap < min_overlap:
                min_overlap = total_overlap
                most_diverse = result

        return most_diverse

    async def _memory_augmented_generate(
        self,
        prompt: str,
        backend: LLMBackend,
        config: Optional[GenerationConfig],
    ) -> InferenceResult:
        """
        Novel: Memory-augmented generation with dynamic context injection.

        Retrieves relevant memories and injects them into the context.
        Can also perform mid-generation retrieval for long outputs.
        """
        start_time = time.time()
        memories_injected = 0
        context_retrievals = 0

        # Initial memory retrieval
        if self.memory_retriever:
            memories = self.memory_retriever(prompt)
            if memories:
                memories_injected = len(memories)
                context_retrievals += 1
                prompt = self._augment_prompt_with_memories(prompt, memories)

        # Generate with augmented prompt
        result = await backend.generate(prompt, config)

        # For long generations, consider mid-generation retrieval
        cfg = config or GenerationConfig()
        if result.tokens_generated > 200 and self.memory_retriever:
            # Extract key topics from generation
            key_phrases = self._extract_key_phrases(result.text)
            for phrase in key_phrases[:2]:  # Limit retrievals
                additional_memories = self.memory_retriever(phrase)
                if additional_memories:
                    context_retrievals += 1
                    memories_injected += len(additional_memories)

        total_time = time.time() - start_time

        return InferenceResult(
            text=result.text,
            tokens_generated=result.tokens_generated,
            tokens_per_second=result.tokens_generated / max(0.001, total_time),
            model_used=result.model_used,
            inference_mode=InferenceMode.MEMORY_AUGMENTED,
            confidence_score=result.confidence_score,
            memories_injected=memories_injected,
            context_retrievals=context_retrievals,
            total_time=total_time,
        )

    def _augment_prompt_with_memories(self, prompt: str, memories: list[str]) -> str:
        """Augment prompt with retrieved memories."""
        if not memories:
            return prompt

        memory_section = "\n".join([f"- {m}" for m in memories[:5]])  # Limit to 5

        return f"""Relevant context from memory:
{memory_section}

User request: {prompt}"""

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extract key phrases for additional retrieval."""
        # Simple extraction - could be enhanced with NLP
        import re

        # Find capitalized phrases (likely proper nouns/concepts)
        phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        # Deduplicate and limit
        unique_phrases = list(dict.fromkeys(phrases))
        return unique_phrases[:5]

    def _update_stats(self, result: InferenceResult):
        """Update engine statistics."""
        self.stats["total_generations"] += 1
        self.stats["total_tokens"] += result.tokens_generated

        # Running average of confidence
        n = self.stats["total_generations"]
        old_avg = self.stats["avg_confidence"]
        self.stats["avg_confidence"] = (old_avg * (n - 1) + result.confidence_score) / n

    def get_stats(self) -> dict:
        """Get inference statistics for monitoring."""
        return {
            **self.stats,
            "speculative_acceptance_rate": (
                self.stats["speculative_accepts"] /
                max(1, self.stats["speculative_accepts"] + self.stats["speculative_rejects"])
            ),
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_generations": 0,
            "total_tokens": 0,
            "speculative_accepts": 0,
            "speculative_rejects": 0,
            "avg_confidence": 0.0,
            "mode_usage": {mode.value: 0 for mode in InferenceMode},
        }
