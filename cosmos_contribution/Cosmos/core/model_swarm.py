"""
cosmos Model Swarm - Collaborative Multi-Model Inference

Novel Approaches:
1. PSO-Based Model Swarm - Particle Swarm Optimization for model collaboration
2. Ensemble Voting - Multiple models vote on best response
3. Mixture of Experts Routing - Dynamic routing to specialized models
4. Speculative Ensemble - Draft with fast models, verify with strong
5. Confidence-Weighted Fusion - Combine outputs weighted by confidence
6. Adaptive Load Balancing - Distribute work based on model availability

Research Sources:
- Model Swarms: Collaborative Search to Adapt LLM Experts (arXiv:2410.11163)
- LLM Ensemble Survey (arXiv:2502.18036)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable
import random
import math
import json
from collections import defaultdict

from loguru import logger


class SwarmStrategy(Enum):
    """Model swarm inference strategies."""
    FASTEST_FIRST = "fastest_first"  # Start with fastest, escalate if needed
    QUALITY_FIRST = "quality_first"  # Start with best, fall back if slow
    PARALLEL_VOTE = "parallel_vote"  # Run all, vote on best
    MIXTURE_OF_EXPERTS = "moe"  # Route to best expert per query
    SPECULATIVE_ENSEMBLE = "speculative"  # Draft + verify
    CONFIDENCE_FUSION = "fusion"  # Weighted combination
    PSO_COLLABORATIVE = "pso"  # Particle swarm optimization


class ModelRole(Enum):
    """Specialized roles for models in swarm."""
    GENERALIST = "generalist"
    REASONING = "reasoning"
    CODING = "coding"
    CREATIVE = "creative"
    MATH = "math"
    MULTILINGUAL = "multilingual"
    SPEED = "speed"
    VERIFIER = "verifier"


@dataclass
class ModelParticle:
    """
    A model treated as a particle in PSO swarm.

    Each particle has:
    - Position: Current model configuration/weights
    - Velocity: Direction of adaptation
    - Best position: Best configuration found
    """
    model_id: str
    model_name: str
    role: ModelRole

    # Performance metrics
    success_rate: float = 0.5
    avg_latency: float = 1.0
    avg_confidence: float = 0.7

    # PSO state
    position: list[float] = field(default_factory=list)  # Abstract position in solution space
    velocity: list[float] = field(default_factory=list)
    personal_best_position: list[float] = field(default_factory=list)
    personal_best_score: float = 0.0

    # Resource requirements
    vram_gb: float = 2.0
    ram_gb: float = 4.0

    # Statistics
    total_requests: int = 0
    successful_requests: int = 0
    total_tokens: int = 0

    def fitness(self) -> float:
        """Calculate particle fitness score."""
        # Multi-objective: quality * speed * efficiency
        quality = self.success_rate * self.avg_confidence
        speed = 1.0 / max(0.1, self.avg_latency)
        efficiency = 1.0 / max(0.1, self.vram_gb)

        return quality * 0.5 + speed * 0.3 + efficiency * 0.2

    def update_stats(self, success: bool, latency: float, confidence: float):
        """Update running statistics."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1

        # Exponential moving average
        alpha = 0.1
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        self.avg_latency = alpha * latency + (1 - alpha) * self.avg_latency
        self.avg_confidence = alpha * confidence + (1 - alpha) * self.avg_confidence

        # Update personal best
        current_fitness = self.fitness()
        if current_fitness > self.personal_best_score:
            self.personal_best_score = current_fitness
            self.personal_best_position = self.position.copy()


@dataclass
class SwarmResponse:
    """Response from model swarm."""
    text: str
    model_id: str
    model_name: str
    strategy_used: SwarmStrategy

    # Quality metrics
    confidence: float = 0.0
    agreement_score: float = 0.0  # How many models agreed

    # Performance
    latency: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0

    # Ensemble details
    num_models_used: int = 1
    model_contributions: dict = field(default_factory=dict)

    # Verification
    verified: bool = False
    verifier_model: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model_id": self.model_id,
            "strategy": self.strategy_used.value,
            "confidence": self.confidence,
            "latency": self.latency,
            "tokens_per_second": self.tokens_per_second,
            "num_models": self.num_models_used,
            "verified": self.verified,
        }


@dataclass
class QueryAnalysis:
    """Analysis of incoming query for routing."""
    query: str
    detected_task: str = "general"
    complexity: float = 0.5
    requires_reasoning: bool = False
    requires_code: bool = False
    requires_math: bool = False
    requires_creativity: bool = False
    is_multilingual: bool = False
    estimated_tokens: int = 100

    def best_role(self) -> ModelRole:
        """Determine best model role for this query."""
        if self.requires_math:
            return ModelRole.MATH
        elif self.requires_code:
            return ModelRole.CODING
        elif self.requires_reasoning and self.complexity > 0.7:
            return ModelRole.REASONING
        elif self.requires_creativity:
            return ModelRole.CREATIVE
        elif self.is_multilingual:
            return ModelRole.MULTILINGUAL
        elif self.complexity < 0.3:
            return ModelRole.SPEED
        return ModelRole.GENERALIST


class QueryAnalyzer:
    """Analyze queries to determine optimal routing."""

    CODE_PATTERNS = [
        r'\b(code|function|class|implement|debug|fix|program|script)\b',
        r'\b(python|javascript|typescript|java|rust|go|cpp|sql)\b',
        r'\b(api|database|server|client|frontend|backend)\b',
        r'```',
    ]

    MATH_PATTERNS = [
        r'\b(calculate|compute|solve|equation|formula|derivative|integral)\b',
        r'\b(probability|statistics|matrix|vector|algebra)\b',
        r'[\d\+\-\*\/\=\^]',
        r'\b(proof|theorem|lemma)\b',
    ]

    REASONING_PATTERNS = [
        r'\b(why|how|explain|analyze|compare|evaluate)\b',
        r'\b(step by step|think through|reason|logic)\b',
        r'\b(cause|effect|consequence|implication)\b',
    ]

    CREATIVE_PATTERNS = [
        r'\b(write|create|compose|generate|imagine|story|poem)\b',
        r'\b(creative|artistic|narrative|fiction)\b',
    ]

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze query for optimal routing."""
        import re

        analysis = QueryAnalysis(query=query)

        # Check patterns
        for pattern in self.CODE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                analysis.requires_code = True
                analysis.detected_task = "coding"
                break

        for pattern in self.MATH_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                analysis.requires_math = True
                analysis.detected_task = "math"
                break

        for pattern in self.REASONING_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                analysis.requires_reasoning = True
                if analysis.detected_task == "general":
                    analysis.detected_task = "reasoning"
                break

        for pattern in self.CREATIVE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                analysis.requires_creativity = True
                analysis.detected_task = "creative"
                break

        # Estimate complexity
        words = len(query.split())
        analysis.complexity = min(1.0, words / 200 + (0.3 if analysis.requires_reasoning else 0))

        # Estimate output tokens
        if analysis.requires_code:
            analysis.estimated_tokens = 500
        elif analysis.requires_creativity:
            analysis.estimated_tokens = 300
        else:
            analysis.estimated_tokens = max(50, words * 2)

        # Check for non-English
        try:
            # Simple heuristic: high ratio of non-ASCII chars
            non_ascii = sum(1 for c in query if ord(c) > 127)
            if non_ascii / max(1, len(query)) > 0.3:
                analysis.is_multilingual = True
        except Exception as e:
            logger.debug(f"Multilingual detection skipped: {e}")

        return analysis


class ModelSwarm:
    """
    Collaborative model swarm for ensemble inference.

    Features:
    - PSO-based collaborative optimization
    - Multiple inference strategies
    - Dynamic model selection
    - Confidence-weighted fusion
    - Verification loops
    - CST: Relativistic speed constraints for geodesic movement
    """

    # CST Speed-of-Light Constant
    # Particles cannot exceed this velocity in solution space
    # This enforces geodesic (straight-line) movement toward solutions
    C_CONSTANT = 1.0  # Normalized speed limit

    def __init__(
        self,
        models: Optional[list[dict]] = None,
        default_strategy: SwarmStrategy = SwarmStrategy.MIXTURE_OF_EXPERTS,
        pso_inertia: float = 0.7,
        pso_cognitive: float = 1.5,
        pso_social: float = 1.5,
        enable_verification: bool = True,
    ):
        self.default_strategy = default_strategy
        self.pso_inertia = pso_inertia
        self.pso_cognitive = pso_cognitive
        self.pso_social = pso_social
        self.enable_verification = enable_verification

        # Model particles
        self.particles: dict[str, ModelParticle] = {}

        # Global best (PSO)
        self.global_best_position: list[float] = []
        self.global_best_score: float = 0.0
        self.global_best_model: Optional[str] = None

        # Query analyzer
        self.analyzer = QueryAnalyzer()

        # Backend references (set by external code)
        self.backends: dict = {}

        # Statistics
        self.total_queries = 0
        self.strategy_stats: dict = defaultdict(lambda: {"count": 0, "success": 0, "latency": []})

        # Initialize with default models if provided
        if models:
            for model_config in models:
                self.register_model(model_config)

    def register_model(
        self,
        config: dict,
        backend: Optional[dict] = None,
    ) -> ModelParticle:
        """Register a model in the swarm."""
        model_id = config.get("id", config.get("name", f"model_{len(self.particles)}"))

        # Determine role from strengths
        strengths = config.get("strengths", [])
        role = ModelRole.GENERALIST
        if "reasoning" in strengths or "math" in strengths:
            role = ModelRole.REASONING
        elif "code" in strengths:
            role = ModelRole.CODING
        elif "creative" in strengths:
            role = ModelRole.CREATIVE
        elif "speed" in strengths or "fast" in strengths:
            role = ModelRole.SPEED
        elif "multilingual" in strengths:
            role = ModelRole.MULTILINGUAL

        # Initialize PSO position (abstract representation)
        position_dim = 10
        position = [random.random() for _ in range(position_dim)]
        velocity = [random.uniform(-0.1, 0.1) for _ in range(position_dim)]

        particle = ModelParticle(
            model_id=model_id,
            model_name=config.get("name", model_id),
            role=role,
            position=position,
            velocity=velocity,
            personal_best_position=position.copy(),
            vram_gb=config.get("vram_gb", 2.0),
            ram_gb=config.get("ram_gb", 4.0),
        )

        self.particles[model_id] = particle

        if backend:
            self.backends[model_id] = backend

        logger.info(f"Registered model {model_id} with role {role.value}")
        return particle

    def set_backend(self, model_id: str, backend: dict):
        """Set backend for a registered model."""
        self.backends[model_id] = backend

    async def infer(
        self,
        prompt: str,
        strategy: Optional[SwarmStrategy] = None,
        max_models: int = 3,
        timeout: float = 30.0,
    ) -> SwarmResponse:
        """
        Run swarm inference with specified strategy.
        """
        strategy = strategy or self.default_strategy
        self.total_queries += 1

        # Analyze query
        analysis = self.analyzer.analyze(prompt)

        # Select strategy-specific method
        if strategy == SwarmStrategy.FASTEST_FIRST:
            response = await self._infer_fastest_first(prompt, analysis, timeout)
        elif strategy == SwarmStrategy.QUALITY_FIRST:
            response = await self._infer_quality_first(prompt, analysis, timeout)
        elif strategy == SwarmStrategy.PARALLEL_VOTE:
            response = await self._infer_parallel_vote(prompt, analysis, max_models, timeout)
        elif strategy == SwarmStrategy.MIXTURE_OF_EXPERTS:
            response = await self._infer_moe(prompt, analysis, timeout)
        elif strategy == SwarmStrategy.SPECULATIVE_ENSEMBLE:
            response = await self._infer_speculative(prompt, analysis, timeout)
        elif strategy == SwarmStrategy.CONFIDENCE_FUSION:
            response = await self._infer_fusion(prompt, analysis, max_models, timeout)
        elif strategy == SwarmStrategy.PSO_COLLABORATIVE:
            response = await self._infer_pso(prompt, analysis, max_models, timeout)
        else:
            response = await self._infer_moe(prompt, analysis, timeout)

        # Update statistics
        self.strategy_stats[strategy.value]["count"] += 1
        self.strategy_stats[strategy.value]["latency"].append(response.latency)
        if response.confidence > 0.7:
            self.strategy_stats[strategy.value]["success"] += 1

        # Verification if enabled
        if self.enable_verification and response.confidence < 0.8:
            response = await self._verify_response(prompt, response)

        return response

    async def _infer_fastest_first(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        timeout: float,
    ) -> SwarmResponse:
        """Start with fastest model, escalate if confidence low."""
        import time

        # Sort by latency
        sorted_particles = sorted(
            self.particles.values(),
            key=lambda p: p.avg_latency
        )

        for particle in sorted_particles:
            if particle.model_id not in self.backends:
                continue

            start = time.time()
            try:
                backend = self.backends[particle.model_id]
                result = await asyncio.wait_for(
                    backend.generate(prompt),
                    timeout=timeout
                )

                latency = time.time() - start
                confidence = result.confidence_score

                particle.update_stats(True, latency, confidence)

                # Check if quality is acceptable
                if confidence >= 0.7 or particle == sorted_particles[-1]:
                    return SwarmResponse(
                        text=result.text,
                        model_id=particle.model_id,
                        model_name=particle.model_name,
                        strategy_used=SwarmStrategy.FASTEST_FIRST,
                        confidence=confidence,
                        latency=latency,
                        tokens_generated=result.tokens_generated,
                        tokens_per_second=result.tokens_per_second,
                    )

                # Low confidence - escalate
                logger.info(f"Escalating from {particle.model_id} (confidence: {confidence:.2f})")

            except asyncio.TimeoutError:
                particle.update_stats(False, timeout, 0.0)
                continue
            except Exception as e:
                logger.warning(f"Model {particle.model_id} failed: {e}")
                particle.update_stats(False, timeout, 0.0)
                continue

        return SwarmResponse(
            text="Error: All models failed",
            model_id="none",
            model_name="none",
            strategy_used=SwarmStrategy.FASTEST_FIRST,
            confidence=0.0,
        )

    async def _infer_quality_first(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        timeout: float,
    ) -> SwarmResponse:
        """Start with highest quality model."""
        import time

        # Sort by fitness (quality)
        sorted_particles = sorted(
            self.particles.values(),
            key=lambda p: p.fitness(),
            reverse=True
        )

        for particle in sorted_particles:
            if particle.model_id not in self.backends:
                continue

            start = time.time()
            try:
                backend = self.backends[particle.model_id]
                result = await asyncio.wait_for(
                    backend.generate(prompt),
                    timeout=timeout
                )

                latency = time.time() - start
                confidence = result.confidence_score

                particle.update_stats(True, latency, confidence)

                return SwarmResponse(
                    text=result.text,
                    model_id=particle.model_id,
                    model_name=particle.model_name,
                    strategy_used=SwarmStrategy.QUALITY_FIRST,
                    confidence=confidence,
                    latency=latency,
                    tokens_generated=result.tokens_generated,
                    tokens_per_second=result.tokens_per_second,
                )

            except asyncio.TimeoutError:
                particle.update_stats(False, timeout, 0.0)
                continue
            except Exception as e:
                logger.warning(f"Model {particle.model_id} failed: {e}")
                continue

        return SwarmResponse(
            text="Error: All models failed",
            model_id="none",
            model_name="none",
            strategy_used=SwarmStrategy.QUALITY_FIRST,
            confidence=0.0,
        )

    async def _infer_parallel_vote(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        max_models: int,
        timeout: float,
    ) -> SwarmResponse:
        """Run multiple models in parallel, vote on best response."""
        import time

        # Select top models
        sorted_particles = sorted(
            self.particles.values(),
            key=lambda p: p.fitness(),
            reverse=True
        )[:max_models]

        available = [p for p in sorted_particles if p.model_id in self.backends]

        if not available:
            return SwarmResponse(
                text="Error: No models available",
                model_id="none",
                model_name="none",
                strategy_used=SwarmStrategy.PARALLEL_VOTE,
                confidence=0.0,
            )

        start = time.time()

        # Run all models in parallel
        async def run_model(particle):
            try:
                backend = self.backends[particle.model_id]
                result = await asyncio.wait_for(
                    backend.generate(prompt),
                    timeout=timeout
                )
                return (particle, result)
            except Exception as e:
                logger.warning(f"Model {particle.model_id} failed: {e}")
                return (particle, None)

        results = await asyncio.gather(*[run_model(p) for p in available])

        # Filter successful results
        successful = [(p, r) for p, r in results if r is not None]

        if not successful:
            return SwarmResponse(
                text="Error: All models failed",
                model_id="none",
                model_name="none",
                strategy_used=SwarmStrategy.PARALLEL_VOTE,
                confidence=0.0,
            )

        # Vote: select response with highest confidence
        best_particle, best_result = max(successful, key=lambda x: x[1].confidence_score)

        # Calculate agreement (simplified: just confidence average)
        avg_confidence = sum(r.confidence_score for _, r in successful) / len(successful)

        latency = time.time() - start

        # Update stats for all participants
        for particle, result in successful:
            particle.update_stats(True, latency, result.confidence_score)

        return SwarmResponse(
            text=best_result.text,
            model_id=best_particle.model_id,
            model_name=best_particle.model_name,
            strategy_used=SwarmStrategy.PARALLEL_VOTE,
            confidence=best_result.confidence_score,
            agreement_score=avg_confidence,
            latency=latency,
            tokens_generated=best_result.tokens_generated,
            tokens_per_second=best_result.tokens_per_second,
            num_models_used=len(successful),
            model_contributions={p.model_id: r.confidence_score for p, r in successful},
        )

    async def _infer_moe(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        timeout: float,
    ) -> SwarmResponse:
        """Mixture of Experts - route to best specialist."""
        import time

        target_role = analysis.best_role()

        # Find best model for role
        candidates = [
            p for p in self.particles.values()
            if p.role == target_role and p.model_id in self.backends
        ]

        # Fallback to generalists
        if not candidates:
            candidates = [
                p for p in self.particles.values()
                if p.model_id in self.backends
            ]

        if not candidates:
            return SwarmResponse(
                text="Error: No models available",
                model_id="none",
                model_name="none",
                strategy_used=SwarmStrategy.MIXTURE_OF_EXPERTS,
                confidence=0.0,
            )

        # Select best candidate by fitness
        particle = max(candidates, key=lambda p: p.fitness())

        start = time.time()
        try:
            backend = self.backends[particle.model_id]
            result = await asyncio.wait_for(
                backend.generate(prompt),
                timeout=timeout
            )

            latency = time.time() - start
            particle.update_stats(True, latency, result.confidence_score)

            return SwarmResponse(
                text=result.text,
                model_id=particle.model_id,
                model_name=particle.model_name,
                strategy_used=SwarmStrategy.MIXTURE_OF_EXPERTS,
                confidence=result.confidence_score,
                latency=latency,
                tokens_generated=result.tokens_generated,
                tokens_per_second=result.tokens_per_second,
            )

        except Exception as e:
            logger.error(f"MoE inference failed: {e}")
            return SwarmResponse(
                text=f"Error: {e}",
                model_id=particle.model_id,
                model_name=particle.model_name,
                strategy_used=SwarmStrategy.MIXTURE_OF_EXPERTS,
                confidence=0.0,
            )

    async def _infer_speculative(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        timeout: float,
    ) -> SwarmResponse:
        """Speculative decoding: draft with fast model, verify with strong."""
        import time

        # Find fastest and strongest models
        available = [p for p in self.particles.values() if p.model_id in self.backends]

        if len(available) < 2:
            return await self._infer_moe(prompt, analysis, timeout)

        fastest = min(available, key=lambda p: p.avg_latency)
        strongest = max(available, key=lambda p: p.fitness())

        if fastest.model_id == strongest.model_id:
            return await self._infer_moe(prompt, analysis, timeout)

        start = time.time()

        # Draft generation
        try:
            draft_backend = self.backends[fastest.model_id]
            draft_result = await asyncio.wait_for(
                draft_backend.generate(prompt),
                timeout=timeout * 0.4
            )
        except Exception as e:
            logger.debug(f"Speculative draft failed, falling back to MoE: {e}")
            return await self._infer_moe(prompt, analysis, timeout)

        # If draft is high confidence, use it
        if draft_result.confidence_score > 0.85:
            latency = time.time() - start
            fastest.update_stats(True, latency, draft_result.confidence_score)

            return SwarmResponse(
                text=draft_result.text,
                model_id=fastest.model_id,
                model_name=fastest.model_name,
                strategy_used=SwarmStrategy.SPECULATIVE_ENSEMBLE,
                confidence=draft_result.confidence_score,
                latency=latency,
                tokens_generated=draft_result.tokens_generated,
                tokens_per_second=draft_result.tokens_per_second,
            )

        # Verify/improve with stronger model
        verify_prompt = f"""Review and improve this response if needed:

Question: {prompt}

Draft Response: {draft_result.text}

Provide the improved response:"""

        try:
            strong_backend = self.backends[strongest.model_id]
            final_result = await asyncio.wait_for(
                strong_backend.generate(verify_prompt),
                timeout=timeout * 0.6
            )

            latency = time.time() - start
            strongest.update_stats(True, latency, final_result.confidence_score)

            return SwarmResponse(
                text=final_result.text,
                model_id=strongest.model_id,
                model_name=strongest.model_name,
                strategy_used=SwarmStrategy.SPECULATIVE_ENSEMBLE,
                confidence=final_result.confidence_score,
                latency=latency,
                tokens_generated=final_result.tokens_generated,
                tokens_per_second=final_result.tokens_per_second,
                verified=True,
                verifier_model=strongest.model_name,
            )

        except Exception as e:
            # Fall back to draft
            logger.debug(f"Verification failed, using draft response: {e}")
            latency = time.time() - start
            return SwarmResponse(
                text=draft_result.text,
                model_id=fastest.model_id,
                model_name=fastest.model_name,
                strategy_used=SwarmStrategy.SPECULATIVE_ENSEMBLE,
                confidence=draft_result.confidence_score,
                latency=latency,
                tokens_generated=draft_result.tokens_generated,
                tokens_per_second=draft_result.tokens_per_second,
            )

    async def _infer_fusion(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        max_models: int,
        timeout: float,
    ) -> SwarmResponse:
        """Confidence-weighted fusion of multiple responses."""
        import time

        # Get parallel responses
        vote_response = await self._infer_parallel_vote(prompt, analysis, max_models, timeout)

        if vote_response.num_models_used <= 1:
            vote_response.strategy_used = SwarmStrategy.CONFIDENCE_FUSION
            return vote_response

        # For true fusion, we'd need to merge responses semantically
        # Simplified: just return the voted best with fusion metadata
        vote_response.strategy_used = SwarmStrategy.CONFIDENCE_FUSION
        return vote_response

    async def _infer_pso(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        max_models: int,
        timeout: float,
    ) -> SwarmResponse:
        """
        PSO-based collaborative inference.

        Models "move" toward better configurations based on:
        - Personal best (their own best performance)
        - Global best (best performance in swarm)
        """
        import time

        # First, run inference with MoE
        response = await self._infer_moe(prompt, analysis, timeout)

        # Update PSO state based on result
        if response.model_id in self.particles:
            particle = self.particles[response.model_id]

            # Update global best if needed
            current_fitness = particle.fitness()
            if current_fitness > self.global_best_score:
                self.global_best_score = current_fitness
                self.global_best_position = particle.position.copy()
                self.global_best_model = particle.model_id

            # Update all particle velocities (PSO step)
            self._pso_step()

        response.strategy_used = SwarmStrategy.PSO_COLLABORATIVE
        return response

    def _pso_step(self):
        """
        Perform one PSO update step for all particles.
        
        CST Enhancement: Uses relativistic constraints instead of random factors.
        Particles move in geodesic lines toward the global best, bounded by C_CONSTANT.
        This eliminates random "wandering" and forces deterministic convergence.
        """
        for particle in self.particles.values():
            if not particle.position or not particle.personal_best_position:
                continue

            # CST: Calculate geometric gradient toward global best
            gradient = self._calculate_geometric_gradient(particle)

            for i in range(len(particle.velocity)):
                # CST: Replace random factors with deterministic gradient
                # velocity = C_CONSTANT * geometric_gradient
                
                # Cognitive component (pull toward personal best)
                cognitive_diff = particle.personal_best_position[i] - particle.position[i]
                
                # Social component (pull toward global best)
                social_diff = 0.0
                if self.global_best_position and i < len(self.global_best_position):
                    social_diff = self.global_best_position[i] - particle.position[i]
                
                # CST: Geodesic direction = weighted combination of personal and social
                geodesic_direction = (
                    self.pso_cognitive * cognitive_diff +
                    self.pso_social * social_diff
                )
                
                # CST: Apply relativistic constraint - velocity cannot exceed C
                # Use inertia as a "mass" term that resists acceleration
                particle.velocity[i] = (
                    self.pso_inertia * particle.velocity[i] +
                    (1.0 - self.pso_inertia) * geodesic_direction
                )
                
                # CST: Clamp velocity to speed-of-light limit
                particle.velocity[i] = max(
                    -self.C_CONSTANT, 
                    min(self.C_CONSTANT, particle.velocity[i])
                )

                # Update position along geodesic
                particle.position[i] += particle.velocity[i]
                particle.position[i] = max(0, min(1, particle.position[i]))

    def _calculate_geometric_gradient(self, particle: ModelParticle) -> list[float]:
        """
        Calculate the geometric gradient toward global best in solution space.
        
        CST Principle: Particles follow geodesics (shortest paths) in the
        solution manifold, rather than random walks.
        
        Returns:
            Unit vector pointing toward global best position
        """
        if not self.global_best_position:
            return [0.0] * len(particle.position)
        
        gradient = []
        magnitude_sq = 0.0
        
        for i in range(len(particle.position)):
            if i < len(self.global_best_position):
                diff = self.global_best_position[i] - particle.position[i]
            else:
                diff = 0.0
            gradient.append(diff)
            magnitude_sq += diff * diff
        
        # Normalize to unit vector (geodesic direction)
        magnitude = magnitude_sq ** 0.5
        if magnitude > 0:
            gradient = [g / magnitude for g in gradient]
        
        return gradient

    async def _verify_response(
        self,
        prompt: str,
        response: SwarmResponse,
    ) -> SwarmResponse:
        """Verify response with a different model."""
        # Find a verifier (preferably reasoning-focused)
        verifiers = [
            p for p in self.particles.values()
            if p.model_id != response.model_id and p.model_id in self.backends
        ]

        if not verifiers:
            return response

        # Prefer reasoning models for verification
        verifier = next(
            (p for p in verifiers if p.role == ModelRole.REASONING),
            verifiers[0]
        )

        verify_prompt = f"""Evaluate this response and rate its quality from 0-10:

Question: {prompt}

Response: {response.text}

Rating (just the number):"""

        try:
            backend = self.backends[verifier.model_id]
            result = await asyncio.wait_for(
                backend.generate(verify_prompt),
                timeout=10.0
            )

            # Parse rating
            import re
            match = re.search(r'(\d+)', result.text)
            if match:
                rating = int(match.group(1))
                response.verified = True
                response.verifier_model = verifier.model_name
                # Adjust confidence based on verification
                response.confidence = (response.confidence + rating / 10) / 2

        except Exception as e:
            logger.debug(f"Verification failed: {e}")

        return response

    def get_stats(self) -> dict:
        """Get swarm statistics."""
        return {
            "total_queries": self.total_queries,
            "num_models": len(self.particles),
            "num_active": len(self.backends),
            "global_best_model": self.global_best_model,
            "global_best_score": self.global_best_score,
            "strategy_stats": {
                k: {
                    "count": v["count"],
                    "success_rate": v["success"] / max(1, v["count"]),
                    "avg_latency": sum(v["latency"]) / max(1, len(v["latency"])),
                }
                for k, v in self.strategy_stats.items()
            },
            "model_stats": {
                model_id: {
                    "role": p.role.value,
                    "fitness": p.fitness(),
                    "success_rate": p.success_rate,
                    "avg_latency": p.avg_latency,
                    "total_requests": p.total_requests,
                }
                for model_id, p in self.particles.items()
            },
        }

    def recommend_strategy(self, analysis: QueryAnalysis) -> SwarmStrategy:
        """Recommend best strategy based on query and history."""
        # Simple heuristics
        if analysis.complexity > 0.8:
            return SwarmStrategy.QUALITY_FIRST
        elif analysis.complexity < 0.3:
            return SwarmStrategy.FASTEST_FIRST
        elif analysis.requires_reasoning or analysis.requires_math:
            return SwarmStrategy.SPECULATIVE_ENSEMBLE
        elif len(self.backends) >= 3:
            return SwarmStrategy.PARALLEL_VOTE
        else:
            return SwarmStrategy.MIXTURE_OF_EXPERTS


# Default model configurations for swarm
DEFAULT_SWARM_MODELS = [
    {
        "id": "deepseek-r1-1.5b",
        "name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "strengths": ["reasoning", "math", "code"],
        "vram_gb": 2.0,
        "ram_gb": 4.0,
    },
    {
        "id": "phi-4-mini",
        "name": "Phi-4-mini",
        "strengths": ["reasoning", "math", "code"],
        "vram_gb": 3.0,
        "ram_gb": 6.0,
    },
    {
        "id": "qwen3-0.6b",
        "name": "Qwen3-0.6B",
        "strengths": ["speed", "multilingual", "fast"],
        "vram_gb": 1.0,
        "ram_gb": 2.0,
    },
    {
        "id": "smollm2-1.7b",
        "name": "SmolLM2-1.7B",
        "strengths": ["generalist", "quality"],
        "vram_gb": 1.5,
        "ram_gb": 3.0,
    },
    {
        "id": "tinyllama-1.1b",
        "name": "TinyLlama-1.1B",
        "strengths": ["speed", "fast", "edge"],
        "vram_gb": 1.0,
        "ram_gb": 2.0,
    },
]
