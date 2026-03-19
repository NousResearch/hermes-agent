"""
COSMOS Latent Space Router
===========================

Semantic routing at the protocol layer — the mesh UNDERSTANDS what it routes.

Instead of routing by model name (--bot phi4), routes by intent:
  1. Embed the incoming query using sentence-transformers
  2. Compare against model profile vectors (built from strengths)
  3. Route to the model with highest cosine similarity
  4. Fall back to keyword matching if embeddings unavailable

Model profiles are multi-dimensional:
  [code, math, reasoning, creative, factual, multilingual]

Each dimension is a float [0, 1] representing model strength.
Profiles are seeded from known model capabilities, then updated
based on actual inference quality over time.

This is the first protocol-level semantic router for AI meshes.
"""
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from .cosmos_auth import blake3


# ── Model Profiles ────────────────────────────────────────────

PROFILE_DIMENSIONS = ["code", "math", "reasoning", "creative", "factual", "multilingual"]

# Known model strength profiles (0.0 to 1.0 per dimension)
# These are seed values — updated dynamically based on performance
DEFAULT_PROFILES: Dict[str, Dict[str, float]] = {
    "qwen3-coder-next-latest": {
        "code": 0.95, "math": 0.80, "reasoning": 0.85,
        "creative": 0.60, "factual": 0.75, "multilingual": 0.90,
    },
    "phi4-latest": {
        "code": 0.70, "math": 0.85, "reasoning": 0.90,
        "creative": 0.65, "factual": 0.80, "multilingual": 0.50,
    },
    "deepseek-r1-8b": {
        "code": 0.75, "math": 0.90, "reasoning": 0.95,
        "creative": 0.50, "factual": 0.70, "multilingual": 0.60,
    },
    "qwen2.5-7b": {
        "code": 0.70, "math": 0.70, "reasoning": 0.70,
        "creative": 0.70, "factual": 0.75, "multilingual": 0.95,
    },
    "mistral-7b": {
        "code": 0.65, "math": 0.60, "reasoning": 0.70,
        "creative": 0.75, "factual": 0.80, "multilingual": 0.70,
    },
    "llama3-8b": {
        "code": 0.65, "math": 0.60, "reasoning": 0.70,
        "creative": 0.80, "factual": 0.75, "multilingual": 0.60,
    },
    "gemma2-9b": {
        "code": 0.60, "math": 0.65, "reasoning": 0.75,
        "creative": 0.65, "factual": 0.85, "multilingual": 0.55,
    },
}

# Keyword → dimension mapping for fallback routing
KEYWORD_MAP: Dict[str, str] = {
    # Code
    "code": "code", "function": "code", "implement": "code", "debug": "code",
    "python": "code", "javascript": "code", "rust": "code", "api": "code",
    "class": "code", "algorithm": "code", "compile": "code", "program": "code",
    "script": "code", "variable": "code", "bug": "code", "refactor": "code",
    "async": "code", "tcp": "code", "server": "code", "database": "code",
    # Math
    "math": "math", "calculate": "math", "equation": "math", "prove": "math",
    "integral": "math", "derivative": "math", "matrix": "math", "solve": "math",
    "probability": "math", "statistics": "math", "theorem": "math",
    # Reasoning
    "explain": "reasoning", "why": "reasoning", "analyze": "reasoning",
    "compare": "reasoning", "implications": "reasoning", "evaluate": "reasoning",
    "logic": "reasoning", "deduce": "reasoning", "infer": "reasoning",
    "think": "reasoning", "reason": "reasoning", "chain": "reasoning",
    # Creative
    "story": "creative", "poem": "creative", "write": "creative",
    "creative": "creative", "imagine": "creative", "fiction": "creative",
    "generate": "creative", "brainstorm": "creative", "idea": "creative",
    # Factual
    "what is": "factual", "define": "factual", "history": "factual",
    "when": "factual", "where": "factual", "fact": "factual",
    "who": "factual", "describe": "factual",
    # Multilingual
    "translate": "multilingual", "chinese": "multilingual", "spanish": "multilingual",
    "french": "multilingual", "japanese": "multilingual", "korean": "multilingual",
    "arabic": "multilingual", "language": "multilingual",
}


@dataclass
class ModelProfile:
    """A model's strength profile in latent space."""
    bot_name: str
    strengths: Dict[str, float]     # dimension → strength [0, 1]
    embedding: Optional[List[float]] = None  # sentence-transformer embedding
    query_count: int = 0
    avg_quality: float = 0.0        # Running average quality score
    avg_latency_ms: float = 0.0     # Average inference time
    last_updated: float = 0.0

    @property
    def vector(self) -> List[float]:
        """Get profile as ordered vector."""
        return [self.strengths.get(d, 0.5) for d in PROFILE_DIMENSIONS]


@dataclass
class RouteDecision:
    """Result of a latent routing decision."""
    selected_bot: str
    confidence: float               # Cosine similarity [0, 1]
    all_scores: Dict[str, float]    # bot → score
    method: str                     # "embedding", "keyword", or "default"
    query_dimensions: Dict[str, float]  # Detected query dimensions
    timestamp: float = 0.0


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class LatentRouter:
    """
    Semantic routing engine for the COSMOS mesh.

    Routes queries to the best model based on:
    1. Sentence embedding similarity (if sentence-transformers available)
    2. Keyword-based dimension detection (fast fallback)
    3. Model strength profiles (seeded + learned)
    """

    def __init__(self):
        self._profiles: Dict[str, ModelProfile] = {}
        self._embedder = None
        self._category_embeddings: Dict[str, List[float]] = {}
        self._route_history: List[RouteDecision] = []
        self._init_profiles()
        self._init_embedder()

    def _init_profiles(self):
        """Initialize model profiles from defaults."""
        for bot_name, strengths in DEFAULT_PROFILES.items():
            self._profiles[bot_name] = ModelProfile(
                bot_name=bot_name,
                strengths=dict(strengths),
                last_updated=time.time(),
            )

    def _init_embedder(self):
        """Try to load sentence-transformers for embedding-based routing."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Latent Router: sentence-transformers loaded (all-MiniLM-L6-v2)")

            # Pre-compute category embeddings
            category_prompts = {
                "code": "Write code, implement a function, debug this program, build an API",
                "math": "Solve this equation, calculate the integral, prove this theorem",
                "reasoning": "Explain why, analyze the implications, evaluate the argument logically",
                "creative": "Write a story, imagine a scenario, brainstorm creative ideas",
                "factual": "What is the definition, describe the history, when did this happen",
                "multilingual": "Translate this text, write in Chinese, say this in Spanish",
            }
            for cat, prompt in category_prompts.items():
                emb = self._embedder.encode(prompt).tolist()
                self._category_embeddings[cat] = emb

        except ImportError:
            logger.info("Latent Router: sentence-transformers not available, using keyword routing")
            self._embedder = None

    def add_model(self, bot_name: str, strengths: Optional[Dict[str, float]] = None):
        """Register a model for routing. Uses defaults if known, balanced otherwise."""
        if bot_name in self._profiles:
            return

        if strengths:
            s = strengths
        elif bot_name in DEFAULT_PROFILES:
            s = DEFAULT_PROFILES[bot_name]
        else:
            # Unknown model — balanced profile
            s = {d: 0.5 for d in PROFILE_DIMENSIONS}

        self._profiles[bot_name] = ModelProfile(
            bot_name=bot_name,
            strengths=s,
            last_updated=time.time(),
        )

    def route(self, prompt: str, available_bots: Optional[List[str]] = None) -> RouteDecision:
        """
        Route a prompt to the best available model.

        Returns RouteDecision with selected bot, confidence, and all scores.
        """
        if not self._profiles:
            return RouteDecision(
                selected_bot="",
                confidence=0.0,
                all_scores={},
                method="none",
                query_dimensions={},
                timestamp=time.time(),
            )

        # Filter to available bots
        candidates = {}
        for name, profile in self._profiles.items():
            if available_bots is None or name in available_bots:
                candidates[name] = profile

        if not candidates:
            # Fall back to first available
            first = (available_bots or [""])[0]
            return RouteDecision(
                selected_bot=first,
                confidence=0.0,
                all_scores={},
                method="fallback",
                query_dimensions={},
                timestamp=time.time(),
            )

        # Detect query dimensions
        query_dims = self._analyze_query(prompt)

        # Score each candidate
        scores = {}
        for name, profile in candidates.items():
            scores[name] = cosine_similarity(
                [query_dims.get(d, 0.0) for d in PROFILE_DIMENSIONS],
                profile.vector,
            )

        # Select best
        best_bot = max(scores, key=scores.get)
        best_score = scores[best_bot]

        method = "embedding" if self._embedder else "keyword"

        decision = RouteDecision(
            selected_bot=best_bot,
            confidence=best_score,
            all_scores=scores,
            method=method,
            query_dimensions=query_dims,
            timestamp=time.time(),
        )

        self._route_history.append(decision)
        # Keep history bounded
        if len(self._route_history) > 1000:
            self._route_history = self._route_history[-500:]

        logger.info(
            f"Latent route: '{prompt[:50]}...' → {best_bot} "
            f"(confidence={best_score:.3f}, method={method})"
        )

        return decision

    def _analyze_query(self, prompt: str) -> Dict[str, float]:
        """Analyze a query to determine its dimension weights."""
        if self._embedder and self._category_embeddings:
            return self._analyze_with_embeddings(prompt)
        return self._analyze_with_keywords(prompt)

    def _analyze_with_embeddings(self, prompt: str) -> Dict[str, float]:
        """Use sentence embeddings to classify query dimensions."""
        query_emb = self._embedder.encode(prompt).tolist()

        dims = {}
        for cat, cat_emb in self._category_embeddings.items():
            sim = cosine_similarity(query_emb, cat_emb)
            # Normalize to [0, 1] range (cosine can be negative)
            dims[cat] = max(0.0, (sim + 1.0) / 2.0)

        # Normalize so max = 1.0
        max_val = max(dims.values()) if dims else 1.0
        if max_val > 0:
            dims = {k: v / max_val for k, v in dims.items()}

        return dims

    def _analyze_with_keywords(self, prompt: str) -> Dict[str, float]:
        """Use keyword matching to classify query dimensions (fast fallback)."""
        prompt_lower = prompt.lower()
        dims = {d: 0.0 for d in PROFILE_DIMENSIONS}

        for keyword, dimension in KEYWORD_MAP.items():
            if keyword in prompt_lower:
                dims[dimension] += 1.0

        # Normalize
        max_val = max(dims.values()) if dims else 1.0
        if max_val > 0:
            dims = {k: v / max_val for k, v in dims.items()}
        else:
            # No keywords matched — uniform distribution
            dims = {d: 0.5 for d in PROFILE_DIMENSIONS}

        return dims

    def update_model_quality(self, bot_name: str, quality: float,
                             latency_ms: float):
        """
        Update a model's profile based on actual performance.

        quality: 0.0 (bad) to 1.0 (perfect)
        latency_ms: inference time in milliseconds

        This allows the router to LEARN which models are best for which tasks.
        """
        profile = self._profiles.get(bot_name)
        if not profile:
            return

        profile.query_count += 1
        # Exponential moving average
        alpha = 0.1
        profile.avg_quality = (1 - alpha) * profile.avg_quality + alpha * quality
        profile.avg_latency_ms = (1 - alpha) * profile.avg_latency_ms + alpha * latency_ms
        profile.last_updated = time.time()

    def record_outcome(self, decision: RouteDecision, quality: float,
                       latency_ms: float = 0) -> None:
        """Record the outcome of a routed task to update model profiles.

        This is the feedback loop that makes routing LEARN:
        - Good outcomes reinforce the model's strengths in relevant dimensions
        - Bad outcomes reduce confidence in those dimensions
        - Profiles drift over time based on actual performance

        Args:
            decision: The RouteDecision that was made
            quality: 0.0 (terrible) to 1.0 (excellent)
            latency_ms: How long the inference took
        """
        bot = decision.selected_bot
        profile = self._profiles.get(bot)
        if not profile:
            return

        # Update running quality/latency averages
        profile.query_count += 1
        alpha = min(0.1, 1.0 / max(profile.query_count, 1))  # Decaying learning rate
        profile.avg_quality += alpha * (quality - profile.avg_quality)
        if latency_ms > 0:
            profile.avg_latency_ms += alpha * (latency_ms - profile.avg_latency_ms)

        # Update strength dimensions based on which dimensions were active
        lr = 0.02  # Small learning rate for gradual drift
        for dim, weight in decision.query_dimensions.items():
            if weight < 0.1:
                continue  # Skip dimensions not relevant to this query

            current = profile.strengths.get(dim, 0.5)
            # Good quality → increase strength, bad → decrease
            delta = lr * weight * (quality - 0.5)  # Centered at 0.5
            new_val = max(0.05, min(0.99, current + delta))
            profile.strengths[dim] = new_val

        profile.last_updated = time.time()

        logger.debug(
            f"Latent router feedback: {bot} quality={quality:.2f}, "
            f"updated dims={[d for d, w in decision.query_dimensions.items() if w >= 0.1]}"
        )

    def get_route_stats(self) -> Dict:
        """Get routing statistics for dashboard display."""
        stats = {
            "total_routes": len(self._route_history),
            "models": {},
        }
        for name, profile in self._profiles.items():
            stats["models"][name] = {
                "strengths": dict(profile.strengths),
                "query_count": profile.query_count,
                "avg_quality": round(profile.avg_quality, 3),
                "avg_latency_ms": round(profile.avg_latency_ms, 1),
            }

        # Recent routing decisions
        recent = self._route_history[-20:] if self._route_history else []
        stats["recent_routes"] = [
            {
                "model": d.selected_bot,
                "confidence": round(d.confidence, 3),
                "method": d.method,
                "dims": {k: round(v, 2) for k, v in d.query_dimensions.items() if v > 0.1},
                "ts": d.timestamp,
            }
            for d in recent
        ]

        return stats

    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        total = len(self._route_history)
        if total == 0:
            return {"total_routes": 0}

        method_counts = {}
        bot_counts = {}
        avg_confidence = 0.0

        for d in self._route_history:
            method_counts[d.method] = method_counts.get(d.method, 0) + 1
            bot_counts[d.selected_bot] = bot_counts.get(d.selected_bot, 0) + 1
            avg_confidence += d.confidence

        return {
            "total_routes": total,
            "avg_confidence": avg_confidence / total,
            "methods": method_counts,
            "bot_distribution": bot_counts,
            "models": {
                name: {
                    "strengths": p.strengths,
                    "queries": p.query_count,
                    "avg_quality": round(p.avg_quality, 3),
                    "avg_latency_ms": round(p.avg_latency_ms, 1),
                }
                for name, p in self._profiles.items()
            },
        }
