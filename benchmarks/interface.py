"""
Common interfaces for the benchmark framework.

All memory backends must implement BenchmarkableStore so the benchmark runner
can swap them transparently.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


class BenchmarkableStore(ABC):
    """Interface that all benchmarkable memory systems must implement."""

    @abstractmethod
    def store(self, content: str, category: str = "factual",
              scope: str = "global", importance: float = 0.5) -> None:
        """Store a memory. Category/scope/importance may be ignored by simple backends."""
        ...

    @abstractmethod
    def recall(self, query: str, top_k: int = 10,
               scope: Optional[str] = None) -> List[str]:
        """
        Recall memories matching the query.
        Returns list of memory content strings, ranked by relevance.
        """
        ...

    @abstractmethod
    def simulate_time(self, days: float) -> None:
        """Advance the simulated clock by N days. For decay/rehearsal testing."""
        ...

    @abstractmethod
    def simulate_access(self, content_substring: str) -> None:
        """Simulate accessing/rehearsing a memory (by content substring match)."""
        ...

    @abstractmethod
    def consolidate(self) -> None:
        """Run consolidation cycle. Noop for backends that don't support it."""
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return backend statistics."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memories. Called between benchmark runs."""
        ...

    def reward_memory(self, memory_id: str, signal: float) -> None:
        """Apply a reward signal to a retrieved memory's Q-value (optional).
        Default no-op for backends that don't support Q-value learning.

        Args:
            memory_id: ID of the memory to reward
            signal: Reward value (positive = useful, negative = not useful)
        """
        pass  # default no-op

    def explore(self, query: str, top_k: int = 20, scope: Optional[str] = None) -> List[str]:
        """Multi-hop exploration. Default: falls back to recall().

        Backends that implement Personalized PageRank graph walking can override
        this to enable multi-hop retrieval across linked memories.

        Args:
            query: The query to explore
            top_k: Number of results to return
            scope: Optional scope filter

        Returns:
            List of memory content strings, ranked by relevance
        """
        return self.recall(query, top_k=top_k, scope=scope)


# --- Result dataclasses ---

@dataclass
class JudgeResult:
    """Result from the LLM judge evaluating a single answer.

    Fields:
        correct:      Binary verdict — True if the answer is correct.
        raw_response: Raw text returned by the judge (LLM output or
                      heuristic explanation string).
        question_type: Optional category tag set by the benchmark runner.
        tokens_used:  LLM tokens consumed for this judgment (0 for
                      heuristic judge).
        scores:       Multi-dimensional rubric scores, each in [0.0, 1.0].
                      Keys produced by MemoryJudge (structured) and
                      HeuristicJudge:
                        - "relevance"            : retrieved facts on-topic
                        - "factual_accuracy"     : facts match gold answer
                        - "completeness"         : all gold parts covered
                        - "temporal_correctness" : correct version retrieved
        confidence:   Judge's confidence in the verdict, in [0.0, 1.0].
                      Derived from mean rubric score for heuristic judge;
                      set explicitly by the LLM structured judge.
        latency_ms:   Wall-clock time (ms) consumed by this judgment.
    """
    correct: bool
    raw_response: str = ""
    question_type: str = ""
    tokens_used: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    latency_ms: float = 0.0


@dataclass
class CategoryResult:
    """Results for a single test category (e.g., A1 semantic recall)."""
    category: str
    total: int
    correct: int
    score: float  # correct / total
    sub_scores: Dict[str, float] = field(default_factory=dict)
    # sub_scores: e.g., {"easy": 0.93, "medium": 0.80, "hard": 0.67}
    details: List[Dict[str, Any]] = field(default_factory=list)
    # per-question details for debugging
    recall_tokens: int = 0  # total tokens in recalled memories for this category
    recall_chars: int = 0   # total chars in recalled memories
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    # retrieval_metrics: aggregated IR metrics for this category,
    # e.g., {"recall_at_5": 0.82, "mrr": 0.74, "ndcg_at_5": 0.79, ...}


@dataclass
class RunResult:
    """Results from a single benchmark run (one seed)."""
    seed: int
    results_by_category: Dict[str, CategoryResult] = field(default_factory=dict)
    overall_score: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    # token_usage: {"recall_tokens": N, "recall_chars": N, "judge_tokens": N,
    #               "embed_calls": N, "store_calls": N, "recall_calls": N}
    wall_time_seconds: float = 0.0
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    # retrieval_metrics: mean IR metrics across all categories in this run
    cost_metrics: Dict[str, float] = field(default_factory=dict)
    # cost_metrics: {tokens_per_query, tokens_per_correct, cost_efficiency, score}


@dataclass
class AggregateResult:
    """Aggregated results across multiple runs."""
    num_runs: int = 0
    mean_score: float = 0.0
    std_score: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    per_category_mean: Dict[str, float] = field(default_factory=dict)
    per_category_std: Dict[str, float] = field(default_factory=dict)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    per_category_retrieval_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # per_category_retrieval_metrics: {category_name: {metric_name: mean_value}}
    mean_retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    # mean_retrieval_metrics: {metric_name: mean_value} averaged across all categories
    cost_metrics: Dict[str, float] = field(default_factory=dict)
    # cost_metrics: mean cost-efficiency metrics across all runs


@dataclass
class SignificanceResult:
    """Result of statistical significance test between two systems."""
    test_name: str = ""  # "paired_t_test" or "wilcoxon"
    p_value: float = 1.0
    effect_size: float = 0.0
    significant: bool = False  # p < 0.05
    baseline_mean: float = 0.0
    experiment_mean: float = 0.0
    improvement: float = 0.0  # percentage points


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    backend_name: str = "baseline-flat"
    profile: str = "balanced"
    embedding_model: str = "auto"
    parameters: Dict[str, Any] = field(default_factory=dict)
    num_runs: int = 5
    judge_model: str = "claude-haiku-4.5"
    output_path: str = "benchmarks/results/"
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
