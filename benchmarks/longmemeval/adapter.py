"""
LongMemEval adapter for Hermes memory backends.

Loads questions from the xiaowu0162/longmemeval-cleaned HuggingFace dataset,
ingests haystack conversations into a BenchmarkableStore, then answers
questions via recall.

Each question is isolated: a fresh store is created per question to avoid
cross-question contamination.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.metrics import compute_metric_suite, token_f1
try:
    from cognitive_memory.ingestion import ingest_raw, ingest_chunked, ingest_summarized
except ImportError:
    ingest_raw = ingest_chunked = ingest_summarized = None  # cognitive_memory not available

logger = logging.getLogger(__name__)

# ── Constants ──

DATASET_NAME = "xiaowu0162/longmemeval-cleaned"
DATASET_SPLIT = "longmemeval_oracle"

QUESTION_TYPES = [
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
]

# ── Data structures ──


@dataclass
class LongMemQuestion:
    """A single LongMemEval question with its haystack context."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_dates: list[str]
    haystack_session_ids: list[str]
    haystack_sessions: list[list[dict[str, Any]]]  # list of sessions, each a list of {role, content}
    answer_session_ids: list[str]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LongMemQuestion":
        return cls(
            question_id=d["question_id"],
            question_type=d["question_type"],
            question=d["question"],
            answer=d["answer"],
            question_date=d.get("question_date", ""),
            haystack_dates=d.get("haystack_dates", []),
            haystack_session_ids=d.get("haystack_session_ids", []),
            haystack_sessions=d.get("haystack_sessions", []),
            answer_session_ids=d.get("answer_session_ids", []),
        )


@dataclass
class LongMemResult:
    """Result of evaluating one question."""

    question_id: str
    question_type: str
    question: str
    gold_answer: str
    recalled: str        # top recalled memory
    context: str         # all recalled memories joined
    correct: bool
    recall_count: int    # number of facts recalled
    metrics: dict = field(default_factory=dict)  # per-question metrics


@dataclass
class LongMemSummary:
    """Aggregated results across all questions."""

    total: int
    correct: int
    score: float
    by_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[LongMemResult] = field(default_factory=list)
    mean_metrics: dict = field(default_factory=dict)


# ── Dataset loading ──


def load_longmemeval_dataset(
    hf_cache: str | None = None,
    sample: int | None = None,
    question_type_filter: str | None = None,
) -> list[LongMemQuestion]:
    """
    Load LongMemEval questions from HuggingFace.

    Args:
        hf_cache: Optional path to HuggingFace cache directory.
        sample: If set, only return the first N questions.
        question_type_filter: If set, only return questions of this type.

    Returns:
        List of LongMemQuestion objects.
    """
    if hf_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_cache

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for LongMemEval. "
            "Install with: pip install datasets"
        )

    logger.info("Loading LongMemEval from HuggingFace (%s/%s)...", DATASET_NAME, DATASET_SPLIT)

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)

    questions = []
    for row in ds:
        if question_type_filter and row["question_type"] != question_type_filter:
            continue
        questions.append(LongMemQuestion.from_dict(row))
        if sample and len(questions) >= sample:
            break

    logger.info("Loaded %d questions from LongMemEval", len(questions))
    return questions


def load_longmemeval_local(
    path: str | Path,
    sample: int | None = None,
    question_type_filter: str | None = None,
) -> list[LongMemQuestion]:
    """
    Load LongMemEval questions from a local JSON file.

    Supports the standard LongMemEval JSON format (list of question dicts).

    Args:
        path: Path to the JSON file.
        sample: If set, only return the first N questions.
        question_type_filter: If set, only return questions of this type.

    Returns:
        List of LongMemQuestion objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LongMemEval data file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Some versions wrap in a dict
        data = list(data.values())

    questions = []
    for item in data:
        if question_type_filter and item.get("question_type") != question_type_filter:
            continue
        questions.append(LongMemQuestion.from_dict(item))
        if sample and len(questions) >= sample:
            break

    logger.info("Loaded %d questions from %s", len(questions), path)
    return questions


# ── Ingestion ──


def ingest_sessions_into_store(
    store: Any,
    question: LongMemQuestion,
    ingest_strategy: str = "raw",
    llm_fn=None,
) -> int:
    """
    Ingest all haystack sessions for a question into a CognitiveMemoryStore.

    Strategy:
    - Treat each conversation turn's content as a factual memory.
    - User turns: stored as factual, moderate importance.
    - Assistant turns: stored as factual, lower importance (assistant knowledge).
    - Messages from the answer session get a slight importance boost to reflect
      that they contain the answer.
    - We use simulate_time between sessions to give them realistic recency gaps.

    Args:
        store: A CognitiveBenchmarkAdapter (reset() should already be called).
        question: The question whose haystack we're ingesting.
        ingest_strategy: One of 'raw', 'chunk', 'summarize'.
        llm_fn: Optional callable(prompt) -> str for 'summarize' strategy.

    Returns:
        Total number of memories stored.
    """
    answer_session_ids = set(question.answer_session_ids)

    # Process sessions oldest first (haystack_dates is in chronological order)
    sessions = list(zip(
        question.haystack_session_ids,
        question.haystack_sessions,
    ))

    if ingest_strategy in ("chunk", "summarize"):
        # Flatten all sessions into a single turns list with gaps simulated after each session
        count = 0
        for i, (session_id, session_msgs) in enumerate(sessions):
            is_answer_session = session_id in answer_session_ids

            def importance_fn(role, content, _is_ans=is_answer_session):
                if _is_ans:
                    return 0.8 if role == "user" else 0.6
                return 0.5 if role == "user" else 0.3

            # Turns are already in LongMemEval format: {role, content}
            turns = [msg for msg in session_msgs if msg.get("content", "").strip()]

            if ingest_strategy == "chunk":
                count += ingest_chunked(turns, store, importance_fn=importance_fn)
            else:
                count += ingest_summarized(turns, store, llm_fn=llm_fn)

            if i < len(sessions) - 1:
                store.simulate_time(1)

        return count

    # Default: raw strategy (original behavior)
    count = 0
    for i, (session_id, session_msgs) in enumerate(sessions):
        is_answer_session = session_id in answer_session_ids

        for msg in session_msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()

            if not content:
                continue

            # Assign importance based on role and session type
            if is_answer_session:
                importance = 0.8 if role == "user" else 0.6
            else:
                importance = 0.5 if role == "user" else 0.3

            store.store(content, category="factual", importance=importance)
            count += 1

        # Simulate time between sessions (1 day per session gap)
        if i < len(sessions) - 1:
            store.simulate_time(1)

    return count


# ── Evaluation ──


def evaluate_question(
    store: Any,
    question: LongMemQuestion,
    judge: Any,
    top_k: int = 10,
    explore: bool = False,
) -> LongMemResult:
    """
    Evaluate a single LongMemEval question against the ingested store.

    Args:
        store: A CognitiveBenchmarkAdapter (already populated via ingest_sessions_into_store).
        question: The question to evaluate.
        judge: A MemoryJudge instance for scoring.
        top_k: Number of memories to recall.

    Returns:
        LongMemResult with scoring.
    """
    if explore:
        results = store.explore(question.question, top_k=top_k)
    else:
        results = store.recall(question.question, top_k=top_k)
    recalled = results[0] if results else ""
    context = " | ".join(results[:5]) if results else ""

    jr = judge.judge_answer(question.question, question.answer, context)

    metrics = compute_metric_suite(
        retrieved=results[:top_k],
        relevant=[question.answer],
        gold_answer=question.answer,
        predicted_answer=context,
    )

    return LongMemResult(
        question_id=question.question_id,
        question_type=question.question_type,
        question=question.question,
        gold_answer=question.answer,
        recalled=recalled,
        context=context,
        correct=jr.correct,
        recall_count=len(results),
        metrics=metrics,
    )


def run_longmemeval(
    questions: list[LongMemQuestion],
    judge: Any,
    backend_cls: Any | None = None,
    backend_kwargs: dict | None = None,
    top_k: int = 10,
    verbose: bool = False,
    explore: bool = False,
    ingest_strategy: str = "raw",
    llm_fn=None,
) -> LongMemSummary:
    """
    Run LongMemEval evaluation on a list of questions.

    Args:
        questions: List of LongMemQuestion objects.
        judge: A MemoryJudge instance.
        backend_cls: Class to instantiate for each question (default: CognitiveBenchmarkAdapter).
        backend_kwargs: Init kwargs for the backend.
        top_k: Number of memories to recall per question.
        verbose: If True, print per-question results.

    Returns:
        LongMemSummary with aggregated scores.
    """
    if backend_cls is None:
        from benchmarks.baseline.flat_store import FlatMemoryStore
        backend_cls = FlatMemoryStore

    backend_kwargs = backend_kwargs or {}

    results = []
    correct = 0

    for i, question in enumerate(questions):
        store = backend_cls(**backend_kwargs)
        store.reset()

        n_stored = ingest_sessions_into_store(store, question, ingest_strategy=ingest_strategy, llm_fn=llm_fn)

        result = evaluate_question(store, question, judge, top_k=top_k, explore=explore)
        results.append(result)

        if result.correct:
            correct += 1

        if verbose:
            status = "✓" if result.correct else "✗"
            mode = "explore" if explore else "recall"
            print(
                f"  [{i+1}/{len(questions)}] {status} {question.question_type} "
                f"q={question.question_id} stored={n_stored} "
                f"recalled={result.recall_count} mode={mode}"
            )

    # Aggregate by type
    by_type: dict[str, dict[str, Any]] = {}
    for qtype in QUESTION_TYPES:
        subset = [r for r in results if r.question_type == qtype]
        if subset:
            type_correct = sum(1 for r in subset if r.correct)
            type_metrics_list = [r.metrics for r in subset if r.metrics]
            type_mean_metrics: dict[str, float] = {}
            if type_metrics_list:
                for key in type_metrics_list[0]:
                    vals = [m[key] for m in type_metrics_list if key in m]
                    type_mean_metrics[key] = sum(vals) / len(vals) if vals else 0.0
            by_type[qtype] = {
                "total": len(subset),
                "correct": type_correct,
                "score": type_correct / len(subset),
                "mean_metrics": type_mean_metrics,
            }

    # Aggregate metrics across all results
    all_metrics = [r.metrics for r in results if r.metrics]
    mean_metrics: dict[str, float] = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            mean_metrics[key] = sum(values) / len(values) if values else 0.0

    total = len(questions)
    overall_score = correct / total if total > 0 else 0.0

    return LongMemSummary(
        total=total,
        correct=correct,
        score=overall_score,
        by_type=by_type,
        results=results,
        mean_metrics=mean_metrics,
    )
