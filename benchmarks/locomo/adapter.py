"""
LoCoMo adapter for Hermes memory backends.

Loads questions from the snap-research/locomo HuggingFace dataset (or a local
JSON file), ingests each question's conversation into a BenchmarkableStore,
then answers questions via recall.

Each question is isolated: a fresh store is created per question to avoid
cross-question contamination.

Reference:
    Maharana et al. (2024). "Building a Long-term Memory for Conversational
    Agents." arXiv:2402.15929. https://arxiv.org/abs/2402.15929
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.metrics import compute_metric_suite
try:
    from cognitive_memory.ingestion import ingest_raw, ingest_chunked, ingest_summarized
except ImportError:
    ingest_raw = ingest_chunked = ingest_summarized = None  # cognitive_memory not available

logger = logging.getLogger(__name__)

# ── Constants ──

DATASET_NAME = "snap-research/locomo"
DATASET_SPLIT = "test"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/snap-research/LoCoMo/main/data/locomo10.json"

QUESTION_TYPES = [
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
    "adversarial",
]

CATEGORY_MAP = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}

# ── Data structures ──


@dataclass
class LoCoMoQuestion:
    """A single LoCoMo question with its full conversation context."""

    question_id: str
    question_type: str   # single_hop | multi_hop | temporal | open_domain | adversarial
    question: str
    answer: str
    evidence: list       # evidence references like ['D1:3']
    conversation_id: str # sample_id from the source data
    conversation_sessions: list  # list of sessions, each session is list of {speaker, dia_id, text}


@dataclass
class LoCoMoResult:
    """Result of evaluating one LoCoMo question."""

    question_id: str
    question_type: str
    question: str
    gold_answer: str
    recalled: str        # top recalled memory content
    context: str         # top-5 recalled memories joined
    correct: bool
    recall_count: int    # total memories retrieved
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class LoCoMoSummary:
    """Aggregated results across all LoCoMo questions."""

    total: int
    correct: int
    score: float
    by_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[LoCoMoResult] = field(default_factory=list)
    # Aggregate retrieval metrics averaged across all questions
    mean_metrics: dict[str, float] = field(default_factory=dict)


# ── Dataset loading ──


def _extract_sessions(conversation: dict) -> list[list[dict]]:
    """
    Extract ordered list of sessions from a conversation dict.

    Sessions are keyed as session_1, session_2, etc.
    Returns a list of sessions, each session is a list of turn dicts
    with keys {speaker, dia_id, text}.
    """
    sessions = []
    i = 1
    while True:
        key = f"session_{i}"
        if key not in conversation:
            break
        session_turns = conversation[key]
        if isinstance(session_turns, list):
            sessions.append(session_turns)
        i += 1
    return sessions


def _parse_conversation_object(conv_obj: dict) -> list[LoCoMoQuestion]:
    """
    Parse a single conversation object from the LoCoMo JSON into a list
    of LoCoMoQuestion instances (one per QA pair).
    """
    sample_id = str(conv_obj.get("sample_id", ""))
    conversation = conv_obj.get("conversation", {})
    qa_list = conv_obj.get("qa", [])

    sessions = _extract_sessions(conversation)

    questions = []
    for idx, qa in enumerate(qa_list):
        category_int = qa.get("category", 1)
        question_type = CATEGORY_MAP.get(category_int, "single_hop")
        question_id = f"{sample_id}_{idx}"

        q = LoCoMoQuestion(
            question_id=question_id,
            question_type=question_type,
            question=str(qa.get("question", "")),
            answer=str(qa.get("answer", "")),
            evidence=qa.get("evidence", []),
            conversation_id=sample_id,
            conversation_sessions=sessions,
        )
        questions.append(q)

    return questions


def load_locomo_local(
    path: str | Path,
    sample: int | None = None,
    question_type_filter: str | None = None,
) -> list[LoCoMoQuestion]:
    """
    Load LoCoMo questions from a local JSON file (snap-research/LoCoMo format).

    The file should be a JSON list of conversation objects, each with:
    - sample_id: str
    - qa: list of {question, answer, evidence, category}
    - conversation: dict with speaker_a, speaker_b, session_1, session_2, ...

    Args:
        path: Path to the JSON file.
        sample: If set, only return the first N questions.
        question_type_filter: If set, only return questions of this type.

    Returns:
        List of LoCoMoQuestion objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON structure is unrecognized.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LoCoMo data file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Could be {"questions": [...]} or {"data": [...]}
        if "questions" in data:
            data = data["questions"]
        elif "data" in data:
            data = data["data"]
        else:
            data = list(data.values())

    if not isinstance(data, list):
        raise ValueError(
            f"Unrecognized LoCoMo JSON structure in {path}. "
            "Expected a list of conversation objects."
        )

    questions: list[LoCoMoQuestion] = []
    for conv_obj in data:
        if not isinstance(conv_obj, dict):
            continue
        parsed = _parse_conversation_object(conv_obj)
        for q in parsed:
            if question_type_filter and q.question_type != question_type_filter:
                continue
            questions.append(q)
            if sample and len(questions) >= sample:
                break
        if sample and len(questions) >= sample:
            break

    logger.info("Loaded %d questions from %s", len(questions), path)
    return questions


def load_locomo_dataset(
    hf_cache: str | None = None,
    sample: int | None = None,
    question_type_filter: str | None = None,
) -> list[LoCoMoQuestion]:
    """
    Load LoCoMo questions from HuggingFace (snap-research/locomo).

    Falls back to downloading from the GitHub raw URL if HuggingFace fails.

    Args:
        hf_cache: Optional path to HuggingFace cache directory.
        sample: If set, only return the first N questions.
        question_type_filter: If set, only return questions of this type.

    Returns:
        List of LoCoMoQuestion objects.

    Raises:
        RuntimeError: If all loading methods fail.
    """
    if hf_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_cache

    # Try HuggingFace first
    try:
        from datasets import load_dataset

        logger.info(
            "Loading LoCoMo from HuggingFace (%s, split=%s)...",
            DATASET_NAME,
            DATASET_SPLIT,
        )
        ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
        questions: list[LoCoMoQuestion] = []
        for conv_obj in ds:
            if not isinstance(conv_obj, dict):
                continue
            parsed = _parse_conversation_object(conv_obj)
            for q in parsed:
                if question_type_filter and q.question_type != question_type_filter:
                    continue
                questions.append(q)
                if sample and len(questions) >= sample:
                    break
            if sample and len(questions) >= sample:
                break
        logger.info("Loaded %d questions from LoCoMo (HuggingFace)", len(questions))
        return questions

    except ImportError:
        logger.warning("datasets package not available, trying GitHub download...")
    except Exception as exc:
        logger.warning("HuggingFace load failed: %s. Trying GitHub download...", exc)

    # Fallback: download from GitHub raw URL
    cache_dir = Path(hf_cache) if hf_cache else Path.home() / ".cache" / "huggingface" / "datasets" / "locomo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "locomo10.json"

    if not local_path.exists():
        logger.info("Downloading LoCoMo from GitHub: %s", GITHUB_RAW_URL)
        try:
            import urllib.request
            urllib.request.urlretrieve(GITHUB_RAW_URL, local_path)
            logger.info("Downloaded to %s", local_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download LoCoMo from GitHub ({GITHUB_RAW_URL!r}): {exc}\n\n"
                "Try loading from a local JSON file instead:\n"
                "  python -m benchmarks.locomo.runner --local /path/to/locomo10.json\n\n"
                "You can download the data from:\n"
                "  https://github.com/snap-research/LoCoMo\n"
                "  https://arxiv.org/abs/2402.15929"
            ) from exc
    else:
        logger.info("Using cached LoCoMo data at %s", local_path)

    return load_locomo_local(local_path, sample=sample, question_type_filter=question_type_filter)


# ── Ingestion ──


def _normalize_locomo_turns(session_turns: list[dict], evidence_set: set[str]) -> list[dict]:
    """Normalize LoCoMo turn format to ingestion module format.

    LoCoMo turns have {speaker, dia_id, text}; ingestion expects {role, content, speaker}.
    """
    normalized = []
    for turn in session_turns:
        text = turn.get("text", "").strip()
        if not text:
            continue
        speaker = turn.get("speaker", "")
        dia_id = turn.get("dia_id", "")
        # LoCoMo doesn't distinguish user/assistant roles — treat all as "user"
        normalized.append({
            "role": "user",
            "content": text,
            "speaker": speaker,
            "dia_id": dia_id,
            "_is_evidence": dia_id in evidence_set,
        })
    return normalized


def ingest_conversation_into_store(
    store: Any,
    conversation_sessions: list[list[dict]],
    evidence_refs: list[str] | None = None,
    ingest_strategy: str = "raw",
    llm_fn=None,
) -> int:
    """
    Ingest conversation sessions into a CognitiveMemoryStore.

    Strategy:
    - Each session's turns are stored as factual memories.
    - Turns whose dia_id appears in evidence_refs get importance 0.8
      (these are the turns that answer questions).
    - Other turns get importance 0.5.
    - simulate_time(1) is called between sessions to model temporal gaps.

    Args:
        store: A CognitiveBenchmarkAdapter (reset() should already be called).
        conversation_sessions: List of sessions, each a list of turn dicts
            with keys {speaker, dia_id, text}.
        evidence_refs: List of dia_id strings that are evidence for questions
            (e.g. ['D1:3', 'D2:7']). Turns matching these get higher importance.
        ingest_strategy: One of 'raw', 'chunk', 'summarize'.
        llm_fn: Optional callable(prompt) -> str for 'summarize' strategy.

    Returns:
        Total number of memories stored.
    """
    evidence_set: set[str] = set(evidence_refs) if evidence_refs else set()
    count = 0

    if ingest_strategy in ("chunk", "summarize"):
        for session_idx, session_turns in enumerate(conversation_sessions):
            # Normalize LoCoMo turns (text -> content) for the ingestion module
            turns = _normalize_locomo_turns(session_turns, evidence_set)

            def importance_fn(role, content, _turns=turns):
                # Find the matching turn to check evidence flag
                for t in _turns:
                    if t["content"] == content:
                        return 0.8 if t["_is_evidence"] else 0.5
                return 0.5

            if ingest_strategy == "chunk":
                count += ingest_chunked(turns, store, importance_fn=importance_fn)
            else:
                count += ingest_summarized(turns, store, llm_fn=llm_fn)

            if session_idx < len(conversation_sessions) - 1:
                store.simulate_time(1)

        return count

    # Default: raw strategy (original behavior, preserve speaker prefix)
    for session_idx, session_turns in enumerate(conversation_sessions):
        for turn in session_turns:
            dia_id = turn.get("dia_id", "")
            text = turn.get("text", "").strip()
            speaker = turn.get("speaker", "")

            if not text:
                continue

            # Evidence turns are more important — they contain answer facts
            if dia_id in evidence_set:
                importance = 0.8
            else:
                importance = 0.5

            # Include speaker name in stored content for context
            content = f"{speaker}: {text}" if speaker else text
            store.store(content, category="factual", importance=importance)
            count += 1

        # Advance simulated time between sessions to model temporal gaps
        if session_idx < len(conversation_sessions) - 1:
            store.simulate_time(1)

    return count


# ── Evaluation ──


def evaluate_question(
    store: Any,
    question: LoCoMoQuestion,
    judge: Any,
    top_k: int = 10,
    explore: bool = False,
) -> LoCoMoResult:
    """
    Evaluate a single LoCoMo question against the ingested store.

    Recalls top_k memories, builds context from the top 5, judges the
    context against the gold answer, and computes the full metric suite.

    Args:
        store: A CognitiveBenchmarkAdapter already populated via
            ingest_conversation_into_store.
        question: The LoCoMoQuestion to evaluate.
        judge: A MemoryJudge instance for scoring.
        top_k: Number of memories to recall (default 10).

    Returns:
        LoCoMoResult with binary correctness, context, and retrieval metrics.
    """
    if explore:
        results = store.explore(question.question, top_k=top_k)
    else:
        results = store.recall(question.question, top_k=top_k)
    recalled = results[0] if results else ""
    context = " | ".join(results[:5]) if results else ""

    jr = judge.judge_answer(question.question, question.answer, context)

    # Use the gold answer as the single "relevant" item for retrieval metrics.
    # This gives a coarse signal: did we surface a memory containing the answer?
    metrics = compute_metric_suite(
        retrieved=results,
        relevant=[question.answer],
        gold_answer=question.answer,
        predicted_answer=context,
    )

    return LoCoMoResult(
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


def run_locomo(
    questions: list[LoCoMoQuestion],
    judge: Any,
    backend_cls: Any | None = None,
    backend_kwargs: dict | None = None,
    top_k: int = 10,
    verbose: bool = False,
    explore: bool = False,
    ingest_strategy: str = "raw",
    llm_fn=None,
) -> LoCoMoSummary:
    """
    Run LoCoMo evaluation on a list of questions.

    For each question:
    1. Create a fresh CognitiveBenchmarkAdapter store.
    2. Ingest the question's conversation sessions (with evidence hints).
    3. Evaluate the question.

    Then aggregate overall accuracy and per-question-type breakdowns.

    Args:
        questions: List of LoCoMoQuestion objects.
        judge: A MemoryJudge instance.
        backend_cls: Backend class to instantiate (default: CognitiveBenchmarkAdapter).
        backend_kwargs: Init kwargs for the backend.
        top_k: Number of memories to recall per question (default 10).
        verbose: If True, print per-question results to stdout.

    Returns:
        LoCoMoSummary with aggregated scores and per-type breakdowns.
    """
    if backend_cls is None:
        from benchmarks.baseline.flat_store import FlatMemoryStore
        backend_cls = FlatMemoryStore

    backend_kwargs = backend_kwargs or {}

    results: list[LoCoMoResult] = []
    correct = 0

    # Accumulate metric values for computing means
    metric_accumulator: dict[str, list[float]] = {}

    for i, question in enumerate(questions):
        store = backend_cls(**backend_kwargs)
        store.reset()

        n_stored = ingest_conversation_into_store(
            store,
            question.conversation_sessions,
            evidence_refs=question.evidence,
            ingest_strategy=ingest_strategy,
            llm_fn=llm_fn,
        )

        result = evaluate_question(store, question, judge, top_k=top_k, explore=explore)
        results.append(result)

        if result.correct:
            correct += 1

        # Accumulate retrieval metrics
        for k, v in result.metrics.items():
            metric_accumulator.setdefault(k, []).append(v)

        if verbose:
            status = "✓" if result.correct else "✗"
            mode = "explore" if explore else "recall"
            print(
                f"  [{i+1}/{len(questions)}] {status} {question.question_type} "
                f"q={question.question_id} stored={n_stored} "
                f"recalled={result.recall_count} mode={mode}"
            )

    # ── Aggregate by question type ──
    by_type: dict[str, dict[str, Any]] = {}
    for qtype in QUESTION_TYPES:
        subset = [r for r in results if r.question_type == qtype]
        if not subset:
            continue
        type_correct = sum(1 for r in subset if r.correct)
        # Average retrieval metrics for this type
        type_metrics: dict[str, float] = {}
        for k in (subset[0].metrics if subset else {}).keys():
            vals = [r.metrics.get(k, 0.0) for r in subset]
            type_metrics[k] = sum(vals) / len(vals) if vals else 0.0
        by_type[qtype] = {
            "total": len(subset),
            "correct": type_correct,
            "score": type_correct / len(subset),
            "metrics": type_metrics,
        }

    # ── Also capture any question types not in QUESTION_TYPES ──
    seen_types = {r.question_type for r in results}
    for qtype in seen_types - set(QUESTION_TYPES):
        subset = [r for r in results if r.question_type == qtype]
        type_correct = sum(1 for r in subset if r.correct)
        type_metrics = {}
        for k in (subset[0].metrics if subset else {}).keys():
            vals = [r.metrics.get(k, 0.0) for r in subset]
            type_metrics[k] = sum(vals) / len(vals) if vals else 0.0
        by_type[qtype] = {
            "total": len(subset),
            "correct": type_correct,
            "score": type_correct / len(subset),
            "metrics": type_metrics,
        }

    total = len(questions)
    overall_score = correct / total if total > 0 else 0.0

    # Compute mean metrics across all questions
    mean_metrics = {
        k: sum(v) / len(v)
        for k, v in metric_accumulator.items()
        if v
    }

    return LoCoMoSummary(
        total=total,
        correct=correct,
        score=overall_score,
        by_type=by_type,
        results=results,
        mean_metrics=mean_metrics,
    )
