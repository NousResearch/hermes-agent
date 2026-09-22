"""
HotpotQA adapter for Hermes memory backends.

Loads questions from the hotpotqa/hotpot_qa HuggingFace dataset (distractor
split), ingests per-question context paragraphs into a BenchmarkableStore,
then answers questions via recall.

The key differentiator for HotpotQA is multi-hop reasoning: questions require
chaining information across two or more supporting paragraphs.  We measure
*supporting_facts_recall* — did the store surface the specific paragraphs
annotated as supporting facts?

Each question is evaluated in isolation: a fresh store is created per question
to avoid cross-question contamination.

Reference
---------
Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., &
Manning, C. D. (2018). HotpotQA: A Dataset for Diverse, Explainable
Multi-hop Question Answering. EMNLP 2018. https://arxiv.org/abs/1809.09600
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──

DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"
FALLBACK_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

QUESTION_TYPES = ["bridge", "comparison"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


# ── Data structures ──


@dataclass
class HotpotQuestion:
    """A single HotpotQA question with its distractor context."""

    question_id: str
    question: str
    answer: str
    question_type: str              # "bridge" or "comparison"
    difficulty: str                 # "easy", "medium", or "hard"
    supporting_facts: list[list]    # list of [title, sent_idx] pairs
    context: list[list]             # list of [title, sentences] pairs

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HotpotQuestion":
        return cls(
            question_id=d["id"],
            question=d["question"],
            answer=d["answer"],
            question_type=d.get("type", "bridge"),
            difficulty=d.get("level", "medium"),
            supporting_facts=list(zip(
                d.get("supporting_facts", {}).get("title", []),
                d.get("supporting_facts", {}).get("sent_id", []),
            )) if isinstance(d.get("supporting_facts"), dict) else d.get("supporting_facts", []),
            context=list(zip(
                d.get("context", {}).get("title", []),
                d.get("context", {}).get("sentences", []),
            )) if isinstance(d.get("context"), dict) else d.get("context", []),
        )


@dataclass
class HotpotResult:
    """Result of evaluating one HotpotQA question."""

    question_id: str
    question_type: str
    difficulty: str
    question: str
    gold_answer: str
    predicted_answer: str           # top recalled memory used as answer
    context: str                    # all recalled items joined
    correct: bool                   # judge verdict
    recall_count: int               # number of items recalled
    supporting_facts_recall: float  # fraction of supporting-fact paragraphs retrieved
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class HotpotSummary:
    """Aggregated results across all HotpotQA questions."""

    total: int
    correct: int
    score: float
    avg_supporting_facts_recall: float
    avg_token_f1: float
    avg_exact_match: float
    by_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[HotpotResult] = field(default_factory=list)


# ── Dataset loading ──


def _stratified_sample(
    questions: list[HotpotQuestion],
    sample: int,
    rng: random.Random | None = None,
) -> list[HotpotQuestion]:
    """
    Return a stratified sample of `sample` questions.

    Stratification is over the 2×3 = 6 cells of (question_type × difficulty).
    Each cell gets an equal share; any remainder is filled by cycling through
    cells in order.
    """
    if rng is None:
        rng = random.Random(42)

    cells: dict[tuple[str, str], list[HotpotQuestion]] = {}
    for q in questions:
        key = (q.question_type, q.difficulty)
        cells.setdefault(key, []).append(q)

    for cell_list in cells.values():
        rng.shuffle(cell_list)

    active_keys = [k for k in cells if cells[k]]
    if not active_keys:
        return []

    per_cell = sample // len(active_keys)
    result: list[HotpotQuestion] = []
    for key in active_keys:
        result.extend(cells[key][:per_cell])

    # Fill remainder by cycling through non-exhausted cells
    remainder = sample - len(result)
    cycle_idx = 0
    while remainder > 0 and cycle_idx < len(active_keys) * 10:
        key = active_keys[cycle_idx % len(active_keys)]
        already_taken = per_cell
        extra = cells[key][already_taken : already_taken + 1]
        if extra:
            result.append(extra[0])
            remainder -= 1
            cells[key] = cells[key][:already_taken] + cells[key][already_taken + 1:]
        cycle_idx += 1

    rng.shuffle(result)
    return result[:sample]


def load_hotpotqa_dataset(
    sample: int = 500,
    hf_cache: str | None = None,
    difficulty_filter: str | None = None,
    question_type_filter: str | None = None,
) -> list[HotpotQuestion]:
    """
    Load HotpotQA questions from HuggingFace (distractor split).

    Uses stratified sampling to get a balanced mix of question types
    (bridge / comparison) and difficulty levels (easy / medium / hard).

    Args:
        sample:               Number of questions to return (stratified).
        hf_cache:             Optional HuggingFace cache directory.
        difficulty_filter:    If set, only return questions of this difficulty.
        question_type_filter: If set, only return questions of this type.

    Returns:
        List of HotpotQuestion objects.
    """
    if hf_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_cache

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for HotpotQA. "
            "Install with: pip install datasets"
        )

    logger.info(
        "Loading HotpotQA from HuggingFace (%s / %s / %s)...",
        DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT,
    )

    # Try primary name, fall back to legacy name
    # Note: newer datasets versions dropped trust_remote_code; try without it first
    for ds_name in [DATASET_NAME, "hotpot_qa"]:
        for kwargs in [
            dict(streaming=True),
            dict(streaming=True, trust_remote_code=True),
        ]:
            try:
                ds = load_dataset(
                    ds_name,
                    DATASET_CONFIG,
                    split=DATASET_SPLIT,
                    **kwargs,
                )
                break
            except TypeError:
                # trust_remote_code not supported in this version
                continue
            except Exception as exc:
                logger.warning("Failed to load '%s': %s", ds_name, exc)
                ds = None
                break
        if ds is not None:
            break

    if ds is None:
        # Fallback: download the official JSON from CMU
        logger.info("HuggingFace load failed. Downloading from official source: %s", FALLBACK_URL)
        cache_dir = Path(hf_cache) if hf_cache else Path.home() / ".cache" / "huggingface" / "datasets" / "hotpotqa"
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / "hotpot_dev_distractor_v1.json"

        if not local_path.exists():
            try:
                import urllib.request
                urllib.request.urlretrieve(FALLBACK_URL, local_path)
                logger.info("Downloaded to %s", local_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download HotpotQA from {FALLBACK_URL}: {exc}\n"
                    "  Try --local to load from a local JSON file."
                ) from exc
        else:
            logger.info("Using cached HotpotQA data at %s", local_path)

        return load_hotpotqa_local(
            local_path, sample=sample,
            difficulty_filter=difficulty_filter,
            question_type_filter=question_type_filter,
        )

    questions: list[HotpotQuestion] = []
    # Stream enough rows to allow stratified sampling (10× sample, cap at 10 000)
    stream_limit = min(max(sample * 10, 2000), 10_000)
    for row in ds:
        q = HotpotQuestion.from_dict(row)
        if difficulty_filter and q.difficulty != difficulty_filter:
            continue
        if question_type_filter and q.question_type != question_type_filter:
            continue
        questions.append(q)
        if len(questions) >= stream_limit:
            break

    logger.info("Streamed %d questions before stratified sampling", len(questions))

    if difficulty_filter or question_type_filter:
        # No need to stratify when a filter is already applied — just truncate
        result = questions[:sample]
    else:
        result = _stratified_sample(questions, sample)

    logger.info("Returning %d questions", len(result))
    return result


def load_hotpotqa_local(
    path: str | Path,
    sample: int | None = None,
    difficulty_filter: str | None = None,
    question_type_filter: str | None = None,
) -> list[HotpotQuestion]:
    """
    Load HotpotQA questions from a local JSON file.

    Supports the official HotpotQA JSON format (a list of question dicts)
    and a nested dict format (keys are question IDs).

    Args:
        path:                 Path to the JSON file.
        sample:               If set, apply stratified sampling to return N questions.
        difficulty_filter:    If set, only return questions of this difficulty.
        question_type_filter: If set, only return questions of this type.

    Returns:
        List of HotpotQuestion objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HotpotQA data file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.values())

    questions: list[HotpotQuestion] = []
    for item in data:
        q = HotpotQuestion.from_dict(item)
        if difficulty_filter and q.difficulty != difficulty_filter:
            continue
        if question_type_filter and q.question_type != question_type_filter:
            continue
        questions.append(q)

    logger.info("Loaded %d questions from %s", len(questions), path)

    if sample:
        if difficulty_filter or question_type_filter:
            questions = questions[:sample]
        else:
            questions = _stratified_sample(questions, sample)
        logger.info("After sampling: %d questions", len(questions))

    return questions


# ── Ingestion ──


def ingest_context_into_store(
    store: Any,
    context_paragraphs: list[list],
) -> int:
    """
    Ingest all context paragraphs for a HotpotQA question into a store.

    Strategy:
    - Each paragraph is a [title, sentences] pair from the question context.
    - Every sentence in each paragraph is stored as a separate factual memory.
    - The paragraph title is included in the text so that supporting_facts
      title matching works correctly during evaluation.
    - Importance is set to 0.6 uniformly (we do not know a priori which
      paragraphs are gold; that would be cheating).

    Args:
        store:               A CognitiveBenchmarkAdapter (reset() already called).
        context_paragraphs:  List of [title, sentences] pairs from question.context.

    Returns:
        Total number of memories stored.
    """
    count = 0
    for title, sentences in context_paragraphs:
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            # Embed the title so retrieval can surface the paragraph source
            text = f"[{title}] {sentence}"
            store.store(text, category="factual", importance=0.6)
            count += 1

    return count


# ── Evaluation ──


def _supporting_fact_titles(question: HotpotQuestion) -> set[str]:
    """Return the set of paragraph titles that contain supporting facts."""
    return {title for title, _sent_idx in question.supporting_facts}


def _retrieved_titles(retrieved: list[str]) -> set[str]:
    """
    Extract paragraph titles from retrieved memory strings.

    Memories are stored as "[Title] sentence text", so we parse the bracket
    prefix.  If no bracket is found, we skip that item.
    """
    titles: set[str] = set()
    for item in retrieved:
        if item.startswith("["):
            end = item.find("]")
            if end > 1:
                titles.add(item[1:end])
    return titles


def evaluate_question(
    store: Any,
    question: HotpotQuestion,
    judge: Any,
    top_k: int = 10,
    explore: bool = False,
) -> HotpotResult:
    """
    Evaluate a single HotpotQA question against the ingested store.

    Steps:
    1. Recall top_k memories matching the question text.
    2. Join results into a context string and use it as the predicted answer.
    3. Judge the context against the gold answer.
    4. Compute supporting_facts_recall: fraction of gold supporting-fact
       paragraph titles that appear in the retrieved set.
    5. Compute the full metric suite from benchmarks.metrics.

    Args:
        store:    A CognitiveBenchmarkAdapter (already populated via
                  ingest_context_into_store).
        question: The HotpotQuestion to evaluate.
        judge:    A MemoryJudge instance for scoring.
        top_k:    Number of memories to recall.

    Returns:
        HotpotResult with all scores populated.
    """
    from benchmarks.metrics import compute_metric_suite

    if explore:
        retrieved = store.explore(question.question, top_k=top_k)
    else:
        retrieved = store.recall(question.question, top_k=top_k)
    predicted_answer = retrieved[0] if retrieved else ""
    context = " | ".join(retrieved) if retrieved else ""

    # Judge answer correctness
    jr = judge.judge_answer(question.question, question.answer, context)

    # Supporting facts recall: did we surface the right paragraphs?
    gold_titles = _supporting_fact_titles(question)
    if gold_titles:
        retrieved_titles_set = _retrieved_titles(retrieved)
        sf_recall = len(gold_titles & retrieved_titles_set) / len(gold_titles)
    else:
        sf_recall = 0.0

    # Full metric suite — use gold_answer sentences from supporting paragraphs
    # as the "relevant" set for retrieval metrics
    relevant_texts = []
    for title, sentences in question.context:
        if title in gold_titles:
            for sent in sentences:
                if sent.strip():
                    relevant_texts.append(f"[{title}] {sent.strip()}")

    metrics = compute_metric_suite(
        retrieved=retrieved,
        relevant=relevant_texts if relevant_texts else [question.answer],
        gold_answer=question.answer,
        predicted_answer=predicted_answer,
    )

    return HotpotResult(
        question_id=question.question_id,
        question_type=question.question_type,
        difficulty=question.difficulty,
        question=question.question,
        gold_answer=question.answer,
        predicted_answer=predicted_answer,
        context=context,
        correct=jr.correct,
        recall_count=len(retrieved),
        supporting_facts_recall=sf_recall,
        metrics=metrics,
    )


def run_hotpotqa(
    questions: list[HotpotQuestion],
    judge: Any,
    backend_cls: Any | None = None,
    backend_kwargs: dict | None = None,
    top_k: int = 10,
    verbose: bool = False,
    explore: bool = False,
) -> HotpotSummary:
    """
    Run the full HotpotQA evaluation loop over a list of questions.

    For each question:
    - Create a fresh store instance.
    - Ingest the question's context paragraphs.
    - Evaluate via evaluate_question.

    Aggregates results overall, per question type, and per difficulty level.

    Args:
        questions:       List of HotpotQuestion objects.
        judge:           A MemoryJudge instance.
        backend_cls:     Backend class to instantiate per question
                         (default: CognitiveBenchmarkAdapter).
        backend_kwargs:  Init kwargs for the backend.
        top_k:           Number of memories to recall per question.
        verbose:         If True, print per-question status to stdout.

    Returns:
        HotpotSummary with aggregated scores.
    """
    if backend_cls is None:
        from benchmarks.baseline.flat_store import FlatMemoryStore
        backend_cls = FlatMemoryStore

    backend_kwargs = backend_kwargs or {}

    results: list[HotpotResult] = []
    correct = 0

    for i, question in enumerate(questions):
        store = backend_cls(**backend_kwargs)
        store.reset()

        n_stored = ingest_context_into_store(store, question.context)

        result = evaluate_question(store, question, judge, top_k=top_k, explore=explore)
        results.append(result)

        if result.correct:
            correct += 1

        if verbose:
            status = "✓" if result.correct else "✗"
            mode = "explore" if explore else "recall"
            print(
                f"  [{i+1}/{len(questions)}] {status} "
                f"type={question.question_type} diff={question.difficulty} "
                f"id={question.question_id} "
                f"stored={n_stored} recalled={result.recall_count} "
                f"sf_recall={result.supporting_facts_recall:.2f} mode={mode}"
            )

    total = len(questions)
    overall_score = correct / total if total > 0 else 0.0
    avg_sf_recall = (
        sum(r.supporting_facts_recall for r in results) / total
        if total > 0 else 0.0
    )
    avg_token_f1 = (
        sum(r.metrics.get("token_f1", 0.0) for r in results) / total
        if total > 0 else 0.0
    )
    avg_exact_match = (
        sum(r.metrics.get("exact_match", 0.0) for r in results) / total
        if total > 0 else 0.0
    )

    # Aggregate by question type
    by_type: dict[str, dict[str, Any]] = {}
    for qtype in QUESTION_TYPES:
        subset = [r for r in results if r.question_type == qtype]
        if subset:
            n = len(subset)
            type_correct = sum(1 for r in subset if r.correct)
            by_type[qtype] = {
                "total": n,
                "correct": type_correct,
                "score": type_correct / n,
                "avg_supporting_facts_recall": sum(r.supporting_facts_recall for r in subset) / n,
                "avg_token_f1": sum(r.metrics.get("token_f1", 0.0) for r in subset) / n,
                "avg_exact_match": sum(r.metrics.get("exact_match", 0.0) for r in subset) / n,
            }

    # Aggregate by difficulty
    by_difficulty: dict[str, dict[str, Any]] = {}
    for diff in DIFFICULTY_LEVELS:
        subset = [r for r in results if r.difficulty == diff]
        if subset:
            n = len(subset)
            diff_correct = sum(1 for r in subset if r.correct)
            by_difficulty[diff] = {
                "total": n,
                "correct": diff_correct,
                "score": diff_correct / n,
                "avg_supporting_facts_recall": sum(r.supporting_facts_recall for r in subset) / n,
                "avg_token_f1": sum(r.metrics.get("token_f1", 0.0) for r in subset) / n,
                "avg_exact_match": sum(r.metrics.get("exact_match", 0.0) for r in subset) / n,
            }

    return HotpotSummary(
        total=total,
        correct=correct,
        score=overall_score,
        avg_supporting_facts_recall=avg_sf_recall,
        avg_token_f1=avg_token_f1,
        avg_exact_match=avg_exact_match,
        by_type=by_type,
        by_difficulty=by_difficulty,
        results=results,
    )
