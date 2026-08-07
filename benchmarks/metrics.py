"""
Retrieval-quality and answer-level evaluation metrics for the benchmark framework.

Implements standard IR and QA metrics from scratch using only Python stdlib.
Suitable for evaluating memory recall quality and answer accuracy in cognitive
memory benchmarks.

References
----------
- Token F1 / Exact Match: Rajpurkar et al. (2016). "SQuAD: 100,000+ Questions
  for Machine Comprehension of Text." EMNLP 2016.
  https://arxiv.org/abs/1606.05250

- NDCG: Järvelin, K. & Kekäläinen, J. (2002). "Cumulated gain-based evaluation
  of IR techniques." ACM TOIS, 20(4), 422-446.
  https://doi.org/10.1145/582415.582418

- MRR: Voorhees, E. M. (1999). "The TREC-8 Question Answering Track Report."
  TREC 1999. https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List

__all__ = [
    # relevance helper
    "is_relevant",
    # retrieval metrics
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "ap",
    # answer-level metrics
    "normalize_answer",
    "exact_match",
    "token_f1",
    "token_precision",
    "token_recall",
    # aggregate helpers
    "mean_metric",
    "compute_metric_suite",
    # cost-efficiency metrics
    "tokens_per_correct",
    "cost_efficiency_ratio",
    "compute_cost_metrics",
]


# ---------------------------------------------------------------------------
# Relevance helper
# ---------------------------------------------------------------------------

def is_relevant(retrieved_str: str, gold_str: str, threshold: float = 0.7) -> bool:
    """
    Determine whether a retrieved string is relevant to a gold reference string.

    Uses two complementary fuzzy-matching strategies:

    1. Substring containment: the retrieved string contains the gold string as
       a case-insensitive substring.
    2. Token overlap: the Jaccard-like token overlap between the two strings
       exceeds `threshold`.

    Parameters
    ----------
    retrieved_str : str
        The candidate string returned by the retrieval system.
    gold_str : str
        The reference/gold string we are trying to recall.
    threshold : float, optional
        Minimum token-overlap ratio to count as relevant (default 0.7).

    Returns
    -------
    bool
        True if the retrieved string is considered relevant to the gold string.

    Examples
    --------
    >>> is_relevant("The cat sat on the mat", "cat sat on the mat")
    True
    >>> is_relevant("Dogs are mammals", "cats are mammals", threshold=0.7)
    False
    """
    if not retrieved_str or not gold_str:
        return False

    r_lower = retrieved_str.lower()
    g_lower = gold_str.lower()

    # Strategy 1: substring containment
    if g_lower in r_lower:
        return True

    # Strategy 2: token overlap (intersection / union of token sets)
    r_tokens = set(r_lower.split())
    g_tokens = set(g_lower.split())
    if not g_tokens:
        return False
    intersection = r_tokens & g_tokens
    union = r_tokens | g_tokens
    if not union:
        return False
    overlap = len(intersection) / len(union)
    return overlap >= threshold


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Recall@K: fraction of relevant items found in the top-K retrieved results.

    Measures the coverage of the retrieved set — what proportion of all known
    relevant items appear in the top-K positions.

    Formula:
        Recall@K = |{relevant} ∩ {retrieved[:k]}| / |{relevant}|

    Relevance is determined via `is_relevant` (fuzzy matching).

    Parameters
    ----------
    retrieved : List[str]
        Ranked list of retrieved strings (position 0 = highest ranked).
    relevant : List[str]
        List of gold/relevant strings.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        Recall@K in [0.0, 1.0]. Returns 0.0 if `relevant` is empty.

    Examples
    --------
    >>> recall_at_k(["a b c", "d e f", "g h i"], ["a b c", "x y z"], k=3)
    0.5
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(
        1 for gold in relevant
        if any(is_relevant(ret, gold) for ret in top_k)
    )
    return hits / len(relevant)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Precision@K: fraction of the top-K retrieved results that are relevant.

    Measures the accuracy of the retrieved set — what proportion of the top-K
    candidates are actually relevant.

    Formula:
        Precision@K = |{relevant} ∩ {retrieved[:k]}| / K

    Parameters
    ----------
    retrieved : List[str]
        Ranked list of retrieved strings (position 0 = highest ranked).
    relevant : List[str]
        List of gold/relevant strings.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        Precision@K in [0.0, 1.0]. Returns 0.0 if k <= 0.

    Examples
    --------
    >>> precision_at_k(["a b c", "irrelevant", "d e f"], ["a b c", "d e f"], k=3)
    0.6666666666666666
    """
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(
        1 for ret in top_k
        if any(is_relevant(ret, gold) for gold in relevant)
    )
    return hits / k


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """
    Mean Reciprocal Rank (MRR): reciprocal of the rank of the first relevant item.

    MRR is the standard metric for evaluating ranked retrieval where only the
    first relevant result matters (e.g., question answering, navigational search).

    Formula:
        MRR = 1 / rank_of_first_relevant_item

    If no relevant item is found, returns 0.0.

    Reference: Voorhees, E. M. (1999). "The TREC-8 Question Answering Track
    Report." TREC 1999. https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf

    Parameters
    ----------
    retrieved : List[str]
        Ranked list of retrieved strings (position 0 = highest ranked).
    relevant : List[str]
        List of gold/relevant strings.

    Returns
    -------
    float
        MRR in [0.0, 1.0]. Returns 0.0 if no relevant item is found.

    Examples
    --------
    >>> mrr(["irrelevant", "the gold answer is here", "other"], ["gold answer"])
    0.5
    """
    if not relevant or not retrieved:
        return 0.0
    for rank, ret in enumerate(retrieved, start=1):
        if any(is_relevant(ret, gold) for gold in relevant):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevance_scores: List[float], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K).

    NDCG measures ranking quality by rewarding highly relevant items at the
    top of the ranking. The gain is discounted logarithmically by rank position,
    then normalized by the ideal DCG (IDCG) — the best possible ranking.

    Formula:
        DCG@K  = sum_{i=1}^{K} rel_i / log2(i + 1)
        IDCG@K = DCG of the ideal ranking (sorted by relevance desc)
        NDCG@K = DCG@K / IDCG@K

    Reference: Järvelin, K. & Kekäläinen, J. (2002). "Cumulated gain-based
    evaluation of IR techniques." ACM TOIS, 20(4), 422-446.
    https://doi.org/10.1145/582415.582418

    Parameters
    ----------
    retrieved : List[str]
        Ranked list of retrieved strings (position 0 = highest ranked).
        The i-th string corresponds to relevance_scores[i].
    relevance_scores : List[float]
        Non-negative relevance score for each retrieved item. Must be the same
        length as `retrieved`. Typically binary (0.0 or 1.0) or graded.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        NDCG@K in [0.0, 1.0]. Returns 0.0 if all relevance scores are 0.

    Notes
    -----
    When using binary relevance (0/1 scores), compute relevance_scores as:
        [1.0 if any(is_relevant(r, g) for g in relevant) else 0.0
         for r in retrieved]

    Examples
    --------
    >>> ndcg_at_k(["a", "b", "c"], [1.0, 0.0, 1.0], k=3)
    0.8154648767857288
    """
    if not retrieved or not relevance_scores or k <= 0:
        return 0.0

    top_k_scores = relevance_scores[:k]

    def dcg(scores: List[float]) -> float:
        return sum(
            rel / math.log2(i + 2)  # i+2 because i is 0-indexed, log2(rank+1)
            for i, rel in enumerate(scores)
        )

    actual_dcg = dcg(top_k_scores)
    if actual_dcg == 0.0:
        return 0.0

    # Ideal DCG: sort scores in descending order
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def ap(retrieved: List[str], relevant: List[str]) -> float:
    """
    Average Precision (AP): mean of precision values at each relevant item's rank.

    AP summarizes the precision-recall curve into a single number and rewards
    systems that rank relevant items higher. Used as a component of MAP
    (Mean Average Precision) in IR evaluation.

    Formula:
        AP = (1 / |relevant|) * sum_{k: retrieved[k] is relevant} Precision@k

    Parameters
    ----------
    retrieved : List[str]
        Ranked list of retrieved strings (position 0 = highest ranked).
    relevant : List[str]
        List of gold/relevant strings.

    Returns
    -------
    float
        Average Precision in [0.0, 1.0]. Returns 0.0 if `relevant` is empty.

    Examples
    --------
    >>> ap(["a b c", "x y z", "d e f"], ["a b c", "d e f"])
    0.8333333333333333
    """
    if not relevant or not retrieved:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, ret in enumerate(retrieved, start=1):
        if any(is_relevant(ret, gold) for gold in relevant):
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return 0.0
    return precision_sum / len(relevant)


# ---------------------------------------------------------------------------
# Answer normalization and answer-level metrics
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """
    Normalize an answer string for comparison.

    Applies the standard normalization used in SQuAD and HotpotQA evaluation:
    1. Lowercase
    2. Remove articles (a, an, the)
    3. Remove punctuation
    4. Collapse whitespace

    Reference: Rajpurkar et al. (2016). "SQuAD: 100,000+ Questions for Machine
    Comprehension of Text." EMNLP 2016. https://arxiv.org/abs/1606.05250

    Parameters
    ----------
    text : str
        Raw text to normalize.

    Returns
    -------
    str
        Normalized text string.

    Examples
    --------
    >>> normalize_answer("The quick brown fox!")
    'quick brown fox'
    >>> normalize_answer("  A  cat   ")
    'cat'
    """
    # Lowercase
    text = text.lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    text = ' '.join(text.split())
    return text


def exact_match(prediction: str, gold: str) -> bool:
    """
    Normalized Exact Match (EM): whether prediction exactly equals gold after normalization.

    Both strings are normalized via `normalize_answer` before comparison.
    This is the standard exact-match metric used in SQuAD, TriviaQA, etc.

    Reference: Rajpurkar et al. (2016). "SQuAD: 100,000+ Questions for Machine
    Comprehension of Text." EMNLP 2016. https://arxiv.org/abs/1606.05250

    Parameters
    ----------
    prediction : str
        The model's predicted answer.
    gold : str
        The reference gold answer.

    Returns
    -------
    bool
        True if the normalized strings are identical.

    Examples
    --------
    >>> exact_match("  the CAT  ", "cat")
    True
    >>> exact_match("a big cat", "cat")
    False
    """
    return normalize_answer(prediction) == normalize_answer(gold)


def _get_tokens(text: str) -> List[str]:
    """Return normalized tokens from text."""
    return normalize_answer(text).split()


def token_precision(prediction: str, gold: str) -> float:
    """
    Token-level Precision: fraction of prediction tokens that appear in gold.

    Formula:
        Token Precision = |common_tokens| / |prediction_tokens|

    Both strings are normalized before tokenization. Token counts use bag-of-words
    (Counter) to handle repeated tokens correctly.

    Reference: Rajpurkar et al. (2016). "SQuAD: 100,000+ Questions for Machine
    Comprehension of Text." EMNLP 2016. https://arxiv.org/abs/1606.05250

    Parameters
    ----------
    prediction : str
        The model's predicted answer.
    gold : str
        The reference gold answer.

    Returns
    -------
    float
        Token precision in [0.0, 1.0]. Returns 0.0 if prediction is empty.
    """
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(gold)
    if not pred_tokens:
        return 0.0
    if not gold_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = sum((pred_counter & gold_counter).values())
    return common / len(pred_tokens)


def token_recall(prediction: str, gold: str) -> float:
    """
    Token-level Recall: fraction of gold tokens that appear in prediction.

    Formula:
        Token Recall = |common_tokens| / |gold_tokens|

    Both strings are normalized before tokenization. Token counts use bag-of-words
    (Counter) to handle repeated tokens correctly.

    Reference: Rajpurkar et al. (2016). "SQuAD: 100,000+ Questions for Machine
    Comprehension of Text." EMNLP 2016. https://arxiv.org/abs/1606.05250

    Parameters
    ----------
    prediction : str
        The model's predicted answer.
    gold : str
        The reference gold answer.

    Returns
    -------
    float
        Token recall in [0.0, 1.0]. Returns 0.0 if gold is empty.
    """
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(gold)
    if not gold_tokens:
        return 0.0
    if not pred_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = sum((pred_counter & gold_counter).values())
    return common / len(gold_tokens)


def token_f1(prediction: str, gold: str) -> float:
    """
    Token-level F1 Score: harmonic mean of token precision and token recall.

    This is the primary answer-quality metric used in SQuAD and HotpotQA. It
    is more lenient than exact match, rewarding partial credit when the prediction
    overlaps with the gold answer in terms of tokens.

    Formula:
        F1 = 2 * (precision * recall) / (precision + recall)

    Both strings are normalized via `normalize_answer` before tokenization.
    Token counts use bag-of-words (Counter) to handle repeated tokens.

    Reference: Rajpurkar et al. (2016). "SQuAD: 100,000+ Questions for Machine
    Comprehension of Text." EMNLP 2016. https://arxiv.org/abs/1606.05250

    Parameters
    ----------
    prediction : str
        The model's predicted answer.
    gold : str
        The reference gold answer.

    Returns
    -------
    float
        Token F1 in [0.0, 1.0]. Returns 0.0 if either string is empty after
        normalization.

    Examples
    --------
    >>> token_f1("the cat sat on the mat", "the cat")
    0.5714285714285715
    >>> token_f1("cat", "cat")
    1.0
    """
    p = token_precision(prediction, gold)
    r = token_recall(prediction, gold)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def mean_metric(values: List[float]) -> float:
    """
    Compute the arithmetic mean of a list of metric values.

    Parameters
    ----------
    values : List[float]
        Metric values to average. May be empty.

    Returns
    -------
    float
        Arithmetic mean. Returns 0.0 for empty list.

    Examples
    --------
    >>> mean_metric([0.8, 0.9, 1.0])
    0.9
    >>> mean_metric([])
    0.0
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_metric_suite(
    retrieved: List[str],
    relevant: List[str],
    gold_answer: str,
    predicted_answer: str,
) -> Dict[str, float]:
    """
    Compute the full suite of retrieval and answer-level metrics in one call.

    This is the primary entry point for evaluating a single query's results.
    Computes all standard IR and QA metrics and returns them in a flat dict.

    Retrieval metrics evaluate the quality of the retrieved memory set.
    Answer metrics evaluate the quality of the generated answer.

    Parameters
    ----------
    retrieved : List[str]
        Ranked list of retrieved memory strings (position 0 = highest ranked).
    relevant : List[str]
        List of gold/relevant memory strings that should have been recalled.
    gold_answer : str
        The reference gold answer to the query.
    predicted_answer : str
        The model's predicted answer to the query.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - recall_at_1, recall_at_3, recall_at_5, recall_at_10
        - mrr
        - ndcg_at_5, ndcg_at_10
        - average_precision
        - exact_match (0.0 or 1.0)
        - token_f1, token_precision, token_recall

    Examples
    --------
    >>> results = compute_metric_suite(
    ...     retrieved=["Alice works at Acme Corp", "Bob likes coffee"],
    ...     relevant=["Alice works at Acme Corp"],
    ...     gold_answer="Acme Corp",
    ...     predicted_answer="Alice works at Acme Corp",
    ... )
    >>> results["recall_at_1"]
    1.0
    >>> results["mrr"]
    1.0
    """
    # Compute binary relevance scores for NDCG
    relevance_scores = [
        1.0 if any(is_relevant(ret, gold) for gold in relevant) else 0.0
        for ret in retrieved
    ]

    return {
        "recall_at_1":       recall_at_k(retrieved, relevant, k=1),
        "recall_at_3":       recall_at_k(retrieved, relevant, k=3),
        "recall_at_5":       recall_at_k(retrieved, relevant, k=5),
        "recall_at_10":      recall_at_k(retrieved, relevant, k=10),
        "mrr":               mrr(retrieved, relevant),
        "ndcg_at_5":         ndcg_at_k(retrieved, relevance_scores, k=5),
        "ndcg_at_10":        ndcg_at_k(retrieved, relevance_scores, k=10),
        "average_precision": ap(retrieved, relevant),
        "exact_match":       float(exact_match(predicted_answer, gold_answer)),
        "token_f1":          token_f1(predicted_answer, gold_answer),
        "token_precision":   token_precision(predicted_answer, gold_answer),
        "token_recall":      token_recall(predicted_answer, gold_answer),
    }


# ---------------------------------------------------------------------------
# Cost-Efficiency Metrics
# ---------------------------------------------------------------------------

def tokens_per_correct(total_tokens: int, correct: int) -> float:
    """Tokens spent per correct answer. Lower is more efficient."""
    return total_tokens / max(correct, 1)


def cost_efficiency_ratio(score: float, tokens_per_query: float) -> float:
    """Score normalized by token cost. Higher is more efficient.

    Defined as: score / log2(tokens_per_query + 1)
    Using log because doubling tokens should not halve efficiency —
    there are diminishing returns to token spending.
    """
    return score / max(math.log2(tokens_per_query + 1), 0.001)


def compute_cost_metrics(total_tokens: int, total_queries: int,
                         correct: int, total: int) -> dict:
    """Compute all cost-efficiency metrics.

    Returns dict with:
      tokens_per_query: average tokens returned per recall
      tokens_per_correct: tokens spent per correct answer
      cost_efficiency: score / log2(tokens_per_query)
      score: correct / total
    """
    score = correct / max(total, 1)
    tpq = total_tokens / max(total_queries, 1)
    return {
        'tokens_per_query': tpq,
        'tokens_per_correct': tokens_per_correct(total_tokens, correct),
        'cost_efficiency': cost_efficiency_ratio(score, tpq),
        'score': score,
        'total_tokens': total_tokens,
        'total_queries': total_queries,
        'correct': correct,
        'total': total,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running metrics self-test...")

    # --- is_relevant ---
    assert is_relevant("The cat sat on the mat", "cat sat on the mat"), \
        "substring containment should match"
    assert not is_relevant("Dogs are mammals", "cats are fish", threshold=0.7), \
        "low overlap should not match"
    assert is_relevant("quick brown fox jumps", "quick brown fox", threshold=0.5), \
        "high token overlap should match"

    # --- recall_at_k ---
    ret = ["The answer is Paris", "London is in England", "Berlin is the capital"]
    rel = ["Paris", "Berlin"]
    assert recall_at_k(ret, rel, k=1) == 0.5, "recall@1 should be 0.5"
    assert recall_at_k(ret, rel, k=3) == 1.0, "recall@3 should be 1.0"
    assert recall_at_k([], rel, k=5) == 0.0, "empty retrieved -> 0.0"

    # --- precision_at_k ---
    p1 = precision_at_k(ret, rel, k=3)
    assert abs(p1 - 2/3) < 1e-9, f"precision@3 should be ~0.667, got {p1}"

    # --- mrr ---
    r1 = mrr(["irrelevant stuff", "Paris is the capital of France", "other"], ["Paris"])
    assert abs(r1 - 0.5) < 1e-9, f"MRR should be 0.5, got {r1}"
    assert mrr([], ["Paris"]) == 0.0, "empty retrieved -> 0.0"
    assert mrr(["no match here"], ["Paris"]) == 0.0, "no match -> 0.0"

    # --- ndcg_at_k ---
    scores = [1.0, 0.0, 1.0]
    n1 = ndcg_at_k(["a", "b", "c"], scores, k=3)
    assert 0.0 < n1 <= 1.0, f"NDCG should be in (0,1], got {n1}"
    # Perfect ranking
    n_perfect = ndcg_at_k(["a", "b", "c"], [1.0, 1.0, 0.0], k=2)
    assert abs(n_perfect - 1.0) < 1e-9, f"Perfect NDCG@2 should be 1.0, got {n_perfect}"

    # --- ap ---
    ap_val = ap(["a b c", "x y z", "d e f"], ["a b c", "d e f"])
    # Hits at rank 1 and 3: P@1=1/1, P@3=2/3 -> AP=(1 + 2/3)/2 = 5/6
    assert abs(ap_val - 5/6) < 1e-9, f"AP should be ~0.833, got {ap_val}"

    # --- normalize_answer ---
    assert normalize_answer("The Quick Brown Fox!") == "quick brown fox", \
        "normalize_answer failed"
    assert normalize_answer("  A  cat   ") == "cat", "normalize_answer strip articles failed"

    # --- exact_match ---
    assert exact_match("  the CAT  ", "cat"), "exact_match should be True after normalization"
    assert exact_match("The cat", "cat"), "exact_match: 'the' is an article and gets stripped"
    assert not exact_match("a big cat", "cat"), "exact_match should be False for non-matching"

    # --- token_f1 ---
    f1 = token_f1("cat sat", "cat sat")
    assert abs(f1 - 1.0) < 1e-9, f"token_f1 should be 1.0 for identical strings, got {f1}"
    f1_partial = token_f1("cat dog bird", "cat fish")
    assert 0.0 < f1_partial < 1.0, f"token_f1 partial match should be in (0,1), got {f1_partial}"
    assert token_f1("", "cat") == 0.0, "empty prediction -> 0.0"
    assert token_f1("cat", "") == 0.0, "empty gold -> 0.0"

    # --- token_precision / token_recall ---
    assert abs(token_precision("cat dog", "cat") - 0.5) < 1e-9, "precision should be 0.5"
    assert abs(token_recall("cat", "cat dog") - 0.5) < 1e-9, "recall should be 0.5"

    # --- mean_metric ---
    assert abs(mean_metric([0.8, 0.9, 1.0]) - 0.9) < 1e-9, "mean_metric failed"
    assert mean_metric([]) == 0.0, "empty list -> 0.0"

    # --- compute_metric_suite ---
    suite = compute_metric_suite(
        retrieved=["Alice works at Acme Corp", "Bob likes coffee", "Charlie is 30"],
        relevant=["Alice works at Acme Corp"],
        gold_answer="Acme Corp",
        predicted_answer="Alice works at Acme Corp",
    )
    assert suite["recall_at_1"] == 1.0, "suite recall@1 should be 1.0"
    assert suite["mrr"] == 1.0, "suite MRR should be 1.0"
    assert suite["exact_match"] == 0.0, "suite EM should be 0.0 (partial match only)"
    assert suite["token_f1"] > 0.0, "suite token_f1 should be > 0"
    assert set(suite.keys()) == {
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "mrr", "ndcg_at_5", "ndcg_at_10", "average_precision",
        "exact_match", "token_f1", "token_precision", "token_recall",
    }, "suite keys mismatch"

    print("All assertions passed.")
