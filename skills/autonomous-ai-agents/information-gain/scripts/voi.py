#!/usr/bin/env python3
"""voi.py — pure value-of-information scoring, gating, ranking, and diversity.

No I/O, no network, no model calls. Every function here is deterministic and
unit-testable in isolation. The model-calling stages live in pipeline.py and the
orchestration loop in infogain.py.

Scoring is grounded in the decision-theoretic Expected Value of Sample Information
(EVSI) and expected-information-gain literature (see references/methodology.md):

    value-of-answering EVSI(q) ≈ Σ_a P(a)·response_change(a)·stakes(a)  # regret avoided
    uncertainty U = normalized_entropy(P(a))·(1 − derivable_prob)       # reducible/epistemic
    exploration value(q) = √(U · EVSI)                                  # the ranking number

`value` is 0 if EITHER the uncertainty gate U or the EVSI is 0 — encoding Howard's
"information has value only if it could change the optimal decision" plus the necessary
conditions (must be uncertain, must be reducible). The geometric mean keeps it on an
interpretable ~0..1 scale (so absolute thresholds like 0.40 are meaningful).

Vocabulary note: the per-answer key `delta_plan` is the legacy name for the response-change
(how much the answer would change your response) — kept stable to avoid churn.
"""

import math
import re

EPS = 1e-9


# ── primitives ───────────────────────────────────────────────────────────────


def clamp01(x):
    """Coerce x to a float in [0, 1]; non-numeric → 0.0."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x):
        return 0.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def normalize_probs(weights):
    """Clamp negatives to 0 and normalize to sum 1.

    Empty → []. All-zero (or empty after clamp) → uniform. This keeps a
    well-formed categorical distribution even when the model emits junk weights.
    """
    w = [max(0.0, float(x)) if _is_num(x) else 0.0 for x in weights]
    n = len(w)
    if n == 0:
        return []
    s = sum(w)
    if s <= 0:
        return [1.0 / n] * n
    return [x / s for x in w]


def _is_num(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def normalized_entropy(probs):
    """Shannon entropy of a categorical distribution, normalized to [0, 1].

    H(P) / log(n). Returns 0 for ≤1 outcome (no uncertainty). Higher = more
    spread = more (apparent) answer-uncertainty.
    """
    p = normalize_probs(probs)
    n = len(p)
    if n <= 1:
        return 0.0
    h = 0.0
    for x in p:
        if x > 0:
            h -= x * math.log(x)
    return h / math.log(n)


# ── uncertainty, EVSI, value ─────────────────────────────────────────────────


def uncertainty(answers, derivable_prob):
    """Reducible/epistemic uncertainty U for a question.

    U = normalized_entropy(P(answers)) · (1 - derivable_prob).

    The entropy term captures answer-spread; the derivability discount removes
    uncertainty that is actually resolvable from the prompt alone (if the answer
    is derivable, asking buys nothing). This is *measured* from simulated answers
    rather than self-rated by the model (CLAMBER: LLMs are poor self-judges of
    ambiguity).
    """
    probs = [a.get("prob", 0.0) for a in (answers or [])]
    return clamp01(normalized_entropy(probs)) * (1.0 - clamp01(derivable_prob))


def evsi(answers):
    """Monte-Carlo EVSI estimate: Σ_a P(a) · Δplan(a) · stakes(a), in [0, 1].

    `answers`: list of dicts with keys `prob`, `delta_plan`, `stakes` (each a
    number; prob need not be pre-normalized). This is the probability-weighted
    expected plan-change-times-stakes — the "regret avoided" by knowing the
    answer (Rao & Daumé 2018 EVPI form, specialized).
    """
    if not answers:
        return 0.0
    probs = normalize_probs([a.get("prob", 0.0) for a in answers])
    total = 0.0
    for p, a in zip(probs, answers):
        total += p * clamp01(a.get("delta_plan", 0.0)) * clamp01(a.get("stakes", 0.0))
    return clamp01(total)


def question_value(u, e):
    """Intrinsic value if resolved = √(U · value-of-answering). Geometric mean of the
    uncertainty gate and the EVSI (value of answering)."""
    return math.sqrt(clamp01(u) * clamp01(e))


def is_gated_out(u, e, eps=EPS):
    """Discard if any necessary condition is ~0: no reducible uncertainty, or no
    expected (probability-weighted) plan-change-times-stakes."""
    return clamp01(u) <= eps or clamp01(e) <= eps


def score_record(rec):
    """Compute and attach u, evsi, value to a question record in place.

    Expects `rec` to carry `answers` (list of {answer, prob, delta_plan, stakes})
    and `derivable_prob`. Returns the same dict with `u`, `evsi`, `value`,
    `gated_out`, and `modal_answer` filled in.
    """
    answers = rec.get("answers") or []
    u = uncertainty(answers, rec.get("derivable_prob", 0.0))
    e = evsi(answers)
    rec["u"] = u
    rec["evsi"] = e  # value of answering
    rec["value"] = question_value(u, e)  # the ranking number = √(U · EVSI)
    rec["gated_out"] = is_gated_out(u, e)
    rec["modal_answer"] = modal_answer(answers)
    return rec


def score_breakdown(rec):
    """Return the arithmetic behind a question's value, for 'show your work' output.

    Pure: reads rec['answers'] ({prob, delta_plan, stakes}) and rec['derivable_prob'].
    `u`, `evsi`, and `value` come from the same canonical functions as score_record,
    so the breakdown never drifts from the real score; the per-answer `evsi_terms`
    are the P·Δplan·stakes contributions shown for explanation.
    """
    answers = rec.get("answers") or []
    raw_probs = [a.get("prob", 0.0) for a in answers]
    probs = normalize_probs(raw_probs)
    ent = normalized_entropy(raw_probs)
    derivable = clamp01(rec.get("derivable_prob", 0.0))
    u = uncertainty(answers, derivable)
    terms = []
    for p, a in zip(probs, answers):
        dp, st = clamp01(a.get("delta_plan", 0.0)), clamp01(a.get("stakes", 0.0))
        terms.append({"answer": a.get("answer", ""), "p": round(p, 4),
                      "delta_plan": dp, "stakes": st, "term": round(p * dp * st, 4)})
    e = evsi(answers)
    return {
        "entropy": round(ent, 4),
        "derivable_prob": derivable,
        "u": round(u, 4),
        "evsi_terms": terms,
        "evsi": round(e, 4),  # value of answering
        "value": round(question_value(u, e), 4),  # exploration value = √(U · EVSI)
    }


# ── classification & defaults ────────────────────────────────────────────────


def classify(value, pre_answer_threshold, discard_threshold):
    """Map a value to a recommendation bucket."""
    if value >= pre_answer_threshold:
        return "PRE_ANSWER"
    if value >= discard_threshold:
        return "ASSUME_DEFAULT"
    return "SKIP"


def modal_answer(answers):
    """The highest-probability projected answer (the suggested default assumption)."""
    if not answers:
        return None
    probs = normalize_probs([a.get("prob", 0.0) for a in answers])
    idx = max(range(len(answers)), key=lambda i: probs[i])
    out = dict(answers[idx])
    out["prob"] = probs[idx]
    return out


# ── similarity, dedup, diversity (MMR) ───────────────────────────────────────


def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def text_jaccard(a, b):
    """Jaccard overlap of word tokens, in [0, 1]."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def question_similarity(a, b):
    """Redundancy between two question records.

    Two questions that resolve the SAME hidden assumption/latent (`target`) are
    fully redundant — their joint information ≈ the max, not the sum (BatchBALD).
    Otherwise fall back to token overlap of the question text.
    """
    ta = (a.get("target") or "").strip().lower()
    tb = (b.get("target") or "").strip().lower()
    if ta and tb and ta == tb:
        return 1.0
    return text_jaccard(a.get("question", ""), b.get("question", ""))


def hierarchical_similarity(a, b, family_sim=0.5):
    """Three-tier redundancy kernel: `family > target > question`.

    same `target` → 1.0 (fully redundant — collapse) · else same `family` → `family_sim`
    (partial — keep both, but MMR spreads selection across families) · else token overlap.
    Generalizes `question_similarity`; with no `family` on the records it reduces to it.
    Use as the `sim_fn` for MMR when the families layer is on.
    """
    ta = (a.get("target") or "").strip().lower()
    tb = (b.get("target") or "").strip().lower()
    if ta and tb and ta == tb:
        return 1.0
    fa = (a.get("family") or "").strip().lower()
    fb = (b.get("family") or "").strip().lower()
    if fa and fb and fa == fb:
        return clamp01(family_sim)
    return text_jaccard(a.get("question", ""), b.get("question", ""))


def is_duplicate(candidate, seen, threshold=0.8):
    """True if `candidate` is ~the same as any record in `seen`."""
    return any(question_similarity(candidate, s) >= threshold for s in seen)


def dedupe(records, threshold=0.8):
    """Greedily drop near-duplicate records, keeping first occurrence."""
    kept = []
    for r in records:
        if not is_duplicate(r, kept, threshold):
            kept.append(r)
    return kept


def mmr_select(candidates, k, lam=0.4, sim_fn=question_similarity):
    """Greedy Maximal Marginal Relevance selection for a diverse, high-value set.

    Each step picks the candidate maximizing  value − λ · max_sim_to_already_kept.
    Independent top-k would over-pick redundant questions (BatchBALD); MMR with a
    submodular-style penalty gives a diverse, non-overlapping bucket.

    `candidates`: records carrying a numeric `value`. Returns up to k records in
    selection order (highest marginal value first).
    """
    pool = list(candidates)
    selected = []
    while pool and len(selected) < k:
        if not selected:
            best = max(pool, key=lambda c: c.get("value", 0.0))
        else:
            def marginal(c):
                red = max(sim_fn(c, s) for s in selected)
                return c.get("value", 0.0) - lam * red

            best = max(pool, key=marginal)
        selected.append(best)
        pool.remove(best)
    return selected


# ── ranking & bucket assembly ────────────────────────────────────────────────


def rank_and_select(records, *, discard_threshold, pre_answer_threshold,
                    hard_cap, mmr_lambda=0.4, redundancy_threshold=0.8,
                    sim_fn=question_similarity):
    """Gate → keep (value ≥ discard) → collapse redundant clusters → MMR-diversify
    to hard_cap → classify.

    Pure: takes already-scored records (each with `value`, `u`, `evsi`,
    `gated_out`, `answers`, `modal_answer`), returns (bucket, discarded). `bucket`
    is the ranked, deduplicated, diversified, classified keepers; `discarded`
    holds everything else, each tagged with a `recommendation`:
      SKIP (low value / gated out) · REDUNDANT (same latent as a kept higher-value
      question) · OVERFLOW (valuable + distinct but beyond hard_cap).
    """
    survivors, discarded = [], []
    for r in records:
        if r.get("gated_out") or r.get("value", 0.0) < discard_threshold:
            r["recommendation"] = "SKIP"
            discarded.append(r)
        else:
            survivors.append(r)

    survivors.sort(key=lambda r: r.get("value", 0.0), reverse=True)

    # Collapse redundant clusters: keep the highest-value representative per latent
    # (survivors are value-desc, so dedupe-keep-first keeps the best). BatchBALD:
    # two questions resolving the same hidden variable have joint value ≈ the max.
    representatives = dedupe(survivors, threshold=redundancy_threshold)
    rep_ids = {id(r) for r in representatives}
    for r in survivors:
        if id(r) not in rep_ids:
            r["recommendation"] = "REDUNDANT"
            discarded.append(r)

    bucket = mmr_select(representatives, hard_cap, lam=mmr_lambda, sim_fn=sim_fn)
    bucket.sort(key=lambda r: r.get("value", 0.0), reverse=True)
    bucket_ids = {id(r) for r in bucket}
    for r in representatives:
        if id(r) not in bucket_ids:
            r["recommendation"] = "OVERFLOW"
            discarded.append(r)

    for r in bucket:
        r["recommendation"] = classify(r["value"], pre_answer_threshold, discard_threshold)
    return bucket, discarded


def best_value(records):
    """Highest `value` among records (0.0 if empty) — used for the refill stop rule."""
    return max((r.get("value", 0.0) for r in records), default=0.0)
