#!/usr/bin/env python3
"""pairwise.py — pure aggregation of pairwise comparisons into per-item [0,1] strengths.

No I/O, no network, no model calls. Deterministic and unit-testable in isolation (like voi.py).

Why this exists (comparative elicitation, #24): a model compares two answers ("which would
change the response more?") far more reliably than it scores one answer in isolation on an
absolute 0-1 scale. The absolute judge's within-task ranking is the skill's one measured
weakness (per-prompt Spearman ρ≈0.34). This module turns a set of pairwise verdicts into a
latent strength per item via Bradley-Terry, then maps strengths onto the SAME [0,1] scale the
absolute judge produced — so the pairwise `delta_plan`/`stakes` are drop-in replacements that
voi.evsi/score_record consume unchanged.

Preserving BETWEEN-task scale (the validated strength, ρ≈0.66) is the subtle part. Pure
within-question normalization would map every question's best answer→1 and worst→0, collapsing
cross-question magnitude and destroying the between-task signal. We avoid that with two virtual
ANCHOR items present in every question's comparison set:
    FLOOR   — "the response stays the baseline (no change)"   → maps to 0.0
    CEILING — "a completely different response"               → maps to 1.0
Real answers are placed between the anchors by how decisively they beat FLOOR / lose to CEILING.
A question whose answers barely beat FLOOR (ties allowed) lands near 0 (low EVSI); a question
whose answers strongly beat FLOOR lands high — so the common anchors carry an absolute-ish scale
across questions, while the pairwise forced choices fix the within-question ordering.

Comparison encoding: a comparison is a tuple (i, j, outcome) with outcome in
    1.0 → item i beats item j   ·   0.0 → item j beats item i   ·   0.5 → tie (≈ equal).
The Bradley-Terry MLE is regularized with a phantom opponent (a conjugate-prior pseudo-count) so
the extreme anchors — FLOOR with ~all losses, CEILING with ~all wins — don't diverge to 0/∞.
"""

import math

EPS = 1e-9


def _tally(n, comparisons):
    """Accumulate per-item wins (ties count 0.5 each) and per-pair game counts."""
    wins = [0.0] * n
    games = [[0.0] * n for _ in range(n)]
    for c in comparisons:
        try:
            i, j, o = int(c[0]), int(c[1]), float(c[2])
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= i < n and 0 <= j < n) or i == j:
            continue
        o = 0.0 if o < 0.0 else 1.0 if o > 1.0 else o
        wins[i] += o
        wins[j] += 1.0 - o
        games[i][j] += 1.0
        games[j][i] += 1.0
    return wins, games


def win_fractions(n, comparisons, prior=0.5, prior_weight=2.0):
    """Beta/Laplace-smoothed win fraction per item, in (0, 1). Cheap, robust, monotone — a
    transparent alternative to Bradley-Terry, kept for offline analysis (NOT wired as a
    runtime fallback — production always uses bradley_terry). `prior_weight` pseudo-games at
    rate `prior` keep items with few or extreme records off the 0/1 rails."""
    wins, games = _tally(n, comparisons)
    played = [sum(row) for row in games]
    return [(wins[k] + prior * prior_weight) / (played[k] + prior_weight) for k in range(n)]


def bradley_terry(n, comparisons, iters=200, tol=1e-10, phantom=1.0):
    """Bradley-Terry strengths via the MM algorithm (Hunter 2004), regularized by a phantom
    opponent of fixed strength 1 that has played `phantom` games (half win / half loss) against
    every item — a conjugate prior that keeps the all-win (CEILING) and all-loss (FLOOR) anchors
    finite. Returns a list of positive strengths normalized to geometric-mean 1 (scale-free).

    The update is  p_i ← (w_i + phantom/2) / ( Σ_{j≠i} n_ij/(p_i+p_j) + phantom/(p_i+1) ).
    Strictly increasing in an item's win count, so more wins ⇒ higher strength (tested).
    """
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    wins, games = _tally(n, comparisons)
    p = [1.0] * n
    for _ in range(iters):
        new = [0.0] * n
        for i in range(n):
            denom = phantom / (p[i] + 1.0)  # phantom opponent at fixed strength 1
            for j in range(n):
                if j != i and games[i][j] > 0:
                    denom += games[i][j] / (p[i] + p[j])
            numer = wins[i] + phantom / 2.0
            new[i] = numer / denom if denom > EPS else p[i]
        # normalize to geometric mean 1 to pin the multiplicative gauge freedom
        gm = math.exp(sum(math.log(max(x, EPS)) for x in new) / n)
        new = [x / gm for x in new]
        if max(abs(new[k] - p[k]) for k in range(n)) < tol:
            p = new
            break
        p = new
    return p


def anchored_scores(strengths, floor_idx, ceil_idx):
    """Map positive BT strengths onto [0,1] on the additive log (Elo) scale, pinned so the FLOOR
    anchor → 0 and the CEILING anchor → 1. Real items land in between by how far their strength
    sits above FLOOR relative to CEILING. Returns clamped [0,1] scores aligned with `strengths`.
    Degenerate spread (ceil ≤ floor) → all zeros (treat as no measurable change)."""
    n = len(strengths)
    if n == 0:
        return []
    if not (0 <= floor_idx < n and 0 <= ceil_idx < n):
        return [0.0] * n
    thetas = [math.log(max(s, EPS)) for s in strengths]
    lo, hi = thetas[floor_idx], thetas[ceil_idx]
    if hi - lo <= EPS:
        return [0.0] * n
    out = []
    for t in thetas:
        x = (t - lo) / (hi - lo)
        out.append(0.0 if x < 0.0 else 1.0 if x > 1.0 else x)
    return out


def all_pairs(n):
    """Unordered index pairs (i<j) for an n-item comparison set — what the elicitation enumerates."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]
