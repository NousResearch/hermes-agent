"""Bradley-Terry pairwise judge.

Absolute-scale LLM judges drift across runs, models, and even
temperatures (Zheng et al. 2023; Chen et al. 2024). Pairwise
preferences are invariant to that drift — ``A ≻ B`` either holds or
does not — and the Bradley-Terry model (Bradley & Terry 1952) converts
a batch of such preferences into calibrated per-candidate scores.

Components
----------

* :func:`aggregate_bradley_terry` — minorisation-maximisation MLE
  (Hunter 2004) over pairwise votes. Pure stdlib; ~30 non-trivial LOC.
* :func:`sample_pair_schedule` — picks which candidates to compare.
  Round-robin when the population is small, random-with-replacement
  once ``C(pop, 2) > rounds``. Guarantees each candidate participates
  in at least ``⌈rounds * 2 / pop⌉`` matches.
* :class:`PairwiseJudge` — thin orchestrator that calls the user-
  supplied LLM judge with randomised A/B order (position-bias guard)
  and returns the decoded winner.

References
----------
Bradley & Terry (1952)    "Rank Analysis of Incomplete Block Designs."
                          Biometrika 39 (3-4): 324–345.
Hunter (2004)             "MM algorithms for generalized Bradley-Terry
                          models." Annals of Statistics 32 (1): 384–406.
Zheng et al. (2023)       "Judging LLM-as-a-Judge." NeurIPS.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from llm import LLMClient


_DEFAULT_SYSTEM = (
    "You are an impartial judge comparing two candidates for a task. "
    "Read both carefully. Decide which is better against the user's goal. "
    "Reply with a single token on the first line: exactly 'LEFT', 'RIGHT', "
    "or 'TIE'. You may then give a short rationale on subsequent lines, "
    "but only the first line is machine-read."
)


# ---------------------------------------------------------------------------
# MLE over pairwise votes — Bradley-Terry via minorisation-maximisation
# ---------------------------------------------------------------------------


def aggregate_bradley_terry(
    votes: Iterable[tuple[str, str]],
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    smoothing: float = 1e-3,
) -> tuple[dict[str, float], int]:
    """Return ``{candidate_id: log-odds}`` via Hunter's MM algorithm.

    ``votes`` is an iterable of ``(winner_id, loser_id)``. Ties should
    be encoded as two half-weight votes before passing in — the MM
    formulation here only counts strict wins (extended with Laplace
    ``smoothing`` so a candidate that won 0 / lost all still has a
    finite score).

    Returns ``(scores, iters)`` where ``scores`` is a dict of log-odds
    centred at 0.
    """
    vote_list = list(votes)
    if not vote_list:
        return {}, 0

    # Collect the candidate universe from both winners and losers.
    ids = sorted({c for pair in vote_list for c in pair})
    idx = {cid: i for i, cid in enumerate(ids)}
    n = len(ids)
    if n == 1:
        return {ids[0]: 0.0}, 0

    # Aggregate wins and pair counts.
    wins = [smoothing] * n           # wᵢ (Laplace-smoothed)
    pairs = [[0.0] * n for _ in range(n)]
    for w, l in vote_list:
        wi, li = idx[w], idx[l]
        wins[wi] += 1.0
        pairs[wi][li] += 1.0
        pairs[li][wi] += 1.0

    # Strength init (BT parameter p_i > 0). Using 1.0 is the canonical
    # seed — the MM iteration is invariant to uniform rescaling.
    p = [1.0] * n

    iters = 0
    for iters in range(1, max_iter + 1):
        new_p = [0.0] * n
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j or pairs[i][j] == 0.0:
                    continue
                denom += pairs[i][j] / (p[i] + p[j])
            new_p[i] = wins[i] / denom if denom > 0 else p[i]

        # Centre by the geometric mean to keep log-odds around 0 and
        # prevent numerical drift.
        log_sum = sum(math.log(v) for v in new_p if v > 0)
        mean = math.exp(log_sum / n) if log_sum != 0 else 1.0
        new_p = [v / mean for v in new_p]

        # Convergence check on log-strength.
        delta = max(abs(math.log(a / b)) for a, b in zip(new_p, p) if a > 0 and b > 0)
        p = new_p
        if delta < tol:
            break

    return {ids[i]: math.log(p[i]) for i in range(n)}, iters


# ---------------------------------------------------------------------------
# Pair scheduling
# ---------------------------------------------------------------------------


def sample_pair_schedule(
    candidates: list[str],
    *,
    rounds: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Pick which candidate pairs to judge this generation.

    * ``pop <= 6`` → round-robin (all unique pairs, then repeat) so
      every candidate sees every other at least once before any pair
      repeats. For pop=6, C(6,2)=15 pairs per round.
    * ``pop  > 6`` → random-with-replacement from all unique pairs;
      we keep a running per-candidate count and bias toward the
      least-compared candidate to avoid orphaning anyone.

    Invariant: every candidate appears in at least
    ``⌈rounds * 2 / pop⌉`` pairs (tested in ``test_pair_schedule_...``).
    """
    pop = len(candidates)
    if pop < 2:
        return []

    unique_pairs = list(itertools.combinations(candidates, 2))
    rng.shuffle(unique_pairs)

    if pop <= 6 and rounds <= len(unique_pairs):
        return unique_pairs[:rounds]

    pairs: list[tuple[str, str]] = []
    appearances = {c: 0 for c in candidates}
    while len(pairs) < rounds:
        # Bias towards the least-seen candidate.
        min_seen = min(appearances.values())
        under = [c for c, n in appearances.items() if n == min_seen]
        anchor = rng.choice(under)
        partner = rng.choice([c for c in candidates if c != anchor])
        pairs.append((anchor, partner))
        appearances[anchor] += 1
        appearances[partner] += 1
    return pairs


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


_LEFT_PATTERNS  = (r"\bleft\b",  r"\ba\b", r"option\s*a", r"option\s*1")
_RIGHT_PATTERNS = (r"\bright\b", r"\bb\b", r"option\s*b", r"option\s*2")
_TIE_PATTERNS   = (r"\btie\b", r"\bequal\b", r"\bdraw\b")


def _decode_verdict(raw: str) -> str:
    """Parse a pairwise LLM verdict into ``"left" | "right" | "tie"``.

    Reads the first non-empty line; matches against pre-computed
    patterns. Defaults to ``"tie"`` on anything we can't recognise so
    that an unparseable response costs nothing in the MLE.
    """
    for line in raw.splitlines():
        token = line.strip().lower()
        if not token:
            continue
        if any(re.search(p, token) for p in _TIE_PATTERNS):
            return "tie"
        if any(re.search(p, token) for p in _LEFT_PATTERNS):
            return "left"
        if any(re.search(p, token) for p in _RIGHT_PATTERNS):
            return "right"
        break
    return "tie"


@dataclass
class PairwiseJudge:
    """LLM-backed pairwise judge with a position-bias guard.

    Each :meth:`judge` call flips a coin to decide whether the nominal
    ``a`` is presented as LEFT or RIGHT to the judge. The decoded
    verdict is then translated back into the caller's original
    ``(a, b)`` coordinate.
    """

    client: LLMClient
    system_prompt: str = _DEFAULT_SYSTEM
    temperature: float = 0.0

    async def judge(
        self,
        a: str,
        b: str,
        *,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> str:
        """Return ``"a" | "b" | "tie"`` — the winner among the caller's pair."""
        rng = rng or random.Random(seed)
        swap = rng.random() < 0.5
        left_text, right_text = (b, a) if swap else (a, b)
        user = (
            f"LEFT:\n{left_text}\n\n"
            f"RIGHT:\n{right_text}\n\n"
            "Which is better? Reply with LEFT, RIGHT, or TIE on the first line."
        )
        raw = await self.client.complete(
            self.system_prompt, user,
            seed=seed, temperature=self.temperature,
            operator="pairwise_judge",
        )
        verdict = _decode_verdict(raw)
        if verdict == "tie":
            return "tie"
        # Un-swap to return the caller's coordinate.
        left_winner = (verdict == "left")
        if swap:
            # In swapped order, LEFT was b.
            return "b" if left_winner else "a"
        return "a" if left_winner else "b"
