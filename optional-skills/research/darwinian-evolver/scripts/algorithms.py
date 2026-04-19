"""Pure evolutionary-algorithm primitives: (μ+λ)-ES, MAP-Elites, NSGA-II.

These functions are side-effect free and deterministic given a seeded
``random.Random`` instance. The evolver driver (``evolver.py``) owns the
population, the fitness evaluator, and the LLM operators; this module is
the library of combinators it composes.

References
----------
Bäck & Schwefel (1993)     "An Overview of Evolutionary Algorithms for
                            Parameter Optimization." Evolutionary
                            Computation, 1(1).
Mouret & Clune (2015)      "Illuminating search spaces by mapping
                            elites." arXiv:1504.04909.
Deb et al. (2002)          "A Fast and Elitist Multi-Objective Genetic
                            Algorithm: NSGA-II." IEEE TEC 6(2).
Auer et al. (2002)         "The Nonstochastic Multiarmed Bandit
                            Problem." SICOMP 32(1).  (Exp3.)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


# ---------------------------------------------------------------------------
# Candidate container
# ---------------------------------------------------------------------------


@dataclass
class Individual:
    """One member of the population.

    ``fitness`` is either a scalar (single-objective) or a dict keyed by
    objective name (multi-objective). ``descriptor`` is the tuple of
    behavioral coordinates used by MAP-Elites binning.
    """

    cid: str
    genome: str
    fitness: float | dict[str, float] = math.nan
    descriptor: tuple = ()
    generation: int = 0
    parents: list[str] = field(default_factory=list)
    operator: str = "seed"


# ---------------------------------------------------------------------------
# Selection operators
# ---------------------------------------------------------------------------


def tournament_select(
    population: list[Individual],
    *,
    k: int = 3,
    n: int = 1,
    rng: random.Random,
    objective: Optional[str] = None,
) -> list[Individual]:
    """Tournament selection with replacement.

    Samples *k* individuals uniformly and returns the best among them;
    repeats *n* times. When fitness is a dict, *objective* is required.
    """
    winners: list[Individual] = []
    for _ in range(n):
        contestants = [rng.choice(population) for _ in range(k)]
        winners.append(max(contestants, key=lambda ind: _scalar(ind, objective)))
    return winners


def rank_select(
    population: list[Individual],
    *,
    n: int = 1,
    rng: random.Random,
    pressure: float = 1.5,
    objective: Optional[str] = None,
) -> list[Individual]:
    """Linear rank-based selection with configurable pressure.

    Pressure 1.0 = uniform; pressure 2.0 = strong bias to the best.
    Invariant: ``1.0 <= pressure <= 2.0``.
    """
    if not 1.0 <= pressure <= 2.0:
        raise ValueError("pressure must be in [1.0, 2.0]")
    ordered = sorted(population, key=lambda ind: _scalar(ind, objective))
    size = len(ordered)
    # linear-rank probabilities
    probs = [
        (2 - pressure) / size + 2 * i * (pressure - 1) / (size * (size - 1))
        for i in range(size)
    ] if size > 1 else [1.0]
    return rng.choices(ordered, weights=probs, k=n)


def _scalar(ind: Individual, objective: Optional[str]) -> float:
    """Extract a scalar value for sort/compare from a fitness record."""
    if isinstance(ind.fitness, dict):
        if objective is None:
            raise ValueError("objective must be provided for multi-objective fitness")
        return float(ind.fitness.get(objective, math.nan))
    return float(ind.fitness)


# ---------------------------------------------------------------------------
# (μ+λ)-ES survival
# ---------------------------------------------------------------------------


def mu_plus_lambda(
    parents: list[Individual],
    offspring: list[Individual],
    *,
    mu: int,
    objective: Optional[str] = None,
) -> list[Individual]:
    """Return the top-*mu* from the combined pool.

    This is the classical plus-replacement (Bäck & Schwefel 1993).
    Elitism is preserved: the best parent is guaranteed to survive.
    """
    combined = list(parents) + list(offspring)
    return sorted(combined, key=lambda ind: -_scalar(ind, objective))[:mu]


# ---------------------------------------------------------------------------
# MAP-Elites
# ---------------------------------------------------------------------------


@dataclass
class MapElitesArchive:
    """Quality-diversity archive binned by behavioral descriptors.

    Each bin stores at most one individual — the best observed so far
    for that behavioral coordinate. The archive size is the number of
    occupied bins; the archive *capacity* is ``∏ bin_counts``.
    """

    bin_counts: tuple[int, ...]
    lows: tuple[float, ...]
    highs: tuple[float, ...]
    objective: Optional[str] = None
    cells: dict[tuple[int, ...], Individual] = field(default_factory=dict)

    def _bin(self, descriptor: Iterable[float]) -> tuple[int, ...]:
        coords = tuple(descriptor)
        if len(coords) != len(self.bin_counts):
            raise ValueError(
                f"descriptor of length {len(coords)} does not match "
                f"archive dimensionality {len(self.bin_counts)}"
            )
        out = []
        for v, lo, hi, n in zip(coords, self.lows, self.highs, self.bin_counts):
            if math.isnan(v):
                out.append(0)
                continue
            clamped = min(max(v, lo), hi - 1e-9)
            span = max(hi - lo, 1e-9)
            out.append(min(int((clamped - lo) / span * n), n - 1))
        return tuple(out)

    def place(self, ind: Individual) -> bool:
        """Insert *ind* into its bin if it beats the current occupant.

        Returns True when the archive was updated.
        """
        key = self._bin(ind.descriptor or (0,) * len(self.bin_counts))
        incumbent = self.cells.get(key)
        if incumbent is None or _scalar(ind, self.objective) > _scalar(incumbent, self.objective):
            self.cells[key] = ind
            return True
        return False

    def sample(self, rng: random.Random, k: int = 1) -> list[Individual]:
        """Uniformly sample *k* occupants, with replacement."""
        if not self.cells:
            return []
        members = list(self.cells.values())
        return [rng.choice(members) for _ in range(k)]

    def best(self, objective: Optional[str] = None) -> Optional[Individual]:
        if not self.cells:
            return None
        obj = objective or self.objective
        return max(self.cells.values(), key=lambda ind: _scalar(ind, obj))

    def coverage(self) -> float:
        capacity = 1
        for c in self.bin_counts:
            capacity *= c
        return len(self.cells) / capacity if capacity else 0.0


# ---------------------------------------------------------------------------
# NSGA-II: fast non-dominated sort + crowding distance
# ---------------------------------------------------------------------------


def dominates(a: dict[str, float], b: dict[str, float], objectives: list[str]) -> bool:
    """Return True iff *a* weakly dominates *b* over *objectives*.

    All objectives are maximised (higher = better). Callers that need
    minimisation should negate their metric at fitness time.
    """
    strictly_better = False
    for k in objectives:
        av, bv = a.get(k, math.nan), b.get(k, math.nan)
        if math.isnan(av) or math.isnan(bv):
            return False
        if av < bv:
            return False
        if av > bv:
            strictly_better = True
    return strictly_better


def fast_non_dominated_sort(
    population: list[Individual],
    objectives: list[str],
) -> list[list[Individual]]:
    """Partition *population* into Pareto fronts (front 0 is the best).

    Implementation follows Deb et al. 2002, §III-A: O(M N²).
    """
    n = len(population)
    fits = []
    for ind in population:
        if not isinstance(ind.fitness, dict):
            raise ValueError("NSGA-II requires dict fitness; got scalar")
        fits.append(ind.fitness)
    dom_count  = [0] * n
    dominated  = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(fits[p], fits[q], objectives):
                dominated[p].append(q)
            elif dominates(fits[q], fits[p], objectives):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return [[population[idx] for idx in front] for front in fronts]


def crowding_distance(front: list[Individual], objectives: list[str]) -> list[float]:
    """Crowding distance per individual within a front (Deb 2002 §III-B).

    Boundary points (best/worst on any objective) get infinite distance
    so they're preserved. Ties on an objective contribute 0.
    """
    size = len(front)
    if size <= 2:
        return [math.inf] * size
    dist = [0.0] * size
    for obj in objectives:
        idx = sorted(range(size), key=lambda i: front[i].fitness[obj])  # type: ignore[index]
        dist[idx[0]]  = math.inf
        dist[idx[-1]] = math.inf
        lo, hi = front[idx[0]].fitness[obj], front[idx[-1]].fitness[obj]  # type: ignore[index]
        span = hi - lo
        if span <= 0:
            continue
        for k in range(1, size - 1):
            prev_f = front[idx[k - 1]].fitness[obj]  # type: ignore[index]
            next_f = front[idx[k + 1]].fitness[obj]  # type: ignore[index]
            dist[idx[k]] += (next_f - prev_f) / span
    return dist


def nsga2_select(
    combined: list[Individual],
    objectives: list[str],
    mu: int,
) -> list[Individual]:
    """Select *mu* survivors via NSGA-II front + crowding-distance rule."""
    fronts = fast_non_dominated_sort(combined, objectives)
    survivors: list[Individual] = []
    for front in fronts:
        if len(survivors) + len(front) <= mu:
            survivors.extend(front)
            continue
        dist = crowding_distance(front, objectives)
        order = sorted(range(len(front)), key=lambda i: -dist[i])
        survivors.extend(front[i] for i in order[: mu - len(survivors)])
        break
    return survivors


# ---------------------------------------------------------------------------
# Exp3 bandit over mutation operators
# ---------------------------------------------------------------------------


@dataclass
class Exp3Bandit:
    """Exp3 (Auer 2002) over a fixed set of arms.

    We use it to pick among mutation operators. ``reward(i, r)`` feeds
    back the observed fitness gain (normalised to [0, 1]). ``pick`` uses
    the current weight distribution to sample an arm. The LLM budget is
    directly visible in arm costs, so cheap operators (paraphrase) get
    more rollouts early and expensive ones (critique-then-edit) only
    fire when the cheap ones plateau.
    """

    arms: list[str]
    gamma: float = 0.2
    weights: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.weights = [1.0] * len(self.arms)

    def _probabilities(self) -> list[float]:
        w_sum = sum(self.weights)
        return [
            (1 - self.gamma) * (w / w_sum) + self.gamma / len(self.arms)
            for w in self.weights
        ]

    def pick(self, rng: random.Random) -> tuple[int, str]:
        probs = self._probabilities()
        idx = rng.choices(range(len(self.arms)), weights=probs, k=1)[0]
        return idx, self.arms[idx]

    def reward(self, idx: int, r: float) -> None:
        r = max(0.0, min(1.0, r))
        probs = self._probabilities()
        estimated = r / probs[idx]
        self.weights[idx] *= math.exp(self.gamma * estimated / len(self.arms))


# ---------------------------------------------------------------------------
# Behavioral descriptors (prompt-oriented)
# ---------------------------------------------------------------------------


def length_bucket(text: str, buckets: int = 8, max_len: int = 2000) -> int:
    """Log-scaled length descriptor in 0..buckets-1."""
    n = min(len(text), max_len)
    if n <= 1:
        return 0
    return min(int(math.log2(max(n, 2)) / math.log2(max_len) * buckets), buckets - 1)


def cot_presence(text: str) -> int:
    """Binary: does the text invoke chain-of-thought scaffolding?"""
    needles = ("let's think", "step by step", "reason step", "explain your reasoning")
    low = text.lower()
    return 1 if any(n in low for n in needles) else 0


def default_prompt_descriptor(text: str) -> tuple[int, int]:
    """Default 2-D descriptor: (length bucket, CoT presence)."""
    return (length_bucket(text), cot_presence(text))
