"""Fitness evaluation: dynamic user fitness loading, async batch eval,
held-out guard, successive halving, budget enforcement.

The user owns the fitness function by writing ``fitness.py`` in the
experiment directory. It must expose ``fitness(candidate, context)``
decorated with :func:`fitness_spec`. We load it via ``importlib`` so the
user doesn't need to package or install anything — a plain script works.

Reward-hacking guard
--------------------
At each generation the runner picks the top-K candidates by training-set
fitness and re-scores them on a *held-out* split seeded deterministically
per candidate. If a candidate's held-out score lags its training score
by more than ``generalization_gap`` (default 15 %), we subtract a
penalty from its archive score so reward-hackers sink below honest but
slightly less optimised competitors.

Successive halving
------------------
When a hard budget is active we don't need to fully evaluate every
offspring — only the promising ones. ``successive_halving`` splits the
budget across rungs: each rung doubles the eval fidelity and keeps the
top half. Follows Li et al. 2017 (Hyperband) without the explore step,
so it's Hyperband-lite. A fidelity-aware fitness function honors
``context["fidelity"]`` (e.g., eval on 25 % of the test set at rung 0).
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from algorithms import Individual


# ---------------------------------------------------------------------------
# Fitness contract
# ---------------------------------------------------------------------------


def fitness_spec(
    *,
    held_out_frac: float = 0.2,
    timeout_s: float = 30.0,
    objectives: Optional[list[str]] = None,
    generalization_gap: float = 0.15,
    judge: str = "scalar",
    pairwise_rounds: int = 40,
    critic: str = "off",
    critic_threshold: float = 0.5,
    critic_top_k: int = 5,
    critic_model: Optional[str] = None,
) -> Callable[[Callable], Callable]:
    """Decorator that attaches evaluator metadata to a user fitness fn.

    Exposed to user code as::

        from evolver_sdk import fitness_spec    # (alias — see sdk.py)

        @fitness_spec(held_out_frac=0.2, objectives=["accuracy", "cost"])
        def fitness(candidate, context): ...

    v0.2 adds four optional metadata keys:

    ``judge``
        ``"scalar"`` (default, existing behaviour) or ``"pairwise"``
        to enable Bradley-Terry pairwise judging. In pairwise mode the
        ``fitness`` function is not called — the LLM judge is the fitness.

    ``pairwise_rounds``
        When ``judge="pairwise"``, the number of pair comparisons per
        generation.

    ``critic`` / ``critic_threshold`` / ``critic_top_k`` / ``critic_model``
        Enable and tune the constitutional reward-hacking critic. See
        ``scripts/critic.py`` for the review schema.

    The metadata is read from ``fn.__evolver_spec__``; the function
    still behaves like a normal callable.
    """

    def deco(fn: Callable) -> Callable:
        fn.__evolver_spec__ = {  # type: ignore[attr-defined]
            "held_out_frac":      float(held_out_frac),
            "timeout_s":          float(timeout_s),
            "objectives":         list(objectives) if objectives else None,
            "generalization_gap": float(generalization_gap),
            "judge":              str(judge),
            "pairwise_rounds":    int(pairwise_rounds),
            "critic":             str(critic),
            "critic_threshold":   float(critic_threshold),
            "critic_top_k":       int(critic_top_k),
            "critic_model":       critic_model,
        }
        return fn

    return deco


def load_fitness(experiment_dir: Path) -> Callable:
    """Dynamically import ``fitness.py`` from *experiment_dir*.

    Returns the ``fitness`` callable. Raises ``FileNotFoundError`` if
    the file is missing and ``AttributeError`` if the ``fitness``
    symbol doesn't exist.

    The experiment directory is prepended to ``sys.path`` before
    import so ``from evolver_sdk import fitness_spec`` (the shim
    created by ``evolver init``) resolves.
    """
    import sys as _sys

    path = experiment_dir / "fitness.py"
    if not path.exists():
        raise FileNotFoundError(f"missing fitness.py at {path}")

    dir_str = str(experiment_dir)
    added = False
    if dir_str not in _sys.path:
        _sys.path.insert(0, dir_str)
        added = True
    try:
        spec = importlib.util.spec_from_file_location(
            f"user_fitness_{experiment_dir.name}", path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load spec from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if added:
            try:
                _sys.path.remove(dir_str)
            except ValueError:
                pass
    if not hasattr(module, "fitness"):
        raise AttributeError(f"fitness.py must define a `fitness` function: {path}")
    return module.fitness


def read_spec(fn: Callable) -> dict:
    """Return the ``fitness_spec`` metadata, with defaults if absent."""
    spec = getattr(fn, "__evolver_spec__", None) or {}
    return {
        "held_out_frac":      spec.get("held_out_frac",      0.2),
        "timeout_s":          spec.get("timeout_s",          30.0),
        "objectives":         spec.get("objectives",         None),
        "generalization_gap": spec.get("generalization_gap", 0.15),
        "judge":              spec.get("judge",              "scalar"),
        "pairwise_rounds":    spec.get("pairwise_rounds",    40),
        "critic":             spec.get("critic",             "off"),
        "critic_threshold":   spec.get("critic_threshold",   0.5),
        "critic_top_k":       spec.get("critic_top_k",       5),
        "critic_model":       spec.get("critic_model",       None),
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalContext:
    """Data passed to each fitness() invocation.

    ``seed`` controls RNG inside the user's fitness (e.g., which held-out
    samples to use). ``fidelity`` is the successive-halving rung, 0..1.
    ``held_out`` flags held-out re-evaluation runs; honest user fitness
    functions typically need this to flip which eval split is used.
    """

    seed: int = 0
    fidelity: float = 1.0
    held_out: bool = False
    extra: dict = field(default_factory=dict)


async def _call_fitness(
    fn: Callable,
    candidate: str,
    context: EvalContext,
    timeout_s: float,
) -> float | dict[str, float]:
    """Invoke the user fitness function with a hard wall-clock timeout.

    If the fitness function is a coroutine we await it directly. If it's
    synchronous we off-load it to the default thread-pool so one blocking
    call can't stall the whole async batch.
    """
    ctx = {
        "seed":     context.seed,
        "fidelity": context.fidelity,
        "held_out": context.held_out,
        **context.extra,
    }
    try:
        if inspect.iscoroutinefunction(fn):
            return await asyncio.wait_for(fn(candidate, ctx), timeout=timeout_s)
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: fn(candidate, ctx)),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        # A timed-out candidate gets the worst possible score for each
        # objective so the selector evicts it without special-casing.
        spec = read_spec(fn)
        if spec["objectives"]:
            return {o: -math.inf for o in spec["objectives"]}
        return -math.inf


async def evaluate_batch(
    population: list[Individual],
    fitness_fn: Callable,
    *,
    seed: int = 0,
    held_out: bool = False,
    fidelity: float = 1.0,
    concurrency: int = 4,
) -> None:
    """Fill in ``fitness`` on every member of *population* in place.

    Uses a bounded Semaphore so heavyweight fitness functions (which
    might themselves call LLMs) don't all fire at once. If the user
    supplied a spec with ``objectives=None`` we expect a scalar; if
    ``objectives`` is a list we expect a dict — but we don't police
    either side beyond this assumption, to keep the contract ergonomic.
    """
    spec = read_spec(fitness_fn)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def worker(ind: Individual) -> None:
        async with sem:
            ctx = EvalContext(
                seed=seed + hash(ind.cid) % 1_000_003,
                fidelity=fidelity,
                held_out=held_out,
            )
            ind.fitness = await _call_fitness(fitness_fn, ind.genome, ctx, spec["timeout_s"])

    await asyncio.gather(*(worker(ind) for ind in population))


# ---------------------------------------------------------------------------
# Held-out reward-hacking guard
# ---------------------------------------------------------------------------


async def held_out_guard(
    population: list[Individual],
    fitness_fn: Callable,
    *,
    top_k: int = 5,
    seed: int = 0,
    objective: Optional[str] = None,
    concurrency: int = 4,
) -> dict[str, float]:
    """Re-score the top-K on the held-out split and return penalties.

    The penalty for candidate *c* is::

        max(0, train(c) - heldout(c) - gap) * weight

    with ``gap`` from the user spec and ``weight = 1.0`` so the penalty
    is comparable with the raw fitness. Returns ``{cid: penalty}``. The
    runner subtracts this from the archive-placement score; the raw
    fitness row in SQLite is untouched for auditability.
    """
    spec = read_spec(fitness_fn)
    gap = spec["generalization_gap"]

    def _score(ind: Individual) -> float:
        f = ind.fitness
        if isinstance(f, dict):
            if objective is None:
                return math.nan
            return float(f.get(objective, math.nan))
        return float(f)

    top = sorted(population, key=_score, reverse=True)[:top_k]
    # Re-evaluate on held-out split in a shadow Individual list to avoid
    # clobbering the training scores already stored.
    shadow = [Individual(cid=ind.cid, genome=ind.genome) for ind in top]
    await evaluate_batch(shadow, fitness_fn, seed=seed, held_out=True, concurrency=concurrency)
    penalties: dict[str, float] = {}
    for train_ind, hout_ind in zip(top, shadow):
        train = _score(train_ind)
        hout = _score(hout_ind)
        if math.isnan(train) or math.isnan(hout):
            continue
        excess = max(0.0, (train - hout) - gap)
        penalties[train_ind.cid] = excess
    return penalties


# ---------------------------------------------------------------------------
# Successive halving (Hyperband-lite)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pairwise evaluation (v0.2 feature 3)
# ---------------------------------------------------------------------------


async def evaluate_pairwise(
    population: list[Individual],
    pj,                             # judge.PairwiseJudge — injected
    *,
    conn,                           # sqlite3.Connection — for vote persistence
    generation: int,
    rounds: int,
    seed: int,
    concurrency: int = 4,
) -> int:
    """Fill fitness via Bradley-Terry aggregation of pairwise verdicts.

    Side-effects:
      * every verdict is written to ``pairwise_votes`` (keyed by
        ``(generation, left_id, right_id, eval_seed)``),
      * the resulting log-odds per candidate are written to
        ``bt_scores`` AND copied onto ``Individual.fitness`` so the
        existing selector pipeline (tournament / MAP-Elites / ES) works
        unchanged.

    Returns the number of verdicts collected (excluding ties).
    """
    import random as _random

    # Lazy import to avoid evaluator.py ↔ judge.py circular import at
    # module load.
    import judge as judge_module
    import storage as storage_module

    if len(population) < 2:
        for ind in population:
            ind.fitness = 0.0
        return 0

    rng = _random.Random(seed)
    cid_to_ind = {ind.cid: ind for ind in population}
    ids = list(cid_to_ind.keys())
    schedule = judge_module.sample_pair_schedule(ids, rounds=rounds, rng=rng)

    sem = asyncio.Semaphore(max(1, concurrency))
    verdicts: list[Optional[str]] = [None] * len(schedule)

    async def worker(i: int, a: str, b: str) -> None:
        async with sem:
            sub_rng = _random.Random(seed * 100003 + i)
            verdicts[i] = await pj.judge(
                cid_to_ind[a].genome, cid_to_ind[b].genome,
                seed=seed * 997 + i, rng=sub_rng,
            )

    await asyncio.gather(*(worker(i, a, b) for i, (a, b) in enumerate(schedule)))

    votes: list[tuple[str, str]] = []
    recorded = 0
    for i, (a, b) in enumerate(schedule):
        verdict = verdicts[i] or "tie"
        if verdict == "a":
            storage_module.record_pairwise_vote(conn, generation, a, b, "left", seed)
            votes.append((a, b))
            recorded += 1
        elif verdict == "b":
            storage_module.record_pairwise_vote(conn, generation, a, b, "right", seed)
            votes.append((b, a))
            recorded += 1
        else:
            storage_module.record_pairwise_vote(conn, generation, a, b, "tie", seed)

    scores, iters = judge_module.aggregate_bradley_terry(votes)
    for ind in population:
        score = float(scores.get(ind.cid, 0.0))
        ind.fitness = score
        storage_module.record_bt_score(conn, generation, ind.cid, score, iters)
    return recorded


async def successive_halving(
    population: list[Individual],
    fitness_fn: Callable,
    *,
    rungs: int = 3,
    keep_frac: float = 0.5,
    seed: int = 0,
    concurrency: int = 4,
    objective: Optional[str] = None,
) -> list[Individual]:
    """Evaluate at increasing fidelity; keep only survivors at each rung.

    The fidelity schedule is ``1/2**(rungs-1) ... 1``. At each rung we
    run :func:`evaluate_batch` and keep the top ``keep_frac`` by score.
    The final returned list carries full-fidelity scores.
    """
    survivors = list(population)
    for rung in range(rungs):
        fid = 1.0 / (2 ** (rungs - 1 - rung))
        await evaluate_batch(
            survivors,
            fitness_fn,
            seed=seed + rung,
            fidelity=fid,
            concurrency=concurrency,
        )
        if rung == rungs - 1:
            break
        if objective is None:
            # Flat score key
            def _key(ind: Individual) -> float:
                f = ind.fitness
                return float(f) if not isinstance(f, dict) else math.nan
        else:
            def _key(ind: Individual) -> float:
                f = ind.fitness
                return float(f[objective]) if isinstance(f, dict) else math.nan  # type: ignore[index]
        survivors.sort(key=_key, reverse=True)
        survivors = survivors[: max(1, int(len(survivors) * keep_frac))]
    return survivors
