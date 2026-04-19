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
) -> Callable[[Callable], Callable]:
    """Decorator that attaches evaluator metadata to a user fitness fn.

    Exposed to user code as::

        from evolver_sdk import fitness_spec    # (alias — see sdk.py)

        @fitness_spec(held_out_frac=0.2, objectives=["accuracy", "cost"])
        def fitness(candidate, context): ...

    The decorated function still behaves like a normal callable; the
    metadata is read from ``fn.__evolver_spec__``.
    """

    def deco(fn: Callable) -> Callable:
        fn.__evolver_spec__ = {  # type: ignore[attr-defined]
            "held_out_frac":      float(held_out_frac),
            "timeout_s":          float(timeout_s),
            "objectives":         list(objectives) if objectives else None,
            "generalization_gap": float(generalization_gap),
        }
        return fn

    return deco


def load_fitness(experiment_dir: Path) -> Callable:
    """Dynamically import ``fitness.py`` from *experiment_dir*.

    Returns the ``fitness`` callable. Raises ``FileNotFoundError`` if
    the file is missing and ``AttributeError`` if the ``fitness``
    symbol doesn't exist.
    """
    path = experiment_dir / "fitness.py"
    if not path.exists():
        raise FileNotFoundError(f"missing fitness.py at {path}")
    spec = importlib.util.spec_from_file_location(f"user_fitness_{experiment_dir.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
