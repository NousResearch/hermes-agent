#!/usr/bin/env python3
"""Darwinian Evolver — CLI entry point.

Usage
-----
    evolver init   <name>  --task {prompt|regex|sql|code}
    evolver run    <dir>   [--generations N --pop M --budget USD
                            --algorithm {es|map-elites|nsga2}
                            --concurrency N --seed N]
    evolver status <dir>
    evolver best   <dir>   [--k K --objective NAME]
    evolver lineage <dir>  --id CID [--format {json|mermaid}]
    evolver budget <dir>
    evolver export <dir>   --format {dspy-jsonl|gepa-trace}
    evolver replay <dir>   --seed N

Every subcommand prints a single JSON object to stdout on success, or a
single JSON object with an ``error`` key and a non-zero exit code on
failure. ``run`` also streams per-generation progress lines (one JSON
object per line) to stdout so a parent agent can follow the loop live.

The CLI deliberately keeps no persistent process state. All state lives
in ``<experiment-dir>/lineage.db`` so a crashed run can be resumed simply
by re-invoking ``evolver run`` on the same directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# Make sibling scripts importable whether we're invoked directly or as
# a module — Hermes invokes skills with the scripts dir on $PATH but not
# on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters              # noqa: E402
import algorithms            # noqa: E402
import bandit_director       # noqa: E402
import cache                 # noqa: E402
import coevolve              # noqa: E402
import critic                # noqa: E402
import descriptor_controller # noqa: E402
import descriptor_dsl        # noqa: E402
import distributed           # noqa: E402
import fitness_synth         # noqa: E402
import hub                   # noqa: E402
import judge                 # noqa: E402
import operators             # noqa: E402
import storage               # noqa: E402
import task_features         # noqa: E402
import transfer              # noqa: E402
import evaluator             # noqa: E402
from algorithms import (  # noqa: E402
    Exp3Bandit,
    Individual,
    MapElitesArchive,
    default_prompt_descriptor,
    mu_plus_lambda,
    nsga2_select,
    tournament_select,
)
from evaluator import evaluate_batch, load_fitness, read_spec  # noqa: E402
from llm import BudgetExceeded, BudgetLedger, LLMClient, discover_endpoint  # noqa: E402


# ---------------------------------------------------------------------------
# Output helpers — every command prints valid JSON
# ---------------------------------------------------------------------------


def _out(obj: Any) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


def _stream(obj: Any) -> None:
    """One-line NDJSON progress record."""
    json.dump(obj, sys.stdout, ensure_ascii=False, default=str)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _err(msg: str, code: int = 1) -> None:
    _out({"error": msg})
    sys.exit(code)


def _hermes_home() -> Path:
    val = os.environ.get("HERMES_HOME", "").strip()
    return Path(val) if val else Path.home() / ".hermes"


def _data_root() -> Path:
    return _hermes_home() / "skills" / "research" / "darwinian-evolver" / "data"


def _resolve_experiment(name_or_path: str) -> Path:
    """Accept either an experiment name (looked up under data_root) or
    an absolute / relative path to an experiment directory."""
    p = Path(name_or_path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (_data_root() / name_or_path).resolve()


# ---------------------------------------------------------------------------
# Fitness templates shipped with the skill
# ---------------------------------------------------------------------------


_FITNESS_TEMPLATES: dict[str, str] = {
    "prompt": textwrap.dedent('''
        """Fitness template for prompt evolution.

        Replace the scoring loop with your own — this stub gives full marks
        to short prompts that contain the word "concise" so the pipeline
        runs end-to-end without an LLM eval while you wire up your real
        judge.
        """

        from evolver_sdk import fitness_spec


        @fitness_spec(held_out_frac=0.2, timeout_s=30)
        def fitness(candidate: str, context: dict) -> float:
            n = len(candidate)
            penalty = max(0, n - 120) / 120          # favour brevity
            bonus   = 0.3 if "concise" in candidate.lower() else 0.0
            return max(0.0, 1.0 - penalty) + bonus
        ''').strip() + "\n",

    "regex": textwrap.dedent('''
        """Fitness template for regex evolution.

        Scores against two lists: strings that should match, and strings
        that should not. Default returns F1 over precision + recall on
        the positive corpus.
        """

        import re
        from evolver_sdk import fitness_spec

        POSITIVES = ["user@example.com", "first.last+tag@sub.domain.org"]
        NEGATIVES = ["not an email", "a@b",                        "@nope.com"]


        @fitness_spec(held_out_frac=0.2, timeout_s=5)
        def fitness(candidate: str, context: dict) -> float:
            try:
                pat = re.compile(candidate)
            except re.error:
                return 0.0
            tp = sum(bool(pat.fullmatch(s)) for s in POSITIVES)
            fp = sum(bool(pat.fullmatch(s)) for s in NEGATIVES)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall    = tp / len(POSITIVES)  if POSITIVES else 0.0
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
        ''').strip() + "\n",

    "sql": textwrap.dedent('''
        """Fitness template for SQL query evolution.

        Stub: rewards queries that return the expected number of rows
        against a local SQLite fixture. Replace with your own schema and
        expected outputs.
        """

        import sqlite3
        from evolver_sdk import fitness_spec

        EXPECTED_ROWS = 5


        @fitness_spec(held_out_frac=0.2, timeout_s=10)
        def fitness(candidate: str, context: dict) -> float:
            conn = sqlite3.connect(":memory:")
            conn.executescript(
                "CREATE TABLE users(id INT, name TEXT);"
                "INSERT INTO users VALUES "
                "(1,'a'),(2,'b'),(3,'c'),(4,'d'),(5,'e');"
            )
            try:
                rows = conn.execute(candidate).fetchall()
            except sqlite3.Error:
                return 0.0
            finally:
                conn.close()
            return 1.0 - min(1.0, abs(len(rows) - EXPECTED_ROWS) / EXPECTED_ROWS)
        ''').strip() + "\n",

    "code": textwrap.dedent('''
        """Fitness template for code evolution (hidden pytest suite).

        Point ``TESTS`` at a pytest file the evaluator should run against
        the candidate. The candidate is written to ``solution.py`` inside
        a subprocess sandbox; fitness = fraction of tests passed.
        """

        from evolver_sdk import fitness_spec

        TESTS = "tests/test_hidden.py"


        @fitness_spec(held_out_frac=0.2, timeout_s=60)
        def fitness(candidate: str, context: dict) -> float:
            # Stub — wire this to sandbox.run_pytest once sandbox.py is
            # enabled in your environment.
            return 0.0
        ''').strip() + "\n",
}


_EVOLVER_SDK_STUB = textwrap.dedent('''
    """Thin shim so fitness.py can ``from evolver_sdk import fitness_spec``.

    This file is auto-generated by ``evolver init``. Re-import the real
    decorator from the skill scripts directory.
    """

    import sys
    from pathlib import Path

    _SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))

    from evaluator import fitness_spec  # noqa: F401
''').strip() + "\n"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> None:
    task = args.task
    if task not in _FITNESS_TEMPLATES:
        _err(f"unknown task {task!r}; choose from {sorted(_FITNESS_TEMPLATES)}")
    exp = _resolve_experiment(args.name)
    if exp.exists():
        _err(f"experiment directory already exists: {exp}")
    (exp / "seed").mkdir(parents=True)
    (exp / "logs").mkdir()
    (exp / "fitness.py").write_text(_FITNESS_TEMPLATES[task], encoding="utf-8")
    (exp / "evolver_sdk.py").write_text(_EVOLVER_SDK_STUB, encoding="utf-8")
    (exp / "experiment.yaml").write_text(
        f"name: {exp.name}\ntask: {task}\ncreated_at: {int(time.time())}\n",
        encoding="utf-8",
    )
    (exp / "seed" / "initial.txt").write_text(
        {"prompt": "Summarize this concisely.\n",
         "regex":  r".+@.+\..+" + "\n",
         "sql":    "SELECT * FROM users;\n",
         "code":   "def solve(x):\n    return x\n"}[task],
        encoding="utf-8",
    )
    conn = storage.open_db(exp / "lineage.db")

    # v0.3 (B3): warm-start from a hub snapshot by adding its top-K
    # genomes as extra seed files. Provenance is recorded so lineage
    # queries can distinguish human-curated from hub-imported seeds.
    imported = 0
    warm_tag = getattr(args, "warm_start", None)
    if warm_tag:
        try:
            genomes = hub.warm_start_seeds(warm_tag, top_k=getattr(args, "warm_k", 5))
        except KeyError as exc:
            conn.close()
            _err(str(exc))
            return
        snap = hub.resolve(warm_tag)
        for i, g in enumerate(genomes, start=1):
            (exp / "seed" / f"warm_{i:02d}.txt").write_text(g, encoding="utf-8")
            imported += 1
            # Candidate rows appear only after ``evolver run``;
            # hub-import provenance is keyed by content-hash genome ID.
            cid = storage.hash_genome(g)
            storage.record_hub_import(
                conn, cid,
                hub_hash=snap.hash if snap else warm_tag,
                hub_tag=snap.tag if snap else warm_tag,
            )
    conn.close()
    _out({"ok": True, "dir": str(exp), "task": task, "warm_start_imports": imported})


def _load_seeds(exp: Path) -> list[str]:
    seeds_dir = exp / "seed"
    if not seeds_dir.exists():
        return []
    return [p.read_text(encoding="utf-8").strip() for p in sorted(seeds_dir.iterdir()) if p.is_file()]


async def _run_loop(args: argparse.Namespace, exp: Path) -> dict:
    """The evolutionary main loop.

    Supports three top-level algorithms:

      * ``es``         — classic (μ+λ)-ES, single-objective.
      * ``map-elites`` — quality-diversity with the default 2-D prompt
                         descriptor (length × CoT-presence).
      * ``nsga2``      — multi-objective, requires dict fitness whose
                         keys are listed in the fitness-spec objectives.
    """
    fitness_fn = load_fitness(exp)
    spec = read_spec(fitness_fn)
    objectives = spec["objectives"]
    primary_obj = objectives[0] if objectives else None
    judge_mode = spec["judge"]
    pairwise_rounds = spec["pairwise_rounds"]

    if args.algorithm == "nsga2" and not objectives:
        _err("nsga2 requires fitness_spec(objectives=[...]) to be set")
    if args.algorithm != "nsga2" and objectives:
        # User returned dict fitness but picked single-objective algo;
        # fall back to primary objective for selection.
        pass
    if judge_mode == "pairwise" and objectives:
        _err(
            "pairwise judge is incompatible with multi-objective fitness "
            "(NSGA-II mode) in v0.2. Set objectives=None or judge='scalar'."
        )

    rng = random.Random(args.seed)
    conn = storage.open_db(exp / "lineage.db")

    ledger = BudgetLedger(
        cap_usd=args.budget,
        input_rate_per_million=args.input_rate,
        output_rate_per_million=args.output_rate,
        on_record=lambda i, o, u, op: storage.record_budget(conn, i, o, u, op),
    )

    # v0.2: per-experiment LLM response cache — invisible to user-facing
    # code but ensures ``evolver replay --seed N`` is bit-for-bit
    # reproducible at zero additional LLM cost. We read the flag via
    # getattr so callers that build Namespaces by hand (mostly tests)
    # continue to work without opting into the newer attribute set.
    no_cache = getattr(args, "no_cache", False)
    response_cache = None if no_cache else cache.ResponseCache(conn)

    async with LLMClient.from_hermes(
        concurrency=args.concurrency, budget=ledger, cache=response_cache,
    ) as llm:
        # Pairwise judge is materialised once per run and reused across
        # generations so the LLM client's connection pool stays warm.
        pairwise_judge = judge.PairwiseJudge(client=llm) if judge_mode == "pairwise" else None

        # Constitutional critic (feature 4) — off by default.
        critic_obj = None
        if spec["critic"] == "on":
            critic_obj = critic.ConstitutionalCritic(
                client=llm,
                threshold=spec["critic_threshold"],
                model_override=spec["critic_model"],
            )

        # v0.4 A2 — bandit director (off by default)
        director: bandit_director.BanditDirector | None = None
        if getattr(args, "bandit_director", "off") == "periodic":
            director = bandit_director.BanditDirector(
                client=llm,
                trigger_every_r=getattr(args, "bandit_every_r", 4),
            )

        # v0.5 B1 — worker backend. "local" uses evaluate_batch's
        # Semaphore path (backward-compatible); "raysim" / "ray" route
        # candidate evaluation through the WorkerBackend protocol so
        # cluster-scale experiments don't bottleneck on a single
        # event loop.
        workers_mode = getattr(args, "workers", "local")
        worker_backend = None
        if workers_mode != "local":
            try:
                worker_backend = distributed.select_backend(
                    workers_mode, workers=args.concurrency,
                )
            except ImportError as exc:
                _err(str(exc))
                return

        async def _evaluate(pop: list[Individual], gen: int, step_seed: int) -> None:
            """Dispatch to the right evaluator given the fitness spec."""
            if pairwise_judge is not None:
                await evaluator.evaluate_pairwise(
                    pop, pairwise_judge,
                    conn=conn, generation=gen,
                    rounds=pairwise_rounds, seed=step_seed,
                    concurrency=args.concurrency,
                )
            else:
                await evaluate_batch(
                    pop, fitness_fn,
                    seed=step_seed, concurrency=args.concurrency,
                    backend=worker_backend,
                )

        async def _apply_critic(pop: list[Individual], gen: int) -> None:
            """Review the top-K and apply placement-score penalties in place.

            Raw fitness rows in SQLite are already written — the
            penalty affects only ``Individual.fitness`` in memory,
            which is what the selector/archive see. The audit trail in
            ``fitness`` table therefore remains pristine.
            """
            if critic_obj is None or not pop:
                return
            k = max(1, int(spec["critic_top_k"]))

            def _score(ind: Individual) -> float:
                f = ind.fitness
                if isinstance(f, dict):
                    return float(f.get(primary_obj or next(iter(f)), float("nan")))
                return float(f)

            top = sorted(pop, key=_score, reverse=True)[:k]
            tasks = [critic_obj.review(ind.genome, fitness_value=ind.fitness) for ind in top]
            reviews = await asyncio.gather(*tasks, return_exceptions=True)

            for ind, review in zip(top, reviews):
                if isinstance(review, Exception):
                    continue
                storage.record_critic_evaluation(
                    conn, ind.cid, gen,
                    risk=review.risk,
                    evidence=review.evidence,
                    signal_tags=review.signal_tags,
                    model=review.model,
                )
                penalty = critic_obj.penalty(review, _score(ind))
                if penalty > 0:
                    if isinstance(ind.fitness, dict):
                        # In multi-objective mode, dock the primary objective.
                        obj_key = primary_obj or next(iter(ind.fitness))
                        ind.fitness[obj_key] = float(ind.fitness[obj_key]) - penalty
                    else:
                        ind.fitness = float(ind.fitness) - penalty

        # ------------ seed population ------------
        seeds = _load_seeds(exp)
        if not seeds:
            _err("no seeds found — put at least one file under seed/")
        population: list[Individual] = []
        for s in seeds:
            ind = Individual(
                cid=storage.hash_genome(s),
                genome=s,
                generation=0,
                descriptor=default_prompt_descriptor(s),
                operator="seed",
            )
            population.append(ind)
            storage.insert_candidate(conn, s, generation=0,
                                     descriptor={"d": list(ind.descriptor)},
                                     parents=())
        await _evaluate(population, gen=0, step_seed=args.seed)
        for ind in population:
            if isinstance(ind.fitness, dict):
                for k, v in ind.fitness.items():
                    storage.record_fitness(conn, ind.cid, k, float(v), eval_seed=args.seed)
            else:
                storage.record_fitness(conn, ind.cid, "fitness", float(ind.fitness), eval_seed=args.seed)
        await _apply_critic(population, gen=0)

        # v0.3 (A1): descriptor is a DSL-parsed function that the
        # optional controller can replace mid-run. The default grid
        # matches v0.2's hard-coded (length, cot_presence) axes, so
        # existing behaviour is byte-identical when the controller is
        # off.
        controller_mode = getattr(args, "descriptor_controller", "off")
        current_descriptor = descriptor_dsl.parse_descriptor(
            "grid(length(bins=8), cot_presence())"
        )

        descriptor_ctrl: descriptor_controller.DescriptorController | None = None
        if controller_mode != "off":
            descriptor_ctrl = descriptor_controller.DescriptorController(
                client=llm,
                trigger_every_k=getattr(args, "descriptor_every_k", 5),
                mode=controller_mode,
            )
            storage.record_descriptor(
                conn, 0, current_descriptor.canonical(),
                "keep", "initial descriptor",
            )

        archive: MapElitesArchive | None = None
        if args.algorithm == "map-elites":
            archive = MapElitesArchive(
                bin_counts=current_descriptor.bin_counts,
                lows=current_descriptor.lows,
                highs=current_descriptor.highs,
                objective=primary_obj,
            )
            # Re-descriptor every seed under the current DSL function so
            # the archive is consistent even when the controller
            # eventually rewrites the grid.
            for ind in population:
                ind.descriptor = current_descriptor(ind.genome)
                archive.place(ind)

        # Fitness-delta ringbuffer (best-of-gen minus previous best)
        # feeds the controller's plateau signal.
        fitness_deltas: list[float] = []
        prev_best_score: float | None = None
        def _record_delta(pop: list[Individual]) -> None:
            nonlocal prev_best_score
            if not pop:
                return
            def _s(i: Individual) -> float:
                f = i.fitness
                if isinstance(f, dict):
                    return float(f.get(primary_obj or next(iter(f)), float("nan")))
                return float(f)
            curr = max(_s(i) for i in pop)
            if prev_best_score is not None:
                fitness_deltas.append(curr - prev_best_score)
            prev_best_score = curr

        bandit = Exp3Bandit(arms=list(operators.MUTATION_OPERATORS.keys()))

        _stream({"gen": 0, "pop": len(population), "budget_used": ledger.spent_usd,
                 "best": _best_summary(population, primary_obj)})

        # ------------ evolution ------------
        try:
            for gen in range(1, args.generations + 1):
                parents = _select_parents(population, archive, args.pop, rng, primary_obj)
                offspring = await _produce_offspring(
                    llm, parents, bandit, rng, gen, primary_obj,
                )
                for child in offspring:
                    parents_edges = [(p, child.operator, "") for p in child.parents]
                    storage.insert_candidate(
                        conn, child.genome, generation=gen,
                        descriptor={"d": list(child.descriptor)},
                        parents=parents_edges,
                    )
                await _evaluate(offspring, gen=gen, step_seed=args.seed + gen)
                for ind in offspring:
                    if isinstance(ind.fitness, dict):
                        for k, v in ind.fitness.items():
                            storage.record_fitness(conn, ind.cid, k, float(v), eval_seed=args.seed + gen)
                    else:
                        storage.record_fitness(conn, ind.cid, "fitness", float(ind.fitness), eval_seed=args.seed + gen)
                await _apply_critic(offspring, gen=gen)

                if args.algorithm == "es":
                    population = mu_plus_lambda(population, offspring,
                                                mu=args.pop, objective=primary_obj)
                elif args.algorithm == "map-elites":
                    assert archive is not None
                    for ind in offspring:
                        ind.descriptor = current_descriptor(ind.genome)
                        archive.place(ind)
                    population = list(archive.cells.values())
                elif args.algorithm == "nsga2":
                    assert objectives is not None
                    population = nsga2_select(population + offspring, objectives, args.pop)
                else:  # pragma: no cover — argparse guards this
                    _err(f"unknown algorithm {args.algorithm!r}")

                # v0.4 (A2): bandit director hook — operates on the
                # in-flight Exp3 bandit independently of algorithm choice.
                if director is not None and director.should_trigger(gen):
                    stats = [
                        bandit_director.OperatorStats(
                            name=arm,
                            weight=bandit.weights[i],
                            selections=0,
                            mean_delta=0.0,
                            last_good_gen=gen,
                        )
                        for i, arm in enumerate(bandit.arms)
                    ]
                    verdict = await director.audit(
                        bandit, stats, fitness_deltas,
                        seed=args.seed + gen,
                    )
                    applied = bandit_director.apply_verdict(
                        verdict, bandit, operators.MUTATION_OPERATORS, director,
                    )
                    for action in applied:
                        if action.type == "add":
                            storage.record_generated_operator(
                                conn,
                                name=action.payload["name"],
                                template=action.payload["template"],
                                temperature=action.payload["temperature"],
                            )
                        elif action.type == "retire":
                            storage.retire_generated_operator(
                                conn, action.payload["name"],
                            )

                # v0.3 (A1): descriptor-controller hook, MAP-Elites only.
                _record_delta(population)
                if (
                    descriptor_ctrl is not None
                    and archive is not None
                    and descriptor_ctrl.should_trigger(gen)
                ):
                    proposal = await descriptor_ctrl.propose(
                        current_descriptor, archive, fitness_deltas,
                        seed=args.seed + gen,
                    )
                    storage.record_descriptor(
                        conn, gen,
                        (proposal.grid or current_descriptor).canonical(),
                        proposal.action, proposal.reason,
                    )
                    if proposal.action == "replace" and proposal.grid is not None:
                        archive = descriptor_controller.remap_archive(archive, proposal.grid)
                        current_descriptor = proposal.grid
                        population = list(archive.cells.values())

                _stream({
                    "gen": gen,
                    "pop": len(population),
                    "archive": archive.coverage() if archive else None,
                    "budget_used": ledger.spent_usd,
                    "calls": ledger.calls,
                    "best": _best_summary(population, primary_obj),
                })
        except BudgetExceeded as exc:
            _stream({"halt": "budget_exceeded", "detail": str(exc)})

    conn.close()
    return {
        "ok": True,
        "generations_run": gen if 'gen' in locals() else 0,
        "final_pop": len(population),
        "budget": {"usd": ledger.spent_usd, "calls": ledger.calls},
    }


def _best_summary(population: list[Individual], objective: str | None) -> dict:
    if not population:
        return {}
    def key(ind: Individual) -> float:
        f = ind.fitness
        if isinstance(f, dict):
            return float(f.get(objective or next(iter(f)), float("nan")))
        return float(f)
    best = max(population, key=key)
    return {"id": best.cid, "fitness": best.fitness, "preview": best.genome[:120]}


def _select_parents(
    population: list[Individual],
    archive: MapElitesArchive | None,
    n: int,
    rng: random.Random,
    objective: str | None,
) -> list[Individual]:
    if archive is not None and archive.cells:
        return archive.sample(rng, k=n)
    return tournament_select(population, k=3, n=n, rng=rng, objective=objective)


async def _produce_offspring(
    llm: LLMClient,
    parents: list[Individual],
    bandit: Exp3Bandit,
    rng: random.Random,
    generation: int,
    objective: str | None,
) -> list[Individual]:
    """Apply one operator per parent and return the resulting offspring."""
    tasks = []
    chosen_ops: list[str] = []
    chosen_idx: list[int] = []
    for parent in parents:
        idx, op = bandit.pick(rng)
        chosen_ops.append(op)
        chosen_idx.append(idx)
        fn = operators.MUTATION_OPERATORS[op]
        tasks.append(fn(llm, parent.genome, seed=rng.randint(0, 2**31 - 1)))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    offspring: list[Individual] = []
    for parent, op, idx, res in zip(parents, chosen_ops, chosen_idx, results):
        if isinstance(res, Exception):
            # Treat operator failure as a null reward; skip child.
            bandit.reward(idx, 0.0)
            continue
        child_genome = res.child.strip()
        if not child_genome or child_genome == parent.genome:
            bandit.reward(idx, 0.0)
            continue
        ind = Individual(
            cid=storage.hash_genome(child_genome),
            genome=child_genome,
            generation=generation,
            descriptor=default_prompt_descriptor(child_genome),
            parents=[parent.cid],
            operator=op,
        )
        offspring.append(ind)
        # Reward signal is populated later by the runner after fitness
        # is measured; at this stage we give a small exploration reward
        # so the bandit keeps trying operators that produce novel text.
        bandit.reward(idx, 0.1)
    return offspring


def cmd_run(args: argparse.Namespace) -> None:
    exp = _resolve_experiment(args.dir)
    if not exp.exists():
        _err(f"experiment not found: {exp}")
    try:
        result = asyncio.run(_run_loop(args, exp))
    except KeyboardInterrupt:
        _err("interrupted", code=130)
    _out(result)


def cmd_status(args: argparse.Namespace) -> None:
    exp = _resolve_experiment(args.dir)
    conn = storage.open_db(exp / "lineage.db")
    budget = storage.get_budget_used(conn)
    gen = storage.count_generations(conn)
    best_rows = []
    for objective in _discover_objectives(conn):
        best_rows.append({"objective": objective,
                          "top": storage.get_best(conn, objective, k=1)})
    _out({
        "experiment": exp.name,
        "generations": gen,
        "budget": budget,
        "objectives": [row["objective"] for row in best_rows],
        "best": best_rows,
    })
    conn.close()


def cmd_best(args: argparse.Namespace) -> None:
    exp = _resolve_experiment(args.dir)
    conn = storage.open_db(exp / "lineage.db")
    objective = args.objective or _discover_objectives(conn)[0]
    rows = storage.get_best(conn, objective, k=args.k)
    _out({"objective": objective, "k": args.k, "candidates": rows})
    conn.close()


def cmd_lineage(args: argparse.Namespace) -> None:
    exp = _resolve_experiment(args.dir)
    conn = storage.open_db(exp / "lineage.db")
    edges = storage.get_ancestry(conn, args.id)
    if args.format == "mermaid":
        lines = ["graph TD"]
        for e in edges:
            lines.append(f'    {e["parent_id"]} -- {e["operator"]} --> {e["child_id"]}')
        _out({"mermaid": "\n".join(lines), "edges": len(edges)})
    else:
        _out({"edges": edges, "count": len(edges)})
    conn.close()


def cmd_budget(args: argparse.Namespace) -> None:
    exp = _resolve_experiment(args.dir)
    conn = storage.open_db(exp / "lineage.db")
    _out({"budget": storage.get_budget_used(conn)})
    conn.close()


def cmd_synthesise_fitness(args: argparse.Namespace) -> None:
    """``evolver synthesise-fitness <dir> --examples file.jsonl`` (A4).

    Reads newline-delimited JSON I/O pairs, asks the LLM to pick an
    archetype, writes the generated fitness.py into the experiment
    directory for the user to review. Never auto-accepts.
    """
    exp = _resolve_experiment(args.dir)
    if not exp.exists():
        _err(f"experiment not found: {exp}")
    examples_path = Path(args.examples).expanduser().resolve()
    if not examples_path.exists():
        _err(f"examples file not found: {examples_path}")

    examples: list[dict] = []
    for line in examples_path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as exc:
            _err(f"bad JSONL line: {exc}")
            return
    if not examples:
        _err("no examples parsed from file")
        return

    async def _run():
        async with LLMClient.from_hermes() as client:
            return await fitness_synth.synthesise(
                client, examples, criterion=args.criterion,
            )

    result = asyncio.run(_run())
    out_path = exp / ("fitness.synthesised.py" if args.no_overwrite else "fitness.py")
    out_path.write_text(result.fitness_src, encoding="utf-8")

    conn = storage.open_db(exp / "lineage.db")
    try:
        synth_id = storage.record_fitness_synthesis(
            conn,
            archetype=result.archetype,
            examples_n=len(examples),
            fitness_src=result.fitness_src,
        )
    finally:
        conn.close()

    _out({
        "ok": True,
        "archetype": result.archetype,
        "rationale": result.rationale,
        "path": str(out_path),
        "synthesis_id": synth_id,
        "examples": len(examples),
    })


def cmd_transfer(args: argparse.Namespace) -> None:
    """``evolver transfer train|apply`` (A5)."""
    if args.action == "train":
        dirs = [_resolve_experiment(d) for d in args.experiments]
        missing = [str(d) for d in dirs if not (d / "lineage.db").exists()]
        if missing:
            _err(f"missing lineage.db in: {missing}")
            return
        policy = transfer.train_policy(dirs, k=args.k)
        out_path = Path(args.out).expanduser().resolve() if args.out \
            else _data_root() / "transfer-policy.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save(out_path)
        _out({
            "ok": True,
            "points": len(policy.points),
            "path": str(out_path),
            "policy_hash": policy.policy_hash,
        })
    elif args.action == "apply":
        policy_path = Path(args.policy).expanduser().resolve()
        if not policy_path.exists():
            _err(f"policy file not found: {policy_path}")
            return
        policy = transfer.TransferPolicy.load(policy_path)
        target_dir = _resolve_experiment(args.target)
        if not target_dir.exists():
            _err(f"target experiment not found: {target_dir}")
            return
        feats = task_features.featurise(target_dir)
        prediction = policy.predict(feats.vector)

        conn = storage.open_db(target_dir / "lineage.db")
        try:
            storage.record_task_features(
                conn, target_dir.name, feats.to_dict(),
                policy_hash=policy.policy_hash,
            )
        finally:
            conn.close()

        seed_dir = target_dir / "seed"
        seed_dir.mkdir(exist_ok=True)
        imported = 0
        for i, g in enumerate(prediction["seeds"], start=1):
            (seed_dir / f"transfer_{i:02d}.txt").write_text(g, encoding="utf-8")
            imported += 1
        _out({
            "ok": True,
            "target": str(target_dir),
            "predicted_operators": prediction["operator_weights"],
            "seeds_imported": imported,
            "confidence": prediction["confidence"],
            "policy_hash": policy.policy_hash,
        })
    else:  # pragma: no cover
        _err(f"unknown transfer action {args.action!r}")


def cmd_coevolve(args: argparse.Namespace) -> None:
    """``evolver coevolve <dir>`` — dual-archive adversarial loop (A3).

    Convenience runner that sets up a solver + adversary pair from
    the experiment's seed directory and alternates evolution for
    *--generations* rounds. Uses existing operators and the user's
    fitness function (which MUST accept ``ctx["input"]``).
    """
    exp = _resolve_experiment(args.dir)
    if not exp.exists():
        _err(f"experiment not found: {exp}")
    fitness_fn = evaluator.load_fitness(exp)
    solver_seeds = _load_seeds(exp)
    if not solver_seeds:
        _err("no solver seeds in seed/")
        return
    adv_seeds = [f"adversarial input {i}" for i in range(1, args.adversaries + 1)]

    solvers = [
        algorithms.Individual(
            cid=storage.hash_genome(s), genome=s, operator="seed",
        )
        for s in solver_seeds
    ]
    advers = [
        algorithms.Individual(
            cid=storage.hash_genome(a), genome=a, operator="adversary_seed",
        )
        for a in adv_seeds
    ]

    conn = storage.open_db(exp / "lineage.db")
    for ind in solvers + advers:
        storage.insert_candidate(conn, ind.genome, generation=0)

    run = coevolve.CoevolveRun(solvers=solvers, adversaries=advers)

    async def _main():
        async with LLMClient.from_hermes(
            concurrency=args.concurrency,
            budget=BudgetLedger(cap_usd=args.budget),
        ) as llm_client:
            await coevolve.coevolve(
                run, llm=llm_client, fitness_fn=fitness_fn,
                conn=conn, generations=args.generations, seed=args.seed,
                max_adversary_generations=args.max_adversary_gens,
            )

    asyncio.run(_main())
    conn.close()

    _out({
        "ok": True,
        "generations": args.generations,
        "solvers_final": len(run.solvers),
        "adversaries_final": len(run.adversaries),
        "history_entries": len(run.history),
    })


def cmd_hub(args: argparse.Namespace) -> None:
    """``evolver hub {push|list|pull}`` — cross-experiment snapshot store."""
    if args.action == "push":
        exp = _resolve_experiment(args.dir)
        if not exp.exists():
            _err(f"experiment not found: {exp}")
        snap = hub.push(exp, tag=args.tag, top_k=args.top_k)
        _out({"ok": True, "hash": snap.hash, "tag": snap.tag, "path": str(snap.path)})
    elif args.action == "list":
        snaps = hub.list_snapshots()
        _out({"snapshots": [
            {"hash": s.hash, "tag": s.tag, "created_at": s.created_at,
             "size": s.size, "objectives": s.manifest.get("objectives", [])}
            for s in snaps
        ]})
    elif args.action == "pull":
        if not args.tag:
            _err("hub pull requires --tag / content-hash")
            return
        dest = Path(args.dest).resolve() if args.dest else Path.cwd() / f"pulled-{args.tag}"
        snap = hub.pull(args.tag, dest)
        _out({"ok": True, "hash": snap.hash, "tag": snap.tag, "dest": str(dest)})
    else:  # pragma: no cover
        _err(f"unknown hub action {args.action!r}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """``evolver dashboard <dir>`` — launches the read-only FastAPI UI.

    We import dashboard.py lazily so ``evolver --help`` and every other
    subcommand works on a Python with no fastapi installed. On import
    failure we print an install hint and exit non-zero.
    """
    exp = _resolve_experiment(args.dir)
    if not exp.exists():
        _err(f"experiment not found: {exp}")
    try:
        import dashboard  # noqa: WPS433 — local lazy import by design
    except ImportError as exc:
        _err(str(exc))
        return
    try:
        import uvicorn  # type: ignore[import]
    except ImportError:
        _err("uvicorn is required for `evolver dashboard`; "
             "install with `pip install uvicorn`.")
        return

    host = args.host
    if host not in ("127.0.0.1", "localhost"):
        print(
            f"WARNING: binding dashboard to {host} exposes read-only "
            f"experiment data on the network. No auth is enforced.",
            flush=True,
        )
    app = dashboard.build_app(exp)
    _out({"ok": True, "listening": f"http://{host}:{args.port}", "experiment": exp.name})
    uvicorn.run(app, host=host, port=args.port, log_level="warning")


def cmd_cache(args: argparse.Namespace) -> None:
    """``evolver cache {stats|purge}`` — LLM response cache control.

    v0.2 ships a transparent cache so replay is deterministic at zero
    extra cost. This subcommand lets the user inspect or clear it
    without hand-editing SQLite.
    """
    exp = _resolve_experiment(args.dir)
    conn = storage.open_db(exp / "lineage.db")
    rc = cache.ResponseCache(conn)
    if args.action == "stats":
        _out({"cache": rc.stats()})
    elif args.action == "purge":
        n = rc.purge()
        _out({"purged": n})
    else:  # pragma: no cover — argparse guards this
        _err(f"unknown cache action {args.action!r}")
    conn.close()


def cmd_export(args: argparse.Namespace) -> None:
    exp = _resolve_experiment(args.dir)
    conn = storage.open_db(exp / "lineage.db")
    out_path = exp / f"export-{args.format}.jsonl"
    if args.format == "dspy-jsonl":
        n = adapters.export_dspy_jsonl(conn, out_path, include_all_generations=args.all)
    elif args.format == "gepa-trace":
        n = adapters.export_gepa_trace(conn, out_path)
    else:
        _err(f"unknown export format {args.format!r}")
    _out({"ok": True, "path": str(out_path), "format": args.format, "records": n})
    conn.close()


def cmd_replay(args: argparse.Namespace) -> None:
    exp = _resolve_experiment(args.dir)
    conn = storage.open_db(exp / "lineage.db")
    h = storage.lineage_hash(conn)
    _out({"lineage_hash": h, "seed": args.seed,
          "hint": "compare this value across runs with the same --seed to check determinism"})
    conn.close()


def _discover_objectives(conn) -> list[str]:
    rows = conn.execute("SELECT DISTINCT objective FROM fitness").fetchall()
    return [r["objective"] for r in rows] or ["fitness"]


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evolver",
        description="LLM-driven evolutionary optimizer for prompts, regex, SQL, and small code.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("init", help="scaffold a new experiment directory")
    pi.add_argument("name")
    pi.add_argument("--task", required=True, choices=list(_FITNESS_TEMPLATES))
    pi.add_argument("--warm-start", dest="warm_start", default=None,
                    help="hub tag or content-hash to seed this experiment from")
    pi.add_argument("--warm-k", dest="warm_k", type=int, default=5,
                    help="number of top-K genomes to pull from the snapshot (default 5)")
    pi.set_defaults(fn=cmd_init)

    pr = sub.add_parser("run", help="run the evolutionary loop")
    pr.add_argument("dir")
    pr.add_argument("--generations", type=int, default=20)
    pr.add_argument("--pop",         type=int, default=8)
    pr.add_argument("--budget",      type=float, default=1.0,
                    help="USD cap (0 to disable)")
    pr.add_argument("--algorithm",   choices=["es", "map-elites", "nsga2"], default="es")
    pr.add_argument("--concurrency", type=int, default=4)
    pr.add_argument("--seed",        type=int, default=42)
    pr.add_argument("--input-rate",  dest="input_rate",  type=float, default=0.0,
                    help="USD per million input tokens (for budget accounting)")
    pr.add_argument("--output-rate", dest="output_rate", type=float, default=0.0,
                    help="USD per million output tokens (for budget accounting)")
    pr.add_argument("--no-cache",    dest="no_cache", action="store_true",
                    help="disable the LLM response cache (always hits the network)")
    pr.add_argument("--descriptor-controller", dest="descriptor_controller",
                    choices=["off", "periodic", "continuous"], default="off",
                    help="v0.3 A1: enable LLM-driven descriptor mutation (map-elites only)")
    pr.add_argument("--descriptor-every-k", dest="descriptor_every_k",
                    type=int, default=5,
                    help="when --descriptor-controller=periodic, trigger every K generations")
    pr.add_argument("--bandit-director", dest="bandit_director",
                    choices=["off", "periodic"], default="off",
                    help="v0.4 A2: LLM-driven operator add/retire/merge")
    pr.add_argument("--workers", dest="workers",
                    choices=["local", "raysim", "ray"], default="local",
                    help="v0.5 B1: evaluator worker backend (local | raysim | ray)")
    pr.set_defaults(fn=cmd_run, no_cache=False)

    ps = sub.add_parser("status", help="show generations, budget, and best-so-far")
    ps.add_argument("dir"); ps.set_defaults(fn=cmd_status)

    pb = sub.add_parser("best", help="top-K candidates by objective")
    pb.add_argument("dir")
    pb.add_argument("--k", type=int, default=5)
    pb.add_argument("--objective", default=None)
    pb.set_defaults(fn=cmd_best)

    pl = sub.add_parser("lineage", help="ancestry DAG for one candidate")
    pl.add_argument("dir")
    pl.add_argument("--id", required=True)
    pl.add_argument("--format", choices=["json", "mermaid"], default="mermaid")
    pl.set_defaults(fn=cmd_lineage)

    pdb = sub.add_parser("budget", help="cumulative LLM spend for this experiment")
    pdb.add_argument("dir"); pdb.set_defaults(fn=cmd_budget)

    pe = sub.add_parser("export", help="export experiment for downstream pipelines")
    pe.add_argument("dir")
    pe.add_argument("--format", choices=["dspy-jsonl", "gepa-trace"], required=True)
    pe.add_argument("--all", action="store_true",
                    help="dspy-jsonl only: emit every candidate (default keeps the best per generation)")
    pe.set_defaults(fn=cmd_export)

    prp = sub.add_parser("replay", help="print the lineage hash for determinism checks")
    prp.add_argument("dir")
    prp.add_argument("--seed", type=int, default=42)
    prp.set_defaults(fn=cmd_replay)

    pc = sub.add_parser("cache", help="inspect or purge the LLM response cache")
    pc.add_argument("dir")
    pc.add_argument("action", choices=["stats", "purge"])
    pc.set_defaults(fn=cmd_cache)

    pd_ = sub.add_parser("dashboard", help="serve a read-only FastAPI dashboard")
    pd_.add_argument("dir")
    pd_.add_argument("--host", default="127.0.0.1")
    pd_.add_argument("--port", type=int, default=8787)
    pd_.set_defaults(fn=cmd_dashboard)

    phub = sub.add_parser("hub", help="persistent archive hub — push/list/pull")
    phub.add_argument("action", choices=["push", "list", "pull"])
    phub.add_argument("--dir",   default=".",     help="experiment dir (for push)")
    phub.add_argument("--tag",   default=None,    help="tag / content-hash (for pull, default=exp name)")
    phub.add_argument("--dest",  default=None,    help="pull destination dir")
    phub.add_argument("--top-k", dest="top_k", type=int, default=10,
                      help="number of best-K genomes to record in the manifest")
    phub.set_defaults(fn=cmd_hub)

    # v0.4 A4 — synthesise-fitness subcommand
    psf = sub.add_parser("synthesise-fitness",
                         help="v0.4 A4: LLM-synthesised fitness.py from I/O pairs")
    psf.add_argument("dir")
    psf.add_argument("--examples", required=True, help="JSONL of {input, output} pairs")
    psf.add_argument("--criterion", default="correctness")
    psf.add_argument("--no-overwrite", dest="no_overwrite", action="store_true",
                     help="write to fitness.synthesised.py instead of fitness.py")
    psf.set_defaults(fn=cmd_synthesise_fitness)

    # v0.6 A5 — transfer train/apply
    pt = sub.add_parser("transfer",
                        help="v0.6 A5: cross-task transfer policy train/apply")
    pt.add_argument("action", choices=["train", "apply"])
    pt.add_argument("--experiments", nargs="+", default=[],
                    help="experiment dirs (train mode)")
    pt.add_argument("--k", type=int, default=3, help="k-NN neighbours (train mode)")
    pt.add_argument("--out", default=None, help="policy output path (train mode)")
    pt.add_argument("--policy", default=None, help="policy input path (apply mode)")
    pt.add_argument("--target", default=None, help="target experiment dir (apply mode)")
    pt.set_defaults(fn=cmd_transfer)

    # v0.4 A3 — coevolve subcommand
    pce = sub.add_parser("coevolve",
                         help="v0.4 A3: solver / adversary co-evolution")
    pce.add_argument("dir")
    pce.add_argument("--generations", type=int, default=10)
    pce.add_argument("--adversaries", type=int, default=4,
                     help="initial adversary seeds")
    pce.add_argument("--concurrency", type=int, default=4)
    pce.add_argument("--budget", type=float, default=0.5)
    pce.add_argument("--seed", type=int, default=42)
    pce.add_argument("--max-adversary-gens", dest="max_adversary_gens",
                     type=int, default=0,
                     help="cap adversary evolution (0 = unbounded)")
    pce.set_defaults(fn=cmd_coevolve)

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
