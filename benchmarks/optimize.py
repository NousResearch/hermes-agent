"""
Parameter fuzzing / optimization framework for CognitiveMemoryConfig.

Searches the config space to find parameter settings that maximize benchmark score.

Usage:
    python -m benchmarks.optimize --strategy random --trials 30 --suite a
    python -m benchmarks.optimize --strategy grid --params gravity_dampening_factor,resolution_boost_factor --suite a
    python -m benchmarks.optimize --strategy bayesian --trials 20 --suite a
    python -m benchmarks.optimize --pareto results/optimize_20250327_082300.json

Reference: Bergstra & Bengio (2012) "Random Search for Hyper-Parameter Optimization"
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Parameter space definitions
# ---------------------------------------------------------------------------

@dataclass
class ParamRange:
    """Defines the search range for a single config parameter."""
    name: str           # must match a field in CognitiveMemoryConfig
    low: float          # inclusive lower bound
    high: float         # inclusive upper bound
    step: float = 0.0   # 0 = continuous; >0 = discrete steps
    log_scale: bool = False  # sample / grid in log space

    def grid_values(self) -> List[float]:
        """Return all discrete grid values for this parameter."""
        if self.step <= 0:
            # Continuous: use 5 evenly spaced points as default grid
            n = 5
            if self.log_scale:
                lo, hi = math.log(self.low), math.log(self.high)
                return [round(math.exp(lo + i * (hi - lo) / (n - 1)), 6)
                        for i in range(n)]
            return [round(self.low + i * (self.high - self.low) / (n - 1), 6)
                    for i in range(n)]

        if self.log_scale:
            lo, hi = math.log(self.low), math.log(self.high)
            step_log = math.log(self.step) if self.step > 1 else self.step
            # discrete log steps: just build list in linear, round to step
            values = []
            v = self.low
            while v <= self.high + 1e-9:
                values.append(round(v, 6))
                v = round(v + self.step, 10)
            return values

        values = []
        v = self.low
        while v <= self.high + 1e-9:
            values.append(round(v, 6))
            v = round(v + self.step, 10)
        return values

    def sample(self, rng: random.Random) -> float:
        """Draw a random sample from this parameter's range."""
        if self.log_scale:
            lo, hi = math.log(self.low), math.log(self.high)
            v = math.exp(rng.uniform(lo, hi))
        else:
            v = rng.uniform(self.low, self.high)

        if self.step > 0:
            # Round to nearest discrete step
            v = round(round((v - self.low) / self.step) * self.step + self.low, 6)
            v = max(self.low, min(self.high, v))

        return round(v, 6)


# Default search space covering tunable memory backend parameters
DEFAULT_SEARCH_SPACE: List[ParamRange] = [
    ParamRange('gravity_dampening_factor',   0.2, 0.8,  step=0.1),
    ParamRange('hub_dampening_max_penalty',  0.3, 0.9,  step=0.1),
    ParamRange('resolution_boost_factor',    1.0, 2.0,  step=0.25),
    ParamRange('w_semantic',                 0.3, 1.0,  step=0.1),
    ParamRange('w_importance',               0.1, 0.5,  step=0.1),
    ParamRange('hebbian_learning_rate',      0.01, 0.2, step=0.01),
    ParamRange('hebbian_homeostasis_target', 0.3, 0.8,  step=0.1),
    ParamRange('d',                          0.3, 0.8,  step=0.1),   # ACT-R decay
    ParamRange('top_k',                      3,   15,   step=1),
]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def make_evaluator(suite: str = 'a', runs: int = 1,
                   judge_model: str = 'heuristic',
                   embedding: str = 'auto') -> Callable[[dict], dict]:
    """Return a callable that evaluates a parameter config dict.

    The returned function accepts a dict of {param_name: value} overrides,
    injects them into CognitiveBenchmarkAdapter via BenchmarkConfig.parameters,
    runs the benchmark suite, and returns a result dict containing at minimum:
        score            - overall accuracy (0.0 – 1.0)
        tokens_per_query - average recall tokens per query (cost proxy)
        wall_time        - seconds taken
        per_category     - {category: score} mapping
    """
    # Import here to avoid circular imports at module level
    from benchmarks.interface import BenchmarkConfig
    from benchmarks.runner import run_single

    base_seeds = [42, 43, 44][:runs]

    def evaluate(params: dict) -> dict:
        config = BenchmarkConfig(
            backend_name='baseline-flat',
            profile='balanced',
            embedding_model=embedding,
            num_runs=runs,
            judge_model=judge_model,
            seeds=base_seeds,
            parameters={
                'profile': 'balanced',
                'embedding_model': embedding,
                'suites': ['a', 'b', 'c', 'd', 'e', 'f', 'g'] if suite == 'all' else [s.strip() for s in suite.split(',')],
                **params,
            },
        )

        scores = []
        tokens_per_query_list = []
        wall_times = []
        per_category_accum: Dict[str, List[float]] = {}

        for seed in base_seeds:
            try:
                run = run_single(config, seed)
            except Exception as exc:
                # If a config combo crashes the backend, treat as zero score
                return {
                    'score': 0.0,
                    'tokens_per_query': 9999,
                    'wall_time': 0.0,
                    'per_category': {},
                    'error': str(exc),
                    'params': params,
                }

            scores.append(run.overall_score)
            tpq = run.token_usage.get('avg_recall_tokens_per_query', 0)
            tokens_per_query_list.append(tpq)
            wall_times.append(run.wall_time_seconds)
            for cat, cr in run.results_by_category.items():
                per_category_accum.setdefault(cat, []).append(cr.score)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        mean_tpq   = sum(tokens_per_query_list) / len(tokens_per_query_list) if tokens_per_query_list else 0
        mean_time  = sum(wall_times) / len(wall_times) if wall_times else 0.0
        per_cat    = {cat: sum(v) / len(v) for cat, v in per_category_accum.items()}

        return {
            'score': mean_score,
            'tokens_per_query': mean_tpq,
            'wall_time': mean_time,
            'per_category': per_cat,
            'params': params,
        }

    return evaluate


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------

def grid_search(
    param_space: List[ParamRange],
    evaluate_fn: Callable[[dict], dict],
    max_configs: int = 100,
    verbose: bool = True,
) -> List[dict]:
    """Exhaustive grid search over discrete parameter values.

    Iterates through all combinations of grid_values() for each parameter
    up to max_configs total evaluations.  Combinations are enumerated via
    a simple Cartesian product (uses itertools internally but no extra deps).

    Returns list of {params, results} dicts sorted by score (descending).
    """
    import itertools

    grid = {pr.name: pr.grid_values() for pr in param_space}
    names = [pr.name for pr in param_space]
    value_lists = [grid[n] for n in names]

    combos = list(itertools.product(*value_lists))
    if len(combos) > max_configs:
        # Subsample deterministically
        step = len(combos) // max_configs
        combos = combos[::step][:max_configs]

    if verbose:
        print(f"Grid search: {len(combos)} configs over {len(names)} params")

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(names, combo))
        if verbose:
            print(f"  [{i}/{len(combos)}] {params}", end=' ', flush=True)
        r = evaluate_fn(params)
        if verbose:
            print(f"-> score={r.get('score', 0):.4f}")
        results.append({'params': params, 'results': r})

    results.sort(key=lambda x: x['results'].get('score', 0.0), reverse=True)
    return results


def random_search(
    param_space: List[ParamRange],
    evaluate_fn: Callable[[dict], dict],
    n_trials: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> List[dict]:
    """Random search over the parameter space.

    More sample-efficient than grid search for high-dimensional spaces.
    Reference: Bergstra & Bengio (2012).

    Returns list of {params, results} sorted by score descending.
    """
    rng = random.Random(seed)
    results = []

    if verbose:
        print(f"Random search: {n_trials} trials over {len(param_space)} params (seed={seed})")

    for i in range(1, n_trials + 1):
        params = {pr.name: pr.sample(rng) for pr in param_space}
        if verbose:
            print(f"  [{i}/{n_trials}] {params}", end=' ', flush=True)
        r = evaluate_fn(params)
        if verbose:
            print(f"-> score={r.get('score', 0):.4f}")
        results.append({'params': params, 'results': r})

    results.sort(key=lambda x: x['results'].get('score', 0.0), reverse=True)
    return results


def bayesian_optimize(
    param_space: List[ParamRange],
    evaluate_fn: Callable[[dict], dict],
    n_trials: int = 30,
    seed: int = 42,
    n_initial: int = 5,
    verbose: bool = True,
) -> List[dict]:
    """Simple Bayesian optimization using a Gaussian Process surrogate.

    Uses scipy.optimize / sklearn if available; falls back to random search
    with a warm-start exploitation phase otherwise.

    The GP is fit on observed (X, y) pairs and the next point is chosen by
    maximizing Expected Improvement (EI) via random sampling of the acquisition
    function (avoids scipy.optimize dependency for the inner loop).

    Returns list of {params, results} sorted by score descending.
    """
    rng = random.Random(seed)
    results_list: List[dict] = []
    observed_X: List[List[float]] = []
    observed_y: List[float] = []
    names = [pr.name for pr in param_space]

    def _params_to_vec(params: dict) -> List[float]:
        vec = []
        for pr in param_space:
            v = params[pr.name]
            if pr.log_scale:
                v = math.log(max(v, 1e-10))
            vec.append(v)
        return vec

    def _vec_to_params(vec: List[float]) -> dict:
        params = {}
        for i, pr in enumerate(param_space):
            v = vec[i]
            if pr.log_scale:
                v = math.exp(v)
            if pr.step > 0:
                v = round(round((v - pr.low) / pr.step) * pr.step + pr.low, 6)
                v = max(pr.low, min(pr.high, v))
            params[pr.name] = round(v, 6)
        return params

    def _bounds_vec() -> List[Tuple[float, float]]:
        bounds = []
        for pr in param_space:
            lo, hi = pr.low, pr.high
            if pr.log_scale:
                lo, hi = math.log(lo), math.log(hi)
            bounds.append((lo, hi))
        return bounds

    def _sample_vec_random() -> List[float]:
        bounds = _bounds_vec()
        return [rng.uniform(lo, hi) for lo, hi in bounds]

    # Try to use sklearn GP; fall back gracefully
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        import numpy as np
        _has_sklearn = True
    except ImportError:
        _has_sklearn = False

    if verbose:
        backend = "GP/sklearn" if _has_sklearn else "random-fallback"
        print(f"Bayesian optimization: {n_trials} trials, {n_initial} initial, backend={backend}")

    def _expected_improvement(X_cand: list, gp, y_best: float) -> List[float]:
        """Compute EI for a list of candidate vectors."""
        import numpy as np
        X_arr = np.array(X_cand)
        mu, sigma = gp.predict(X_arr, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        Z = (mu - y_best) / sigma
        from math import erf, sqrt
        # Vectorised-ish normal CDF / PDF via scipy if available
        try:
            from scipy.stats import norm
            ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
        except ImportError:
            # Manual approximation
            def _phi(z):
                return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
            def _Phi(z):
                return 0.5 * (1 + erf(z / math.sqrt(2)))
            ei = np.array([(mu[j] - y_best) * _Phi(Z[j]) + sigma[j] * _phi(Z[j])
                           for j in range(len(Z))])
        return ei.tolist()

    def _next_candidate(gp, y_best: float) -> List[float]:
        """Pick next candidate via random EI maximization."""
        import numpy as np
        n_cands = 500
        cands = [_sample_vec_random() for _ in range(n_cands)]
        ei_vals = _expected_improvement(cands, gp, y_best)
        best_idx = int(np.argmax(ei_vals))
        return cands[best_idx]

    # Phase 1: random initial samples
    for i in range(1, n_initial + 1):
        vec = _sample_vec_random()
        params = _vec_to_params(vec)
        if verbose:
            print(f"  [init {i}/{n_initial}] {params}", end=' ', flush=True)
        r = evaluate_fn(params)
        score = r.get('score', 0.0)
        if verbose:
            print(f"-> score={score:.4f}")
        results_list.append({'params': params, 'results': r})
        observed_X.append(_params_to_vec(params))
        observed_y.append(score)

    # Phase 2: Bayesian acquisition
    remaining = n_trials - n_initial
    for i in range(1, remaining + 1):
        if _has_sklearn and len(observed_y) >= 2:
            import numpy as np
            X_arr = np.array(observed_X)
            y_arr = np.array(observed_y)
            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                random_state=seed + i,
            )
            try:
                gp.fit(X_arr, y_arr)
                y_best = float(np.max(y_arr))
                vec = _next_candidate(gp, y_best)
            except Exception:
                vec = _sample_vec_random()
        else:
            vec = _sample_vec_random()

        params = _vec_to_params(vec)
        if verbose:
            trial_n = n_initial + i
            print(f"  [{trial_n}/{n_trials}] {params}", end=' ', flush=True)
        r = evaluate_fn(params)
        score = r.get('score', 0.0)
        if verbose:
            print(f"-> score={score:.4f}")
        results_list.append({'params': params, 'results': r})
        observed_X.append(_params_to_vec(params))
        observed_y.append(score)

    results_list.sort(key=lambda x: x['results'].get('score', 0.0), reverse=True)
    return results_list


# ---------------------------------------------------------------------------
# Pareto frontier analysis
# ---------------------------------------------------------------------------

def pareto_frontier(
    results: List[dict],
    x_metric: str = 'tokens_per_query',
    y_metric: str = 'score',
) -> List[dict]:
    """Find Pareto-optimal configurations.

    A configuration is Pareto-optimal if no other configuration is both
    cheaper (lower x_metric) AND better (higher y_metric).

    The returned list is sorted by x_metric ascending.
    """
    def _get(entry: dict, metric: str) -> float:
        r = entry.get('results', entry)
        return float(r.get(metric, 0.0))

    # Remove entries missing either metric
    valid = [e for e in results if _get(e, x_metric) > 0 or _get(e, y_metric) > 0]

    pareto = []
    for cand in valid:
        cand_x = _get(cand, x_metric)
        cand_y = _get(cand, y_metric)
        dominated = False
        for other in valid:
            if other is cand:
                continue
            o_x = _get(other, x_metric)
            o_y = _get(other, y_metric)
            # 'other' dominates 'cand' if it is at least as cheap and strictly better,
            # or strictly cheaper and at least as good
            if o_x <= cand_x and o_y >= cand_y and (o_x < cand_x or o_y > cand_y):
                dominated = True
                break
        if not dominated:
            pareto.append(cand)

    pareto.sort(key=lambda e: _get(e, x_metric))
    return pareto


# ---------------------------------------------------------------------------
# Result I/O helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"


def save_results(results: List[dict], strategy: str, suite: str) -> Path:
    """Persist optimization results to a timestamped JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"optimize_{ts}_{strategy}_{suite}.json"
    payload = {
        "strategy": strategy,
        "suite": suite,
        "timestamp": ts,
        "num_trials": len(results),
        "best_score": results[0]["results"].get("score", 0) if results else 0,
        "best_params": results[0]["params"] if results else {},
        "trials": results,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_results(path: str) -> List[dict]:
    """Load a saved results file and return the trials list."""
    with open(path) as fh:
        data = json.load(fh)
    # Support both raw list and wrapped payload
    if isinstance(data, list):
        return data
    return data.get("trials", [])


def print_top_results(results: List[dict], n: int = 10) -> None:
    """Print the top-N results in a compact table."""
    print(f"\nTop {min(n, len(results))} configurations:")
    print(f"  {'Rank':>4}  {'Score':>8}  {'Tok/Q':>6}  Params")
    print(f"  {'----':>4}  {'-----':>8}  {'-----':>6}  ------")
    for rank, entry in enumerate(results[:n], 1):
        r = entry.get("results", {})
        score = r.get("score", 0.0)
        tpq   = r.get("tokens_per_query", 0)
        p     = entry.get("params", {})
        # Compact param string
        pstr  = "  ".join(f"{k}={v}" for k, v in p.items())
        print(f"  {rank:>4}  {score:>8.4f}  {tpq:>6.0f}  {pstr}")
    print()


def print_pareto(pareto: List[dict],
                 x_metric: str = 'tokens_per_query',
                 y_metric: str = 'score') -> None:
    """Print Pareto-optimal frontier in a compact table."""
    print(f"\nPareto frontier ({x_metric} vs {y_metric}) — {len(pareto)} points:")
    print(f"  {'Score':>8}  {'Tok/Q':>8}  Params")
    print(f"  {'-----':>8}  {'-----':>8}  ------")
    for entry in pareto:
        r = entry.get("results", entry)
        score = r.get(y_metric, 0.0)
        tpq   = r.get(x_metric, 0)
        p     = entry.get("params", {})
        pstr  = "  ".join(f"{k}={v}" for k, v in p.items())
        print(f"  {score:>8.4f}  {tpq:>8.0f}  {pstr}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_param_space(param_names: Optional[str]) -> List[ParamRange]:
    """Build ParamRange list from comma-separated param names or use all defaults."""
    if not param_names:
        return DEFAULT_SEARCH_SPACE

    requested = [n.strip() for n in param_names.split(",")]
    lookup = {pr.name: pr for pr in DEFAULT_SEARCH_SPACE}
    space = []
    for name in requested:
        if name not in lookup:
            print(f"WARNING: unknown param '{name}', skipping", file=sys.stderr)
            continue
        space.append(lookup[name])
    if not space:
        print("ERROR: no valid params specified", file=sys.stderr)
        sys.exit(1)
    return space


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.optimize",
        description="Parameter optimization for CognitiveMemoryConfig",
    )
    parser.add_argument("--strategy", default="random",
                        choices=["random", "grid", "bayesian"],
                        help="Search strategy (default: random)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of trials / max configs (default: 30)")
    parser.add_argument("--suite", default="a",
                        help="Benchmark suite(s) to evaluate on (default: a)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Benchmark runs per trial (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--params", default=None,
                        help="Comma-separated param names to optimize (default: all)")
    parser.add_argument("--judge-model", default="heuristic",
                        help="Judge model (default: heuristic)")
    parser.add_argument("--embedding", default="auto",
                        help="Embedding model (default: auto)")
    parser.add_argument("--pareto", default=None, metavar="RESULTS_JSON",
                        help="Analyze Pareto frontier from an existing results JSON")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top results to display (default: 10)")
    parser.add_argument("--x-metric", default="tokens_per_query",
                        help="X-axis metric for Pareto analysis (default: tokens_per_query)")
    parser.add_argument("--y-metric", default="score",
                        help="Y-axis metric for Pareto analysis (default: score)")

    args = parser.parse_args()

    # ---- Pareto analysis only mode ----
    if args.pareto:
        trials = load_results(args.pareto)
        frontier = pareto_frontier(trials, x_metric=args.x_metric, y_metric=args.y_metric)
        print_top_results(trials, n=args.top)
        print_pareto(frontier, x_metric=args.x_metric, y_metric=args.y_metric)
        return

    # ---- Run optimization ----
    param_space = _build_param_space(args.params)
    evaluate_fn = make_evaluator(
        suite=args.suite,
        runs=args.runs,
        judge_model=args.judge_model,
        embedding=args.embedding,
    )

    t0 = time.time()
    if args.strategy == "grid":
        results = grid_search(param_space, evaluate_fn, max_configs=args.trials)
    elif args.strategy == "bayesian":
        n_init = max(3, args.trials // 6)
        results = bayesian_optimize(
            param_space, evaluate_fn,
            n_trials=args.trials,
            seed=args.seed,
            n_initial=n_init,
        )
    else:
        results = random_search(
            param_space, evaluate_fn,
            n_trials=args.trials,
            seed=args.seed,
        )
    elapsed = time.time() - t0

    print(f"\nOptimization complete in {elapsed:.1f}s")
    print_top_results(results, n=args.top)

    # Pareto frontier
    frontier = pareto_frontier(results, x_metric=args.x_metric, y_metric=args.y_metric)
    print_pareto(frontier, x_metric=args.x_metric, y_metric=args.y_metric)

    # Save
    out_path = save_results(results, args.strategy, args.suite)
    print(f"Results saved to {out_path}")

    if results:
        best = results[0]
        print(f"\nBest config: score={best['results'].get('score', 0):.4f}")
        for k, v in best['params'].items():
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
