"""
Hermes Benchmark Suite - Unified CLI dispatcher.

Usage:
  python -m benchmarks                          # default: suite=a, runs=1
  python -m benchmarks --quick                   # Suite A only, 1 run
  python -m benchmarks --full                    # All suites, 3 runs
  python -m benchmarks --academic                # LongMemEval + LoCoMo + HotpotQA
  python -m benchmarks --compare before.json after.json
  python -m benchmarks --report                  # Markdown report from latest results
  python -m benchmarks --dashboard               # ASCII dashboard from latest results
  python -m benchmarks --ablation param=v1,v2,v3

All standard runner flags (--backend, --suite, --runs, --json, --judge-model,
--profile, --embedding, --output-dir, etc.) are passed through unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_RESULT = RESULTS_DIR / "baseline-flat.json"


def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {p}", file=sys.stderr)
        sys.exit(1)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _latest_result() -> dict:
    """Load the default/latest result file."""
    if not DEFAULT_RESULT.exists():
        # Fall back to any json in results/
        candidates = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime,
                            reverse=True)
        if not candidates:
            print(f"ERROR: no result files found in {RESULTS_DIR}", file=sys.stderr)
            sys.exit(1)
        return _load_json(str(candidates[0]))
    return _load_json(str(DEFAULT_RESULT))


def _inject_sys_argv(extra: list[str]) -> None:
    """Replace sys.argv with a fresh list for sub-parsers."""
    sys.argv = [sys.argv[0]] + extra


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def mode_standard(runner_args: list[str]) -> None:
    """Run the standard benchmark suite (default, --quick, or --full)."""
    # Import here so we only pay the cost when needed
    import benchmarks.runner as runner_mod

    _inject_sys_argv(runner_args)
    runner_mod.main()


def mode_academic(full: bool, passthrough: list[str]) -> None:
    """Run all three academic benchmarks sequentially."""
    print()
    print("=" * 65)
    print("  Running Academic Benchmark Suite...")
    print("=" * 65)

    lme_sample  = None if full else 50
    loc_sample  = None if full else 50
    hpq_sample  = None if full else 100

    summaries = {}

    # ---- LongMemEval ----
    try:
        from benchmarks.longmemeval.runner import main as lme_main
        argv = [sys.argv[0]]
        if lme_sample is not None:
            argv += ["--sample", str(lme_sample)]
        argv += passthrough
        _inject_sys_argv(argv[1:])
        print(f"\n[1/3] LongMemEval  (sample={lme_sample or 'all'})")
        lme_main()
        result_path = Path("benchmarks/results/longmemeval.json")
        if result_path.exists():
            with open(result_path, encoding="utf-8") as f:
                summaries["longmemeval"] = json.load(f)
    except Exception as exc:
        print(f"  WARNING: LongMemEval failed: {exc}", file=sys.stderr)

    # ---- LoCoMo ----
    try:
        from benchmarks.locomo.runner import main as loc_main
        argv = [sys.argv[0]]
        if loc_sample is not None:
            argv += ["--sample", str(loc_sample)]
        argv += passthrough
        _inject_sys_argv(argv[1:])
        print(f"\n[2/3] LoCoMo  (sample={loc_sample or 'all'})")
        loc_main()
        result_path = Path("benchmarks/results/locomo.json")
        if result_path.exists():
            with open(result_path, encoding="utf-8") as f:
                summaries["locomo"] = json.load(f)
    except Exception as exc:
        print(f"  WARNING: LoCoMo failed: {exc}", file=sys.stderr)

    # ---- HotpotQA ----
    try:
        from benchmarks.hotpotqa.runner import main as hpq_main
        argv = [sys.argv[0]]
        if hpq_sample is not None:
            argv += ["--sample", str(hpq_sample)]
        argv += passthrough
        _inject_sys_argv(argv[1:])
        print(f"\n[3/3] HotpotQA  (sample={hpq_sample or 'all'})")
        hpq_main()
        result_path = Path("benchmarks/results/hotpotqa.json")
        if result_path.exists():
            with open(result_path, encoding="utf-8") as f:
                summaries["hotpotqa"] = json.load(f)
    except Exception as exc:
        print(f"  WARNING: HotpotQA failed: {exc}", file=sys.stderr)

    # ---- Combined summary ----
    print()
    print("=" * 65)
    print("  ACADEMIC BENCHMARK SUITE - COMBINED SUMMARY")
    print("=" * 65)
    if not summaries:
        print("  No results collected.")
    else:
        print(f"  {'Benchmark':<20} {'N':>6}  {'Score':>8}")
        print(f"  {'-'*20} {'-'*6}  {'-'*8}")
        scores = []
        for name, data in summaries.items():
            score = data.get("score", 0.0)
            total = data.get("total", 0)
            scores.append(score)
            print(f"  {name:<20} {total:>6}  {score:>7.1%}")
        print(f"  {'-'*20} {'-'*6}  {'-'*8}")
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  {'Average':<20} {'':>6}  {avg:>7.1%}")
    print("=" * 65)
    print()


def mode_compare(before_path: str, after_path: str) -> None:
    """Load two JSON files and print comparison."""
    before = _load_json(before_path)
    after = _load_json(after_path)

    before_cats = before.get("per_category_mean", {})
    after_cats = after.get("per_category_mean", {})

    from benchmarks.visualize.charts import comparison_chart
    title = f"{before.get('backend', 'before')} vs {after.get('backend', 'after')}"
    print(comparison_chart(before_cats, after_cats, title=title))

    # Statistical comparison if both have run lists
    before_runs = before.get("runs", [])
    after_runs  = after.get("runs", [])
    if before_runs and after_runs:
        try:
            from benchmarks.statistical import compare_runs
            from benchmarks.interface import RunResult

            def _to_run_results(run_list):
                """Convert raw dicts to minimal RunResult objects."""
                results = []
                for r in run_list:
                    rr = RunResult(
                        seed=r.get("seed", 0),
                        results_by_category={},
                        overall_score=r.get("overall_score", 0.0),
                        token_usage={},
                        wall_time_seconds=r.get("wall_time_seconds", 0.0),
                    )
                    results.append(rr)
                return results

            runs_a = _to_run_results(before_runs)
            runs_b = _to_run_results(after_runs)
            sig = compare_runs(runs_a, runs_b)

            bname = before.get("backend", "before")
            aname = after.get("backend", "after")
            print("=" * 65)
            print("  STATISTICAL COMPARISON")
            print("=" * 65)
            print(f"  {bname}: {sig.baseline_mean:.4f}")
            print(f"  {aname}: {sig.experiment_mean:.4f}")
            print(f"  Improvement:     {sig.improvement:+.2f} pp")
            print(f"  Effect size (d): {sig.effect_size:.4f}")
            print(f"  p-value:         {sig.p_value:.4f}")
            print(f"  Significant:     {'YES' if sig.significant else 'NO'} (p<0.05)")
            print(f"  Test:            {sig.test_name}")
            print("=" * 65)
            print()
        except Exception as exc:
            print(f"  (Statistical test skipped: {exc})")


def mode_report(result_path: str | None = None) -> None:
    """Generate a markdown report from the latest results."""
    if result_path:
        data = _load_json(result_path)
    else:
        data = _latest_result()

    from benchmarks.visualize.report import generate_report
    backend = data.get("backend", "results")
    out_path = RESULTS_DIR / f"{backend}_report.md"
    report = generate_report(data, output_path=out_path)
    print(report)


def mode_dashboard(result_path: str | None = None) -> None:
    """Print ASCII dashboard from the latest results."""
    if result_path:
        data = _load_json(result_path)
    else:
        data = _latest_result()

    from benchmarks.visualize.charts import summary_dashboard
    print(summary_dashboard(data))


def _mode_optimize(args) -> None:
    """Delegate to benchmarks.optimize with the given strategy and args."""
    import benchmarks.optimize as opt_mod

    # Build param space
    param_space = opt_mod._build_param_space(getattr(args, 'optimize_params', None))

    suite = args.suite or 'a'
    runs  = args.runs or 1
    judge = args.judge_model or 'heuristic'
    emb   = args.embedding or 'auto'
    trials = args.trials

    evaluate_fn = opt_mod.make_evaluator(suite=suite, runs=runs,
                                         judge_model=judge, embedding=emb)

    import time
    strategy = args.optimize
    t0 = time.time()
    if strategy == 'grid':
        results = opt_mod.grid_search(param_space, evaluate_fn, max_configs=trials)
    elif strategy == 'bayesian':
        n_init = max(3, trials // 6)
        results = opt_mod.bayesian_optimize(param_space, evaluate_fn,
                                            n_trials=trials, n_initial=n_init)
    else:
        results = opt_mod.random_search(param_space, evaluate_fn, n_trials=trials)
    elapsed = time.time() - t0

    print(f"\nOptimization complete in {elapsed:.1f}s")
    opt_mod.print_top_results(results, n=10)
    frontier = opt_mod.pareto_frontier(results)
    opt_mod.print_pareto(frontier)
    out_path = opt_mod.save_results(results, strategy, suite)
    print(f"Results saved to {out_path}")


def mode_ablation(spec: str, passthrough: list[str]) -> None:
    """Delegate a parameter sweep to compare_configs."""
    import benchmarks.compare_configs as cc_mod

    # parse spec: param=v1,v2,v3
    argv = [sys.argv[0], "--sweep", spec] + passthrough
    _inject_sys_argv(argv[1:])
    cc_mod.main()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Hermes Benchmark Suite - Unified CLI",
        add_help=True,
    )

    # Mode flags (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick",    action="store_true",
                            help="Suite A only, 1 run")
    mode_group.add_argument("--full",     action="store_true",
                            help="All suites, 3 runs")
    mode_group.add_argument("--academic", action="store_true",
                            help="LongMemEval + LoCoMo + HotpotQA")
    mode_group.add_argument("--compare",  nargs=2, metavar=("BEFORE", "AFTER"),
                            help="Compare two result JSON files")
    mode_group.add_argument("--report",   action="store_true",
                            help="Generate markdown report from latest results")
    mode_group.add_argument("--dashboard", action="store_true",
                            help="Print ASCII dashboard from latest results")
    mode_group.add_argument("--ablation", metavar="PARAM=v1,v2,...",
                            help="Parameter sweep (delegates to compare_configs)")
    mode_group.add_argument("--optimize", metavar="STRATEGY",
                            choices=["random", "grid", "bayesian"],
                            help="Parameter optimization strategy (random/grid/bayesian)")

    # Pass-through args for runner.main() (standard benchmark modes)
    parser.add_argument("--backend",    default=None,
                        help="Memory backend (default: baseline-flat)")
    parser.add_argument("--profile",    default=None,
                        help="Config profile for cognitive backend")
    parser.add_argument("--embedding",  default=None,
                        help="Embedding model: auto, sentence-transformers, tfidf")
    parser.add_argument("--suite",      default=None,
                        help="Suite(s) to run: a,b,c,d,e,f or 'all'")
    parser.add_argument("--runs",       type=int, default=None,
                        help="Number of runs per benchmark")
    parser.add_argument("--judge-model", default=None,
                        help="Model for LLM-as-judge (default: heuristic)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for JSON results")
    parser.add_argument("--seeds",      nargs="+", type=int, default=None,
                        help="Random seeds for runs")
    parser.add_argument("--contradiction-llm", default=None,
                        help="LLM model for contradiction fallback")
    parser.add_argument("--json",       action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--result-file", default=None,
                        help="Path to result JSON for --report / --dashboard")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of trials for --optimize (default: 30)")
    parser.add_argument("--optimize-params", default=None,
                        help="Comma-separated param names to optimize (default: all)")

    args, unknown = parser.parse_known_args()

    # Build pass-through argv for runner.main() or academic sub-runners
    runner_argv: list[str] = list(unknown)  # anything unrecognised goes straight through
    if args.backend:        runner_argv += ["--backend",           args.backend]
    if args.profile:        runner_argv += ["--profile",           args.profile]
    if args.embedding:      runner_argv += ["--embedding",         args.embedding]
    if args.suite:          runner_argv += ["--suite",             args.suite]
    if args.runs is not None: runner_argv += ["--runs",            str(args.runs)]
    if args.judge_model:    runner_argv += ["--judge-model",       args.judge_model]
    if args.output_dir:     runner_argv += ["--output-dir",        args.output_dir]
    if args.seeds:          runner_argv += ["--seeds"] + [str(s) for s in args.seeds]
    if args.contradiction_llm: runner_argv += ["--contradiction-llm", args.contradiction_llm]
    if args.json:           runner_argv += ["--json"]

    # ---- Dispatch ----

    if args.optimize:
        _mode_optimize(args)

    elif args.compare:
        mode_compare(args.compare[0], args.compare[1])

    elif args.report:
        mode_report(args.result_file)

    elif args.dashboard:
        mode_dashboard(args.result_file)

    elif args.ablation:
        mode_ablation(args.ablation, runner_argv)

    elif args.academic:
        mode_academic(full=args.full, passthrough=runner_argv)

    elif args.quick:
        # Suite A, 1 run, baseline-flat backend
        argv = ["--suite", "a", "--runs", "1"]
        if not any(a.startswith("--backend") for a in runner_argv):
            argv += ["--backend", "baseline-flat"]
        argv += runner_argv
        mode_standard(argv)

    elif args.full:
        # All suites, 3 runs, baseline-flat backend
        argv = ["--suite", "all", "--runs", "3"]
        if not any(a.startswith("--backend") for a in runner_argv):
            argv += ["--backend", "baseline-flat"]
        argv += runner_argv
        mode_standard(argv)

    else:
        # Default: suite=a, runs=1
        argv = []
        if not any(a.startswith("--suite") for a in runner_argv):
            argv += ["--suite", "a"]
        if not any(a.startswith("--runs") for a in runner_argv):
            argv += ["--runs", "1"]
        argv += runner_argv
        mode_standard(argv)


if __name__ == "__main__":
    main()
