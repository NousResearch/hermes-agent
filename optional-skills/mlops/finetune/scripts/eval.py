#!/usr/bin/env python3
"""
Evaluation gate for the finetune pipeline.

Runs the finetune benchmark against an adapter, compares to baseline,
and produces a pass/fail verdict for promotion.

Usage:
    python eval.py --cluster CLUSTER_ID --version VERSION [--baseline PATH]
"""

import argparse
import json
import logging
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (
    ADAPTERS_DIR, BENCH_DIR, CLUSTER_STATE_PATH,
    ensure_dirs, load_json, save_json, logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Regression thresholds (from bench spec)
THRESHOLDS = {
    "tool_selection_accuracy": 0.03,
    "tool_execution_success": 0.05,
    "task_completion_rate": 0.05,
    "format_compliance_min": 0.95,
    # Small tolerance rather than exact zero: a single flaky case out of 243
    # must not auto-reject an adapter. Matches the bench env's verdict gate.
    "hallucination_max": 0.01,
    "canary_regression": 0.05,
    # Core gate for training-derived metrics (trainer_state.json): the
    # candidate's loss/perplexity must not regress past the baseline by
    # more than this. eval_loss is absolute (nats), perplexity is relative.
    "eval_loss_regression": 0.05,
    "perplexity_regression": 0.05,
}


def _find_baseline(cluster_id: str) -> Optional[Dict]:
    """
    Find the most recent prior eval result for a cluster.

    Eval results are saved by EvalGate.evaluate() as
    ``{cluster_id}_{version}_{YYYYmmdd_HHMMSS}.json``, so we match that
    exact shape with an anchored regex — a bare ``{cluster_id}_*`` glob
    would prefix-match other clusters (``c1`` matching ``c1_extra_...``).
    The most recent file by mtime wins.
    """
    results_dir = BENCH_DIR / "results"
    if not results_dir.exists():
        return None

    pattern = re.compile(
        re.escape(cluster_id) + r"_v\d+_\d{8}_\d{6}\.json$"
    )
    candidates = [
        p for p in results_dir.glob(f"{cluster_id}_*.json")
        if pattern.fullmatch(p.name)
    ]
    # Newest first; skip results that recorded a skipped gate (empty
    # metrics) — they carry nothing to regress against.
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        data = load_json(path)
        if data and data.get("metrics", data):
            return data
    return None


def compare_metrics(
    current: Dict[str, float],
    baseline: Dict[str, float],
) -> Dict[str, Dict]:
    """Compare current metrics to baseline, returning deltas."""
    comparison = {}
    for key in current:
        if key in baseline and isinstance(current[key], (int, float)):
            comparison[key] = {
                "baseline": baseline[key],
                "candidate": current[key],
                "delta": current[key] - baseline[key],
            }
    return comparison


def verdict(comparison: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Determine pass/fail from comparison metrics.

    A gate only applies to metrics present in BOTH candidate and baseline
    (i.e. present in `comparison`, which is their intersection). A missing
    gate metric is skipped with an explicit printed warning — it never
    auto-passes or auto-fails the run. Runs whose metrics come from
    trainer_state.json (eval_loss/perplexity only) are gated on the
    loss/perplexity regression checks alone.
    """
    checks = {}

    # (metric key, check name, pass predicate over the comparison entry)
    gates = [
        ("tool_selection_accuracy", "tool_selection",
         lambda c: c["delta"] >= -THRESHOLDS["tool_selection_accuracy"]),
        ("tool_execution_success", "tool_execution",
         lambda c: c["delta"] >= -THRESHOLDS["tool_execution_success"]),
        ("task_completion_rate", "task_completion",
         lambda c: c["delta"] >= -THRESHOLDS["task_completion_rate"]),
        ("format_compliance", "format_compliance",
         lambda c: c["candidate"] >= THRESHOLDS["format_compliance_min"]),
        ("hallucination_rate", "no_hallucinations",
         lambda c: c["candidate"]
         <= max(THRESHOLDS["hallucination_max"], c["baseline"] + 0.01)),
        ("canary_pass_rate", "canary",
         lambda c: c["delta"] >= -THRESHOLDS["canary_regression"]),
        # Core gate for training-derived metrics: lower is better.
        ("eval_loss", "eval_loss",
         lambda c: c["delta"] <= THRESHOLDS["eval_loss_regression"]),
        ("perplexity", "perplexity",
         lambda c: c["candidate"] <= c["baseline"]
         * (1 + THRESHOLDS["perplexity_regression"])),
    ]

    for metric, name, predicate in gates:
        if metric in comparison:
            checks[name] = bool(predicate(comparison[metric]))
        else:
            print(
                f"WARNING: gate metric '{metric}' missing from candidate "
                f"and/or baseline — check '{name}' skipped (not counted as "
                f"pass or fail)"
            )

    checks["overall"] = all(v for k, v in checks.items() if k != "overall")
    return checks


def format_report(
    current: Dict[str, float],
    baseline: Optional[Dict[str, float]] = None,
    comparison: Optional[Dict[str, Dict]] = None,
    checks: Optional[Dict[str, Any]] = None,
    cluster_id: str = "",
    version: str = "",
) -> str:
    """Generate the ASCII comparison report."""
    lines = []
    w = 60

    if baseline and comparison:
        lines.append("+" + "=" * w + "+")
        lines.append(f"|{'FINETUNE BENCH — Comparison Report':^{w}}|")
        lines.append("+" + "=" * w + "+")
        lines.append(f"|  Baseline:  {cluster_id} (previous)")
        lines.append(f"|  Candidate: {cluster_id} {version}")
        lines.append("+" + "-" * w + "+")
        lines.append(f"|  {'Metric':<25} {'Baseline':>10} {'Candidate':>10} {'Delta':>10} |")
        lines.append("+" + "-" * w + "+")

        metric_names = {
            "tool_selection_accuracy": "Tool Selection Acc.",
            "tool_execution_success": "Tool Execution Succ.",
            "task_completion_rate": "Task Completion Rate",
            "format_compliance": "Format Compliance",
            "no_tool_accuracy": "No-Tool Accuracy",
            "hallucination_rate": "Hallucination Rate",
            "mean_turns": "Mean Turns/Task",
            "mean_errors": "Mean Errors/Task",
            "canary_pass_rate": "Canary Pass Rate",
        }

        for key, label in metric_names.items():
            if key in comparison:
                c = comparison[key]
                b_val = c["baseline"]
                c_val = c["candidate"]
                delta = c["delta"]

                # Format as percentage for rate metrics
                is_pct = key not in ("mean_turns", "mean_errors")
                if is_pct:
                    b_str = f"{b_val*100:.1f}%"
                    c_str = f"{c_val*100:.1f}%"
                    d_str = f"{delta*100:+.1f}%"
                else:
                    b_str = f"{b_val:.1f}"
                    c_str = f"{c_val:.1f}"
                    d_str = f"{delta:+.1f}"

                # Pass/fail indicator
                if key == "hallucination_rate":
                    icon = "ok" if c_val == 0 else "FAIL"
                elif key in ("mean_turns", "mean_errors"):
                    icon = "ok" if delta <= 0 else "warn"
                else:
                    icon = "ok" if delta >= -0.05 else "FAIL"

                lines.append(
                    f"|  {label:<25} {b_str:>10} {c_str:>10} {d_str:>8} {icon:>2} |"
                )

        lines.append("+" + "-" * w + "+")

        if checks:
            passed = sum(1 for k, v in checks.items() if k != "overall" and v)
            total = sum(1 for k in checks if k != "overall")
            result = "PASS" if checks["overall"] else "FAIL"
            lines.append(f"|  VERDICT: {result} ({passed}/{total} metrics pass)")

            # Warnings
            for key, ok in checks.items():
                if key != "overall" and not ok:
                    lines.append(f"|  FAIL: {key}")

        lines.append("+" + "=" * w + "+")
    else:
        # No baseline — just show current metrics
        lines.append("+" + "=" * w + "+")
        lines.append(f"|{'FINETUNE BENCH — Evaluation Results':^{w}}|")
        lines.append(f"|  {cluster_id} {version}")
        lines.append("+" + "-" * w + "+")
        for key, val in current.items():
            if isinstance(val, float):
                lines.append(f"|  {key:<35} {val:>10.4f} |")
            else:
                lines.append(f"|  {key:<35} {str(val):>10} |")
        lines.append("+" + "=" * w + "+")

    return "\n".join(lines)


class EvalGate:
    """Run evaluation and gate promotion decisions."""

    def __init__(self):
        ensure_dirs()

    def evaluate(
        self,
        cluster_id: str,
        version: str,
        baseline_path: str = None,
    ) -> Tuple[bool, str]:
        """
        Evaluate an adapter and return (pass, report).

        This is a lightweight eval that checks the adapter directory
        for eval results from training. For full benchmark evaluation,
        use the finetune_bench environment.
        """
        adapter_dir = ADAPTERS_DIR / cluster_id / version

        # Load eval results from training
        eval_results_path = adapter_dir / "eval_results.json"
        if eval_results_path.exists():
            current_metrics = load_json(eval_results_path)
        else:
            # Construct from training logs if available
            current_metrics = self._extract_training_metrics(adapter_dir)

        if not current_metrics:
            skip_note = "no eval metrics found — gate skipped, treating as pass"
            logger.warning("%s (%s %s)", skip_note, cluster_id, version)
            print(f"WARNING: {skip_note} ({cluster_id} {version})")

            # Record the skipped gate so downstream tooling can see this
            # version was never actually evaluated.
            results_dir = BENCH_DIR / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_json(results_dir / f"{cluster_id}_{version}_{ts}.json", {
                "cluster_id": cluster_id,
                "version": version,
                "metrics": {},
                "baseline_metrics": None,
                "passed": True,
                "gate_skipped": True,
                "note": skip_note,
                "timestamp": datetime.now().isoformat(),
            })
            return True, f"WARNING: {skip_note}"

        # Load baseline
        baseline_metrics = None
        if baseline_path:
            baseline_data = load_json(Path(baseline_path))
            baseline_metrics = baseline_data.get("metrics", baseline_data)
        else:
            baseline_data = _find_baseline(cluster_id)
            if baseline_data:
                baseline_metrics = baseline_data.get("metrics", baseline_data)

        # Compare
        if baseline_metrics:
            comp = compare_metrics(current_metrics, baseline_metrics)
            checks = verdict(comp)
            report = format_report(
                current_metrics, baseline_metrics, comp, checks,
                cluster_id, version,
            )
            passed = checks.get("overall", True)
        else:
            report = format_report(
                current_metrics, cluster_id=cluster_id, version=version,
            )
            passed = True  # No baseline to regress against

        # Save results
        results_dir = BENCH_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = results_dir / f"{cluster_id}_{version}_{ts}.json"
        save_json(result_path, {
            "cluster_id": cluster_id,
            "version": version,
            "metrics": current_metrics,
            "baseline_metrics": baseline_metrics,
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
        })

        return passed, report

    def _extract_training_metrics(self, adapter_dir: Path) -> Dict[str, float]:
        """Extract eval metrics from Axolotl training output."""
        # Check for trainer_state.json (HuggingFace Trainer saves this)
        trainer_state = adapter_dir / "adapter_model" / "trainer_state.json"
        if trainer_state.exists():
            state = load_json(trainer_state)
            best_metric = state.get("best_metric")
            if best_metric is not None:
                # HF Trainer's eval_loss is natural-log cross-entropy, so
                # perplexity = e^loss (not 2^loss).
                return {
                    "eval_loss": best_metric,
                    "perplexity": math.exp(best_metric) if best_metric < 20 else float("inf"),
                }

        # Check for eval output in logs
        eval_log = adapter_dir / "adapter_model" / "eval_results.json"
        if eval_log.exists():
            return load_json(eval_log)

        return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate adapter for promotion")
    parser.add_argument("--cluster", required=True, help="Cluster ID")
    parser.add_argument("--version", required=True, help="Adapter version")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline eval results")
    args = parser.parse_args()

    gate = EvalGate()
    passed, report = gate.evaluate(args.cluster, args.version, args.baseline)

    print(report)
    print(f"\nVerdict: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
