"""
Statistical analysis for benchmark results.

Provides:
- Aggregation across multiple runs (mean, std, CI)
- Significance testing between backends (paired t-test, Wilcoxon)
- Effect size computation (Cohen's d)
"""

import math
from typing import List, Tuple

from benchmarks.interface import (
    RunResult, AggregateResult, SignificanceResult,
)


def aggregate_results(runs: List[RunResult]) -> AggregateResult:
    """Aggregate results across multiple benchmark runs."""
    if not runs:
        return AggregateResult()

    scores = [r.overall_score for r in runs]
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
    std = math.sqrt(variance)

    ci_lower, ci_upper = compute_confidence_interval(scores)

    # Per-category aggregation
    all_categories = set()
    for r in runs:
        all_categories.update(r.results_by_category.keys())

    per_cat_mean = {}
    per_cat_std = {}
    for cat in all_categories:
        cat_scores = [
            r.results_by_category[cat].score
            for r in runs
            if cat in r.results_by_category
        ]
        if cat_scores:
            cm = sum(cat_scores) / len(cat_scores)
            cv = sum((s - cm) ** 2 for s in cat_scores) / max(len(cat_scores) - 1, 1)
            per_cat_mean[cat] = cm
            per_cat_std[cat] = math.sqrt(cv)

    total_tokens = sum(
        r.token_usage.get("total_input", 0) + r.token_usage.get("total_output", 0)
        for r in runs
    )

    return AggregateResult(
        num_runs=n,
        mean_score=mean,
        std_score=std,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        per_category_mean=per_cat_mean,
        per_category_std=per_cat_std,
        total_tokens=total_tokens,
    )


def compute_confidence_interval(
    scores: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute confidence interval for a list of scores.

    Uses t-distribution for small samples.
    """
    n = len(scores)
    if n < 2:
        mean = scores[0] if scores else 0
        return (mean, mean)

    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)

    # t-value for 95% CI (two-tailed)
    # Approximation for small n; use scipy for exact values
    t_values = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
        15: 2.145, 20: 2.093, 30: 2.045, 50: 2.009, 100: 1.984,
    }
    # Find closest t-value
    t_val = 1.96  # default for large n
    for k in sorted(t_values.keys()):
        if n <= k:
            t_val = t_values[k]
            break

    margin = t_val * se
    return (mean - margin, mean + margin)


def compare_runs(
    baseline_runs: List[RunResult],
    experiment_runs: List[RunResult],
) -> SignificanceResult:
    """Compare two sets of benchmark runs for statistical significance.

    Uses paired t-test (parametric) and Wilcoxon signed-rank (non-parametric).
    Reports the more conservative result.
    """
    if len(baseline_runs) != len(experiment_runs):
        raise ValueError("Must have equal number of runs for paired comparison")

    baseline_scores = [r.overall_score for r in baseline_runs]
    experiment_scores = [r.overall_score for r in experiment_runs]

    b_mean = sum(baseline_scores) / len(baseline_scores)
    e_mean = sum(experiment_scores) / len(experiment_scores)

    if len(baseline_scores) < 2:
        return SignificanceResult(
            test_name="not_enough_runs",
            p_value=1.0,
            effect_size=0.0,
            significant=False,
            baseline_mean=b_mean,
            experiment_mean=e_mean,
            improvement=(e_mean - b_mean) * 100,
        )

    try:
        from scipy import stats as sp_stats

        # Paired t-test
        t_stat, t_pvalue = sp_stats.ttest_rel(experiment_scores, baseline_scores)

        # Wilcoxon signed-rank (non-parametric alternative)
        try:
            w_stat, w_pvalue = sp_stats.wilcoxon(
                [e - b for e, b in zip(experiment_scores, baseline_scores)]
            )
        except ValueError:
            # Wilcoxon fails if all differences are zero
            w_pvalue = 1.0

        # Use more conservative p-value
        p_value = max(t_pvalue, w_pvalue)
        test_name = "paired_t_test + wilcoxon (conservative)"

    except ImportError:
        # Fallback: simple paired t-test without scipy
        n = len(baseline_scores)
        diffs = [e - b for e, b in zip(experiment_scores, baseline_scores)]
        d_mean = sum(diffs) / n
        d_var = sum((d - d_mean) ** 2 for d in diffs) / max(n - 1, 1)
        d_std = math.sqrt(d_var)
        t_stat = d_mean / (d_std / math.sqrt(n)) if d_std > 0 else 0

        # Rough p-value approximation (two-tailed)
        # For proper p-values, install scipy
        p_value = 1.0  # Conservative: always "not significant" without scipy
        test_name = "paired_t_test (no scipy - install for real p-values)"

    # Cohen's d effect size
    pooled_std = math.sqrt(
        (sum((s - b_mean) ** 2 for s in baseline_scores) +
         sum((s - e_mean) ** 2 for s in experiment_scores)) /
        max(2 * len(baseline_scores) - 2, 1)
    )
    effect_size = (e_mean - b_mean) / pooled_std if pooled_std > 0 else 0

    return SignificanceResult(
        test_name=test_name,
        p_value=p_value,
        effect_size=effect_size,
        significant=p_value < 0.05,
        baseline_mean=b_mean,
        experiment_mean=e_mean,
        improvement=(e_mean - b_mean) * 100,  # percentage points
    )
