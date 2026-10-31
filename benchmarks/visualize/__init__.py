"""
benchmarks.visualize
~~~~~~~~~~~~~~~~~~~~
ASCII/text-based visualization and markdown report generation
for benchmark results.  Pure Python stdlib — no external deps.

Exported functions
------------------
charts:
    radar_chart(categories, title='') -> str
    comparison_chart(before, after, title='') -> str
    metrics_table(metrics, title='') -> str
    regression_chart(history, metric='overall') -> str
    summary_dashboard(result_json) -> str

report:
    generate_report(result_json, output_path=None) -> str
    generate_comparison_report(before_json, after_json) -> str
"""

from .charts import (
    radar_chart,
    comparison_chart,
    metrics_table,
    regression_chart,
    summary_dashboard,
)

from .report import (
    generate_report,
    generate_comparison_report,
)

__all__ = [
    # charts
    "radar_chart",
    "comparison_chart",
    "metrics_table",
    "regression_chart",
    "summary_dashboard",
    # reports
    "generate_report",
    "generate_comparison_report",
]
