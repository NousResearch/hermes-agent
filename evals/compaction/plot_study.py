"""Render the study's aggregate.json as a publication-ready SVG (requires matplotlib)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def plot(aggregate, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    data = json.loads(aggregate.read_text())
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": "#334155",
        "axes.labelcolor": "#475569", "xtick.color": "#64748b", "ytick.color": "#64748b",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1", "svg.fonttype": "none", "svg.hashsalt": "hermes-study",
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.11, top=0.77, hspace=0.55, wspace=0.23)
    fig.suptitle("Summary + original evidence", x=0.08, y=0.97, ha="left", fontsize=20, weight="bold")
    fig.text(0.08, 0.91, "Frozen summaries · Jev Top 0–5 · exact-field scoring", color="#64748b", fontsize=11)
    handles = []
    for row, (cohort, title) in enumerate((
        ("primary", "Four primary tasks"), ("controls", "Ten short / query-only controls"),
    )):
        values = {r["arm"]: r for r in data[cohort]}
        for col, (metric, label) in enumerate((("facts", "Field accuracy"), ("input_tokens", "Answer input tokens"))):
            ax = axes[row, col]
            for kind, color, marker, name in (
                ("S", "#4f65d5", "o", "S · generic summary"),
                ("H", "#16867c", "s", "H · Hermes summary"),
            ):
                series = [values[kind if n == 0 else f"{kind}+top{n}"][metric] for n in range(6)]
                line, = ax.plot(range(6), series, color=color, marker=marker, markersize=5,
                                linewidth=2.2, label=name, zorder=3)
                if row == col == 0:
                    handles.append(line)
            baseline = ax.axhline(values["full"][metric], color="#94a3b8", linestyle=(0, (4, 3)),
                                  linewidth=1.4, label="Full history")
            if row == col == 0:
                handles.append(baseline)
            ax.set(ylabel=label, xlabel="Jev Top N", xlim=(-0.15, 5.15), xticks=range(6))
            ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=12)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e8edf3", linewidth=0.8)
            ax.tick_params(length=0, pad=7)
            if metric == "facts":
                ax.set(ylim=(0, 1.05), yticks=[0, 0.25, 0.5, 0.75, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            else:
                ax.set(ylim=(0, 4400), yticks=[0, 1000, 2000, 3000, 4000])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.875),
               ncol=3, frameon=False, handlelength=2.8, columnspacing=2)
    fig.text(0.08, 0.025, "Top 0 = summary alone. Inputs exclude summary/ranking preparation. "
             "Task means; repeated trials are not independent tasks.", fontsize=9, color="#64748b")
    fig.savefig(output, format="svg", metadata={
        "Date": None, "Title": "Summary and original-evidence study",
        "Description": "Source aggregate SHA-256: " + hashlib.sha256(aggregate.read_bytes()).hexdigest(),
    })
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.aggregate, args.out)


if __name__ == "__main__":
    main()
