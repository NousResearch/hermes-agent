"""
Render publication-style charts with matplotlib and seaborn into a PNG asset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .models import VizSpecification


def render_chart_png(spec: VizSpecification, dest: Path) -> Path:
    """
    Write a single high-DPI PNG under `dest` (full file path).

    Stateless: mutates only the filesystem at `dest`.
    """
    sns.set_theme(style=spec.seaborn_style, palette=spec.palette)
    fig, ax = plt.subplots(figsize=spec.figure_size, dpi=spec.dpi)
    _draw(spec, ax)
    fig.suptitle(spec.title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return dest


def _df_from_spec(spec: VizSpecification) -> pd.DataFrame | None:
    rec = spec.data.records
    if not rec:
        return None
    return pd.DataFrame(rec)


def _draw(spec: VizSpecification, ax: plt.Axes) -> None:
    df = _df_from_spec(spec)
    kind = spec.chart_type

    if kind == "heatmap":
        mat = spec.data.matrix
        if mat is None:
            raise ValueError("heatmap requires matrix data")
        sns.heatmap(mat, ax=ax, cmap="viridis", cbar=True)
        ax.set_xlabel(spec.x_label)
        ax.set_ylabel(spec.y_label)
        return

    if kind in ("line", "scatter", "regression", "distribution", "bar") and df is not None:
        if kind == "line":
            sns.lineplot(
                data=df,
                x=spec.x_field,
                y=spec.y_field,
                hue=spec.hue_column,
                ax=ax,
                marker="o",
            )
        elif kind == "scatter":
            sns.scatterplot(
                data=df, x=spec.x_field, y=spec.y_field, hue=spec.hue_column, ax=ax, s=60
            )
        elif kind == "regression":
            sns.regplot(data=df, x=spec.x_field, y=spec.y_field, ax=ax, scatter_kws={"s": 50})
        elif kind == "distribution":
            sns.kdeplot(data=df, y=spec.y_field, ax=ax, fill=True, cut=0)
        else:
            sns.barplot(
                data=df,
                x=spec.category_field,
                y=spec.value_field,
                hue=spec.hue_column,
                ax=ax,
            )
        ax.set_xlabel(spec.x_label or spec.x_field)
        ax.set_ylabel(spec.y_label or spec.y_field)
        return

    xs, ys = spec.data.x, spec.data.y
    if xs is None or ys is None:
        raise ValueError(f"{kind} requires records or x/y arrays")
    if kind == "line":
        ax.plot(xs, ys, marker="o")
    elif kind == "scatter":
        ax.scatter(xs, ys)
    elif kind == "regression":
        sns.regplot(x=xs, y=ys, ax=ax)
    elif kind == "distribution":
        sns.kdeplot(y=ys, ax=ax, fill=True, cut=0)
    elif kind == "bar":
        ax.bar([str(i) for i in range(len(ys))], ys)
    else:
        raise ValueError(f"Unsupported chart_type for array mode: {kind}")
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(spec.y_label)
