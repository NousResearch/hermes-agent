#!/usr/bin/env python3
"""Render a validated static chart and emit a Hermes MEDIA directive.

The renderer deliberately accepts a small JSON chart specification instead of
arbitrary Python. This keeps generated charts predictable and makes it easy to
validate data before a messaging gateway uploads the resulting PNG.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_CHART_TYPES = frozenset({"line", "bar", "scatter", "histogram", "pie"})
MAX_SERIES = 8
MAX_POINTS_PER_SERIES = 500
MAX_PIE_SLICES = 12
MAX_TITLE_LENGTH = 160
MAX_LABEL_LENGTH = 100
MAX_SOURCE_LENGTH = 240


class ChartSpecError(ValueError):
    """Raised when a chart specification is incomplete or unsafe to render."""


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChartSpecError(f"{field} must be an object")
    return value


def _require_list(value: Any, field: str, *, minimum: int = 1, maximum: int = MAX_POINTS_PER_SERIES) -> list[Any]:
    if not isinstance(value, list):
        raise ChartSpecError(f"{field} must be an array")
    if not minimum <= len(value) <= maximum:
        raise ChartSpecError(f"{field} must contain between {minimum} and {maximum} values")
    return value


def _require_text(value: Any, field: str, *, maximum: int = MAX_LABEL_LENGTH, required: bool = True) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise ChartSpecError(f"{field} must be text")
    value = value.strip()
    if required and not value:
        raise ChartSpecError(f"{field} cannot be empty")
    if len(value) > maximum:
        raise ChartSpecError(f"{field} must be at most {maximum} characters")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChartSpecError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ChartSpecError(f"{field} must be a finite number")
    return number


def _validate_xy_series(raw_series: Any, chart_type: str) -> list[dict[str, Any]]:
    series_items = _require_list(raw_series, "series", maximum=MAX_SERIES)
    normalized: list[dict[str, Any]] = []
    first_x: list[Any] | None = None

    for index, item in enumerate(series_items, start=1):
        series = _require_mapping(item, f"series[{index}]")
        name = _require_text(series.get("name", f"Series {index}"), f"series[{index}].name")
        x_values = _require_list(series.get("x"), f"series[{index}].x")
        y_values = _require_list(series.get("y"), f"series[{index}].y")
        if len(x_values) != len(y_values):
            raise ChartSpecError(f"series[{index}].x and series[{index}].y must have the same length")

        if chart_type == "scatter":
            x_normalized = [_require_number(value, f"series[{index}].x[{point_index}]")
                            for point_index, value in enumerate(x_values, start=1)]
        else:
            x_normalized = []
            for point_index, value in enumerate(x_values, start=1):
                if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                    raise ChartSpecError(
                        f"series[{index}].x[{point_index}] must be text or a finite number"
                    )
                if isinstance(value, float) and not math.isfinite(value):
                    raise ChartSpecError(f"series[{index}].x[{point_index}] must be finite")
                x_normalized.append(value)

        y_normalized = [_require_number(value, f"series[{index}].y[{point_index}]")
                        for point_index, value in enumerate(y_values, start=1)]
        if chart_type == "bar":
            if first_x is None:
                first_x = x_normalized
            elif x_normalized != first_x:
                raise ChartSpecError("all bar series must use the same x labels")
        normalized.append({"name": name, "x": x_normalized, "y": y_normalized})

    return normalized


def _validate_histogram_series(raw_series: Any) -> list[dict[str, Any]]:
    series_items = _require_list(raw_series, "series", maximum=MAX_SERIES)
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(series_items, start=1):
        series = _require_mapping(item, f"series[{index}]")
        name = _require_text(series.get("name", f"Series {index}"), f"series[{index}].name")
        values = _require_list(series.get("values"), f"series[{index}].values")
        normalized.append(
            {
                "name": name,
                "values": [
                    _require_number(value, f"series[{index}].values[{point_index}]")
                    for point_index, value in enumerate(values, start=1)
                ],
            }
        )
    return normalized


def _validate_pie_series(raw_series: Any) -> list[dict[str, Any]]:
    series_items = _require_list(raw_series, "series", maximum=1)
    series = _require_mapping(series_items[0], "series[1]")
    name = _require_text(series.get("name", "Breakdown"), "series[1].name")
    labels = _require_list(series.get("labels"), "series[1].labels", maximum=MAX_PIE_SLICES)
    values = _require_list(series.get("values"), "series[1].values", maximum=MAX_PIE_SLICES)
    if len(labels) != len(values):
        raise ChartSpecError("series[1].labels and series[1].values must have the same length")

    normalized_labels = [
        _require_text(value, f"series[1].labels[{index}]")
        for index, value in enumerate(labels, start=1)
    ]
    normalized_values = [
        _require_number(value, f"series[1].values[{index}]")
        for index, value in enumerate(values, start=1)
    ]
    if any(value < 0 for value in normalized_values) or not any(normalized_values):
        raise ChartSpecError("pie values must be non-negative and add up to more than zero")
    return [{"name": name, "labels": normalized_labels, "values": normalized_values}]


def validate_spec(raw_spec: Any) -> dict[str, Any]:
    """Normalize and validate the JSON chart schema before importing plotting libraries."""
    spec = _require_mapping(raw_spec, "chart spec")
    chart_type = _require_text(spec.get("type"), "type").lower()
    if chart_type not in SUPPORTED_CHART_TYPES:
        choices = ", ".join(sorted(SUPPORTED_CHART_TYPES))
        raise ChartSpecError(f"type must be one of: {choices}")

    title = _require_text(spec.get("title"), "title", maximum=MAX_TITLE_LENGTH)
    x_label = _require_text(spec.get("x_label", ""), "x_label", required=False)
    y_label = _require_text(spec.get("y_label", ""), "y_label", required=False)
    source = _require_text(
        spec.get("source", ""), "source", maximum=MAX_SOURCE_LENGTH, required=False
    )

    if chart_type == "histogram":
        series = _validate_histogram_series(spec.get("series"))
    elif chart_type == "pie":
        series = _validate_pie_series(spec.get("series"))
    else:
        series = _validate_xy_series(spec.get("series"), chart_type)

    bins = spec.get("bins")
    if chart_type == "histogram" and bins is not None:
        if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 100:
            raise ChartSpecError("bins must be an integer between 2 and 100")
    elif bins is not None:
        raise ChartSpecError("bins is only supported for histogram charts")

    return {
        "type": chart_type,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "source": source,
        "series": series,
        "bins": bins,
    }


def _output_root() -> Path:
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if not hermes_home:
        raise ChartSpecError("HERMES_HOME is required so charts can be delivered safely")
    return (Path(hermes_home).expanduser().resolve() / "cache" / "images")


def resolve_output_path(raw_output: str | None) -> Path:
    """Return an absolute PNG path contained in Hermes' image cache."""
    root = _output_root()
    if raw_output:
        output = Path(raw_output).expanduser().resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = root / f"chart-{timestamp}-{secrets.token_hex(3)}.png"

    if output.suffix.lower() != ".png":
        raise ChartSpecError("output must use the .png extension")
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ChartSpecError(f"output must be inside {root}") from exc
    return output


def load_spec(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ChartSpecError(f"specification file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ChartSpecError("specification file must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ChartSpecError(f"invalid JSON: {exc.msg}") from exc
    return validate_spec(raw)


def _load_plotting_libraries():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
        import japanize_matplotlib
        import seaborn
    except ImportError as exc:
        raise ChartSpecError(
            "missing chart dependencies; install the skill requirements into the renderer Python environment"
        ) from exc
    return matplotlib, pyplot, seaborn, japanize_matplotlib


def _configure_japanese_font(matplotlib: Any, japanize_matplotlib: Any) -> str:
    """Apply japanize-matplotlib after any Seaborn theme resets font settings."""
    japanize_matplotlib.japanize()
    matplotlib.rcParams["axes.unicode_minus"] = False
    return str(matplotlib.rcParams["font.family"][0])


def _render_line(axis: Any, series: list[dict[str, Any]], colors: list[Any]) -> None:
    for item, color in zip(series, colors):
        axis.plot(item["x"], item["y"], marker="o", linewidth=2, label=item["name"], color=color)


def _render_bar(axis: Any, series: list[dict[str, Any]], colors: list[Any]) -> None:
    labels = series[0]["x"]
    positions = list(range(len(labels)))
    width = 0.8 / len(series)
    for index, (item, color) in enumerate(zip(series, colors)):
        offset = (index - (len(series) - 1) / 2) * width
        axis.bar([position + offset for position in positions], item["y"], width, label=item["name"], color=color)
    axis.set_xticks(positions, labels)


def _render_scatter(axis: Any, series: list[dict[str, Any]], colors: list[Any]) -> None:
    for item, color in zip(series, colors):
        axis.scatter(item["x"], item["y"], s=55, label=item["name"], color=color)


def _render_histogram(axis: Any, series: list[dict[str, Any]], colors: list[Any], bins: int | None) -> None:
    values = [item["values"] for item in series]
    names = [item["name"] for item in series]
    axis.hist(
        values,
        bins=bins or 20,
        label=names,
        color=colors[:len(values)],
        alpha=0.72,
        edgecolor="white",
    )


def _render_pie(axis: Any, series: list[dict[str, Any]], colors: list[Any]) -> None:
    item = series[0]
    axis.pie(
        item["values"],
        labels=item["labels"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    axis.axis("equal")


def render_chart(spec: dict[str, Any], output_path: Path) -> str:
    """Render one validated chart and return the active Japanese font name."""
    matplotlib, pyplot, seaborn, japanize_matplotlib = _load_plotting_libraries()
    seaborn.set_theme(style="whitegrid", context="notebook")
    # Seaborn's theme sets a default sans-serif family, so reapply the bundled
    # IPAexGothic font afterwards rather than allowing it to reset to Arial.
    selected_font = _configure_japanese_font(matplotlib, japanize_matplotlib)
    figure, axis = pyplot.subplots(figsize=(10, 6), layout="constrained")
    colors = seaborn.color_palette("colorblind", n_colors=max(3, len(spec["series"])))

    try:
        chart_type = spec["type"]
        if chart_type == "line":
            _render_line(axis, spec["series"], colors)
        elif chart_type == "bar":
            _render_bar(axis, spec["series"], colors)
        elif chart_type == "scatter":
            _render_scatter(axis, spec["series"], colors)
        elif chart_type == "histogram":
            _render_histogram(axis, spec["series"], colors, spec["bins"])
        else:
            _render_pie(axis, spec["series"], colors)

        axis.set_title(spec["title"], loc="left", fontsize=16, pad=16)
        if chart_type != "pie":
            axis.set_xlabel(spec["x_label"])
            axis.set_ylabel(spec["y_label"])
            if len(spec["series"]) > 1:
                axis.legend(frameon=False)
            axis.tick_params(axis="x", rotation=25 if chart_type == "bar" else 0)
        if spec["source"]:
            figure.text(0.01, 0.01, f"Source: {spec['source']}", ha="left", va="bottom", fontsize=8, color="#555555")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="png", dpi=180, bbox_inches="tight")
    finally:
        pyplot.close(figure)
    return selected_font


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a validated static chart for Hermes")
    parser.add_argument("spec", type=Path, help="UTF-8 JSON chart specification")
    parser.add_argument("--output", help="PNG path under $HERMES_HOME/cache/images")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec = load_spec(args.spec)
        output_path = resolve_output_path(args.output)
        selected_font = render_chart(spec, output_path)
    except (ChartSpecError, OSError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2

    payload = {
        "success": True,
        "file_path": str(output_path),
        "media": f"MEDIA:{output_path}",
    }
    if selected_font:
        payload["font"] = selected_font
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
