#!/usr/bin/env python3
"""Generate a local-first investment portfolio review brief.

Input files may be JSON, YAML, or CSV. Expected fields per position:
name, value, cost_basis, category, thesis, last_reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

SILENT_MARKER = "[SILENT]"


@dataclass(frozen=True)
class Position:
    name: str
    value: float
    cost_basis: float | None = None
    category: str = "uncategorized"
    thesis: str = ""
    last_reviewed: date | None = None


@dataclass(frozen=True)
class ReviewAnalysis:
    as_of: date
    total_value: float
    unrealized_pnl: float | None
    positions: list[dict[str, Any]]
    concentration_flags: list[dict[str, Any]]
    missing_thesis: list[dict[str, Any]]
    stale_reviews: list[dict[str, Any]]

    @property
    def needs_attention(self) -> bool:
        return bool(self.concentration_flags or self.missing_thesis or self.stale_reviews)


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD")


def _parse_float(value: Any, field: str, name: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"Position {name!r} has invalid {field}: {value!r}") from exc


def _coerce_position(raw: dict[str, Any], index: int) -> Position:
    name = str(raw.get("name") or raw.get("asset") or "").strip()
    if not name:
        raise ValueError(f"Position #{index} is missing required field: name")

    value = _parse_float(raw.get("value"), "value", name)
    if value is None:
        raise ValueError(f"Position {name!r} is missing required field: value")
    if value < 0:
        raise ValueError(f"Position {name!r} has negative value: {value}")

    return Position(
        name=name,
        value=value,
        cost_basis=_parse_float(raw.get("cost_basis"), "cost_basis", name),
        category=str(raw.get("category") or "uncategorized").strip() or "uncategorized",
        thesis=str(raw.get("thesis") or "").strip(),
        last_reviewed=_parse_date(raw.get("last_reviewed") or raw.get("reviewed_at")),
    )


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ValueError("YAML input requires PyYAML; use JSON/CSV or install pyyaml") from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_positions(path: Path) -> list[Position]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml(path)
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            data = list(csv.DictReader(fh))
    else:
        raise ValueError("Unsupported input format; use .json, .yaml, .yml, or .csv")

    if isinstance(data, dict):
        data = data.get("positions")
    if not isinstance(data, list) or not data:
        raise ValueError("No investment positions found; expected a non-empty list")

    return [_coerce_position(dict(item), index) for index, item in enumerate(data, start=1)]


def analyze_positions(
    positions: list[Position],
    *,
    as_of: date,
    review_after_days: int = 90,
    concentration_threshold: float = 0.5,
) -> ReviewAnalysis:
    total = sum(position.value for position in positions)
    if total <= 0:
        raise ValueError("Total portfolio value must be greater than zero")

    total_cost = sum(position.cost_basis for position in positions if position.cost_basis is not None)
    has_cost_basis = any(position.cost_basis is not None for position in positions)
    rendered_positions: list[dict[str, Any]] = []
    concentration_flags: list[dict[str, Any]] = []
    missing_thesis: list[dict[str, Any]] = []
    stale_reviews: list[dict[str, Any]] = []

    for position in sorted(positions, key=lambda item: item.value, reverse=True):
        weight = position.value / total
        pnl = None if position.cost_basis is None else position.value - position.cost_basis
        row = {
            "name": position.name,
            "category": position.category,
            "value": position.value,
            "weight": weight,
            "pnl": pnl,
            "thesis": position.thesis,
            "last_reviewed": position.last_reviewed,
        }
        rendered_positions.append(row)

        if weight > concentration_threshold:
            concentration_flags.append(row)
        if not position.thesis:
            missing_thesis.append(row)
        if position.last_reviewed is None:
            stale_reviews.append({**row, "days_since_review": None})
        else:
            days_since_review = (as_of - position.last_reviewed).days
            if days_since_review > review_after_days:
                stale_reviews.append({**row, "days_since_review": days_since_review})

    return ReviewAnalysis(
        as_of=as_of,
        total_value=total,
        unrealized_pnl=(total - total_cost) if has_cost_basis else None,
        positions=rendered_positions,
        concentration_flags=concentration_flags,
        missing_thesis=missing_thesis,
        stale_reviews=stale_reviews,
    )


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_brief(analysis: ReviewAnalysis, *, silent_if_clear: bool = False) -> str:
    if silent_if_clear and not analysis.needs_attention:
        return SILENT_MARKER

    lines = [
        "# 投資組合 Review Brief",
        "",
        "## TL;DR",
        f"- Fact / verified：總資產：{_money(analysis.total_value)}；部位數：{len(analysis.positions)}。",
    ]
    if analysis.unrealized_pnl is not None:
        lines.append(f"- Fact / verified：未實現損益：{_money(analysis.unrealized_pnl)}。")
    lines.append(
        f"- Recommendation：優先處理 {len(analysis.concentration_flags)} 個集中風險、"
        f"{len(analysis.missing_thesis)} 個缺少 thesis、{len(analysis.stale_reviews)} 個過期 review。"
    )

    lines.extend(["", "## 部位概覽"])
    for row in analysis.positions:
        pnl_text = "n/a" if row["pnl"] is None else _money(row["pnl"])
        lines.append(
            f"- {row['name']} — {_pct(row['weight'])} / {_money(row['value'])} "
            f"/ {row['category']} / PnL {pnl_text}"
        )

    lines.extend(["", "## 需要 Joe review 的事項"])
    if not analysis.needs_attention:
        lines.append("- Fact / verified：目前沒有超過門檻的集中風險、缺少 thesis 或過期 review。")
    for row in analysis.concentration_flags:
        lines.append(f"- 集中風險：{row['name']} — {_pct(row['weight'])}。")
    for row in analysis.missing_thesis:
        lines.append(f"- 缺少 thesis：{row['name']}。")
    for row in analysis.stale_reviews:
        days = row.get("days_since_review")
        if days is None:
            lines.append(f"- 過期 review：{row['name']}（沒有 last_reviewed）。")
        else:
            lines.append(f"- 過期 review：{row['name']}（{days} 天未 review）。")

    lines.extend(
        [
            "",
            "## 建議下一步",
            "- 補齊缺少 thesis 的部位：一句話 thesis、失效條件、下一次 review 日期。",
            "- 對集中部位設定可量化 action：加倉、減倉、hedge 或維持，但要寫明條件。",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a local investment review brief.")
    parser.add_argument("path", type=Path, help="Portfolio file (.json, .yaml, .yml, .csv)")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="Review date (YYYY-MM-DD)")
    parser.add_argument("--review-after-days", type=int, default=90)
    parser.add_argument("--concentration-threshold", type=float, default=0.5)
    parser.add_argument("--silent-if-clear", action="store_true", help="Print exact [SILENT] when no flags exist")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        positions = load_positions(args.path)
        analysis = analyze_positions(
            positions,
            as_of=_parse_date(args.as_of) or date.today(),
            review_after_days=args.review_after_days,
            concentration_threshold=args.concentration_threshold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(render_brief(analysis, silent_if_clear=args.silent_if_clear))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
