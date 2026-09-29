#!/usr/bin/env python3
"""Generate a local-first body-composition brief for Joe.

Inputs stay local: JSON/YAML arrays, JSON objects with a ``records`` key, or CSV.
The script prints Traditional Chinese markdown suitable for cron delivery, or exact
``[SILENT]`` when there is nothing actionable and suppression is requested.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

SILENT = "[SILENT]"
TARGET_BODY_FAT_MIN = 18.0
TARGET_BODY_FAT_MAX = 20.0


@dataclass(frozen=True)
class BodyRecord:
    date: date
    weight_kg: float | None = None
    body_fat_pct: float | None = None
    waist_cm: float | None = None
    note: str = ""


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if value is None:
        raise ValueError("record is missing date")
    return date.fromisoformat(str(value)[:10])


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _normalise_record(raw: dict[str, Any]) -> BodyRecord:
    return BodyRecord(
        date=_parse_date(raw.get("date") or raw.get("logged_at")),
        weight_kg=_to_float(raw.get("weight_kg") or raw.get("weight")),
        body_fat_pct=_to_float(raw.get("body_fat_pct") or raw.get("body_fat") or raw.get("bf_pct")),
        waist_cm=_to_float(raw.get("waist_cm") or raw.get("waist")),
        note=str(raw.get("note") or raw.get("notes") or ""),
    )


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
        raise RuntimeError("YAML input requires PyYAML; use JSON/CSV or install pyyaml") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_raw(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: str | Path) -> list[BodyRecord]:
    """Load body records from JSON/YAML/CSV and sort oldest to newest."""
    raw = _load_raw(Path(path))
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("records", [])
    if not isinstance(raw, list):
        raise ValueError("body-composition input must be a list or an object with records")
    return sorted((_normalise_record(item) for item in raw), key=lambda item: item.date)


def filter_records(records: Iterable[BodyRecord], days: int | None) -> list[BodyRecord]:
    ordered = sorted(records, key=lambda item: item.date)
    if not ordered or not days:
        return ordered
    cutoff = ordered[-1].date - timedelta(days=days - 1)
    return [record for record in ordered if record.date >= cutoff]


def _fmt_delta(value: float | None, unit: str) -> str:
    if value is None:
        return "資料不足"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}{unit}"


def _latest_with(records: Sequence[BodyRecord], attr: str) -> BodyRecord | None:
    for record in reversed(records):
        if getattr(record, attr) is not None:
            return record
    return None


def summarise(records: Sequence[BodyRecord]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: item.date)
    latest = ordered[-1] if ordered else None
    latest_weight = _latest_with(ordered, "weight_kg")
    latest_body_fat = _latest_with(ordered, "body_fat_pct")
    first_weight = next((item for item in ordered if item.weight_kg is not None), None)
    first_body_fat = next((item for item in ordered if item.body_fat_pct is not None), None)

    weight_delta = None
    if (
        first_weight
        and latest_weight
        and first_weight != latest_weight
        and first_weight.weight_kg is not None
        and latest_weight.weight_kg is not None
    ):
        weight_delta = round(latest_weight.weight_kg - first_weight.weight_kg, 1)

    body_fat_delta = None
    if (
        first_body_fat
        and latest_body_fat
        and first_body_fat != latest_body_fat
        and first_body_fat.body_fat_pct is not None
        and latest_body_fat.body_fat_pct is not None
    ):
        body_fat_delta = round(latest_body_fat.body_fat_pct - first_body_fat.body_fat_pct, 1)

    target_gap = None
    if latest_body_fat and latest_body_fat.body_fat_pct is not None:
        target_gap = round(max(0.0, latest_body_fat.body_fat_pct - TARGET_BODY_FAT_MAX), 1)

    return {
        "latest": latest,
        "latest_weight": latest_weight,
        "latest_body_fat": latest_body_fat,
        "weight_delta": weight_delta,
        "body_fat_delta": body_fat_delta,
        "target_gap": target_gap,
        "count": len(ordered),
    }


def render_brief(records: Sequence[BodyRecord], *, silent_if_empty: bool = False, days: int | None = None) -> str:
    scoped = filter_records(records, days)
    if not scoped:
        return SILENT if silent_if_empty else "## TL;DR\n- 沒有可用的身體組成紀錄。"

    summary = summarise(scoped)
    latest = summary["latest"]
    latest_weight = summary["latest_weight"]
    latest_body_fat = summary["latest_body_fat"]
    target_gap = summary["target_gap"]

    weight_text = "無體重資料" if latest_weight is None else f"{latest_weight.weight_kg:.1f}kg"
    body_fat_text = "無體脂資料" if latest_body_fat is None else f"{latest_body_fat.body_fat_pct:.1f}%"
    gap_text = "資料不足" if target_gap is None else f"距離 20% 還差 {target_gap:.1f} 個百分點"

    direction = "資料不足"
    if summary["body_fat_delta"] is not None:
        direction = "下降中" if summary["body_fat_delta"] < 0 else "上升中" if summary["body_fat_delta"] > 0 else "持平"

    lines = [
        "## TL;DR",
        f"- 最新紀錄（{latest.date.isoformat()}）：{weight_text}，體脂 {body_fat_text}；{gap_text}。",
        f"- 期間趨勢：體重 {_fmt_delta(summary['weight_delta'], 'kg')}，體脂 {_fmt_delta(summary['body_fat_delta'], 'pct')}（{direction}）。",
        "",
        "## Fact / verified",
        f"- 本次讀取 {summary['count']} 筆本地紀錄；未連接任何新資料源、未傳送訊息。",
        f"- Joe 的目標區間：體脂 {TARGET_BODY_FAT_MIN:.0f}–{TARGET_BODY_FAT_MAX:.0f}%。",
        "",
        "## Hypothesis",
        "- 如果體脂趨勢連續兩週沒有下降，問題更可能是飲食紀錄/週末熱量，而不是訓練不足。",
        "",
        "## Action for Joe",
        "- 今天只做一件可衡量的事：記錄體重、體脂/腰圍、蛋白質是否達標（是/否）。",
        "- 若已有 7 天以上資料：用週平均看趨勢，不要被單日水分波動騙走。",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Traditional Chinese body-composition brief.")
    parser.add_argument("--input", required=True, help="Path to JSON/YAML/CSV body-composition records")
    parser.add_argument("--days", type=int, default=14, help="Only consider the most recent N-day window")
    parser.add_argument("--silent-if-empty", action="store_true", help="Print exact [SILENT] when no records exist")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records = load_records(args.input)
    print(render_brief(records, silent_if_empty=args.silent_if_empty, days=args.days))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
