#!/usr/bin/env python3
"""Parse Hermes observability weekly reports and label trend comparability.

This helper is intentionally dependency-free so cron/report agents can run it from
any Hermes checkout before writing a weekly report. It reads Obsidian markdown
reports with YAML-like frontmatter, finds the latest prior report in the same
folder, and emits deterministic delta/query-note text for the next report.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_GLOB = "*Hermes Observability Weekly.md"
WINDOW_TOLERANCE_SECONDS = 5 * 60
CANONICAL_BOARD_ID = "bb4u4ZSG6zm"
CANONICAL_BOARD_URL = "https://ui.honeycomb.io/hermes-32/environments/prod/board/bb4u4ZSG6zm"
CANONICAL_QUERY_SET = [
    {
        "purpose": "timed_out_agent_turns",
        "query_id": "cD18ADe6uZD",
        "run_pk": "BwaKeEDHL6e",
        "url": "https://ui.honeycomb.io/hermes-32/environments/prod/datasets/hermes-agent/result/BwaKeEDHL6e",
        "spec": {
            "breakdowns": ["session_id", "hermes.profile", "hermes.cron.job_id", "hermes.session.kind"],
            "calculations": [
                {"name": "timed_out_turns", "op": "COUNT"},
                {"column": "duration_ms", "name": "p95_duration_ms", "op": "P95"},
                {"column": "duration_ms", "name": "max_duration_ms", "op": "MAX"},
            ],
            "filters": [
                {"column": "name", "op": "=", "value": "agent"},
                {"column": "hermes.turn.final_status", "op": "=", "value": "timed_out"},
            ],
            "breakdown_limit": 100,
            "order": "timed_out_turns descending",
        },
    },
    {
        "purpose": "agent_final_status_distribution",
        "query_id": "p59L5Wm7BKU",
        "run_pk": "hDhknpdiYsb",
        "url": "https://ui.honeycomb.io/hermes-32/environments/prod/datasets/hermes-agent/result/hDhknpdiYsb",
        "spec": {
            "breakdowns": ["hermes.turn.final_status", "hermes.session.completed", "hermes.session.interrupted"],
            "calculations": [
                {"name": "turns", "op": "COUNT"},
                {"column": "duration_ms", "name": "p95_duration_ms", "op": "P95"},
                {"column": "duration_ms", "name": "max_duration_ms", "op": "MAX"},
            ],
            "filters": [{"column": "name", "op": "=", "value": "agent"}],
            "breakdown_limit": 20,
            "order": "turns descending",
        },
    },
    {
        "purpose": "api_token_spend_by_session",
        "query_id": "jwwNTcpUUrs",
        "run_pk": "D7BK65D6V4w",
        "url": "https://ui.honeycomb.io/hermes-32/environments/prod/datasets/hermes-agent/result/D7BK65D6V4w",
        "spec": {
            "breakdowns": ["session_id", "hermes.profile"],
            "calculations": [
                {"name": "api_calls", "op": "COUNT"},
                {"column": "gen_ai.usage.total_tokens", "name": "total_tokens", "op": "SUM"},
                {"column": "duration_ms", "name": "max_api_duration_ms", "op": "MAX"},
            ],
            "filters": [
                {"column": "name", "op": "starts-with", "value": "api."},
                {"column": "session_id", "op": "exists"},
            ],
            "breakdown_limit": 20,
            "order": "total_tokens descending",
        },
    },
    {
        "purpose": "cache_read_ratio_absolute_prompt_tokens",
        "query_id": "fxDceAHGyM6",
        "run_pk": "ozu15hfPbxN",
        "url": "https://ui.honeycomb.io/hermes-32/environments/prod/datasets/hermes-agent/result/ozu15hfPbxN",
        "spec": {
            "calculations": [
                {"column": "gen_ai.usage.cache_read.input_tokens", "name": "cache_read_tokens", "op": "SUM"},
                {"column": "gen_ai.usage.input_tokens", "name": "prompt_tokens", "op": "SUM"},
                {"column": "gen_ai.usage.output_tokens", "name": "completion_tokens", "op": "SUM"},
                {"name": "count_token_spans", "op": "COUNT"},
            ],
            "formulas": [
                {"expression": "$cache_read_tokens / $prompt_tokens * 100", "name": "cache_read_ratio_pct"},
                {"expression": "$prompt_tokens", "name": "prompt_tokens_abs"},
                {"expression": "$cache_read_tokens", "name": "cache_read_tokens_abs"},
                {"expression": "$completion_tokens", "name": "completion_tokens_abs"},
                {"expression": "$count_token_spans", "name": "token_span_count"},
            ],
            "filters": [{"column": "gen_ai.usage.input_tokens", "op": "exists"}],
        },
    },
    {
        "purpose": "max_api_duration_by_session_model",
        "query_id": "uwGjV1nKRyK",
        "run_pk": "jmzgzDGzw5h",
        "url": "https://ui.honeycomb.io/hermes-32/environments/prod/datasets/hermes-agent/result/jmzgzDGzw5h",
        "spec": {
            "breakdowns": ["session_id", "hermes.profile", "name"],
            "calculations": [
                {"name": "api_calls", "op": "COUNT"},
                {"column": "duration_ms", "name": "max_api_duration_ms", "op": "MAX"},
                {"column": "duration_ms", "name": "p95_api_duration_ms", "op": "P95"},
            ],
            "filters": [
                {"column": "name", "op": "starts-with", "value": "api."},
                {"column": "session_id", "op": "exists"},
            ],
            "breakdown_limit": 20,
            "order": "max_api_duration_ms descending",
        },
    },
]

METRIC_FIELDS = (
    "trace_count",
    "agent_turn_count",
    "api_call_count",
    "tool_call_count",
    "error_ratio",
    "tool_error_ratio",
    "p95_agent_latency_ms",
    "p95_api_latency_ms",
    "p95_tool_latency_ms",
    "cache_read_ratio",
)
RATIO_FIELDS = {"error_ratio", "tool_error_ratio", "cache_read_ratio"}


@dataclass(frozen=True)
class ReportMetrics:
    path: Path
    frontmatter: dict[str, Any]

    @property
    def start(self) -> datetime | None:
        return parse_datetime(self.frontmatter.get("period_analyzed_start"))

    @property
    def end(self) -> datetime | None:
        return parse_datetime(self.frontmatter.get("period_analyzed_end"))

    @property
    def duration_seconds(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return (self.end - self.start).total_seconds()

    def metric(self, name: str) -> float | int | None:
        value = self.frontmatter.get(name)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("markdown report is missing YAML frontmatter opener")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration as exc:
        raise ValueError("markdown report is missing YAML frontmatter closer") from exc
    return parse_simple_yaml(lines[1:end]), "\n".join(lines[end + 1 :])


def parse_simple_yaml(lines: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        data[key.strip()] = parse_scalar(raw.strip())
    return data


def parse_scalar(raw: str) -> Any:
    if raw in {"", "null", "~"}:
        return None
    if raw in {"true", "false"}:
        return raw == "true"
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    if re.fullmatch(r"[-+]?\d+", raw):
        try:
            return int(raw)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", raw):
        try:
            return float(raw)
        except ValueError:
            pass
    return raw


def load_report(path: Path) -> ReportMetrics:
    frontmatter, _body = split_frontmatter(path.read_text(encoding="utf-8"))
    return ReportMetrics(path=path, frontmatter=frontmatter)


def find_prior_report(current_path: Path, directory: Path | None = None) -> Path | None:
    current = load_report(current_path)
    root = directory or current_path.parent
    candidates: list[ReportMetrics] = []
    for path in root.glob(REPORT_GLOB):
        if path.resolve() == current_path.resolve():
            continue
        try:
            candidate = load_report(path)
        except ValueError:
            continue
        if candidate.end is None or current.end is None:
            continue
        if candidate.end <= current.end:
            candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda report: (report.end or datetime.min.replace(tzinfo=timezone.utc), report.path.name)).path


def comparability(current: ReportMetrics, prior: ReportMetrics | None) -> tuple[str, str]:
    if prior is None:
        return "incomparable", "No prior report was found; this report is the baseline."
    shared_keys = ("team", "environment", "dataset")
    mismatches = [key for key in shared_keys if current.frontmatter.get(key) != prior.frontmatter.get(key)]
    if mismatches:
        return "incomparable", "Different " + ", ".join(mismatches) + "; do not compute deltas."
    if current.start is None or current.end is None or prior.start is None or prior.end is None:
        return "incomparable", "One or both reports lack fixed analysis timestamps."
    if current.duration_seconds is None or prior.duration_seconds is None:
        return "incomparable", "One or both reports lack a complete analysis window."
    if abs(current.duration_seconds - prior.duration_seconds) > WINDOW_TOLERANCE_SECONDS:
        return "incomparable", (
            f"Analysis window durations differ: current {format_duration(current.duration_seconds)}, "
            f"prior {format_duration(prior.duration_seconds)}."
        )
    overlap_seconds = min(current.end, prior.end) - max(current.start, prior.start)
    if overlap_seconds.total_seconds() > WINDOW_TOLERANCE_SECONDS:
        return "overlapping", (
            "Windows have the same duration but overlap by "
            f"{format_duration(overlap_seconds.total_seconds())}; deltas are directional, not independent week-over-week evidence."
        )
    return "independent", "Windows have matching duration and do not overlap; deltas can be treated as independent trend evidence."


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and not days:
        parts.append(f"{minutes}m")
    return " ".join(parts) or "0m"


def metric_deltas(current: ReportMetrics, prior: ReportMetrics | None) -> dict[str, dict[str, float | int | None]]:
    if prior is None:
        return {}
    deltas: dict[str, dict[str, float | int | None]] = {}
    for field in METRIC_FIELDS:
        current_value = current.metric(field)
        prior_value = prior.metric(field)
        if current_value is None or prior_value is None:
            continue
        row: dict[str, float | int | None] = {
            "current": current_value,
            "prior": prior_value,
            "delta": current_value - prior_value,
        }
        if field in RATIO_FIELDS:
            row["delta_percentage_points"] = (current_value - prior_value) * 100
        elif prior_value != 0:
            row["delta_percent"] = ((current_value - prior_value) / prior_value) * 100
        else:
            row["delta_percent"] = None
        deltas[field] = row
    return deltas


def format_metric_deltas(deltas: dict[str, dict[str, float | int | None]]) -> list[str]:
    lines: list[str] = []
    for field in METRIC_FIELDS:
        row = deltas.get(field)
        if not row:
            continue
        if field in RATIO_FIELDS:
            lines.append(
                f"- `{field}`: {row['prior']:.4g} → {row['current']:.4g} "
                f"({row['delta_percentage_points']:+.2f} percentage points)"
            )
        else:
            delta_percent = row.get("delta_percent")
            pct = "n/a" if delta_percent is None else f"{delta_percent:+.1f}%"
            lines.append(f"- `{field}`: {row['prior']:.4g} → {row['current']:.4g} ({pct})")
    return lines


def render_query_notes(current: ReportMetrics, prior: ReportMetrics | None) -> str:
    label, reason = comparability(current, prior)
    deltas = metric_deltas(current, prior)
    lines = [
        "- Trend comparability label: "
        f"`{label}`. {reason}",
    ]
    if prior is not None:
        lines.append(f"- Prior report used for mechanical deltas: `{prior.path}`.")
    if deltas:
        lines.append("- Frontmatter metric deltas computed by `scripts/observability_report_checker.py`:")
        lines.extend(format_metric_deltas(deltas))
    lines.extend(
        [
            f"- Runtime-health board: `{CANONICAL_BOARD_ID}` ({CANONICAL_BOARD_URL}).",
            "- Canonical fixed-window query set to preserve stable trend shapes:",
        ]
    )
    window = {
        "start": current.frontmatter.get("period_analyzed_start"),
        "end": current.frontmatter.get("period_analyzed_end"),
    }
    for query in CANONICAL_QUERY_SET:
        spec = dict(query["spec"])
        spec["fixed_window"] = window
        lines.append(
            f"  - `{query['purpose']}`: board query `{query['query_id']}`, run `{query['run_pk']}`, "
            f"{query['url']}; spec `{json.dumps(spec, sort_keys=True, separators=(',', ':'))}`"
        )
    return "\n".join(lines)


def build_result(current_path: Path, prior_path: Path | None = None) -> dict[str, Any]:
    current = load_report(current_path)
    prior = load_report(prior_path) if prior_path else None
    if prior is None:
        found = find_prior_report(current_path)
        prior = load_report(found) if found else None
    label, reason = comparability(current, prior)
    return {
        "current_report": str(current.path),
        "prior_report": str(prior.path) if prior else None,
        "comparability": label,
        "comparability_reason": reason,
        "deltas": metric_deltas(current, prior),
        "query_notes": render_query_notes(current, prior),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Current Hermes Observability Weekly markdown report")
    parser.add_argument("--prior", type=Path, help="Explicit prior report path; otherwise auto-detect latest prior")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown query notes")
    args = parser.parse_args()

    result = build_result(args.report, args.prior)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result["query_notes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
