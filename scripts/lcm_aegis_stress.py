#!/usr/bin/env python3
"""Deterministic PRD-6 Aegis LCM stress gate.

The default path is a dry-run simulator: it exercises scoring, persistence,
provider-down degradation, recall, long tool-output rows, and restart-like
store reopens without mutating live Hermes profiles.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DEFAULT_TURNS = 80
DEFAULT_COMPACTION_EVERY = 4
DEFAULT_PROVIDER_DOWN_COMPACTIONS = frozenset({5})
DEFAULT_RESTART_TURNS = frozenset({33, 66})
DEFAULT_DEGRADED_RATE_MAX = 0.05
DEFAULT_CONSECUTIVE_DEGRADED_MAX = 3
DEFAULT_INPUT_USD_PER_MTOK = 5.0
DEFAULT_OUTPUT_USD_PER_MTOK = 15.0


@dataclass(frozen=True)
class StressThresholds:
    max_consecutive_degraded: int = DEFAULT_CONSECUTIVE_DEGRADED_MAX
    max_degraded_rate: float = DEFAULT_DEGRADED_RATE_MAX


@dataclass(frozen=True)
class SpendPolicy:
    input_usd_per_mtok: float = DEFAULT_INPUT_USD_PER_MTOK
    output_usd_per_mtok: float = DEFAULT_OUTPUT_USD_PER_MTOK

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1_000_000.0 * self.input_usd_per_mtok) + (
            completion_tokens / 1_000_000.0 * self.output_usd_per_mtok
        )


@dataclass(frozen=True)
class StressRecord:
    turn: int
    shapes: tuple[str, ...]
    compaction_index: int | None
    degraded: bool
    fail_closed: bool
    recall_ok: bool | None
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    observed_prompt_tokens: int
    observed_completion_tokens: int
    note: str


@dataclass(frozen=True)
class StressRun:
    dry_run: bool
    turns: int
    compaction_every: int
    compaction_count: int
    degraded_count: int
    degraded_rate: float
    max_consecutive_degraded: int
    fail_closed_count: int
    store_persistence: bool
    restart_count: int
    coverage_counts: dict[str, int]
    estimated_spend_usd: float
    observed_spend_usd: float
    status: str
    failures: list[str]
    records: list[StressRecord]
    out_path: Path | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


class SimulatedStore:
    """Tiny append-only store used to prove dry-run restart persistence."""

    def __init__(self, rows: dict[str, str] | None = None) -> None:
        self.rows = dict(rows or {})

    def ingest(self, key: str, value: str) -> None:
        self.rows[key] = value

    def recall(self, key: str) -> str | None:
        return self.rows.get(key)

    def snapshot(self) -> str:
        return json.dumps(self.rows, sort_keys=True)

    @classmethod
    def restore(cls, payload: str) -> "SimulatedStore":
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("store snapshot must be a JSON object")
        return cls({str(k): str(v) for k, v in parsed.items()})


def estimate_text_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _cost_policy() -> SpendPolicy:
    return SpendPolicy()


def _parse_int_set(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    values: set[int] = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.add(int(item))
    return values


def _turn_text(turn: int, shapes: Iterable[str]) -> str:
    base = f"turn={turn:03d}; ordinary Aegis QA conversation about LCM cutover safety."
    if "tool_call_like_row" in shapes:
        base += " tool_call={\"name\":\"lcm_status\",\"arguments\":{\"scope\":\"stress\"}}"
    if "long_tool_output" in shapes:
        base += " long_tool_output=" + ("LCM-STRESS-LOG-LINE " * 900)
    if "recall_query" in shapes:
        base += " recall_query=retrieve the oldest stored stress sentinel."
    return base


def run_stress_gate(
    *,
    dry_run: bool,
    turns: int = DEFAULT_TURNS,
    compaction_every: int = DEFAULT_COMPACTION_EVERY,
    provider_down_compactions: set[int] | frozenset[int] = DEFAULT_PROVIDER_DOWN_COMPACTIONS,
    restart_turns: set[int] | frozenset[int] = DEFAULT_RESTART_TURNS,
    fail_closed_turns: set[int] | frozenset[int] = frozenset(),
    thresholds: StressThresholds | None = None,
    spend_policy: SpendPolicy | None = None,
    out_path: Path | None = None,
) -> StressRun:
    if not dry_run:
        raise ValueError("live stress mode is intentionally not implemented here; run with --dry-run")
    if turns <= 0:
        raise ValueError("turns must be positive")
    if compaction_every <= 0:
        raise ValueError("compaction_every must be positive")

    thresholds = thresholds or StressThresholds()
    spend_policy = spend_policy or _cost_policy()
    store = SimulatedStore()
    records: list[StressRecord] = []
    coverage: Counter[str] = Counter()
    compaction_count = 0
    degraded_count = 0
    consecutive_degraded = 0
    max_consecutive_degraded = 0
    fail_closed_count = 0
    restart_count = 0
    persistence_checks: list[bool] = []
    estimated_spend = 0.0
    observed_spend = 0.0
    known_keys: list[str] = []

    for turn in range(1, turns + 1):
        shapes: list[str] = ["normal_chat"]
        if turn % 6 == 0:
            shapes.append("tool_call_like_row")
        if turn % 9 == 0:
            shapes.append("long_tool_output")
        if turn % 10 == 0:
            shapes.append("recall_query")

        key = f"stress-sentinel-{turn:03d}"
        value = f"LCM stress sentinel stored at turn {turn:03d}"
        if turn % 5 == 0:
            store.ingest(key, value)
            known_keys.append(key)

        compaction_index: int | None = None
        degraded = False
        if turn % compaction_every == 0:
            compaction_count += 1
            compaction_index = compaction_count
            if compaction_index in provider_down_compactions:
                degraded = True
                degraded_count += 1
                consecutive_degraded += 1
                shapes.append("provider_down_window")
            else:
                consecutive_degraded = 0
            max_consecutive_degraded = max(max_consecutive_degraded, consecutive_degraded)

        recall_ok: bool | None = None
        if "recall_query" in shapes:
            target = known_keys[0] if known_keys else key
            recall_ok = store.recall(target) is not None

        fail_closed = turn in fail_closed_turns
        if fail_closed:
            fail_closed_count += 1
            recall_ok = False

        if turn in restart_turns:
            before = store.snapshot()
            restored = SimulatedStore.restore(before)
            persistence_checks.append(restored.snapshot() == before and bool(restored.rows))
            store = restored
            restart_count += 1
            shapes.append("simulated_restart")

        for shape in shapes:
            coverage[shape] += 1

        text = _turn_text(turn, shapes)
        estimated_prompt = estimate_text_tokens(text)
        estimated_completion = 96 if "recall_query" in shapes else 32
        observed_prompt = max(1, int(estimated_prompt * (0.92 if degraded else 0.85)))
        observed_completion = max(1, int(estimated_completion * (0.80 if degraded else 0.70)))
        estimated_spend += spend_policy.cost(estimated_prompt, estimated_completion)
        observed_spend += spend_policy.cost(observed_prompt, observed_completion)
        note_parts: list[str] = []
        if degraded:
            note_parts.append("provider-down window: compaction failed open and marked degraded")
        if fail_closed:
            note_parts.append("fail-closed recall: store unavailable for query")
        if turn in restart_turns:
            note_parts.append("simulated restart: store snapshot reopened and checked")
        if not note_parts:
            note_parts.append("ok")

        records.append(
            StressRecord(
                turn=turn,
                shapes=tuple(shapes),
                compaction_index=compaction_index,
                degraded=degraded,
                fail_closed=fail_closed,
                recall_ok=recall_ok,
                estimated_prompt_tokens=estimated_prompt,
                estimated_completion_tokens=estimated_completion,
                observed_prompt_tokens=observed_prompt,
                observed_completion_tokens=observed_completion,
                note="; ".join(note_parts),
            )
        )

    degraded_rate = degraded_count / compaction_count if compaction_count else 0.0
    store_persistence = bool(persistence_checks) and all(persistence_checks)
    failures: list[str] = []
    if max_consecutive_degraded >= thresholds.max_consecutive_degraded:
        failures.append(
            f"degraded alert threshold tripped: {max_consecutive_degraded} consecutive compactions "
            f">= {thresholds.max_consecutive_degraded} consecutive"
        )
    if degraded_rate > thresholds.max_degraded_rate:
        failures.append(
            f"degraded alert threshold tripped: degraded rate {degraded_rate:.3%} "
            f"> {thresholds.max_degraded_rate:.3%}"
        )
    if fail_closed_count:
        failures.append(f"fail-closed count must be zero for cutover; saw {fail_closed_count}")
    if not store_persistence:
        failures.append("store persistence check failed across simulated restart")
    status = "FAIL-LOUD" if failures else "PASS"

    run = StressRun(
        dry_run=dry_run,
        turns=turns,
        compaction_every=compaction_every,
        compaction_count=compaction_count,
        degraded_count=degraded_count,
        degraded_rate=degraded_rate,
        max_consecutive_degraded=max_consecutive_degraded,
        fail_closed_count=fail_closed_count,
        store_persistence=store_persistence,
        restart_count=restart_count,
        coverage_counts=dict(coverage),
        estimated_spend_usd=estimated_spend,
        observed_spend_usd=observed_spend,
        status=status,
        failures=failures,
        records=records,
        out_path=out_path,
    )
    if out_path is not None:
        write_report(out_path, run)
    return run


def write_report(out_path: Path, run: StressRun) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# PRD-6 Aegis LCM Stress Gate",
        "",
        f"Generated: {ts}",
        f"Mode: {'dry-run' if run.dry_run else 'live'}",
        f"Status: {run.status}",
        "",
        "## Gate summary",
        "",
        f"- Turns: {run.turns}",
        f"- Compaction count: {run.compaction_count}",
        f"- Degraded count: {run.degraded_count}",
        f"- Degraded rate: {run.degraded_rate:.3%}",
        f"- Max consecutive degraded: {run.max_consecutive_degraded}",
        f"- Fail-closed count: {run.fail_closed_count}",
        f"- Store persistence: {'PASS' if run.store_persistence else 'FAIL'}",
        f"- Simulated restart count: {run.restart_count}",
        f"- estimated spend: ${run.estimated_spend_usd:.6f}",
        f"- observed spend: ${run.observed_spend_usd:.6f}",
        "",
        "## Alert contract",
        "",
        "- LOUD failure status is FAIL-LOUD.",
        f"- Trigger: degraded compactions >= {DEFAULT_CONSECUTIVE_DEGRADED_MAX} consecutive or degraded rate > {DEFAULT_DEGRADED_RATE_MAX:.3%}.",
        "- Apollo routing target: #alerts when Status is FAIL-LOUD.",
        "",
        "## Scenario coverage",
        "",
    ]
    for name in sorted(run.coverage_counts):
        label = name.replace("_", "-")
        lines.append(f"- {label}: {run.coverage_counts[name]}")
    lines.extend(["", "## Failures", ""])
    if run.failures:
        lines.extend(f"- {failure}" for failure in run.failures)
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Stress records",
        "",
        "| turn | shapes | compaction | degraded | fail_closed | recall_ok | est_tokens | obs_tokens | note |",
        "|---:|---|---:|---|---|---|---:|---:|---|",
    ])
    for record in run.records:
        est_tokens = record.estimated_prompt_tokens + record.estimated_completion_tokens
        obs_tokens = record.observed_prompt_tokens + record.observed_completion_tokens
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.turn),
                    _md(", ".join(record.shapes)),
                    "" if record.compaction_index is None else str(record.compaction_index),
                    str(record.degraded),
                    str(record.fail_closed),
                    "" if record.recall_ok is None else str(record.recall_ok),
                    str(est_tokens),
                    str(obs_tokens),
                    _md(record.note),
                ]
            )
            + " |"
        )
    lines.extend([
        "",
        "## Safety notes",
        "",
        "- This stress driver never mutates Hermes config and never invokes a live lifecycle command.",
        "- Dry-run provider-down windows are simulated so degraded-rate alert routing is testable without live spend.",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Run deterministic offline stress mode.")
    parser.add_argument("--out", required=True, help="Markdown report path.")
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--compaction-every", type=int, default=DEFAULT_COMPACTION_EVERY)
    parser.add_argument(
        "--provider-down-compactions",
        default=",".join(str(v) for v in sorted(DEFAULT_PROVIDER_DOWN_COMPACTIONS)),
        help="Comma-separated compaction indexes to mark provider-down/degraded.",
    )
    parser.add_argument(
        "--restart-turns",
        default=",".join(str(v) for v in sorted(DEFAULT_RESTART_TURNS)),
        help="Comma-separated turn indexes where the dry-run store is serialized and reopened.",
    )
    parser.add_argument(
        "--fail-closed-turns",
        default="",
        help="Comma-separated turn indexes to simulate fail-closed recall errors.",
    )
    parser.add_argument("--max-degraded-rate", type=float, default=DEFAULT_DEGRADED_RATE_MAX)
    parser.add_argument("--max-consecutive-degraded", type=int, default=DEFAULT_CONSECUTIVE_DEGRADED_MAX)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.dry_run:
        print("live stress mode is not implemented in this tool; pass --dry-run", file=sys.stderr)
        return 2
    thresholds = StressThresholds(
        max_consecutive_degraded=args.max_consecutive_degraded,
        max_degraded_rate=args.max_degraded_rate,
    )
    run = run_stress_gate(
        dry_run=True,
        turns=args.turns,
        compaction_every=args.compaction_every,
        provider_down_compactions=_parse_int_set(args.provider_down_compactions),
        restart_turns=_parse_int_set(args.restart_turns),
        fail_closed_turns=_parse_int_set(args.fail_closed_turns),
        thresholds=thresholds,
        out_path=Path(args.out).expanduser().resolve(),
    )
    print(
        f"status={run.status} compactions={run.compaction_count} degraded={run.degraded_count} "
        f"rate={run.degraded_rate:.3%} fail_closed={run.fail_closed_count} report={run.out_path}"
    )
    if run.failures:
        for failure in run.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
    return 0 if run.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
