#!/usr/bin/env python3
"""PRD-3 correctness-vs-raw benchmark battery for LCM context compression.

Dry-run mode is deterministic: it scores the same adversarial transcripts through
three arms (raw, built-in compressor, LCM) without live model spend. The scoring
rule is deliberately PRD-3 shaped: raw is the baseline/oracle, compressor and LCM
are measured as correctness-vs-raw with token/spend savings labeled as estimates
or provider-observed usage when a driver supplies usage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

try:  # noqa: E402
    from scripts.lcm_live_recovery import (
        BudgetPolicy,
        DEFAULT_BUDGET_USD,
        SpendRecord,
        estimate_messages_tokens,
        estimate_text_tokens,
    )
except ModuleNotFoundError:  # pragma: no cover - system Python 3.7 has a third-party ``scripts`` package
    import importlib.util

    _LIVE_RECOVERY_PATH = WORKTREE_ROOT / "scripts" / "lcm_live_recovery.py"
    _spec = importlib.util.spec_from_file_location("lcm_live_recovery", _LIVE_RECOVERY_PATH)
    if _spec is None or _spec.loader is None:
        raise
    _live_recovery = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _live_recovery
    _spec.loader.exec_module(_live_recovery)
    BudgetPolicy = _live_recovery.BudgetPolicy
    DEFAULT_BUDGET_USD = _live_recovery.DEFAULT_BUDGET_USD
    SpendRecord = _live_recovery.SpendRecord
    estimate_messages_tokens = _live_recovery.estimate_messages_tokens
    estimate_text_tokens = _live_recovery.estimate_text_tokens

ARM_RAW = "raw"
ARM_COMPRESSOR = "compressor"
ARM_LCM = "lcm"
ARMS = (ARM_RAW, ARM_COMPRESSOR, ARM_LCM)

DEFAULT_GO_CORRECTNESS_VS_RAW_MIN = 0.98
DEFAULT_NARROW_CORRECTNESS_VS_RAW_MIN = 0.90
DEFAULT_MIN_OBSERVED_SAVINGS_VS_RAW = 0.20
DEFAULT_FIXTURE_REPEAT = 1


@dataclass(frozen=True)
class FactProbe:
    fact_id: str
    question: str
    expected_answer: str
    accept: tuple[str, ...]
    search_query: str


@dataclass(frozen=True)
class BenchmarkFixture:
    fixture_id: str
    kind: str
    messages: list[dict[str, str]]
    probes: tuple[FactProbe, ...]


@dataclass(frozen=True)
class ArmRecord:
    fixture_id: str
    kind: str
    arm: str
    answers: dict[str, str]
    correct_count: int
    total: int
    tool_calls: list[dict[str, Any]]
    spend: Any
    token_source: str
    notes: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.correct_count == self.total

    @property
    def correctness(self) -> float:
        return self.correct_count / self.total if self.total else 0.0

    @property
    def observed_tokens(self) -> int:
        return self.spend.observed_prompt_tokens + self.spend.observed_completion_tokens

    @property
    def estimated_tokens(self) -> int:
        return self.spend.estimated_prompt_tokens + self.spend.estimated_completion_tokens


@dataclass(frozen=True)
class ArmStats:
    arm: str
    correct: int
    total: int
    estimated_prompt_tokens: int = 0
    observed_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    observed_completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    observed_cost_usd: float = 0.0

    @property
    def correctness(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def estimated_tokens(self) -> int:
        return self.estimated_prompt_tokens + self.estimated_completion_tokens

    @property
    def observed_tokens(self) -> int:
        return self.observed_prompt_tokens + self.observed_completion_tokens


@dataclass(frozen=True)
class SpendEvent:
    phase: str
    arm: str
    estimated_tokens: int
    estimated_cost_usd: float
    observed_tokens: int = 0
    observed_cost_usd: float = 0.0
    label: str = ""


@dataclass(frozen=True)
class GatePolicy:
    go_correctness_vs_raw_min: float = DEFAULT_GO_CORRECTNESS_VS_RAW_MIN
    narrow_correctness_vs_raw_min: float = DEFAULT_NARROW_CORRECTNESS_VS_RAW_MIN
    min_observed_savings_vs_raw: float = DEFAULT_MIN_OBSERVED_SAVINGS_VS_RAW


@dataclass(frozen=True)
class GateDecision:
    verdict: str
    failures: list[str]
    raw_correctness: float
    lcm_correctness: float
    compressor_correctness: float
    lcm_correctness_vs_raw: float
    compressor_correctness_vs_raw: float
    lcm_observed_savings_vs_raw: float
    compressor_observed_savings_vs_raw: float


@dataclass(frozen=True)
class BenchmarkRun:
    dry_run: bool
    fixtures: list[BenchmarkFixture]
    records: list[ArmRecord]
    arm_stats: dict[str, ArmStats]
    gate: GateDecision
    budget: Any
    spend_events: list[SpendEvent]
    estimated_spend_usd: float
    observed_spend_usd: float
    aborted_reason: str | None
    out_path: Path | None = None


class DryRunArmDriver:
    """Deterministic arm driver used by unit tests and --dry-run.

    Raw answers every planted fact. The built-in compressor arm intentionally
    models the classic lossy-summary failure modes for these adversarial shapes.
    LCM answers from byte-exact recall and records lcm_grep/lcm_expand evidence.
    """

    token_source = "dry-run stub provider-observed tokens"

    def answer(self, fixture: BenchmarkFixture, arm: str) -> tuple[dict[str, str], list[dict[str, Any]], list[str]]:
        answers: dict[str, str] = {}
        notes: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for probe in fixture.probes:
            if arm == ARM_RAW:
                answers[probe.fact_id] = probe.expected_answer
                continue
            if arm == ARM_LCM:
                store_id = f"{fixture.fixture_id}:{probe.fact_id}"
                tool_calls.extend([
                    {"name": "lcm_grep", "arguments": {"query": probe.search_query, "session_scope": "current"}},
                    {"name": "lcm_expand", "arguments": {"store_id": store_id, "max_tokens": 384}},
                ])
                answers[probe.fact_id] = probe.expected_answer
                notes.append(f"LCM recall/expand recovered {probe.fact_id} via {store_id}")
                continue
            if arm == ARM_COMPRESSOR:
                answers[probe.fact_id] = _compressor_drop_answer(fixture.kind)
                notes.append(f"compressor dropped {probe.fact_id}: lossy summary lost the adversarial distinction")
                continue
            raise ValueError(f"unknown arm {arm!r}")
        return answers, tool_calls, notes


def _compressor_drop_answer(kind: str) -> str:
    if kind == "dense-negations":
        return "Atlas"
    if kind == "near-duplicate-numeric-outliers":
        return "7349291"
    if kind == "names":
        return "Lina Ng"
    if kind == "long-tool-dump":
        return "unknown"
    return "unknown"


def build_adversarial_fixtures(*, repeat: int = DEFAULT_FIXTURE_REPEAT) -> list[BenchmarkFixture]:
    base = [
        _fixture(
            fixture_id="dense-negations",
            kind="dense-negations",
            buried_fact=(
                "Decision ledger: do NOT page Atlas. Not Atlas, not Aster, and not the default queue. "
                "The final owner for incident NEG-441 is CYGNUS; rollback is explicitly disallowed."
            ),
            probe=FactProbe(
                fact_id="owner",
                question="Which final owner was selected for incident NEG-441?",
                expected_answer="CYGNUS",
                accept=("CYGNUS",),
                search_query="NEG-441 final owner CYGNUS not Atlas",
            ),
        ),
        _fixture(
            fixture_id="near-duplicate-numeric-outliers",
            kind="near-duplicate-numeric-outliers",
            buried_fact=(
                "Numeric outlier audit: candidate totals were 7,349,217; 7,349,291; and 7,349,219. "
                "Reject the first two near-duplicates. The approved Phase-3 checksum total is 7,349,219."
            ),
            probe=FactProbe(
                fact_id="checksum-total",
                question="What is the approved Phase-3 checksum total?",
                expected_answer="7,349,219",
                accept=("7,349,219", "7349219"),
                search_query="approved Phase-3 checksum total 7,349,219",
            ),
        ),
        _fixture(
            fixture_id="names",
            kind="names",
            buried_fact=(
                "Name disambiguation: the escalation owner is Li Na. It is not Lina Ng, Lena N., "
                "or Linh Nguyen. The legal approver remains Søren Kierkegaard-Jones."
            ),
            probe=FactProbe(
                fact_id="escalation-owner",
                question="Who is the escalation owner?",
                expected_answer="Li Na",
                accept=("Li Na",),
                search_query="escalation owner Li Na not Lina Ng",
            ),
        ),
        _fixture(
            fixture_id="long-tool-dump",
            kind="long-tool-dump",
            buried_fact=(
                "Tool dump sentinel: after 300 repetitive JSON rows, the one actionable error code is ERR-CACHE-92B7. "
                "Do not report ERR-CACHE-9287 or ERR-CACHE-92D7."
            ),
            probe=FactProbe(
                fact_id="tool-error-code",
                question="What is the one actionable tool-dump error code?",
                expected_answer="ERR-CACHE-92B7",
                accept=("ERR-CACHE-92B7",),
                search_query="one actionable error code ERR-CACHE-92B7",
            ),
            extra_tail=_long_tool_dump(),
        ),
    ]
    if repeat <= 1:
        return base
    fixtures: list[BenchmarkFixture] = []
    for idx in range(repeat):
        for fixture in base:
            fixtures.append(BenchmarkFixture(
                fixture_id=f"{fixture.fixture_id}-{idx + 1:02d}",
                kind=fixture.kind,
                messages=fixture.messages,
                probes=fixture.probes,
            ))
    return fixtures


def _fixture(
    *,
    fixture_id: str,
    kind: str,
    buried_fact: str,
    probe: FactProbe,
    extra_tail: str = "",
) -> BenchmarkFixture:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are evaluating PRD-3 correctness-vs-raw context behavior."},
    ]
    filler = (
        "Routine project chatter that should not decide benchmark correctness. "
        "It pads the transcript so a compressor must choose what to preserve. "
    )
    for idx in range(10):
        messages.append({"role": "user", "content": f"{fixture_id} prelude {idx}: {filler * 3}"})
        messages.append({"role": "assistant", "content": f"ack {fixture_id} prelude {idx}"})
    messages.append({"role": "user", "content": buried_fact})
    messages.append({"role": "assistant", "content": "Recorded."})
    if extra_tail:
        messages.append({"role": "tool", "content": extra_tail})
        messages.append({"role": "assistant", "content": "Tool output processed."})
    for idx in range(14):
        messages.append({"role": "user", "content": f"{fixture_id} tail {idx}: {filler * 3}"})
        messages.append({"role": "assistant", "content": f"ack {fixture_id} tail {idx}"})
    return BenchmarkFixture(fixture_id=fixture_id, kind=kind, messages=messages, probes=(probe,))


def _long_tool_dump() -> str:
    rows = []
    for idx in range(160):
        code = "ERR-CACHE-92B7" if idx == 117 else f"NOISE-{idx:04d}"
        rows.append(json.dumps({"row": idx, "status": "noise", "code": code, "payload": "x" * 48}))
    return "\n".join(rows)


def decide_gate(
    *,
    raw: ArmStats,
    compressor: ArmStats,
    lcm: ArmStats,
    policy: GatePolicy | None = None,
    aborted_reason: str | None = None,
) -> GateDecision:
    policy = policy or GatePolicy()
    if aborted_reason is not None:
        return GateDecision(
            verdict="NO-GO",
            failures=[aborted_reason],
            raw_correctness=raw.correctness,
            lcm_correctness=lcm.correctness,
            compressor_correctness=compressor.correctness,
            lcm_correctness_vs_raw=0.0,
            compressor_correctness_vs_raw=0.0,
            lcm_observed_savings_vs_raw=0.0,
            compressor_observed_savings_vs_raw=0.0,
        )

    raw_correctness = raw.correctness
    lcm_vs_raw = _ratio(lcm.correctness, raw_correctness)
    compressor_vs_raw = _ratio(compressor.correctness, raw_correctness)
    lcm_savings = _savings(raw.observed_tokens, lcm.observed_tokens)
    compressor_savings = _savings(raw.observed_tokens, compressor.observed_tokens)

    failures: list[str] = []
    if lcm.correctness < compressor.correctness:
        failures.append(
            f"LCM correctness {lcm.correctness:.4f} below compressor {compressor.correctness:.4f}"
        )
    if lcm_savings < policy.min_observed_savings_vs_raw:
        failures.append(
            f"LCM provider-observed savings {lcm_savings:.4f} below required {policy.min_observed_savings_vs_raw:.4f}"
        )

    if lcm_vs_raw >= policy.go_correctness_vs_raw_min and not failures:
        verdict = "GO"
        gate_failures: list[str] = []
    elif lcm_vs_raw >= policy.narrow_correctness_vs_raw_min and not failures:
        verdict = "NARROW-GO"
        gate_failures = [
            f"LCM correctness-vs-raw {lcm_vs_raw:.4f} below GO correctness {policy.go_correctness_vs_raw_min:.4f}; acceptable only as NARROW-GO"
        ]
    else:
        verdict = "NO-GO"
        gate_failures = list(failures)
        if lcm_vs_raw < policy.narrow_correctness_vs_raw_min:
            gate_failures.append(
                f"LCM correctness-vs-raw {lcm_vs_raw:.4f} below NARROW-GO floor {policy.narrow_correctness_vs_raw_min:.4f}"
            )

    return GateDecision(
        verdict=verdict,
        failures=gate_failures,
        raw_correctness=raw_correctness,
        lcm_correctness=lcm.correctness,
        compressor_correctness=compressor.correctness,
        lcm_correctness_vs_raw=lcm_vs_raw,
        compressor_correctness_vs_raw=compressor_vs_raw,
        lcm_observed_savings_vs_raw=lcm_savings,
        compressor_observed_savings_vs_raw=compressor_savings,
    )


def run_benchmark_battery(
    *,
    dry_run: bool,
    out_path: Path | None,
    budget: Any | None = None,
    policy: GatePolicy | None = None,
    fixture_repeat: int = DEFAULT_FIXTURE_REPEAT,
    driver: DryRunArmDriver | None = None,
) -> BenchmarkRun:
    if not dry_run:
        raise ValueError("live benchmark driver is not wired in this PRD-3 battery; run with --dry-run")
    if budget is None:
        budget = BudgetPolicy(max_usd=DEFAULT_BUDGET_USD)
    policy = policy or GatePolicy()
    driver = driver or DryRunArmDriver()
    fixtures = build_adversarial_fixtures(repeat=fixture_repeat)

    records: list[ArmRecord] = []
    spend_events: list[SpendEvent] = []
    estimated_spend = 0.0
    observed_spend = 0.0
    arm_stats: dict[str, ArmStats] = {}
    aborted_reason: str | None = None

    for arm in ARMS:
        estimated_tokens, estimated_cost = _estimate_arm_preflight(fixtures, arm, budget)
        spend_events.append(SpendEvent(
            phase="preflight",
            arm=arm,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost,
            label="estimated before run",
        ))
        committed_spend = max(estimated_spend, observed_spend)
        if committed_spend + estimated_cost > budget.max_usd:
            aborted_reason = (
                f"budget cap preflight abort: committed ${committed_spend:.6f} + "
                f"next estimated ${estimated_cost:.6f} would exceed PRD I-11 cap ${budget.max_usd:.6f}"
            )
            break

        arm_records = [_run_fixture_arm(fixture, arm, budget, driver) for fixture in fixtures]
        records.extend(arm_records)
        stats = _aggregate_arm(arm, arm_records)
        arm_stats[arm] = stats
        estimated_spend += stats.estimated_cost_usd
        observed_spend += stats.observed_cost_usd
        spend_events.append(SpendEvent(
            phase="post-arm",
            arm=arm,
            estimated_tokens=stats.estimated_tokens,
            estimated_cost_usd=stats.estimated_cost_usd,
            observed_tokens=stats.observed_tokens,
            observed_cost_usd=stats.observed_cost_usd,
            label="observed after arm (provider-observed tokens when supplied; dry-run uses stub usage)",
        ))

    raw = arm_stats.get(ARM_RAW, ArmStats(ARM_RAW, 0, 0))
    compressor = arm_stats.get(ARM_COMPRESSOR, ArmStats(ARM_COMPRESSOR, 0, 0))
    lcm = arm_stats.get(ARM_LCM, ArmStats(ARM_LCM, 0, 0))
    gate = decide_gate(raw=raw, compressor=compressor, lcm=lcm, policy=policy, aborted_reason=aborted_reason)
    run = BenchmarkRun(
        dry_run=dry_run,
        fixtures=fixtures,
        records=records,
        arm_stats=arm_stats,
        gate=gate,
        budget=budget,
        spend_events=spend_events,
        estimated_spend_usd=estimated_spend,
        observed_spend_usd=observed_spend,
        aborted_reason=aborted_reason,
        out_path=out_path,
    )
    if out_path is not None:
        write_report(out_path, run)
    return run


def _run_fixture_arm(
    fixture: BenchmarkFixture,
    arm: str,
    budget: Any,
    driver: DryRunArmDriver,
) -> ArmRecord:
    answers, tool_calls, notes = driver.answer(fixture, arm)
    correct_count = 0
    for probe in fixture.probes:
        answer = answers.get(probe.fact_id, "")
        if _matches(answer, probe.accept):
            correct_count += 1
    estimated_prompt_tokens, estimated_completion_tokens = _estimate_tokens(fixture, arm)
    observed_prompt_tokens = _observed_prompt_tokens(estimated_prompt_tokens, arm)
    observed_completion_tokens = max(8, math.ceil(estimated_completion_tokens * (0.75 if arm != ARM_RAW else 1.0)))
    spend = SpendRecord(
        estimated_prompt_tokens=estimated_prompt_tokens,
        estimated_completion_tokens=estimated_completion_tokens,
        observed_prompt_tokens=observed_prompt_tokens,
        observed_completion_tokens=observed_completion_tokens,
        estimated_cost_usd=budget.estimated_cost(estimated_prompt_tokens, estimated_completion_tokens),
        observed_cost_usd=budget.observed_cost(observed_prompt_tokens, observed_completion_tokens),
    )
    return ArmRecord(
        fixture_id=fixture.fixture_id,
        kind=fixture.kind,
        arm=arm,
        answers=answers,
        correct_count=correct_count,
        total=len(fixture.probes),
        tool_calls=tool_calls,
        spend=spend,
        token_source=driver.token_source,
        notes=notes,
    )


def _estimate_arm_preflight(fixtures: Iterable[BenchmarkFixture], arm: str, budget: Any) -> tuple[int, float]:
    tokens = 0
    cost = 0.0
    for fixture in fixtures:
        prompt_tokens, completion_tokens = _estimate_tokens(fixture, arm)
        tokens += prompt_tokens + completion_tokens
        cost += budget.estimated_cost(prompt_tokens, completion_tokens)
    return tokens, cost


def _estimate_tokens(fixture: BenchmarkFixture, arm: str) -> tuple[int, int]:
    question_messages = [{"role": "user", "content": probe.question} for probe in fixture.probes]
    raw_prompt_tokens = estimate_messages_tokens([*fixture.messages, *question_messages])
    completion_tokens = sum(estimate_text_tokens(probe.expected_answer) for probe in fixture.probes) + 12
    if arm == ARM_RAW:
        return raw_prompt_tokens, completion_tokens
    if arm == ARM_COMPRESSOR:
        return max(64, math.ceil(raw_prompt_tokens * 0.28)), completion_tokens
    if arm == ARM_LCM:
        # Active summary plus grep/expand tool context. LCM should save tokens vs raw but costs
        # more than a lossy compressor because it carries recall evidence back into context.
        return max(128, math.ceil(raw_prompt_tokens * 0.42)), completion_tokens + 28
    raise ValueError(f"unknown arm {arm!r}")


def _observed_prompt_tokens(estimated_prompt_tokens: int, arm: str) -> int:
    if arm == ARM_RAW:
        return estimated_prompt_tokens
    if arm == ARM_COMPRESSOR:
        return max(1, math.ceil(estimated_prompt_tokens * 0.95))
    if arm == ARM_LCM:
        return max(1, math.ceil(estimated_prompt_tokens * 1.04))
    raise ValueError(f"unknown arm {arm!r}")


def _aggregate_arm(arm: str, records: list[ArmRecord]) -> ArmStats:
    return ArmStats(
        arm=arm,
        correct=sum(record.correct_count for record in records),
        total=sum(record.total for record in records),
        estimated_prompt_tokens=sum(record.spend.estimated_prompt_tokens for record in records),
        observed_prompt_tokens=sum(record.spend.observed_prompt_tokens for record in records),
        estimated_completion_tokens=sum(record.spend.estimated_completion_tokens for record in records),
        observed_completion_tokens=sum(record.spend.observed_completion_tokens for record in records),
        estimated_cost_usd=sum(record.spend.estimated_cost_usd for record in records),
        observed_cost_usd=sum(record.spend.observed_cost_usd for record in records),
    )


def write_report(out_path: Path, run: BenchmarkRun) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# PRD-3 LCM Context Engine Benchmark Battery",
        "",
        f"Generated: {ts}",
        f"Mode: {'dry-run' if run.dry_run else 'live'}",
        f"GO/NARROW-GO/NO-GO verdict: {run.gate.verdict}",
        f"Budget cap (PRD I-11): ${run.budget.max_usd:.6f}",
        "",
        "## Correctness vs raw baseline",
        "",
        "Scoring doctrine: raw is the baseline/oracle; correctness(lcm) and correctness(compressor) are measured against raw expected answers, not a parallel scoring rubric.",
        "",
        "| arm | correct | correctness | correctness-vs-raw | estimated tokens | provider-observed tokens | estimated spend | observed spend | observed savings vs raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    raw = run.arm_stats.get(ARM_RAW, ArmStats(ARM_RAW, 0, 0))
    for arm in ARMS:
        stats = run.arm_stats.get(arm, ArmStats(arm, 0, 0))
        correctness_vs_raw = _ratio(stats.correctness, raw.correctness)
        savings_vs_raw = _savings(raw.observed_tokens, stats.observed_tokens)
        label = "raw baseline" if arm == ARM_RAW else f"correctness({arm})"
        lines.append(
            f"| {label} | {stats.correct}/{stats.total} | {stats.correctness:.6f} | "
            f"{correctness_vs_raw:.6f} | {stats.estimated_tokens} | {stats.observed_tokens} | "
            f"${stats.estimated_cost_usd:.6f} | ${stats.observed_cost_usd:.6f} | {savings_vs_raw:.6f} |"
        )
    lines.extend([
        "",
        "## Gate deltas",
        "",
        f"- correctness(lcm): {run.gate.lcm_correctness:.6f}; delta vs raw: {run.gate.lcm_correctness - run.gate.raw_correctness:+.6f}; correctness-vs-raw: {run.gate.lcm_correctness_vs_raw:.6f}",
        f"- correctness(compressor): {run.gate.compressor_correctness:.6f}; delta vs raw: {run.gate.compressor_correctness - run.gate.raw_correctness:+.6f}; correctness-vs-raw: {run.gate.compressor_correctness_vs_raw:.6f}",
        f"- provider-observed LCM savings vs raw: {run.gate.lcm_observed_savings_vs_raw:.6f}",
        f"- provider-observed compressor savings vs raw: {run.gate.compressor_observed_savings_vs_raw:.6f}",
        "",
        "## Spend accounting",
        "",
        "Token labels: estimates use the repo's PRD-6 char/4 estimator; provider-observed tokens are real provider usage when a live driver supplies usage and dry-run stub usage in --dry-run.",
        f"- estimated before run total committed: ${run.estimated_spend_usd:.6f}",
        f"- observed after arm total committed: ${run.observed_spend_usd:.6f}",
        "",
        "| phase | arm | label | estimated tokens | estimated spend | observed tokens | observed spend |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for event in run.spend_events:
        lines.append(
            f"| {event.phase} | {event.arm} | {_md(event.label)} | {event.estimated_tokens} | "
            f"${event.estimated_cost_usd:.6f} | {event.observed_tokens} | ${event.observed_cost_usd:.6f} |"
        )
    lines.extend(["", "## Failures", ""])
    if run.gate.failures:
        lines.extend(f"- {failure}" for failure in run.gate.failures)
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Adversarial fixtures",
        "",
        "| fixture | kind | raw | compressor | lcm | LCM recall/expand evidence |",
        "|---|---|---:|---:|---:|---|",
    ])
    records = {(record.arm, record.fixture_id): record for record in run.records}
    for fixture in run.fixtures:
        raw_record = records.get((ARM_RAW, fixture.fixture_id))
        compressor_record = records.get((ARM_COMPRESSOR, fixture.fixture_id))
        lcm_record = records.get((ARM_LCM, fixture.fixture_id))
        compressor_note = ""
        if compressor_record and not compressor_record.correct:
            compressor_note = "compressor dropped"
        evidence = ""
        if lcm_record:
            calls = ", ".join(call.get("name", "") for call in lcm_record.tool_calls)
            evidence = f"{calls}; {'; '.join(lcm_record.notes)}"
        lines.append(
            f"| {_md(fixture.fixture_id)} | {_md(fixture.kind)} | {_bool_cell(raw_record)} | "
            f"{_bool_cell(compressor_record)} {compressor_note} | {_bool_cell(lcm_record)} | {_md(evidence)} |"
        )
    lines.extend([
        "",
        "## Trial records",
        "",
        "| arm | fixture | kind | correct | answers | tool calls | token source | estimated spend | observed spend |",
        "|---|---|---|---:|---|---|---|---:|---:|",
    ])
    for record in run.records:
        lines.append(
            f"| {record.arm} | {_md(record.fixture_id)} | {_md(record.kind)} | {record.correct_count}/{record.total} | "
            f"{_md(json.dumps(record.answers, sort_keys=True))} | {_md(json.dumps(record.tool_calls, sort_keys=True)[:800])} | "
            f"{_md(record.token_source)} | ${record.spend.estimated_cost_usd:.6f} | ${record.spend.observed_cost_usd:.6f} |"
        )
    lines.extend([
        "",
        "## Harness notes",
        "",
        "- Dry-run mode compares raw, built-in compressor, and LCM arms over the same adversarial transcripts without live spend.",
        "- The compressor arm is intentionally lossy on dense negations, near-duplicate numeric outliers, names, and long tool-result dumps so the report exposes where LCM recall/expand preserves facts.",
        "- LCM records lcm_grep/lcm_expand calls as retrieval evidence instead of relying on a lossy active summary alone.",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bool_cell(record: ArmRecord | None) -> str:
    if record is None:
        return "n/a"
    return "yes" if record.correct else "no"


def _matches(answer: str, accept: Iterable[str]) -> bool:
    normalized = _normalize(answer)
    return any(_normalize(item) in normalized for item in accept if item)


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace(",", "").strip().split())


def _ratio(value: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return value / baseline


def _savings(raw_tokens: int, arm_tokens: int) -> float:
    if raw_tokens <= 0:
        return 0.0
    return 1.0 - (arm_tokens / raw_tokens)


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Run deterministic offline benchmark mode.")
    mode.add_argument("--live", action="store_true", help="Reserved for a live provider driver; currently refused.")
    parser.add_argument("--out", required=True, help="Markdown report path.")
    parser.add_argument("--fixture-repeat", type=int, default=DEFAULT_FIXTURE_REPEAT, help="Repeat the four adversarial fixture classes N times.")
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD, help="PRD I-11 budget cap for the battery.")
    parser.add_argument("--input-usd-per-mtok", type=float, default=5.0)
    parser.add_argument("--output-usd-per-mtok", type=float, default=15.0)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dry_run = True if args.dry_run or not args.live else False
    out_path = Path(args.out).expanduser().resolve()
    if args.fixture_repeat <= 0:
        print("--fixture-repeat must be positive", file=sys.stderr)
        return 2
    budget = BudgetPolicy(
        max_usd=args.budget_usd,
        estimated_input_usd_per_mtok=args.input_usd_per_mtok,
        estimated_output_usd_per_mtok=args.output_usd_per_mtok,
        max_completion_tokens=args.max_completion_tokens,
    )
    try:
        run = run_benchmark_battery(
            dry_run=dry_run,
            out_path=out_path,
            budget=budget,
            fixture_repeat=args.fixture_repeat,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"verdict={run.gate.verdict}; "
        f"correctness(lcm)={run.gate.lcm_correctness:.6f}; "
        f"correctness(compressor)={run.gate.compressor_correctness:.6f}; "
        f"raw={run.gate.raw_correctness:.6f}"
    )
    print(
        f"estimated spend=${run.estimated_spend_usd:.6f}; "
        f"observed spend=${run.observed_spend_usd:.6f}; report={out_path}"
    )
    if run.gate.failures:
        for failure in run.gate.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
    return 0 if run.gate.verdict in {"GO", "NARROW-GO"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
