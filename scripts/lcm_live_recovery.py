#!/usr/bin/env python3
"""PRD-6 Phase-2 live-model recovery statistical harness.

Dry-run mode is deterministic and uses stubbed transcripts/responses so the gate
can be unit-tested without Aegis or live model spend. Live mode drives an Aegis
Hermes process through a configurable command and records the same per-trial
schema before applying the statistical gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LIVE_MIN_TRIALS = 180
WILSON_Z = 1.96
DEFAULT_RECALL_POINT_MIN = 0.95
DEFAULT_WILSON_LOWER_MIN = 0.90  # PRD-6: Wilson 95% lower bound >=0.90 (binding); point recall gate is the 0.95 lever
DEFAULT_JUDGE_MIN = 0.95
DEFAULT_BUDGET_USD = 25.0
DEFAULT_MAX_COMPLETION_TOKENS = 512
DEFAULT_INPUT_USD_PER_MTOK = 5.0
DEFAULT_OUTPUT_USD_PER_MTOK = 15.0
DEFAULT_COMPACTION_THRESHOLD_TOKENS = 5000


@dataclass(frozen=True)
class SamplingParams:
    """Sampling knobs recorded for every trial."""

    temperature: float | None = 0.0
    seed: int | None = 1729


@dataclass(frozen=True)
class SpendRecord:
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    observed_prompt_tokens: int = 0
    observed_completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    observed_cost_usd: float = 0.0


@dataclass(frozen=True)
class BudgetPolicy:
    """Token and cost cap used before each trial is sent."""

    max_usd: float = DEFAULT_BUDGET_USD
    estimated_input_usd_per_mtok: float = DEFAULT_INPUT_USD_PER_MTOK
    estimated_output_usd_per_mtok: float = DEFAULT_OUTPUT_USD_PER_MTOK
    observed_input_usd_per_mtok: float | None = None
    observed_output_usd_per_mtok: float | None = None
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS

    def estimated_cost(self, prompt_tokens: int, completion_tokens: int | None = None) -> float:
        completion = self.max_completion_tokens if completion_tokens is None else completion_tokens
        return _cost_usd(
            prompt_tokens,
            completion,
            self.estimated_input_usd_per_mtok,
            self.estimated_output_usd_per_mtok,
        )

    def observed_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_rate = self.observed_input_usd_per_mtok
        output_rate = self.observed_output_usd_per_mtok
        if input_rate is None:
            input_rate = self.estimated_input_usd_per_mtok
        if output_rate is None:
            output_rate = self.estimated_output_usd_per_mtok
        return _cost_usd(prompt_tokens, completion_tokens, input_rate, output_rate)


@dataclass(frozen=True)
class GateThresholds:
    min_trials: int = LIVE_MIN_TRIALS
    recall_point_min: float = DEFAULT_RECALL_POINT_MIN
    wilson_lower_min: float = DEFAULT_WILSON_LOWER_MIN
    judge_precision_min: float = DEFAULT_JUDGE_MIN
    judge_recall_min: float = DEFAULT_JUDGE_MIN
    require_tool_call_evidence: bool = True


@dataclass(frozen=True)
class TrialFixture:
    prompt_id: str
    arm: str
    buried_fact: str
    expected_answer: str
    semantic_accept: list[str]
    messages: list[dict[str, str]]
    question: str
    compaction_threshold_tokens: int = DEFAULT_COMPACTION_THRESHOLD_TOKENS


@dataclass(frozen=True)
class DriverResponse:
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    confidence_wrong: bool | None = None


@dataclass(frozen=True)
class TrialRecord:
    prompt_id: str
    arm: str
    buried_fact: str
    tool_calls: list[dict[str, Any]]
    answer: str
    correct: bool
    confidence_wrong: bool
    spend: SpendRecord


@dataclass(frozen=True)
class JudgeCalibrationResult:
    precision: float
    recall: float
    passed: bool
    details: str
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str]
    total_trials: int
    correct_trials: int
    point_recall: float
    wilson_lower: float
    confident_wrong: int
    missing_tool_call_evidence: int
    semantic_calibration: JudgeCalibrationResult | None


@dataclass(frozen=True)
class WilsonStats:
    successes: int
    total: int
    z: float
    point: float
    denominator: float
    centre: float
    margin: float
    lower: float


@dataclass(frozen=True)
class RecoveryRun:
    mode: str
    n_requested: int
    sampling: SamplingParams
    thresholds: GateThresholds
    budget: BudgetPolicy
    trials: list[TrialRecord]
    gate: GateResult
    semantic_calibration: JudgeCalibrationResult | None
    estimated_spend_usd: float
    observed_spend_usd: float
    aborted_reason: str | None
    out_path: Path | None = None


class RecoveryDriver:
    """Driver boundary used by dry-run stubs, unit tests, and live Aegis mode."""

    def run_trial(self, fixture: TrialFixture, sampling: SamplingParams) -> DriverResponse:
        raise NotImplementedError


class OfflineStubDriver(RecoveryDriver):
    """Deterministic offline driver that simulates successful LCM retrieval."""

    def run_trial(self, fixture: TrialFixture, sampling: SamplingParams) -> DriverResponse:
        query = fixture.buried_fact if fixture.arm == "exact" else fixture.expected_answer
        answer = fixture.expected_answer
        if fixture.arm == "semantic":
            answer = f"The recovered fact points to {fixture.expected_answer}."
        prompt_tokens = estimate_messages_tokens(fixture.messages + [{"role": "user", "content": fixture.question}])
        completion_tokens = max(16, estimate_text_tokens(answer))
        return DriverResponse(
            answer=answer,
            tool_calls=[
                {"name": "lcm_grep", "arguments": {"query": query}},
                {"name": "lcm_expand", "arguments": {"store_id": f"stub-{fixture.prompt_id}"}},
            ],
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            confidence_wrong=False,
        )


class LiveAegisDriver(RecoveryDriver):
    """Subprocess driver for a pre-activated Aegis profile.

    The default command is intentionally conservative and does not mutate Hermes
    configuration or restart gateways. Operators can pass --aegis-command when
    they have a richer live driver that emits JSON tool/usage markers.
    """

    def __init__(self, command: list[str] | None = None, timeout_seconds: int = 300) -> None:
        self.command = command or ["hermes", "-p", "aegis", "chat", "-q"]
        self.timeout_seconds = timeout_seconds

    def run_trial(self, fixture: TrialFixture, sampling: SamplingParams) -> DriverResponse:
        prompt = render_trial_prompt(fixture)
        env = os.environ.copy()
        if sampling.temperature is not None:
            env["HERMES_SAMPLING_TEMPERATURE"] = str(sampling.temperature)
        if sampling.seed is not None:
            env["HERMES_SAMPLING_SEED"] = str(sampling.seed)
        proc = subprocess.run(
            [*self.command, prompt],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        if proc.returncode != 0:
            answer = stdout.strip() or stderr.strip()
            return DriverResponse(answer=answer, tool_calls=[], usage={})
        return DriverResponse(
            answer=_extract_answer(stdout),
            tool_calls=_extract_tool_calls(stdout + "\n" + stderr),
            usage=_extract_usage(stdout + "\n" + stderr),
        )


def _clean_cli_answer(text: str) -> str:
    """Strip Hermes CLI banner/warning noise so only the model reply remains."""
    skip_prefixes = (
        "Warning:", "⚠", "session_id:", "Query:", "Initializing agent",
        "Resume this session", "↻", "⟳", "Normalized model",
        "claude-pool", "  hermes --resume", "Session:", "Duration:", "Messages:",
        "─", "╭", "╰",
    )
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if any(s.startswith(p) for p in skip_prefixes):
            continue
        out.append(s)
    return "\n".join(out).strip()


class LiveAegisSessionDriver(RecoveryDriver):
    """Persistent multi-turn driver that exercises the REAL LCM store.

    Unlike LiveAegisDriver (which renders the whole transcript inline in one
    prompt, so the model can read the buried fact directly and never needs an
    LCM tool), this driver:

      1. Plants the buried fact in a fresh Aegis session (captures session_id).
      2. Drives filler turns under a LOW ``LCM_CONTEXT_THRESHOLD`` so the fact
         ages out of the active tail and the store compacts/rolls over.
      3. Asks for the fact back, instructing the model to use its LCM retrieval
         tools, with ``-Q`` so stdout carries only the final answer (no prompt
         echo to contaminate recall scoring).
      4. Detects the ACTUAL ``lcm_*`` tool call by reading the Aegis lcm.db
         (the authoritative signal; the quiet CLI emits no JSON tool markers).

    Cost: 1 plant + N_FILLER filler + 1 recovery call per trial. Keep filler
    small; the point is crossing the (lowered) threshold, not raw volume.
    """

    def __init__(
        self,
        *,
        profile: str = "aegis",
        timeout_seconds: int = 300,
        threshold: float | None = 0.02,
        filler_turns: int = 4,
        filler_tokens: int = 2500,
        lcm_db: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        self.profile = profile
        self.timeout_seconds = timeout_seconds
        self.threshold = threshold
        self.filler_turns = filler_turns
        self.filler_tokens = filler_tokens
        self.model = model
        self.provider = provider
        self.lcm_db = lcm_db or os.path.expanduser(
            f"~/.hermes/profiles/{profile}/lcm.db"
        )

    def _hermes(self, args: list[str], prompt: str) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        if self.threshold is not None:
            env["LCM_CONTEXT_THRESHOLD"] = str(self.threshold)
        cmd = ["hermes", "-p", self.profile, "chat", "-Q"]
        if self.model:
            cmd.extend(["-m", self.model])
        if self.provider:
            cmd.extend(["--provider", self.provider])
        cmd.extend([*args, "-q", prompt])
        return subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=self.timeout_seconds,
            env=env, check=False,
        )

    @staticmethod
    def _session_id(stdout: str) -> str | None:
        m = re.findall(r"session_id:\s*([0-9A-Za-z_]+)", stdout or "")
        return m[-1] if m else None

    def _grep_called_since(self, since_ts: float) -> list[dict[str, Any]]:
        """Read lcm.db for real lcm_* tool messages written after since_ts."""
        calls: list[dict[str, Any]] = []
        try:
            import sqlite3
            conn = sqlite3.connect(self.lcm_db)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT tool_name, content, timestamp FROM messages "
                "WHERE role='tool' AND tool_name LIKE 'lcm_%' AND timestamp >= ? "
                "ORDER BY timestamp DESC LIMIT 20",
                (since_ts,),
            ).fetchall()
            conn.close()
            for r in rows:
                name = r["tool_name"]
                if name in ("lcm_grep", "lcm_expand", "lcm_expand_query", "lcm_load_session"):
                    calls.append({"name": name, "arguments": (r["content"] or "")[:200]})
        except Exception:
            pass
        return calls

    def run_trial(self, fixture: TrialFixture, sampling: SamplingParams) -> DriverResponse:
        import time
        filler = ("Routine deterministic project chatter for compaction. "
                  * max(1, self.filler_tokens // 8))
        # 1. plant
        plant = self._hermes(
            ["--pass-session-id"],
            f"Remember this exact fact for later, verbatim: {fixture.buried_fact} "
            f"Reply with only OK.",
        )
        sid = self._session_id((plant.stdout or '') + (plant.stderr or ''))
        if not sid:
            return DriverResponse(answer="", tool_calls=[], usage={},
                                  confidence_wrong=False)
        # 2. filler turns to age the fact out + trigger compaction
        for i in range(self.filler_turns):
            _r = self._hermes(
                ["--resume", sid],
                f"Status turn {i}, reply 'ack {i}' only. Context notes: {filler}",
            )
            sid = self._session_id((_r.stdout or "") + (_r.stderr or "")) or sid
        # 3. recovery (mark time so we only count tool calls from THIS step)
        t0 = time.time() - 1.0
        rec = self._hermes(
            ["--resume", sid],
            "Earlier I gave you a fact that is no longer in your active context. "
            "Use your LCM retrieval tools (lcm_grep / lcm_load_session / lcm_expand) "
            "to search the stored conversation and recover it. " + fixture.question,
        )
        answer = _clean_cli_answer(rec.stdout or "")
        if rec.returncode != 0:
            answer = answer or _clean_cli_answer(rec.stderr or "")
        # 4. authoritative tool-call detection from the store
        tool_calls = self._grep_called_since(t0)
        prompt_tokens = estimate_messages_tokens(
            fixture.messages + [{"role": "user", "content": fixture.question}]
        )
        completion_tokens = max(16, estimate_text_tokens(answer))
        return DriverResponse(
            answer=answer,
            tool_calls=tool_calls,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            confidence_wrong=False,
        )


class LexicalJudge:
    """Deterministic planted judge used for dry-run and calibration."""

    def contains_expected(self, answer: str, expected: list[str]) -> bool:
        haystack = _normalize(answer)
        return any(_normalize(item) in haystack for item in expected if item)


def _cost_usd(prompt_tokens: int, completion_tokens: int, input_rate: float, output_rate: float) -> float:
    return (prompt_tokens / 1_000_000.0 * input_rate) + (completion_tokens / 1_000_000.0 * output_rate)


def estimate_text_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def estimate_messages_tokens(messages: Iterable[dict[str, str]]) -> int:
    total_chars = 0
    for message in messages:
        total_chars += len(message.get("role", "")) + len(message.get("content", "")) + 8
    return max(1, math.ceil(total_chars / 4))


def wilson_stats(successes: int, total: int, z: float = WILSON_Z) -> WilsonStats:
    if total <= 0:
        return WilsonStats(successes, total, z, 0.0, 1.0, 0.0, 0.0, 0.0)
    point = successes / total
    z2 = z * z
    denominator = 1.0 + (z2 / total)
    centre = point + (z2 / (2 * total))
    margin = z * math.sqrt(((point * (1.0 - point)) + (z2 / (4 * total))) / total)
    lower = (centre - margin) / denominator
    return WilsonStats(successes, total, z, point, denominator, centre, margin, lower)


def wilson_lower_bound(successes: int, total: int, z: float = WILSON_Z) -> float:
    return wilson_stats(successes, total, z).lower


def validate_run_config(*, mode: str, n: int, allow_underpowered_live: bool = False) -> None:
    if mode not in {"dry-run", "live"}:
        raise ValueError(f"mode must be 'dry-run' or 'live', got {mode!r}")
    if n <= 0:
        raise ValueError("n must be positive")
    if mode == "live" and n < LIVE_MIN_TRIALS and not allow_underpowered_live:
        raise ValueError(f"live Phase-2 gate requires N=180 minimum; got n={n}")


def make_fixtures(n: int, *, seed: int, probe_kind: str = "mixed") -> list[TrialFixture]:
    """Build N recovery fixtures.

    probe_kind:
      - "exact":    all exact-string sentinel probes (the raw-store / FTS path's
                    real contract). Use this for the Arm-A gate run.
      - "semantic": all owner-name semantic probes (recovery of meaning). The
                    DAG / Arm-B owns this contract.
      - "mixed":    legacy 2/3-exact, 1/3-semantic interleave (cross-contaminated;
                    kept only for backward compat, NOT for a gate run).
    """
    rng = random.Random(seed)
    if probe_kind == "exact":
        arms = ["exact"] * n
    elif probe_kind == "semantic":
        arms = ["semantic"] * n
    elif probe_kind == "mixed":
        arms = ["semantic" if i % 3 == 1 else "exact" for i in range(n)]
    else:
        raise ValueError(f"probe_kind must be exact|semantic|mixed, got {probe_kind!r}")
    fixtures = [_make_fixture(index=i, arm=arms[i], seed=seed) for i in range(n)]
    rng.shuffle(fixtures)
    return fixtures


def _make_fixture(*, index: int, arm: str, seed: int) -> TrialFixture:
    prompt_id = f"{arm}-{seed}-{index:03d}"
    if arm == "semantic":
        names = ["Ada Lovelace", "Grace Hopper", "Katherine Johnson", "Margaret Hamilton"]
        owner = names[index % len(names)]
        handoff = f"recover-{seed}-{index:03d}"
        buried_fact = f"The recovery owner is {owner}; the handoff phrase is {handoff}."
        expected = owner
        accept = [owner, owner.split()[0], handoff]
        question = "After compaction, which recovery owner was planted in the buried fact?"
    else:
        sentinel = f"LCM-LIVE-RECOVERY-{seed}-{index:03d}"
        buried_fact = f"The exact recovery sentinel is {sentinel}."
        expected = sentinel
        accept = [sentinel]
        question = "After compaction, what is the exact recovery sentinel? Return only the sentinel."
    messages = build_long_conversation(prompt_id=prompt_id, buried_fact=buried_fact)
    return TrialFixture(
        prompt_id=prompt_id,
        arm=arm,
        buried_fact=buried_fact,
        expected_answer=expected,
        semantic_accept=accept,
        messages=messages,
        question=question,
    )


def build_long_conversation(*, prompt_id: str, buried_fact: str) -> list[dict[str, str]]:
    filler = (
        "This is deterministic compaction filler for PRD-6 recovery trials. "
        "It contains ordinary project chatter, status updates, and repeated low-value text. "
        "The planted fact is intentionally surrounded by irrelevant turns so the active tail alone is insufficient. "
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": "You are Aegis validating LCM live recovery."}]
    for idx in range(18):
        messages.append({"role": "user", "content": f"{prompt_id} pre-bury filler {idx}: {filler * 2}"})
        messages.append({"role": "assistant", "content": f"ack pre-bury filler {idx}"})
    messages.append({"role": "user", "content": buried_fact})
    messages.append({"role": "assistant", "content": "Recorded for later retrieval."})
    for idx in range(28):
        messages.append({"role": "user", "content": f"{prompt_id} post-bury filler {idx}: {filler * 2}"})
        messages.append({"role": "assistant", "content": f"ack post-bury filler {idx}"})
    return messages


def render_trial_prompt(fixture: TrialFixture) -> str:
    lines = [
        "You are running the PRD-6 LCM live recovery harness.",
        "Use available long-context memory/retrieval tools if the buried fact is no longer in the active tail.",
        "Sampling contract: temperature=0 where supported; seed recorded by harness where supported.",
        "",
        "Transcript follows:",
    ]
    for message in fixture.messages:
        lines.append(f"[{message['role']}] {message['content']}")
    lines.extend(["", "Final recovery question:", fixture.question])
    return "\n".join(lines)


def calibrate_judge(
    judge: Any,
    *,
    precision_min: float = DEFAULT_JUDGE_MIN,
    recall_min: float = DEFAULT_JUDGE_MIN,
) -> JudgeCalibrationResult:
    cases = [
        ("The recovery owner is Ada Lovelace.", ["Ada Lovelace"], True),
        ("The fallback duration was 48 hours.", ["48 hours", "two days"], True),
        ("The recovery owner is Grace Hopper.", ["Ada Lovelace"], False),
        ("I cannot recover that fact from the transcript.", ["48 hours"], False),
    ]
    tp = fp = fn = tn = 0
    for answer, expected, label in cases:
        predicted = bool(judge.contains_expected(answer, expected))
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif (not predicted) and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    passed = precision >= precision_min and recall >= recall_min
    details = (
        f"tp={tp} fp={fp} fn={fn} tn={tn}; "
        f"precision={precision:.3f} recall={recall:.3f}; "
        f"required precision>={precision_min:.2f} recall>={recall_min:.2f}"
    )
    return JudgeCalibrationResult(precision, recall, passed, details, tp, fp, fn, tn)


def score_response(fixture: TrialFixture, response: DriverResponse, judge: Any) -> TrialRecord:
    if fixture.arm == "exact":
        correct = fixture.expected_answer in response.answer
    else:
        correct = bool(judge.contains_expected(response.answer, fixture.semantic_accept))
    confidence_wrong = response.confidence_wrong
    if confidence_wrong is None:
        confidence_wrong = detect_confident_wrong(response.answer, correct)
    return TrialRecord(
        prompt_id=fixture.prompt_id,
        arm=fixture.arm,
        buried_fact=fixture.buried_fact,
        tool_calls=response.tool_calls,
        answer=response.answer,
        correct=correct,
        confidence_wrong=bool(confidence_wrong),
        spend=SpendRecord(),
    )


def detect_confident_wrong(answer: str, correct: bool) -> bool:
    if correct:
        return False
    normalized = _normalize(answer)
    uncertainty = ("not sure", "cannot", "can't", "unable", "unknown", "do not know", "don't know")
    if any(marker in normalized for marker in uncertainty):
        return False
    confidence = ("definitely", "certainly", "the answer is", "it is", "must be", "clearly")
    return any(marker in normalized for marker in confidence) or len(normalized.split()) <= 8


def evaluate_gate(
    trials: list[TrialRecord],
    *,
    thresholds: GateThresholds,
    mode: str,
    semantic_calibration: JudgeCalibrationResult | None,
) -> GateResult:
    total = len(trials)
    correct = sum(1 for trial in trials if trial.correct)
    stats = wilson_stats(correct, total)
    confident_wrong = sum(1 for trial in trials if trial.confidence_wrong)
    missing_tool_calls = sum(1 for trial in trials if not trial.tool_calls)
    failures: list[str] = []
    if semantic_calibration is not None and not semantic_calibration.passed:
        failures.append(f"judge calibration failed before semantic scoring: {semantic_calibration.details}")
    if total == 0:
        failures.append("no trials scored")
    if mode == "live" and total < thresholds.min_trials:
        failures.append(f"live Phase-2 gate requires N={thresholds.min_trials} minimum; scored {total}")
    if total and stats.point < thresholds.recall_point_min:
        failures.append(
            f"point recall {stats.point:.4f} below required {thresholds.recall_point_min:.4f}"
        )
    if total and stats.lower < thresholds.wilson_lower_min:
        failures.append(
            f"Wilson lower bound {stats.lower:.4f} below required {thresholds.wilson_lower_min:.4f}"
        )
    if confident_wrong:
        failures.append(f"observed confident-wrong count must be zero; saw {confident_wrong}")
    if thresholds.require_tool_call_evidence and missing_tool_calls:
        failures.append(f"missing tool-call evidence on {missing_tool_calls} trial(s)")
    return GateResult(
        passed=not failures,
        failures=failures,
        total_trials=total,
        correct_trials=correct,
        point_recall=stats.point,
        wilson_lower=stats.lower,
        confident_wrong=confident_wrong,
        missing_tool_call_evidence=missing_tool_calls,
        semantic_calibration=semantic_calibration,
    )


def run_recovery_gate(
    *,
    mode: str,
    n: int,
    out_path: Path | None,
    seed: int = 1729,
    sampling: SamplingParams | None = None,
    thresholds: GateThresholds | None = None,
    budget: BudgetPolicy | None = None,
    driver: RecoveryDriver | None = None,
    judge: Any | None = None,
    allow_underpowered_live: bool = False,
    probe_kind: str = "mixed",
) -> RecoveryRun:
    validate_run_config(mode=mode, n=n, allow_underpowered_live=allow_underpowered_live)
    sampling = sampling or SamplingParams(seed=seed)
    if thresholds is None:
        thresholds = GateThresholds(min_trials=1, wilson_lower_min=0.0) if mode == "dry-run" else GateThresholds()
    budget = budget or BudgetPolicy()
    judge = judge or LexicalJudge()
    if driver is None:
        driver = OfflineStubDriver() if mode == "dry-run" else LiveAegisDriver()

    fixtures = make_fixtures(n, seed=seed, probe_kind=probe_kind)
    has_semantic = any(fixture.arm == "semantic" for fixture in fixtures)
    calibration: JudgeCalibrationResult | None = None
    if has_semantic:
        calibration = calibrate_judge(
            judge,
            precision_min=thresholds.judge_precision_min,
            recall_min=thresholds.judge_recall_min,
        )
        if not calibration.passed:
            gate = evaluate_gate([], thresholds=thresholds, mode=mode, semantic_calibration=calibration)
            run = RecoveryRun(mode, n, sampling, thresholds, budget, [], gate, calibration, 0.0, 0.0, None, out_path)
            if out_path is not None:
                write_report(out_path, run)
            return run

    trials: list[TrialRecord] = []
    estimated_spend = 0.0
    observed_spend = 0.0
    aborted_reason: str | None = None
    for fixture in fixtures:
        prompt_tokens = estimate_messages_tokens(fixture.messages + [{"role": "user", "content": fixture.question}])
        estimated_completion = budget.max_completion_tokens
        estimated_cost = budget.estimated_cost(prompt_tokens, estimated_completion)
        committed_spend = max(estimated_spend, observed_spend)
        if committed_spend + estimated_cost > budget.max_usd:
            aborted_reason = (
                f"budget cap preflight abort: committed ${committed_spend:.6f} + "
                f"next estimated ${estimated_cost:.6f} would exceed cap ${budget.max_usd:.6f}"
            )
            break

        response = driver.run_trial(fixture, sampling)
        observed_prompt = int(response.usage.get("prompt_tokens", 0) or 0)
        observed_completion = int(response.usage.get("completion_tokens", 0) or 0)
        observed_cost = budget.observed_cost(observed_prompt, observed_completion)
        estimated_spend += estimated_cost
        observed_spend += observed_cost
        scored = score_response(fixture, response, judge)
        trials.append(
            TrialRecord(
                prompt_id=scored.prompt_id,
                arm=scored.arm,
                buried_fact=scored.buried_fact,
                tool_calls=scored.tool_calls,
                answer=scored.answer,
                correct=scored.correct,
                confidence_wrong=scored.confidence_wrong,
                spend=SpendRecord(
                    estimated_prompt_tokens=prompt_tokens,
                    estimated_completion_tokens=estimated_completion,
                    observed_prompt_tokens=observed_prompt,
                    observed_completion_tokens=observed_completion,
                    estimated_cost_usd=estimated_cost,
                    observed_cost_usd=observed_cost,
                ),
            )
        )

    gate = evaluate_gate(trials, thresholds=thresholds, mode=mode, semantic_calibration=calibration)
    if aborted_reason is not None:
        gate = GateResult(
            passed=False,
            failures=[*gate.failures, aborted_reason],
            total_trials=gate.total_trials,
            correct_trials=gate.correct_trials,
            point_recall=gate.point_recall,
            wilson_lower=gate.wilson_lower,
            confident_wrong=gate.confident_wrong,
            missing_tool_call_evidence=gate.missing_tool_call_evidence,
            semantic_calibration=gate.semantic_calibration,
        )
    run = RecoveryRun(
        mode=mode,
        n_requested=n,
        sampling=sampling,
        thresholds=thresholds,
        budget=budget,
        trials=trials,
        gate=gate,
        semantic_calibration=calibration,
        estimated_spend_usd=estimated_spend,
        observed_spend_usd=observed_spend,
        aborted_reason=aborted_reason,
        out_path=out_path,
    )
    if out_path is not None:
        write_report(out_path, run)
    return run


def write_report(out_path: Path, run: RecoveryRun) -> None:
    stats = wilson_stats(run.gate.correct_trials, run.gate.total_trials)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict = "GO" if run.gate.passed else "BLOCKED"
    lines = [
        "# PRD-6 Phase-2 LCM Live Recovery Gate",
        "",
        f"Generated: {ts}",
        f"Mode: {run.mode}",
        f"Verdict: {verdict}",
        f"Trials requested: {run.n_requested}; trials scored: {run.gate.total_trials}",
        f"Sampling: temperature={run.sampling.temperature!r}, seed={run.sampling.seed!r}; fixture order shuffled by seed",
        "",
        "## Gate summary",
        "",
        f"- Correct: {run.gate.correct_trials}/{run.gate.total_trials}",
        f"- Point recall: {run.gate.point_recall:.6f} (required >= {run.thresholds.recall_point_min:.3f})",
        f"- Wilson 95% lower bound: {run.gate.wilson_lower:.6f} (required >= {run.thresholds.wilson_lower_min:.3f})",
        f"- Confident-wrong: {run.gate.confident_wrong} (required 0)",
        f"- Missing tool-call evidence: {run.gate.missing_tool_call_evidence}",
        f"- estimated spend: ${run.estimated_spend_usd:.6f} / cap ${run.budget.max_usd:.6f}",
        f"- observed spend: ${run.observed_spend_usd:.6f} / cap ${run.budget.max_usd:.6f}",
        "",
        "## Wilson arithmetic",
        "",
        f"Arithmetic: successes={stats.successes}, n={stats.total}, z={stats.z:.2f}, phat=successes/n={stats.point:.6f}",
        f"denominator = 1 + z^2/n = {stats.denominator:.6f}",
        f"centre = phat + z^2/(2n) = {stats.centre:.6f}",
        f"margin = z * sqrt((phat*(1-phat) + z^2/(4n))/n) = {stats.margin:.6f}",
        f"lower = (centre - margin) / denominator = {stats.lower:.6f}",
        "",
        "## Judge calibration",
        "",
    ]
    if run.semantic_calibration is None:
        lines.append("- Semantic arms absent; judge calibration not required.")
    else:
        lines.extend([
            f"- precision: {run.semantic_calibration.precision:.6f} (required >= {run.thresholds.judge_precision_min:.3f})",
            f"- recall: {run.semantic_calibration.recall:.6f} (required >= {run.thresholds.judge_recall_min:.3f})",
            f"- passed: {run.semantic_calibration.passed}",
            f"- details: {run.semantic_calibration.details}",
        ])
    lines.extend(["", "## Failures", ""])
    if run.gate.failures:
        lines.extend(f"- {failure}" for failure in run.gate.failures)
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Trial records",
        "",
        "| prompt_id | buried_fact | tool_calls | answer | correct | confidence_wrong | estimated spend | observed spend |",
        "|---|---|---|---|---|---|---:|---:|",
    ])
    for trial in run.trials:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(trial.prompt_id),
                    _md(trial.buried_fact),
                    _md(json.dumps(trial.tool_calls, sort_keys=True)[:500]),
                    _md(trial.answer[:500]),
                    str(trial.correct),
                    str(trial.confidence_wrong),
                    f"${trial.spend.estimated_cost_usd:.6f}",
                    f"${trial.spend.observed_cost_usd:.6f}",
                ]
            )
            + " |"
        )
    lines.extend([
        "",
        "## Harness notes",
        "",
        "- Dry-run mode uses stubbed transcripts and responses; it exercises scoring, report, Wilson, judge, and budget gates without live spend.",
        "- Live mode does not restart gateways or flip configs; it assumes Apollo has activated the Aegis profile and only invokes the configured Aegis command.",
        "- Live mode records per-trial prompt id, buried fact, tool calls, answer, correctness, confidence-wrong adjudication, and spend.",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _extract_answer(output: str) -> str:
    stripped = output.strip()
    parsed = _extract_first_json_object(stripped)
    if isinstance(parsed, dict):
        for key in ("answer", "final_response", "response", "content"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return stripped


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    parsed = _extract_first_json_object(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
        return [item for item in parsed["tool_calls"] if isinstance(item, dict)]
    calls: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("HERMES_TOOL_CALLS_JSON="):
            try:
                data = json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                calls.extend(item for item in data if isinstance(item, dict))
    return calls


def _extract_usage(text: str) -> dict[str, int]:
    parsed = _extract_first_json_object(text)
    if isinstance(parsed, dict):
        usage = parsed.get("usage")
        if isinstance(usage, dict):
            return {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            }
    for line in text.splitlines():
        if line.startswith("HERMES_USAGE_JSON="):
            try:
                data = json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return {
                    "prompt_tokens": int(data.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(data.get("completion_tokens", 0) or 0),
                }
    return {}


def _extract_first_json_object(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Run deterministic offline stub mode.")
    mode.add_argument("--live", action="store_true", help="Run live Aegis subprocess mode.")
    parser.add_argument("--n", type=int, default=LIVE_MIN_TRIALS, help="Number of recovery trials.")
    parser.add_argument("--out", required=True, help="Markdown report path.")
    parser.add_argument("--seed", type=int, default=1729, help="Fixture shuffle seed and recorded sampling seed.")
    parser.add_argument("--budget-usd", type=float, default=float(os.environ.get("LCM_LIVE_RECOVERY_BUDGET_USD", DEFAULT_BUDGET_USD)))
    parser.add_argument("--input-usd-per-mtok", type=float, default=DEFAULT_INPUT_USD_PER_MTOK)
    parser.add_argument("--output-usd-per-mtok", type=float, default=DEFAULT_OUTPUT_USD_PER_MTOK)
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--aegis-command", help="Live command prefix; prompt is appended as final argv item.")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--recall-point-min", type=float, default=DEFAULT_RECALL_POINT_MIN)
    parser.add_argument("--wilson-lower-min", type=float, default=None)
    parser.add_argument("--judge-precision-min", type=float, default=DEFAULT_JUDGE_MIN)
    parser.add_argument("--judge-recall-min", type=float, default=DEFAULT_JUDGE_MIN)
    parser.add_argument("--no-tool-call-required", action="store_true", help="Record tool calls but do not fail the gate when they are absent.")
    parser.add_argument("--session-mode", action="store_true", help="Use the persistent multi-turn LiveAegisSessionDriver (plant->filler->recover against the real LCM store) instead of the inline single-prompt driver.")
    parser.add_argument("--profile", default="aegis", help="Hermes profile to drive in --session-mode.")
    parser.add_argument("--model", help="Model override for each Hermes chat call in --session-mode (e.g. claude-haiku-4-5-20251001).")
    parser.add_argument("--provider", help="Provider override for each Hermes chat call in --session-mode.")
    parser.add_argument("--lcm-threshold", type=float, default=0.02, help="LCM_CONTEXT_THRESHOLD override for --session-mode filler turns.")
    parser.add_argument("--natural-threshold", action="store_true", help="Do not override LCM_CONTEXT_THRESHOLD; use the profile/model natural threshold.")
    parser.add_argument("--filler-turns", type=int, default=4, help="Filler turns per trial in --session-mode.")
    parser.add_argument("--filler-tokens", type=int, default=2500, help="Approximate filler-token budget per filler turn in --session-mode.")
    parser.add_argument("--allow-underpowered-live", action="store_true", help="Allow N<180 live runs for E2E shakedown ONLY; not a Phase-2 promotion gate.")
    parser.add_argument("--probe-kind", choices=["exact", "semantic", "mixed"], default="exact", help="Probe contract: 'exact' = raw-store/FTS gate (Arm-A, DEFAULT per PRD-8.1), 'semantic' = DAG/meaning, 'mixed' = legacy interleave (NOT a gate run; explicit opt-in only).")
    parser.add_argument("--lcm-db", help="Path to the lcm.db the session driver reads tool-call evidence from (default: ~/.hermes/profiles/<profile>/lcm.db). NOTE: this only redirects the evidence-READ path; the live gateway still WRITES to the profile db. True write-isolation needs a gateway config change (out of scope per AC-5).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = "live" if args.live else "dry-run" if args.dry_run else "live"
    out_path = Path(args.out).expanduser().resolve()
    try:
        validate_run_config(mode=mode, n=args.n, allow_underpowered_live=args.allow_underpowered_live)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.wilson_lower_min is None:
        wilson_lower_min = DEFAULT_WILSON_LOWER_MIN if mode == "live" else 0.0
    else:
        wilson_lower_min = args.wilson_lower_min
    thresholds = GateThresholds(
        min_trials=args.n if args.allow_underpowered_live and mode == "live" and args.n < LIVE_MIN_TRIALS else LIVE_MIN_TRIALS,
        recall_point_min=args.recall_point_min,
        wilson_lower_min=wilson_lower_min,
        judge_precision_min=args.judge_precision_min,
        judge_recall_min=args.judge_recall_min,
        require_tool_call_evidence=not args.no_tool_call_required,
    )
    budget = BudgetPolicy(
        max_usd=args.budget_usd,
        estimated_input_usd_per_mtok=args.input_usd_per_mtok,
        estimated_output_usd_per_mtok=args.output_usd_per_mtok,
        max_completion_tokens=args.max_completion_tokens,
    )
    command = shlex.split(args.aegis_command) if args.aegis_command else None
    driver: RecoveryDriver | None = None
    if mode == "live":
        if getattr(args, "session_mode", False):
            driver = LiveAegisSessionDriver(
                profile=args.profile,
                timeout_seconds=args.timeout_seconds,
                threshold=None if args.natural_threshold else args.lcm_threshold,
                filler_turns=args.filler_turns,
                filler_tokens=args.filler_tokens,
                model=args.model,
                provider=args.provider,
                lcm_db=args.lcm_db,
            )
        else:
            driver = LiveAegisDriver(command=command, timeout_seconds=args.timeout_seconds)
    run = run_recovery_gate(
        mode=mode,
        n=args.n,
        out_path=out_path,
        seed=args.seed,
        sampling=SamplingParams(temperature=0.0, seed=args.seed),
        thresholds=thresholds,
        budget=budget,
        driver=driver,
        allow_underpowered_live=args.allow_underpowered_live,
        probe_kind=args.probe_kind,
    )
    print(f"{run.gate.correct_trials}/{run.gate.total_trials} correct; Wilson lower={run.gate.wilson_lower:.6f}")
    print(f"estimated spend=${run.estimated_spend_usd:.6f}; observed spend=${run.observed_spend_usd:.6f}; report={out_path}")
    if run.gate.failures:
        for failure in run.gate.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
    return 0 if run.gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
