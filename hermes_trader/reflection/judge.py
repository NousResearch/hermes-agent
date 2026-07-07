"""LLM-as-Judge — scores decision quality independent of outcome."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from hermes_trader.memory.episodes import TradeEpisode

JudgeFn = Callable[["ReflectionInput"], "JudgeScore"]

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


@dataclass
class ReflectionInput:
    episode: TradeEpisode
    pnl_usd: float
    holding_hours: Optional[float] = None
    ohlcv_summary: Optional[dict[str, Any]] = None
    outcome_notes: str = ""


@dataclass
class JudgeScore:
    thesis: float
    timing: float
    sizing: float
    execution: float
    overall: float
    lessons: List[str] = field(default_factory=list)
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "thesis": self.thesis,
            "timing": self.timing,
            "sizing": self.sizing,
            "execution": self.execution,
            "overall": self.overall,
            "lessons": list(self.lessons),
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, source: str = "llm") -> "JudgeScore":
        return cls(
            thesis=_clamp_score(data.get("thesis")),
            timing=_clamp_score(data.get("timing")),
            sizing=_clamp_score(data.get("sizing")),
            execution=_clamp_score(data.get("execution")),
            overall=_clamp_score(data.get("overall")),
            lessons=[str(x) for x in (data.get("lessons") or []) if str(x).strip()],
            source=source,
        )


def _clamp_score(value: Any, default: float = 3.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(1.0, min(5.0, score))


def _prompt_template() -> str:
    path = (
        Path(__file__).resolve().parents[2]
        / "optional-skills"
        / "trading"
        / "hermes-agentic-trader"
        / "prompts"
        / "reflect.md"
    )
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return (
        "You are a trading decision auditor. Score the ORIGINAL intent, not the outcome.\n"
        "Output JSON only: "
        '{"thesis":1-5,"timing":1-5,"sizing":1-5,"execution":1-5,"overall":1-5,"lessons":[]}\n'
        "Intent: ${intent_json}\nOutcome pnl_usd: ${pnl_usd}\n"
    )


def build_judge_prompt(reflection: ReflectionInput) -> str:
    from string import Template

    intent = reflection.episode.intent or {}
    return Template(_prompt_template()).safe_substitute(
        intent_json=json.dumps(intent, ensure_ascii=False),
        gate_decision=reflection.episode.gate_decision,
        gate_reason=reflection.episode.gate_reason or "",
        pnl_usd=reflection.pnl_usd,
        holding_hours=reflection.holding_hours or "",
        ohlcv_json=json.dumps(reflection.ohlcv_summary or {}, ensure_ascii=False),
        outcome_notes=reflection.outcome_notes,
        reasoning=intent.get("reasoning", ""),
        confidence=intent.get("confidence", ""),
    )


def parse_judge_response(payload: str | dict[str, Any]) -> JudgeScore:
    if isinstance(payload, dict):
        return JudgeScore.from_mapping(payload, source="llm")
    text = str(payload).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        block = _JSON_BLOCK_RE.search(text)
        if not block:
            raise ValueError("no JSON judge scores found in response")
        data = json.loads(block.group(1))
    if not isinstance(data, dict):
        raise ValueError("judge response must be a JSON object")
    return JudgeScore.from_mapping(data, source="llm")


def heuristic_judge(reflection: ReflectionInput) -> JudgeScore:
    """Deterministic judge for tests and offline reflection."""
    intent = reflection.episode.intent or {}
    confidence = float(intent.get("confidence") or 0.0)
    liquidity = float(intent.get("pool_liquidity_usd") or reflection.episode.liquidity_usd or 0.0)
    pnl = reflection.pnl_usd

    thesis = 3.0
    if intent.get("reasoning"):
        thesis += 0.5
    if reflection.episode.gate_decision == "APPROVE":
        thesis += 0.5
    else:
        thesis -= 0.5

    timing = 3.0
    if pnl > 0:
        timing += 0.5
    elif pnl < 0:
        timing -= 0.5

    sizing = 3.0
    size_usd = float(intent.get("size_usd") or 0.0)
    if 0 < size_usd <= 200:
        sizing += 0.5
    if confidence >= 0.7:
        sizing += 0.25

    execution = 3.5 if reflection.episode.tx_hash else 3.0
    if reflection.episode.gate_decision == "REJECT":
        execution = 4.0

    overall = (thesis + timing + sizing + execution) / 4.0
    if liquidity < 50_000:
        overall -= 0.5
        lessons = ["Liquidity was below preferred threshold at decision time."]
    elif pnl < 0 and overall < 3:
        lessons = ["Loss with weak thesis — tighten entry criteria."]
    elif pnl > 0 and overall >= 4:
        lessons = ["Repeat sizing discipline from this setup."]
    else:
        lessons = ["Review gate logs and OHLCV context for this token."]

    return JudgeScore(
        thesis=_clamp_score(thesis),
        timing=_clamp_score(timing),
        sizing=_clamp_score(sizing),
        execution=_clamp_score(execution),
        overall=_clamp_score(overall),
        lessons=lessons,
        source="heuristic",
    )


def run_judge(
    reflection: ReflectionInput,
    *,
    judge_fn: Optional[JudgeFn] = None,
    llm_response: Optional[str] = None,
) -> JudgeScore:
    if llm_response is not None:
        return parse_judge_response(llm_response)
    fn = judge_fn or heuristic_judge
    return fn(reflection)