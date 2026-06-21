"""Deterministic synthesis over structured lane results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .lanes import LaneResult, LaneStatus
from .redaction import redact_text


@dataclass
class LaneDigest:
    lane_id: str
    agent: str
    status: str
    digest: str


@dataclass
class SynthesisResult:
    chosen_lane_id: str | None
    summary: str
    lanes: list[LaneDigest]
    conflicts: list[str]
    confidence: str


def _digest(result: LaneResult, *, max_chars: int = 240) -> str:
    text = result.output if result.output else result.error if result.error else ""
    text = " ".join(redact_text(text).split())
    return text[:max_chars]


def synthesize(results: Sequence[LaneResult]) -> SynthesisResult:
    """Summarize structured lane results without reading terminal scrollback."""

    lanes = [LaneDigest(result.lane_id, result.agent, result.status.value, _digest(result)) for result in results]
    successes = [result for result in results if result.status is LaneStatus.SUCCEEDED]
    chosen = successes[0].lane_id if successes else None

    conflicts: list[str] = []
    success_digests = {_digest(result) for result in successes if _digest(result)}
    if len(success_digests) > 1:
        conflicts.append("successful lanes produced different digests")

    if not successes:
        confidence = "low"
        summary = "성공 lane 없음. 실패/timeout/skipped 결과만 수집됐습니다."
    else:
        non_success = [result for result in results if result.status is not LaneStatus.SUCCEEDED]
        confidence = "high" if not non_success and not conflicts else "medium"
        summary = f"선택 lane: {chosen}. 성공 {len(successes)}개, 비성공 {len(non_success)}개."

    if lanes:
        details = "; ".join(f"{lane.lane_id}={lane.status}" for lane in lanes)
        summary = f"{summary} lanes: {details}"
    if conflicts:
        summary = f"{summary} conflicts: {', '.join(conflicts)}"

    return SynthesisResult(
        chosen_lane_id=chosen,
        summary=summary,
        lanes=lanes,
        conflicts=conflicts,
        confidence=confidence,
    )
