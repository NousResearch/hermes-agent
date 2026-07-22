"""Confidence calibration report and weekly cron helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from hermes_trader.config import TraderConfig, load_trader_config
from hermes_trader.memory.episodes import TradeEpisode
from hermes_trader.reflection.judge import JudgeScore

CONFIDENCE_BUCKETS: list[tuple[float, float, str]] = [
    (0.5, 0.6, "0.5–0.6"),
    (0.6, 0.7, "0.6–0.7"),
    (0.7, 0.8, "0.7–0.8"),
    (0.8, 0.9, "0.8–0.9"),
    (0.9, 1.01, "0.9–1.0"),
]


@dataclass(frozen=True)
class BucketStats:
    bucket: str
    trades: int
    win_rate: float
    avg_pnl: float
    judge_avg: Optional[float]
    avg_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "trades": self.trades,
            "win_rate": self.win_rate,
            "avg_pnl": self.avg_pnl,
            "judge_avg": self.judge_avg,
            "avg_confidence": self.avg_confidence,
        }


def confidence_bucket(confidence: float) -> Optional[str]:
    for low, high, label in CONFIDENCE_BUCKETS:
        if low <= confidence < high:
            return label
    return None


def build_calibration_report(
    episodes: Iterable[TradeEpisode],
    *,
    reflections: Optional[dict[str, JudgeScore]] = None,
) -> List[BucketStats]:
    """Aggregate closed episodes by confidence bucket."""
    reflections = reflections or {}
    grouped: dict[str, list[TradeEpisode]] = {label: [] for *_l, label in CONFIDENCE_BUCKETS}

    for episode in episodes:
        if episode.pnl_usd is None:
            continue
        confidence = float((episode.intent or {}).get("confidence") or 0.0)
        label = confidence_bucket(confidence)
        if label:
            grouped[label].append(episode)

    rows: list[BucketStats] = []
    for low, high, label in CONFIDENCE_BUCKETS:
        items = grouped[label]
        if not items:
            continue
        wins = sum(1 for ep in items if (ep.pnl_usd or 0) > 0)
        pnls = [float(ep.pnl_usd or 0) for ep in items]
        judge_scores = [
            reflections[ep.episode_id].overall
            for ep in items
            if ep.episode_id in reflections
        ]
        confidences = [float((ep.intent or {}).get("confidence") or 0.0) for ep in items]
        rows.append(
            BucketStats(
                bucket=label,
                trades=len(items),
                win_rate=wins / len(items),
                avg_pnl=sum(pnls) / len(pnls),
                judge_avg=(sum(judge_scores) / len(judge_scores)) if judge_scores else None,
                avg_confidence=sum(confidences) / len(confidences),
            )
        )
    return rows


def suggest_min_confidence(
    report: List[BucketStats],
    *,
    current_min: float = 0.6,
    miscalibration_gap: float = 0.15,
) -> Optional[float]:
    """Tighten min_confidence when win rate trails stated confidence."""
    suggestion: Optional[float] = None
    for row in report:
        if row.trades < 3:
            continue
        if row.win_rate < row.avg_confidence - miscalibration_gap:
            candidate = min(0.95, row.avg_confidence + 0.05)
            suggestion = max(suggestion or current_min, candidate)
    if suggestion and suggestion > current_min:
        return round(suggestion, 2)
    return None


def format_calibration_report_markdown(
    report: List[BucketStats],
    *,
    suggested_min_confidence: Optional[float] = None,
) -> str:
    lines = [
        "# Hermes Trader — Confidence Calibration",
        "",
        "| Confidence bucket | Trades | Win rate | Avg PnL | Judge avg |",
        "|-------------------|--------|----------|---------|-----------|",
    ]
    for row in report:
        judge = f"{row.judge_avg:.1f}" if row.judge_avg is not None else "—"
        lines.append(
            f"| {row.bucket} | {row.trades} | {row.win_rate:.0%} | "
            f"${row.avg_pnl:+.2f} | {judge} |"
        )
    if suggested_min_confidence is not None:
        lines.extend(
            [
                "",
                f"Suggested `min_confidence`: **{suggested_min_confidence:.2f}** "
                "(win rate trailed confidence in one or more buckets).",
            ]
        )
    return "\n".join(lines)


def build_weekly_reflection_cron_spec(
    config: Optional[TraderConfig] = None,
    *,
    job_id: str = "trader-reflection-weekly",
) -> dict[str, Any]:
    """Cron kwargs for weekly calibration + reflection summary."""
    cfg = config or load_trader_config()
    return {
        "prompt": (
            "Run weekly Hermes Agentic Trader reflection: "
            "execute `python -m hermes_trader.reflection.pipeline --weekly` "
            "and deliver the calibration markdown summary."
        ),
        "schedule": "0 9 * * 1",
        "name": job_id,
        "skill": "hermes-agentic-trader",
        "repeat": None,
        "deliver": "local",
        "script": None,
    }