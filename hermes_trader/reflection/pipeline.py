"""Reflection pipeline orchestration."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_trader.config import TraderConfig, load_trader_config
from hermes_trader.memory.episodes import EpisodeStore, TradeEpisode
from hermes_trader.memory.strategic_rules import default_strategic_rules_path
from hermes_trader.reflection.calibration_report import (
    build_calibration_report,
    format_calibration_report_markdown,
    suggest_min_confidence,
)
from hermes_trader.reflection.distill import apply_distillation
from hermes_trader.reflection.judge import (
    JudgeFn,
    JudgeScore,
    ReflectionInput,
    run_judge,
)


@dataclass(frozen=True)
class ReflectionResult:
    episode_id: str
    score: JudgeScore
    rules_updated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "score": self.score.to_dict(),
            "rules_updated": self.rules_updated,
        }


def run_reflection(
    episode_id: str,
    *,
    pnl_usd: float,
    store: Optional[EpisodeStore] = None,
    config: Optional[TraderConfig] = None,
    judge_fn: Optional[JudgeFn] = None,
    llm_response: Optional[str] = None,
    holding_hours: Optional[float] = None,
    ohlcv_summary: Optional[dict[str, Any]] = None,
    outcome_notes: str = "",
    rules_path: Optional[Path | str] = None,
    apply_rules: bool = True,
) -> ReflectionResult:
    """Run judge + optional distillation for a closed episode."""
    cfg = config or load_trader_config()
    episode_store = store or EpisodeStore()
    episode = episode_store.get_episode(episode_id)
    if episode is None:
        raise ValueError(f"episode not found: {episode_id}")

    episode_store.update_outcome(
        episode_id,
        pnl_usd=pnl_usd,
        holding_hours=holding_hours,
    )
    episode = episode_store.get_episode(episode_id)
    assert episode is not None

    reflection_input = ReflectionInput(
        episode=episode,
        pnl_usd=pnl_usd,
        holding_hours=holding_hours,
        ohlcv_summary=ohlcv_summary,
        outcome_notes=outcome_notes,
    )
    score = run_judge(reflection_input, judge_fn=judge_fn, llm_response=llm_response)
    episode_store.save_reflection(episode_id, score.to_dict())

    rules_updated = False
    if apply_rules:
        _rules, rules_updated = apply_distillation(
            episode,
            score,
            cfg,
            rules_path=rules_path or default_strategic_rules_path(),
        )

    return ReflectionResult(
        episode_id=episode_id,
        score=score,
        rules_updated=rules_updated,
    )


def _reflections_from_episodes(episodes: list[TradeEpisode]) -> dict[str, JudgeScore]:
    out: dict[str, JudgeScore] = {}
    for ep in episodes:
        if ep.reflection:
            out[ep.episode_id] = JudgeScore.from_mapping(
                ep.reflection,
                source=str(ep.reflection.get("source", "stored")),
            )
    return out


def run_weekly_calibration(
    *,
    store: Optional[EpisodeStore] = None,
    config: Optional[TraderConfig] = None,
) -> str:
    """Build markdown calibration report from closed episodes."""
    cfg = config or load_trader_config()
    episode_store = store or EpisodeStore()
    closed = episode_store.list_closed_episodes(limit=500)
    reflections = _reflections_from_episodes(closed)
    report = build_calibration_report(closed, reflections=reflections)
    suggested = suggest_min_confidence(
        report,
        current_min=cfg.min_confidence,
        miscalibration_gap=cfg.calibration_miscalibration_gap,
    )
    return format_calibration_report_markdown(report, suggested_min_confidence=suggested)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes trader reflection pipeline")
    parser.add_argument("--weekly", action="store_true", help="Print weekly calibration report")
    parser.add_argument("--episode-id", help="Run reflection for one episode")
    parser.add_argument("--pnl-usd", type=float, default=0.0)
    parser.add_argument("--holding-hours", type=float, default=None)
    args = parser.parse_args(argv)

    if args.weekly:
        print(run_weekly_calibration())
        return 0

    if args.episode_id:
        result = run_reflection(
            args.episode_id,
            pnl_usd=args.pnl_usd,
            holding_hours=args.holding_hours,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())