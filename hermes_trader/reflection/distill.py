"""Distill reflection outcomes into strategic_rules.yaml."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from hermes_trader.config import TraderConfig
from hermes_trader.memory.episodes import TradeEpisode
from hermes_trader.memory.strategic_rules import (
    StrategicRules,
    default_strategic_rules_path,
    load_strategic_rules,
    save_strategic_rules,
)
from hermes_trader.reflection.judge import JudgeScore


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:48] or uuid.uuid4().hex[:8]


def should_add_negative_constraint(
    score: JudgeScore,
    pnl_usd: float,
    *,
    loss_threshold_usd: float,
) -> bool:
    return score.overall < 3.0 or pnl_usd <= -abs(loss_threshold_usd)


def should_add_positive_heuristic(score: JudgeScore, pnl_usd: float) -> bool:
    return score.overall >= 4.0 and pnl_usd > 0


def _rule_exists(rules: list[dict[str, str]], rule_id: str) -> bool:
    return any(item.get("id") == rule_id for item in rules)


def apply_distillation(
    episode: TradeEpisode,
    score: JudgeScore,
    config: TraderConfig,
    *,
    rules_path: Optional[Path | str] = None,
) -> tuple[StrategicRules, bool]:
    """Update strategic rules from judge outcome. Returns (rules, changed)."""
    path = Path(rules_path) if rules_path is not None else default_strategic_rules_path()
    rules = load_strategic_rules(path if path.is_file() else None)
    pnl = float(episode.pnl_usd if episode.pnl_usd is not None else 0.0)
    changed = False
    tag = episode.strategy_tag or "general"
    intent = episode.intent or {}

    if should_add_negative_constraint(
        score,
        pnl,
        loss_threshold_usd=config.reflection_loss_threshold_usd,
    ):
        rule_id = f"neg_{_slug(tag)}_{episode.episode_id[:8]}"
        lesson = score.lessons[0] if score.lessons else intent.get("reasoning", "Poor decision quality")
        entry = {
            "id": rule_id,
            "rule": (
                f"Avoid repeating '{tag}' setups like {episode.token_address[:10]}… "
                f"when judge overall={score.overall:.1f} (pnl=${pnl:.2f}): {lesson}"
            ),
        }
        if not _rule_exists(rules.negative_constraints, rule_id):
            rules.negative_constraints.append(entry)
            changed = True

    if should_add_positive_heuristic(score, pnl):
        rule_id = f"pos_{_slug(tag)}_{episode.episode_id[:8]}"
        lesson = score.lessons[0] if score.lessons else intent.get("reasoning", "Strong decision quality")
        entry = {
            "id": rule_id,
            "rule": (
                f"Favor '{tag}' pattern on {episode.chain} when judge overall={score.overall:.1f}: "
                f"{lesson}"
            ),
        }
        if not _rule_exists(rules.positive_heuristics, rule_id):
            rules.positive_heuristics.append(entry)
            changed = True

    if changed:
        rules.updated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        save_strategic_rules(rules, path)

    return rules, changed