"""Reward shaping for the LLM-as-skill-selector foraging env.

The LLM-side reward is computed *per episode*, not per skill burst.
ScoredDataGroup gets one score per rollout (Atropos convention for chat
envs). We attribute the score to the LLM's full sequence of skill picks.

Components:
    +10.0 per battery collected
    +20.0 bonus if all batteries collected before timeout (success)
    -0.05 per skill pick (efficiency penalty — fewer picks = higher score)
    -5.0  if the robot fell (terminal failure)
    +0.5  × Δ(distance-to-nearest-battery) summed over picks (dense shaping)

Tunable via `RewardConfig`. Defaults chosen so a successful 5-battery
episode scores ~70-90 and a failure scores ~-5 to 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import math


@dataclass
class RewardConfig:
    per_battery: float = 10.0
    success_bonus: float = 20.0
    skill_pick_cost: float = 0.05
    fall_penalty: float = 5.0
    distance_decrement_weight: float = 0.5
    max_episode_score_clip: float = 200.0  # safety against runaway shaping


@dataclass
class EpisodeMetrics:
    """Per-rollout aggregates handed to the reward function."""

    batteries_collected: int
    total_batteries: int
    skill_picks: int
    fell: bool
    timed_out: bool
    nearest_battery_distance_history: List[float]  # one entry per pick


def compute_reward(metrics: EpisodeMetrics, cfg: RewardConfig | None = None) -> float:
    """Single scalar score for the LLM's full skill-pick trajectory."""
    cfg = cfg or RewardConfig()

    # Collection.
    score = cfg.per_battery * metrics.batteries_collected

    # Success bonus.
    if (
        metrics.batteries_collected == metrics.total_batteries
        and metrics.total_batteries > 0
    ):
        score += cfg.success_bonus

    # Efficiency penalty.
    score -= cfg.skill_pick_cost * metrics.skill_picks

    # Death penalty.
    if metrics.fell:
        score -= cfg.fall_penalty

    # Distance shaping: positive when distances trend down across picks.
    dists = metrics.nearest_battery_distance_history
    if len(dists) >= 2:
        # Total decrement: how much closer we got over the episode.
        # Use the cumulative best-distance progression so we don't reward
        # walking-toward-and-then-away cycles.
        best = math.inf
        decrement_sum = 0.0
        for d in dists:
            if d < best:
                decrement_sum += (best - d) if best != math.inf else 0.0
                best = d
        score += cfg.distance_decrement_weight * decrement_sum

    return float(max(-cfg.max_episode_score_clip, min(cfg.max_episode_score_clip, score)))
