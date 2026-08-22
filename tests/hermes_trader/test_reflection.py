"""Tests for hermes_trader.reflection — P4 post-trade reflection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hermes_trader.config import TraderConfig
from hermes_trader.memory.episodes import EpisodeStore
from hermes_trader.memory.strategic_rules import load_strategic_rules
from hermes_trader.reflection.calibration_report import (
    build_calibration_report,
    confidence_bucket,
    format_calibration_report_markdown,
    suggest_min_confidence,
)
from hermes_trader.reflection.distill import (
    apply_distillation,
    should_add_negative_constraint,
    should_add_positive_heuristic,
)
from hermes_trader.reflection.judge import (
    JudgeScore,
    ReflectionInput,
    build_judge_prompt,
    heuristic_judge,
    parse_judge_response,
    run_judge,
)
from hermes_trader.reflection.pipeline import run_reflection, run_weekly_calibration
from tests.hermes_trader.test_memory import _sample_cycle_result


@pytest.fixture
def episode_db(tmp_path):
    return EpisodeStore(tmp_path / "trade_episodes.db")


@pytest.fixture
def rules_path(tmp_path):
    return tmp_path / "strategic_rules.yaml"


def test_parse_judge_response_json():
    score = parse_judge_response(
        '{"thesis":4,"timing":3,"sizing":4,"execution":3,"overall":3.5,"lessons":["ok"]}'
    )
    assert score.thesis == 4.0
    assert score.overall == 3.5
    assert score.lessons == ["ok"]


def test_heuristic_judge_produces_bounded_scores(episode_db):
    episode = episode_db.record_cycle(_sample_cycle_result())
    reflection = ReflectionInput(episode=episode, pnl_usd=-10.0)
    score = heuristic_judge(reflection)
    assert 1.0 <= score.overall <= 5.0
    assert score.lessons


def test_build_judge_prompt_includes_intent():
    from hermes_trader.memory.episodes import episode_from_cycle

    episode = episode_from_cycle(_sample_cycle_result())
    prompt = build_judge_prompt(ReflectionInput(episode=episode, pnl_usd=5.0))
    assert "buy" in prompt.lower() or "intent" in prompt.lower()


def test_distill_adds_negative_on_bad_score(episode_db, rules_path, tmp_path):
    cfg = TraderConfig(reflection_loss_threshold_usd=5.0)
    episode = episode_db.record_cycle(_sample_cycle_result())
    episode_db.update_outcome(episode.episode_id, pnl_usd=-12.0)
    episode = episode_db.get_episode(episode.episode_id)
    assert episode is not None

    score = JudgeScore(thesis=2, timing=2, sizing=2, execution=2, overall=2, lessons=["avoid"])
    assert should_add_negative_constraint(score, -12.0, loss_threshold_usd=5.0)
    _rules, changed = apply_distillation(episode, score, cfg, rules_path=rules_path)
    assert changed
    loaded = load_strategic_rules(rules_path)
    assert any("avoid" in r.get("rule", "") for r in loaded.negative_constraints)


def test_distill_adds_positive_on_win(episode_db, rules_path):
    cfg = TraderConfig()
    episode = episode_db.record_cycle(_sample_cycle_result())
    episode_db.update_outcome(episode.episode_id, pnl_usd=25.0)
    episode = episode_db.get_episode(episode.episode_id)
    assert episode is not None
    score = JudgeScore(thesis=4.5, timing=4, sizing=4, execution=4, overall=4.2, lessons=["repeat"])
    assert should_add_positive_heuristic(score, 25.0)
    _rules, changed = apply_distillation(episode, score, cfg, rules_path=rules_path)
    assert changed


def test_calibration_report_buckets(episode_db):
    for i, (pnl, conf) in enumerate([(5.0, 0.55), (-3.0, 0.55), (10.0, 0.75)]):
        ep = episode_db.record_cycle(_sample_cycle_result())
        episode_db.update_outcome(ep.episode_id, pnl_usd=pnl)
        row = episode_db.get_episode(ep.episode_id)
        assert row is not None
        row.intent["confidence"] = conf
        # confidence is in stored intent_json — re-record not needed for bucket if set at record
    closed = episode_db.list_closed_episodes()
    report = build_calibration_report(closed)
    assert report
    assert confidence_bucket(0.55) == "0.5–0.6"


def test_suggest_min_confidence_tightens_when_miscalibrated():
    from hermes_trader.reflection.calibration_report import BucketStats

    report = [
        BucketStats(
            bucket="0.6–0.7",
            trades=10,
            win_rate=0.35,
            avg_pnl=-2.0,
            judge_avg=2.5,
            avg_confidence=0.65,
        )
    ]
    suggested = suggest_min_confidence(report, current_min=0.6, miscalibration_gap=0.15)
    assert suggested is not None
    assert suggested > 0.6


def test_run_reflection_pipeline(episode_db, rules_path):
    cfg = TraderConfig(reflection_loss_threshold_usd=1.0)
    episode = episode_db.record_cycle(_sample_cycle_result())
    result = run_reflection(
        episode.episode_id,
        pnl_usd=-8.0,
        store=episode_db,
        config=cfg,
        rules_path=rules_path,
    )
    assert result.score.overall >= 1.0
    stored = episode_db.get_episode(episode.episode_id)
    assert stored is not None
    assert stored.reflection is not None
    assert stored.pnl_usd == -8.0


def test_weekly_calibration_markdown(episode_db):
    ep = episode_db.record_cycle(_sample_cycle_result())
    episode_db.update_outcome(ep.episode_id, pnl_usd=3.0)
    run_reflection(ep.episode_id, pnl_usd=3.0, store=episode_db, apply_rules=False)
    md = run_weekly_calibration(store=episode_db)
    assert "Calibration" in md
    assert "|" in md


def test_schema_v2_migration(episode_db):
    version = episode_db.migrate()
    assert version == 2
    with episode_db.connect() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trade_episodes)")}
    assert "reflection_json" in cols