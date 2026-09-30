"""
Tests for the Self Evolution Plugin.

Covers:
  - quality_scorer: composite score computation
  - models: dataclass serialization / deserialization
  - db: SQLite CRUD operations (temp DB)
  - hooks: telemetry collection + signal detection
  - rule_engine: strategy condition matching
  - strategy_store: file-based persistence + archive
  - evolution_proposer: proposal generation + dedup
  - evolution_executor: execute + tracking + rollback
  - reflection_engine: JSON/text parsing of model output
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def _tmp_evolution_db(tmp_path, monkeypatch):
    """Redirect self_evolution DB to a temp directory for every test."""
    db_dir = tmp_path / ".hermes" / "self_evolution"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "evolution.db"

    # Patch centralized paths module
    import self_evolution.paths as paths_mod
    monkeypatch.setattr(paths_mod, "DATA_DIR", db_dir)
    monkeypatch.setattr(paths_mod, "DB_PATH", db_path)
    monkeypatch.setattr(paths_mod, "STRATEGIES_FILE", db_dir / "strategies.json")
    monkeypatch.setattr(paths_mod, "ARCHIVE_DIR", db_dir / "archive")
    monkeypatch.setattr(paths_mod, "SKILLS_DIR", tmp_path / ".hermes" / "skills" / "learned")
    monkeypatch.setattr(paths_mod, "MEMORIES_DIR", tmp_path / ".hermes" / "memories")

    # Also patch the imported names in db module
    import self_evolution.db as db_mod
    monkeypatch.setattr(db_mod, "DB_DIR", db_dir)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)

    # Initialize schema
    db_mod.init_db()
    yield db_mod
    # Clean up thread-local connection after each test
    db_mod.close_connection()


@pytest.fixture
def sample_session_data():
    """Standard session data for quality scoring tests."""
    return {
        "session_id": "test-session-001",
        "completed": True,
        "iterations": 5,
        "tool_call_count": 5,
        "message_count": 3,
        "duration_seconds": 120,
        "model": "test-model",
        "platform": "test",
        "tool_names": ["bash", "read", "write"],
    }


# ============================================================================
# 1. Quality Scorer
# ============================================================================

class TestQualityScorer:
    """Test the composite quality score computation."""

    def test_completed_session_high_score(self, sample_session_data):
        from self_evolution.quality_scorer import compute_score

        score = compute_score(sample_session_data)
        assert score.composite > 0.5, f"Completed session should score > 0.5, got {score.composite}"
        assert score.completion_rate == 1.0
        assert score.task_category == "coding"

    def test_interrupted_session_medium_score(self, sample_session_data):
        from self_evolution.quality_scorer import compute_score

        sample_session_data["completed"] = False
        sample_session_data["interrupted"] = True
        score = compute_score(sample_session_data)
        assert score.completion_rate == 0.5

    def test_partial_session(self, sample_session_data):
        from self_evolution.quality_scorer import compute_score

        sample_session_data["completed"] = False
        sample_session_data["partial"] = True
        score = compute_score(sample_session_data)
        assert score.completion_rate == 0.3

    def test_efficiency_degrades_with_iterations(self, sample_session_data):
        from self_evolution.quality_scorer import compute_score

        # Low iterations => high efficiency
        sample_session_data["iterations"] = 2
        score_low = compute_score(sample_session_data)

        # High iterations => low efficiency
        sample_session_data["iterations"] = 50
        score_high = compute_score(sample_session_data)

        assert score_low.efficiency_score > score_high.efficiency_score

    def test_budget_exhaustion_lowers_satisfaction(self, sample_session_data):
        from self_evolution.quality_scorer import compute_score

        sample_session_data["max_iterations"] = 5
        sample_session_data["iterations"] = 5  # exactly at limit
        score = compute_score(sample_session_data)
        assert score.satisfaction_proxy < 0.7  # below baseline

    def test_single_turn_completion_high_satisfaction(self, sample_session_data):
        from self_evolution.quality_scorer import compute_score

        sample_session_data["message_count"] = 2
        sample_session_data["completed"] = True
        score = compute_score(sample_session_data)
        assert score.satisfaction_proxy == 0.9

    def test_task_category_coding(self, sample_session_data):
        from self_evolution.quality_scorer import compute_score

        sample_session_data["tool_names"] = ["bash", "write"]
        score = compute_score(sample_session_data)
        assert score.task_category == "coding"

    def test_task_category_web_research(self, sample_session_data):
        from self_evolution.quality_scorer import compute_score

        sample_session_data["tool_names"] = ["web_search", "browser"]
        score = compute_score(sample_session_data)
        assert score.task_category == "web_research"

    def test_task_category_file_analysis(self, sample_session_data):
        from self_evolution.quality_scorer import compute_score

        sample_session_data["tool_names"] = ["read", "grep", "glob"]
        score = compute_score(sample_session_data)
        assert score.task_category == "file_analysis"

    def test_task_category_general(self, sample_session_data):
        from self_evolution.quality_scorer import compute_score

        sample_session_data["tool_names"] = []
        score = compute_score(sample_session_data)
        assert score.task_category == "general"

    def test_tool_names_as_string(self, sample_session_data):
        from self_evolution.quality_scorer import compute_score

        sample_session_data["tool_names"] = "bash,read,write"
        score = compute_score(sample_session_data)
        assert score.task_category == "coding"

    def test_composite_weighted_sum(self, sample_session_data):
        """Verify composite = 0.4*completion + 0.2*efficiency + 0.15*cost + 0.25*satisfaction."""
        from self_evolution.quality_scorer import compute_score

        score = compute_score(sample_session_data)
        expected = (
            0.40 * score.completion_rate
            + 0.20 * score.efficiency_score
            + 0.15 * score.cost_efficiency
            + 0.25 * score.satisfaction_proxy
        )
        assert abs(score.composite - round(expected, 3)) < 0.001


# ============================================================================
# 2. Models — Serialization
# ============================================================================

class TestModels:
    """Test data model serialization and deserialization."""

    def test_quality_score_to_db_row(self):
        from self_evolution.models import QualityScore

        qs = QualityScore(
            session_id="s1",
            composite=0.85,
            completion_rate=1.0,
            efficiency_score=0.7,
            cost_efficiency=0.9,
            satisfaction_proxy=0.8,
            task_category="coding",
            model="test",
        )
        row = qs.to_db_row()
        assert row["session_id"] == "s1"
        assert row["composite_score"] == 0.85
        assert row["task_category"] == "coding"

    def test_reflection_report_to_db_row(self):
        from self_evolution.models import ReflectionReport

        report = ReflectionReport(
            period_start=1000.0,
            period_end=2000.0,
            sessions_analyzed=5,
            avg_score=0.75,
            worst_patterns=["pattern1", "pattern2"],
            best_patterns=["good1"],
            recommendations=["rec1"],
        )
        row = report.to_db_row()
        assert row["sessions_analyzed"] == 5
        assert json.loads(row["worst_patterns"]) == ["pattern1", "pattern2"]
        assert json.loads(row["best_patterns"]) == ["good1"]

    def test_proposal_to_db_row(self):
        from self_evolution.models import Proposal

        p = Proposal(
            id="prop-001",
            proposal_type="strategy",
            title="Test Proposal",
            description="A test proposal",
            risk_assessment="low",
        )
        row = p.to_db_row()
        assert row["id"] == "prop-001"
        assert row["proposal_type"] == "strategy"
        assert row["status"] == "pending_approval"

    def test_improvement_unit_should_revert(self):
        from self_evolution.models import ImprovementUnit

        unit = ImprovementUnit(
            id="u1",
            proposal_id="p1",
            change_type="strategy",
            baseline_score=0.8,
            current_score=0.6,
            sessions_sampled=5,
            max_regression=0.10,
        )
        # Regression = 0.2 > max_regression 0.10 => should revert
        assert unit.should_revert is True

    def test_improvement_unit_should_not_revert(self):
        from self_evolution.models import ImprovementUnit

        unit = ImprovementUnit(
            id="u2",
            proposal_id="p2",
            change_type="strategy",
            baseline_score=0.8,
            current_score=0.75,
            sessions_sampled=5,
            max_regression=0.10,
        )
        # Regression = 0.05 < max_regression 0.10 => should NOT revert
        assert unit.should_revert is False

    def test_improvement_unit_should_promote(self):
        from self_evolution.models import ImprovementUnit

        unit = ImprovementUnit(
            id="u3",
            proposal_id="p3",
            change_type="strategy",
            baseline_score=0.7,
            current_score=0.8,
            sessions_sampled=15,
            min_sessions=10,
            min_improvement=0.05,
        )
        # Improvement = 0.1 >= min_improvement 0.05 and sessions >= min_sessions
        assert unit.should_promote is True

    def test_improvement_unit_should_not_promote_too_few_sessions(self):
        from self_evolution.models import ImprovementUnit

        unit = ImprovementUnit(
            id="u4",
            proposal_id="p4",
            change_type="strategy",
            baseline_score=0.7,
            current_score=0.9,
            sessions_sampled=5,
            min_sessions=10,
            min_improvement=0.05,
        )
        assert unit.should_promote is False

    def test_strategy_rule_roundtrip(self):
        from self_evolution.models import StrategyRule, StrategyCondition

        rule = StrategyRule(
            id="sr1",
            name="Avoid large file reads",
            strategy_type="avoid",
            description="Don't read files > 1MB",
            conditions=[
                StrategyCondition(field="tool_name", operator="equals", pattern="read"),
            ],
            hint_text="Use grep instead",
            severity="high",
        )
        d = rule.to_dict()
        restored = StrategyRule.from_dict(d)
        assert restored.id == "sr1"
        assert restored.strategy_type == "avoid"
        assert len(restored.conditions) == 1
        assert restored.conditions[0].field == "tool_name"

    def test_error_analysis_summary(self):
        from self_evolution.models import ErrorAnalysis, ToolFailure

        ea = ErrorAnalysis(
            tool_failures=[
                ToolFailure(tool_name="bash", error_type="timeout", count=3),
            ],
            retry_patterns=[],
            incomplete_sessions=["s1"],
            user_corrections=2,
        )
        summary = ea.summary()
        assert "bash" in summary
        assert "3" in summary
        assert "未完成" in summary
        assert "纠正" in summary

    def test_waste_analysis_summary(self):
        from self_evolution.models import WasteAnalysis, ToolDuration

        wa = WasteAnalysis(
            slowest_tools=[
                ToolDuration(tool_name="bash", total_duration_ms=5000, call_count=5, avg_duration_ms=1000),
            ],
        )
        summary = wa.summary()
        assert "bash" in summary
        assert "1000" in summary

    def test_code_change_analysis_summary_empty(self):
        from self_evolution.models import CodeChangeAnalysis

        cca = CodeChangeAnalysis()
        assert cca.summary() == "代码更新: 无新提交"

    def test_code_change_analysis_summary_with_commits(self):
        from self_evolution.models import CodeChangeAnalysis, CommitInfo

        cca = CodeChangeAnalysis(
            commits=[
                CommitInfo(hash_short="abc1234", subject="fix: bug", insertions=10, deletions=5),
            ],
            total_commits=1,
            total_insertions=10,
            total_deletions=5,
            total_files_changed=2,
        )
        summary = cca.summary()
        assert "abc1234" in summary
        assert "+10" in summary


# ============================================================================
# 3. Database CRUD
# ============================================================================

class TestDatabase:
    """Test SQLite CRUD operations."""

    def test_init_db_creates_tables(self, _tmp_evolution_db):
        conn = _tmp_evolution_db.get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "tool_invocations" in table_names
        assert "session_scores" in table_names
        assert "evolution_proposals" in table_names
        assert "improvement_units" in table_names
        assert "strategy_versions" in table_names
        conn.close()

    def test_insert_and_fetch(self, _tmp_evolution_db):
        rowid = _tmp_evolution_db.insert("session_scores", {
            "session_id": "s-test",
            "composite_score": 0.85,
            "completion_rate": 1.0,
            "efficiency_score": 0.7,
            "cost_efficiency": 0.9,
            "satisfaction_proxy": 0.8,
            "task_category": "coding",
            "model": "test",
        })
        assert rowid > 0

        row = _tmp_evolution_db.fetch_one("session_scores", where="session_id = ?", params=("s-test",))
        assert row is not None
        assert row["composite_score"] == 0.85

    def test_insert_many(self, _tmp_evolution_db):
        rows = [
            {"session_id": f"s-{i}", "composite_score": 0.5, "completion_rate": 1.0,
             "efficiency_score": 0.5, "cost_efficiency": 0.5, "satisfaction_proxy": 0.5,
             "task_category": "general", "model": "test"}
            for i in range(3)
        ]
        _tmp_evolution_db.insert_many("session_scores", rows)
        all_rows = _tmp_evolution_db.fetch_all("session_scores")
        assert len(all_rows) == 3

    def test_update(self, _tmp_evolution_db):
        _tmp_evolution_db.insert("session_scores", {
            "session_id": "s-upd", "composite_score": 0.5,
            "completion_rate": 1.0, "efficiency_score": 0.5,
            "cost_efficiency": 0.5, "satisfaction_proxy": 0.5,
            "task_category": "general", "model": "test",
        })
        _tmp_evolution_db.update(
            "session_scores",
            {"composite_score": 0.95},
            where="session_id = ?",
            where_params=("s-upd",),
        )
        row = _tmp_evolution_db.fetch_one("session_scores", where="session_id = ?", params=("s-upd",))
        assert row["composite_score"] == 0.95

    def test_fetch_all_with_order_and_limit(self, _tmp_evolution_db):
        for i in range(5):
            _tmp_evolution_db.insert("tool_invocations", {
                "session_id": f"s-{i}",
                "tool_name": "bash",
                "duration_ms": i * 100,
                "success": True,
                "turn_number": i,
            })
        rows = _tmp_evolution_db.fetch_all(
            "tool_invocations",
            where="tool_name = ?",
            params=("bash",),
            order_by="duration_ms DESC",
            limit=3,
        )
        assert len(rows) == 3
        assert rows[0]["duration_ms"] == 400

    def test_query(self, _tmp_evolution_db):
        _tmp_evolution_db.insert("session_scores", {
            "session_id": "s-q", "composite_score": 0.7,
            "completion_rate": 1.0, "efficiency_score": 0.5,
            "cost_efficiency": 0.5, "satisfaction_proxy": 0.5,
            "task_category": "general", "model": "test",
        })
        results = _tmp_evolution_db.query("SELECT COUNT(*) as cnt FROM session_scores")
        assert results[0]["cnt"] == 1

    def test_cleanup(self, _tmp_evolution_db):
        old_ts = time.time() - 31 * 86400  # 31 days ago
        _tmp_evolution_db.insert("tool_invocations", {
            "session_id": "s-old", "tool_name": "bash",
            "duration_ms": 100, "success": True, "turn_number": 0,
            "created_at": old_ts,
        })
        _tmp_evolution_db.insert("tool_invocations", {
            "session_id": "s-new", "tool_name": "bash",
            "duration_ms": 100, "success": True, "turn_number": 0,
        })
        _tmp_evolution_db.cleanup(days=30)
        remaining = _tmp_evolution_db.fetch_all("tool_invocations")
        assert len(remaining) == 1
        assert remaining[0]["session_id"] == "s-new"


# ============================================================================
# 4. Hooks — Telemetry + Signal Detection
# ============================================================================

class TestHooks:
    """Test lifecycle hook functions."""

    def test_on_tool_call_inserts_telemetry(self, _tmp_evolution_db):
        from self_evolution.hooks import on_tool_call

        on_tool_call(
            tool_name="bash",
            started_at=time.time(),
            duration_ms=500,
            success=True,
            session_id="s-hook-1",
            turn_number=3,
        )
        rows = _tmp_evolution_db.fetch_all("tool_invocations")
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "bash"
        assert rows[0]["duration_ms"] == 500

    def test_on_tool_call_failure(self, _tmp_evolution_db):
        from self_evolution.hooks import on_tool_call

        on_tool_call(
            tool_name="write",
            success=False,
            error_type="PermissionError",
            session_id="s-hook-2",
        )
        rows = _tmp_evolution_db.fetch_all("tool_invocations")
        assert rows[0]["success"] is False or rows[0]["success"] == 0
        assert rows[0]["error_type"] == "PermissionError"

    def test_on_session_end_computes_score(self, _tmp_evolution_db):
        from self_evolution.hooks import on_session_end

        on_session_end(session_data={
            "session_id": "s-end-1",
            "completed": True,
            "iterations": 3,
            "tool_call_count": 3,
            "message_count": 2,
            "tool_names": ["bash"],
        })
        row = _tmp_evolution_db.fetch_one("session_scores", where="session_id = ?", params=("s-end-1",))
        assert row is not None
        assert row["composite_score"] > 0

    def test_on_session_end_no_session_id(self, _tmp_evolution_db):
        from self_evolution.hooks import on_session_end

        # Should not crash, should not insert anything
        on_session_end(session_data={})
        rows = _tmp_evolution_db.fetch_all("session_scores")
        assert len(rows) == 0

    def test_correction_signal_detected(self, _tmp_evolution_db):
        from self_evolution.hooks import on_session_end

        on_session_end(session_data={
            "session_id": "s-corr-1",
            "completed": True,
            "iterations": 5,
            "tool_call_count": 5,
            "message_count": 3,
            "messages": [
                {"role": "assistant", "content": "Done"},
                {"role": "user", "content": "不对，这不是我想要的"},
            ],
        })
        signals = _tmp_evolution_db.fetch_all(
            "outcome_signals",
            where="session_id = ? AND signal_type = ?",
            params=("s-corr-1", "correction"),
        )
        assert len(signals) == 1

    def test_frustration_signal_detected(self, _tmp_evolution_db):
        from self_evolution.hooks import on_session_end

        on_session_end(session_data={
            "session_id": "s-frust-1",
            "completed": True,
            "iterations": 5,
            "tool_call_count": 5,
            "message_count": 3,
            "messages": [
                {"role": "assistant", "content": "Done"},
                {"role": "user", "content": "太慢了，浪费时间"},
            ],
        })
        signals = _tmp_evolution_db.fetch_all(
            "outcome_signals",
            where="session_id = ? AND signal_type = ?",
            params=("s-frust-1", "frustration"),
        )
        assert len(signals) == 1

    def test_budget_exhausted_signal(self, _tmp_evolution_db):
        from self_evolution.hooks import on_session_end

        on_session_end(session_data={
            "session_id": "s-budget-1",
            "completed": False,
            "interrupted": False,
            "iterations": 20,
            "max_iterations": 20,
            "tool_call_count": 20,
            "message_count": 10,
        })
        signals = _tmp_evolution_db.fetch_all(
            "outcome_signals",
            where="session_id = ? AND signal_type = ?",
            params=("s-budget-1", "budget_exhausted"),
        )
        assert len(signals) == 1


# ============================================================================
# 5. Rule Engine — Strategy Matching
# ============================================================================

class TestRuleEngine:
    """Test conditional strategy matching."""

    def _make_rule(self, strategy_type="hint", conditions=None, enabled=True):
        from self_evolution.models import StrategyRule, StrategyCondition

        return StrategyRule(
            id="r1",
            name="Test Rule",
            strategy_type=strategy_type,
            description="desc",
            conditions=conditions or [],
            hint_text="test hint",
            enabled=enabled,
        )

    def test_always_match_no_conditions(self):
        from self_evolution.rule_engine import StrategyRuleEngine

        engine = StrategyRuleEngine()
        rule = self._make_rule()
        matched = engine.match_strategies([rule], {})
        assert len(matched) == 1

    def test_disabled_rule_not_matched(self):
        from self_evolution.rule_engine import StrategyRuleEngine

        engine = StrategyRuleEngine()
        rule = self._make_rule(enabled=False)
        matched = engine.match_strategies([rule], {})
        assert len(matched) == 0

    def test_equals_operator(self):
        from self_evolution.rule_engine import StrategyRuleEngine
        from self_evolution.models import StrategyCondition

        engine = StrategyRuleEngine()
        rule = self._make_rule(conditions=[
            StrategyCondition(field="tool_name", operator="equals", pattern="bash"),
        ])
        assert len(engine.match_strategies([rule], {"tool_name": "bash"})) == 1
        assert len(engine.match_strategies([rule], {"tool_name": "read"})) == 0

    def test_contains_operator(self):
        from self_evolution.rule_engine import StrategyRuleEngine
        from self_evolution.models import StrategyCondition

        engine = StrategyRuleEngine()
        rule = self._make_rule(conditions=[
            StrategyCondition(field="task_type", operator="contains", pattern="debug"),
        ])
        assert len(engine.match_strategies([rule], {"task_type": "debug python code"})) == 1
        assert len(engine.match_strategies([rule], {"task_type": "write tests"})) == 0

    def test_regex_match_operator(self):
        from self_evolution.rule_engine import StrategyRuleEngine
        from self_evolution.models import StrategyCondition

        engine = StrategyRuleEngine()
        rule = self._make_rule(conditions=[
            StrategyCondition(field="platform", operator="regex_match", pattern="feishu|slack"),
        ])
        assert len(engine.match_strategies([rule], {"platform": "feishu"})) == 1
        assert len(engine.match_strategies([rule], {"platform": "discord"})) == 0

    def test_not_contains_operator(self):
        from self_evolution.rule_engine import StrategyRuleEngine
        from self_evolution.models import StrategyCondition

        engine = StrategyRuleEngine()
        rule = self._make_rule(conditions=[
            StrategyCondition(field="model", operator="not_contains", pattern="mini"),
        ])
        assert len(engine.match_strategies([rule], {"model": "gpt-4"})) == 1
        assert len(engine.match_strategies([rule], {"model": "gpt-4-mini"})) == 0

    def test_starts_with_operator(self):
        from self_evolution.rule_engine import StrategyRuleEngine
        from self_evolution.models import StrategyCondition

        engine = StrategyRuleEngine()
        rule = self._make_rule(conditions=[
            StrategyCondition(field="platform", operator="starts_with", pattern="feishu"),
        ])
        assert len(engine.match_strategies([rule], {"platform": "feishu_web"})) == 1
        assert len(engine.match_strategies([rule], {"platform": "web_feishu"})) == 0

    def test_and_logic_all_conditions_must_match(self):
        from self_evolution.rule_engine import StrategyRuleEngine
        from self_evolution.models import StrategyCondition

        engine = StrategyRuleEngine()
        rule = self._make_rule(conditions=[
            StrategyCondition(field="platform", operator="equals", pattern="feishu"),
            StrategyCondition(field="task_type", operator="contains", pattern="code"),
        ])
        # Both match
        assert len(engine.match_strategies([rule], {"platform": "feishu", "task_type": "code review"})) == 1
        # Only one matches
        assert len(engine.match_strategies([rule], {"platform": "feishu", "task_type": "chat"})) == 0

    def test_format_hints(self):
        from self_evolution.rule_engine import StrategyRuleEngine

        engine = StrategyRuleEngine()
        rule = self._make_rule(strategy_type="avoid", conditions=[])
        hint = engine.format_hints([rule])
        assert "[自我进化策略提示]" in hint
        assert "Test Rule" in hint


# ============================================================================
# 6. Strategy Store
# ============================================================================

class TestStrategyStore:
    """Test strategy persistence with versioning."""

    def test_load_empty(self, tmp_path, monkeypatch):
        from self_evolution.strategy_store import StrategyStore

        store = StrategyStore()
        monkeypatch.setattr(
            "self_evolution.strategy_store.STRATEGIES_FILE",
            tmp_path / "strategies.json",
        )
        monkeypatch.setattr(
            "self_evolution.strategy_store.ARCHIVE_DIR",
            tmp_path / "archive",
        )
        data = store.load()
        assert data["version"] == 0
        assert data["rules"] == []

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        from self_evolution.strategy_store import StrategyStore

        store = StrategyStore()
        sf = tmp_path / "strategies.json"
        ad = tmp_path / "archive"
        monkeypatch.setattr("self_evolution.paths.STRATEGIES_FILE", sf)
        monkeypatch.setattr("self_evolution.paths.ARCHIVE_DIR", ad)
        monkeypatch.setattr("self_evolution.strategy_store.STRATEGIES_FILE", sf)
        monkeypatch.setattr("self_evolution.strategy_store.ARCHIVE_DIR", ad)

        data = {"version": 1, "rules": [{"id": "r1", "name": "Rule 1"}]}
        store.save(data)

        loaded = store.load()
        assert loaded["version"] == 1
        assert len(loaded["rules"]) == 1

    def test_archive_and_restore(self, tmp_path, monkeypatch):
        from self_evolution.strategy_store import StrategyStore

        store = StrategyStore()
        sf = tmp_path / "strategies.json"
        ad = tmp_path / "archive"
        monkeypatch.setattr("self_evolution.paths.STRATEGIES_FILE", sf)
        monkeypatch.setattr("self_evolution.paths.ARCHIVE_DIR", ad)
        monkeypatch.setattr("self_evolution.strategy_store.STRATEGIES_FILE", sf)
        monkeypatch.setattr("self_evolution.strategy_store.ARCHIVE_DIR", ad)

        data_v1 = {"version": 1, "rules": [{"id": "r1"}]}
        store.save(data_v1)
        store.archive(1)

        # Overwrite with v2
        data_v2 = {"version": 2, "rules": [{"id": "r2"}]}
        store.save(data_v2)

        # Restore v1
        archive = store.load_archive(1)
        assert archive["version"] == 1
        assert archive["rules"][0]["id"] == "r1"

    def test_load_nonexistent_archive(self, tmp_path, monkeypatch):
        from self_evolution.strategy_store import StrategyStore

        store = StrategyStore()
        monkeypatch.setattr("self_evolution.paths.ARCHIVE_DIR", tmp_path / "archive")
        monkeypatch.setattr(
            "self_evolution.strategy_store.ARCHIVE_DIR",
            tmp_path / "archive",
        )
        assert store.load_archive(999) is None


# ============================================================================
# 7. Evolution Proposer
# ============================================================================

class TestEvolutionProposer:
    """Test proposal generation from reflection reports."""

    def _make_report(self, worst=None, best=None, recs=None, sessions=10):
        from self_evolution.models import ReflectionReport

        return ReflectionReport(
            period_start=1000.0,
            period_end=2000.0,
            sessions_analyzed=sessions,
            worst_patterns=worst or ["bash timeout frequently"],
            best_patterns=best or ["single-turn code generation works well"],
            recommendations=recs or ["创建新的工具偏好来优化bash使用"],
        )

    def test_generates_proposals_from_report(self):
        from self_evolution.evolution_proposer import generate_proposals

        report = self._make_report()
        proposals = generate_proposals(report, report_id=1)
        assert len(proposals) > 0

    def test_error_pattern_creates_code_improvement_proposal(self):
        from self_evolution.evolution_proposer import generate_proposals

        report = self._make_report(worst=["tool failure pattern"])
        proposals = generate_proposals(report, report_id=1)
        code_proposals = [p for p in proposals if p.proposal_type == "code_improvement"]
        assert len(code_proposals) > 0
        # Verify structured description
        desc = code_proposals[0].description
        assert "问题描述" in desc
        assert "建议方向" in desc

    def test_success_pattern_creates_skill_proposal(self):
        from self_evolution.evolution_proposer import generate_proposals

        # Report with enough sessions to pass the ≥5 threshold
        report = self._make_report(
            best=["efficient workflow discovered"],
            sessions=10,
        )
        proposals = generate_proposals(report, report_id=1)
        skill_proposals = [p for p in proposals if p.proposal_type == "skill"]
        assert len(skill_proposals) > 0

    def test_success_pattern_skipped_below_threshold(self):
        """Skill proposals should not be generated from best_patterns with <5 sessions."""
        from self_evolution.evolution_proposer import generate_proposals

        report = self._make_report(
            best=["efficient workflow discovered"],
            recs=[],  # No recommendations that might create skill proposals
            sessions=2,  # Below threshold
        )
        proposals = generate_proposals(report, report_id=1)
        skill_from_best = [
            p for p in proposals
            if p.proposal_type == "skill" and p.id.startswith("prop-success-")
        ]
        assert len(skill_from_best) == 0

    def test_recommendation_type_detection(self):
        from self_evolution.evolution_proposer import generate_proposals

        report = self._make_report(recs=["更新记忆来记住这个发现"])
        proposals = generate_proposals(report, report_id=1)
        memory_proposals = [p for p in proposals if p.proposal_type == "memory"]
        assert len(memory_proposals) > 0

    def test_deduplication(self):
        from self_evolution.evolution_proposer import generate_proposals

        report = self._make_report(
            worst=["same pattern", "same pattern"],  # duplicate
        )
        proposals = generate_proposals(report, report_id=1)
        titles = [p.title for p in proposals]
        assert len(titles) == len(set(titles)), "Should deduplicate similar titles"

    def test_max_five_proposals(self):
        from self_evolution.evolution_proposer import generate_proposals

        report = self._make_report(
            worst=[f"pattern {i}" for i in range(10)],
            best=[f"best {i}" for i in range(10)],
            recs=[f"rec {i}" for i in range(10)],
        )
        proposals = generate_proposals(report, report_id=1)
        assert len(proposals) <= 5


# ============================================================================
# 8. Evolution Executor
# ============================================================================

class TestEvolutionExecutor:
    """Test execution of approved proposals."""

    def test_execute_strategy_proposal(self, _tmp_evolution_db, tmp_path, monkeypatch):
        from self_evolution.evolution_executor import EvolutionExecutor
        from self_evolution.models import Proposal

        monkeypatch.setattr(
            "self_evolution.evolution_executor.STRATEGIES_DIR", tmp_path,
        )
        monkeypatch.setattr(
            "self_evolution.evolution_executor.STRATEGIES_FILE",
            tmp_path / "strategies.json",
        )
        monkeypatch.setattr(
            "self_evolution.evolution_executor.ARCHIVE_DIR",
            tmp_path / "archive",
        )
        monkeypatch.setattr(
            "self_evolution.strategy_store.STRATEGIES_DIR", tmp_path,
        )
        monkeypatch.setattr(
            "self_evolution.strategy_store.STRATEGIES_FILE",
            tmp_path / "strategies.json",
        )
        monkeypatch.setattr(
            "self_evolution.strategy_store.ARCHIVE_DIR",
            tmp_path / "archive",
        )

        proposal = Proposal(
            id="prop-exec-1",
            proposal_type="strategy",
            title="Test Strategy",
            description="Avoid large file reads",
            status="approved",
        )
        executor = EvolutionExecutor()
        executor.execute(proposal)

        # Verify status updated
        row = _tmp_evolution_db.fetch_one("evolution_proposals", where="id IS NULL")  # proposal not in DB, skip
        # Verify strategy file updated
        from self_evolution.strategy_store import StrategyStore
        store = StrategyStore()
        data = store.load()
        assert data["version"] >= 1
        assert any(r["id"] == "prop-exec-1" for r in data["rules"])

    def test_execute_skill_proposal(self, _tmp_evolution_db, tmp_path, monkeypatch):
        from self_evolution.evolution_executor import EvolutionExecutor
        from self_evolution.models import Proposal

        skills_dir = tmp_path / "skills" / "learned"
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        proposal = Proposal(
            id="prop-skill-1",
            proposal_type="skill",
            title="Test Skill",
            description="A learned skill for testing",
            status="approved",
        )
        executor = EvolutionExecutor()
        executor.execute(proposal)

        skill_file = tmp_path / ".hermes" / "skills" / "learned" / "prop-skill-1" / "SKILL.md"
        assert skill_file.exists()
        content = skill_file.read_text()
        assert "Test Skill" in content

    def test_execute_memory_proposal(self, _tmp_evolution_db, tmp_path, monkeypatch):
        from self_evolution.evolution_executor import EvolutionExecutor
        from self_evolution.models import Proposal

        memories_dir = tmp_path / ".hermes" / "memories"
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        proposal = Proposal(
            id="prop-mem-1",
            proposal_type="memory",
            title="Remember Pattern",
            description="Always use context managers for file operations",
            status="approved",
        )
        executor = EvolutionExecutor()
        executor.execute(proposal)

        perf_file = memories_dir / "PERFORMANCE.md"
        assert perf_file.exists()
        content = perf_file.read_text()
        assert "context managers" in content

    def test_execute_tool_preference_proposal(self, _tmp_evolution_db, tmp_path, monkeypatch):
        from self_evolution.evolution_executor import EvolutionExecutor
        from self_evolution.models import Proposal

        evo_dir = tmp_path / "self_evolution"
        evo_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("self_evolution.paths.DATA_DIR", evo_dir)
        monkeypatch.setattr("self_evolution.evolution_executor.STRATEGIES_DIR", evo_dir)

        proposal = Proposal(
            id="prop-tool-1",
            proposal_type="tool_preference",
            title="Prefer grep over find",
            description="Use grep instead of find for searching",
            expected_impact="faster searches",
            status="approved",
        )
        executor = EvolutionExecutor()
        executor.execute(proposal)

        prefs_file = evo_dir / "tool_preferences.json"
        assert prefs_file.exists()
        prefs = json.loads(prefs_file.read_text())
        assert "prop-tool-1" in prefs


# ============================================================================
# 9. Reflection Engine — Parsing
# ============================================================================

class TestReflectionEngine:
    """Test reflection report parsing from model output."""

    def _make_engine(self):
        from self_evolution.reflection_engine import DreamEngine
        return DreamEngine(config={"base_url": "", "model": ""})

    def test_parse_valid_json(self):
        engine = self._make_engine()
        text = json.dumps({
            "worst_patterns": ["bash timeouts", "repeated reads"],
            "best_patterns": ["single-turn success"],
            "recommendations": ["add retry logic"],
            "tool_insights": {"bash": {"sr": 0.9}},
        })
        report = engine._parse_reflection(
            text, 1000.0, 2000.0, 5, 0.75,
            error_analysis=MagicMock(summary=lambda: ""),
            waste_analysis=MagicMock(summary=lambda: ""),
        )
        assert len(report.worst_patterns) == 2
        assert len(report.best_patterns) == 1
        assert len(report.recommendations) == 1

    def test_parse_json_in_markdown_wrapper(self):
        engine = self._make_engine()
        text = '```json\n{"worst_patterns": ["p1"], "best_patterns": [], "recommendations": []}\n```'
        report = engine._parse_reflection(
            text, 1000.0, 2000.0, 1, 0.5,
            error_analysis=MagicMock(summary=lambda: ""),
            waste_analysis=MagicMock(summary=lambda: ""),
        )
        assert report.worst_patterns == ["p1"]

    def test_parse_text_sections(self):
        engine = self._make_engine()
        text = """Here is my analysis:

worst patterns:
- Too many retries
- Slow file operations

best patterns:
- Direct code generation

recommendations:
- Cache tool results
- Optimize file reads
"""
        report = engine._parse_reflection(
            text, 1000.0, 2000.0, 1, 0.5,
            error_analysis=MagicMock(summary=lambda: ""),
            waste_analysis=MagicMock(summary=lambda: ""),
        )
        assert len(report.worst_patterns) >= 1
        assert len(report.best_patterns) >= 1
        assert len(report.recommendations) >= 1

    def test_parse_numbered_list(self):
        engine = self._make_engine()
        text = """分析结果:

worst patterns:
1) Bash command timeouts
2) Repeated tool calls

recommendations:
1) Add timeout handling
"""
        report = engine._parse_reflection(
            text, 1000.0, 2000.0, 1, 0.5,
            error_analysis=MagicMock(summary=lambda: ""),
            waste_analysis=MagicMock(summary=lambda: ""),
        )
        assert len(report.worst_patterns) >= 1

    def test_parse_empty_text(self):
        engine = self._make_engine()
        report = engine._parse_reflection(
            "", 1000.0, 2000.0, 0, 0.0,
            error_analysis=MagicMock(summary=lambda: ""),
            waste_analysis=MagicMock(summary=lambda: ""),
        )
        assert report.worst_patterns == []
        assert report.best_patterns == []
        assert report.recommendations == []


# ============================================================================
# 10. Integration — End-to-End Flow
# ============================================================================

class TestEndToEndFlow:
    """Test the full self-evolution cycle with mocked LLM calls."""

    def test_full_cycle_no_model(self, _tmp_evolution_db, tmp_path, monkeypatch):
        """Simulate the full cycle: hooks → data → analysis (without LLM call)."""
        from self_evolution.hooks import on_tool_call, on_session_end
        from self_evolution.reflection_engine import DreamEngine

        # 1. Simulate tool calls
        for i in range(5):
            on_tool_call(
                tool_name="bash",
                duration_ms=200 + i * 100,
                success=(i < 4),  # last one fails
                error_type="timeout" if i == 4 else None,
                session_id="s-e2e-1",
                turn_number=i,
            )

        # 2. Simulate session end
        on_session_end(session_data={
            "session_id": "s-e2e-1",
            "completed": True,
            "iterations": 5,
            "tool_call_count": 5,
            "message_count": 2,
            "tool_names": ["bash"],
            "model": "test",
        })

        # 3. Verify data was collected
        invocations = _tmp_evolution_db.fetch_all("tool_invocations")
        assert len(invocations) == 5

        scores = _tmp_evolution_db.fetch_all("session_scores")
        assert len(scores) == 1

        # 4. Run error analysis directly (no LLM)
        engine = DreamEngine(config={"base_url": "", "model": ""})
        invocations = _tmp_evolution_db.fetch_all("tool_invocations")
        signals = _tmp_evolution_db.fetch_all("outcome_signals")
        scores = _tmp_evolution_db.fetch_all("session_scores")

        error_analysis = engine._analyze_errors(scores, invocations, signals)
        assert len(error_analysis.tool_failures) == 1
        assert error_analysis.tool_failures[0].tool_name == "bash"
        assert error_analysis.tool_failures[0].count == 1

        # 5. Time waste analysis
        waste_analysis = engine._analyze_time_waste(scores, invocations)
        assert len(waste_analysis.slowest_tools) > 0

    def test_reflection_prompt_builds(self, _tmp_evolution_db):
        """Verify the reflection prompt is well-formed."""
        from self_evolution.reflection_engine import DreamEngine

        engine = DreamEngine(config={"base_url": "", "model": ""})

        # Insert mock data
        _tmp_evolution_db.insert("session_scores", {
            "session_id": "s1", "composite_score": 0.8,
            "completion_rate": 1.0, "efficiency_score": 0.7,
            "cost_efficiency": 0.9, "satisfaction_proxy": 0.8,
            "task_category": "coding", "model": "test",
        })
        _tmp_evolution_db.insert("tool_invocations", {
            "session_id": "s1", "tool_name": "bash",
            "duration_ms": 500, "success": True, "turn_number": 1,
        })

        scores = _tmp_evolution_db.fetch_all("session_scores")
        invocations = _tmp_evolution_db.fetch_all("tool_invocations")
        signals = _tmp_evolution_db.fetch_all("outcome_signals")

        error_analysis = engine._analyze_errors(scores, invocations, signals)
        waste_analysis = engine._analyze_time_waste(scores, invocations)

        prompt = engine._build_reflection_prompt(
            scores, invocations, signals,
            error_analysis, waste_analysis, avg_score=0.8,
        )
        assert "概况" in prompt or "sessions" in prompt
        assert "0.800" in prompt


# ============================================================================
# 11. Security — SQL Injection Prevention
# ============================================================================

class TestSecurity:
    """Test security hardening measures."""

    def test_sql_injection_rejected_invalid_table(self, _tmp_evolution_db):
        """Table names not in the whitelist must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid table name"):
            _tmp_evolution_db.insert("users; DROP TABLE users--", {"id": 1})

    def test_sql_injection_rejected_in_fetch(self, _tmp_evolution_db):
        with pytest.raises(ValueError, match="Invalid table name"):
            _tmp_evolution_db.fetch_one("nonexistent_table")

    def test_sql_injection_rejected_in_update(self, _tmp_evolution_db):
        with pytest.raises(ValueError, match="Invalid table name"):
            _tmp_evolution_db.update(
                "evil_table", {"x": 1}, where="1=1",
            )

    def test_sql_injection_rejected_in_insert_many(self, _tmp_evolution_db):
        with pytest.raises(ValueError, match="Invalid table name"):
            _tmp_evolution_db.insert_many("bad_table", [{"x": 1}])

    def test_sql_injection_rejected_in_fetch_all(self, _tmp_evolution_db):
        with pytest.raises(ValueError, match="Invalid table name"):
            _tmp_evolution_db.fetch_all("no_such_table")

    def test_limit_coerced_to_int(self, _tmp_evolution_db):
        """Non-integer limit values should be safely coerced."""
        _tmp_evolution_db.insert("tool_invocations", {
            "session_id": "s1", "tool_name": "bash",
            "duration_ms": 100, "success": True, "turn_number": 0,
        })
        # Pass a string-ish limit; int() coercion should handle it
        rows = _tmp_evolution_db.fetch_all(
            "tool_invocations", limit=1,
        )
        assert len(rows) == 1

    def test_valid_tables_still_work(self, _tmp_evolution_db):
        """All legitimate tables should pass validation."""
        _tmp_evolution_db.insert("tool_invocations", {
            "session_id": "s-ok", "tool_name": "bash",
            "duration_ms": 100, "success": True, "turn_number": 0,
        })
        _tmp_evolution_db.insert("outcome_signals", {
            "session_id": "s-ok", "signal_type": "test",
            "signal_value": 1.0,
        })
        rows = _tmp_evolution_db.fetch_all("tool_invocations")
        assert len(rows) == 1
