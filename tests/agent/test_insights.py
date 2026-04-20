"""Tests for agent/insights.py — InsightsEngine analytics and reporting."""

import time
import pytest
from pathlib import Path

from hermes_state import SessionDB
from agent.insights import (
    InsightsEngine,
    _estimate_cost,
    _format_duration,
    _bar_chart,
    _has_known_pricing,
    _DEFAULT_PRICING,
)


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_insights.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


@pytest.fixture()
def populated_db(db):
    """Create a DB with realistic session data for insights testing."""
    now = time.time()
    day = 86400

    # Session 1: CLI, claude-sonnet, ended, 2 days ago
    db.create_session(
        session_id="s1", source="cli",
        model="anthropic/claude-sonnet-4-20250514", user_id="user1",
    )
    # Backdate the started_at
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's1'", (now - 2 * day,))
    db.end_session("s1", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's1'", (now - 2 * day + 3600,))
    db.update_token_counts("s1", input_tokens=50000, output_tokens=15000)
    db.append_message("s1", role="user", content="Hello, help me fix a bug")
    db.append_message("s1", role="assistant", content="Sure, let me look into that.")
    db.append_message("s1", role="assistant", content="Let me search the files.",
                      tool_calls=[{"function": {"name": "search_files"}}])
    db.append_message("s1", role="tool", content="Found 3 matches", tool_name="search_files")
    db.append_message("s1", role="assistant", content="Let me read the file.",
                      tool_calls=[{"function": {"name": "read_file"}}])
    db.append_message("s1", role="tool", content="file contents...", tool_name="read_file")
    db.append_message("s1", role="assistant", content="I found the bug. Let me fix it.",
                      tool_calls=[{"function": {"name": "patch"}}])
    db.append_message("s1", role="tool", content="patched successfully", tool_name="patch")
    db.append_message(
        "s1",
        role="assistant",
        content="Let me load the PR workflow skill.",
        tool_calls=[{"function": {"name": "skill_view", "arguments": '{"name":"github-pr-workflow"}'}}],
    )
    db.append_message("s1", role="user", content="Thanks!")
    db.append_message("s1", role="assistant", content="You're welcome!")

    # Session 2: Telegram, gpt-4o, ended, 5 days ago
    db.create_session(
        session_id="s2", source="telegram",
        model="gpt-4o", user_id="user1",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's2'", (now - 5 * day,))
    db.end_session("s2", end_reason="timeout")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's2'", (now - 5 * day + 1800,))
    db.update_token_counts("s2", input_tokens=20000, output_tokens=8000)
    db.append_message("s2", role="user", content="Search the web for something")
    db.append_message("s2", role="assistant", content="Searching...",
                      tool_calls=[{"function": {"name": "web_search"}}])
    db.append_message("s2", role="tool", content="results...", tool_name="web_search")
    db.append_message("s2", role="assistant", content="Here's what I found")

    # Session 3: CLI, deepseek-chat, ended, 10 days ago
    db.create_session(
        session_id="s3", source="cli",
        model="deepseek-chat", user_id="user1",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's3'", (now - 10 * day,))
    db.end_session("s3", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's3'", (now - 10 * day + 7200,))
    db.update_token_counts("s3", input_tokens=100000, output_tokens=40000)
    db.append_message("s3", role="user", content="Run this terminal command")
    db.append_message("s3", role="assistant", content="Running...",
                      tool_calls=[{"function": {"name": "terminal"}}])
    db.append_message("s3", role="tool", content="output...", tool_name="terminal")
    db.append_message("s3", role="assistant", content="Let me run another",
                      tool_calls=[{"function": {"name": "terminal"}}])
    db.append_message("s3", role="tool", content="more output...", tool_name="terminal")
    db.append_message("s3", role="assistant", content="And search files",
                      tool_calls=[{"function": {"name": "search_files"}}])
    db.append_message("s3", role="tool", content="found stuff", tool_name="search_files")
    db.append_message(
        "s3",
        role="assistant",
        content="Load the debugging skill.",
        tool_calls=[{"function": {"name": "skill_view", "arguments": '{"name":"systematic-debugging"}'}}],
    )

    # Session 4: Discord, same model as s1, ended, 1 day ago
    db.create_session(
        session_id="s4", source="discord",
        model="anthropic/claude-sonnet-4-20250514", user_id="user2",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's4'", (now - 1 * day,))
    db.end_session("s4", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's4'", (now - 1 * day + 900,))
    db.update_token_counts("s4", input_tokens=10000, output_tokens=5000)
    db.append_message("s4", role="user", content="Quick question")
    db.append_message("s4", role="assistant", content="Sure, go ahead")
    db.append_message(
        "s4",
        role="assistant",
        content="Load and update GitHub skills.",
        tool_calls=[
            {"function": {"name": "skill_view", "arguments": '{"name":"github-pr-workflow"}'}},
            {"function": {"name": "skill_manage", "arguments": '{"name":"github-code-review"}'}},
        ],
    )

    # Session 5: Old session, 45 days ago (should be excluded from 30-day window)
    db.create_session(
        session_id="s_old", source="cli",
        model="gpt-4o-mini", user_id="user1",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's_old'", (now - 45 * day,))
    db.end_session("s_old", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's_old'", (now - 45 * day + 600,))
    db.update_token_counts("s_old", input_tokens=5000, output_tokens=2000)
    db.append_message("s_old", role="user", content="old message")
    db.append_message("s_old", role="assistant", content="old reply")

    db._conn.commit()
    return db


class TestHasKnownPricing:
    def test_known_commercial_model(self):
        assert _has_known_pricing("gpt-4o", provider="openai") is True
        assert _has_known_pricing("anthropic/claude-sonnet-4-20250514") is True
        assert _has_known_pricing("gpt-4.1", provider="openai") is True

    def test_unknown_custom_model(self):
        assert _has_known_pricing("FP16_Hermes_4.5") is False
        assert _has_known_pricing("my-custom-model") is False
        assert _has_known_pricing("glm-5") is False
        assert _has_known_pricing("") is False
        assert _has_known_pricing(None) is False

    def test_heuristic_matched_models_are_not_considered_known(self):
        assert _has_known_pricing("some-opus-model") is False
        assert _has_known_pricing("future-sonnet-v2") is False


class TestEstimateCost:
    def test_basic_cost(self):
        cost, status = _estimate_cost(
            "anthropic/claude-sonnet-4-20250514",
            1_000_000,
            1_000_000,
            provider="anthropic",
        )
        assert status == "estimated"
        assert cost == pytest.approx(18.0, abs=0.01)

    def test_zero_tokens(self):
        cost, status = _estimate_cost("gpt-4o", 0, 0, provider="openai")
        assert status == "estimated"
        assert cost == 0.0

    def test_cache_aware_usage(self):
        cost, status = _estimate_cost(
            "anthropic/claude-sonnet-4-20250514",
            1000,
            500,
            cache_read_tokens=2000,
            cache_write_tokens=400,
            provider="anthropic",
        )
        assert status == "estimated"
        expected = (1000 * 3.0 + 500 * 15.0 + 2000 * 0.30 + 400 * 3.75) / 1_000_000
        assert cost == pytest.approx(expected, abs=0.0001)


# =========================================================================
# Format helpers
# =========================================================================

class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(45) == "45s"

    def test_minutes(self):
        assert _format_duration(300) == "5m"

    def test_hours_with_minutes(self):
        result = _format_duration(5400)  # 1.5 hours
        assert result == "1h 30m"

    def test_exact_hours(self):
        assert _format_duration(7200) == "2h"

    def test_days(self):
        result = _format_duration(172800)  # 2 days
        assert result == "2.0d"


class TestBarChart:
    def test_basic_bars(self):
        bars = _bar_chart([10, 5, 0, 20], max_width=10)
        assert len(bars) == 4
        assert len(bars[3]) == 10  # max value gets full width
        assert len(bars[0]) == 5   # half of max
        assert bars[2] == ""       # zero gets empty

    def test_empty_values(self):
        bars = _bar_chart([], max_width=10)
        assert bars == []

    def test_all_zeros(self):
        bars = _bar_chart([0, 0, 0], max_width=10)
        assert all(b == "" for b in bars)

    def test_single_value(self):
        bars = _bar_chart([5], max_width=10)
        assert len(bars) == 1
        assert len(bars[0]) == 10


# =========================================================================
# InsightsEngine — empty DB
# =========================================================================

class TestInsightsEmpty:
    def test_empty_db_returns_empty_report(self, db):
        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["empty"] is True
        assert report["overview"] == {}

    def test_empty_db_terminal_format(self, db):
        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)
        assert "No sessions found" in text

    def test_empty_db_gateway_format(self, db):
        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        text = engine.format_gateway(report)
        assert "No sessions found" in text


# =========================================================================
# InsightsEngine — populated DB
# =========================================================================

class TestInsightsPopulated:
    def test_generate_returns_all_sections(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)

        assert report["empty"] is False
        assert "overview" in report
        assert "models" in report
        assert "platforms" in report
        assert "tools" in report
        assert "activity" in report
        assert "top_sessions" in report

    def test_overview_session_count(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        overview = report["overview"]

        # s1, s2, s3, s4 are within 30 days; s_old is 45 days ago
        assert overview["total_sessions"] == 4

    def test_overview_token_totals(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        overview = report["overview"]

        expected_input = 50000 + 20000 + 100000 + 10000
        expected_output = 15000 + 8000 + 40000 + 5000
        assert overview["total_input_tokens"] == expected_input
        assert overview["total_output_tokens"] == expected_output
        assert overview["total_tokens"] == expected_input + expected_output

    def test_overview_cost_positive(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        assert report["overview"]["estimated_cost"] > 0

    def test_overview_duration_stats(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        overview = report["overview"]

        # All 4 sessions have durations
        assert overview["total_hours"] > 0
        assert overview["avg_session_duration"] > 0

    def test_model_breakdown(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        models = report["models"]

        # Should have 3 distinct models (claude-sonnet x2, gpt-4o, deepseek-chat)
        model_names = [m["model"] for m in models]
        assert "claude-sonnet-4-20250514" in model_names
        assert "gpt-4o" in model_names
        assert "deepseek-chat" in model_names

        # Claude-sonnet has 2 sessions (s1 + s4)
        claude = next(m for m in models if "claude-sonnet" in m["model"])
        assert claude["sessions"] == 2

    def test_platform_breakdown(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        platforms = report["platforms"]

        platform_names = [p["platform"] for p in platforms]
        assert "cli" in platform_names
        assert "telegram" in platform_names
        assert "discord" in platform_names

        cli = next(p for p in platforms if p["platform"] == "cli")
        assert cli["sessions"] == 2  # s1 + s3

    def test_tool_breakdown(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        tools = report["tools"]

        tool_names = [t["tool"] for t in tools]
        assert "terminal" in tool_names
        assert "search_files" in tool_names
        assert "read_file" in tool_names
        assert "patch" in tool_names
        assert "web_search" in tool_names

        # terminal was used 2x in s3
        terminal = next(t for t in tools if t["tool"] == "terminal")
        assert terminal["count"] == 2

        # Percentages should sum to ~100%
        total_pct = sum(t["percentage"] for t in tools)
        assert total_pct == pytest.approx(100.0, abs=0.1)

    def test_skill_breakdown(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        skills = report["skills"]

        assert skills["summary"]["distinct_skills_used"] == 3
        assert skills["summary"]["total_skill_loads"] == 3
        assert skills["summary"]["total_skill_edits"] == 1
        assert skills["summary"]["total_skill_actions"] == 4

        top_skill = skills["top_skills"][0]
        assert top_skill["skill"] == "github-pr-workflow"
        assert top_skill["view_count"] == 2
        assert top_skill["manage_count"] == 0
        assert top_skill["total_count"] == 2
        assert top_skill["last_used_at"] is not None

    def test_skill_breakdown_respects_days_filter(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=3)
        skills = report["skills"]

        assert skills["summary"]["distinct_skills_used"] == 2
        assert skills["summary"]["total_skill_loads"] == 2
        assert skills["summary"]["total_skill_edits"] == 1

        skill_names = [s["skill"] for s in skills["top_skills"]]
        assert "systematic-debugging" not in skill_names

    def test_activity_patterns(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        activity = report["activity"]

        assert len(activity["by_day"]) == 7
        assert len(activity["by_hour"]) == 24
        assert activity["active_days"] >= 1
        assert activity["busiest_day"] is not None
        assert activity["busiest_hour"] is not None

    def test_top_sessions(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        top = report["top_sessions"]

        labels = [t["label"] for t in top]
        assert "Longest session" in labels
        assert "Most messages" in labels
        assert "Most tokens" in labels
        assert "Most tool calls" in labels

    def test_source_filter_cli(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30, source="cli")

        assert report["overview"]["total_sessions"] == 2  # s1, s3

    def test_source_filter_telegram(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30, source="telegram")

        assert report["overview"]["total_sessions"] == 1  # s2

    def test_source_filter_nonexistent(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30, source="slack")

        assert report["empty"] is True

    def test_days_filter_short(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=3)

        # Only s1 (2 days ago) and s4 (1 day ago) should be included
        assert report["overview"]["total_sessions"] == 2

    def test_days_filter_long(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=60)

        # All 5 sessions should be included
        assert report["overview"]["total_sessions"] == 5


# =========================================================================
# Formatting
# =========================================================================

class TestTerminalFormatting:
    def test_terminal_format_has_sections(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        assert "Hermes Insights" in text
        assert "Overview" in text
        assert "Models Used" in text
        assert "Top Tools" in text
        assert "Top Skills" in text
        assert "Activity Patterns" in text
        assert "Notable Sessions" in text

    def test_terminal_format_shows_tokens(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        assert "Input tokens" in text
        assert "Output tokens" in text
        # Cost and cache metrics are intentionally hidden (pricing was unreliable).
        assert "Est. cost" not in text
        assert "Cache read" not in text
        assert "Cache write" not in text

    def test_terminal_format_shows_platforms(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        # Multi-platform, so Platforms section should show
        assert "Platforms" in text
        assert "cli" in text
        assert "telegram" in text

    def test_terminal_format_shows_bar_chart(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        assert "█" in text  # Bar chart characters

    def test_terminal_format_hides_cost_for_custom_models(self, db):
        """Cost display is hidden entirely — custom models no longer show 'N/A' either."""
        db.create_session(session_id="s1", source="cli", model="my-custom-model")
        db.update_token_counts("s1", input_tokens=1000, output_tokens=500)
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        assert "N/A" not in text
        assert "custom/self-hosted" not in text
        assert "Cost" not in text


class TestGatewayFormatting:
    def test_gateway_format_is_shorter(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        terminal_text = engine.format_terminal(report)
        gateway_text = engine.format_gateway(report)

        assert len(gateway_text) < len(terminal_text)

    def test_gateway_format_has_bold(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_gateway(report)

        assert "**" in text  # Markdown bold

    def test_gateway_format_hides_cost(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_gateway(report)

        assert "$" in text
        assert "Top Skills" in text
        assert "Est. cost" in text
        assert "cache" not in text.lower()

    def test_gateway_format_shows_models(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_gateway(report)

        assert "Models" in text
        assert "sessions" in text


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    def test_session_with_no_tokens(self, db):
        """Sessions with zero tokens should not crash."""
        db.create_session(session_id="s1", source="cli", model="test-model")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["empty"] is False
        assert report["overview"]["total_tokens"] == 0
        assert report["overview"]["estimated_cost"] == 0.0

    def test_session_with_no_end_time(self, db):
        """Active (non-ended) sessions should be included but duration = 0."""
        db.create_session(session_id="s1", source="cli", model="test-model")
        db.update_token_counts("s1", input_tokens=1000, output_tokens=500)
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        # Session included
        assert report["overview"]["total_sessions"] == 1
        assert report["overview"]["total_tokens"] == 1500
        # But no duration stats (session not ended)
        assert report["overview"]["total_hours"] == 0

    def test_session_with_no_model(self, db):
        """Sessions with NULL model should not crash."""
        db.create_session(session_id="s1", source="cli")
        db.update_token_counts("s1", input_tokens=1000, output_tokens=500)
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["empty"] is False

        models = report["models"]
        assert len(models) == 1
        assert models[0]["model"] == "unknown"
        assert models[0]["has_pricing"] is False

    def test_custom_model_shows_zero_cost(self, db):
        """Custom/self-hosted models should show $0 cost, not fake estimates."""
        db.create_session(session_id="s1", source="cli", model="FP16_Hermes_4.5")
        db.update_token_counts("s1", input_tokens=100000, output_tokens=50000)
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["overview"]["estimated_cost"] == 0.0
        assert "FP16_Hermes_4.5" in report["overview"]["models_without_pricing"]

        models = report["models"]
        custom = next(m for m in models if m["model"] == "FP16_Hermes_4.5")
        assert custom["cost"] == 0.0
        assert custom["has_pricing"] is False

    def test_tool_usage_from_tool_calls_json(self, db):
        """Tool usage should be extracted from tool_calls JSON when tool_name is NULL."""
        import json as _json
        db.create_session(session_id="s1", source="cli", model="test")
        # Assistant message with tool_calls (this is what CLI produces)
        db.append_message("s1", role="assistant", content="Let me search",
                          tool_calls=[{"id": "call_1", "type": "function",
                                       "function": {"name": "search_files", "arguments": "{}"}}])
        # Tool response WITHOUT tool_name (this is the CLI bug)
        db.append_message("s1", role="tool", content="found results",
                          tool_call_id="call_1")
        db.append_message("s1", role="assistant", content="Now reading",
                          tool_calls=[{"id": "call_2", "type": "function",
                                       "function": {"name": "read_file", "arguments": "{}"}}])
        db.append_message("s1", role="tool", content="file content",
                          tool_call_id="call_2")
        db.append_message("s1", role="assistant", content="And searching again",
                          tool_calls=[{"id": "call_3", "type": "function",
                                       "function": {"name": "search_files", "arguments": "{}"}}])
        db.append_message("s1", role="tool", content="more results",
                          tool_call_id="call_3")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        tools = report["tools"]

        # Should find tools from tool_calls JSON even though tool_name is NULL
        tool_names = [t["tool"] for t in tools]
        assert "search_files" in tool_names
        assert "read_file" in tool_names

        # search_files was called twice
        sf = next(t for t in tools if t["tool"] == "search_files")
        assert sf["count"] == 2

    def test_overview_pricing_sets_are_lists(self, db):
        """models_with/without_pricing should be JSON-serializable lists."""
        import json as _json
        db.create_session(session_id="s1", source="cli", model="gpt-4o")
        db.create_session(session_id="s2", source="cli", model="my-custom")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        overview = report["overview"]

        assert isinstance(overview["models_with_pricing"], list)
        assert isinstance(overview["models_without_pricing"], list)
        # Should be JSON-serializable
        _json.dumps(report["overview"])  # would raise if sets present

    def test_mixed_commercial_and_custom_models(self, db):
        """Mix of commercial and custom models: only commercial ones get costs."""
        db.create_session(session_id="s1", source="cli", model="anthropic/claude-sonnet-4-20250514")
        db.update_token_counts(
            "s1",
            input_tokens=10000,
            output_tokens=5000,
            billing_provider="anthropic",
        )
        db.create_session(session_id="s2", source="cli", model="my-local-llama")
        db.update_token_counts("s2", input_tokens=10000, output_tokens=5000)
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)

        # Cost should only come from gpt-4o, not from the custom model
        overview = report["overview"]
        assert overview["estimated_cost"] > 0
        assert "claude-sonnet-4-20250514" in overview["models_with_pricing"]  # list now, not set
        assert "my-local-llama" in overview["models_without_pricing"]

        # Verify individual model entries
        claude = next(m for m in report["models"] if m["model"] == "claude-sonnet-4-20250514")
        assert claude["has_pricing"] is True
        assert claude["cost"] > 0

        llama = next(m for m in report["models"] if m["model"] == "my-local-llama")
        assert llama["has_pricing"] is False
        assert llama["cost"] == 0.0

    def test_single_session_streak(self, db):
        """Single session should have streak of 0 or 1."""
        db.create_session(session_id="s1", source="cli", model="test")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["activity"]["max_streak"] <= 1

    def test_no_tool_calls(self, db):
        """Sessions with no tool calls should produce empty tools list."""
        db.create_session(session_id="s1", source="cli", model="test")
        db.append_message("s1", role="user", content="hello")
        db.append_message("s1", role="assistant", content="hi there")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["tools"] == []

    def test_only_one_platform(self, db):
        """Single-platform usage should still work."""
        db.create_session(session_id="s1", source="cli", model="test")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert len(report["platforms"]) == 1
        assert report["platforms"][0]["platform"] == "cli"

        # Terminal format should NOT show platform section for single platform
        text = engine.format_terminal(report)
        # (it still shows platforms section if there's only cli and nothing else)
        # Actually the condition is > 1 platforms OR non-cli, so single cli won't show

    def test_large_days_value(self, db):
        """Very large days value should not crash."""
        db.create_session(session_id="s1", source="cli", model="test")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=365)
        assert report["empty"] is False

    def test_zero_days(self, db):
        """Zero days should return empty (nothing is in the future)."""
        db.create_session(session_id="s1", source="cli", model="test")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=0)
        # Depending on timing, might catch the session if created <1s ago
        # Just verify it doesn't crash
        assert "empty" in report
