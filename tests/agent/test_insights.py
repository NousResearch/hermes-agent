"""Tests for agent/insights.py — InsightsEngine analytics and reporting."""

import sqlite3
import time
import pytest

from hermes_state import SessionDB
from agent.insights import (
    InsightsEngine,
    _estimate_cost,
    _bar_chart,
    _safe_float,
)
from agent.usage_pricing import (
    format_duration_compact as _format_duration,
    has_known_pricing as _has_known_pricing,
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


    def test_hours_with_minutes(self):
        result = _format_duration(5400)  # 1.5 hours
        assert result == "1h 30m"




class TestBarChart:
    def test_basic_bars(self):
        bars = _bar_chart([10, 5, 0, 20], max_width=10)
        assert len(bars) == 4
        assert len(bars[3]) == 10  # max value gets full width
        assert len(bars[0]) == 5   # half of max
        assert bars[2] == ""       # zero gets empty


    def test_all_zeros(self):
        bars = _bar_chart([0, 0, 0], max_width=10)
        assert all(b == "" for b in bars)



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


    def test_overview_token_totals(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        overview = report["overview"]

        expected_input = 50000 + 20000 + 100000 + 10000
        expected_output = 15000 + 8000 + 40000 + 5000
        assert overview["total_input_tokens"] == expected_input
        assert overview["total_output_tokens"] == expected_output
        assert overview["total_tokens"] == expected_input + expected_output




    def test_model_breakdown_splits_mid_session_switch(self, db):
        """A session that switches models mid-flight is split across both
        models in the breakdown, not dumped on the initial model (#51607).
        """
        now = time.time()
        db.create_session(session_id="sw", source="cli",
                          model="deepseek/deepseek-v4-pro")
        # 40k tokens on deepseek, then switch and 50k on opus.
        db.update_token_counts("sw", input_tokens=40000, output_tokens=8000,
                               model="deepseek/deepseek-v4-pro",
                               billing_provider="deepseek", api_call_count=2)
        db.update_session_model("sw", "anthropic/claude-opus-4.8")
        db.update_token_counts("sw", input_tokens=50000, output_tokens=4000,
                               model="anthropic/claude-opus-4.8",
                               billing_provider="openrouter", api_call_count=3)
        db._conn.commit()

        report = InsightsEngine(db).generate(days=30)
        models = {m["model"]: m for m in report["models"]}
        assert "deepseek-v4-pro" in models
        assert "claude-opus-4.8" in models
        # Tokens attributed to the model that actually incurred them.
        assert models["deepseek-v4-pro"]["input_tokens"] == 40000
        assert models["claude-opus-4.8"]["input_tokens"] == 50000
        assert models["claude-opus-4.8"]["api_calls"] == 3
        # The summary row's single model would have hidden one of these.
        assert models["deepseek-v4-pro"]["total_tokens"] == 48000
        assert models["claude-opus-4.8"]["total_tokens"] == 54000


    def test_overview_cost_matches_per_model_stored_cost(self, db):
        db.create_session(session_id="cost", source="cli", model="model-a")
        db.update_token_counts(
            "cost", input_tokens=10, model="model-a", billing_provider="custom",
            estimated_cost_usd=1.25, actual_cost_usd=1.0,
            cost_status="estimated", cost_source="provider", api_call_count=1,
        )
        db.update_session_model("cost", "model-b")
        db.update_session_billing_route("cost", provider="custom-b", base_url=None)
        db.update_token_counts(
            "cost", input_tokens=20, model="model-b", billing_provider="custom-b",
            estimated_cost_usd=2.5, actual_cost_usd=2.0,
            cost_status="estimated", cost_source="provider", api_call_count=1,
        )

        report = InsightsEngine(db).generate(days=30)
        assert sum(m["cost"] for m in report["models"]) == pytest.approx(3.75)
        assert report["overview"]["estimated_cost"] == pytest.approx(3.75)
        assert report["overview"]["actual_cost"] == pytest.approx(3.0)


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


    def test_activity_patterns(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        activity = report["activity"]

        assert len(activity["by_day"]) == 7
        assert len(activity["by_hour"]) == 24
        assert activity["active_days"] >= 1
        assert activity["busiest_day"] is not None
        assert activity["busiest_hour"] is not None







    # The Insights assistant tool-call queries pin
    # idx_messages_assistant_calls_by_session via INDEXED BY.  These tests prove
    # (a) the planner uses that index for BOTH the unfiltered and source-filtered
    # branches on a fresh DB *without* ANALYZE, and (b) the index is a pure
    # optimization — output is identical whether or not it is selected.
    _INDEX = "idx_messages_assistant_calls_by_session"
    _PINNED_QUERIES = (
        ("_GET_TOOL_CALLS_ALL", (0.0,)),
        ("_GET_TOOL_CALLS_WITH_SOURCE", (0.0, "cli")),
        ("_GET_SKILL_CALLS_ALL", (0.0,)),
        ("_GET_SKILL_CALLS_WITH_SOURCE", (0.0, "cli")),
    )

    def test_assistant_call_queries_use_partial_index_without_analyze(
        self, populated_db
    ):
        """Every fixed-predicate branch selects the partial index on a fresh DB.

        No ANALYZE is run, so this covers the default-statistics case a freshly
        initialized state.db is actually in. Both the unfiltered and the
        source-filtered (``s.source = ?``) branches are checked.
        """
        # Guard against the fresh-DB planner regression the reviewers found:
        # without INDEXED BY the source-filtered branch fell back to
        # idx_messages_session_active.
        assert "ANALYZE" not in "".join(
            r["sql"] or ""
            for r in populated_db._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index'"
            )
        )
        for attr, params in self._PINNED_QUERIES:
            sql = getattr(InsightsEngine, attr)
            plan = "\n".join(
                row["detail"]
                for row in populated_db._conn.execute(
                    "EXPLAIN QUERY PLAN " + sql, params
                ).fetchall()
            )
            assert self._INDEX in plan, f"{attr} did not use the index:\n{plan}"

    def test_assistant_call_rows_invariant_to_index_selection(self, populated_db):
        """The pinned index only changes the plan, never the result set.

        For every branch, the index-pinned query and the un-pinned form (whose
        plan the optimizer chooses freely) must return identical rows — proving
        the index is a pure optimization — for both the unfiltered and
        source-filtered scopes.
        """
        assert populated_db._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
            (self._INDEX,),
        ).fetchone() is not None

        for attr, params in self._PINNED_QUERIES:
            pinned_sql = getattr(InsightsEngine, attr)
            unpinned_sql = pinned_sql.replace(f" INDEXED BY {self._INDEX}", "")
            pinned = [
                tuple(r) for r in
                populated_db._conn.execute(pinned_sql, params).fetchall()
            ]
            unpinned = [
                tuple(r) for r in
                populated_db._conn.execute(unpinned_sql, params).fetchall()
            ]
            assert sorted(pinned) == sorted(unpinned), attr

    def test_tool_and_skill_usage_invariant_to_partial_index(self, populated_db):
        """The public tool/skill usage output is stable and exercises the
        assistant tool_calls path for both scopes."""
        engine = InsightsEngine(populated_db)
        cutoff = 0.0

        tools = engine._get_tool_usage(cutoff)
        tools_cli = engine._get_tool_usage(cutoff, source="cli")
        skills = engine._get_skill_usage(cutoff)
        skills_cli = engine._get_skill_usage(cutoff, source="cli")

        # Sanity: the fixture actually drives the assistant tool_calls path.
        assert any(t["tool_name"] == "search_files" for t in tools)
        assert any(t["tool_name"] == "search_files" for t in tools_cli)
        assert isinstance(skills, list) and isinstance(skills_cli, list)

    def test_missing_index_falls_back_to_unpinned_queries(self, populated_db):
        """INDEXED BY would be a hard error if the index is missing — which
        happens on read-only opens of a state.db written by an older version
        (web dashboard analytics). The engine must probe and fall back to the
        unpinned variants instead of crashing, returning identical rows."""
        engine_pinned = InsightsEngine(populated_db)
        tools_before = engine_pinned._get_tool_usage(0.0)

        populated_db._conn.execute(f"DROP INDEX IF EXISTS {self._INDEX}")
        populated_db._conn.commit()

        engine = InsightsEngine(populated_db)
        assert engine._has_assistant_calls_index is False
        assert "INDEXED BY" not in engine._GET_TOOL_CALLS_ALL
        tools_after = engine._get_tool_usage(0.0)
        assert sorted(t["tool_name"] for t in tools_after) == sorted(
            t["tool_name"] for t in tools_before
        )
        # And with the index present, the pin stays.
        assert "INDEXED BY" in InsightsEngine._GET_TOOL_CALLS_ALL

    def test_get_skill_breakdown_matches_full_generate(self, populated_db):
        engine = InsightsEngine(populated_db)
        full = engine.generate(days=30)
        focused = engine.get_usage_breakdown(days=30)["skills"]
        assert focused == full["skills"]

    def test_get_usage_breakdown_matches_full_generate(self, populated_db):
        engine = InsightsEngine(populated_db)
        full = engine.generate(days=30)
        focused = engine.get_usage_breakdown(days=30)
        assert focused["skills"] == full["skills"]
        assert focused["tools"] == full["tools"]

    def test_get_skill_breakdown_respects_source_filter(self, populated_db):
        engine = InsightsEngine(populated_db)
        # Only s1 (cli) has skill_view "github-pr-workflow"
        focused = engine.get_usage_breakdown(days=30, source="cli")["skills"]
        skill_names = [s["skill"] for s in focused["top_skills"]]
        assert "github-pr-workflow" in skill_names
        # github-code-review was in discord (s4), not cli
        assert "github-code-review" not in skill_names

    def test_get_skill_breakdown_empty_db(self, db):
        focused = InsightsEngine(db).get_usage_breakdown(days=30)["skills"]
        assert focused == {
            "summary": {
                "total_skill_loads": 0,
                "total_skill_edits": 0,
                "total_skill_actions": 0,
                "distinct_skills_used": 0,
            },
            "top_skills": [],
        }

    def test_get_skill_usage_prefilter_ignores_non_skill_substring(self, db):
        # "my_skill_view_helper" contains "skill_view" as a substring; instr()
        # will match but the Python-side name check keeps the set clean.
        # More importantly, messages with no skill_* tools must be excluded.
        db.create_session(session_id="sx", source="cli", model="gpt-4o")
        db.append_message(
            "sx",
            role="assistant",
            content="Just using read_file.",
            tool_calls=[{"function": {"name": "read_file", "arguments": '{"path":"/tmp/x"}'}}],
        )
        db._conn.commit()
        focused = InsightsEngine(db).get_usage_breakdown(days=30)["skills"]
        assert focused["summary"]["total_skill_actions"] == 0
        assert focused["top_skills"] == []


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




    def test_terminal_format_unknown_bucket_for_custom_models(self, db):
        """Custom models with no pricing surface as the Unknown bucket (#77223)."""
        db.create_session(session_id="s1", source="cli", model="my-custom-model")
        db.update_token_counts("s1", input_tokens=1000, output_tokens=500)
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        assert "N/A" not in text
        assert "custom/self-hosted" not in text
        # Cost section surfaces unknown-cost sessions (#77223) instead of
        # hiding them — a custom model with no pricing data shows in the
        # Unknown bucket rather than silently reporting $0.
        assert "Unknown" in text
        assert "no pricing data" in text


class TestGatewayFormatting:
    def test_gateway_format_is_shorter(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        terminal_text = engine.format_terminal(report)
        gateway_text = engine.format_gateway(report)

        assert len(gateway_text) < len(terminal_text)


    def test_gateway_format_hides_cache_details(self, populated_db):
        """Gateway format omits internal cache details.

        Dollar figures now appear when there are estimated/included/unknown
        cost buckets (#77223) — the old assertion that '$' is absent is no
        longer correct because surfacing cost buckets is the fix.
        """
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_gateway(report)

        assert "cache" not in text.lower()



# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:


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


    def test_cost_buckets_displayed_in_terminal_format(self, db):
        """#77223: included/estimated/unknown cost buckets surface in terminal."""
        # Estimated cost session
        db.create_session(session_id="est", source="cli", model="model-a")
        db.update_token_counts(
            "est", input_tokens=100, model="model-a",
            billing_provider="custom",
            estimated_cost_usd=1.50, actual_cost_usd=1.0,
            cost_status="estimated", cost_source="provider", api_call_count=1,
        )
        # Included cost session (subscription)
        db.create_session(session_id="inc", source="cli", model="gpt-5.4-mini")
        db.update_token_counts(
            "inc", input_tokens=200, model="gpt-5.4-mini",
            billing_provider="openai-codex",
            estimated_cost_usd=0.0, actual_cost_usd=0.0,
            cost_status="included", cost_source="none", api_call_count=1,
        )

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        # The cost section should appear with the buckets this DB has
        # (estimated + included; no unknown-cost session is created here)
        assert "💰 Cost" in text
        assert "~$1.50" in text  # estimated
        assert "included" in text.lower()
        assert "subscription" in text.lower()

    def test_sub_cent_aggregate_estimated_cost_not_zero(self, db):
        """A sub-cent aggregate must not render 'Estimated: ~$0.00' (#79220).

        The insights formatters share format_cost_label with per-response
        labels; a cheap-model period totaling $0.0046 shows 4dp, not $0.00.
        """
        db.create_session(session_id="est", source="cli", model="model-a")
        db.update_token_counts(
            "est", input_tokens=100, model="model-a",
            billing_provider="custom",
            estimated_cost_usd=0.0046, actual_cost_usd=0.0,
            cost_status="estimated", cost_source="provider", api_call_count=1,
        )

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        terminal_text = engine.format_terminal(report)
        gateway_text = engine.format_gateway(report)

        assert "~$0.00\n" not in terminal_text
        assert "~$0.0046" in terminal_text
        assert "~$0.00 estimated" not in gateway_text
        assert "~$0.0046 estimated" in gateway_text

    def test_cost_buckets_displayed_in_gateway_format(self, db):
        """#77223: included/estimated/unknown cost buckets surface in gateway."""
        db.create_session(session_id="est", source="cli", model="model-a")
        db.update_token_counts(
            "est", input_tokens=100, model="model-a",
            billing_provider="custom",
            estimated_cost_usd=2.25, actual_cost_usd=0.0,
            cost_status="estimated", cost_source="provider", api_call_count=1,
        )
        db.create_session(session_id="inc", source="cli", model="gpt-5.4-mini")
        db.update_token_counts(
            "inc", input_tokens=200, model="gpt-5.4-mini",
            billing_provider="openai-codex",
            estimated_cost_usd=0.0, actual_cost_usd=0.0,
            cost_status="included", cost_source="none", api_call_count=1,
        )

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        text = engine.format_gateway(report)

        assert "**Cost:**" in text
        assert "~$2.25" in text
        assert "included" in text.lower()

    def test_unknown_bucket_shown_for_costless_session(self, db):
        """A session with no model still shows unknown cost bucket (#77223).

        The unknown bucket is surfaced so users can see they have sessions
        with no pricing data, rather than silently reporting $0.
        """
        db.create_session(session_id="s1", source="cli", model="test")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)
        # The session has no cost data, so it falls in the "unknown" bucket.
        assert "💰 Cost" in text
        assert "Unknown" in text


class TestSafeFloat:
    """Regression tests for _safe_float guarding corrupted DB values."""

    def test_normal_float(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float(0) == 0.0

    def test_none_returns_zero(self):
        assert _safe_float(None) == 0.0

    def test_corrupted_string_returns_zero(self):
        assert _safe_float("corrupted") == 0.0
        assert _safe_float("not a number") == 0.0

    def test_safe_int_corrupted_string_returns_zero(self):
        from agent.insights import _safe_int
        assert _safe_int("corrupted") == 0
        assert _safe_int("not a number") == 0
        assert _safe_int(None) == 0

    def test_numeric_string_coerces(self):
        assert _safe_float("3.14") == 3.14
        assert _safe_float("42") == 42.0

    def test_boolean_coerces(self):
        assert _safe_float(True) == 1.0
        assert _safe_float(False) == 0.0


class TestInsightsCorruptedCost:
    """Regression: non-numeric cost columns must not crash /insights."""

    def test_corrupted_estimated_cost_usd_does_not_crash(self, db):
        """_safe_float neutralizes a string in estimated_cost_usd."""
        now = time.time()
        db.create_session(session_id="s1", source="cli", model="gpt-4o")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = 's1'", (now - 3600,)
        )
        db.update_token_counts("s1", input_tokens=1000, output_tokens=500)
        # Inject corrupted cost value directly into session_model_usage
        try:
            db._conn.execute("""
                INSERT INTO session_model_usage
                (session_id, model, billing_provider, billing_base_url,
                 api_call_count, input_tokens, output_tokens,
                 cache_read_tokens, cache_write_tokens, reasoning_tokens,
                 estimated_cost_usd, actual_cost_usd, cost_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("s1", "gpt-4o", "openai", "",
                  1, 1000, 500, 0, 0, 0,
                  "corrupted", 0.0, "included"))
        except Exception:
            # Table might not exist in older DB schemas
            pytest.skip("session_model_usage table not available")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        # Must not crash — estimated_cost should be 0 for the corrupted row
        assert report["overview"]["estimated_cost"] >= 0

    def test_corrupted_actual_cost_usd_does_not_crash(self, db):
        """_safe_float neutralizes a string in actual_cost_usd."""
        now = time.time()
        db.create_session(session_id="s1", source="cli", model="gpt-4o")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = 's1'", (now - 3600,)
        )
        db.update_token_counts("s1", input_tokens=1000, output_tokens=500)
        try:
            db._conn.execute("""
                INSERT INTO session_model_usage
                (session_id, model, billing_provider, billing_base_url,
                 api_call_count, input_tokens, output_tokens,
                 cache_read_tokens, cache_write_tokens, reasoning_tokens,
                 estimated_cost_usd, actual_cost_usd, cost_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("s1", "gpt-4o", "openai", "",
                  1, 1000, 500, 0, 0, 0,
                  0.0001, "corrupted", "included"))
        except Exception:
            pytest.skip("session_model_usage table not available")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["overview"]["estimated_cost"] >= 0

    def test_corrupted_api_call_count_does_not_crash(self, db):
        """_safe_int neutralizes a non-numeric api_call_count."""
        now = time.time()
        db.create_session(session_id="s1", source="cli", model="gpt-4o")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = 's1'", (now - 3600,)
        )
        db.update_token_counts("s1", input_tokens=1000, output_tokens=500)
        try:
            db._conn.execute("""
                INSERT INTO session_model_usage
                (session_id, model, billing_provider, billing_base_url,
                 api_call_count, input_tokens, output_tokens,
                 cache_read_tokens, cache_write_tokens, reasoning_tokens,
                 estimated_cost_usd, actual_cost_usd, cost_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("s1", "gpt-4o", "openai", "",
                  "corrupted", 1000, 500, 0, 0, 0,
                  0.0001, 0.0, "included"))
        except Exception:
            pytest.skip("session_model_usage table not available")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["overview"]["total_input_tokens"] >= 0


    def test_corrupted_input_tokens_in_usage_does_not_crash(self, db):
        """_safe_int neutralizes corrupted input_tokens in _accumulate path."""
        now = time.time()
        db.create_session(session_id="s1", source="cli", model="gpt-4o")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = 's1'", (now - 3600,)
        )
        try:
            db._conn.execute("""
                INSERT INTO session_model_usage
                (session_id, model, billing_provider, billing_base_url,
                 api_call_count, input_tokens, output_tokens,
                 cache_read_tokens, cache_write_tokens, reasoning_tokens,
                 estimated_cost_usd, actual_cost_usd, cost_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("s1", "gpt-4o", "openai", "",
                  1, "corrupted", 500, 0, 0, 0,
                  0.0, 0.0, "included"))
        except Exception:
            pytest.skip("session_model_usage table not available")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["overview"]["total_input_tokens"] >= 0

    def test_corrupted_aggregate_input_tokens_does_not_crash(self, db):
        """_safe_int guards aggregate sessions.input_tokens in platform breakdown."""
        now = time.time()
        db.create_session(session_id="s1", source="cli", model="gpt-4o")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = 's1'", (now - 3600,)
        )
        # Inject corrupted token value into the sessions aggregate row
        db._conn.execute(
            "UPDATE sessions SET input_tokens = ? WHERE id = 's1'",
            ("corrupted",)
        )
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["overview"]["total_input_tokens"] >= 0
