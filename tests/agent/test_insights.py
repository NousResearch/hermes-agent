"""Tests for agent/insights.py — InsightsEngine analytics and reporting."""

import time
import pytest

from agent.insights import (
    InsightsEngine,
    _estimate_cost,
    _bar_chart,
)
from persistence.sqlite.conversation.usage_analytics import _InsightQueries
from agent.usage_pricing import (
    format_duration_compact as _format_duration,
    has_known_pricing as _has_known_pricing,
)
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope, store_location
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session, seed_usage

@pytest.fixture()
def db(tmp_path):
    """A store handle on a temp profile home."""
    store = open_persistence(profile_scope(home=tmp_path))
    yield store
    store.close()


def _sql(store, sql, params=()):
    """One raw statement in a write transaction, for the timestamps and schema damage no verb writes."""
    raw_holder(store)._execute_write(lambda conn: conn.execute(sql, params))

@pytest.fixture()
def populated_db(db):
    """Create a DB with realistic session data for insights testing."""
    now = time.time()
    day = 86400

    # Session 1: CLI, claude-sonnet, ended, 2 days ago
    seed_session(
        db, session_id="s1", source="cli",
        model="anthropic/claude-sonnet-4-20250514", user_id="user1",
    )
    # Backdate the started_at
    _sql(db, "UPDATE sessions SET started_at = ? WHERE id = 's1'", (now - 2 * day,))
    db.sessions.end("s1", "user_exit")
    _sql(db, "UPDATE sessions SET ended_at = ? WHERE id = 's1'", (now - 2 * day + 3600,))
    seed_usage(db, "s1", input_tokens=50000, output_tokens=15000)
    seed_messages(db, "s1", dict(role="user", content="Hello, help me fix a bug"))
    seed_messages(db, "s1", dict(role="assistant", content="Sure, let me look into that."))
    seed_messages(db, "s1", dict(role="assistant", content="Let me search the files.",
                                 tool_calls=[{"function": {"name": "search_files"}}]))
    seed_messages(db, "s1", dict(role="tool", content="Found 3 matches", tool_name="search_files"))
    seed_messages(db, "s1", dict(role="assistant", content="Let me read the file.",
                                 tool_calls=[{"function": {"name": "read_file"}}]))
    seed_messages(db, "s1", dict(role="tool", content="file contents...", tool_name="read_file"))
    seed_messages(db, "s1", dict(role="assistant", content="I found the bug. Let me fix it.",
                                 tool_calls=[{"function": {"name": "patch"}}]))
    seed_messages(db, "s1", dict(role="tool", content="patched successfully", tool_name="patch"))
    seed_messages(db, "s1", dict(
        role="assistant",
        content="Let me load the PR workflow skill.",
        tool_calls=[{"function": {"name": "skill_view", "arguments": '{"name":"github-pr-workflow"}'}}],
    ))
    seed_messages(db, "s1", dict(role="user", content="Thanks!"))
    seed_messages(db, "s1", dict(role="assistant", content="You're welcome!"))

    # Session 2: Telegram, gpt-4o, ended, 5 days ago
    seed_session(
        db, session_id="s2", source="telegram",
        model="gpt-4o", user_id="user1",
    )
    _sql(db, "UPDATE sessions SET started_at = ? WHERE id = 's2'", (now - 5 * day,))
    db.sessions.end("s2", "timeout")
    _sql(db, "UPDATE sessions SET ended_at = ? WHERE id = 's2'", (now - 5 * day + 1800,))
    seed_usage(db, "s2", input_tokens=20000, output_tokens=8000)
    seed_messages(db, "s2", dict(role="user", content="Search the web for something"))
    seed_messages(db, "s2", dict(role="assistant", content="Searching...",
                                 tool_calls=[{"function": {"name": "web_search"}}]))
    seed_messages(db, "s2", dict(role="tool", content="results...", tool_name="web_search"))
    seed_messages(db, "s2", dict(role="assistant", content="Here's what I found"))

    # Session 3: CLI, deepseek-chat, ended, 10 days ago
    seed_session(
        db, session_id="s3", source="cli",
        model="deepseek-chat", user_id="user1",
    )
    _sql(db, "UPDATE sessions SET started_at = ? WHERE id = 's3'", (now - 10 * day,))
    db.sessions.end("s3", "user_exit")
    _sql(db, "UPDATE sessions SET ended_at = ? WHERE id = 's3'", (now - 10 * day + 7200,))
    seed_usage(db, "s3", input_tokens=100000, output_tokens=40000)
    seed_messages(db, "s3", dict(role="user", content="Run this terminal command"))
    seed_messages(db, "s3", dict(role="assistant", content="Running...",
                                 tool_calls=[{"function": {"name": "terminal"}}]))
    seed_messages(db, "s3", dict(role="tool", content="output...", tool_name="terminal"))
    seed_messages(db, "s3", dict(role="assistant", content="Let me run another",
                                 tool_calls=[{"function": {"name": "terminal"}}]))
    seed_messages(db, "s3", dict(role="tool", content="more output...", tool_name="terminal"))
    seed_messages(db, "s3", dict(role="assistant", content="And search files",
                                 tool_calls=[{"function": {"name": "search_files"}}]))
    seed_messages(db, "s3", dict(role="tool", content="found stuff", tool_name="search_files"))
    seed_messages(db, "s3", dict(
        role="assistant",
        content="Load the debugging skill.",
        tool_calls=[{"function": {"name": "skill_view", "arguments": '{"name":"systematic-debugging"}'}}],
    ))

    # Session 4: Discord, same model as s1, ended, 1 day ago
    seed_session(
        db, session_id="s4", source="discord",
        model="anthropic/claude-sonnet-4-20250514", user_id="user2",
    )
    _sql(db, "UPDATE sessions SET started_at = ? WHERE id = 's4'", (now - 1 * day,))
    db.sessions.end("s4", "user_exit")
    _sql(db, "UPDATE sessions SET ended_at = ? WHERE id = 's4'", (now - 1 * day + 900,))
    seed_usage(db, "s4", input_tokens=10000, output_tokens=5000)
    seed_messages(db, "s4", dict(role="user", content="Quick question"))
    seed_messages(db, "s4", dict(role="assistant", content="Sure, go ahead"))
    seed_messages(db, "s4", dict(
        role="assistant",
        content="Load and update GitHub skills.",
        tool_calls=[
            {"function": {"name": "skill_view", "arguments": '{"name":"github-pr-workflow"}'}},
            {"function": {"name": "skill_manage", "arguments": '{"name":"github-code-review"}'}},
        ],
    ))

    # Session 5: Old session, 45 days ago (should be excluded from 30-day window)
    seed_session(
        db, session_id="s_old", source="cli",
        model="gpt-4o-mini", user_id="user1",
    )
    _sql(db, "UPDATE sessions SET started_at = ? WHERE id = 's_old'", (now - 45 * day,))
    db.sessions.end("s_old", "user_exit")
    _sql(db, "UPDATE sessions SET ended_at = ? WHERE id = 's_old'", (now - 45 * day + 600,))
    seed_usage(db, "s_old", input_tokens=5000, output_tokens=2000)
    seed_messages(db, "s_old", dict(role="user", content="old message"))
    seed_messages(db, "s_old", dict(role="assistant", content="old reply"))

    return db

class TestHasKnownPricing:

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
        engine = InsightsEngine(db, persistence=db)
        report = engine.generate(days=30)
        assert report["empty"] is True
        assert report["overview"] == {}
        # Both renderers must handle the empty report without crashing.
        assert engine.format_terminal(report)
        assert engine.format_gateway(report)

# =========================================================================
# InsightsEngine — populated DB
# =========================================================================

class TestInsightsPopulated:

    def test_overview_token_totals(self, populated_db):
        engine = InsightsEngine(populated_db, persistence=populated_db)
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
        seed_session(db, session_id="sw", source="cli",
                         model="deepseek/deepseek-v4-pro")
        # 40k tokens on deepseek, then switch and 50k on opus.
        seed_usage(db, "sw", input_tokens=40000, output_tokens=8000,
                       model="deepseek/deepseek-v4-pro",
                       billing_provider="deepseek", api_call_count=2)
        db.sessions.update_model("sw", "anthropic/claude-opus-4.8")
        seed_usage(db, "sw", input_tokens=50000, output_tokens=4000,
                       model="anthropic/claude-opus-4.8",
                       billing_provider="openrouter", api_call_count=3)

        report = InsightsEngine(db, persistence=db).generate(days=30)
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
        seed_session(db, session_id="cost", source="cli", model="model-a")
        seed_usage(db,
            "cost", input_tokens=10, model="model-a", billing_provider="custom",
            estimated_cost_usd=1.25, actual_cost_usd=1.0,
            cost_status="estimated", cost_source="provider", api_call_count=1,
        )
        db.sessions.update_model("cost", "model-b")
        db.usage.set_billing_route("cost", provider="custom-b", base_url="")
        seed_usage(db,
            "cost", input_tokens=20, model="model-b", billing_provider="custom-b",
            estimated_cost_usd=2.5, actual_cost_usd=2.0,
            cost_status="estimated", cost_source="provider", api_call_count=1,
        )

        report = InsightsEngine(db, persistence=db).generate(days=30)
        assert sum(m["cost"] for m in report["models"]) == pytest.approx(3.75)
        assert report["overview"]["estimated_cost"] == pytest.approx(3.75)
        assert report["overview"]["actual_cost"] == pytest.approx(3.0)

    def test_tool_usage_sums_disjoint_sessions_without_double_counting_pairs(self, db):
        """One session records a call as tool_name only, another as tool_calls only: both count.
        A session carrying BOTH representations of the same call still counts it once (#9814)."""
        seed_session(db, session_id="gw", source="gateway", model="m")
        seed_messages(db, "gw", dict(role="tool", content="r", tool_name="search_files"))
        seed_session(db, session_id="cli", source="cli", model="m")
        seed_messages(db, "cli", dict(role="assistant", content="x",
                                      tool_calls=[{"function": {"name": "search_files", "arguments": "{}"}}]))
        seed_session(db, session_id="both", source="cli", model="m")
        seed_messages(db, "both", dict(role="assistant", content="x",
                                       tool_calls=[{"function": {"name": "search_files", "arguments": "{}"}}]))
        seed_messages(db, "both", dict(role="tool", content="r", tool_name="search_files"))

        tools = InsightsEngine(db, persistence=db).generate(days=30)["tools"]
        assert next(t["count"] for t in tools if t["tool"] == "search_files") == 3

    def test_tool_breakdown(self, populated_db):
        engine = InsightsEngine(populated_db, persistence=populated_db)
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
        engine = InsightsEngine(populated_db, persistence=populated_db)
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

    # The Insights assistant tool-call queries (the store's usage analytics reads) pin
    # idx_messages_assistant_calls_by_session via INDEXED BY.  These tests prove
    # (a) the planner uses that index for BOTH the unfiltered and source-filtered
    # branches on a fresh DB *without* ANALYZE, and (b) the index is a pure
    # optimization — output is identical whether or not it is selected.
    _INDEX = "idx_messages_assistant_calls_by_session"
    _PINNED_QUERIES = (
        ("_GET_TOOL_CALLS_ALL", (0.0,)),
        ("_GET_TOOL_CALLS_WITH_SOURCE", (0.0, "cli", "cli")),  # label, then created_source of a local row
        ("_GET_SKILL_CALLS_ALL", (0.0,)),
        ("_GET_SKILL_CALLS_WITH_SOURCE", (0.0, "cli", "cli")),
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
        with raw_holder(populated_db)._read_ctx() as conn:
            assert "ANALYZE" not in "".join(
                r["sql"] or ""
                for r in conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'index'"
                )
            )
            for attr, params in self._PINNED_QUERIES:
                sql = getattr(_InsightQueries, attr)
                plan = "\n".join(
                    row["detail"]
                    for row in conn.execute(
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
        with raw_holder(populated_db)._read_ctx() as conn:
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
                (self._INDEX,),
            ).fetchone() is not None

            for attr, params in self._PINNED_QUERIES:
                pinned_sql = getattr(_InsightQueries, attr)
                unpinned_sql = pinned_sql.replace(f" INDEXED BY {self._INDEX}", "")
                pinned = [
                    tuple(r) for r in
                    conn.execute(pinned_sql, params).fetchall()
                ]
                unpinned = [
                    tuple(r) for r in
                    conn.execute(unpinned_sql, params).fetchall()
                ]
                assert sorted(pinned) == sorted(unpinned), attr

    def test_missing_index_falls_back_to_unpinned_queries(self, populated_db):
        """INDEXED BY would be a hard error if the index is missing — which
        happens on read-only opens of a state.db written by an older version
        (web dashboard analytics). The engine must probe and fall back to the
        unpinned variants instead of crashing, returning identical rows."""
        engine_pinned = InsightsEngine(populated_db, persistence=populated_db)
        tools_before = engine_pinned._get_tool_usage(0.0)

        _sql(populated_db, f"DROP INDEX IF EXISTS {self._INDEX}")

        with open_persistence_reader(store_location(populated_db.scope)) as reader:
            engine = InsightsEngine(None, persistence=reader)
            with raw_holder(populated_db)._read_ctx() as conn:
                queries = _InsightQueries(conn)
            assert queries._has_assistant_calls_index is False
            assert "INDEXED BY" not in queries._GET_TOOL_CALLS_ALL
            tools_after = engine._get_tool_usage(0.0)
        assert sorted(t["tool_name"] for t in tools_after) == sorted(
            t["tool_name"] for t in tools_before
        )
        # And with the index present, the pin stays.
        assert "INDEXED BY" in _InsightQueries._GET_TOOL_CALLS_ALL

    def test_get_usage_breakdown_matches_full_generate(self, populated_db):
        engine = InsightsEngine(populated_db, persistence=populated_db)
        full = engine.generate(days=30)
        focused = engine.get_usage_breakdown(days=30)
        assert focused["skills"] == full["skills"]
        assert focused["tools"] == full["tools"]

    def test_get_skill_breakdown_respects_source_filter(self, populated_db):
        engine = InsightsEngine(populated_db, persistence=populated_db)
        # Only s1 (cli) has skill_view "github-pr-workflow"
        focused = engine.get_usage_breakdown(days=30, source="cli")["skills"]
        skill_names = [s["skill"] for s in focused["top_skills"]]
        assert "github-pr-workflow" in skill_names
        # github-code-review was in discord (s4), not cli
        assert "github-code-review" not in skill_names

    def test_get_skill_usage_prefilter_ignores_non_skill_substring(self, db):
        # "my_skill_view_helper" contains "skill_view" as a substring; instr()
        # will match but the Python-side name check keeps the set clean.
        # More importantly, messages with no skill_* tools must be excluded.
        seed_session(db, session_id="sx", source="cli", model="gpt-4o")
        seed_messages(db, "sx", dict(
            role="assistant",
            content="Just using read_file.",
            tool_calls=[{"function": {"name": "read_file", "arguments": '{"path":"/tmp/x"}'}}],
        ))
        focused = InsightsEngine(db, persistence=db).get_usage_breakdown(days=30)["skills"]
        assert focused["summary"]["total_skill_actions"] == 0
        assert focused["top_skills"] == []

# =========================================================================
# Formatting
# =========================================================================

class TestTerminalFormatting:

    def test_terminal_format_unknown_bucket_for_custom_models(self, db):
        """Custom models with no pricing surface as the Unknown bucket (#77223)."""
        seed_session(db, session_id="s1", source="cli", model="my-custom-model")
        seed_usage(db, "s1", input_tokens=1000, output_tokens=500)

        engine = InsightsEngine(db, persistence=db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        # Cost section surfaces unknown-cost sessions (#77223) instead of
        # hiding them — a custom model with no pricing data shows in the
        # Unknown bucket rather than silently reporting $0.
        assert "Unknown" in text

class TestGatewayFormatting:
    def test_gateway_format_is_shorter(self, populated_db):
        engine = InsightsEngine(populated_db, persistence=populated_db)
        report = engine.generate(days=30)
        terminal_text = engine.format_terminal(report)
        gateway_text = engine.format_gateway(report)

        assert len(gateway_text) < len(terminal_text)

# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:

    def test_session_with_no_model(self, db):
        """Sessions with NULL model should not crash."""
        seed_session(db, session_id="s1", source="cli")
        seed_usage(db, "s1", input_tokens=1000, output_tokens=500)

        engine = InsightsEngine(db, persistence=db)
        report = engine.generate(days=30)
        assert report["empty"] is False

        models = report["models"]
        assert len(models) == 1
        assert models[0]["model"] == "unknown"
        assert models[0]["has_pricing"] is False

    def test_mixed_commercial_and_custom_models(self, db):
        """Mix of commercial and custom models: only commercial ones get costs."""
        seed_session(db, session_id="s1", source="cli", model="anthropic/claude-sonnet-4-20250514")
        seed_usage(db,
            "s1",
            input_tokens=10000,
            output_tokens=5000,
            billing_provider="anthropic",
        )
        seed_session(db, session_id="s2", source="cli", model="my-local-llama")
        seed_usage(db, "s2", input_tokens=10000, output_tokens=5000)

        engine = InsightsEngine(db, persistence=db)
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
        seed_session(db, session_id="s1", source="cli", model="test")

        engine = InsightsEngine(db, persistence=db)
        report = engine.generate(days=30)
        assert len(report["platforms"]) == 1
        assert report["platforms"][0]["platform"] == "cli"

        # Terminal format should NOT show platform section for single platform
        engine.format_terminal(report)
        # (it still shows platforms section if there's only cli and nothing else)
        # Actually the condition is > 1 platforms OR non-cli, so single cli won't show

    def test_cost_buckets_displayed_in_terminal_format(self, db):
        """#77223: included/estimated/unknown cost buckets surface in terminal."""
        # Estimated cost session
        seed_session(db, session_id="est", source="cli", model="model-a")
        seed_usage(db,
            "est", input_tokens=100, model="model-a",
            billing_provider="custom",
            estimated_cost_usd=1.50, actual_cost_usd=1.0,
            cost_status="estimated", cost_source="provider", api_call_count=1,
        )
        # Included cost session (subscription)
        seed_session(db, session_id="inc", source="cli", model="gpt-5.4-mini")
        seed_usage(db,
            "inc", input_tokens=200, model="gpt-5.4-mini",
            billing_provider="openai-codex",
            estimated_cost_usd=0.0, actual_cost_usd=0.0,
            cost_status="included", cost_source="none", api_call_count=1,
        )

        engine = InsightsEngine(db, persistence=db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        # The cost section should appear with the buckets this DB has
        # (estimated + included; no unknown-cost session is created here)
        assert "~$1.50" in text  # estimated
        assert "included" in text.lower()
        assert "subscription" in text.lower()

    def test_sub_cent_aggregate_estimated_cost_not_zero(self, db):
        """A sub-cent aggregate must not render 'Estimated: ~$0.00' (#79220).

        The insights formatters share format_cost_label with per-response
        labels; a cheap-model period totaling $0.0046 shows 4dp, not $0.00.
        """
        seed_session(db, session_id="est", source="cli", model="model-a")
        seed_usage(db,
            "est", input_tokens=100, model="model-a",
            billing_provider="custom",
            estimated_cost_usd=0.0046, actual_cost_usd=0.0,
            cost_status="estimated", cost_source="provider", api_call_count=1,
        )

        engine = InsightsEngine(db, persistence=db)
        report = engine.generate(days=30)
        terminal_text = engine.format_terminal(report)
        gateway_text = engine.format_gateway(report)

        assert "~$0.00\n" not in terminal_text
        assert "~$0.0046" in terminal_text
        assert "~$0.00 estimated" not in gateway_text
        assert "~$0.0046 estimated" in gateway_text


def _label(db, session_id, source, created_source):
    seed_session(db, session_id=session_id, source=source, model="model-a")
    seed_messages(db, session_id, dict(role="user", content="hi"))
    _sql(db, "UPDATE sessions SET created_source = ? WHERE id = ?", (created_source, session_id))


def test_source_filter_includes_gateway_local_rows_that_surface_created(db):
    """The gateway stores interactive CLI/TUI rows as ``local`` (``created_source`` keeps the
    surface), so ``/insights --source cli`` reads both labels (tracker #134889 CLI-23)."""
    _label(db, "legacy-cli", "cli", "cli")
    _label(db, "local-cli", "local", "cli")
    _label(db, "local-tui", "local", "tui")
    _label(db, "tg", "telegram", "telegram")
    engine = InsightsEngine(db, persistence=db)
    assert engine.generate(days=30, source="cli")["overview"]["total_sessions"] == 2
    assert engine.generate(days=30, source="tui")["overview"]["total_sessions"] == 1
    assert engine.generate(days=30, source="telegram")["overview"]["total_sessions"] == 1


def test_source_filter_on_a_db_without_created_source(db):
    """A DB an older writer left behind (read-only opens skip migrations) still filters by label."""
    _label(db, "legacy-cli", "cli", "cli")
    # SQLite refuses DROP COLUMN while an index references the column.
    _sql(db, "DROP INDEX IF EXISTS idx_sessions_local_created_started")
    _sql(db, "ALTER TABLE sessions DROP COLUMN created_source")
    with open_persistence_reader(store_location(db.scope)) as reader:
        assert InsightsEngine(None, persistence=reader).generate(days=30, source="cli")["overview"]["total_sessions"] == 1


@pytest.mark.parametrize("read_only", [False, True], ids=["writer_handle", "reader_handle"])
def test_generate_drains_the_writers_queue_when_it_holds_one(db, monkeypatch, read_only):
    """A writer's handle beside the store drains its accounting queue; beside a reader the store drains its own."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    store_flush = MagicMock()
    store = SimpleNamespace(flush_token_counts=store_flush)

    db.ready()
    with open_persistence_reader(store_location(db.scope)) as reader:
        persistence = SimpleNamespace(read_only=read_only, usage=MagicMock(), usage_analytics=reader.usage_analytics)
        InsightsEngine(store, persistence=persistence).generate(days=30)

    assert persistence.usage.flush_usage.call_count == (0 if read_only else 1)
    assert store_flush.call_count == (1 if read_only else 0)


def test_reads_go_through_the_handle_usage_analytics(populated_db):
    """With a handle the engine reads its usage analytics, the same report a reader beside the store gives."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    with open_persistence_reader(store_location(populated_db.scope)) as reader:
        expected = InsightsEngine(None, persistence=reader).generate(days=30)
        analytics = MagicMock(wraps=reader.usage_analytics)
        handle = SimpleNamespace(read_only=True, usage_analytics=analytics)
        report = InsightsEngine(populated_db, persistence=handle).generate(days=30)
    assert analytics.insight_sessions.call_count == 1 and analytics.insight_model_usage.call_count == 1
    expected.pop("generated_at")
    report.pop("generated_at")
    assert report == expected
