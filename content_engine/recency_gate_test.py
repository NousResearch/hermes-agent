"""Test suite for temporal relevance gates, variable injection, and topic recency.

Run: python3 recency_gate_test.py
"""
import sys
import os
import uuid
from datetime import UTC, datetime, timedelta, date

# Ensure repo on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_drafts import (
    _build_topic_variables,
    _is_future_fixture,
    _reject_stale_tech,
    _fill_variables,
    _select_and_fill_template,
    generate_drafts,
)
from topics import (
    has_stale_tech_reference,
    STALE_TECH_REFS,
    TOPIC_BANKS,
    _signal_to_topic,
    get_topics,
)
from database import init_db, is_topic_recently_used, log_topic_usage, get_recently_used_topics
from config import DB_PATH

PASS = 0
FAIL = 0

def assert_true(expr, name):
    global PASS, FAIL
    if expr:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}")

def test_variable_injection():
    print("\n--- Variable Injection ---")
    # Fixture topic
    topic = {
        "pillar": "live_predictions",
        "topic": "Arsenal vs City",
        "fixture": {"home": "Arsenal", "away": "City", "date": "2026-06-01", "time": "15:00"},
    }
    vars_out = _build_topic_variables("matchdaymaestro", topic)
    assert_true(vars_out["home"] == "Arsenal", "fixture home injected")
    assert_true(vars_out["away"] == "City", "fixture away injected")
    assert_true(vars_out["date"] == "2026-06-01", "fixture date injected")

    # Template fill
    template = "{home} vs {away} on {date}. Kick off {time}."
    filled = _fill_variables(template, vars_out)
    assert_true(filled is not None and "Arsenal vs City on 2026-06-01" in filled, "template fills correctly")

    # Activity topic
    topic2 = {
        "pillar": "build_in_public",
        "topic": "Pushed repo",
        "activity_data": {
            "signal_type": "github_push",
            "variables": {"repo_name": "KenseiAgent", "description": "AI agent infra"},
            "signal_id": "sig_123",
        },
    }
    vars2 = _build_topic_variables("sahil_twitter", topic2)
    assert_true(vars2["repo_name"] == "KenseiAgent", "activity repo_name injected")
    assert_true(vars2["signal_type"] == "github_push", "activity signal_type injected")


def test_temporal_gates():
    print("\n--- Temporal Relevance Gates ---")

    # Future fixture passes
    future_topic = {"fixture": {"date": (datetime.now(UTC) + timedelta(days=3)).strftime("%Y-%m-%d")}}
    assert_true(_is_future_fixture(future_topic), "future fixture passes gate")

    # Today fixture passes
    today_topic = {"fixture": {"date": datetime.now(UTC).strftime("%Y-%m-%d")}}
    assert_true(_is_future_fixture(today_topic), "today fixture passes gate")

    # Past fixture fails
    past_topic = {"fixture": {"date": "2025-01-01"}}
    assert_true(not _is_future_fixture(past_topic), "past fixture rejected")

    # Non-fixture topic always passes
    no_fixture = {"pillar": "game_modes", "topic": "Daily quiz"}
    assert_true(_is_future_fixture(no_fixture), "non-fixture topic passes")


def test_stale_tech_gate():
    print("\n--- Stale Tech Gate ---")

    for ref in ["GPT-4o", "gpt-4o", "ChatGPT 4"]:
        has_stale, match = has_stale_tech_reference(f"Using {ref} for my app.")
        assert_true(has_stale, f"detects stale ref: {ref}")
        assert_true(match == ref, f"returns correct match: {ref}")

    # Clean text passes
    ok, _ = _reject_stale_tech("Claude Code workflow with Convex schema.")
    assert_true(ok, "clean text passes stale-tech gate")

    # Stale text rejected
    ok2, _ = _reject_stale_tech("Built with GPT-4o and loving it.")
    assert_true(not ok2, "stale text rejected")


def test_topic_recency_db():
    print("\n--- Topic Recency DB ---")

    # Use a temp DB for isolation
    original_db = str(DB_PATH)
    test_db = original_db.replace(".db", "_test.db")
    import database
    database.DB_PATH = __import__("pathlib").Path(test_db)

    init_db()
    tid = "test_topic_001"
    assert_true(not is_topic_recently_used(tid), "unused topic not recently used")

    log_topic_usage(tid, "matchdaymaestro", "Derby day", "twitter")
    assert_true(is_topic_recently_used(tid), "logged topic is recently used")

    recent = get_recently_used_topics("matchdaymaestro", days=30)
    assert_true(tid in recent, "recent topics list includes logged topic")

    # Cleanup
    os.remove(test_db) if os.path.exists(test_db) else None
    database.DB_PATH = __import__("pathlib").Path(original_db)


def test_generate_drafts_integration():
    print("\n--- generate_drafts Integration ---")

    topics = [
        {
            "id": "fx_001",
            "pillar": "live_predictions",
            "topic": "Arsenal vs City",
            "fixture": {"home": "Arsenal", "away": "City", "date": (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d"), "time": "15:00"},
        },
        {
            "id": "gm_001",
            "pillar": "game_modes",
            "topic": "Strike501 daily",
        },
    ]
    drafts = generate_drafts("matchdaymaestro", topics, platform="twitter")
    assert_true(len(drafts) == 2, f"generates 2 drafts (got {len(drafts)})")

    # Verify variable injection in output
    d0 = drafts[0]
    body = d0["body_text"]
    assert_true("Arsenal" in body or "City" in body, "fixture vars injected into body")
    assert_true(d0.get("slop_audit") is not None, "slop audit present")

    # Past fixture should be skipped
    past_topics = [
        {
            "id": "fx_past",
            "pillar": "live_predictions",
            "topic": "Old match",
            "fixture": {"home": "Liverpool", "away": "Spurs", "date": "2025-01-01", "time": "15:00"},
        },
    ]
    past_drafts = generate_drafts("matchdaymaestro", past_topics, platform="twitter")
    assert_true(len(past_drafts) == 0, "past fixture produces zero drafts")


def test_topic_bank_counts():
    print("\n--- Topic Bank Counts ---")
    for brand, bank in TOPIC_BANKS.items():
        count = len(bank)
        assert_true(count >= 30, f"{brand}: {count} topics (min 30)")
        print(f"  {brand}: {count} topics")


def main():
    print("=" * 60)
    print("Recency Gate Test Suite")
    print("=" * 60)

    test_variable_injection()
    test_temporal_gates()
    test_stale_tech_gate()
    test_topic_recency_db()
    test_generate_drafts_integration()
    test_topic_bank_counts()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
