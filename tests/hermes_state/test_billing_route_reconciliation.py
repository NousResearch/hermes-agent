"""Sessions-row billing-route reconciliation: the row follows the LATEST accounted call.

A configured-route writer (create_session's COALESCE backfill, /model switches,
/new's config-default reset) can stamp a route that never served accounted usage;
with the old ``api_call_count == 0`` gate such a stamp froze onto the row forever.
The row must follow the latest call that actually carried accounted usage, and a
usage-less write must never stomp the stored route.
"""

from __future__ import annotations

from hermes_state import SessionDB

SUB = "claude-subscription-route"  # any route other than the config default


def test_route_follows_latest_accounted_call(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.update_token_counts(
        "s1", input_tokens=100, model="z-ai/glm-5.3", billing_provider="openrouter",
        api_call_count=1, estimated_cost_usd=0.05,
    )
    # A configured-route writer stamps a route that never served accounted usage
    # (/new's config-default reset did exactly this to the outgoing session).
    db.update_session_billing_route("s1", provider="other", base_url="https://other.example/v1")
    db.update_token_counts(
        "s1", input_tokens=50, model="claude-max-route", billing_provider=SUB,
        billing_base_url="process://sub", api_call_count=1,
    )
    with db._lock:
        row = db._conn.execute(
            "SELECT model, billing_provider, billing_base_url FROM sessions WHERE id='s1'").fetchone()
        routes = db._conn.execute(
            "SELECT model, billing_provider FROM session_model_usage WHERE session_id='s1' ORDER BY rowid"
        ).fetchall()
    assert row["billing_provider"] == SUB
    assert row["model"] == "claude-max-route"
    assert row["billing_base_url"] == "process://sub"
    # the exact per-route split survives on the usage rows
    assert [(r["billing_provider"], r["model"]) for r in routes] == [
        ("openrouter", "z-ai/glm-5.3"), (SUB, "claude-max-route")]


def test_first_accounted_route_still_replaces_create_time_route(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db._insert_session_row("s2", "cli", model="z-ai/glm-5.3")
    db.update_token_counts(
        "s2", input_tokens=10, model="claude-max-route", billing_provider=SUB,
        api_call_count=1,
    )
    with db._lock:
        row = db._conn.execute(
            "SELECT model, billing_provider FROM sessions WHERE id='s2'").fetchone()
    assert row["billing_provider"] == SUB
    assert row["model"] == "claude-max-route"


def test_usage_less_write_never_stomps_the_route(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.update_token_counts(
        "s3", input_tokens=10, model="claude-max-route", billing_provider=SUB,
        api_call_count=1,
    )
    # No cost and no provider on the delta: no route rewrite even though the
    # stored names differ (has_accounted_usage gates on tokens/cost, not names).
    db.update_token_counts("s3", input_tokens=5, api_call_count=1)
    with db._lock:
        row = db._conn.execute(
            "SELECT model, billing_provider FROM sessions WHERE id='s3'").fetchone()
    assert row["billing_provider"] == SUB