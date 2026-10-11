"""Usage-tab by_model must attribute tokens per API call, not per session (#103063).

The usage analytics route built ``by_model`` with ``GROUP BY model`` over the
``sessions`` table — the session's *current* model — so a session that switched
models via ``/model`` credited every token it ever burned to the last model
selected, and the raw model string (with ``provider/`` prefixes) reached the
Desktop Command Center unnormalized. ``session_model_usage`` records each API
call with the model active at call time; ``InsightsEngine._compute_model_breakdown``
aggregates it with a legacy residual fallback and normalizes names via
``_short_model``. The route now reuses that instead of the sessions GROUP BY.

Aux folding: ``_compute_model_breakdown`` reads ALL ``session_model_usage`` rows
(main + aux), so aux tokens are included per-model — dropping the old
``_merge_aux_into_by_model`` merge must not lose aux-only models (e.g. a vision
model that never appears in ``sessions.model``) nor double-count them.
"""

import time

import pytest

from hermes_cli.web_routers import analytics as web_server
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


@pytest.fixture
def usage(monkeypatch, db):
    """Run _get_usage_analytics against the test DB."""
    monkeypatch.setattr(
        web_server,
        "_open_session_db_for_profile",
        lambda profile, read_only=True: db,
    )
    # The helper closes the DB it is handed; keep it open for assertions.
    monkeypatch.setattr(db, "close", lambda: None)
    return lambda: web_server._get_usage_analytics(days=30)


def _touch(db, session_id, model, provider, *, input_tokens, output_tokens, api_calls=1):
    """Record one main-loop API call's delta on the route active at call time."""
    db.update_token_counts(
        session_id, input_tokens=input_tokens, output_tokens=output_tokens,
        model=model, billing_provider=provider, api_call_count=api_calls,
    )


def _by_model(result):
    return {m["model"]: m for m in result["by_model"]}


def test_mid_session_model_switch_splits_tokens_across_both_models(db, usage):
    """A /model switch must reattribute tokens to the model that burned them."""
    db.create_session("s1", source="cli", model="anthropic/claude-opus-5")
    _touch(db, "s1", "claude-opus-5", "anthropic", input_tokens=30_000, output_tokens=1_000)
    _touch(db, "s1", "deepseek-v4-pro", "deepseek", input_tokens=31_000, output_tokens=2_000)
    # The sessions row keeps only the final model — the old misattribution source.
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET model = ? WHERE id = ?", ("deepseek-v4-pro", "s1"),
        )
        db._conn.commit()

    models = _by_model(usage())

    # Names come back normalized (no provider/ prefix), consistent with the
    # status bar and picker.
    assert set(models) == {"claude-opus-5", "deepseek-v4-pro"}
    assert models["claude-opus-5"]["input_tokens"] == 30_000
    assert models["deepseek-v4-pro"]["input_tokens"] == 31_000
    # A session counted once per model it actually used, never COUNT(*)-inflated.
    assert models["claude-opus-5"]["sessions"] == 1
    assert models["deepseek-v4-pro"]["sessions"] == 1


def test_aux_only_model_keeps_its_own_row_without_double_counting(db, usage):
    """Dropping _merge_aux_into_by_model must not lose aux-only models (#23270)."""
    db.create_session("s1", source="cli", model="gpt-5.6-sol")
    _touch(db, "s1", "gpt-5.6-sol", "openai-codex", input_tokens=100, output_tokens=10)
    db.record_auxiliary_usage(
        "s1", "vision", model="gemini-3-flash", billing_provider="gemini",
        input_tokens=500, output_tokens=50,
    )

    models = _by_model(usage())

    assert models["gemini-3-flash"]["input_tokens"] == 500
    # The main model's row is not inflated by the aux call's tokens.
    assert models["gpt-5.6-sol"]["input_tokens"] == 100
    # Aux usage happened inside an already-counted session: the main card's
    # session count is not inflated by the aux row.
    assert models["gpt-5.6-sol"]["sessions"] == 1


def test_legacy_session_without_per_call_rows_falls_back_to_aggregate(db, usage):
    """A session with no session_model_usage rows still appears in by_model."""
    db.create_session("s1", source="cli", model="anthropic/claude-opus-5")
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET input_tokens = 7_000, output_tokens = 900 WHERE id = ?",
            ("s1",),
        )
        db._conn.commit()

    models = _by_model(usage())

    assert models["claude-opus-5"]["input_tokens"] == 7_000
    assert models["claude-opus-5"]["output_tokens"] == 900


def test_by_task_summary_still_reports_aux_tasks(db, usage):
    """The by_task summary keeps using _aux_usage_rows after the merge removal."""
    db.create_session("s1", source="cli", model="gpt-5.6-sol")
    _touch(db, "s1", "gpt-5.6-sol", "openai-codex", input_tokens=100, output_tokens=10)
    db.record_auxiliary_usage(
        "s1", "vision", model="gemini-3-flash", billing_provider="gemini",
        input_tokens=500, output_tokens=50,
    )

    result = usage()

    tasks = {t["task"]: t for t in result["by_task"]}
    assert tasks["vision"]["input_tokens"] == 500
