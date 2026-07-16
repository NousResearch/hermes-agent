"""Unit tests for the stalled-thread follow-up detector.

The autouse hermetic fixture (tests/conftest.py) points HERMES_HOME (and thus
the kanban + state DBs) at a per-test tempdir and pins TZ=UTC. We patch the
module's classify + delivery seams so no model/network is needed, and assert on
the real detection/dedup/consent orchestration over real tmp DBs.
"""

import time
from unittest.mock import patch

import pytest

from agent import stalled_threads as st


def _cfg(**overrides):
    base = {
        "enabled": True,
        "scan_interval_hours": 12,
        "staleness_hours": 48,
        "cooldown_hours": 72,
        "lookback_hours": 336,
        "min_confidence": 0.6,
        "scan_threads": True,
        "max_items_per_digest": 5,
        "board": "",
        "exclude_sources": ["tool", "tui"],
    }
    base.update(overrides)
    return base


@pytest.fixture
def cfg_patch():
    def _apply(**overrides):
        return patch.object(st, "_cfg", return_value=_cfg(**overrides))

    return _apply


def _seed_card(*, title, body=None, status="running", age_hours=0.0):
    """Create a kanban card, optionally back-dating created_at by age_hours."""
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        # create_task forces running/blocked/triage; use triage for an open card.
        tid = kb.create_task(conn, title=title, body=body, triage=True)
        if age_hours:
            old = int(time.time() - age_hours * 3600)
            conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (old, tid))
            conn.commit()
        return tid
    finally:
        conn.close()


# ── _parse_due ───────────────────────────────────────────────────────────────


class TestParseDue:
    def test_iso_with_offset(self):
        assert st._parse_due("Due: 2026-07-25T03:59:00+00:00\n\nx") is not None

    def test_none_sentinel(self):
        assert st._parse_due("Due: none\n\nx") is None

    def test_date_only(self):
        assert st._parse_due("Due: 2026-07-25") is not None

    def test_z_suffix(self):
        assert st._parse_due("Due: 2026-07-25T03:59:00Z") is not None

    def test_missing_line(self):
        assert st._parse_due("no due here") is None

    def test_empty_body(self):
        assert st._parse_due(None) is None

    def test_prose_due_phrase(self):
        # Real dispatcher-rewritten cards embed the date in prose, no 'Due:' line.
        body = (
            "Goal: Stand up a working TE ready for a demo by the due date 2026-07-17."
        )
        assert st._parse_due(body) is not None

    def test_prose_due_word(self):
        body = "Complete the PLG prep work. Due 2026-07-25."
        assert st._parse_due(body) is not None


# ── consent / disabled ───────────────────────────────────────────────────────


def test_disabled_scan_skips(cfg_patch):
    with cfg_patch(enabled=False):
        assert st.run_stalled_thread_scan() == {"skipped": "disabled"}


def test_disabled_list_skips(cfg_patch):
    with cfg_patch(enabled=False):
        assert st.list_stalled_candidates() == {"skipped": "disabled"}


# ── Source A: commitments ────────────────────────────────────────────────────


def test_past_due_card_detected(cfg_patch):
    yesterday = "2000-01-01"  # firmly in the past
    _seed_card(title="Send the deck", body=f"Due: {yesterday}\n\n> Send the deck")
    with cfg_patch(scan_threads=False):
        result = st.list_stalled_candidates()
    ids = [c["candidate_id"] for c in result["candidates"]]
    assert any(cid.startswith("card:") for cid in ids)


def test_untouched_card_detected(cfg_patch):
    _seed_card(title="Old task", body="Due: none", age_hours=100)  # > 48h staleness
    with cfg_patch(scan_threads=False):
        result = st.list_stalled_candidates()
    assert len(result["candidates"]) == 1


def test_fresh_card_not_detected(cfg_patch):
    _seed_card(title="Fresh", body="Due: none", age_hours=1)  # < staleness
    with cfg_patch(scan_threads=False):
        result = st.list_stalled_candidates()
    assert result["candidates"] == []


def test_due_none_not_past_due(cfg_patch):
    # A fresh card with 'Due: none' is neither past-due nor untouched.
    _seed_card(title="No due", body="Due: none", age_hours=1)
    with cfg_patch(scan_threads=False):
        result = st.list_stalled_candidates()
    assert result["candidates"] == []


# ── classify + governor orchestration ────────────────────────────────────────


def _confirm_all(candidates):
    """Fake classifier: mark every candidate open with high confidence."""
    return {
        c["candidate_id"]: {
            "candidate_id": c["candidate_id"],
            "owed_summary": f"follow up: {c['text'][:30]}",
            "still_open": True,
            "confidence": 0.9,
        }
        for c in candidates
    }


def test_scan_nudges_confirmed_open(cfg_patch):
    _seed_card(title="Send the deck", body="Due: 2000-01-01\n\n> deck")
    with (
        cfg_patch(scan_threads=False),
        patch.object(st, "_classify", side_effect=_confirm_all),
        patch("agent.notification_budget.should_deliver") as should,
        patch("tools.send_message_tool.send_message_tool") as send,
    ):
        from agent.notification_budget import BudgetDecision

        should.return_value = BudgetDecision(
            allow=True,
            reason="under-budget",
            score=0.9,
            threshold=0.5,
            category="stalled_thread",
            ledger_id="x",
        )
        result = st.run_stalled_thread_scan()
    assert result["nudged"] == 1
    assert result["delivered"] == 1
    send.assert_called_once()
    # exactly one batched digest, not one send per item
    should.assert_called_once()


def test_governor_suppresses_digest(cfg_patch):
    _seed_card(title="Send the deck", body="Due: 2000-01-01\n\n> deck")
    with (
        cfg_patch(scan_threads=False),
        patch.object(st, "_classify", side_effect=_confirm_all),
        patch("agent.notification_budget.should_deliver") as should,
        patch("tools.send_message_tool.send_message_tool") as send,
    ):
        from agent.notification_budget import BudgetDecision

        should.return_value = BudgetDecision(
            allow=False,
            reason="below-threshold",
            score=0.1,
            threshold=0.5,
            category="stalled_thread",
            ledger_id="x",
        )
        result = st.run_stalled_thread_scan()
    assert result["delivered"] == 0
    send.assert_not_called()


def test_resolved_items_dropped(cfg_patch):
    _seed_card(title="Done thing", body="Due: 2000-01-01\n\n> x")

    def _resolve_all(candidates):
        return {
            c["candidate_id"]: {
                "candidate_id": c["candidate_id"],
                "owed_summary": "",
                "still_open": False,
                "confidence": 0.9,
            }
            for c in candidates
        }

    with (
        cfg_patch(scan_threads=False),
        patch.object(st, "_classify", side_effect=_resolve_all),
        patch("tools.send_message_tool.send_message_tool") as send,
    ):
        result = st.run_stalled_thread_scan()
    assert result["candidates"] == 0
    send.assert_not_called()


def test_low_confidence_dropped(cfg_patch):
    _seed_card(title="Maybe", body="Due: 2000-01-01\n\n> x")

    def _low(candidates):
        return {
            c["candidate_id"]: {
                "candidate_id": c["candidate_id"],
                "owed_summary": "x",
                "still_open": True,
                "confidence": 0.3,  # below min_confidence 0.6
            }
            for c in candidates
        }

    with (
        cfg_patch(scan_threads=False),
        patch.object(st, "_classify", side_effect=_low),
        patch("tools.send_message_tool.send_message_tool") as send,
    ):
        result = st.run_stalled_thread_scan()
    assert result["candidates"] == 0
    send.assert_not_called()


def test_dedup_within_cooldown(cfg_patch):
    _seed_card(title="Send the deck", body="Due: 2000-01-01\n\n> deck")
    with (
        cfg_patch(scan_threads=False),
        patch.object(st, "_classify", side_effect=_confirm_all),
        patch("agent.notification_budget.should_deliver") as should,
        patch("tools.send_message_tool.send_message_tool"),
    ):
        from agent.notification_budget import BudgetDecision

        should.return_value = BudgetDecision(
            allow=True,
            reason="under-budget",
            score=0.9,
            threshold=0.5,
            category="stalled_thread",
            ledger_id="x",
        )
        first = st.run_stalled_thread_scan()
        second = st.run_stalled_thread_scan()
    assert first["nudged"] == 1
    # second run: same card is within cooldown -> filtered out before classify
    assert second["candidates"] == 0
    assert second["nudged"] == 0


def test_classify_failure_is_fail_soft(cfg_patch):
    _seed_card(title="Send the deck", body="Due: 2000-01-01\n\n> deck")
    with (
        cfg_patch(scan_threads=False),
        patch.object(st, "_classify", return_value={}),
        patch("tools.send_message_tool.send_message_tool") as send,
    ):
        result = st.run_stalled_thread_scan()
    # No verdicts -> nothing confirmed open -> no nudge, no crash
    assert result["candidates"] == 0
    send.assert_not_called()


def test_max_items_caps_digest(cfg_patch):
    for i in range(4):
        _seed_card(title=f"Task {i}", body="Due: 2000-01-01\n\n> x")
    with (
        cfg_patch(scan_threads=False, max_items_per_digest=2),
        patch.object(st, "_classify", side_effect=_confirm_all),
        patch("agent.notification_budget.should_deliver") as should,
        patch("tools.send_message_tool.send_message_tool"),
    ):
        from agent.notification_budget import BudgetDecision

        should.return_value = BudgetDecision(
            allow=True,
            reason="under-budget",
            score=0.9,
            threshold=0.5,
            category="stalled_thread",
            ledger_id="x",
        )
        result = st.run_stalled_thread_scan()
    assert result["nudged"] == 2  # capped


def test_empty_board_no_candidates(cfg_patch):
    with cfg_patch(scan_threads=False):
        result = st.run_stalled_thread_scan()
    assert result == {"scanned": 0, "candidates": 0, "nudged": 0, "delivered": 0}


# ── Source B: threads (via stubbed live-thread rows) ─────────────────────────


def test_thread_awaiting_reply_detected(cfg_patch):
    old = time.time() - 60 * 3600  # 60h ago > 48h staleness
    thread = {
        "session_key": "telegram:12345",
        "id": "sess-1",
        "source": "telegram",
        "chat_id": "12345",
        "display_name": "Sarah",
        "last_active": old,
        "last_role": "user",
        "last_content": "did you get my note?",
        "last_observed": 0,
    }
    with (
        cfg_patch(),
        patch.object(
            st.SessionDB, "list_live_threads_for_stall", return_value=[thread]
        ),
    ):
        # scan_threads True by default; commitments source is empty (no cards)
        result = st.list_stalled_candidates()
    ids = [c["candidate_id"] for c in result["candidates"]]
    assert any(cid.startswith("thread:") for cid in ids)


def test_thread_bot_spoke_last_not_detected(cfg_patch):
    old = time.time() - 60 * 3600
    thread = {
        "session_key": "telegram:12345",
        "id": "sess-1",
        "source": "telegram",
        "last_active": old,
        "last_role": "assistant",  # bot spoke last -> not awaiting the user
        "last_content": "here you go",
        "last_observed": 0,
    }
    with (
        cfg_patch(),
        patch.object(
            st.SessionDB, "list_live_threads_for_stall", return_value=[thread]
        ),
    ):
        result = st.list_stalled_candidates()
    assert result["candidates"] == []


# ── Fail-soft: uncaught errors become an error summary, never propagate ──────


def test_parse_due_malformed_date_returns_none():
    # _to_epoch swallows bad dates; _parse_due must not raise.
    assert st._parse_due("Due: 2026-99-99") is None
    assert st._parse_due("Complete it. Due 2026-13-45.") is None


def test_record_nudge_failure_is_fail_soft(cfg_patch):
    _seed_card(title="Send the deck", body="Due: 2000-01-01\n\n> deck")
    with (
        cfg_patch(scan_threads=False),
        patch.object(st, "_classify", side_effect=_confirm_all),
        patch("agent.notification_budget.should_deliver") as should,
        patch("tools.send_message_tool.send_message_tool"),
        patch.object(
            st.SessionDB,
            "record_stall_nudge",
            side_effect=RuntimeError("db write failed"),
        ),
    ):
        from agent.notification_budget import BudgetDecision

        should.return_value = BudgetDecision(
            allow=True,
            reason="under-budget",
            score=0.9,
            threshold=0.5,
            category="stalled_thread",
            ledger_id="x",
        )
        result = st.run_stalled_thread_scan()
    # The digest was sent, but the record write blew up — must not propagate.
    assert "error" in result


def test_scan_db_open_failure_is_fail_soft(cfg_patch):
    with (
        cfg_patch(),
        patch.object(st, "SessionDB", side_effect=RuntimeError("cannot open state.db")),
    ):
        result = st.run_stalled_thread_scan()
    assert "error" in result
    assert result["nudged"] == 0
