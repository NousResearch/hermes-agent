"""Tests for the shadow ACK delivery ledger (M1 root ACK)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_ack_ledger as ack


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Schema / round-trip
# ---------------------------------------------------------------------------


def test_init_creates_ack_ledger_tables(kanban_home):
    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    names = {r["name"] for r in rows}
    assert {
        "ack_task_verdict",
        "ack_subscription",
        "ack_passive_delivery",
        "ack_active_wake",
        "ack_operator_receipt",
    } <= names


def test_task_verdict_round_trip(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="verdict task", assignee="worker")
        rid = ack.record_ack_task_verdict(
            conn,
            task_id=tid,
            run_id=7,
            event_id=42,
            verdict="GO",
            status="done",
            summary_ref="run_7_summary",
            summary_safe="All good.\nSecond line.",
        )
        rows = ack.list_ack_task_verdicts(conn, tid)

    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == rid
    assert row["task_id"] == tid
    assert row["run_id"] == 7
    assert row["event_id"] == 42
    assert row["verdict"] == "GO"
    assert row["status"] == "done"
    assert row["summary_ref"] == "run_7_summary"
    assert row["summary_safe"] == "All good. Second line."


def test_subscription_round_trip(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="sub task", assignee="worker")
        rid = ack.record_ack_subscription(
            conn,
            task_id=tid,
            subscription_id=3,
            platform="discord",
            chat_id="1499390151393284106",
            thread_id="123",
            notifier_profile="gateway-a",
            desired_delivery_mode="passive",
            active_wake_required=True,
            operator_receipt_required=True,
        )
        rows = ack.list_ack_subscriptions(conn, tid)

    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == rid
    assert row["task_id"] == tid
    assert row["subscription_id"] == 3
    assert row["platform"] == "discord"
    assert row["chat_id"] == "1499390151393284106"
    assert row["thread_id"] == "123"
    assert row["notifier_profile"] == "gateway-a"
    assert row["desired_delivery_mode"] == "passive"
    assert int(row["active_wake_required"]) == 1
    assert int(row["operator_receipt_required"]) == 1


def test_passive_delivery_sanitizes_error(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="passive task", assignee="worker")
        rid = ack.record_ack_passive_delivery(
            conn,
            task_id=tid,
            subscription_id=5,
            message_id="msg-1",
            status="failed",
            error="send failed: access_token=secret123 chat_id=1499390151393284106",
        )
        rows = ack.list_ack_passive_deliveries(conn, tid)

    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == rid
    assert row["status"] == "failed"
    assert "secret123" not in row["error_safe"]
    assert "access_token=***" in row["error_safe"]
    assert row["correlation_id"].startswith("ack_")


def test_active_wake_sanitizes_trigger_error(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="wake task", assignee="worker")
        rid = ack.record_ack_active_wake(
            conn,
            task_id=tid,
            subscription_id=6,
            triggered_agent=False,
            trigger_error="wake refused: api_key=super_secret",
            correlation_id="corr-1",
            status="started",
            accepted_by_session=True,
            started_by_session=True,
            target_session_key="agent:main:discord:group:1497895797579190357",
        )
        rows = ack.list_ack_active_wakes(conn, tid)

    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == rid
    assert int(row["triggered_agent"]) == 0
    assert "super_secret" not in row["trigger_error"]
    assert "api_key=***" in row["trigger_error"]
    assert row["correlation_id"] == "corr-1"
    assert row["status"] == "started"
    assert int(row["accepted_by_session"]) == 1
    assert int(row["started_by_session"]) == 1
    assert row["target_session_key"] == "agent:main:discord:group:1497895797579190357"


def test_operator_receipt_status_validation(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="receipt task", assignee="worker")
        with pytest.raises(ValueError):
            ack.record_ack_operator_receipt(conn, task_id=tid, status="bad_status")
        rid = ack.record_ack_operator_receipt(
            conn,
            task_id=tid,
            status="pending",
            actor="operator-x",
            actor_ref="discord:u:123",
        )
        rows = ack.list_ack_operator_receipts(conn, tid)

    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == rid
    assert row["status"] == "pending"
    assert row["actor"] == "operator-x"
    assert row["actor_ref"] == "discord:u:123"


# ---------------------------------------------------------------------------
# Completion shadow-write behavior
# ---------------------------------------------------------------------------


def test_complete_task_with_explicit_subscription_shadow_writes_ledger(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="wired task", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="discord",
            chat_id="1499390151393284106",
            thread_id="123",
            notifier_profile="gateway-a",
            trigger_agent=True,
        )
        kb.complete_task(
            conn,
            tid,
            summary="Verdict: GO\nShipped.",
            result="done",
        )
        verdicts = ack.list_ack_task_verdicts(conn, tid)
        subs = ack.list_ack_subscriptions(conn, tid)

    assert len(verdicts) == 1
    assert verdicts[0]["verdict"] == "GO"
    assert verdicts[0]["status"] == "done"
    assert verdicts[0]["summary_safe"].startswith("Verdict: GO")
    assert "Shipped." in verdicts[0]["summary_safe"]

    assert len(subs) == 1
    assert subs[0]["platform"] == "discord"
    assert subs[0]["chat_id"] == "1499390151393284106"
    assert subs[0]["thread_id"] == "123"
    assert subs[0]["notifier_profile"] == "gateway-a"
    assert int(subs[0]["active_wake_required"]) == 1


def test_complete_task_snapshots_all_explicit_subscriptions(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="multi sub task", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="discord",
            chat_id="1499390151393284106",
            thread_id="123",
            notifier_profile="gateway-a",
            trigger_agent=True,
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="987654321",
            notifier_profile="gateway-b",
            trigger_agent=False,
        )
        kb.complete_task(conn, tid, summary="Verdict: GO\nDone.")
        subs = ack.list_ack_subscriptions(conn, tid)

    assert len(subs) == 2
    by_target = {(row["platform"], row["chat_id"]): row for row in subs}
    assert set(by_target) == {
        ("discord", "1499390151393284106"),
        ("telegram", "987654321"),
    }
    assert int(by_target[("discord", "1499390151393284106")]["active_wake_required"]) == 1
    assert int(by_target[("telegram", "987654321")]["active_wake_required"]) == 0


def test_complete_task_without_subscription_does_not_invent_one(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="unwired task", assignee="worker")
        kb.complete_task(conn, tid, summary="Verdict: BLOCK\nNo origin.")
        verdicts = ack.list_ack_task_verdicts(conn, tid)
        subs = ack.list_ack_subscriptions(conn, tid)

    assert len(verdicts) == 1
    assert verdicts[0]["verdict"] == "BLOCK"
    assert subs == []


def test_complete_task_with_origin_body_only_classified_missing_subscription(kanban_home):
    """Fixture reproducing t_a11319e9 shape.

    The task is created directly via ``create_task`` with a body that contains
    ``Origin/return_to:`` prose but no explicit subscription. Completion must
    record a verdict but must NOT infer a subscription from the body text.
    """
    body = "Origin/return_to: Discord Devhub #research (<#1499390151393284106>)\nDo the work."
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="origin prose task",
            body=body,
            assignee="ccreviewer",
        )
        kb.complete_task(conn, tid, summary="Verdict: GO\nDone.")
        verdicts = ack.list_ack_task_verdicts(conn, tid)
        subs = ack.list_ack_subscriptions(conn, tid)
        notify_subs = kb.list_notify_subs(conn, tid)

    assert notify_subs == [], "prose body must not create a real subscription"
    assert len(verdicts) == 1
    assert verdicts[0]["verdict"] == "GO"
    assert subs == [], "prose body must not be shadow-copied as subscription"


# ---------------------------------------------------------------------------
# Helper sanity checks
# ---------------------------------------------------------------------------


def test_safe_summary_collapses_and_caps_text():
    assert ack._safe_summary(None) is None
    assert ack._safe_summary("   \n  ") is None
    assert ack._safe_summary("line1\nline2") == "line1 line2"
    long_text = "x" * 500
    assert ack._safe_summary(long_text) == "x" * 400


def test_correlation_id_is_stable_and_non_secret():
    c1 = ack._ack_correlation_id("passive", "t_a", "discord", "c1", "th1", 12345)
    c2 = ack._ack_correlation_id("passive", "t_a", "discord", "c1", "th1", 12345)
    c3 = ack._ack_correlation_id("passive", "t_b", "discord", "c1", "th1", 12345)
    assert c1 == c2
    assert c1 != c3
    assert c1.startswith("ack_")
    assert "t_a" not in c1
