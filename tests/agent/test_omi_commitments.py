"""Unit tests for Omi commitment extraction → kanban cards.

The autouse hermetic fixture points HERMES_HOME (and thus the kanban + state
DBs) at a per-test tempdir. We patch the module's MCP fetch and LLM extraction
seams so no network/model is needed, then assert on the real kanban rows and
the real dedup/consent/notification orchestration.
"""

from unittest.mock import patch

import pytest

from agent import omi_commitments as oc


def _cfg(**overrides):
    base = {
        "enabled": True,
        "scan_interval_hours": 6,
        "lookback_hours": 24,
        "min_confidence": 0.6,
        "board": "",
        "assignee": "",
        "create_notification": False,  # keep tests offline unless asserted
        "max_conversations_per_scan": 25,
    }
    base.update(overrides)
    return base


@pytest.fixture
def cfg_patch():
    def _apply(**overrides):
        return patch.object(oc, "_cfg", return_value=_cfg(**overrides))

    return _apply


def _list_cards():
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        return kb.list_tasks(conn, include_archived=True)
    finally:
        conn.close()


def test_disabled_skips_without_mcp(cfg_patch):
    with cfg_patch(enabled=False), patch.object(oc, "_call_mcp") as mcp:
        result = oc.run_omi_commitment_scan()
    assert result == {"skipped": "disabled"}
    mcp.assert_not_called()


def test_creates_card_for_owner_commitment(cfg_patch):
    convs = [{"id": "conv1", "transcript": "I'll send the report by 3pm"}]
    commitments = [
        {
            "text": "Send the report",
            "due_iso": "2026-07-16T15:00:00",
            "confidence": 0.9,
            "made_by_user": True,
        }
    ]
    with (
        cfg_patch(),
        patch.object(oc, "_call_mcp", return_value=convs),
        patch.object(oc, "_extract_commitments", return_value=commitments),
    ):
        result = oc.run_omi_commitment_scan()

    assert result["created"] == 1
    cards = _list_cards()
    assert len(cards) == 1
    assert cards[0].status == "triage"
    assert "Due: 2026-07-16T15:00:00" in cards[0].body


def test_bystander_commitment_excluded(cfg_patch):
    convs = [{"id": "conv1", "transcript": "TV: buy now!"}]
    commitments = [
        {
            "text": "Buy the product",
            "due_iso": None,
            "confidence": 0.95,
            "made_by_user": False,  # not the device owner
        }
    ]
    with (
        cfg_patch(),
        patch.object(oc, "_call_mcp", return_value=convs),
        patch.object(oc, "_extract_commitments", return_value=commitments),
    ):
        result = oc.run_omi_commitment_scan()

    assert result["created"] == 0
    assert _list_cards() == []


def test_low_confidence_dropped(cfg_patch):
    convs = [{"id": "conv1", "transcript": "maybe I'll look into it"}]
    commitments = [
        {
            "text": "Look into it",
            "due_iso": None,
            "confidence": 0.3,  # below min_confidence 0.6
            "made_by_user": True,
        }
    ]
    with (
        cfg_patch(),
        patch.object(oc, "_call_mcp", return_value=convs),
        patch.object(oc, "_extract_commitments", return_value=commitments),
    ):
        result = oc.run_omi_commitment_scan()

    assert result["created"] == 0
    assert _list_cards() == []


def test_dedup_on_rescan(cfg_patch):
    convs = [{"id": "conv1", "transcript": "I'll email Sarah"}]
    commitments = [
        {
            "text": "Email Sarah",
            "due_iso": None,
            "confidence": 0.8,
            "made_by_user": True,
        }
    ]
    # First scan creates the card and marks the conversation processed.
    with (
        cfg_patch(),
        patch.object(oc, "_call_mcp", return_value=convs),
        patch.object(oc, "_extract_commitments", return_value=commitments),
    ):
        first = oc.run_omi_commitment_scan()
    assert first["created"] == 1

    # Second scan of the same conversation must not create a duplicate.
    # (Even if the seen-guard were bypassed, the idempotency_key would dedup.)
    with (
        cfg_patch(),
        patch.object(oc, "_call_mcp", return_value=convs),
        patch.object(oc, "_extract_commitments", return_value=commitments),
    ):
        second = oc.run_omi_commitment_scan()
    assert second["scanned"] == 0  # already-seen conversation skipped
    assert len(_list_cards()) == 1


def test_mcp_error_handled_gracefully(cfg_patch):
    # _call_mcp returns None on an MCP error dict — scan yields zero, no crash.
    with cfg_patch(), patch.object(oc, "_call_mcp", return_value=None):
        result = oc.run_omi_commitment_scan()
    assert result == {"scanned": 0, "extracted": 0, "created": 0, "notified": 0}
    assert _list_cards() == []


def test_notification_routed_through_governor(cfg_patch):
    convs = [{"id": "conv1", "transcript": "I'll ship it tonight"}]
    commitments = [
        {
            "text": "Ship it",
            "due_iso": None,
            "confidence": 0.9,
            "made_by_user": True,
        }
    ]
    with (
        cfg_patch(create_notification=True),
        patch.object(oc, "_call_mcp", return_value=convs),
        patch.object(oc, "_extract_commitments", return_value=commitments),
        patch("agent.notification_budget.should_deliver") as should,
        patch("tools.send_message_tool.send_message_tool") as send,
    ):
        from agent.notification_budget import BudgetDecision

        should.return_value = BudgetDecision(
            allow=True,
            reason="under-budget",
            score=0.9,
            threshold=0.5,
            category="omi_commitment",
            ledger_id="x",
        )
        result = oc.run_omi_commitment_scan()

    assert result["notified"] == 1
    should.assert_called_once()
    send.assert_called_once()


def test_notification_suppressed_when_budget_denies(cfg_patch):
    convs = [{"id": "conv1", "transcript": "I'll ship it tonight"}]
    commitments = [
        {"text": "Ship it", "due_iso": None, "confidence": 0.9, "made_by_user": True}
    ]
    with (
        cfg_patch(create_notification=True),
        patch.object(oc, "_call_mcp", return_value=convs),
        patch.object(oc, "_extract_commitments", return_value=commitments),
        patch("agent.notification_budget.should_deliver") as should,
        patch("tools.send_message_tool.send_message_tool") as send,
    ):
        from agent.notification_budget import BudgetDecision

        should.return_value = BudgetDecision(
            allow=False,
            reason="below-threshold",
            score=0.1,
            threshold=0.5,
            category="omi_commitment",
            ledger_id="x",
        )
        result = oc.run_omi_commitment_scan()

    assert result["created"] == 1  # card still filed
    assert result["notified"] == 0  # but no ping
    send.assert_not_called()


class TestParseCommitments:
    """The real JSON parser must survive fences and malformed output."""

    def test_plain_json(self):
        out = oc._parse_commitments('{"commitments": [{"text": "x"}]}')
        assert out == [{"text": "x"}]

    def test_code_fenced_json(self):
        raw = '```json\n{"commitments": [{"text": "y"}]}\n```'
        assert oc._parse_commitments(raw) == [{"text": "y"}]

    def test_malformed_returns_empty(self):
        assert oc._parse_commitments("not json at all") == []

    def test_missing_key_returns_empty(self):
        assert oc._parse_commitments('{"other": 1}') == []
