"""Tests for gateway/handoffs.py — handoff record persistence and validation."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Patch HERMES_HOME to use a temp directory for all tests
_TEMP_DIR = None


@pytest.fixture(autouse=True)
def temp_hermes_home(tmp_path):
    """Redirect handoffs storage to a temp directory for each test."""
    global _TEMP_DIR
    _TEMP_DIR = tmp_path
    with mock.patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


from gateway.handoffs import (
    VALID_TRANSITIONS,
    create_handoff,
    get_handoff,
    list_handoffs,
    render_origin_message,
    should_auto_resume,
    update_handoff_status,
    validate_status_transition,
)


# ── create / get ─────────────────────────────────────────────


class TestCreateAndGet:
    def test_create_persists_record(self, tmp_path):
        record = create_handoff(
            handoff_id="hc_test_001",
            origin_platform="discord",
            origin_channel_id="123",
            origin_thread_id="456",
            origin_message_id="789",
            requestor="Kevin",
            request_summary="Fix the auth bug",
            done_when="Tests pass",
        )
        assert record["handoff_id"] == "hc_test_001"
        assert record["status"] == "requested"
        assert record["requestor"] == "Kevin"
        assert record["origin"]["channel_id"] == "123"
        assert record["origin"]["thread_id"] == "456"

        # Verify it's actually on disk
        loaded = get_handoff("hc_test_001")
        assert loaded is not None
        assert loaded["handoff_id"] == "hc_test_001"

    def test_create_rejects_duplicate(self, tmp_path):
        create_handoff(handoff_id="hc_dup_001")
        with pytest.raises(ValueError, match="already exists"):
            create_handoff(handoff_id="hc_dup_001")

    def test_get_returns_none_for_unknown(self, tmp_path):
        assert get_handoff("hc_nonexistent") is None


# ── status transitions ───────────────────────────────────────


class TestStatusTransitions:
    def test_valid_transitions(self):
        assert validate_status_transition("requested", "in_progress")
        assert validate_status_transition("requested", "blocked")
        assert validate_status_transition("requested", "failed")
        assert validate_status_transition("in_progress", "done")
        assert validate_status_transition("in_progress", "blocked")
        assert validate_status_transition("in_progress", "failed")
        # Idempotent terminal states
        assert validate_status_transition("done", "done")
        assert validate_status_transition("blocked", "blocked")
        assert validate_status_transition("failed", "failed")
        # Recovery from blocked/failed
        assert validate_status_transition("blocked", "in_progress")
        assert validate_status_transition("failed", "in_progress")

    def test_invalid_transitions(self):
        assert not validate_status_transition("requested", "done")
        assert not validate_status_transition("done", "in_progress")
        assert not validate_status_transition("done", "failed")
        assert not validate_status_transition("done", "blocked")

    def test_update_status(self, tmp_path):
        create_handoff(handoff_id="hc_trans_001")
        record = update_handoff_status("hc_trans_001", "in_progress")
        assert record["status"] == "in_progress"

        record = update_handoff_status(
            "hc_trans_001",
            "done",
            callback_payload={"summary": "All fixed"},
        )
        assert record["status"] == "done"
        assert record["callback_received"]["summary"] == "All fixed"

    def test_update_rejects_invalid_transition(self, tmp_path):
        create_handoff(handoff_id="hc_invalid_001")
        with pytest.raises(ValueError, match="Invalid status transition"):
            update_handoff_status("hc_invalid_001", "done")

    def test_update_rejects_unknown_id(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown handoff_id"):
            update_handoff_status("hc_ghost", "done")

    def test_idempotent_callback(self, tmp_path):
        create_handoff(handoff_id="hc_idem_001")
        update_handoff_status("hc_idem_001", "in_progress")
        update_handoff_status(
            "hc_idem_001",
            "done",
            callback_payload={"summary": "First"},
        )
        # Second callback with same status should be idempotent
        record = update_handoff_status(
            "hc_idem_001",
            "done",
            callback_payload={"summary": "Second"},
        )
        # Should still have the first callback (idempotent = no-op)
        assert record["callback_received"]["summary"] == "First"


# ── list ─────────────────────────────────────────────────────


class TestListHandoffs:
    def test_list_all(self, tmp_path):
        create_handoff(handoff_id="hc_list_001")
        create_handoff(handoff_id="hc_list_002")
        records = list_handoffs()
        assert len(records) == 2

    def test_list_with_filter(self, tmp_path):
        create_handoff(handoff_id="hc_filter_001")
        create_handoff(handoff_id="hc_filter_002")
        update_handoff_status("hc_filter_002", "in_progress")
        assert len(list_handoffs("requested")) == 1
        assert len(list_handoffs("in_progress")) == 1
        assert len(list_handoffs("done")) == 0


# ── render_origin_message ────────────────────────────────────


class TestRenderOriginMessage:
    def test_basic_done_message(self, tmp_path):
        create_handoff(handoff_id="hc_render_001", request_summary="Fix auth")
        update_handoff_status("hc_render_001", "in_progress")
        record = update_handoff_status(
            "hc_render_001",
            "done",
            callback_payload={
                "summary": "Fixed the auth middleware",
                "artifacts": {
                    "pr": "https://github.com/example/repo/pull/42",
                    "branch": "fix/auth",
                },
                "verification": {"results": "All 15 tests pass"},
                "needs_kevin": False,
            },
        )

        msg = render_origin_message(record)
        assert "hc_render_001" in msg
        assert "done" in msg
        assert "Fixed the auth middleware" in msg
        assert "pull/42" in msg
        assert "fix/auth" in msg
        assert "All 15 tests pass" in msg

    def test_blocked_needs_kevin(self, tmp_path):
        create_handoff(handoff_id="hc_render_002")
        update_handoff_status("hc_render_002", "in_progress")
        record = update_handoff_status(
            "hc_render_002",
            "blocked",
            callback_payload={
                "summary": "Missing API key",
                "needs_kevin": True,
                "next_recommended_action": "Provide staging API key",
            },
        )

        msg = render_origin_message(record)
        assert "blocked" in msg
        assert "Needs Kevin" in msg
        assert "Provide staging API key" in msg


# ── should_auto_resume ───────────────────────────────────────


class TestAutoResume:
    def test_auto_resume_when_safe(self, tmp_path):
        create_handoff(
            handoff_id="hc_resume_001",
            next_action="Deploy to staging",
            next_action_safe=True,
        )
        update_handoff_status("hc_resume_001", "in_progress")
        record = update_handoff_status(
            "hc_resume_001",
            "done",
            callback_payload={"needs_kevin": False},
        )
        assert should_auto_resume(record) is True

    def test_no_resume_when_needs_kevin(self, tmp_path):
        create_handoff(
            handoff_id="hc_resume_002",
            next_action="Deploy to prod",
            next_action_safe=True,
        )
        update_handoff_status("hc_resume_002", "in_progress")
        record = update_handoff_status(
            "hc_resume_002",
            "done",
            callback_payload={"needs_kevin": True},
        )
        assert should_auto_resume(record) is False

    def test_no_resume_when_blocked(self, tmp_path):
        create_handoff(
            handoff_id="hc_resume_003",
            next_action="Continue",
            next_action_safe=True,
        )
        update_handoff_status("hc_resume_003", "in_progress")
        record = update_handoff_status(
            "hc_resume_003",
            "blocked",
            callback_payload={"needs_kevin": False},
        )
        assert should_auto_resume(record) is False

    def test_no_resume_when_action_unsafe(self, tmp_path):
        create_handoff(
            handoff_id="hc_resume_004",
            next_action="Deploy to prod",
            next_action_safe=False,
        )
        update_handoff_status("hc_resume_004", "in_progress")
        record = update_handoff_status(
            "hc_resume_004",
            "done",
            callback_payload={"needs_kevin": False},
        )
        assert should_auto_resume(record) is False

    def test_no_resume_when_no_action(self, tmp_path):
        create_handoff(handoff_id="hc_resume_005")
        update_handoff_status("hc_resume_005", "in_progress")
        record = update_handoff_status(
            "hc_resume_005",
            "done",
            callback_payload={"needs_kevin": False},
        )
        assert should_auto_resume(record) is False
