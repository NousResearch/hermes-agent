"""Unit tests for agent_bus.finalizer validators.

Covers spec §10 unit tests 1-4 + 6 (amend_learning / session exit gate are
deferred to later slices).
"""

import os
from unittest import mock

import pytest

from agent_bus import finalizer as fz


# -------- validate_close_payload --------
class TestValidatePayload:
    def test_missing_task_id(self):
        ok, code = fz.validate_close_payload({"outcome": "done", "summary": "ok"})
        assert not ok
        assert code == fz.ERR_MISSING_FIELD

    def test_empty_task_id(self):
        ok, code = fz.validate_close_payload({"task_id": "", "outcome": "done", "summary": "ok"})
        assert not ok and code == fz.ERR_MISSING_FIELD

    def test_whitespace_task_id(self):
        ok, code = fz.validate_close_payload({"task_id": "   ", "outcome": "done", "summary": "ok"})
        assert not ok and code == fz.ERR_MISSING_FIELD

    def test_missing_outcome(self):
        ok, code = fz.validate_close_payload({"task_id": "T-1", "summary": "ok"})
        assert not ok
        assert code == fz.ERR_INVALID_OUTCOME

    def test_unknown_outcome(self):
        ok, code = fz.validate_close_payload({"task_id": "T-1", "outcome": "cancelled", "summary": "ok"})
        assert not ok
        assert code == fz.ERR_INVALID_OUTCOME

    def test_missing_summary(self):
        ok, code = fz.validate_close_payload({"task_id": "T-1", "outcome": "done"})
        assert not ok
        assert code == fz.ERR_MISSING_FIELD

    def test_summary_empty(self):
        ok, code = fz.validate_close_payload({"task_id": "T-1", "outcome": "done", "summary": "   "})
        assert not ok and code == fz.ERR_MISSING_FIELD

    def test_non_dict_payload(self):
        ok, code = fz.validate_close_payload("not-a-dict")
        assert not ok and code == fz.ERR_MISSING_FIELD

    def test_minimal_valid_done(self):
        ok, code = fz.validate_close_payload({
            "task_id": "T-ABC123",
            "outcome": "done",
            "summary": "finished the thing",
        })
        assert ok and code is None

    def test_minimal_valid_fail(self):
        ok, code = fz.validate_close_payload({
            "task_id": "T-ABC123",
            "outcome": "fail",
            "summary": "broken because X",
        })
        assert ok and code is None

    def test_minimal_valid_keep_alive(self):
        ok, code = fz.validate_close_payload({
            "task_id": "T-ABC123",
            "outcome": "keep-alive",
            "summary": "still working on it",
        })
        assert ok and code is None


# -------- validate_transition --------
class TestValidateTransition:
    @pytest.mark.parametrize("src", ["ack", "progress", "keep-alive"])
    @pytest.mark.parametrize("dst", ["done", "fail", "keep-alive"])
    def test_legal_non_terminal_transitions(self, src, dst):
        ok, code = fz.validate_transition(src, dst)
        assert ok and code is None

    def test_pending_to_done_is_illegal(self):
        ok, code = fz.validate_transition("pending", "done")
        assert not ok and code == fz.ERR_INVALID_TRANSITION

    def test_pending_to_fail_is_illegal(self):
        ok, code = fz.validate_transition("pending", "fail")
        assert not ok and code == fz.ERR_INVALID_TRANSITION

    def test_pending_to_keep_alive_is_illegal(self):
        ok, code = fz.validate_transition("pending", "keep-alive")
        assert not ok and code == fz.ERR_INVALID_TRANSITION

    def test_done_to_done_is_idempotent(self):
        ok, code = fz.validate_transition("done", "done")
        assert ok and code == fz.ERR_ALREADY_TERMINAL_IDENTICAL
        assert fz.is_soft_ok(code)

    def test_fail_to_fail_is_idempotent(self):
        ok, code = fz.validate_transition("fail", "fail")
        assert ok and code == fz.ERR_ALREADY_TERMINAL_IDENTICAL

    def test_done_to_fail_is_flip(self):
        ok, code = fz.validate_transition("done", "fail")
        assert not ok and code == fz.ERR_INVALID_TERMINAL_FLIP

    def test_fail_to_done_is_flip(self):
        ok, code = fz.validate_transition("fail", "done")
        assert not ok and code == fz.ERR_INVALID_TERMINAL_FLIP

    def test_terminal_to_keep_alive_is_illegal(self):
        ok, code = fz.validate_transition("done", "keep-alive")
        assert not ok and code == fz.ERR_INVALID_TRANSITION

    def test_timeout_to_same_outcome_is_not_expressible(self):
        # timeout is a terminal status but not in VALID_OUTCOMES
        # so timeout→done or timeout→fail is a flip
        ok, code = fz.validate_transition("timeout", "done")
        assert not ok and code == fz.ERR_INVALID_TERMINAL_FLIP

    def test_invalid_outcome_name(self):
        ok, code = fz.validate_transition("ack", "cancelled")
        assert not ok and code == fz.ERR_INVALID_OUTCOME


# -------- enforce_close helper --------
class TestEnforceClose:
    def test_proceed_on_valid_and_legal(self):
        decision, code = fz.enforce_close(
            {"task_id": "T-1", "outcome": "done", "summary": "finished"},
            current_status="progress",
        )
        assert decision == "proceed" and code is None

    def test_idempotent_when_same_terminal(self):
        decision, code = fz.enforce_close(
            {"task_id": "T-1", "outcome": "done", "summary": "finished again"},
            current_status="done",
        )
        assert decision == "idempotent" and code == fz.ERR_ALREADY_TERMINAL_IDENTICAL

    def test_reject_on_missing_field(self):
        decision, code = fz.enforce_close(
            {"task_id": "T-1", "outcome": "done"},
            current_status="progress",
        )
        assert decision == "reject" and code == fz.ERR_MISSING_FIELD

    def test_reject_on_flip(self):
        decision, code = fz.enforce_close(
            {"task_id": "T-1", "outcome": "fail", "summary": "flipping"},
            current_status="done",
        )
        assert decision == "reject" and code == fz.ERR_INVALID_TERMINAL_FLIP

    def test_reject_on_pending_close(self):
        decision, code = fz.enforce_close(
            {"task_id": "T-1", "outcome": "done", "summary": "skipping ack"},
            current_status="pending",
        )
        assert decision == "reject" and code == fz.ERR_INVALID_TRANSITION


# -------- gate mode --------
class TestGateMode:
    def test_default_mode_is_core(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_FINALIZER_GATE", None)
            assert fz.get_mode() == fz.MODE_CORE
            assert fz.is_enforcing()

    def test_off_mode(self):
        with mock.patch.dict(os.environ, {"HERMES_FINALIZER_GATE": "off"}):
            assert fz.get_mode() == fz.MODE_OFF
            assert not fz.is_enforcing()

    def test_advisory_mode(self):
        with mock.patch.dict(os.environ, {"HERMES_FINALIZER_GATE": "advisory"}):
            assert fz.get_mode() == fz.MODE_ADVISORY
            assert not fz.is_enforcing()

    def test_full_mode(self):
        with mock.patch.dict(os.environ, {"HERMES_FINALIZER_GATE": "full"}):
            assert fz.get_mode() == fz.MODE_FULL
            assert fz.is_enforcing()

    def test_unknown_mode_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"HERMES_FINALIZER_GATE": "nonsense"}):
            assert fz.get_mode() == fz.DEFAULT_MODE

    def test_mode_is_case_insensitive(self):
        with mock.patch.dict(os.environ, {"HERMES_FINALIZER_GATE": "ADVISORY"}):
            assert fz.get_mode() == fz.MODE_ADVISORY


# -------- keepalive timeout --------
class TestKeepaliveTimeout:
    def test_default_is_1800(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FINALIZER_KEEPALIVE_TIMEOUT_SEC", None)
            assert fz.keepalive_timeout_sec() == 1800

    def test_custom_value(self):
        with mock.patch.dict(os.environ, {"FINALIZER_KEEPALIVE_TIMEOUT_SEC": "600"}):
            assert fz.keepalive_timeout_sec() == 600

    def test_invalid_falls_back(self):
        with mock.patch.dict(os.environ, {"FINALIZER_KEEPALIVE_TIMEOUT_SEC": "abc"}):
            assert fz.keepalive_timeout_sec() == 1800

    def test_floor_at_60(self):
        with mock.patch.dict(os.environ, {"FINALIZER_KEEPALIVE_TIMEOUT_SEC": "10"}):
            assert fz.keepalive_timeout_sec() == 60
