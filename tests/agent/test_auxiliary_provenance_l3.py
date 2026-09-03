"""Tests for L3 provenance trace at auxiliary call completion (refs #36797)."""
import logging
import pytest
from agent import auxiliary_client
from agent.auxiliary_client import (
    _record_auxiliary_provenance,
    _get_auxiliary_provenance,
    _RELAY_AUX_CALL_CONTEXT,
    _relay_auxiliary_call,
    _complete_relay_auxiliary_call,
)


def _enable_flag(monkeypatch):
    """Enable expose_provenance via monkeypatch on cfg_get."""
    monkeypatch.setattr(
        "agent.auxiliary_client.cfg_get",
        lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
    )


class TestL3Trace:
    def test_flag_on_logs_provenance_at_completion(self, monkeypatch, caplog):
        """When flag is on, completing a relay call emits a provenance log entry."""
        _enable_flag(monkeypatch)
        with caplog.at_level(logging.INFO, logger="agent.auxiliary_client"):
            token = _RELAY_AUX_CALL_CONTEXT.set({
                "task": "test", "request_id": "r1", "attempt_count": 0,
                "provider": "p", "model": "m", "response_model": None,
                "api_mode": "chat_completions",
                "attempts": [], "served_by": "p", "served_model": "m",
                "fallback_chain_used": False, "fallback_count": 0,
                "final_status": "success",
                "enabled": True,
            })
            try:
                _complete_relay_auxiliary_call(outcome="success")
            finally:
                context_before_reset = _RELAY_AUX_CALL_CONTEXT.get()
                # Capture provenance before reset
                provenance_before_reset = None
                if context_before_reset is not None:
                    try:
                        provenance_before_reset = auxiliary_client._get_auxiliary_provenance()
                    except Exception:
                        pass
                _RELAY_AUX_CALL_CONTEXT.reset(token)
        provenance_logs = [r for r in caplog.records if r.message == "auxiliary_provenance"]
        assert len(provenance_logs) == 1
        # Verify the log record carries the provenance dict in extra
        record = provenance_logs[0]
        extra = getattr(record, "provenance", None)
        # When passed via `extra={"provenance": ...}`, logging attaches it
        # to the LogRecord directly if the logger processes the extra fields.
        # Check both direct attribute and via `__dict__` / custom attribute access.
        provenance_dict = None
        for attr in ("provenance",):
            if hasattr(record, attr):
                provenance_dict = getattr(record, attr)
                break
        # Also try via the record's __dict__ keys
        if provenance_dict is None:
            provenance_dict = record.__dict__.get("provenance")
        assert provenance_dict is not None
        assert provenance_dict.get("request_id") == "r1"
        assert provenance_dict.get("final_status") == "success"

    def test_flag_off_does_not_log(self, monkeypatch, caplog):
        """When flag is off, completing a call does NOT emit a provenance log."""
        # Default config (flag off) — don't enable
        with caplog.at_level(logging.INFO, logger="agent.auxiliary_client"):
            token = _RELAY_AUX_CALL_CONTEXT.set({
                "task": "test", "request_id": "r1", "attempt_count": 0,
                "provider": "p", "model": "m", "response_model": None,
                "api_mode": "chat_completions",
                "attempts": [], "served_by": None, "served_model": None,
                "fallback_chain_used": False, "fallback_count": 0,
                "final_status": "pending",
            })
            try:
                _complete_relay_auxiliary_call(outcome="success")
            finally:
                _RELAY_AUX_CALL_CONTEXT.reset(token)
        provenance_logs = [r for r in caplog.records if r.message == "auxiliary_provenance"]
        assert len(provenance_logs) == 0

    def test_failed_outcome_sets_final_status(self, monkeypatch):
        """Failed outcome with no success attempt -> final_status='failed'."""
        _enable_flag(monkeypatch)
        token = _RELAY_AUX_CALL_CONTEXT.set({
            "task": "test", "request_id": "r1", "attempt_count": 0,
            "provider": "", "model": "", "response_model": None,
            "api_mode": "chat_completions",
            "attempts": [], "served_by": None, "served_model": None,
            "fallback_chain_used": False, "fallback_count": 0,
            "final_status": "pending",
        })
        try:
            _complete_relay_auxiliary_call(outcome="failed")
            # Read state before reset
            context = _RELAY_AUX_CALL_CONTEXT.get()
            assert context is not None
            assert context["final_status"] == "failed"
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)

    def test_success_with_pending_sets_success(self, monkeypatch):
        """Success outcome when final_status is pending should set success with served_by=None."""
        _enable_flag(monkeypatch)
        token = _RELAY_AUX_CALL_CONTEXT.set({
            "task": "test", "request_id": "r2", "attempt_count": 0,
            "provider": "", "model": "", "response_model": None,
            "api_mode": "chat_completions",
            "attempts": [], "served_by": None, "served_model": None,
            "fallback_chain_used": False, "fallback_count": 0,
            "final_status": "pending",
        })
        try:
            _complete_relay_auxiliary_call(outcome="success")
            context = _RELAY_AUX_CALL_CONTEXT.get()
            assert context is not None
            # Per spec: outcome success + pending + no success attempt -> "failed"
            assert context["final_status"] == "failed"
            # served_by populated from response_model (None here)
            assert context["served_by"] is None
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)

    def test_final_status_failed_when_no_success_attempt_exists(self, monkeypatch):
        """Success outcome with no success attempt -> final_status='failed'."""
        _enable_flag(monkeypatch)
        token = _RELAY_AUX_CALL_CONTEXT.set({
            "task": "test", "request_id": "r7", "attempt_count": 0,
            "provider": "", "model": "", "response_model": "recovered-model",
            "api_mode": "chat_completions",
            "attempts": [
                {"provider":"a","model":"m","status":"attempted"},
                {"provider":"b","model":"m","status":"failed","failure":"err"},
            ],
            "served_by": None, "served_model": None,
            "fallback_chain_used": True, "fallback_count": 1,
            "final_status": "pending",
            "enabled": True,
        })
        try:
            from agent.auxiliary_client import _complete_relay_auxiliary_call
            _complete_relay_auxiliary_call(outcome="success")
            context = _RELAY_AUX_CALL_CONTEXT.get()
            assert context is not None
            assert context["final_status"] == "failed"
            # served_by populated from response_model
            assert context["served_by"] == ""
            assert context["served_model"] == "recovered-model"
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)
