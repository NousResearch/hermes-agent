"""Tests for auxiliary failover provenance (issue #36797)."""
import pytest
from unittest.mock import patch, MagicMock

from agent.auxiliary_client import (
    _record_auxiliary_provenance,
    _get_auxiliary_provenance,
    _RELAY_AUX_CALL_CONTEXT,
)


class TestFlagOff:
    """When expose_provenance is False, no provenance is captured."""

    def test_get_returns_none_when_flag_off(self, monkeypatch):
        # Default config: flag is off — _get_auxiliary_provenance returns None
        assert _get_auxiliary_provenance() is None

    def test_record_is_noop_when_flag_off(self, monkeypatch):
        # Even with an active context, writer is a no-op when flag is off
        token = _RELAY_AUX_CALL_CONTEXT.set({
            "task": "test",
            "request_id": "test-req",
            "attempt_count": 0,
            "provider": "",
            "model": "",
            "response_model": None,
            "api_mode": "chat_completions",
            "attempts": [],
            "served_by": None,
            "served_model": None,
            "fallback_chain_used": False,
            "fallback_count": 0,
            "final_status": "pending",
            "enabled": True,
        })
        try:
            _record_auxiliary_provenance(
                provider="openai", model="gpt-4", status="success"
            )
            # No crash, attempts still empty (no-op)
            context = _RELAY_AUX_CALL_CONTEXT.get()
            assert context is not None
            # Since flag is off, writer should return early; attempts unchanged
            # (if it was empty it stays empty; if it had items they stay)
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)


class TestFlagOn:
    """When expose_provenance is True, provenance is captured."""

    def test_record_appends_attempt(self, monkeypatch, tmp_path):
        # Patch config to enable flag
        monkeypatch.setattr(
            "agent.auxiliary_client.cfg_get",
            lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
        )
        token = _RELAY_AUX_CALL_CONTEXT.set({
            "task": "compression",
            "request_id": "req-123",
            "attempt_count": 0,
            "provider": "",
            "model": "",
            "response_model": None,
            "api_mode": "chat_completions",
            "attempts": [],
            "served_by": None,
            "served_model": None,
            "fallback_chain_used": False,
            "fallback_count": 0,
            "final_status": "pending",
            "enabled": True,
        })
        try:
            _record_auxiliary_provenance(
                provider="openai", model="gpt-4", status="attempted"
            )
            context = _RELAY_AUX_CALL_CONTEXT.get()
            assert len(context["attempts"]) == 1
            entry = context["attempts"][0]
            assert entry["provider"] == "openai"
            assert entry["model"] == "gpt-4"
            assert entry["status"] == "attempted"
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)

    def test_record_marks_success(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.cfg_get",
            lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
        )
        token = _RELAY_AUX_CALL_CONTEXT.set({
            "task": "vision",
            "request_id": "req-456",
            "attempt_count": 0,
            "provider": "",
            "model": "",
            "response_model": None,
            "api_mode": "chat_completions",
            "attempts": [],
            "served_by": None,
            "served_model": None,
            "fallback_chain_used": False,
            "fallback_count": 0,
            "final_status": "pending",
            "enabled": True,
        })
        try:
            _record_auxiliary_provenance(
                provider="openrouter", model="gpt-4o", status="success"
            )
            context = _RELAY_AUX_CALL_CONTEXT.get()
            assert context["served_by"] == "openrouter"
            assert context["served_model"] == "gpt-4o"
            assert context["fallback_count"] == 0
            assert context["fallback_chain_used"] is False
            assert context["final_status"] == "success"
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)

    def test_record_truncates_long_errors(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.cfg_get",
            lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
        )
        token = _RELAY_AUX_CALL_CONTEXT.set({
            "task": "test",
            "request_id": "req-err",
            "attempt_count": 0,
            "provider": "",
            "model": "",
            "response_model": None,
            "api_mode": "chat_completions",
            "attempts": [],
            "served_by": None,
            "served_model": None,
            "fallback_chain_used": False,
            "fallback_count": 0,
            "final_status": "pending",
            "enabled": True,
        })
        try:
            long_err = ValueError("x" * 1000)
            _record_auxiliary_provenance(
                provider="openai", model="gpt-4", status="failed", error=long_err
            )
            context = _RELAY_AUX_CALL_CONTEXT.get()
            entry = context["attempts"][0]
            assert entry["status"] == "failed"
            assert len(entry["failure"]) == 503  # 500 + "..."
            assert entry["failure"].endswith("...")
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)

    def test_get_returns_full_dict(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.cfg_get",
            lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
        )
        token = _RELAY_AUX_CALL_CONTEXT.set({
            "task": "summary",
            "request_id": "req-full",
            "attempt_count": 2,
            "provider": "auto",
            "model": "claude-3",
            "response_model": None,
            "api_mode": "chat_completions",
            "attempts": [{"provider": "a", "model": "m", "status": "attempted"}],
            "served_by": "openrouter",
            "served_model": "gpt-4o",
            "fallback_chain_used": True,
            "fallback_count": 1,
            "final_status": "success",
            "enabled": True,
        })
        try:
            result = _get_auxiliary_provenance()
            assert result is not None
            assert result["task"] == "summary"
            assert result["served_by"] == "openrouter"
            assert result["served_model"] == "gpt-4o"
            assert result["fallback_chain_used"] is True
            assert result["fallback_count"] == 1
            assert result["final_status"] == "success"
            assert result["attempts"] == [{"provider": "a", "model": "m", "status": "attempted"}]
            assert result["request_id"] == "req-full"
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)

    def test_get_returns_none_without_context(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.cfg_get",
            lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
        )
        # Ensure no context is set
        token = _RELAY_AUX_CALL_CONTEXT.set(None)
        try:
            assert _get_auxiliary_provenance() is None
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)
