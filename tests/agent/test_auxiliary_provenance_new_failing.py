"""New failing tests for Q3, Q5, Q6, Q9, Q4 (BLOCKER + SHOULD-FIX)."""
import pytest
from unittest.mock import patch, MagicMock
from agent.auxiliary_client import (
    _record_auxiliary_provenance,
    _get_auxiliary_provenance,
    get_auxiliary_provenance,
    _RELAY_AUX_CALL_CONTEXT,
)


def test_record_appends_success_status_after_response(monkeypatch):
    """After a successful LLM response, status='success' should be recorded."""
    monkeypatch.setattr(
        "agent.auxiliary_client.cfg_get",
        lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
    )
    token = _RELAY_AUX_CALL_CONTEXT.set({
        "task": "test", "request_id": "r1", "attempt_count": 0,
        "provider": "p", "model": "m", "response_model": None,
        "api_mode": "chat_completions",
        "attempts": [{"provider":"openai","model":"gpt-4","status":"attempted"}],
        "served_by": None, "served_model": None,
        "fallback_chain_used": False, "fallback_count": 0,
        "final_status": "pending",
        "enabled": True,
    })
    try:
        _record_auxiliary_provenance(provider="openai", model="gpt-4", status="success")
        context = _RELAY_AUX_CALL_CONTEXT.get()
        assert context["attempts"][1]["status"] == "success"
        assert context["final_status"] == "success"
        assert context["served_by"] == "openai"
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)


def test_record_appends_failed_status_after_exception(monkeypatch):
    """After an exception, status='failed' + error should be recorded."""
    monkeypatch.setattr(
        "agent.auxiliary_client.cfg_get",
        lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
    )
    token = _RELAY_AUX_CALL_CONTEXT.set({
        "task": "test", "request_id": "r2", "attempt_count": 0,
        "provider": "p", "model": "m", "response_model": None,
        "api_mode": "chat_completions",
        "attempts": [{"provider":"openai","model":"gpt-4","status":"attempted"}],
        "served_by": None, "served_model": None,
        "fallback_chain_used": False, "fallback_count": 0,
        "final_status": "pending",
        "enabled": True,
    })
    try:
        exc = ValueError("service unavailable")
        _record_auxiliary_provenance(provider="openai", model="gpt-4", status="failed", error=exc)
        context = _RELAY_AUX_CALL_CONTEXT.get()
        assert context["attempts"][1]["status"] == "failed"
        assert "service unavailable" in context["attempts"][1]["failure"]
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)


def test_sanitize_strips_bearer_token(monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.cfg_get",
        lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
    )
    token = _RELAY_AUX_CALL_CONTEXT.set({
        "task": "test", "request_id": "r3", "attempt_count": 0,
        "provider": "", "model": "", "response_model": None,
        "api_mode": "chat_completions",
        "attempts": [], "served_by": None,
        "served_model": None,
        "fallback_chain_used": False, "fallback_count": 0,
        "final_status": "pending",
        "enabled": True,
    })
    try:
        from agent.auxiliary_client import _sanitize_error_message
        msg = "Error: Bearer secret-token-abc123 occurred"
        cleaned = _sanitize_error_message(msg)
        assert "secret-token-abc123" not in cleaned
        assert "[REDACTED]" in cleaned
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)


def test_sanitize_strips_api_key_in_url(monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.cfg_get",
        lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
    )
    token = _RELAY_AUX_CALL_CONTEXT.set({
        "task": "test", "request_id": "r4", "attempt_count": 0,
        "provider": "", "model": "", "response_model": None,
        "api_mode": "chat_completions",
        "attempts": [], "served_by": None,
        "served_model": None,
        "fallback_chain_used": False, "fallback_count": 0,
        "final_status": "pending",
        "enabled": True,
    })
    try:
        from agent.auxiliary_client import _sanitize_error_message
        msg = "https://example.com?key=sk-secret123&token=abc"
        cleaned = _sanitize_error_message(msg)
        assert "sk-secret123" not in cleaned
        assert "abc" not in cleaned
        assert "[REDACTED]" in cleaned
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)


def test_cap_at_max_attempts_drops_oldest(monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.cfg_get",
        lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
    )
    token = _RELAY_AUX_CALL_CONTEXT.set({
        "task": "test", "request_id": "r5", "attempt_count": 0,
        "provider": "", "model": "", "response_model": None,
        "api_mode": "chat_completions",
        "attempts": [], "served_by": None,
        "served_model": None,
        "fallback_chain_used": False, "fallback_count": 0,
        "final_status": "pending",
        "enabled": True,
    })
    try:
        from agent.auxiliary_client import MAX_PROVENANCE_ATTEMPTS
        for i in range(MAX_PROVENANCE_ATTEMPTS):
            _record_auxiliary_provenance(provider=f"p{i}", model=f"m{i}", status="attempted")
        context = _RELAY_AUX_CALL_CONTEXT.get()
        assert len(context["attempts"]) == MAX_PROVENANCE_ATTEMPTS
        _record_auxiliary_provenance(provider="new", model="m", status="attempted")
        assert len(context["attempts"]) == MAX_PROVENANCE_ATTEMPTS
        assert context["attempts"][0]["provider"] == "p1"
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)


def test_final_status_failed_when_no_success_attempt(monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.cfg_get",
        lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
    )
    token = _RELAY_AUX_CALL_CONTEXT.set({
        "task": "test", "request_id": "r6", "attempt_count": 0,
        "provider": "", "model": "", "response_model": None,
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
        assert context["final_status"] == "failed"
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)


def test_public_api_get_auxiliary_provenance_alias(monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.cfg_get",
        lambda cfg, *keys, default=None: True if keys == ("auxiliary", "expose_provenance") else default,
    )
    assert callable(get_auxiliary_provenance)
    assert get_auxiliary_provenance is _get_auxiliary_provenance
