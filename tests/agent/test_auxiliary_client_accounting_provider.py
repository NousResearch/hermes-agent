"""Auxiliary-task accounting records the concrete billing provider and base URL (#78953).

The relay call context carries the originally-resolved route; retry and fallback paths
call the accounting chokepoint without provider/base_url, so the row must be backfilled
from that context instead of landing with an empty route.
"""

from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from agent.aux_accounting import (
    set_accounting_context,
    reset_accounting_context,
)
from agent.auxiliary_client import (
    _validate_llm_response,
    _set_relay_auxiliary_route,
    _RELAY_AUX_CALL_CONTEXT,
)


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _usage_rows(db, session_id):
    with db._lock:
        rows = db._conn.execute(
            "SELECT * FROM session_model_usage WHERE session_id = ? ORDER BY task",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _relay_context(request_id: str):
    return {
        "task": "approval",
        "request_id": request_id,
        "attempt_count": 1,
        "provider": "nous",
        "model": "anthropic/claude-opus-5",
        "base_url": "https://inference-api.nousresearch.com/v1",
        "response_model": None,
        "api_mode": "chat_completions",
    }


def test_validate_llm_response_backfills_omitted_provider_and_base_url(db):
    """Verify that _validate_llm_response backfills the resolved provider and base
    URL from _RELAY_AUX_CALL_CONTEXT when the caller omits them (retry/fallback paths)."""
    db.create_session("s_test", source="cli")
    token = set_accounting_context(db, "s_test")

    # Set up the context as it would be resolved by _call_llm_impl / _set_relay_auxiliary_route
    context_token = _RELAY_AUX_CALL_CONTEXT.set(_relay_context("aux-req-123"))

    # Raw response from client
    mock_resp = SimpleNamespace(
        model="anthropic/claude-opus-5",
        choices=[SimpleNamespace(message=SimpleNamespace(content="APPROVE"))],
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=20,
            total_tokens=1020,
            cache_read_input_tokens=800,
            cache_creation_input_tokens=100,
        )
    )

    try:
        # Call with provider/base_url omitted, as the retry and fallback paths do
        _validate_llm_response(mock_resp, "approval")
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(context_token)
        reset_accounting_context(token)

    rows = _usage_rows(db, "s_test")
    assert len(rows) == 1
    r = rows[0]
    assert r["task"] == "approval"
    # Concrete billing provider and base URL must be populated
    assert r["billing_provider"] == "nous"
    assert r["billing_base_url"] == "https://inference-api.nousresearch.com/v1"
    # Cache read/write/input tokens must be correctly computed/parsed
    assert r["cache_read_tokens"] == 800
    assert r["cache_write_tokens"] == 100
    assert r["input_tokens"] == 1000 - 800 - 100  # 100


def test_validate_llm_response_keeps_concrete_hints(db):
    """Only omitted hints are backfilled: a concrete provider hint is never overridden."""
    db.create_session("s_test_hint", source="cli")
    token = set_accounting_context(db, "s_test_hint")
    context_token = _RELAY_AUX_CALL_CONTEXT.set(_relay_context("aux-req-hint"))

    mock_resp = SimpleNamespace(
        model="anthropic/claude-opus-5",
        choices=[SimpleNamespace(message=SimpleNamespace(content="APPROVE"))],
        usage=SimpleNamespace(prompt_tokens=800, completion_tokens=10, total_tokens=810),
    )

    try:
        _validate_llm_response(mock_resp, "approval", provider="openrouter")
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(context_token)
        reset_accounting_context(token)

    rows = _usage_rows(db, "s_test_hint")
    assert len(rows) == 1
    assert rows[0]["billing_provider"] == "openrouter"


def test_set_relay_auxiliary_route_publishes_base_url():
    """The publication half: _set_relay_auxiliary_route carries the call's base URL
    into the relay call context; the legacy three-argument form leaves an
    already-published base URL untouched."""
    token = _RELAY_AUX_CALL_CONTEXT.set({
        "task": "approval",
        "request_id": "aux-req-pub",
        "attempt_count": 0,
        "provider": "",
        "model": "",
        "response_model": None,
        "api_mode": None,
    })
    try:
        _set_relay_auxiliary_route(
            "nous", "anthropic/claude-opus-5", "chat_completions",
            base_url="https://inference-api.nousresearch.com/v1",
        )
        ctx = _RELAY_AUX_CALL_CONTEXT.get()
        assert ctx["provider"] == "nous"
        assert ctx["base_url"] == "https://inference-api.nousresearch.com/v1"

        _set_relay_auxiliary_route("nous", "anthropic/claude-opus-5", None)
        assert _RELAY_AUX_CALL_CONTEXT.get()["base_url"] == "https://inference-api.nousresearch.com/v1"
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)
