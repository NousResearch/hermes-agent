"""Tests that auxiliary task token/cache accounting resolves the correct concrete billing provider

and does not discard Anthropic cache read/creation token counts (issue #78953).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from hermes_state import SessionDB
from agent.aux_accounting import (
    set_accounting_context,
    reset_accounting_context,
    record_aux_usage,
)
from agent.auxiliary_client import (
    _validate_llm_response,
    _RELAY_AUX_CALL_CONTEXT,
    _AnthropicCompletionsAdapter,
    call_llm,
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


def test_anthropic_adapter_preserves_cache_tokens():
    """Verify that _AnthropicCompletionsAdapter retains cache read/creation

    input tokens in its normalized usage output.
    """
    mock_real_client = MagicMock()
    mock_raw_response = SimpleNamespace(
        id="msg_123",
        content=[SimpleNamespace(type="text", text="Hello from Kimi!")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=150,
            cache_read_input_tokens=800,
            cache_creation_input_tokens=100,
        )
    )

    adapter = _AnthropicCompletionsAdapter(
        real_client=mock_real_client,
        model="claude-3-opus",
        is_oauth=False,
        base_url="https://api.anthropic.com"
    )

    with patch("agent.anthropic_adapter.create_anthropic_message", return_value=mock_raw_response):
        res = adapter.create(messages=[{"role": "user", "content": "hi"}])

    assert res.usage.prompt_tokens == 1000
    assert res.usage.completion_tokens == 150
    assert res.usage.input_tokens == 1000
    assert res.usage.output_tokens == 150
    assert res.usage.cache_read_input_tokens == 800
    assert res.usage.cache_creation_input_tokens == 100


def test_validate_llm_response_resolves_auto_provider_and_base_url(db):
    """Verify that _validate_llm_response fetches resolved provider

    and base URL from _RELAY_AUX_CALL_CONTEXT if they are omitted or 'auto'.
    """
    db.create_session("s_test", source="cli")
    token = set_accounting_context(db, "s_test")

    # Set up the context as it would be resolved by _call_llm_impl / _set_relay_auxiliary_route
    context_token = _RELAY_AUX_CALL_CONTEXT.set({
        "task": "approval",
        "request_id": "aux-req-123",
        "attempt_count": 1,
        "provider": "nous",
        "model": "anthropic/claude-opus-5",
        "base_url": "https://inference-api.nousresearch.com/v1",
        "response_model": None,
        "api_mode": "chat_completions",
    })

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
        # Call with provider="auto" or omitted provider to test fallback resolution
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
