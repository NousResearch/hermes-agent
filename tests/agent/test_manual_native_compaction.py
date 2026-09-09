"""Explicit native Responses compaction for Codex OAuth sessions."""

from types import SimpleNamespace
from unittest.mock import MagicMock


def _agent() -> SimpleNamespace:
    client = MagicMock()
    client.responses.compact.return_value = SimpleNamespace(
        status="completed",
        output=[
            SimpleNamespace(
                type="compaction",
                encrypted_content="opaque-checkpoint",
            )
        ],
    )
    transport = MagicMock()
    transport.preflight_kwargs.side_effect = lambda kwargs, **_kwargs: kwargs
    return SimpleNamespace(
        api_mode="codex_responses",
        provider="openai-codex",
        model="gpt-5.6-terra",
        base_url="https://chatgpt.com/backend-api/codex",
        _base_url_hostname="chatgpt.com",
        _base_url_lower="https://chatgpt.com/backend-api/codex",
        runtime_capabilities={"native_compaction": True},
        codex_responses_native_compaction=True,
        compression_enabled=True,
        compression_checkpoint_required=False,
        codex_responses_compact_threshold=700_000,
        context_compressor=SimpleNamespace(protect_last_n=2, threshold_tokens=750_000),
        capabilities={},
        _get_transport=MagicMock(return_value=transport),
        _build_api_kwargs=MagicMock(
            return_value={
                "model": "gpt-5.6-terra",
                "input": [{"role": "user", "content": "older context"}],
                "instructions": "keep these instructions",
                "prompt_cache_key": "cache-key",
                "timeout": 42,
                "extra_headers": {"session_id": "session-1"},
            }
        ),
        _ensure_primary_openai_client=MagicMock(return_value=client),
    )


def test_manual_native_compaction_stores_checkpoint_at_preserved_assistant_boundary():
    """The manual endpoint compacts an old prefix without rewriting readable history."""
    from agent.native_compaction import manual_native_responses_compaction

    agent = _agent()
    history = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "middle user"},
        {"role": "assistant", "content": "preserved assistant"},
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent assistant"},
    ]

    result = manual_native_responses_compaction(agent, history)

    agent._build_api_kwargs.assert_called_once_with(history[:3])
    agent._get_transport.return_value.preflight_kwargs.assert_called_once()
    agent._ensure_primary_openai_client.assert_called_once_with(
        reason="manual_native_responses_compaction"
    )
    agent._ensure_primary_openai_client.return_value.responses.compact.assert_called_once_with(
        model="gpt-5.6-terra",
        input=[{"role": "user", "content": "older context"}],
        instructions="keep these instructions",
        prompt_cache_key="cache-key",
        timeout=42,
        extra_headers={"session_id": "session-1"},
    )
    assert result.checkpoint_count == 1
    assert result.messages is not history
    assert [message["role"] for message in result.messages] == [
        message["role"] for message in history
    ]
    assert "codex_reasoning_items" not in history[3]
    assert result.messages[3]["codex_reasoning_items"] == [
        {
            "type": "compaction",
            "encrypted_content": "opaque-checkpoint",
            "_issuer_kind": "codex_backend",
        }
    ]
