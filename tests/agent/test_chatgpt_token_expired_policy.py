"""A ChatGPT plan auth rejection must not enter Codex's stale-reasoning replay recovery."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import time

import pytest

from agent.agent_runtime_helpers import extract_api_error_context, recover_with_credential_pool
from agent.turn_api_error import handle_api_error
from agent.turn_retry_state import TurnRetryState


class _ExpiredToken(Exception):
    status_code = 401
    message = "Provided authentication token is expired. Please try signing in again."

    def __init__(self):
        super().__init__(self.message)
        self.body = {"error": {"code": "token_expired", "message": self.message,
                               "type": "invalid_request_error"}}


def _agent(provider):
    from run_agent import AIAgent

    agent = SimpleNamespace(
        provider=provider, model="account-model", api_mode="codex_responses",
        base_url=("https://api.openai.com/v1" if provider == "openai-chatgpt"
                  else "https://chatgpt.com/backend-api/codex"),
        api_key="fixture-bearer", log_prefix="", thinking_callback=None,
        context_compressor=None, _interrupt_requested=False, _image_rejecting_models=set(),
        _credential_pool=None, _fallback_index=0, _fallback_chain=[object()],
        _codex_reasoning_replay_enabled=True, _codex_reasoning_replay_rejected=False,
        _current_streamed_assistant_text="", verbose_logging=False,
        _extract_api_error_context=extract_api_error_context,
        _summarize_api_error=str, _client_log_context=lambda: provider,
        _is_openrouter_url=lambda: False,
        _try_refresh_codex_client_credentials=Mock(return_value=False),
        _try_activate_fallback=Mock(return_value=True),
        _has_pending_fallback=lambda: True,
    )
    agent._recover_with_credential_pool = lambda **kwargs: recover_with_credential_pool(agent, **kwargs)
    agent._disable_codex_reasoning_replay = Mock(
        side_effect=lambda messages, **kwargs: AIAgent._disable_codex_reasoning_replay(agent, messages, **kwargs),
    )
    for method in ("_invoke_api_request_error_hook", "_touch_activity", "_buffer_vprint", "_vprint",
                   "_dump_api_request_debug", "_flush_status_buffer", "_emit_diagnostic_status", "_persist_session"):
        setattr(agent, method, Mock())
    return agent


@pytest.mark.parametrize("provider", ["openai-chatgpt", "openai-codex"])
def test_expired_token_respects_provider_policy_before_replaying_reasoning(provider):
    agent, retry = _agent(provider), TurnRetryState()
    messages = [
        {"role": "assistant", "content": "Earlier answer", "codex_reasoning_items": [
            {"type": "reasoning", "id": "rs_fixture", "encrypted_content": "fixture-encrypted-reasoning"},
        ]},
        {"role": "user", "content": "Continue"},
    ]
    original = deepcopy(messages)
    api_messages = deepcopy(messages)
    verdict = handle_api_error(
        agent, api_error=_ExpiredToken(), _retry=retry, thinking_spinner=None,
        messages=messages, api_messages=api_messages, api_kwargs={"model": agent.model},
        system_message="", active_system_prompt="", conversation_history=deepcopy(messages[:-1]),
        approx_tokens=100, retry_count=0, max_retries=3, compression_attempts=0,
        max_compression_attempts=3, api_call_count=1, api_request_id="request-fixture",
        api_start_time=time.time(), effective_task_id=None, turn_id="turn-fixture",
        current_turn_user_idx=1,
    )

    agent._try_refresh_codex_client_credentials.assert_not_called()
    agent._try_activate_fallback.assert_not_called()
    if provider == "openai-chatgpt":
        assert verdict.action == "return"
        assert verdict.result["failed"] is True
        assert verdict.result["failure_reason"] == "provider_policy_blocked"
        assert verdict.result["failure_retryable"] is False
        assert messages == original and api_messages == original
        assert retry.invalid_encrypted_content_retry_attempted is False
        agent._disable_codex_reasoning_replay.assert_not_called()
    else:
        assert verdict.action == "continue"
        assert retry.invalid_encrypted_content_retry_attempted is True
        assert "codex_reasoning_items" not in messages[0]
        assert "codex_reasoning_items" not in api_messages[0]
        agent._disable_codex_reasoning_replay.assert_called_once()
