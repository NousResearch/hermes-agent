"""An Azure cross-resource rejection repairs encrypted replay, not credentials.

All error bodies and conversation items are synthetic. These contracts exercise
real SDK errors and Hermes recovery code without contacting a provider.
"""
from __future__ import annotations

from copy import deepcopy

import httpx
import pytest
from openai import AuthenticationError, BadRequestError, RateLimitError

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import recover_after_classification
from agent.turn_retry_state import TurnRetryState
from run_agent import AIAgent

_CROSS_RESOURCE = (
    "The requested item was created under a different Azure OpenAI resource. "
    "Use the same resource that created the item to access it."
)


def _api_error(message, status=400):
    body = {"error": {"message": message, "type": "invalid_request_error", "code": 0}}
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://azure-relay.example/v1/responses"), json=body,
    )
    error_type = {400: BadRequestError, 401: AuthenticationError, 429: RateLimitError}[status]
    return error_type(message, response=response, body=body)


class _Agent(AIAgent):
    log_prefix = ""
    provider = "custom"
    api_mode = "codex_responses"
    model = "test-model"
    api_key = "test-only"
    _codex_reasoning_replay_enabled = True
    _codex_reasoning_replay_rejected = False

    def __init__(self):
        self.base_url = "https://azure-relay.example/v1"
        self.refresh_calls = 0

    def _recover_with_credential_pool(self, **kwargs):
        return False, False

    def _try_refresh_codex_client_credentials(self, *, force=True):
        self.refresh_calls += 1
        return False

    def _extract_api_error_context(self, error):
        from agent.agent_runtime_helpers import extract_api_error_context
        return extract_api_error_context(error)

    def _get_transport(self, api_mode=None):
        return None

    def _vprint(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("provider,message,status,repair", [
    pytest.param("custom", _CROSS_RESOURCE, 400, True, id="custom-relay"),
    pytest.param("azure-foundry", _CROSS_RESOURCE, 400, True, id="azure-foundry"),
    pytest.param("custom", _CROSS_RESOURCE.upper(), 400, True, id="case-insensitive"),
    pytest.param(
        "custom",
        "The requested item was created under a different *** OpenAI resource. "
        "Use the same resource that created the item to access it.",
        400, True, id="masked-word-qualifier",
    ),
    pytest.param(
        "custom",
        "The requested item was created under a different Azure OpenAI deployment. "
        "Use the same deployment that created the item to access it.",
        400, False, id="different-frame-not-resource",
    ),
    pytest.param(
        "custom",
        "The requested item was created under a different Microsoft Azure OpenAI resource. "
        "Use the same resource that created the item to access it.",
        400, True, id="multi-word-qualifier",
    ),
    pytest.param(
        "custom",
        "The requested item was created under a different Azure OpenAI resource. "
        "Request id: req-123.",
        400, True, id="trailing-text-not-matched",
    ),
    pytest.param(
        "custom",
        "The requested item was created under a different Azure OpenAI resource quota exceeded. "
        "Retry after the limit resets.",
        400, False, id="text-after-resource",
    ),
    pytest.param(
        "custom",
        "The requested item was created under a different Azure Cognitive Search index. "
        "Use the same openai resource that created the item.",
        400, False, id="resource-mention-in-other-sentence",
    ),
    pytest.param(
        "custom",
        "The request failed for an unrelated reason. Use the same openai resource.",
        400, False, id="second-sentence-only",
    ),
    pytest.param("custom", "The Azure OpenAI resource was not found.", 400, False, id="other-resource-error"),
    pytest.param("custom", "Invalid API key for this Azure OpenAI resource.", 401, False, id="auth-error"),
    pytest.param("custom", "Rate limit exceeded for this Azure OpenAI resource.", 429, False, id="quota-error"),
])
def test_only_cross_resource_replay_rejections_reach_repair(provider, message, status, repair):
    result = classify_api_error(_api_error(message, status), provider=provider, model="test-model")
    assert (result.reason == FailoverReason.invalid_encrypted_content) is repair
    if repair:
        assert result.retryable is True
        assert result.should_fallback is False


def test_cross_resource_400_strips_cached_replay_and_retries_once():
    agent, retry = _Agent(), TurnRetryState()
    messages = [
        {"role": "user", "content": "Check a page and report its title."},
        {
            "role": "assistant", "content": "I will check the page.",
            "tool_calls": [{"id": "call_test", "type": "function", "function": {
                "name": "browser", "arguments": "{}",
            }}],
            "codex_reasoning_items": [{
                "type": "reasoning", "id": "rs_test", "encrypted_content": "test-ciphertext",
            }],
        },
        {"role": "tool", "tool_call_id": "call_test", "content": "Example page"},
        {"role": "assistant", "content": "The page title is Example page."},
    ]
    api_messages, expected = deepcopy(messages), deepcopy(messages)
    expected[1].pop("codex_reasoning_items")
    err = _api_error(_CROSS_RESOURCE)
    classified = classify_api_error(err, provider=agent.provider, model=agent.model)
    assert classified.reason == FailoverReason.invalid_encrypted_content

    retried, _ = recover_after_classification(
        agent, err, classified, retry, status_code=400,
        error_context=agent._extract_api_error_context(err),
        messages=messages, api_messages=api_messages,
    )

    assert retried is True
    assert agent.refresh_calls == 0
    assert messages == expected
    assert api_messages == expected
    assert agent._codex_reasoning_replay_enabled is True
    assert agent._codex_reasoning_replay_rejected is True

    # The same turn cannot repeatedly consume the repair rung after its blobs are gone.
    retried_again, _ = recover_after_classification(
        agent, err, classified, retry, status_code=400,
        error_context=agent._extract_api_error_context(err),
        messages=messages, api_messages=api_messages,
    )
    assert retried_again is False
    assert messages == expected
    assert api_messages == expected


@pytest.mark.parametrize("replay_state", ["unverified", "current-route", "ids-only", "replay-disabled"])
def test_recovery_rebuilds_input_without_stale_message_ids(monkeypatch, replay_state):
    from agent.transports.codex import ResponsesApiTransport

    agent, retry, transport = _Agent(), TurnRetryState(), ResponsesApiTransport()
    monkeypatch.setattr(agent, "_get_transport", lambda api_mode=None: transport)
    agent._codex_reasoning_replay_enabled = replay_state != "replay-disabled"
    message_item = {
        "type": "message", "role": "assistant", "id": "msg_stale",
        "status": "incomplete", "phase": "commentary",
        "content": [{"type": "output_text", "text": "Checking the page."}],
    }
    messages = [
        {"role": "user", "content": "Check the page."},
        {"role": "assistant", "content": "Checking the page.",
         "codex_message_items": [message_item],
         "tool_calls": [{"id": "call_test", "type": "function", "function": {
             "name": "browser", "arguments": "{}",
         }}]},
        {"role": "tool", "tool_call_id": "call_test", "content": "Example page"},
        {"role": "assistant", "content": "The title is Example page.",
         "codex_message_items": [{**deepcopy(message_item), "id": "msg_final",
                                  "status": "completed", "phase": "final_answer",
                                  "content": [{"type": "output_text", "text": "The title is Example page."}]}]},
        {"role": "user", "content": "Continue."},
    ]
    params = {"model": agent.model, "base_url": agent.base_url,
              "replay_encrypted_reasoning": agent._codex_reasoning_replay_enabled}
    transport.convert_messages(messages, **params)
    if replay_state in {"unverified", "current-route"}:
        reasoning = {"type": "reasoning", "id": "rs_test", "encrypted_content": "test-ciphertext"}
        if replay_state == "current-route":
            reasoning.update(_issuer_kind=transport._last_issuer_kind, _issuer_model=transport._last_issuer_model)
        messages[1]["codex_reasoning_items"] = [reasoning]
    original, api_messages, expected = deepcopy(messages), deepcopy(messages), deepcopy(messages)
    expected[1].pop("codex_reasoning_items", None)
    for index in (1, 3):
        expected[index]["codex_message_items"][0].pop("id")
    before = transport.convert_messages(api_messages, **params)
    assert messages == original  # Normal serialization never mutates persisted history.
    assert any(item.get("id") == "msg_final" for item in before)
    err = _api_error(_CROSS_RESOURCE)
    classified = classify_api_error(err, provider=agent.provider, model=agent.model)

    retried, _ = recover_after_classification(
        agent, err, classified, retry, status_code=400,
        error_context=agent._extract_api_error_context(err), messages=messages, api_messages=api_messages,
    )

    assert retried is True
    assert messages == expected
    assert api_messages == expected
    wire = transport.convert_messages(api_messages, **params)
    assistant_items = [item for item in wire if item.get("type") == "message" and item.get("role") == "assistant"]
    assert assistant_items == [expected[index]["codex_message_items"][0] for index in (1, 3)]
    assert not any(item.get("type") == "reasoning" for item in wire)
    calls = [item for item in wire if item.get("type") == "function_call"]
    outputs = [item for item in wire if item.get("type") == "function_call_output"]
    assert [(item["call_id"], item["name"], item["arguments"]) for item in calls] == [("call_test", "browser", "{}")]
    assert [(item["call_id"], item["output"]) for item in outputs] == [("call_test", "Example page")]
    assert agent.refresh_calls == 0
    assert agent._codex_reasoning_replay_enabled is (replay_state != "replay-disabled")
    assert agent._codex_reasoning_replay_rejected is (replay_state == "current-route")
    retried_again, _ = recover_after_classification(
        agent, err, classified, retry, status_code=400,
        error_context=agent._extract_api_error_context(err), messages=messages, api_messages=api_messages,
    )
    assert retried_again is False
    assert messages == api_messages == expected


@pytest.mark.parametrize("message,status,api_mode,has_id,already_retried", [
    pytest.param("Unsupported response format.", 400, "codex_responses", True, False, id="unrelated-error"),
    pytest.param(_CROSS_RESOURCE, 400, "chat_completions", True, False, id="chat-completions"),
    pytest.param("Token expired. Sign in again.", 401, "codex_responses", True, False, id="token-expired"),
    pytest.param(_CROSS_RESOURCE, 400, "codex_responses", False, False, id="no-ids"),
    pytest.param(_CROSS_RESOURCE, 400, "codex_responses", True, True, id="already-retried"),
])
def test_message_id_repair_does_not_consume_unrelated_recovery(message, status, api_mode, has_id, already_retried):
    from agent.route_verdicts import route_visit

    agent, retry = _Agent(), TurnRetryState()
    agent.provider, agent.api_mode = "openai-codex", api_mode
    messages = [{"role": "assistant", "content": "Saved answer.", "codex_message_items": [{
        "type": "message", "role": "assistant", "id": "msg_saved",
        "content": [{"type": "output_text", "text": "Saved answer."}],
    }]}]
    err = _api_error(message, status)
    if status == 401:
        err.body["error"]["code"] = "token_expired"
    if not has_id:
        messages[0]["codex_message_items"][0].pop("id")
    if already_retried:
        retry.invalid_encrypted_content_retry_visit = route_visit(agent)
    api_messages, expected = deepcopy(messages), deepcopy(messages)
    classified = classify_api_error(err, provider=agent.provider, model=agent.model)
    retried, _ = recover_after_classification(
        agent, err, classified, retry, status_code=err.status_code,
        error_context=agent._extract_api_error_context(err), messages=messages, api_messages=api_messages,
    )
    assert retried is False
    assert messages == api_messages == expected
    assert agent._codex_reasoning_replay_rejected is False
    assert agent.refresh_calls == (1 if status == 401 else 0)
