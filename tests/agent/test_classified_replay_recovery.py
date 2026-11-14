"""A profile-classified replay 401 repairs only attributable items on the actual SDK wire."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import APIError, AuthenticationError, OpenAI
import pytest

from agent.error_classifier import ClassifiedError, FailoverReason
from agent.transports.codex import ResponsesApiTransport
from agent.turn_recovery import recover_after_classification
from agent.turn_retry_state import TurnRetryState

BASE_URL = "https://replay.example/v1"
MODEL = "test-model"


def _item(blob, **metadata):
    return {"type": "reasoning", "encrypted_content": blob, **metadata}


def _history(items):
    return [{"role": "user", "content": "question"},
            {"role": "assistant", "content": "plaintext stays", "codex_reasoning_items": items},
            {"role": "user", "content": "continue"}]


def _agent(transport):
    return SimpleNamespace(
        provider="replay-test", api_mode="codex_responses", model=MODEL, base_url=BASE_URL,
        log_prefix="", _codex_reasoning_replay_enabled=True, _codex_reasoning_replay_rejected=False,
        _get_transport=lambda: transport, _buffer_vprint=Mock(), _vprint=Mock(),
        _extract_api_error_context=lambda error: {},
        _recover_with_credential_pool=Mock(return_value=(True, False)),
        _try_refresh_provider_runtime_credentials=Mock(return_value=True),
    )


def _wire_error(transport, request_history, *, replacement=None, base_url=BASE_URL):
    """Exercise the builder, FINAL preflight and SDK serialization, not imagined input dicts."""
    bodies = []

    def reject(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(401, json={"error": {"message": "input item does not belong to this connection"}})

    kwargs = transport.build_kwargs(model=MODEL, messages=request_history, tools=[], base_url=base_url)
    if replacement is not None:
        kwargs["extra_body"] = {"input": replacement}
    kwargs = transport.preflight_kwargs(kwargs, drop_replay_item_ids=True)
    with OpenAI(api_key="offline", base_url=base_url, max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(reject))) as client:
        with pytest.raises(AuthenticationError) as caught:
            client.responses.create(**kwargs)
    assert len(bodies) == 1
    return caught.value, bodies[0]


def _recover(agent, error, retry, history, request_history, reason=FailoverReason.invalid_encrypted_content):
    classified = ClassifiedError(reason=reason, status_code=401, retryable=False)
    return recover_after_classification(
        agent, error, classified, retry, status_code=getattr(error, "status_code", None), error_context={},
        messages=history, api_messages=request_history,
    )


@pytest.mark.parametrize("error_form", ["http-status", "request-only-sdk-error"])
def test_classified_replay_401_repairs_sent_unverified_before_auth_without_disabling_replay(error_form):
    transport = ResponsesApiTransport()
    stamped = {"_issuer_kind": "other:" + BASE_URL, "_issuer_model": MODEL}
    history = _history([_item("rejected", id="rs_old"), _item("valid", **stamped),
                        _item("foreign", _issuer_kind="other:https://elsewhere.example/v1")])
    history[1]["tool_calls"] = [{"id": "call_offline", "type": "function",
                                 "function": {"name": "lookup", "arguments": "{}"}}]
    history.insert(2, {"role": "tool", "tool_call_id": "call_offline", "content": "tool result stays"})
    history[1]["codex_reasoning_items"][0]["summary"] = [{"type": "summary_text", "text": "display"}]
    request_history = deepcopy(history)
    history[1]["codex_reasoning_items"].append(_item("unsent-legacy"))
    original = deepcopy(history)
    request_history[1]["codex_reasoning_items"][0].update(summary=[], status="completed")
    error, body = _wire_error(transport, request_history)
    assert [i["encrypted_content"] for i in body["input"] if i.get("type") == "reasoning"] == ["rejected", "valid"]
    # Preflight added summary=[] and stripped id/provenance: raw dict equality would miss this.
    assert body["input"][1] == _item("rejected", summary=[])
    if error_form == "request-only-sdk-error":
        error = APIError("input item does not belong to this connection", request=error.request,
                         body={"error": {"code": 401, "message": "input item does not belong to this connection"}})
    agent, retry = _agent(transport), TurnRetryState()

    assert _recover(agent, error, retry, history, request_history) == (True, False)
    assert history[1]["codex_reasoning_items"] == original[1]["codex_reasoning_items"][1:]
    expected_request = deepcopy(history)
    expected_request[1]["codex_reasoning_items"].pop()  # unsent source-only legacy item survives
    assert request_history == expected_request
    _, repaired_body = _wire_error(transport, request_history)
    assert [i["encrypted_content"] for i in repaired_body["input"] if i.get("type") == "reasoning"] == ["valid"]
    assert [i for i in repaired_body["input"] if i.get("type") in ("function_call", "function_call_output")] == [
        i for i in body["input"] if i.get("type") in ("function_call", "function_call_output")]
    assert any(i.get("output") == "tool result stays" for i in repaired_body["input"])
    assert [m["content"] for m in history] == [m["content"] for m in original]
    assert agent._codex_reasoning_replay_enabled is True
    agent._recover_with_credential_pool.assert_not_called()
    agent._try_refresh_provider_runtime_credentials.assert_not_called()
    # Same visit has spent its repair even though another valid opaque item remains.
    assert _recover(agent, error, retry, history, request_history) == (False, False)
    agent._recover_with_credential_pool.assert_not_called()


@pytest.mark.parametrize("case", [
    "wire-duplicate", "request-duplicate", "source-duplicate", "source-omitted",
    "wire-omitted", "request-omitted", "foreign-source", "unknown-type", "empty",
    "nonstring", "missing-body", "missing-request", "ordinary-auth", "no-opaque",
])
def test_ambiguous_unsent_or_unclassified_replay_is_never_swept(case):
    transport = ResponsesApiTransport()
    history = _history([_item("opaque", id="rs_one")])
    request_history = deepcopy(history)
    replacements = {
        "wire-duplicate": [_item("opaque", summary=[]), _item("opaque", summary=[{"type": "summary_text", "text": "different"}])],
        "wire-omitted": [{"role": "user", "content": "override replaces input"}],
    }
    mutations = {
        "request-duplicate": lambda: request_history[1]["codex_reasoning_items"].append(_item("opaque", id="rs_two", summary=[])),
        "source-duplicate": lambda: history[1]["codex_reasoning_items"].append(_item("opaque", id="rs_two", summary=[])),
        "source-omitted": lambda: history[1].pop("codex_reasoning_items"),
        "request-omitted": lambda: request_history[1].pop("codex_reasoning_items"),
        "no-opaque": lambda: (history[1].pop("codex_reasoning_items"), request_history[1].pop("codex_reasoning_items")),
        "foreign-source": lambda: history[1]["codex_reasoning_items"][0].update(_issuer_kind="other:https://foreign.example/v1"),
        "unknown-type": lambda: history[1]["codex_reasoning_items"][0].update(type="unknown"),
        "empty": lambda: history[1]["codex_reasoning_items"][0].update(encrypted_content=""),
        "nonstring": lambda: history[1]["codex_reasoning_items"][0].update(encrypted_content=42),
    }
    mutations.get(case, lambda: None)()
    replacement = replacements.get(case)
    before, request_before = deepcopy(history), deepcopy(request_history)
    error, body = _wire_error(transport, request_history, replacement=replacement)
    if replacement is not None:
        assert body["input"] == replacement
    if case == "missing-body":
        error = SimpleNamespace(status_code=401)
    if case == "missing-request":
        error = SimpleNamespace(status_code=401, response=httpx.Response(401))
    agent = _agent(transport)
    reason = FailoverReason.auth if case == "ordinary-auth" else FailoverReason.invalid_encrypted_content
    assert _recover(agent, error, TurnRetryState(), history, request_history, reason) == (
        (True, True) if case == "ordinary-auth" else (False, False))
    assert history == before
    assert request_history == request_before
    if case == "ordinary-auth":
        agent._recover_with_credential_pool.assert_called_once()
    else:
        agent._recover_with_credential_pool.assert_not_called()
    assert agent._codex_reasoning_replay_enabled is True


@pytest.mark.parametrize("raw_status", [401, None])
def test_unrepairable_replay_is_terminal_even_with_a_broad_stale_credential_predicate(monkeypatch, raw_status):
    from agent import conversation_loop, turn_api_error
    from agent.turn_api_error import settle_unrecovered_error

    agent = _agent(ResponsesApiTransport())
    agent._has_pending_fallback = Mock(return_value=True)
    agent._try_activate_fallback = Mock(return_value=True)
    stale = Mock(return_value=True)
    profile = SimpleNamespace(is_stale_credential_error=stale, display_name="test",
                              credential_refresh_notice=lambda **kwargs: "refreshed")
    monkeypatch.setattr(conversation_loop, "_credential_refresh_profile", lambda agent: profile)
    monkeypatch.setattr("agent.fallback_cooldown._mark_entitlement_rejected_model", lambda *args: None)
    monkeypatch.setattr(turn_api_error, "settle_delivered_partial", lambda *args: None)
    monkeypatch.setattr(turn_api_error, "nonretryable_client_error_result", lambda *args, **kw: {"failed": True})
    verdict = settle_unrecovered_error(
        agent, api_error=RuntimeError("replay rejected"),
        classified=ClassifiedError(FailoverReason.invalid_encrypted_content, status_code=401, retryable=False),
        _retry=TurnRetryState(), status_code=raw_status, error_msg="replay rejected", is_context_length_error=False,
        is_rate_limited=False, _is_zai_coding_overload=False, _provider=agent.provider,
        _base=BASE_URL, _model=MODEL, messages=[], api_messages=[], api_kwargs={},
        active_system_prompt="system", conversation_history=[], approx_tokens=10, retry_count=1,
        max_retries=3, compression_attempts=0, api_call_count=1,
    )
    assert verdict.action == "return"
    stale.assert_not_called()
    agent._try_refresh_provider_runtime_credentials.assert_not_called()
    agent._try_activate_fallback.assert_not_called()
