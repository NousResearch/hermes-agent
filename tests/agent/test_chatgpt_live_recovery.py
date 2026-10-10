"""Exercise SIWC recovery through real agent, pool, SDK and auxiliary request paths."""

import asyncio
import json

import httpx
import pytest

from hermes_cli import auth, auth_chatgpt


def _sse(events):
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          text="".join("data: " + json.dumps(event) + "\n\n" for event in events))


def _answer(text="Recovered", status="completed", reason=None, tool=None):
    item = tool or {"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}
    return _sse([
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response." + status, "response": {"id": "resp_test", "status": status,
          "incomplete_details": {"reason": reason} if reason else None, "output": [item],
          "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}},
    ])


@pytest.fixture
def session(monkeypatch, tmp_path):
    from agent import auxiliary_client as aux

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: vendor/model\n"
        "auxiliary:\n  transient_retries: 2\n  compression:\n"
        "    provider: openai-chatgpt\n    model: gpt-5.4\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-main-key")
    row = {"id": "selected", "label": "Personal", "source": "manual:chatgpt",
           "auth_type": "oauth", "priority": 0, "access_token": "access-before",
           "refresh_token": "refresh-before", "expires_at_ms": 4102444800000,
           "chatgpt": {"client_id": "issued-client", "subject": "subject-one",
                       "scopes": [auth_chatgpt.DIRECT_SCOPE]}}
    auth._save_auth_store({"version": 1,
                          "providers": {auth_chatgpt.PROVIDER: {"active_credential_id": row["id"]}},
                          "credential_pool": {auth_chatgpt.PROVIDER: [row]}})
    requests, refreshes = [], []
    replies = []

    def send(request):
        requests.append(request)
        assert request.url.host == "api.openai.com", "Implicit billing fallback dispatched"
        assert request.url.path == "/v1/responses"
        response = replies.pop(0) if replies else _answer()
        return response() if callable(response) else response

    def refresh(entry):
        refreshes.append(entry.refresh_token)
        return {"access_token": "access-after", "refresh_token": "refresh-after",
                "expires_at_ms": 4102444800000, "chatgpt": dict(entry.extra["chatgpt"])}

    monkeypatch.setattr("agent.process_bootstrap.build_keepalive_http_client",
                        lambda *a, **kw: httpx.Client(transport=httpx.MockTransport(send)))
    monkeypatch.setattr(auth_chatgpt, "refresh_credential", refresh)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", dict)
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **kw: 0)
    monkeypatch.setattr(aux, "_TRANSIENT_RETRY_BACKOFF_BASE", 0)
    aux._evict_cached_clients("openai-chatgpt")
    yield row, requests, replies, refreshes
    aux._evict_cached_clients("openai-chatgpt")


def _agent():
    from agent.credential_pool import load_pool
    from run_agent import AIAgent

    pool = load_pool("openai-chatgpt")
    entry = pool.select()
    return AIAgent(model="gpt-5.4", provider="openai-chatgpt", api_mode="codex_responses",
                   api_key=entry.access_token, base_url="https://api.openai.com/v1", credential_pool=pool,
                   quiet_mode=True, max_iterations=4, skip_context_files=True, skip_memory=True,
                   save_trajectories=False)


@pytest.mark.parametrize("failure", ["bare-401", "token_expired", "peer-rotation"])
def test_live_agent_recovers_same_registration_without_restarting(session, failure):
    row, requests, replies, refreshes = session
    agent = _agent()
    assert agent.run_conversation("First turn")["completed"] is True
    if failure == "peer-rotation":
        row.update(access_token="access-after", refresh_token="refresh-after")
        auth.write_credential_pool("openai-chatgpt", [row])
    else:
        replies.append(httpx.Response(401, json={"error": {
            "message": "The authentication token is expired", "type": "invalid_request_error",
            "code": "token_expired" if failure == "token_expired" else None}}))
    result = agent.run_conversation("Second turn")
    assert result.get("completed") is True, result
    assert result["final_response"] == "Recovered"
    assert agent.api_key == "access-after"
    assert refreshes == ([] if failure == "peer-rotation" else ["refresh-before"])
    assert requests[-1].headers["authorization"] == "Bearer access-after"


@pytest.mark.parametrize("reason", ["content_filter", "max_output_tokens"])
def test_live_agent_preserves_explicit_incomplete_semantics(session, reason):
    _, requests, replies, _ = session
    replies.append(_answer("Partial answer", status="incomplete", reason=reason))
    result = _agent().run_conversation("Respond")
    if reason == "content_filter":
        assert len(requests) == 1
        assert "safety" in result["final_response"].lower() or "declined" in result["final_response"].lower()
    else:
        assert result.get("completed") is True
        assert len(requests) == 2
        assert "Partial answer" in requests[-1].content.decode()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("code", ["subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable"])
def test_auxiliary_midstream_transient_is_bounded_without_billing_fallback(session, async_mode, code):
    from agent.auxiliary_client import async_call_llm, call_llm
    from run_agent import _StreamErrorEvent

    _, requests, replies, _ = session
    attempts = 2 if async_mode else 3
    replies.extend(_sse([{"type": "response.failed", "response": {
        "status": "failed", "error": {"code": code, "message": "Temporary plan service outage"}}}])
                   for _ in range(attempts))
    kwargs = {"task": "compression", "messages": [{"role": "user", "content": "Summarize"}]}
    with pytest.raises(_StreamErrorEvent) as raised:
        asyncio.run(async_call_llm(**kwargs)) if async_mode else call_llm(**kwargs)
    assert raised.value.code == code
    assert len(requests) == attempts


@pytest.mark.parametrize("failure", ["bare-401", "peer-rotation", "sse-token-expired"])
@pytest.mark.parametrize("async_mode", [False, True])
def test_auxiliary_recovers_only_the_issuing_registration(session, failure, async_mode):
    from agent.auxiliary_client import async_call_llm, call_llm, resolve_provider_client

    row, requests, replies, refreshes = session
    resolve_provider_client("openai-chatgpt", model="gpt-5.4")
    if failure == "peer-rotation":
        row.update(access_token="access-after", refresh_token="refresh-after")
        auth.write_credential_pool("openai-chatgpt", [row])
    elif failure == "sse-token-expired":
        replies.append(_sse([{"type": "response.failed", "response": {
            "status": "failed", "error": {"code": "token_expired", "message": "Token expired"}}}]))
    else:
        replies.append(httpx.Response(401, json={"error": {"message": "Token expired"}}))
    kwargs = {"task": "compression", "messages": [{"role": "user", "content": "Summarize"}]}
    result = asyncio.run(async_call_llm(**kwargs)) if async_mode else call_llm(**kwargs)
    assert result.choices[0].message.content == "Recovered"
    assert requests[-1].headers["authorization"] == "Bearer access-after"
    assert refreshes == ([] if failure == "peer-rotation" else ["refresh-before"])


@pytest.mark.parametrize("change", ["logout", "account", "scope", "pending"])
def test_live_agent_cannot_refresh_past_a_terminal_account_change(session, change):
    row, requests, _, refreshes = session
    agent = _agent()
    if change == "logout":
        row.update(access_token="", refresh_token="")
    elif change == "account":
        state = auth._load_auth_store()
        state["providers"]["openai-chatgpt"]["active_credential_id"] = "other"
        auth._save_auth_store(state)
    elif change == "scope":
        row["chatgpt"]["scopes"] = []
    else:
        row["chatgpt"]["pending_refresh"] = {"response": {"access_token": "unverified"}}
    auth.write_credential_pool("openai-chatgpt", [row])
    result = agent.run_conversation("Do not dispatch")
    assert result.get("failed") is True
    assert result["failure_reason"] == "provider_policy_blocked"
    assert requests == refreshes == []


def test_live_agent_repeated_unauthorized_is_bounded(session):
    _, requests, replies, refreshes = session
    replies.extend(httpx.Response(401, json={"error": {"message": "Token rejected"}}) for _ in range(10))
    result = _agent().run_conversation("Respond")
    assert result.get("failed") is True
    assert 1 <= len(refreshes) <= 3
    assert len(requests) <= 4


@pytest.mark.parametrize("tool_status", [None, "completed", "in_progress"])
def test_auxiliary_incomplete_preserves_partial_text_and_tool_completion(session, tool_status):
    from agent.auxiliary_client import call_llm

    _, requests, replies, _ = session
    tool = ({"type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "read_file",
             "arguments": '{"path":"note.txt"}', "status": tool_status} if tool_status else None)
    replies.append(_answer("Partial checkpoint", status="incomplete", reason="max_output_tokens", tool=tool))
    result = call_llm(task="compression", messages=[{"role": "user", "content": "Summarize"}])
    assert len(requests) == 1
    if tool_status == "completed":
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls[0].function.name == "read_file"
    elif tool_status is None:
        assert result.choices[0].finish_reason == "length"
        assert result.choices[0].message.content == "Partial checkpoint"
    else:
        assert result.choices[0].finish_reason == "length"
        assert not result.choices[0].message.tool_calls


def test_live_agent_unknown_incomplete_is_not_certified_complete(session):
    _, requests, replies, _ = session
    replies.extend(_answer("Unfinished", status="incomplete", reason="unknown_reason") for _ in range(5))
    result = _agent().run_conversation("Respond")
    assert result.get("completed") is not True
    assert len(requests) <= 4


@pytest.mark.parametrize("auxiliary", [False, True])
def test_account_change_during_unauthorized_request_cannot_refresh_or_switch(session, auxiliary):
    from agent.auxiliary_client import call_llm
    from hermes_cli.auth_constants import AuthError

    row, requests, replies, refreshes = session

    def change_account():
        other = {**row, "id": "another", "access_token": "other-account", "refresh_token": "other-refresh"}
        state = auth._load_auth_store()
        state["providers"]["openai-chatgpt"]["active_credential_id"] = "another"
        state["credential_pool"]["openai-chatgpt"].append(other)
        auth._save_auth_store(state)
        return httpx.Response(401, json={"error": {"message": "Token rejected"}})

    replies.append(change_account)
    if auxiliary:
        with pytest.raises(AuthError) as raised:
            call_llm(task="compression", messages=[{"role": "user", "content": "Summarize"}])
        assert raised.value.code == "chatgpt_session_changed"
    else:
        result = _agent().run_conversation("Respond")
        assert result.get("failed") is True
        assert result["failure_reason"] == "provider_policy_blocked"
    assert len(requests) == 1 and not refreshes


@pytest.mark.parametrize("async_mode", [False, True])
def test_auxiliary_auth_retry_cannot_recover_a_narrowed_terminal_plan_error(session, async_mode, monkeypatch):
    from agent import auxiliary_client as aux
    from agent.auxiliary_client import async_call_llm, call_llm
    from openai import RateLimitError

    _, requests, replies, refreshes = session
    recoveries = []
    recover = aux._recover_provider_pool

    def observe_recovery(*args, **kwargs):
        recoveries.append(args[1])
        return recover(*args, **kwargs)

    monkeypatch.setattr(aux, "_recover_provider_pool", observe_recovery)
    replies.extend([
        httpx.Response(401, json={"error": {"message": "Token rejected"}}),
        httpx.Response(429, json={"error": {"code": "subscription_sharing_usage_limit_exceeded",
                                          "message": "Usage limit exceeded"}}),
    ])
    kwargs = {"task": "compression", "messages": [{"role": "user", "content": "Summarize"}]}
    with pytest.raises(RateLimitError) as raised:
        asyncio.run(async_call_llm(**kwargs)) if async_mode else call_llm(**kwargs)
    assert raised.value.code == "subscription_sharing_usage_limit_exceeded"
    assert len(requests) == 2 and refreshes == ["refresh-before"]
    assert not recoveries
    row = auth.read_credential_pool("openai-chatgpt")[0]
    assert row.get("last_status") in {None, "ok"}
    assert not row.get("cooldown_until")
