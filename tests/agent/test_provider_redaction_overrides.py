"""The SDK's effective payload must pass the same provider-copy gate."""

import copy

import pytest

from agent.provider_redaction import redact_provider_api_kwargs
from agent.secret_scope import reset_secret_scope, set_secret_scope


@pytest.mark.parametrize("extra_body", [False, True])
@pytest.mark.parametrize("field", ["input", "messages"])
def test_effective_native_input_is_masked_without_mutating_request(monkeypatch, extra_body, field):
    secret = "sdk-override-secret-77487"
    payload = {field: secret if field == "input" else [{"role": "user", "content": secret}], "instructions": f"Use {secret}"}
    request = {"model": "fixture", "extra_body": payload} if extra_body else {"model": "fixture", **payload}
    original = copy.deepcopy(request)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"ACTIVE_TOKEN": secret})
    try:
        redacted = redact_provider_api_kwargs(request)
    finally:
        reset_secret_scope(token)
    effective = {**redacted, **redacted.get("extra_body", {})}
    assert secret not in str(effective[field])
    assert secret not in effective["instructions"]
    assert request == original


def test_auxiliary_override_and_codex_conversion_are_gated(monkeypatch):
    from types import SimpleNamespace
    from agent import auxiliary_client

    content = [{"type": "text", "text": "ordinary fragment"}]
    secret = str(content)
    captured = {}
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(auxiliary_client, "_call_llm_impl", lambda **kwargs: captured.update(kwargs))
    token = set_secret_scope({"ACTIVE_TOKEN": secret})
    try:
        auxiliary_client.call_llm(messages=[], extra_body={"input": secret})
        adapter = auxiliary_client._CodexCompletionsAdapter(
            SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex"), "fixture"
        )
        native, _model, _timeout = adapter._build_responses_kwargs(
            {"messages": [{"role": "system", "content": content}]}
        )
    finally:
        reset_secret_scope(token)
    assert secret not in captured["extra_body"]["input"]
    assert secret not in native["instructions"]
    assert content == [{"type": "text", "text": "ordinary fragment"}]


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_relay_replacement_is_masked_at_the_provider_callback(monkeypatch, mode):
    import asyncio
    from types import SimpleNamespace
    from agent import auxiliary_client, relay_llm

    secret = "relay-replacement-secret-77487"
    replacement = {"messages": [{"role": "user", "content": secret}], "extra_body": {"input": secret}}
    captured = []
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(auxiliary_client, "_relay_auxiliary_metadata", lambda **kw: ("fixture", "fixture", {}))

    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(choices=[])

    async def async_create(**kwargs):
        return create(**kwargs)

    async def relay_async(_request, callback, **kwargs):
        return await callback(replacement)

    monkeypatch.setattr(relay_llm, "execute_current", lambda _request, callback, **kw: callback(replacement))
    monkeypatch.setattr(relay_llm, "execute_current_async", relay_async)
    monkeypatch.setattr(relay_llm, "stream_current", lambda _request, callback, **kw: callback(replacement))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=async_create if mode == "async" else create)))
    token = set_secret_scope({"RELAY_TOKEN": secret})
    try:
        if mode == "sync":
            auxiliary_client._relay_sync_completion(client, {"messages": []})
        elif mode == "async":
            asyncio.run(auxiliary_client._relay_async_completion(client, {"messages": []}))
        else:
            auxiliary_client._relay_sync_stream(client, {"messages": []})
    finally:
        reset_secret_scope(token)
    assert len(captured) == 1
    assert secret not in str(captured[0])
    assert replacement["messages"][0]["content"] == secret
    assert replacement["extra_body"]["input"] == secret
