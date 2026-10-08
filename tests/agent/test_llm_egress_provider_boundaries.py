from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.chat_completion_helpers import (
    _dispatch_nonstreaming_api_request,
    _dispatch_provider_request,
)
from agent.llm_egress_firewall import EgressBlocked


def _agent(tmp_path, *, provider="nous", api_mode="chat_completions"):
    if provider == "openai-codex":
        base_url = "https://chatgpt.com/backend-api/codex"
    elif provider == "anthropic":
        base_url = "https://api.anthropic.com/v1"
    else:
        base_url = "https://inference-api.nousresearch.com/v1"
    return SimpleNamespace(
        provider=provider,
        model="test-model",
        base_url=base_url,
        api_mode=api_mode,
        session_id="session-1",
        _current_turn_id="turn-1",
        _current_api_request_id="request-1",
        _llm_egress_policy_digest=sha256(b"policy").hexdigest(),
        _llm_egress_state_dir=tmp_path,
    )


@pytest.mark.parametrize(
    "provider", ["openai-codex", "nous", "nous-portal", "nousresearch", "anthropic"]
)
def test_protected_main_provider_denies_before_callback(tmp_path, provider):
    agent = _agent(tmp_path, provider=provider)
    callback = MagicMock()

    with pytest.raises(EgressBlocked):
        _dispatch_provider_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "token=super-secret-value"}],
            },
            callback,
        )

    callback.assert_not_called()


def test_local_main_provider_keeps_zero_firewall_overhead(tmp_path):
    agent = _agent(tmp_path, provider="ollama-launch")
    agent.base_url = "http://127.0.0.1:11434/v1"
    request = {"messages": [{"role": "user", "content": "/Users/private/file.py"}]}
    callback = MagicMock(return_value="local")

    assert _dispatch_provider_request(agent, request, callback) == "local"
    callback.assert_called_once_with(request)


def test_loopback_provider_receives_only_sdk_fields(tmp_path):
    agent = _agent(tmp_path)
    agent.base_url = "http://127.0.0.1:11434/v1"
    request = {
        "messages": [{"role": "user", "content": "/Users/private/file.py"}],
        "_hermes_source_provenance": [{"message_index": 0}],
        "timeout": 2,
    }
    callback = MagicMock(return_value="local")
    assert _dispatch_provider_request(agent, request, callback) == "local"
    callback.assert_called_once_with(
        {"messages": request["messages"], "timeout": 2}
    )
    assert "_hermes_source_provenance" in request


@pytest.mark.parametrize("local_auxiliary", [False, True])
def test_auxiliary_sdk_url_binds_its_own_route(
    tmp_path, monkeypatch, local_auxiliary
):
    from openai import OpenAI
    from agent import auxiliary_client

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = (
        "http://127.0.0.1:11434/v1" if local_auxiliary
        else "https://inference-api.nousresearch.com/v1"
    )
    request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "token=synthetic-secret"}],
    }
    with OpenAI(api_key="synthetic-key", base_url=endpoint) as client:
        callback = MagicMock(return_value="local")
        monkeypatch.setattr(client.chat.completions, "create", callback)
        with auxiliary_client.scoped_runtime_main(
            {"provider": "nous", "base_url": "http://127.0.0.1:11434/v1"}
        ):
            if local_auxiliary:
                assert auxiliary_client._relay_sync_completion(
                    client, request, provider="nous"
                ) == "local"
                callback.assert_called_once_with(**request)
            else:
                with pytest.raises(EgressBlocked):
                    auxiliary_client._relay_sync_completion(
                        client, request, provider="nous"
                    )
                callback.assert_not_called()


@pytest.mark.parametrize("local_candidate", [False, True])
def test_policy_fallback_does_not_resolve_remote_candidates(
    tmp_path, monkeypatch, local_candidate
):
    import yaml
    from agent import auxiliary_client

    chain = [{
        "provider": "custom", "model": "remote-model",
        "base_url": "https://remote.example.test/v1",
    }]
    if local_candidate:
        chain.append({
            "provider": "custom", "model": "local-model",
            "base_url": "http://127.0.0.1:11434/v1",
        })
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(
        {"auxiliary": {"title_generation": {"fallback_chain": chain}}}
    ))
    resolved = []
    main_reads = []

    def resolve(provider, *, model, explicit_base_url, **kwargs):
        assert explicit_base_url == "http://127.0.0.1:11434/v1"
        resolved.append((provider, model, explicit_base_url))
        return SimpleNamespace(base_url=explicit_base_url), model

    monkeypatch.setattr(auxiliary_client, "resolve_provider_client", resolve)
    monkeypatch.setattr(auxiliary_client, "_read_main_provider", lambda: main_reads.append(True) or "nous")
    monkeypatch.setattr(auxiliary_client, "_read_main_model", lambda: "main-remote")
    monkeypatch.setattr(auxiliary_client, "_is_provider_unhealthy", lambda provider: False)
    with pytest.raises(EgressBlocked) as blocked:
        _dispatch_provider_request(
            _agent(tmp_path),
            {"messages": [{"role": "user", "content": "token=synthetic-secret"}]},
            lambda request: pytest.fail("unsafe primary payload was dispatched"),
        )
    ladder = auxiliary_client._aux_recovery_ladder(
        blocked.value, client=SimpleNamespace(),
        kwargs={"messages": [{"role": "user", "content": "token=synthetic-secret"}]},
        task="title_generation", async_mode=False, base_info="",
        resolved_provider="nous", resolved_model="unsafe-model",
        resolved_base_url=None, resolved_api_key=None, resolved_api_mode=None,
        final_model="unsafe-model", max_tokens=None, main_runtime=None, route_info={},
    )
    if local_candidate:
        step = next(ladder)
        assert step.kind == "fallback"
        assert resolved == [("custom", "local-model", "http://127.0.0.1:11434/v1")]
        with pytest.raises(StopIteration) as finished:
            ladder.send("recovered")
        assert finished.value.value == "recovered"
        assert main_reads == []
    else:
        with pytest.raises(StopIteration) as finished:
            next(ladder)
        assert finished.value.value is auxiliary_client._RERAISE_ORIGINAL
        assert resolved == []
        assert main_reads == [True]


@pytest.mark.parametrize(
    "provider", ["openai-codex", "nous", "nous-portal", "nousresearch", "anthropic"]
)
@pytest.mark.parametrize("protected_flag", [None, "0", "1"])
def test_protected_provider_denies_ungranted_terminal_output(
    tmp_path, monkeypatch, provider, protected_flag
):
    if protected_flag is None:
        monkeypatch.delenv("HERMES_KANBAN_PROTECTED_REMOTE", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", protected_flag)
    request = {
        "model": "test-model",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_terminal123",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_terminal123",
                "content": "def calculate_total(items):\n    return sum(items)\n",
            },
        ],
    }
    callback = MagicMock()
    with pytest.raises(EgressBlocked) as exc_info:
        _dispatch_provider_request(
            _agent(tmp_path, provider=provider), request, callback
        )
    assert "untrusted_provenance" in exc_info.value.decision.reason_codes
    callback.assert_not_called()

    local = _agent(tmp_path, provider="ollama-launch")
    local.base_url = "http://127.0.0.1:11434/v1"
    callback.return_value = "local"
    assert _dispatch_provider_request(local, request, callback) == "local"
    callback.assert_called_once_with(request)


@pytest.mark.parametrize("surface", ["message", "extra_headers", "extra_query"])
@pytest.mark.parametrize(
    "text,denied",
    [
        ("token=super-secret-value", True),
        ("TOKEN=short", True),
        ("export TOKEN=short", True),
        ("token: 'short'", True),
        ("token=os.getenv('EXTERNAL_VALUE', 'super-secret-value')", True),
        ("The prose discusses token=CPU as a technical example.", False),
        ("token=os.getenv('EXTERNAL_VALUE')", False),
        ("TOKEN=process.env.EXTERNAL_VALUE", False),
        ("token=128", True),
        ("max_tokens=128", False),
    ],
)
def test_protected_boundary_credential_assignments_and_reference_controls(
    tmp_path, surface, text, denied
):
    request = {"messages": [{"role": "user", "content": "Review carefully."}]}
    if surface == "message":
        request["messages"][0]["content"] = text
    else:
        request[surface] = {"x-fixture": text}
    callback = MagicMock(return_value="allowed")
    if denied:
        with pytest.raises(EgressBlocked):
            _dispatch_provider_request(_agent(tmp_path), request, callback)
        callback.assert_not_called()
    else:
        assert (
            _dispatch_provider_request(_agent(tmp_path), request, callback) == "allowed"
        )
        callback.assert_called_once_with(request)


def test_nous_chat_completions_entrypoint_uses_firewall(tmp_path):
    agent = _agent(tmp_path)
    client = MagicMock()

    with pytest.raises(EgressBlocked):
        _dispatch_nonstreaming_api_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "/Users/private/file.py"}],
            },
            make_client=lambda *_args, **_kwargs: client,
        )

    client.chat.completions.create.assert_not_called()


def test_nous_anthropic_entrypoint_uses_firewall(tmp_path):
    agent = _agent(tmp_path, api_mode="anthropic_messages")
    agent._anthropic_messages_create = MagicMock()
    client = MagicMock()

    with pytest.raises(EgressBlocked):
        _dispatch_nonstreaming_api_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "/Users/private/file.py"}],
            },
            make_client=lambda *_args, **_kwargs: client,
        )

    agent._anthropic_messages_create.assert_not_called()
