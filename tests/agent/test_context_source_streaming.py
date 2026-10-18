"""Physical streaming SDK entrypoints preserve the same egress boundary."""

import json

import httpx
import pytest
from openai import OpenAI

from tests.agent.test_run_agent import agent as _agent_fixture

agent = _agent_fixture


@pytest.mark.parametrize("wire", ["chat", "anthropic"])
def test_protected_stream_denies_secret_before_physical_sdk_io(agent, tmp_path, monkeypatch, wire):
    from agent.chat_completion_helpers import _StreamingCall
    from agent.llm_egress_firewall import EgressBlocked
    captured = []
    def io(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
    agent.provider = "nous" if wire == "chat" else "anthropic"
    agent.base_url = ("https://inference-api.nousresearch.com/v1" if wire == "chat"
                      else "https://api.anthropic.com")
    agent._base_url_lower = agent.base_url.lower()
    agent.api_mode = "chat_completions" if wire == "chat" else "anthropic_messages"
    agent.session_id = "session-1"
    agent._current_turn_id = "turn-1"
    agent._current_api_request_id = "turn-1:api:1"
    agent._llm_egress_state_dir = tmp_path / "egress"
    kwargs = {"model": agent.model, "messages": [
        {"role": "user", "content": "token=super-secret-value"},
    ]}
    if wire == "chat":
        kwargs["stream"] = True
        with OpenAI(api_key="synthetic-key", base_url=agent.base_url,
                    http_client=httpx.Client(transport=httpx.MockTransport(io))) as sdk:
            agent._create_request_openai_client = lambda **_k: sdk
            call = _StreamingCall(agent, kwargs, None)
            with pytest.raises(EgressBlocked):
                call._open_chat_stream(dict(kwargs))
    else:
        from anthropic import Anthropic
        kwargs["max_tokens"] = 16
        def reach_open(request, opener, **_k):
            opener(request)
            raise AssertionError("protected streaming SDK boundary was reached before denial")
        monkeypatch.setattr("agent.relay_llm.stream", reach_open)
        with Anthropic(api_key="synthetic-key", base_url=agent.base_url,
                       http_client=httpx.Client(transport=httpx.MockTransport(io))) as sdk:
            call = _StreamingCall(agent, kwargs, None)
            with pytest.raises(EgressBlocked):
                call._call_anthropic(sdk)
    assert captured == []


@pytest.mark.parametrize("local", [False, True])
def test_chat_stream_source_wire_strips_internal_sidecar_on_a_copy(
    agent, tmp_path, local,
):
    from agent.chat_completion_helpers import _StreamingCall
    from tests.agent.test_context_source_lifecycle import _prepare
    _, message, turn = _prepare(agent, tmp_path)
    if local:
        agent.base_url = "http://127.0.0.1:11434/v1"
        agent._base_url_lower = agent.base_url.lower()
    agent._current_api_request_id = f"{turn}:api:1"
    kwargs = agent._build_api_kwargs([{"role": "user", "content": message}], tools_for_api=[])
    kwargs.update(stream=True, _hermes_source_provenance=[])
    captured = []
    def io(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=b"data: [DONE]\n\n")
    with OpenAI(api_key="synthetic-key", base_url=agent.base_url,
                http_client=httpx.Client(transport=httpx.MockTransport(io))) as sdk:
        agent._create_request_openai_client = lambda **_k: sdk
        call = _StreamingCall(agent, kwargs, None)
        stream = call._open_chat_stream(dict(kwargs))
        stream.close()
    assert len(captured) == 1 and "_hermes_source_provenance" not in captured[0]
    assert kwargs["_hermes_source_provenance"] == []
    receipts = tmp_path / "egress/llm-egress-receipts.jsonl"
    if local:
        assert not receipts.exists()
    else:
        assert json.loads(receipts.read_text().splitlines()[-1])["source_grant_count"] == 1


def test_final_firewall_uses_the_same_lf_bounded_raw_slice(agent, tmp_path):
    from agent.context_references import preprocess_context_references
    from agent.source_provenance import admit_agent_context_sources, provenance_kwargs_for_agent
    from tests.agent.test_context_source_lifecycle import _dispatch, _sdk
    source = tmp_path / "lf.py"
    source.write_bytes(b"first\rsecond\nnext\n" + b"tail\n" * 100000)
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    agent.api_mode = "chat_completions"
    agent.session_id = "parent"
    agent._llm_egress_state_dir = tmp_path / "egress"
    expanded = preprocess_context_references(
        "Inspect @file:lf.py:1-1", cwd=tmp_path, allowed_root=tmp_path,
        context_length=240_000, **provenance_kwargs_for_agent(agent, establish_turn=True),
    )
    turn = agent._source_provenance_pending_turn_id
    agent._current_turn_id = agent._relay_pending_turn_id = turn
    agent._source_provenance_pending_turn_id = None
    admit_agent_context_sources(agent, turn_id=turn, prepared_session_id="parent")
    captured = []
    with _sdk(agent, captured) as sdk:
        _dispatch(agent, sdk, expanded.message, turn, 1)
    assert len(captured) == 1
    assert "first\rsecond\n" in captured[0]["messages"][0]["content"]


@pytest.mark.parametrize("repeat", [False, True])
def test_local_fallback_walk_advances_and_stops_repeated_lanes(monkeypatch, repeat):
    from types import SimpleNamespace
    from agent import auxiliary_egress_recovery as recovery
    clients = [SimpleNamespace(base_url="http://127.0.0.1:18434/v1"),
               SimpleNamespace(base_url="http://127.0.0.1:18435/v1")]
    selections = iter([(clients[0], "first", "custom"),
                       (clients[0] if repeat else clients[1], "first" if repeat else "second", "custom"),
                       (None, None, "")])
    monkeypatch.setattr(recovery, "try_configured_fallback_chain", lambda *a, **k: next(selections))
    monkeypatch.setattr(recovery, "try_main_agent_model_fallback", lambda *a, **k: (None, None, ""))
    route = SimpleNamespace(task="summary", resolved_provider="nous", final_model="remote", route_info={})
    walk = recovery.local_fallback_steps(route, lambda kind, payload: payload)
    assert next(walk)[1] == "first"
    if repeat:
        with pytest.raises(StopIteration):
            walk.send(None)
    else:
        assert walk.send(None)[1] == "second"
        with pytest.raises(StopIteration) as done:
            walk.send("healthy")
        assert done.value.value == "healthy"


@pytest.mark.parametrize("secret", [False, True])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("provider", ["openai-codex", "nous"])
def test_codex_auxiliary_adapter_binds_receipt_to_final_responses_wire(tmp_path, monkeypatch, secret, async_call, provider):
    from hashlib import sha256
    from agent import auxiliary_client as auxiliary
    from agent.auxiliary_egress_recovery import authorize_auxiliary_request
    from agent.llm_egress_firewall import EgressBlocked
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    captured = []
    def io(request):
        captured.append(json.loads(request.content))
        final = {"id": "response-1", "object": "response", "model": "gpt-5-codex",
                 "status": "completed", "output": [{"type": "message", "id": "message-1",
                 "role": "assistant", "status": "completed", "content": [
                 {"type": "output_text", "text": "ok", "annotations": []}]}],
                 "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=("data: " + json.dumps({"type": "response.completed", "response": final}) + "\n\n").encode())
    base = ("https://chatgpt.com/backend-api/codex" if provider == "openai-codex"
            else "https://inference-api.nousresearch.com/v1")
    kwargs = {"model": "gpt-5-codex", "messages": [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "token=super-secret-value" if secret else "hello"}], "stream": True}
    with OpenAI(api_key="synthetic-key", base_url=base,
                http_client=httpx.Client(transport=httpx.MockTransport(io))) as sdk:
        client = auxiliary.CodexAuxiliaryClient(sdk, "gpt-5-codex")
        if async_call:
            client = auxiliary.AsyncCodexAuxiliaryClient(client)
        with auxiliary.scoped_runtime_main({"provider": provider, "model": "gpt-5-codex",
                                            "session_id": "session-1", "turn_id": "turn-1", "base_url": base}):
            def dispatch():
                result = authorize_auxiliary_request(client, kwargs,
                    lambda request: client.chat.completions.create(**request),
                    provider=provider, api_mode="codex_responses", metadata=None)
                if async_call:
                    import asyncio
                    return asyncio.run(result)
                return result
            if secret:
                with pytest.raises(EgressBlocked):
                    dispatch()
                assert not captured
            else:
                dispatch()
                assert len(captured) == 1 and "input" in captured[0] and "messages" not in captured[0]
                receipts = (tmp_path / "egress/llm-egress-receipts.jsonl").read_text().splitlines()
                assert len(receipts) == 1
                receipt = json.loads(receipts[0])
                wire = json.dumps(captured[0], ensure_ascii=False, allow_nan=False,
                                  separators=(",", ":"), sort_keys=True).encode()
                assert receipt["payload_sha256"] == sha256(wire).hexdigest()
    assert "_hermes_aux_request_provider" not in kwargs
