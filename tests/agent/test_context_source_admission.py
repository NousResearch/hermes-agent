"""Context authority follows real admission and both consuming SDK requests."""

import json

import httpx
import pytest
from openai import OpenAI

from tests.agent.test_run_agent import agent as _agent_fixture

agent = _agent_fixture


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("lines", [8, 1600])
def test_context_slice_survives_durable_admission_and_tool_loop(
    agent, tmp_path, monkeypatch, rotate, lines,
):
    from agent.context_references import preprocess_context_references
    from agent.llm_egress_runtime import dispatch_provider_request
    from agent.source_provenance import provenance_kwargs_for_agent
    from tests.agent.test_cross_process_turn_lease import _DB

    source = tmp_path / "source.py"
    raw = "my_value = other_value + 123\n" * lines
    source.write_text(raw)
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    agent._base_url_lower = agent.base_url.lower()
    agent.api_mode = "chat_completions"
    agent.session_id = "stale-parent"
    agent.platform = "desktop"
    agent._llm_egress_state_dir = tmp_path / "egress"
    db = _DB()
    if rotate:
        def acquire(session_id, holder, **kwargs):
            kwargs["on_contended"]()
            return True
        db.acquire_session_turn_lease = acquire
    agent._session_db = db
    agent._persist_disabled = False
    expanded = preprocess_context_references(
        f"Inspect @file:source.py:1-{lines}", cwd=tmp_path, allowed_root=tmp_path,
        context_length=240_000, **provenance_kwargs_for_agent(agent, establish_turn=True),
    )
    assert not expanded.blocked and raw in expanded.message
    pending = agent._source_provenance_pending_turn_id
    captured = []

    def sdk_io(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "context-completion", "object": "chat.completion", "created": 0,
            "model": agent.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
        })

    def consume(actual, message, *_args, **_kwargs):
        assert actual.session_id == ("compressed-tip" if rotate else "stale-parent")
        actual._current_turn_id = actual._relay_pending_turn_id
        assert actual._current_turn_id == pending
        with OpenAI(api_key="synthetic-key", base_url=actual.base_url,
                    http_client=httpx.Client(transport=httpx.MockTransport(sdk_io))) as sdk:
            for number in (1, 2):
                actual._current_api_request_id = f"{pending}:api:{number}"
                kwargs = actual._build_api_kwargs(
                    [{"role": "user", "content": message}], tools_for_api=[],
                )
                dispatch_provider_request(actual, kwargs,
                                          lambda request: sdk.chat.completions.create(**request))
        return {"final_response": "ok", "messages": [], "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", consume)
    result = agent.run_conversation(expanded.message)
    assert result["final_response"] == "ok"
    assert len(captured) == 2
    assert all(any(raw in message.get("content", "") for message in request["messages"])
               for request in captured)
    receipts = [json.loads(line) for line in
                (tmp_path / "egress/llm-egress-receipts.jsonl").read_text().splitlines()]
    assert [receipt["source_grant_count"] for receipt in receipts] == [1, 1]
    assert [receipt["source_segment_count"] for receipt in receipts] == [1, 1]
    assert agent._source_provenance_pending_turn_id is None
    assert not agent._source_provenance_registry.grants_for_request(f"{pending}:api:1")
    assert not agent._source_provenance_registry.grants_for_request(f"{pending}:api:2")


@pytest.mark.parametrize("surface", ["cli", "tui"])
def test_abandoned_context_preparation_clears_only_pending_authority(
    tmp_path, monkeypatch, surface,
):
    from types import SimpleNamespace
    from agent.context_references import preprocess_context_references
    from agent.source_provenance import provenance_kwargs_for_agent

    source = tmp_path / "source.py"
    source.write_text("my_value = 123\n")
    consumer = SimpleNamespace(session_id="session-1")

    def expand(message):
        result = preprocess_context_references(
            message, cwd=tmp_path, allowed_root=tmp_path, context_length=240_000,
            **provenance_kwargs_for_agent(consumer, establish_turn=True),
        )
        assert not result.blocked
        return result.message, None

    if surface == "cli":
        from cli import HermesCLI
        host = HermesCLI.__new__(HermesCLI)
        host.agent = consumer
        host._active_agent_route_signature = "route"
        host._secret_capture_callback = lambda *_a, **_k: None
        host._ensure_runtime_credentials = lambda: True
        host._resolve_turn_agent_config = lambda _m: {
            "signature": "route", "model": "model", "runtime": None,
        }
        host._init_agent = lambda **_kwargs: True
        host._sync_fallback_chain_with_config = lambda _a: None
        host._chat_route_images = lambda message, _images: message
        host._chat_expand_context_references = expand

        def failed_stage(*_a):
            raise RuntimeError("preparation failed")
        host._chat_stage_user_message = failed_stage
        with pytest.raises(RuntimeError, match="preparation failed"):
            host.chat("Inspect @file:source.py:1-1")
    else:
        from tui_gateway import server
        expand("Inspect @file:source.py:1-1")
        st = server._TurnRun(consumer, None, None, False)
        monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
        server._release_turn_scopes("test", {}, st)

    assert getattr(consumer, "_source_provenance_pending_turn_id", None) is None
    assert not consumer._source_provenance_registry._grants
