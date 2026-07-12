from types import SimpleNamespace

import pytest

from agent.codex_websocket_transport import (
    CachedWebSocketConnection,
    CachedWebSocketContinuationState,
    WebSocketNotStartedError,
    build_cached_websocket_request_body,
    cleanup_codex_websocket_sessions,
    get_cached_websocket_input_delta,
    is_supported_codex_websocket_backend,
    request_bodies_match_except_input,
    resolve_codex_websocket_url,
    run_codex_websocket_stream,
)


def _body(extra=None, input_items=None):
    data = {
        "model": "codex",
        "instructions": "You are helpful",
        "input": input_items or [{"role": "user", "content": "one"}],
        "store": False,
        "prompt_cache_key": "s",
        "extra_headers": {"session_id": "s"},
    }
    if extra:
        data.update(extra)
    return data


def test_resolve_codex_websocket_url_from_backend_base():
    assert resolve_codex_websocket_url("https://chatgpt.com/backend-api/codex") == (
        "wss://chatgpt.com/backend-api/codex/responses"
    )
    assert resolve_codex_websocket_url("http://localhost:8080/codex/responses") == (
        "ws://localhost:8080/codex/responses"
    )


def test_supported_backend_is_narrow():
    assert is_supported_codex_websocket_backend(
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )
    assert not is_supported_codex_websocket_backend(
        provider="custom",
        base_url="https://chatgpt.com/backend-api/codex",
    )
    assert not is_supported_codex_websocket_backend(
        provider="openai-codex",
        base_url="https://api.openai.com/v1",
    )


def test_request_bodies_match_ignores_input_and_previous_response_id():
    a = _body(input_items=[{"role": "user", "content": "one"}])
    b = _body(input_items=[{"role": "user", "content": "two"}], extra={"previous_response_id": "resp_1"})
    assert request_bodies_match_except_input(a, b)
    c = _body(extra={"reasoning": {"effort": "high"}})
    assert not request_bodies_match_except_input(a, c)


def test_cached_delta_second_request_sends_suffix_and_previous_response_id():
    first = _body(input_items=[{"role": "user", "content": "one"}])
    response_items = [{"role": "assistant", "content": "two"}]
    continuation = CachedWebSocketContinuationState(
        last_request_body=first,
        last_response_id="resp_1",
        last_response_items=response_items,
    )
    second = _body(input_items=first["input"] + response_items + [{"role": "user", "content": "three"}])
    delta = get_cached_websocket_input_delta(second, continuation)
    assert delta == [{"role": "user", "content": "three"}]
    entry = CachedWebSocketConnection(ws=SimpleNamespace())
    entry.continuation = continuation
    request = build_cached_websocket_request_body(entry, second)
    assert request["previous_response_id"] == "resp_1"
    assert request["input"] == [{"role": "user", "content": "three"}]


def test_cached_delta_config_change_or_prefix_mismatch_sends_full_context():
    first = _body(input_items=[{"role": "user", "content": "one"}])
    continuation = CachedWebSocketContinuationState(
        last_request_body=first,
        last_response_id="resp_1",
        last_response_items=[{"role": "assistant", "content": "two"}],
    )
    entry = CachedWebSocketConnection(ws=SimpleNamespace())
    entry.continuation = continuation
    changed = _body(
        extra={"reasoning": {"effort": "high"}},
        input_items=first["input"] + continuation.last_response_items + [{"role": "user", "content": "three"}],
    )
    assert build_cached_websocket_request_body(entry, changed) == changed
    assert entry.continuation is None

    entry.continuation = continuation
    branched = _body(input_items=[{"role": "user", "content": "different"}])
    assert build_cached_websocket_request_body(entry, branched) == branched
    assert entry.continuation is None


def test_continuation_response_items_are_generated_through_adapter():
    from agent.codex_websocket_transport import _response_items_for_continuation

    response = SimpleNamespace(
        id="resp_1",
        status="completed",
        output=[
            SimpleNamespace(
                type="reasoning",
                id="rs_1",
                encrypted_content="enc",
                summary=[SimpleNamespace(type="summary_text", text="thought")],
            ),
            SimpleNamespace(
                type="message",
                id="msg_1",
                phase="final_answer",
                status="completed",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text="answer")],
            ),
        ],
    )
    items = _response_items_for_continuation(response)
    assert items[0]["type"] == "reasoning"
    assert "id" not in items[0]
    assert items[0]["encrypted_content"] == "enc"
    assert items[1] == {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "answer"}],
        "id": "msg_1",
        "phase": "final_answer",
    }


def test_websocket_unsupported_provider_fails_before_start():
    with pytest.raises(WebSocketNotStartedError):
        run_codex_websocket_stream(
            api_kwargs=_body(),
            client=SimpleNamespace(api_key="k", default_headers={}),
            provider="custom",
            base_url="https://api.openai.com/v1",
            session_id="s",
            transport="websocket-cached",
            collect_events=lambda events, getter: getter(),
        )


def test_missing_websockets_hint_for_forced_mode(monkeypatch):
    def _missing():
        raise RuntimeError("Codex Responses WebSocket transport requires the optional 'websockets' package.")

    monkeypatch.setattr("agent.codex_websocket_transport._require_websockets_connect", _missing)
    cleanup_codex_websocket_sessions()
    with pytest.raises(WebSocketNotStartedError) as excinfo:
        run_codex_websocket_stream(
            api_kwargs=_body(),
            client=SimpleNamespace(api_key="k", default_headers={}),
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            session_id="missing-websockets",
            transport="websocket",
            collect_events=lambda events, getter: getter(),
        )
    assert "websockets" in str(excinfo.value)


def test_connection_reuse_busy_socket_uses_temporary(monkeypatch):
    import agent.codex_websocket_transport as mod

    cleanup_codex_websocket_sessions()
    sockets = []

    class FakeWS:
        def __init__(self, name):
            self.name = name
            self.closed = False

        def close(self):
            self.closed = True

    def fake_connect(url, headers, timeout=None):
        ws = FakeWS(f"ws{len(sockets)}")
        sockets.append(ws)
        return ws

    monkeypatch.setattr(mod, "_connect_websocket", fake_connect)
    ws1, entry1, reused1, release1 = mod._acquire_websocket("wss://x", [], "s")
    assert not reused1
    release1(keep=True)
    ws2, entry2, reused2, release2 = mod._acquire_websocket("wss://x", [], "s")
    assert ws2 is ws1
    assert reused2
    ws3, entry3, reused3, release3 = mod._acquire_websocket("wss://x", [], "s")
    assert ws3 is not ws1
    assert entry3 is None
    assert not reused3
    release2(keep=True)
    release3(keep=True)
    cleanup_codex_websocket_sessions()
