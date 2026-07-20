import threading
from types import SimpleNamespace

import pytest

from agent.codex_websocket_transport import (
    CachedWebSocketConnection,
    CachedWebSocketContinuationState,
    WebSocketNotStartedError,
    build_cached_websocket_request_body,
    build_codex_websocket_wire_body,
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
    d = _body(extra={"client_metadata": {"x-codex-turn-state": "turn-1"}})
    assert request_bodies_match_except_input(a, d)


def test_wire_body_excludes_sdk_only_options():
    body = build_codex_websocket_wire_body(
        _body(extra={
            "extra_body": {
                "client_metadata": {"originator": "hermes"},
                "text": {"verbosity": "low"},
            },
            "timeout": 12.0,
            "stream": True,
            "extra_query": {"api-version": "future"},
            "previous_response_id": "resp_1",
        })
    )

    assert body["previous_response_id"] == "resp_1"
    assert body["client_metadata"] == {"originator": "hermes"}
    assert body["text"] == {"verbosity": "low"}
    assert not ({"extra_headers", "extra_query", "extra_body", "timeout", "stream"} & body.keys())


def test_emitted_frame_contains_only_wire_fields(monkeypatch):
    import agent.codex_websocket_transport as mod

    class FakeWS:
        closed = False

        def send(self, payload):
            self.sent = payload

        def recv(self, timeout=None):
            return '{"type":"response.completed","response":{"id":"resp_1","status":"completed"}}'

        def close(self):
            self.closed = True

    ws = FakeWS()
    handshake = {}

    def fake_connect(url, headers, timeout=None):
        handshake["headers"] = dict(headers)
        handshake["timeout"] = timeout
        return ws

    monkeypatch.setattr(mod, "_connect_websocket", fake_connect)
    api_kwargs = _body(extra={
        "extra_body": {"client_metadata": {"originator": "hermes"}},
        "timeout": 8.0,
        "stream": True,
        "previous_response_id": "resp_1",
    })
    run_codex_websocket_stream(
        api_kwargs=api_kwargs,
        client=SimpleNamespace(api_key="k", default_headers={}),
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        session_id=None,
        transport="websocket",
        timeout=8.0,
        collect_events=lambda events, _getter: (
            list(events),
            SimpleNamespace(id="resp_1", status="completed", output=[]),
        )[1],
    )

    frame = ws.sent
    assert frame["type"] == "response.create"
    assert frame["previous_response_id"] == "resp_1"
    assert frame["client_metadata"] == {"originator": "hermes"}
    assert not ({"extra_headers", "extra_body", "timeout", "stream"} & frame.keys())
    assert handshake["headers"]["session-id"]
    assert handshake["headers"]["thread-id"]
    assert handshake["timeout"] == 8.0


def test_active_connection_abort_is_registered_and_unregistered(monkeypatch):
    import agent.codex_websocket_transport as mod

    class FakeWS:
        closed = False

        def send(self, payload):
            self.sent = payload

        def recv(self, timeout=None):
            return '{"type":"response.completed","response":{"id":"resp_1","status":"completed"}}'

        def close(self):
            self.closed = True

    ws = FakeWS()
    registration = {}

    def register(abort):
        registration["abort"] = abort

        def unregister():
            registration["unregistered"] = True

        return unregister

    monkeypatch.setattr(mod, "_connect_websocket", lambda client, headers, timeout=None: ws)
    run_codex_websocket_stream(
        api_kwargs=_body(),
        client=SimpleNamespace(),
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        session_id=None,
        transport="websocket",
        collect_events=lambda events, _getter: (
            list(events),
            SimpleNamespace(id="resp_1", status="completed", output=[]),
        )[1],
        register_connection_abort=register,
    )

    assert callable(registration["abort"])
    assert registration["unregistered"] is True
    assert ws.closed


def test_turn_state_metadata_is_recorded_and_added_to_request(monkeypatch):
    import agent.codex_websocket_transport as mod

    class FakeWS:
        closed = False

        def __init__(self):
            self.frames = iter(
                [
                    '{"type":"response.metadata","headers":'
                    '{"X-Codex-Turn-State":"sticky-turn"}}',
                    '{"type":"response.completed","response":'
                    '{"id":"resp_1","status":"completed"}}',
                ]
            )

        def send(self, payload):
            self.sent = payload

        def recv(self, timeout=None):
            return next(self.frames)

        def close(self):
            self.closed = True

    ws = FakeWS()
    recorded = []
    monkeypatch.setattr(
        mod,
        "_connect_websocket",
        lambda _client, _headers, timeout=None: ws,
    )

    run_codex_websocket_stream(
        api_kwargs=_body(
            extra={"extra_body": {"client_metadata": {"originator": "hermes"}}}
        ),
        client=SimpleNamespace(),
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        session_id=None,
        transport="websocket",
        turn_state="prior-state",
        record_turn_state=recorded.append,
        collect_events=lambda events, _getter: (
            list(events),
            SimpleNamespace(id="resp_1", status="completed", output=[]),
        )[1],
    )

    assert ws.sent["client_metadata"] == {
        "originator": "hermes",
        "x-codex-turn-state": "prior-state",
    }
    assert recorded == ["sticky-turn"]


def test_cross_thread_abort_shuts_down_underlying_socket():
    import agent.codex_websocket_transport as mod

    calls = []
    raw_connection = SimpleNamespace(
        socket=SimpleNamespace(shutdown=calls.append),
    )
    ws = SimpleNamespace(_connection=raw_connection)

    mod._abort_websocket_silently(ws)

    assert calls == [mod.socket.SHUT_RDWR]


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


def test_cached_delta_preserves_explicit_conflicting_previous_response_id():
    first = _body(input_items=[{"role": "user", "content": "one"}])
    continuation = CachedWebSocketContinuationState(
        last_request_body=first,
        last_response_id="resp_cached",
        last_response_items=[{"role": "assistant", "content": "two"}],
    )
    entry = CachedWebSocketConnection(ws=SimpleNamespace(), continuation=continuation)
    request = _body(
        extra={"previous_response_id": "resp_explicit"},
        input_items=[{"role": "user", "content": "new branch"}],
    )

    assert build_cached_websocket_request_body(entry, request) == request
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
            collect_events=lambda events, _getter: list(events),
        )


def test_missing_sdk_websocket_support_for_forced_mode():
    class Responses:
        def connect(self, **_kwargs):
            raise RuntimeError("install openai[realtime] to use responses.connect()")

    cleanup_codex_websocket_sessions()
    with pytest.raises(WebSocketNotStartedError) as excinfo:
        run_codex_websocket_stream(
            api_kwargs=_body(),
            client=SimpleNamespace(responses=Responses()),
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            session_id="missing-websockets",
            transport="websocket",
            collect_events=lambda events, getter: getter(),
        )
    assert "openai[realtime]" in str(excinfo.value)


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
    client = SimpleNamespace()
    ws1, entry1, reused1, release1 = mod._acquire_websocket(client, "wss://x", {}, "s")
    assert not reused1
    release1(keep=True)
    ws2, entry2, reused2, release2 = mod._acquire_websocket(client, "wss://x", {}, "s")
    assert ws2 is ws1
    assert reused2
    ws3, entry3, reused3, release3 = mod._acquire_websocket(client, "wss://x", {}, "s")
    assert ws3 is not ws1
    assert entry3 is None
    assert not reused3
    release2(keep=True)
    release3(keep=True)
    cleanup_codex_websocket_sessions()


def test_connection_cache_key_includes_auth_headers(monkeypatch):
    import agent.codex_websocket_transport as mod

    cleanup_codex_websocket_sessions()
    sockets = []

    class FakeWS:
        closed = False

        def close(self):
            self.closed = True

    def fake_connect(url, headers, timeout=None):
        ws = FakeWS()
        sockets.append(ws)
        return ws

    monkeypatch.setattr(mod, "_connect_websocket", fake_connect)
    client = SimpleNamespace()
    ws1, _entry1, _reused1, release1 = mod._acquire_websocket(
        client, "wss://x", {"Authorization": "Bearer old"}, "s"
    )
    release1(keep=True)
    ws2, _entry2, reused2, release2 = mod._acquire_websocket(
        client, "wss://x", {"Authorization": "Bearer new"}, "s"
    )
    assert ws2 is not ws1
    assert not reused2
    release2(keep=True)
    mod.cleanup_codex_websocket_session("s")
    assert all(ws.closed for ws in sockets)


def test_connection_cache_key_includes_sdk_auth_identity(monkeypatch):
    import agent.codex_websocket_transport as mod

    cleanup_codex_websocket_sessions()
    sockets = []

    class FakeWS:
        closed = False

        def close(self):
            self.closed = True

    def fake_connect(client, headers, timeout=None):
        ws = FakeWS()
        sockets.append(ws)
        return ws

    monkeypatch.setattr(mod, "_connect_websocket", fake_connect)
    old_client = SimpleNamespace(
        auth_headers={"Authorization": "Bearer old"},
        base_url="https://chatgpt.com/backend-api/codex",
    )
    new_client = SimpleNamespace(
        auth_headers={"Authorization": "Bearer new"},
        base_url="https://chatgpt.com/backend-api/codex",
    )
    ws1, _entry1, _reused1, release1 = mod._acquire_websocket(
        old_client, "wss://x", {}, "rotating-token"
    )
    release1(keep=True)
    ws2, _entry2, reused2, release2 = mod._acquire_websocket(
        new_client, "wss://x", {}, "rotating-token"
    )

    assert ws2 is not ws1
    assert not reused2
    release2(keep=True)
    cleanup_codex_websocket_sessions()
    assert all(ws.closed for ws in sockets)


def test_cleanup_codex_websocket_session_only_closes_matching_session(monkeypatch):
    import agent.codex_websocket_transport as mod

    cleanup_codex_websocket_sessions()
    sockets = []

    class FakeWS:
        closed = False

        def close(self):
            self.closed = True

    def fake_connect(url, headers, timeout=None):
        ws = FakeWS()
        sockets.append(ws)
        return ws

    monkeypatch.setattr(mod, "_connect_websocket", fake_connect)
    client = SimpleNamespace()
    ws1, _entry1, _reused1, release1 = mod._acquire_websocket(client, "wss://x", {}, "s1")
    ws2, _entry2, _reused2, release2 = mod._acquire_websocket(client, "wss://x", {}, "s2")
    release1(keep=True)
    release2(keep=True)
    mod.cleanup_codex_websocket_session("s1")
    assert ws1.closed
    assert not ws2.closed
    mod.cleanup_codex_websocket_sessions()


def test_recv_timeout_polls_interruption(monkeypatch):
    import agent.codex_websocket_transport as mod

    cleanup_codex_websocket_sessions()

    class FakeWS:
        closed = False

        def send(self, payload):
            self.sent = payload

        def recv(self, timeout=None):
            raise TimeoutError()

        def close(self):
            self.closed = True

    ws = FakeWS()
    monkeypatch.setattr(mod, "_connect_websocket", lambda url, headers, timeout=None: ws)
    calls = {"n": 0}

    def interrupted():
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(InterruptedError):
        run_codex_websocket_stream(
            api_kwargs=_body(),
            client=SimpleNamespace(api_key="k", default_headers={}),
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            session_id="timeout-interrupt",
            transport="websocket-cached",
            collect_events=lambda events, _getter: list(events),
            interrupted=interrupted,
        )
    assert calls["n"] >= 2
    assert ws.closed


def test_recv_timeout_enforces_stream_idle_timeout(monkeypatch):
    import agent.codex_websocket_transport as mod

    cleanup_codex_websocket_sessions()

    class FakeWS:
        closed = False

        def send(self, payload):
            self.sent = payload

        def recv(self, timeout=None):
            raise TimeoutError()

        def close(self):
            self.closed = True

    ws = FakeWS()
    timestamps = iter([10.0, 11.1])
    monkeypatch.setattr(mod, "_connect_websocket", lambda url, headers, timeout=None: ws)
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(timestamps))

    with pytest.raises(mod.WebSocketStartedError, match="idle for 1 seconds") as excinfo:
        run_codex_websocket_stream(
            api_kwargs=_body(),
            client=SimpleNamespace(api_key="k", default_headers={}),
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            session_id="idle-timeout",
            transport="websocket-cached",
            collect_events=lambda events, _getter: list(events),
            timeout=1.0,
        )

    assert excinfo.value.retryable
    assert ws.closed
    assert not mod._websocket_session_cache


def test_incomplete_response_does_not_seed_continuation(monkeypatch):
    import agent.codex_websocket_transport as mod

    cleanup_codex_websocket_sessions()

    class FakeWS:
        closed = False

        def send(self, payload):
            self.sent = payload

        def recv(self, timeout=None):
            return '{"type":"response.incomplete","response":{"id":"resp_incomplete","status":"incomplete","output":[]}}'

        def close(self):
            self.closed = True

    ws = FakeWS()
    monkeypatch.setattr(mod, "_connect_websocket", lambda url, headers, timeout=None: ws)
    response = run_codex_websocket_stream(
        api_kwargs=_body(),
        client=SimpleNamespace(api_key="k", default_headers={}),
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        session_id="incomplete-session",
        transport="websocket-cached",
        collect_events=lambda events, getter: SimpleNamespace(id="resp_incomplete", status="incomplete", output=[]),
    )
    assert response.status == "incomplete"
    assert ws.closed
    assert not mod._websocket_session_cache


def test_continuation_conversion_failure_keeps_completed_response(monkeypatch):
    import agent.codex_websocket_transport as mod

    cleanup_codex_websocket_sessions()

    class FakeWS:
        closed = False

        def send(self, payload):
            self.sent = payload

        def recv(self, timeout=None):
            return '{"type":"response.completed","response":{"id":"resp_1","status":"completed"}}'

        def close(self):
            self.closed = True

    ws = FakeWS()
    monkeypatch.setattr(mod, "_connect_websocket", lambda _client, _headers, timeout=None: ws)
    monkeypatch.setattr(
        mod,
        "_response_items_for_continuation",
        lambda _response: (_ for _ in ()).throw(ValueError("bad output item")),
    )
    final = SimpleNamespace(id="resp_1", status="completed", output=[])

    response = run_codex_websocket_stream(
        api_kwargs=_body(),
        client=SimpleNamespace(),
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        session_id="continuation-conversion",
        transport="websocket-cached",
        collect_events=lambda events, _getter: (list(events), final)[1],
    )

    assert response is final
    entry = next(iter(mod._websocket_session_cache.values()))
    assert entry.continuation is None
    assert not ws.closed
    cleanup_codex_websocket_sessions()


def test_native_sdk_connect_receives_handshake_options():
    import agent.codex_websocket_transport as mod

    captured = {}
    connection = SimpleNamespace(close=lambda: None)

    class Manager:
        def enter(self):
            return connection

    class Responses:
        def connect(self, **kwargs):
            captured.update(kwargs)
            return Manager()

    result = mod._connect_websocket(
        SimpleNamespace(responses=Responses()),
        {"OpenAI-Beta": "beta", "session-id": "s", "thread-id": "s"},
        extra_query={"api-version": "future"},
        timeout=9.0,
    )

    assert result is connection
    assert captured["extra_headers"]["session-id"] == "s"
    assert captured["extra_headers"]["thread-id"] == "s"
    assert captured["extra_query"] == {"api-version": "future"}
    assert captured["websocket_connection_options"]["open_timeout"] == 9.0


def test_native_sdk_connect_caps_handshake_timeout_independently():
    import agent.codex_websocket_transport as mod

    captured = {}

    class Manager:
        def enter(self):
            return SimpleNamespace(close=lambda: None)

    class Responses:
        def connect(self, **kwargs):
            captured.update(kwargs)
            return Manager()

    mod._connect_websocket(
        SimpleNamespace(responses=Responses()),
        {},
        timeout=900.0,
    )

    assert captured["websocket_connection_options"]["open_timeout"] == 15.0


def test_native_sdk_connect_normalizes_base_url_with_responses_suffix():
    import agent.codex_websocket_transport as mod

    captured = {}
    connection = SimpleNamespace(close=lambda: None)

    class Manager:
        def enter(self):
            return connection

    class Responses:
        def connect(self, **_kwargs):
            return Manager()

    copied_client = SimpleNamespace(responses=Responses())

    def copy_client(**kwargs):
        captured.update(kwargs)
        return copied_client

    client = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex/responses/",
        copy=copy_client,
        responses=SimpleNamespace(),
    )
    result = mod._connect_websocket(client, {})

    assert result is connection
    assert captured["websocket_base_url"] == "wss://chatgpt.com/backend-api/codex"


def test_handshake_headers_exclude_sdk_default_header_sentinels():
    import agent.codex_websocket_transport as mod

    client = SimpleNamespace(
        default_headers={
            "Authorization": "Bearer secret",
            "OpenAI-Organization": object(),
            "OpenAI-Project": object(),
            "Content-Type": "application/json",
        }
    )
    headers = mod._build_headers_from_client(
        client,
        {"extra_headers": {"x-custom": "yes"}},
        "session",
    )

    assert headers["x-custom"] == "yes"
    assert headers["session-id"] == "session"
    assert headers["thread-id"] == "session"
    assert "Authorization" not in headers
    assert "OpenAI-Organization" not in headers
    assert "OpenAI-Project" not in headers
    assert "Content-Type" not in headers


def test_handshake_headers_preserve_codex_client_custom_headers():
    import agent.codex_websocket_transport as mod

    client = SimpleNamespace(
        _custom_headers={
            "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
            "originator": "codex_cli_rs",
            "ChatGPT-Account-ID": "acct_123",
        }
    )
    headers = mod._build_headers_from_client(
        client,
        {"extra_headers": {"x-custom": "yes"}},
        "session",
    )

    assert headers["User-Agent"].startswith("codex_cli_rs/")
    assert headers["originator"] == "codex_cli_rs"
    assert headers["ChatGPT-Account-ID"] == "acct_123"
    assert headers["x-custom"] == "yes"


def test_handshake_headers_merge_case_insensitively():
    import agent.codex_websocket_transport as mod

    client = SimpleNamespace(
        _custom_headers={
            "Originator": "client-originator",
            "authorization": "Bearer override",
            "OPENAI-BETA": "stale-beta",
            "X-Custom": "client",
        }
    )
    headers = mod._build_headers_from_client(
        client,
        {
            "extra_headers": {
                "originator": "request-originator",
                "x-custom": "request",
                "Session-Id": "caller-session",
            }
        },
        "managed-session",
    )

    assert headers["Authorization"] == "Bearer override"
    assert headers["originator"] == "request-originator"
    assert headers["x-custom"] == "request"
    assert headers["OpenAI-Beta"] == mod.OPENAI_BETA_RESPONSES_WEBSOCKETS
    assert headers["session-id"] == "managed-session"
    for expected_name in {
        "authorization",
        "originator",
        "openai-beta",
        "x-custom",
        "session-id",
    }:
        assert sum(name.lower() == expected_name for name in headers) == 1


def test_concurrent_first_acquire_keeps_every_connection_managed(monkeypatch):
    import agent.codex_websocket_transport as mod

    cleanup_codex_websocket_sessions()
    connect_started = threading.Event()
    allow_first_connect = threading.Event()
    sockets = []
    results = []

    class FakeWS:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def fake_connect(_client, _headers, timeout=None):
        ws = FakeWS()
        sockets.append(ws)
        if len(sockets) == 1:
            connect_started.set()
            assert allow_first_connect.wait(timeout=2)
        return ws

    def acquire():
        results.append(
            mod._acquire_websocket(SimpleNamespace(), "wss://x", {}, "same-session")
        )

    monkeypatch.setattr(mod, "_connect_websocket", fake_connect)
    first = threading.Thread(target=acquire)
    second = threading.Thread(target=acquire)
    first.start()
    assert connect_started.wait(timeout=2)
    second.start()
    allow_first_connect.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert len(results) == 2
    assert len(sockets) == 2
    assert sum(entry is not None for _, entry, _, _ in results) == 1
    for _, _, _, release in results:
        release(keep=True)
    cleanup_codex_websocket_sessions()
    assert all(ws.closed for ws in sockets)


def test_cleanup_during_connect_prevents_cache_resurrection(monkeypatch):
    import agent.codex_websocket_transport as mod

    cleanup_codex_websocket_sessions()
    connect_started = threading.Event()
    allow_connect = threading.Event()
    result = []

    class FakeWS:
        closed = False

        def close(self):
            self.closed = True

    ws = FakeWS()

    def fake_connect(_client, _headers, timeout=None):
        connect_started.set()
        assert allow_connect.wait(timeout=2)
        return ws

    monkeypatch.setattr(mod, "_connect_websocket", fake_connect)
    thread = threading.Thread(
        target=lambda: result.append(
            mod._acquire_websocket(SimpleNamespace(), "wss://x", {}, "closing-session")
        )
    )
    thread.start()
    assert connect_started.wait(timeout=2)
    mod.cleanup_codex_websocket_session("closing-session")
    allow_connect.set()
    thread.join(timeout=2)

    assert len(result) == 1
    _, entry, _, release = result[0]
    assert entry is None
    assert not mod._websocket_session_cache
    release(keep=True)
    assert ws.closed


def test_websocket_rejection_preserves_status_code_body_and_code(monkeypatch):
    import agent.codex_websocket_transport as mod

    class FakeWS:
        closed = False

        def send(self, _payload):
            pass

        def recv(self, timeout=None):
            return (
                '{"type":"error","status":429,'
                '"error":{"code":"rate_limit_exceeded","message":"slow down"}}'
            )

        def close(self):
            self.closed = True

    ws = FakeWS()
    monkeypatch.setattr(mod, "_connect_websocket", lambda _client, _headers, timeout=None: ws)

    with pytest.raises(mod.WebSocketRejectedError) as excinfo:
        run_codex_websocket_stream(
            api_kwargs=_body(),
            client=SimpleNamespace(),
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            session_id="structured-error",
            transport="websocket-cached",
            collect_events=lambda events, _getter: list(events),
        )

    assert excinfo.value.status_code == 429
    assert excinfo.value.request_replay_safe is True
    assert excinfo.value.code == "rate_limit_exceeded"
    assert excinfo.value.body["error"]["message"] == "slow down"


def test_websocket_rejection_accepts_status_code_alias(monkeypatch):
    import agent.codex_websocket_transport as mod

    class FakeWS:
        def send(self, _payload):
            pass

        def recv(self, timeout=None):
            return (
                '{"type":"error","status_code":401,'
                '"error":{"code":"unauthorized","message":"refresh auth"}}'
            )

        def close(self):
            pass

    monkeypatch.setattr(
        mod,
        "_connect_websocket",
        lambda _client, _headers, timeout=None: FakeWS(),
    )

    with pytest.raises(mod.WebSocketRejectedError) as excinfo:
        run_codex_websocket_stream(
            api_kwargs=_body(),
            client=SimpleNamespace(),
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            session_id="structured-status-code-error",
            transport="websocket-cached",
            collect_events=lambda events, _getter: list(events),
        )

    assert excinfo.value.status_code == 401
    assert excinfo.value.code == "unauthorized"


def test_send_failure_is_ambiguous_and_never_replay_safe(monkeypatch):
    import agent.codex_websocket_transport as mod

    class FakeWS:
        closed = False

        def send(self, _payload):
            raise ConnectionError("write outcome unknown")

        def close(self):
            self.closed = True

    ws = FakeWS()
    monkeypatch.setattr(
        mod,
        "_connect_websocket",
        lambda _client, _headers, timeout=None: ws,
    )

    with pytest.raises(mod.WebSocketStartedError) as excinfo:
        run_codex_websocket_stream(
            api_kwargs=_body(),
            client=SimpleNamespace(),
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            session_id="ambiguous-send",
            transport="websocket-cached",
            collect_events=lambda events, _getter: list(events),
        )

    assert excinfo.value.request_replay_safe is False
    assert excinfo.value.retryable is True
    assert ws.closed


def test_connection_lifetime_error_is_replay_safe_and_retryable(monkeypatch):
    import agent.codex_websocket_transport as mod

    class FakeWS:
        closed = False

        def send(self, _payload):
            pass

        def recv(self, timeout=None):
            return (
                '{"type":"error","status":400,"error":{'
                '"code":"websocket_connection_limit_reached",'
                '"message":"create a new websocket connection"}}'
            )

        def close(self):
            self.closed = True

    ws = FakeWS()
    monkeypatch.setattr(
        mod,
        "_connect_websocket",
        lambda _client, _headers, timeout=None: ws,
    )

    with pytest.raises(mod.WebSocketNotStartedError) as excinfo:
        run_codex_websocket_stream(
            api_kwargs=_body(),
            client=SimpleNamespace(),
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            session_id="expired-connection",
            transport="websocket-cached",
            collect_events=lambda events, _getter: list(events),
        )

    assert excinfo.value.request_replay_safe is True
    assert excinfo.value.retryable is True
    assert ws.closed


def test_response_done_uses_embedded_failed_status_and_error(monkeypatch):
    import agent.codex_websocket_transport as mod

    class FakeWS:
        closed = False

        def send(self, _payload):
            pass

        def recv(self, timeout=None):
            return (
                '{"type":"response.done","response":{"id":"resp_failed",'
                '"status":"failed","error":{"code":"overloaded","message":"busy"}}}'
            )

        def close(self):
            self.closed = True

    ws = FakeWS()
    monkeypatch.setattr(mod, "_connect_websocket", lambda _client, _headers, timeout=None: ws)
    seen = []

    def collect(events, _getter):
        seen.extend(events)
        event = seen[-1]
        return SimpleNamespace(
            id=event.response.id,
            status=event.response.status,
            error=event.response.error,
            output=[],
        )

    response = run_codex_websocket_stream(
        api_kwargs=_body(),
        client=SimpleNamespace(),
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        session_id="failed-done",
        transport="websocket-cached",
        collect_events=collect,
    )

    assert seen[-1].type == "response.failed"
    assert response.status == "failed"
    assert response.error.code == "overloaded"
    assert ws.closed
