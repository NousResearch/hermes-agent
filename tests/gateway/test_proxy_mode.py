"""Tests for gateway proxy mode — forwarding messages to a remote API server."""

import json
import asyncio

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import resolve_proxy_url
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner
from gateway.recall_scope import (
    GATEWAY_PROXY_CHAT_COMPLETIONS_PATH,
    GATEWAY_PROXY_MARKER_HEADER,
    GATEWAY_PROXY_ORIGIN_HEADER,
    GATEWAY_PROXY_SESSION_KEY_HEADER,
    decode_gateway_proxy_origin,
    decode_gateway_proxy_session_key,
    encode_gateway_proxy_origin,
    encode_gateway_proxy_session_key,
)
from gateway.session import SessionSource, build_session_key
from hermes_state import SessionDB
from tools.session_search_tool import session_search


def _make_runner(proxy_url=None):
    """Create a minimal GatewayRunner for proxy tests."""
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = MagicMock()
    runner.config.streaming = StreamingConfig()
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    return runner


def _make_source(platform=Platform.MATRIX):
    return SessionSource(
        platform=platform,
        chat_id="!room:server.org",
        chat_name="Test Room",
        chat_type="group",
        user_id="@user:server.org",
        user_name="testuser",
        thread_id=None,
    )


def _gateway_session_key(source: SessionSource) -> str:
    return build_session_key(
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=True,
        profile=source.profile,
    )


def _proxy_headers(
    source: SessionSource,
    *,
    session_id: str,
    session_key: str | None = None,
) -> dict[str, str]:
    exact_key = session_key or _gateway_session_key(source)
    return {
        "Authorization": "Bearer test-key-123",
        "Content-Type": "application/json",
        "X-Hermes-Session-Id": session_id,
        GATEWAY_PROXY_MARKER_HEADER: "1",
        GATEWAY_PROXY_ORIGIN_HEADER: encode_gateway_proxy_origin(source.to_dict()),
        GATEWAY_PROXY_SESSION_KEY_HEADER: encode_gateway_proxy_session_key(
            exact_key
        ),
    }


def _proxy_api_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    wanted = {GATEWAY_PROXY_CHAT_COMPLETIONS_PATH, "/v1/chat/completions"}
    for method, path, handler in adapter._http_route_table():
        if path in wanted:
            app.router.add_route(method, path, handler)
    return app


async def _direct_api_dispatch(adapter, path, body, headers):
    """Run the real middleware, auth decorator, and handler without a socket."""
    request = MagicMock()
    request.headers = headers
    request.json = AsyncMock(return_value=body)
    request.remote = "127.0.0.1"
    request.transport = None
    request.app = {}
    request.match_info = {}
    handler = (
        adapter._handle_gateway_proxy_chat_completions
        if path == GATEWAY_PROXY_CHAT_COMPLETIONS_PATH
        else adapter._handle_chat_completions
    )
    middleware = adapter._make_profile_prefix_middleware()
    return await middleware(request, handler)


class _InProcessProxyResponse:
    def __init__(self, adapter, path, body, headers):
        self._adapter = adapter
        self._path = path
        self._body = body
        self._headers = headers
        self.status = 500
        self._text = ""
        self._chunks = []
        self.content = self

    async def __aenter__(self):
        request_body = dict(self._body)
        request_body["stream"] = False
        response = await _direct_api_dispatch(
            self._adapter,
            self._path,
            request_body,
            self._headers,
        )
        self.status = response.status
        self._text = response.text
        if response.status == 200:
            payload = json.loads(response.text)
            content = payload["choices"][0]["message"]["content"]
            self._chunks = [
                "data: "
                + json.dumps({"choices": [{"delta": {"content": content}}]})
                + "\n\ndata: [DONE]\n\n"
            ]
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self):
        return self._text

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk.encode("utf-8")


class _InProcessProxySession:
    def __init__(self, adapter):
        self._adapter = adapter

    def post(self, url, json=None, headers=None, **_kwargs):
        path = (
            GATEWAY_PROXY_CHAT_COMPLETIONS_PATH
            if url.endswith(GATEWAY_PROXY_CHAT_COMPLETIONS_PATH)
            else "/v1/chat/completions"
        )
        return _InProcessProxyResponse(
            self._adapter, path, json or {}, headers or {}
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _FakeSSEResponse:
    """Simulates an aiohttp response with SSE streaming."""

    def __init__(self, status=200, sse_chunks=None, error_text=""):
        self.status = status
        self._sse_chunks = sse_chunks or []
        self._error_text = error_text
        self.content = self

    async def text(self):
        return self._error_text

    async def iter_any(self):
        for chunk in self._sse_chunks:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeSession:
    """Simulates an aiohttp.ClientSession with captured request args."""

    def __init__(self, response):
        self._response = response
        self.captured_url = None
        self.captured_json = None
        self.captured_headers = None

    def post(self, url, json=None, headers=None, **kwargs):
        self.captured_url = url
        self.captured_json = json
        self.captured_headers = headers
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _patch_aiohttp(session):
    """Patch aiohttp.ClientSession to return our fake session."""
    return patch(
        "aiohttp.ClientSession",
        return_value=session,
    )


class TestGetProxyUrl:
    """Test _get_proxy_url() config resolution."""

    def test_returns_none_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
        runner = _make_runner()
        with patch("gateway.run._load_gateway_config", return_value={}):
            assert runner._get_proxy_url() is None


    def test_reads_from_config_yaml(self, monkeypatch):
        monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
        runner = _make_runner()
        cfg = {"gateway": {"proxy_url": "http://10.0.0.1:8642"}}
        with patch("gateway.run._load_gateway_config", return_value=cfg):
            assert runner._get_proxy_url() == "http://10.0.0.1:8642"


class TestResolveProxyUrl:

    def test_no_proxy_bypasses_matching_host(self, monkeypatch):
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                    "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
        monkeypatch.setenv("NO_PROXY", "api.telegram.org")

        assert resolve_proxy_url(target_hosts="api.telegram.org") is None

    def test_no_proxy_bypasses_cidr_target(self, monkeypatch):
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                    "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
        monkeypatch.setenv("NO_PROXY", "149.154.160.0/20")

        assert resolve_proxy_url(target_hosts=["149.154.167.220"]) is None


class TestRunAgentProxyDispatch:
    """Test that _run_agent() delegates to proxy when configured."""

    def test_api_server_registers_dedicated_gateway_proxy_route(self):
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": "test-key-123"})
        )
        routes = {
            (method, path): handler
            for method, path, handler in adapter._http_route_table()
        }
        assert (
            routes[("POST", GATEWAY_PROXY_CHAT_COMPLETIONS_PATH)].__func__
            is APIServerAdapter._handle_gateway_proxy_chat_completions
        )
        adapter._response_store.close()

    @pytest.mark.asyncio
    async def test_run_agent_delegates_to_proxy(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        runner = _make_runner()
        source = _make_source()

        expected_result = {
            "final_response": "Hello from remote!",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello from remote!"},
            ],
            "api_calls": 1,
            "tools": [],
        }

        runner._run_agent_via_proxy = AsyncMock(return_value=expected_result)

        result = await runner._run_agent(
            message="hi",
            context_prompt="",
            history=[],
            source=source,
            session_id="test-session-123",
            session_key="test-key",
            run_generation=7,
        )

        assert result["final_response"] == "Hello from remote!"
        runner._run_agent_via_proxy.assert_called_once()
        assert runner._run_agent_via_proxy.call_args.kwargs["run_generation"] == 7


class TestRunAgentViaProxy:
    """Test the actual proxy HTTP forwarding logic."""

    @pytest.mark.asyncio
    async def test_builds_correct_request(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.setenv("GATEWAY_PROXY_KEY", "test-key-123")
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[
                'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
                "data: [DONE]\n\n"
            ],
        )
        session = _FakeSession(resp)

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    result = await runner._run_agent_via_proxy(
                        message="How are you?",
                        context_prompt="You are helpful.",
                        history=[
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there!"},
                        ],
                        source=source,
                        session_id="session-abc",
                        session_key=_gateway_session_key(source),
                    )

        # Verify request URL
        assert session.captured_url == (
            "http://host:8642" + GATEWAY_PROXY_CHAT_COMPLETIONS_PATH
        )

        # Verify auth header
        assert session.captured_headers["Authorization"] == "Bearer test-key-123"

        # Verify session ID header
        assert session.captured_headers["X-Hermes-Session-Id"] == "session-abc"
        assert session.captured_headers[GATEWAY_PROXY_MARKER_HEADER] == "1"
        assert decode_gateway_proxy_origin(
            session.captured_headers[GATEWAY_PROXY_ORIGIN_HEADER]
        ) == {
            "platform": "matrix",
            "chat_id": "!room:server.org",
            "chat_type": "group",
            "user_id": "@user:server.org",
        }
        assert decode_gateway_proxy_session_key(
            session.captured_headers[GATEWAY_PROXY_SESSION_KEY_HEADER]
        ) == _gateway_session_key(source)

        # Verify messages include system, history, and current message
        messages = session.captured_json["messages"]
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert messages[2] == {"role": "assistant", "content": "Hi there!"}
        assert messages[3] == {"role": "user", "content": "How are you?"}

        # Verify streaming is requested
        assert session.captured_json["stream"] is True

        # Verify response was assembled
        assert result["final_response"] == "Hello world"


    @pytest.mark.asyncio
    async def test_handles_connection_error(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://unreachable:8642")
        monkeypatch.setenv("GATEWAY_PROXY_KEY", "test-key-123")
        runner = _make_runner()
        source = _make_source()

        class _ErrorSession:
            def post(self, *args, **kwargs):
                raise ConnectionError("Connection refused")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("gateway.run._load_gateway_config", return_value={}):
            with patch("aiohttp.ClientSession", return_value=_ErrorSession()):
                with patch("aiohttp.ClientTimeout"):
                    result = await runner._run_agent_via_proxy(
                        message="hi",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="test",
                        session_key=_gateway_session_key(source),
                    )

        assert "Proxy connection error" in result["final_response"]


    @pytest.mark.asyncio
    async def test_no_system_message_when_context_empty(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.setenv("GATEWAY_PROXY_KEY", "test-key-123")
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'],
        )
        session = _FakeSession(resp)

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    await runner._run_agent_via_proxy(
                        message="hello",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="test",
                        session_key=_gateway_session_key(source),
                    )

        # No system message should appear when context_prompt is empty
        messages = session.captured_json["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_non_gateway_api_source_omits_gateway_scope_headers(
        self, monkeypatch
    ):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.setenv("GATEWAY_PROXY_KEY", "test-key-123")
        runner = _make_runner()
        response = _FakeSSEResponse(
            status=200,
            sse_chunks=[
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b"data: [DONE]\n\n"
            ],
        )
        session = _FakeSession(response)

        with patch("gateway.run._load_gateway_config", return_value={}), _patch_aiohttp(
            session
        ), patch("aiohttp.ClientTimeout"):
            result = await runner._run_agent_via_proxy(
                message="ordinary API",
                context_prompt="",
                history=[],
                source=_make_source(Platform.API_SERVER),
                session_id="api-session",
            )

        assert result["final_response"] == "ok"
        assert session.captured_url == "http://host:8642/v1/chat/completions"
        assert GATEWAY_PROXY_MARKER_HEADER not in session.captured_headers
        assert GATEWAY_PROXY_ORIGIN_HEADER not in session.captured_headers
        assert GATEWAY_PROXY_SESSION_KEY_HEADER not in session.captured_headers

    @pytest.mark.asyncio
    async def test_missing_proxy_key_fails_before_http_and_never_broadens(
        self, monkeypatch
    ):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()

        with patch("gateway.run._load_gateway_config", return_value={}):
            result = await runner._run_agent_via_proxy(
                message="hello",
                context_prompt="",
                history=[],
                source=_make_source(),
                session_id="test",
            )

        assert "requires GATEWAY_PROXY_KEY" in result["final_response"]

    @pytest.mark.asyncio
    async def test_missing_or_oversized_proxy_context_fails_before_http(
        self, monkeypatch
    ):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.setenv("GATEWAY_PROXY_KEY", "test-key-123")
        source = _make_source()

        class _NoHTTP:
            def __init__(self):
                self.called = False

            def post(self, *_args, **_kwargs):
                self.called = True
                raise AssertionError("invalid proxy context reached HTTP")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        for session_key, expected in (
            (None, "missing or malformed"),
            ("k" * 6143, "too large"),
        ):
            client = _NoHTTP()
            runner = _make_runner()
            with patch("aiohttp.ClientSession", return_value=client), patch(
                "aiohttp.ClientTimeout"
            ):
                result = await runner._run_agent_via_proxy(
                    message="hello",
                    context_prompt="",
                    history=[],
                    source=source,
                    session_id="test",
                    session_key=session_key,
                )
            assert expected in result["final_response"]
            assert client.called is False

        oversized_origin_source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-a",
            chat_type="group",
            user_id="x" * 6025,
            scope_id="guild-1",
        )
        client = _NoHTTP()
        with patch("aiohttp.ClientSession", return_value=client), patch(
            "aiohttp.ClientTimeout"
        ):
            result = await _make_runner()._run_agent_via_proxy(
                message="hello",
                context_prompt="",
                history=[],
                source=oversized_origin_source,
                session_id="test",
                session_key=_gateway_session_key(oversized_origin_source),
            )
        assert "origin is too large" in result["final_response"]
        assert client.called is False

    @pytest.mark.asyncio
    async def test_authenticated_proxy_to_api_preserves_scope_and_fails_closed(
        self, monkeypatch, tmp_path
    ):
        db = SessionDB(tmp_path / "state.db")
        current_source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-a",
            chat_type="group",
            user_id="user-1",
            scope_id="guild-1",
        )
        peer_source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-a",
            chat_type="group",
            user_id="user-2",
            scope_id="guild-1",
        )
        other_source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-b",
            chat_type="group",
            user_id="user-1",
            scope_id="guild-1",
        )
        current_key = _gateway_session_key(current_source)
        peer_key = _gateway_session_key(peer_source)
        other_key = _gateway_session_key(other_source)
        db.create_session("old-ambiguous", source="discord")
        for session_id, source, session_key, content in (
            ("current", current_source, current_key, "active turn"),
            ("peer", peer_source, peer_key, "proxy scope needle"),
            ("other", other_source, other_key, "proxy scope needle"),
        ):
            db.create_session(
                session_id,
                source=source.platform.value,
                session_key=session_key,
                origin_json=json.dumps(source.to_dict()),
            )
            db.append_message(session_id, role="user", content=content)

        remote = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": "test-key-123"})
        )
        remote._session_db = db
        calls = []
        bound_contexts = []

        async def _run_bound_search(
            *,
            user_message,
            session_id,
            gateway_session_key=None,
            gateway_origin=None,
            stream_delta_callback=None,
            **_kwargs,
        ):
            from gateway.session_context import (
                clear_session_vars,
                gateway_context_active,
                get_bound_gateway_origin,
            )

            tokens = remote._bind_api_server_session(
                chat_id=session_id,
                session_key=gateway_session_key or session_id,
                session_id=session_id,
                gateway_origin=gateway_origin,
            )
            try:
                calls.append(user_message)
                bound_contexts.append(
                    (gateway_context_active(), get_bound_gateway_origin())
                )
                payload = session_search(
                    query="proxy scope needle",
                    scope="all" if user_message == "all" else None,
                    db=db,
                    current_session_id=session_id,
                    limit=10,
                )
                if stream_delta_callback is not None:
                    stream_delta_callback(payload)
                return (
                    {
                        "final_response": payload,
                        "messages": [],
                        "api_calls": 1,
                        "tools": [],
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
            finally:
                clear_session_vars(tokens)

        monkeypatch.setattr(remote, "_run_agent", _run_bound_search)
        async def _inline_to_thread(function, /, *args, **kwargs):
            return function(*args, **kwargs)

        # This sandbox forbids sockets and has an unreliable default-executor
        # wakeup for asyncio.to_thread. Keep the real middleware/auth/handler,
        # durable verifier, and binder; only the tiny DB reads execute inline.
        monkeypatch.setattr(
            "gateway.platforms.api_server.asyncio.to_thread",
            _inline_to_thread,
        )
        body = {
            "model": "hermes-agent",
            "messages": [{"role": "user", "content": "ordinary"}],
        }
        runner = _make_runner()
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://in-process-proxy")
        monkeypatch.setenv("GATEWAY_PROXY_KEY", "test-key-123")
        with patch(
            "aiohttp.ClientSession",
            return_value=_InProcessProxySession(remote),
        ), patch("aiohttp.ClientTimeout"):
            fresh_result = await asyncio.wait_for(
                runner._run_agent_via_proxy(
                    message="fresh",
                    context_prompt="",
                    history=[],
                    source=current_source,
                    session_id="fresh-current",
                    session_key=current_key,
                ),
                timeout=5,
            )
            current_result = await asyncio.wait_for(
                runner._run_agent_via_proxy(
                    message="current",
                    context_prompt="",
                    history=[],
                    source=current_source,
                    session_id="current",
                    session_key=current_key,
                ),
                timeout=5,
            )
            all_result = await asyncio.wait_for(
                runner._run_agent_via_proxy(
                    message="all",
                    context_prompt="",
                    history=[],
                    source=current_source,
                    session_id="current",
                    session_key=current_key,
                ),
                timeout=5,
            )
        ordinary_resp = await _direct_api_dispatch(
            remote,
            "/v1/chat/completions",
            body,
            _proxy_headers(
                current_source,
                session_id="ordinary-session",
                session_key=current_key,
            ),
        )
        ordinary_body = json.loads(ordinary_resp.text)

        user_b_resp = await _direct_api_dispatch(
            remote,
            GATEWAY_PROXY_CHAT_COMPLETIONS_PATH,
            body,
            _proxy_headers(
                peer_source,
                session_id="current",
                session_key=peer_key,
            ),
        )
        wrong_origin = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-b",
            chat_type="group",
            user_id="user-1",
            scope_id="guild-1",
        )
        tampered_resp = await _direct_api_dispatch(
            remote,
            GATEWAY_PROXY_CHAT_COMPLETIONS_PATH,
            body,
            _proxy_headers(
                wrong_origin,
                session_id="current",
                session_key=current_key,
            ),
        )
        ambiguous_resp = await _direct_api_dispatch(
            remote,
            GATEWAY_PROXY_CHAT_COMPLETIONS_PATH,
            body,
            _proxy_headers(
                current_source,
                session_id="old-ambiguous",
                session_key=current_key,
            ),
        )
        wrong_profile_origin = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-a",
            chat_type="group",
            user_id="user-1",
            scope_id="guild-1",
            profile="work",
        )
        profile_mismatch_resp = await _direct_api_dispatch(
            remote,
            GATEWAY_PROXY_CHAT_COMPLETIONS_PATH,
            body,
            _proxy_headers(
                wrong_profile_origin,
                session_id="current",
            ),
        )

        fresh_payload = json.loads(fresh_result["final_response"])
        current_payload = json.loads(current_result["final_response"])
        all_payload = json.loads(all_result["final_response"])
        ordinary_payload = json.loads(
            ordinary_body["choices"][0]["message"]["content"]
        )
        assert {row["session_id"] for row in fresh_payload["results"]} == {"peer"}
        assert json.loads(db.get_session("fresh-current")["origin_json"])[
            "chat_id"
        ] == "chat-a"
        assert db.get_session("fresh-current")["session_key"] == current_key
        assert {row["session_id"] for row in current_payload["results"]} == {"peer"}
        assert {row["session_id"] for row in all_payload["results"]} == {
            "peer",
            "other",
        }
        assert ordinary_resp.status == 200
        assert {row["session_id"] for row in ordinary_payload["results"]} == {
            "peer",
            "other",
        }
        assert user_b_resp.status == 409
        assert json.loads(user_b_resp.text)["error"]["code"] == (
            "gateway_proxy_session_key_mismatch"
        )
        assert tampered_resp.status == 409
        assert ambiguous_resp.status == 409
        assert profile_mismatch_resp.status == 409
        assert calls == ["fresh", "current", "all", "ordinary"]
        assert [active for active, _origin in bound_contexts] == [
            True,
            True,
            True,
            False,
        ]
        assert bound_contexts[-1][1] is None
        db.close()
        remote._response_store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("label", "changes"),
        [
            ("marker missing", {GATEWAY_PROXY_MARKER_HEADER: None}),
            ("marker empty", {GATEWAY_PROXY_MARKER_HEADER: ""}),
            ("marker malformed", {GATEWAY_PROXY_MARKER_HEADER: "0"}),
            ("origin missing", {GATEWAY_PROXY_ORIGIN_HEADER: None}),
            ("origin empty", {GATEWAY_PROXY_ORIGIN_HEADER: ""}),
            ("origin malformed", {GATEWAY_PROXY_ORIGIN_HEADER: "%%%"}),
            ("origin oversized", {GATEWAY_PROXY_ORIGIN_HEADER: "a" * 8191}),
            ("key missing", {GATEWAY_PROXY_SESSION_KEY_HEADER: None}),
            ("key empty", {GATEWAY_PROXY_SESSION_KEY_HEADER: ""}),
            ("key malformed", {GATEWAY_PROXY_SESSION_KEY_HEADER: "%%%"}),
            ("key oversized", {GATEWAY_PROXY_SESSION_KEY_HEADER: "a" * 8191}),
            ("session id missing", {"X-Hermes-Session-Id": None}),
            (
                "proxy headers stripped",
                {
                    GATEWAY_PROXY_MARKER_HEADER: None,
                    GATEWAY_PROXY_ORIGIN_HEADER: None,
                    GATEWAY_PROXY_SESSION_KEY_HEADER: None,
                },
            ),
        ],
    )
    async def test_dedicated_route_rejects_missing_empty_malformed_or_oversized_context(
        self, tmp_path, label, changes
    ):
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-a",
            chat_type="group",
            user_id="user-a",
            scope_id="guild-1",
        )
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": "test-key-123"})
        )
        adapter._session_db = SessionDB(tmp_path / "state.db")
        adapter._create_agent = MagicMock(
            side_effect=AssertionError(f"{label} reached agent construction")
        )
        headers = _proxy_headers(source, session_id="new-session")
        for name, value in changes.items():
            if value is None:
                headers.pop(name, None)
            else:
                headers[name] = value
        body = {
            "model": "hermes-agent",
            "messages": [{"role": "user", "content": "hello"}],
        }
        response = await _direct_api_dispatch(
            adapter,
            GATEWAY_PROXY_CHAT_COMPLETIONS_PATH,
            body,
            headers,
        )
        assert response.status == 400, label
        adapter._create_agent.assert_not_called()
        adapter._session_db.close()
        adapter._response_store.close()

    @pytest.mark.asyncio
    async def test_dedicated_route_uses_real_bearer_auth_before_context_parser(
        self, tmp_path
    ):
        source = _make_source(Platform.DISCORD)
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": "test-key-123"})
        )
        adapter._session_db = SessionDB(tmp_path / "state.db")
        headers = _proxy_headers(source, session_id="new-session")
        headers.pop("Authorization")
        response = await _direct_api_dispatch(
            adapter,
            GATEWAY_PROXY_CHAT_COMPLETIONS_PATH,
            {"messages": [{"role": "user", "content": "hello"}]},
            headers,
        )
        assert response.status == 401
        adapter._session_db.close()
        adapter._response_store.close()

    def test_proxy_header_encoders_match_aiohttp_8190_byte_boundary(self):
        origin = {
            "platform": "discord",
            "chat_id": "chat-a",
            "chat_type": "group",
            "scope_id": "guild-1",
            "user_id": "x" * 6024,
        }
        encoded_origin = encode_gateway_proxy_origin(origin)
        encoded_key = encode_gateway_proxy_session_key("k" * 6142)
        assert len(encoded_origin.encode("ascii")) == 8190
        assert len(encoded_key.encode("ascii")) == 8190
        assert decode_gateway_proxy_origin(encoded_origin) == origin
        assert decode_gateway_proxy_session_key(encoded_key) == "k" * 6142
        with pytest.raises(ValueError, match="too large"):
            encode_gateway_proxy_origin({**origin, "user_id": "x" * 6025})
        with pytest.raises(ValueError, match="too large"):
            encode_gateway_proxy_session_key("k" * 6143)

    @pytest.mark.parametrize(
        "header_name",
        [GATEWAY_PROXY_ORIGIN_HEADER, GATEWAY_PROXY_SESSION_KEY_HEADER],
    )
    def test_real_aiohttp_header_parser_uses_same_8190_value_limit(
        self, header_name
    ):
        from aiohttp.http_exceptions import LineTooLong
        from aiohttp.http_parser import HttpRequestParser

        def parse(value_size):
            loop = asyncio.new_event_loop()
            try:
                parser = HttpRequestParser(
                    None,
                    loop,
                    2**16,
                    max_field_size=8190,
                )
                request = (
                    b"POST / HTTP/1.1\r\nHost: localhost\r\n"
                    + header_name.encode("ascii")
                    + b": "
                    + (b"a" * value_size)
                    + b"\r\nContent-Length: 0\r\n\r\n"
                )
                return parser.feed_data(request)
            finally:
                loop.close()

        messages, _upgraded, _tail = parse(8190)
        assert len(messages) == 1
        with pytest.raises(LineTooLong):
            parse(8191)

    @pytest.mark.asyncio
    async def test_real_aiohttp_parser_accepts_encoder_limit_and_rejects_overflow(
        self, monkeypatch, tmp_path
    ):
        """Exercise the TCP parser when the execution sandbox permits sockets."""
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-a",
            chat_type="group",
            user_id="x" * 6003,
            scope_id="guild-1",
        )
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": "test-key-123"})
        )
        adapter._session_db = SessionDB(tmp_path / "state.db")

        class _Agent:
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_total_tokens = 0
            provider = "test"
            model = "test"

            def __init__(self, session_id):
                self.session_id = session_id

            def run_conversation(self, **_kwargs):
                return {
                    "final_response": "ok",
                    "messages": [],
                    "api_calls": 1,
                    "tools": [],
                }

        monkeypatch.setattr(
            adapter,
            "_create_agent",
            lambda **kwargs: _Agent(kwargs["session_id"]),
        )
        client = TestClient(TestServer(_proxy_api_app(adapter)))
        try:
            try:
                await client.start_server()
            except PermissionError:
                pytest.skip("sandbox forbids loopback sockets")
            body = {"messages": [{"role": "user", "content": "hello"}]}
            headers = _proxy_headers(source, session_id="at-limit")
            assert len(headers[GATEWAY_PROXY_ORIGIN_HEADER]) == 8190
            accepted = await client.post(
                GATEWAY_PROXY_CHAT_COMPLETIONS_PATH,
                json=body,
                headers=headers,
            )
            assert accepted.status == 200

            overflow_headers = dict(headers)
            overflow_headers[GATEWAY_PROXY_ORIGIN_HEADER] = "a" * 8191
            rejected = await client.post(
                GATEWAY_PROXY_CHAT_COMPLETIONS_PATH,
                json=body,
                headers=overflow_headers,
            )
            assert rejected.status == 400
        finally:
            await client.close()
            adapter._session_db.close()
            adapter._response_store.close()

    @pytest.mark.asyncio
    async def test_new_session_insert_conflict_reads_back_exact_key_and_origin(
        self, monkeypatch, tmp_path
    ):
        db = SessionDB(tmp_path / "state.db")
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": "test-key-123"})
        )
        adapter._session_db = db
        source_a = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-a",
            chat_type="group",
            user_id="user-a",
            scope_id="guild-1",
        )
        source_b = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-a",
            chat_type="group",
            user_id="user-b",
            scope_id="guild-1",
        )
        key_a = _gateway_session_key(source_a)
        key_b = _gateway_session_key(source_b)
        original_execute_write = db._execute_write

        def inject_conflicting_winner(callback, *args, **kwargs):
            def wrapped(conn):
                conn.execute(
                    """INSERT INTO sessions (
                           id, source, user_id, session_key, chat_id,
                           chat_type, origin_json, started_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        "raced",
                        "discord",
                        "user-b",
                        key_b,
                        "chat-a",
                        "group",
                        json.dumps(source_b.to_dict()),
                        1.0,
                    ),
                )
                return callback(conn)

            return original_execute_write(wrapped, *args, **kwargs)

        async def _inline_to_thread(function, /, *args, **kwargs):
            return function(*args, **kwargs)

        monkeypatch.setattr(db, "_execute_write", inject_conflicting_winner)
        monkeypatch.setattr(
            "gateway.platforms.api_server.asyncio.to_thread",
            _inline_to_thread,
        )
        rejected = await adapter._verify_gateway_proxy_session_context(
            session_id="raced",
            origin=source_a.to_dict(),
            session_key=key_a,
        )
        assert rejected is not None
        assert rejected.status == 409
        assert json.loads(rejected.text)["error"]["code"] == (
            "gateway_proxy_session_key_mismatch"
        )
        row = db.get_session("raced")
        durable_origin = json.loads(row["origin_json"])
        assert durable_origin["user_id"] == "user-b"
        assert row["session_key"] == key_b
        db.close()
        adapter._response_store.close()


class TestEnvVarRegistration:
    """Verify GATEWAY_PROXY_URL and GATEWAY_PROXY_KEY are registered."""

    def test_proxy_url_in_optional_env_vars(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "GATEWAY_PROXY_URL" in OPTIONAL_ENV_VARS
        info = OPTIONAL_ENV_VARS["GATEWAY_PROXY_URL"]
        assert info["category"] == "messaging"
        assert info["password"] is False
