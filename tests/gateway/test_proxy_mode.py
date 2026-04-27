"""Tests for gateway proxy mode — forwarding messages to a remote API server."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.platforms.base import resolve_proxy_url
from gateway.run import GatewayRunner
from gateway.session import SessionSource


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

    def test_reads_from_env_var(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://192.168.1.100:8642")
        runner = _make_runner()
        assert runner._get_proxy_url() == "http://192.168.1.100:8642"

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642/")
        runner = _make_runner()
        assert runner._get_proxy_url() == "http://host:8642"

    def test_reads_from_config_yaml(self, monkeypatch):
        monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
        runner = _make_runner()
        cfg = {"gateway": {"proxy_url": "http://10.0.0.1:8642"}}
        with patch("gateway.run._load_gateway_config", return_value=cfg):
            assert runner._get_proxy_url() == "http://10.0.0.1:8642"

    def test_env_var_overrides_config(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://env-host:8642")
        runner = _make_runner()
        cfg = {"gateway": {"proxy_url": "http://config-host:8642"}}
        with patch("gateway.run._load_gateway_config", return_value=cfg):
            assert runner._get_proxy_url() == "http://env-host:8642"

    def test_empty_string_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "  ")
        runner = _make_runner()
        with patch("gateway.run._load_gateway_config", return_value={}):
            assert runner._get_proxy_url() is None


class TestResolveProxyUrl:
    def test_normalizes_socks_alias_from_all_proxy(self, monkeypatch):
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                    "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:1080/")
        assert resolve_proxy_url() == "socks5://127.0.0.1:1080/"

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

    def test_no_proxy_ignored_without_target(self, monkeypatch):
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                    "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
        monkeypatch.setenv("NO_PROXY", "*")

        assert resolve_proxy_url() == "http://proxy.example:8080"


class TestRunAgentProxyDispatch:
    """Test that _run_agent() delegates to proxy when configured."""

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

    @pytest.mark.asyncio
    async def test_run_agent_skips_proxy_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
        runner = _make_runner()

        runner._run_agent_via_proxy = AsyncMock()

        with patch("gateway.run._load_gateway_config", return_value={}):
            try:
                await runner._run_agent(
                    message="hi",
                    context_prompt="",
                    history=[],
                    source=_make_source(),
                    session_id="test-session",
                )
            except Exception:
                pass  # Expected — bare runner can't create a real agent

        runner._run_agent_via_proxy.assert_not_called()


class TestRunAgentViaProxy:
    """Test the actual proxy HTTP forwarding logic."""

    @pytest.mark.asyncio
    async def test_stamped_process_proxy_run_exposes_interruptible_handle(
        self, monkeypatch
    ):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        runner = _make_runner()
        source = _make_source()
        session_key = "agent:main:matrix:group:test"
        generation = runner._begin_session_run_generation(session_key)
        monkeypatch.setattr(
            runner, "_is_stale_process_notification", lambda _payload: False
        )

        request_started = asyncio.Event()
        hold_request = asyncio.Event()

        class _BlockingResponse(_FakeSSEResponse):
            async def __aenter__(self):
                request_started.set()
                await hold_request.wait()
                return self

        session = _FakeSession(_BlockingResponse())
        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    task = asyncio.create_task(
                        runner._run_agent_via_proxy(
                            message="stale completion",
                            context_prompt="",
                            history=[],
                            source=source,
                            session_id="sess-old",
                            session_key=session_key,
                            run_generation=generation,
                            expected_process_session_id="sess-old",
                        )
                    )
                    await request_started.wait()
                    handle = runner._running_agents[session_key]

                    # Mirrors /new: invalidate before interrupting the concrete
                    # execution handle, so no pending-sentinel gap remains.
                    runner._invalidate_session_run_generation(
                        session_key, reason="test-reset"
                    )
                    handle.interrupt("test reset")
                    with pytest.raises(asyncio.CancelledError):
                        await task

        # Direct backend invocation has no outer gateway turn finalizer; emulate
        # the explicit reset cleanup that owns the published identities.
        runner._release_running_agent_state(session_key)
        assert session_key not in runner._running_agents
        assert session_key not in runner._stamped_process_running_agents

    @pytest.mark.asyncio
    @pytest.mark.parametrize("busy_mode", ["interrupt", "steer"])
    async def test_foreground_input_does_not_interrupt_stamped_proxy_process_turn(
        self, monkeypatch, busy_mode
    ):
        from types import SimpleNamespace

        from gateway.platforms.base import MessageEvent, MessageType

        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
        runner = _make_runner()
        runner._stamped_process_running_agents = {}
        runner._draining = False
        runner._busy_input_mode = busy_mode
        runner._busy_text_mode = "interrupt"
        monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
        source = _make_source()
        adapter = SimpleNamespace(_pending_messages={})
        runner.adapters[source.platform] = adapter
        session_key = "agent:main:matrix:group:!room:server.org:@user:server.org"
        generation = runner._begin_session_run_generation(session_key)
        monkeypatch.setattr(
            runner, "_is_stale_process_notification", lambda _payload: False
        )
        request_started = asyncio.Event()
        hold_request = asyncio.Event()

        class _BlockingResponse(_FakeSSEResponse):
            async def __aenter__(self):
                request_started.set()
                await hold_request.wait()
                return self

        session = _FakeSession(
            _BlockingResponse(
                sse_chunks=[b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n']
            )
        )
        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    proxy_task = asyncio.create_task(
                        runner._run_agent_via_proxy(
                            message="stale completion",
                            context_prompt="",
                            history=[],
                            source=source,
                            session_id="sess-old",
                            session_key=session_key,
                            run_generation=generation,
                            expected_process_session_id="sess-old",
                        )
                    )
                    await request_started.wait()
                    handle = runner._running_agents[session_key]
                    assert runner._stamped_process_running_agents[session_key] is handle
                    foreground = MessageEvent(
                        text="real foreground request",
                        message_type=MessageType.TEXT,
                        source=source,
                    )

                    handled = await runner._handle_active_session_busy_message(
                        foreground, session_key
                    )

                    assert handled is True
                    assert not proxy_task.done()
                    assert adapter._pending_messages[session_key] is foreground
                    hold_request.set()
                    result = await proxy_task

        assert result["final_response"] == "ok"
        # Backend completion is not the turn boundary: ownership remains visible
        # until the outer gateway lifecycle releases the complete synthetic turn.
        handle = runner._running_agents[session_key]
        assert runner._stamped_process_running_agents[session_key] is handle
        runner._release_running_agent_state(session_key)
        assert session_key not in runner._running_agents
        assert session_key not in runner._stamped_process_running_agents

    @pytest.mark.asyncio
    @pytest.mark.parametrize("busy_mode", ["interrupt", "steer"])
    async def test_proxy_backend_completion_keeps_stamped_tail_owned(
        self, monkeypatch, busy_mode
    ):
        from types import SimpleNamespace

        from gateway.platforms.base import MessageEvent, MessageType

        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        runner._stamped_process_running_agents = {}
        runner._draining = False
        runner._busy_input_mode = busy_mode
        runner._busy_text_mode = "interrupt"
        monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
        source = _make_source()
        adapter = SimpleNamespace(_pending_messages={})
        runner.adapters[source.platform] = adapter
        session_key = "agent:main:matrix:group:proxy-tail"
        generation = runner._begin_session_run_generation(session_key)
        monkeypatch.setattr(
            runner, "_is_stale_process_notification", lambda _payload: False
        )
        session = _FakeSession(
            _FakeSSEResponse(
                sse_chunks=[b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n']
            )
        )

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    result = await runner._run_agent_via_proxy(
                        message="stale completion",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="sess-old",
                        session_key=session_key,
                        run_generation=generation,
                        expected_process_session_id="sess-old",
                    )

        assert result["final_response"] == "ok"
        handle = runner._running_agents[session_key]
        assert runner._stamped_process_running_agents[session_key] is handle
        handle.interrupt = MagicMock()
        handle.steer = MagicMock(return_value=True)
        tail_foreground = MessageEvent(
            text="foreground during proxy postprocessing tail",
            message_type=MessageType.TEXT,
            source=source,
        )

        handled = await runner._handle_active_session_busy_message(
            tail_foreground, session_key
        )

        assert handled is True
        handle.interrupt.assert_not_called()
        handle.steer.assert_not_called()
        assert adapter._pending_messages[session_key] is tail_foreground
        runner._release_running_agent_state(session_key)
        assert session_key not in runner._running_agents
        assert session_key not in runner._stamped_process_running_agents

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
                    )

        # Verify request URL
        assert session.captured_url == "http://host:8642/v1/chat/completions"

        # Verify auth header
        assert session.captured_headers["Authorization"] == "Bearer test-key-123"

        # Verify session ID header
        assert session.captured_headers["X-Hermes-Session-Id"] == "session-abc"

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
    async def test_handles_http_error(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(status=401, error_text="Unauthorized: invalid API key")
        session = _FakeSession(resp)

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    result = await runner._run_agent_via_proxy(
                        message="hi",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="test",
                    )

        assert "Proxy error (401)" in result["final_response"]
        assert result["api_calls"] == 0

    @pytest.mark.asyncio
    async def test_handles_connection_error(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://unreachable:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
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
                    )

        assert "Proxy connection error" in result["final_response"]

    @pytest.mark.asyncio
    async def test_rejects_proxy_sse_without_line_boundary_after_buffer_cap(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        monkeypatch.setattr("gateway.run._GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS", 16)
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(status=200, sse_chunks=[b"data: ", b"x" * 20])
        session = _FakeSession(resp)

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    result = await runner._run_agent_via_proxy(
                        message="hi",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="test",
                    )

        assert "Proxy connection error" in result["final_response"]
        assert "exceeded max buffer size" in result["final_response"]
        assert result["api_calls"] == 0

    @pytest.mark.asyncio
    async def test_skips_tool_messages_in_history(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'],
        )
        session = _FakeSession(resp)

        history = [
            {"role": "user", "content": "search for X"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "content": "search results...", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "Found results."},
        ]

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    await runner._run_agent_via_proxy(
                        message="tell me more",
                        context_prompt="",
                        history=history,
                        source=source,
                        session_id="test",
                    )

        # Only user and assistant with content should be forwarded
        messages = session.captured_json["messages"]
        roles = [m["role"] for m in messages]
        assert "tool" not in roles
        # assistant with None content should be skipped
        assert all(m.get("content") for m in messages)

    @pytest.mark.asyncio
    async def test_result_shape_matches_run_agent(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n'],
        )
        session = _FakeSession(resp)

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    result = await runner._run_agent_via_proxy(
                        message="hi",
                        context_prompt="",
                        history=[{"role": "user", "content": "prev"}, {"role": "assistant", "content": "ok"}],
                        source=source,
                        session_id="sess-123",
                    )

        # Required keys that callers depend on
        assert "final_response" in result
        assert result["final_response"] == "answer"
        assert "messages" in result
        assert "api_calls" in result
        assert "tools" in result
        assert "history_offset" in result
        assert result["history_offset"] == 2  # len(history)
        assert "session_id" in result
        assert result["session_id"] == "sess-123"

    @pytest.mark.asyncio
    async def test_proxy_stale_generation_returns_empty_result(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        source = _make_source()
        runner._session_run_generation["test-key"] = 2

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[
                'data: {"choices":[{"delta":{"content":"stale"}}]}\n\n',
                "data: [DONE]\n\n",
            ],
        )
        session = _FakeSession(resp)

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    result = await runner._run_agent_via_proxy(
                        message="hi",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="sess-123",
                        session_key="test-key",
                        run_generation=1,
                    )

        assert result["final_response"] == ""
        assert result["messages"] == []
        assert result["api_calls"] == 0

    @pytest.mark.asyncio
    async def test_no_auth_header_without_key(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
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
                        message="hi",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="test",
                    )

        assert "Authorization" not in session.captured_headers

    @pytest.mark.asyncio
    async def test_no_system_message_when_context_empty(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
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
                    )

        # No system message should appear when context_prompt is empty
        messages = session.captured_json["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"


class TestEnvVarRegistration:
    """Verify GATEWAY_PROXY_URL and GATEWAY_PROXY_KEY are registered."""

    def test_proxy_url_in_optional_env_vars(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "GATEWAY_PROXY_URL" in OPTIONAL_ENV_VARS
        info = OPTIONAL_ENV_VARS["GATEWAY_PROXY_URL"]
        assert info["category"] == "messaging"
        assert info["password"] is False

    def test_proxy_key_in_optional_env_vars(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "GATEWAY_PROXY_KEY" in OPTIONAL_ENV_VARS
        info = OPTIONAL_ENV_VARS["GATEWAY_PROXY_KEY"]
        assert info["category"] == "messaging"
        assert info["password"] is True
