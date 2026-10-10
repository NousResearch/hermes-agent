"""Tests for gateway proxy mode — forwarding messages to a remote API server."""

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.platforms.base import resolve_proxy_url
from gateway.run import GatewayRunner
from gateway.session import SessionSource

# Reused harness from the streaming / queued-delivery suites: a recording adapter, the fuller
# runner the completion seam needs, and ``_completion_seam`` — the helper that runs a result
# through ``_hmwa_deliver_turn_response`` and returns the text the caller would send (None =
# already delivered). The streamed-proxy tests below need all three.
from tests.gateway.test_queued_followup_final_delivery import _completion_seam
from tests.gateway.test_run_progress_topics import (
    MetadataEditProgressCaptureAdapter,
    _make_runner as _make_streaming_runner,
)

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
    """Install the optional aiohttp boundary without requiring the extra."""
    module = types.SimpleNamespace(
        ClientSession=MagicMock(return_value=session),
        ClientTimeout=MagicMock(),
    )
    return patch.dict("sys.modules", {"aiohttp": module})

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

class _SelectiveScope(dict):
    """Bound scope that resolves GATEWAY_PROXY_URL but fails on the KEY read."""
    def get(self, name, default=None):
        if name == "GATEWAY_PROXY_URL":
            return "http://proxy.local:8642"
        if name == "GATEWAY_PROXY_KEY":
            raise RuntimeError("resolver boom")
        return dict.get(self, name, default)


class TestProxyKeyScopeFailure:
    """The proxy key read must propagate a bound-scope failure -- the ambient env
    may hold another profile's credential (pre-fix: ``except Exception -> os.getenv``)."""

    @pytest.mark.asyncio
    async def test_proxy_key_scope_failure_never_borrows_env(self, monkeypatch):
        from agent import secret_scope as ss

        monkeypatch.setenv("GATEWAY_PROXY_KEY", "foreign-key")
        runner = _make_runner()
        runner._run_still_current_fn = lambda *a, **k: True

        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(_SelectiveScope())
        try:
            with _patch_aiohttp(MagicMock()):
                with pytest.raises(RuntimeError, match="resolver boom"):
                    await runner._run_agent_via_proxy("hi", "ctx", [], _make_source(), "sess-1")
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)


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

@pytest.mark.platforms("macos")
class TestMacosProxyProbeCache:
    """``scutil --proxy`` is a ~11 ms fork and resolve_proxy_url runs it on the SEND path —
    per chunk of an outbound message and per media attachment."""

    SCUTIL_OUT = "<dictionary> {\n  HTTPEnable : 1\n  HTTPProxy : 10.0.0.1\n  HTTPPort : 3128\n}"

    @pytest.fixture(autouse=True)
    def _isolate(self):
        from gateway.platforms import base
        base.reset_macos_proxy_cache()
        yield
        base.reset_macos_proxy_cache()

    def _count_forks(self, monkeypatch):
        from gateway.platforms import base
        calls = []

        def fake(*a, **kw):
            calls.append(a)
            return self.SCUTIL_OUT
        monkeypatch.setattr(base.subprocess, "check_output", fake)
        return base, calls

    def test_repeated_probes_fork_scutil_once(self, monkeypatch):
        base, calls = self._count_forks(monkeypatch)
        results = [base._detect_macos_system_proxy() for _ in range(10)]
        assert len(calls) == 1, f"expected 1 scutil fork for 10 probes, got {len(calls)}"
        assert results == ["http://10.0.0.1:3128"] * 10

    def test_expired_ttl_re_reads(self, monkeypatch):
        base, calls = self._count_forks(monkeypatch)
        clock = {"t": 1000.0}
        monkeypatch.setattr(base.time, "monotonic", lambda: clock["t"])
        base._detect_macos_system_proxy()
        clock["t"] += base._MACOS_PROXY_TTL_SECONDS + 1
        base._detect_macos_system_proxy()
        assert len(calls) == 2


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
            with _patch_aiohttp(_ErrorSession()):
                with patch("aiohttp.ClientTimeout"):
                    result = await runner._run_agent_via_proxy(
                        message="hi",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="test",
                    )

        assert "Connection refused" in result["final_response"]
        assert result["api_calls"] == 0

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

class TestStreamingResilience:
    """Tests for SSE streaming robustness — hang avoidance and malformed-chunk tolerance."""

    @pytest.mark.asyncio
    async def test_done_marker_stops_reading_trailing_chunks(self, monkeypatch):
        """After `[DONE]`, no further SSE chunks must be processed.

        A buggy upstream that holds the connection open and streams more
        chunks after `[DONE]` should not leak those chunks into the
        response. Regression test for the inner `break` that only exited
        the line-parse loop, leaving the outer chunk loop to keep reading
        until sock_read timeout.
        """
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        source = _make_source()

        # Content → [DONE] → MORE content. The trailing chunk must be
        # dropped. With the pre-fix code it would be appended to
        # full_response, since `break` only exited the inner loop.
        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[
                'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
                'data: [DONE]\n',
                'data: {"choices":[{"delta":{"content":" IGNORED"}}]}\n',
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
                        session_id="test",
                    )

        assert result["final_response"] == "Hello"

    @pytest.mark.asyncio
    async def test_residual_buffer_flushed_after_eof(self, monkeypatch):
        """A final SSE frame without a trailing newline must not be dropped.

        The line loop only consumes complete lines; if the upstream's last
        frame lacks the newline, its content sat in ``buffer`` at EOF and
        was silently discarded (pi#8997's bug class). The residual buffer
        is now flushed after the read loop.
        """
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[
                'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
                'data: {"choices":[{"delta":{"content":" world"}}]}',  # no newline, then EOF
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
                        session_id="test",
                    )

        assert result["final_response"] == "Hello world"

    @pytest.mark.asyncio
    async def test_eof_without_done_and_no_content_is_an_error(self, monkeypatch):
        """Clean EOF with no [DONE] and no content must surface an error, not
        an empty 'response'. With content, the partial text is kept (and the
        truncation is logged) rather than thrown away."""
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(status=200, sse_chunks=[])
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

        assert result["final_response"]
        assert result["api_calls"] == 0

    @pytest.mark.asyncio
    async def test_client_timeout_sets_sock_connect(self, monkeypatch):
        """ClientTimeout must bound the TCP connect phase.

        Without an explicit ``sock_connect``, an unreachable proxy host
        hangs for the OS default (minutes) before failing. The fix sets
        a short connect cap so the gateway surfaces the error quickly.
        """
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(status=200, sse_chunks=['data: [DONE]\n'])
        session = _FakeSession(resp)

        captured = {}

        def _capture_timeout(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout", side_effect=_capture_timeout):
                    await runner._run_agent_via_proxy(
                        message="hi",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="test",
                    )

        assert "sock_connect" in captured, (
            "ClientTimeout should set sock_connect to bound TCP connect"
        )
        assert 0 < captured["sock_connect"] <= 60, (
            f"sock_connect should be a short, reasonable cap — got {captured['sock_connect']}"
        )

    @pytest.mark.asyncio
    async def test_malformed_chunk_is_skipped_not_fatal(self, monkeypatch):
        """One bad SSE chunk must not abort the whole stream.

        Pre-fix: `choices[0].get(...)` raised ``AttributeError`` when
        ``choices[0]`` was ``None``, escaping the narrow
        ``except json.JSONDecodeError`` and bubbling to the outer
        ``except Exception`` which returned whatever partial response
        was accumulated. All later chunks were lost.

        Post-fix: type guards + broader exception handling skip the bad
        chunk and keep parsing.
        """
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[
                'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
                'data: {"choices":[null]}\n',
                'data: {"choices":"wrong-type"}\n',
                'data: {"choices":[{"delta":"wrong-type"}]}\n',
                'data: {"choices":[{"delta":{"content":" world"}}]}\n',
                'data: [DONE]\n',
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
                        session_id="test",
                    )

        assert result["final_response"] == "Hello world"


# ---------------------------------------------------------------------------------------------
# Streamed proxy turns reconcile the streamed body before the normal final send.
#
# The proxy lane builds its OWN GatewayStreamConsumer when streaming is enabled
# (``_proxy_stream_consumer``) and feeds it the SSE deltas. ``_run_agent_via_proxy`` then calls
# ``_run_agent_mark_streamed_delivery`` so the lane behaves like the local ``_run_agent_inner``
# path: when the consumer already delivered the final body the result is marked ``already_sent``
# so the completion seam (``_hmwa_deliver_turn_response``) does not send the same text twice, and
# a response-filter replacement that arrived after the body streamed is written back over the
# streamed message instead of being lost.
#
# Every other test in this file leaves the runner on the default ``StreamingConfig()`` (whose
# ``globally_enabled`` is False), so ``_proxy_stream_consumer`` returns None and none of this
# executes. These tests supply a consumer (streaming on) and pin both directions: streamed
# (suppressed / edited) and unstreamed (the completion seam owns the final send).


class _ResponseRedactor:
    """Subscriber offering a replacement for every outbound response text."""

    async def emit_collect(self, event_type, context):
        if event_type == "agent:response:filter":
            return [{"response": "redacted answer"}]
        return []


def _sse_streamed(*parts):
    """SSE body streaming ``parts`` in order, then the terminal ``[DONE]``."""
    frames = "".join(
        'data: {"choices":[{"delta":{"content":"%s"}}]}\n\n' % part for part in parts
    )
    return _FakeSSEResponse(status=200, sse_chunks=[frames + "data: [DONE]\n\n"])


def _proxy_runner(adapter, *, streaming):
    """Proxy runner whose config decides whether ``_proxy_stream_consumer`` builds one."""
    runner = _make_streaming_runner(adapter)
    runner.config.streaming = StreamingConfig(
        enabled=streaming, edit_interval=0.01, buffer_threshold=1,
    )
    return runner


async def _run_streamed_proxy(monkeypatch, runner, source, *parts):
    monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
    monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
    session = _FakeSession(_sse_streamed(*parts))
    with patch("gateway.run._load_gateway_config", return_value={}):
        with _patch_aiohttp(session):
            with patch("aiohttp.ClientTimeout"):
                return await runner._run_agent_via_proxy(
                    message="hi", context_prompt="", history=[], source=source,
                    session_id="s-stream-proxy",
                )


@pytest.mark.asyncio
async def test_streamed_proxy_turn_is_not_sent_twice(monkeypatch):
    """The proxy's own stream consumer delivered the body: the completion seam must not resend it."""
    adapter = MetadataEditProgressCaptureAdapter(platform=Platform.MATRIX)
    runner = _proxy_runner(adapter, streaming=True)
    source = _make_source()

    result = await _run_streamed_proxy(monkeypatch, runner, source, "Hello", " world")

    assert result["final_response"] == "Hello world"
    # Adapter-visible delivery: the body reached the user through the lane's consumer.
    delivered = [c["content"] for c in adapter.sent] + [e["content"] for e in adapter.edits]
    assert any("Hello world" in text for text in delivered), delivered
    # ...and the completion seam sends nothing more.
    assert result.get("already_sent") is True
    sends_before = list(adapter.sent)
    assert await _completion_seam(adapter, result, result["final_response"]) is None
    assert adapter.sent == sends_before


@pytest.mark.asyncio
async def test_proxy_turn_without_streaming_defers_to_the_completion_send(monkeypatch):
    """Streaming off (the default): no consumer, so the completion seam owns the final send."""
    adapter = MetadataEditProgressCaptureAdapter(platform=Platform.MATRIX)
    runner = _proxy_runner(adapter, streaming=False)
    source = _make_source()

    result = await _run_streamed_proxy(monkeypatch, runner, source, "Hello", " world")

    assert result["final_response"] == "Hello world"
    assert not result.get("already_sent")
    assert adapter.sent == [] and adapter.edits == []
    assert await _completion_seam(adapter, result, result["final_response"]) == "Hello world"


@pytest.mark.asyncio
async def test_streamed_proxy_turn_delivers_the_response_filter_replacement(monkeypatch):
    """A replacement applied after the body streamed must reach the user, not be dropped."""
    adapter = MetadataEditProgressCaptureAdapter(platform=Platform.MATRIX)
    runner = _proxy_runner(adapter, streaming=True)
    runner.hooks = _ResponseRedactor()
    source = _make_source()

    result = await _run_streamed_proxy(monkeypatch, runner, source, "raw", " answer")

    assert result["final_response"] == "redacted answer"
    visible = [c["content"] for c in adapter.sent] + [e["content"] for e in adapter.edits]
    assert "redacted answer" in visible, visible
    assert await _completion_seam(adapter, result, result["final_response"]) is None
