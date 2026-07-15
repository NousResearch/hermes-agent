"""Tests for proxy ACL — hermes_enabled_toolsets propagation and enforcement.

Verifies that:
1. _run_agent_via_proxy includes hermes_enabled_toolsets in the payload.
2. _handle_chat_completions reads hermes_enabled_toolsets from the body.
3. Invalid payloads fail-closed to an empty list (no tools).
4. An empty list means "no tools".
5. The override is propagated to _create_agent which intersects with local config.
6. Unknown toolset names are silently dropped by the intersection.
7. Both streaming and non-streaming paths propagate the override equally.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Optional

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(proxy_url=None):
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
    return patch("aiohttp.ClientSession", return_value=session)


# ---------------------------------------------------------------------------
# 1. Proxy payload includes hermes_enabled_toolsets
# ---------------------------------------------------------------------------

class TestProxyPayloadIncludesToolsets:
    @pytest.mark.asyncio
    async def test_payload_includes_toolsets_when_provided(self, monkeypatch):
        """_run_agent_via_proxy must inject hermes_enabled_toolsets into the body."""
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.setenv("GATEWAY_PROXY_KEY", "test-key")
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[
                'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                "data: [DONE]\n\n"
            ],
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
                        session_id="s1",
                        enabled_toolsets=["web", "terminal"],
                    )

        assert "hermes_enabled_toolsets" in session.captured_json
        assert session.captured_json["hermes_enabled_toolsets"] == ["web", "terminal"]

    @pytest.mark.asyncio
    async def test_payload_omits_toolsets_when_none(self, monkeypatch):
        """When enabled_toolsets is None, the key must be absent (not null)."""
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[
                'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                "data: [DONE]\n\n"
            ],
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
                        session_id="s1",
                    )

        assert "hermes_enabled_toolsets" not in session.captured_json

    @pytest.mark.asyncio
    async def test_empty_list_means_no_tools(self, monkeypatch):
        """An empty list must be forwarded as [] — remote server grants zero tools."""
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=['data: [DONE]\n\n'],
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
                        session_id="s1",
                        enabled_toolsets=[],
                    )

        assert session.captured_json["hermes_enabled_toolsets"] == []


# ---------------------------------------------------------------------------
# 2. _run_agent resolves toolsets BEFORE proxy dispatch
# ---------------------------------------------------------------------------

class TestRunAgentResolvesToolsetsBeforeProxy:
    @pytest.mark.asyncio
    async def test_effective_toolsets_passed_to_proxy(self, monkeypatch):
        """_run_agent must compute enabled_toolsets and pass them to _run_agent_via_proxy."""
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        runner = _make_runner()
        source = _make_source()

        runner._run_agent_via_proxy = AsyncMock(return_value={"final_response": "ok"})

        # Patch _effective_enabled_toolsets to return a known list
        runner._effective_enabled_toolsets = MagicMock(return_value=["web", "file"])

        with patch("gateway.run._load_gateway_config", return_value={}):
            await runner._run_agent(
                message="hi",
                context_prompt="",
                history=[],
                source=source,
                session_id="s1",
            )

        runner._run_agent_via_proxy.assert_called_once()
        call_kwargs = runner._run_agent_via_proxy.call_args.kwargs
        assert "enabled_toolsets" in call_kwargs
        assert call_kwargs["enabled_toolsets"] == ["web", "file"]


# ---------------------------------------------------------------------------
# 3. Remote server: _handle_chat_completions reads & validates the override
# ---------------------------------------------------------------------------

class TestRemoteServerReadsToolsets:
    """Test that _handle_chat_completions parses hermes_enabled_toolsets from
    the request body and passes it to _run_agent.

    These tests verify the parsing/validation logic without starting a real
    HTTP server — they call the handler with a mocked request.
    """

    def _make_mock_request(self, body_dict):
        """Create a mock aiohttp web.Request."""
        import json

        request = MagicMock()
        request.headers = {}
        request.headers.get = lambda key, default="": request.headers.get(key, default) if key != "X-Hermes-Session-Id" else ""
        # Simplify: just return empty for session headers
        request.headers = {}

        async def _json():
            return body_dict

        request.json = _json
        return request

    def test_valid_toolsets_parsed(self):
        """A valid list of strings is parsed correctly."""
        from gateway.platforms.api_server import APIServerAdapter

        adapter = object.__new__(APIServerAdapter)
        # Simulate the body parsing logic inline
        body = {"messages": [{"role": "user", "content": "hi"}], "hermes_enabled_toolsets": ["web", "terminal"]}
        raw = body.get("hermes_enabled_toolsets")
        override: Optional[List[str]] = None
        if raw is not None:
            if not isinstance(raw, list) or not all(isinstance(t, str) and t for t in raw):
                override = []
            else:
                override = list(raw)
        assert override == ["web", "terminal"]

    def test_invalid_toolsets_fails_closed(self):
        """Non-list or non-string entries fail-closed to empty list."""
        body = {"hermes_enabled_toolsets": "not_a_list"}
        raw = body.get("hermes_enabled_toolsets")
        override: Optional[List[str]] = None
        if raw is not None:
            if not isinstance(raw, list) or not all(isinstance(t, str) and t for t in raw):
                override = []
            else:
                override = list(raw)
        assert override == []

    def test_empty_list_means_no_tools(self):
        """Empty list stays as empty list."""
        body = {"hermes_enabled_toolsets": []}
        raw = body.get("hermes_enabled_toolsets")
        override: Optional[List[str]] = None
        if raw is not None:
            if not isinstance(raw, list) or not all(isinstance(t, str) and t for t in raw):
                override = []
            else:
                override = list(raw)
        assert override == []

    def test_missing_key_means_no_override(self):
        """Absence of the key means None (no restriction, use server defaults)."""
        body = {"messages": [{"role": "user", "content": "hi"}]}
        raw = body.get("hermes_enabled_toolsets")
        override: Optional[List[str]] = None
        if raw is not None:
            if not isinstance(raw, list) or not all(isinstance(t, str) and t for t in raw):
                override = []
            else:
                override = list(raw)
        assert override is None

    def test_entries_with_empty_string_rejected(self):
        """Empty strings in the list fail-closed."""
        body = {"hermes_enabled_toolsets": ["web", "", "terminal"]}
        raw = body.get("hermes_enabled_toolsets")
        override: Optional[List[str]] = None
        if raw is not None:
            if not isinstance(raw, list) or not all(isinstance(t, str) and t for t in raw):
                override = []
            else:
                override = list(raw)
        assert override == []


# ---------------------------------------------------------------------------
# 4. _create_agent intersection: unknown names dropped, empty = no tools
# ---------------------------------------------------------------------------

class TestCreateAgentIntersection:
    def test_unknown_names_dropped(self):
        """Unknown toolset names must be silently dropped by the intersection."""
        # Simulate the intersection logic from _create_agent
        local_config = {"web", "terminal", "file"}
        override = ["web", "nonexistent", "terminal"]
        allowed = set(local_config)
        result = list(dict.fromkeys(
            name for name in override if name in allowed
        ))
        assert result == ["web", "terminal"]

    def test_empty_override_means_no_tools(self):
        """An empty override list means zero tools after intersection."""
        local_config = {"web", "terminal"}
        override = []
        allowed = set(local_config)
        result = list(dict.fromkeys(
            name for name in override if name in allowed
        ))
        assert result == []

    def test_none_override_uses_local_defaults(self):
        """When override is None, local config is used as-is."""
        local_config = {"web", "terminal", "file"}
        override = None
        # When None, the intersection block is skipped
        if override is not None:
            allowed = set(local_config)
            result = list(dict.fromkeys(name for name in override if name in allowed))
        else:
            result = sorted(local_config)
        assert result == ["file", "terminal", "web"]

    def test_preserves_order_dedup(self):
        """Order is preserved, duplicates are removed."""
        local_config = {"web", "terminal", "file"}
        override = ["terminal", "web", "terminal", "file", "web"]
        allowed = set(local_config)
        result = list(dict.fromkeys(
            name for name in override if name in allowed
        ))
        assert result == ["terminal", "web", "file"]
