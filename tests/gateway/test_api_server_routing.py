"""Tests for the api_server adapter's multi-agent routing wiring.

Covers ``APIServerAdapter._resolve_agent_profile`` — the bridge between
inbound ``X-Hermes-Chat-Id`` (etc.) headers and the shared
``_attach_agent_id`` resolver / ``AgentProfile`` registry.

These tests construct the adapter in-process (no HTTP listener) and feed
it ``MagicMock``-backed ``aiohttp.web.Request`` stand-ins, mirroring the
style used elsewhere in ``test_api_server.py``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from agent.profile import AgentProfile, DEFAULT_AGENT_ID, get_active_profile
from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(
    *,
    routes=None,
    registry=None,
    default_agent: str = "main",
) -> APIServerAdapter:
    """Build an APIServerAdapter wired to *routes* and *registry*.

    Mirrors what ``GatewayRunner`` does at startup: calls
    ``set_routing_context`` with a fake gateway that exposes
    ``_agent_registry``.
    """
    cfg = PlatformConfig()
    adapter = APIServerAdapter(cfg)

    fake_gateway = MagicMock()
    fake_gateway._agent_registry = registry or {}
    adapter.set_routing_context(
        routes=routes or [],
        default_agent=default_agent,
        gateway=fake_gateway,
    )
    return adapter


def _request_with_headers(headers: dict) -> MagicMock:
    """Stub aiohttp.web.Request — only ``.headers.get`` is exercised."""
    req = MagicMock()
    req.headers = headers
    return req


# ---------------------------------------------------------------------------
# Header sanitisation
# ---------------------------------------------------------------------------


class TestReadRoutingHeader:
    def test_returns_value_when_present(self):
        adapter = _make_adapter()
        req = _request_with_headers({"X-Hermes-Chat-Id": "calendar-propose"})
        assert adapter._read_routing_header(req, "X-Hermes-Chat-Id") == "calendar-propose"

    def test_returns_none_when_absent(self):
        adapter = _make_adapter()
        req = _request_with_headers({})
        assert adapter._read_routing_header(req, "X-Hermes-Chat-Id") is None

    def test_strips_whitespace(self):
        adapter = _make_adapter()
        req = _request_with_headers({"X-Hermes-Chat-Id": "  coder  "})
        assert adapter._read_routing_header(req, "X-Hermes-Chat-Id") == "coder"

    def test_rejects_crlf_injection(self):
        adapter = _make_adapter()
        req = _request_with_headers({"X-Hermes-Chat-Id": "ok\r\nX-Injected: yes"})
        assert adapter._read_routing_header(req, "X-Hermes-Chat-Id") is None

    def test_rejects_overlong_value(self):
        adapter = _make_adapter()
        req = _request_with_headers({"X-Hermes-Chat-Id": "x" * 1024})
        assert adapter._read_routing_header(req, "X-Hermes-Chat-Id") is None


# ---------------------------------------------------------------------------
# _resolve_agent_profile — core routing wiring
# ---------------------------------------------------------------------------


class TestResolveAgentProfile:
    def test_header_match_routes_to_specified_agent(self, tmp_path):
        calendar = AgentProfile(id="calendar-propose", home_dir=tmp_path / "calendar")
        main = AgentProfile(id="main")
        registry = {"main": main, "calendar-propose": calendar}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "calendar-propose"}, "agent": "calendar-propose"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({"X-Hermes-Chat-Id": "calendar-propose"})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "calendar-propose"
        assert profile is calendar

    def test_no_header_falls_through_to_default_agent(self):
        main = AgentProfile(id="main")
        calendar = AgentProfile(id="calendar-propose")
        registry = {"main": main, "calendar-propose": calendar}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "calendar-propose"}, "agent": "calendar-propose"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "main"
        assert profile is main

    def test_unmatched_header_falls_through_to_default(self):
        main = AgentProfile(id="main")
        calendar = AgentProfile(id="calendar-propose")
        registry = {"main": main, "calendar-propose": calendar}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "calendar-propose"}, "agent": "calendar-propose"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({"X-Hermes-Chat-Id": "unknown-agent"})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "main"
        assert profile is main

    def test_platform_only_route_matches_any_request(self):
        """A bare ``platform: api_server`` route is the catch-all."""
        coder = AgentProfile(id="coder")
        main = AgentProfile(id="main")
        registry = {"main": main, "coder": coder}
        routes = [
            {"match": {"platform": "api_server"}, "agent": "coder"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        # No headers — platform alone matches.
        req = _request_with_headers({})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "coder"
        assert profile is coder

    def test_user_id_route_matches(self):
        vip = AgentProfile(id="vip")
        main = AgentProfile(id="main")
        registry = {"main": main, "vip": vip}
        routes = [
            {"match": {"platform": "api_server", "user_id": "alice"}, "agent": "vip"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({"X-Hermes-User-Id": "alice"})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "vip"
        assert profile is vip

    def test_thread_id_route_matches(self):
        thread_agent = AgentProfile(id="thread-agent")
        main = AgentProfile(id="main")
        registry = {"main": main, "thread-agent": thread_agent}
        routes = [
            {"match": {"platform": "api_server", "thread_id": "T-42"}, "agent": "thread-agent"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({"X-Hermes-Thread-Id": "T-42"})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "thread-agent"
        assert profile is thread_agent

    def test_more_specific_route_wins_when_declared_first(self):
        coder = AgentProfile(id="coder")
        general = AgentProfile(id="general")
        main = AgentProfile(id="main")
        registry = {"main": main, "coder": coder, "general": general}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "code", "user_id": "alice"}, "agent": "coder"},
            {"match": {"platform": "api_server", "chat_id": "code"}, "agent": "general"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({
            "X-Hermes-Chat-Id": "code",
            "X-Hermes-User-Id": "alice",
        })

        profile, agent_id = adapter._resolve_agent_profile(req)
        assert agent_id == "coder"
        assert profile is coder

    def test_empty_registry_returns_none_profile(self):
        """Legacy single-agent install: registry empty → profile is None."""
        adapter = _make_adapter(routes=[], registry={})
        req = _request_with_headers({})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "main"
        assert profile is None

    def test_crlf_header_is_treated_as_absent(self):
        """A header with control chars must not bypass routing."""
        main = AgentProfile(id="main")
        calendar = AgentProfile(id="calendar-propose")
        registry = {"main": main, "calendar-propose": calendar}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "calendar-propose"}, "agent": "calendar-propose"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({
            "X-Hermes-Chat-Id": "calendar-propose\r\nX-Injected: yes",
        })

        profile, agent_id = adapter._resolve_agent_profile(req)

        # Tainted header dropped → no match → default agent.
        assert agent_id == "main"
        assert profile is main

    def test_gateway_ref_missing_does_not_raise(self):
        """No gateway wired (single-agent install never calls
        set_routing_context with a gateway): profile resolution must
        still succeed and return ``None``."""
        cfg = PlatformConfig()
        adapter = APIServerAdapter(cfg)
        # No set_routing_context call: routes and gateway_ref remain unset.
        req = _request_with_headers({"X-Hermes-Chat-Id": "anything"})

        profile, agent_id = adapter._resolve_agent_profile(req)

        # No registry → no profile, default agent_id falls back to "main".
        assert profile is None
        assert agent_id == "main"


# ---------------------------------------------------------------------------
# ContextVar isolation under concurrency
# ---------------------------------------------------------------------------


class TestContextVarIsolation:
    """Two concurrent requests with different chat_ids must route to
    independent agents without leaking state."""

    def test_concurrent_use_profile_does_not_cross_contaminate(self):
        """Verify ``use_profile`` honours asyncio task isolation.

        This is the foundation the api_server adapter relies on when
        wrapping concurrent agent runs.  If this test fails, the patch
        does not actually isolate requests.
        """
        from agent.profile import use_profile

        code = AgentProfile(id="code", home_dir="/tmp/code")
        chat = AgentProfile(id="chat", home_dir="/tmp/chat")

        observed = {"code": None, "chat": None}

        async def _under_profile(name: str, profile: AgentProfile, hold: float) -> None:
            with use_profile(profile):
                # Yield to the other task — if profiles leak we'll see it.
                await asyncio.sleep(hold)
                observed[name] = get_active_profile()

        async def _main() -> None:
            await asyncio.gather(
                _under_profile("code", code, 0.05),
                _under_profile("chat", chat, 0.01),
            )

        asyncio.run(_main())

        assert observed["code"] is code
        assert observed["chat"] is chat
        # Outer context is restored.
        assert get_active_profile() is None

    def test_concurrent_resolve_returns_independent_profiles(self, tmp_path):
        """Run ``_resolve_agent_profile`` from two parallel tasks with
        different headers and assert each task sees its own profile."""
        code = AgentProfile(id="code", home_dir=tmp_path / "code")
        chat = AgentProfile(id="chat", home_dir=tmp_path / "chat")
        main = AgentProfile(id="main")
        registry = {"main": main, "code": code, "chat": chat}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "code"}, "agent": "code"},
            {"match": {"platform": "api_server", "chat_id": "chat"}, "agent": "chat"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)

        from agent.profile import use_profile

        observed = {}

        async def _one(label: str, header_value: str, hold: float) -> None:
            req = _request_with_headers({"X-Hermes-Chat-Id": header_value})
            profile, agent_id = adapter._resolve_agent_profile(req)
            with use_profile(profile):
                await asyncio.sleep(hold)
                observed[label] = (agent_id, get_active_profile())

        async def _main() -> None:
            await asyncio.gather(
                _one("a", "code", 0.05),
                _one("b", "chat", 0.01),
            )

        asyncio.run(_main())

        assert observed["a"] == ("code", code)
        assert observed["b"] == ("chat", chat)


# ---------------------------------------------------------------------------
# Backward compatibility — adapter unchanged for legacy callers
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_routing_constants_match_documented_headers(self):
        """Header names are part of the public contract — guard against
        silent renames."""
        assert APIServerAdapter._AGENT_CHAT_ID_HEADER == "X-Hermes-Chat-Id"
        assert APIServerAdapter._AGENT_USER_ID_HEADER == "X-Hermes-User-Id"
        assert APIServerAdapter._AGENT_THREAD_ID_HEADER == "X-Hermes-Thread-Id"

    def test_legacy_no_routes_no_agents_returns_main(self):
        """Existing single-agent installs: no agents config, no routes →
        every request resolves to ``main`` with a ``None`` profile (i.e.
        the legacy ``HERMES_HOME`` env-driven path)."""
        adapter = _make_adapter(routes=[], registry={})
        req = _request_with_headers({"X-Hermes-Chat-Id": "ignored"})
        profile, agent_id = adapter._resolve_agent_profile(req)
        assert agent_id == "main"
        assert profile is None

    def test_default_agent_id_constant_is_main(self):
        """Anchor the default-agent contract."""
        assert DEFAULT_AGENT_ID == "main"
