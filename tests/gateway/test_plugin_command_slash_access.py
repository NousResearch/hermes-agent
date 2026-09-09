"""Gateway plugin slash commands: slash-access gating + PluginInvocation context.

Drives the real ``GatewayRunner._hm_dispatch_quick_and_plugin_commands`` sink
(object.__new__ construction, same pattern as test_slash_access_dispatch.py) so
the actual gate and the surface-side context builder are exercised.

Coverage targets:
  - Backward compat: no ``allow_admin_from`` for the scope → no gate, handler runs.
  - Admin runs any plugin command.
  - Non-admin denied unless the command is in ``user_allowed_commands``.
  - Plugin commands receive an immutable gateway-bound PluginInvocation (surface,
    platform, session identity); fresh sessions (no agent) fail closed.
  - A session with a usable cached agent authorizes dispatch against the agent's
    effective tool set, and dispatch outside that set is refused.
  - Legacy one-argument handlers are still called with exactly one argument.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def _make_source(
    *,
    platform: Platform = Platform.DISCORD,
    user_id: str = "user1",
    chat_type: str = "dm",
    chat_id: str = "c1",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name=f"name-{user_id}",
        chat_type=chat_type,
    )


def _make_event(text: str, source: SessionSource) -> MessageEvent:
    return MessageEvent(text=text, source=source, message_id="m1")


def _make_runner(*, platform_extra: dict | None = None,
                 platform: Platform = Platform.DISCORD):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            platform: PlatformConfig(
                enabled=True,
                token="***",
                extra=platform_extra or {},
            )
        }
    )
    runner.adapters = {platform: SimpleNamespace(send=None)}
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_sources = {}
    return runner


@pytest.fixture
def plugin_handler_patch(monkeypatch):
    """Install a fake ``get_plugin_command_handler`` resolving name ``plug``."""

    def install(fake):
        monkeypatch.setattr(
            "hermes_cli.plugins.get_plugin_command_handler",
            lambda name: fake if name == "plug" else None,
        )

    return install


@pytest.mark.asyncio
async def test_plugin_command_denied_for_non_admin_when_gating_enabled(plugin_handler_patch):
    runner = _make_runner(platform_extra={"allow_admin_from": ["admin1"]})
    calls = []
    plugin_handler_patch(lambda args: calls.append(args) or "should not run")
    source = _make_source(user_id="user1")
    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
        _make_event("/plug go", source), source, "plug")
    assert handled is True
    assert command == "plug"
    assert "admin-only" in (result or "")
    assert calls == []


@pytest.mark.asyncio
async def test_plugin_command_allowed_for_admin(plugin_handler_patch):
    runner = _make_runner(platform_extra={"allow_admin_from": ["admin1"]})
    plugin_handler_patch(lambda args: f"ran:{args}")
    source = _make_source(user_id="admin1")
    handled, result, _command = await runner._hm_dispatch_quick_and_plugin_commands(
        _make_event("/plug go", source), source, "plug")
    assert handled is True
    assert result == "ran:go"


@pytest.mark.asyncio
async def test_plugin_command_allowed_for_non_admin_in_user_allowed_commands(plugin_handler_patch):
    runner = _make_runner(platform_extra={
        "allow_admin_from": ["admin1"], "user_allowed_commands": ["plug"]})
    plugin_handler_patch(lambda args: f"ran:{args}")
    source = _make_source(user_id="user1")
    handled, result, _command = await runner._hm_dispatch_quick_and_plugin_commands(
        _make_event("/plug go", source), source, "plug")
    assert handled is True
    assert result == "ran:go"


@pytest.mark.asyncio
async def test_plugin_command_ungated_when_no_admin_listed(plugin_handler_patch):
    # No allow_admin_from for the scope → policy disabled → exact pre-gate behavior.
    runner = _make_runner(platform_extra={})
    plugin_handler_patch(lambda args: f"ran:{args}")
    source = _make_source(user_id="user1")
    handled, result, _command = await runner._hm_dispatch_quick_and_plugin_commands(
        _make_event("/plug go", source), source, "plug")
    assert handled is True
    assert result == "ran:go"


@pytest.mark.asyncio
async def test_plugin_command_receives_gateway_invocation(plugin_handler_patch):
    runner = _make_runner(platform_extra={"allow_admin_from": ["admin1"]})
    source = _make_source(user_id="admin1", chat_type="dm", chat_id="c1")
    received = {}

    def context_aware(args, *, invocation=None):
        received["args"] = args
        received["invocation"] = invocation
        return "context-ok"

    plugin_handler_patch(context_aware)
    handled, result, _command = await runner._hm_dispatch_quick_and_plugin_commands(
        _make_event("/plug alpha", source), source, "plug")
    assert handled is True
    assert result == "context-ok"
    assert received["args"] == "alpha"
    invocation = received["invocation"]
    assert invocation is not None
    assert invocation.surface == "gateway"
    assert invocation.platform == "discord"
    assert invocation.session_key == runner._session_key_for_source(source)
    assert invocation.authorized is False  # fresh session: no agent yet → fail closed
    assert invocation.tool_names == frozenset()
    with pytest.raises(PermissionError, match="not authorized"):
        invocation.dispatch_tool("read_file", {"path": "x"})


@pytest.mark.asyncio
async def test_legacy_plugin_handler_keeps_exact_one_argument_contract(plugin_handler_patch):
    runner = _make_runner(platform_extra={"allow_admin_from": ["admin1"]})
    source = _make_source(user_id="admin1")
    received = []

    def legacy(args):
        received.append(args)
        return "legacy-ok"

    plugin_handler_patch(legacy)
    handled, result, _command = await runner._hm_dispatch_quick_and_plugin_commands(
        _make_event("/plug beta", source), source, "plug")
    assert handled is True
    assert result == "legacy-ok"
    assert received == ["beta"]


class _FakeAgent:
    session_id = "sess-9"
    platform = "discord"
    valid_tool_names = ["read_file", "terminal"]


@pytest.mark.asyncio
async def test_plugin_command_invocation_authorized_with_cached_agent(plugin_handler_patch):
    runner = _make_runner(platform_extra={"allow_admin_from": ["admin1"]})
    source = _make_source(user_id="admin1")
    quick_key = runner._session_key_for_source(source)
    runner._agent_cache[quick_key] = (_FakeAgent(), "signature")
    received = {}

    def context_aware(args, *, invocation=None):
        received["invocation"] = invocation
        return "auth-ok"

    plugin_handler_patch(context_aware)
    handled, result, _command = await runner._hm_dispatch_quick_and_plugin_commands(
        _make_event("/plug gamma", source), source, "plug")
    assert handled is True
    assert result == "auth-ok"
    invocation = received["invocation"]
    assert invocation.authorized is True
    assert invocation.session_id == "sess-9"
    assert invocation.tool_names == frozenset({"read_file", "terminal"})
    # Dispatch outside the session's effective tool set refuses even when authorized.
    with pytest.raises(PermissionError, match="not available"):
        invocation.dispatch_tool("web_search", {"query": "x"})
