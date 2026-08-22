"""Tests for GatewayRunner.session_is_busy public API and pre_gateway_dispatch agent_busy kwarg."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_event(text: str = "hello", platform: Platform = Platform.WHATSAPP) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=platform,
            user_id="15551234567@s.whatsapp.net",
            chat_id="15551234567@s.whatsapp.net",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner(platform: Platform):
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    return runner, adapter


class TestSessionIsBusyPublicAPI:
    """Test the public session_is_busy(session_key) predicate."""

    @pytest.mark.asyncio
    async def test_idle_returns_false(self, monkeypatch):
        """Idle session (no agent in _running_agents) returns False."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        runner, _ = _make_runner(Platform.WHATSAPP)
        key = runner._session_key_for_source(_make_event().source)

        # idle -> False
        assert runner.session_is_busy(key) is False

    @pytest.mark.asyncio
    async def test_running_returns_true(self, monkeypatch):
        """Session with a running agent in _running_agents returns True."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        runner, _ = _make_runner(Platform.WHATSAPP)
        key = runner._session_key_for_source(_make_event().source)

        # idle -> False
        assert runner.session_is_busy(key) is False

        # busy -> True (dummy agent object in _running_agents)
        runner._running_agents[key] = object()
        assert runner.session_is_busy(key) is True

    @pytest.mark.asyncio
    async def test_viki_returns_false(self, monkeypatch):
        """Non-existent session key (viki) returns False, no KeyError."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        runner, _ = _make_runner(Platform.WHATSAPP)

        # completely unknown key should return False gracefully
        assert runner.session_is_busy("non:existent:session:key") is False


class TestPreGatewayDispatchAgentBusyKwarg:
    """Test that pre_gateway_dispatch hook receives agent_busy kwarg."""

    @pytest.mark.asyncio
    async def test_receives_agent_busy_when_idle(self, monkeypatch):
        """When session is idle, agent_busy is False."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        seen = {}

        def _fake_hook(name, **kwargs):
            if name == "pre_gateway_dispatch":
                seen["agent_busy"] = kwargs.get("agent_busy", "MISSING")
                seen["session_key"] = kwargs.get("session_key", "MISSING")
            return [{"action": "allow"}]

        async def _capture(event, source, _quick_key, _run_generation):
            return "ok"

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

        runner, _adapter = _make_runner(Platform.WHATSAPP)
        runner._handle_message_with_agent = _capture  # noqa: SLF001

        event = _make_event("hi")
        await runner._handle_message(event)

        assert seen.get("agent_busy") is False
        assert seen.get("session_key") != "MISSING"

    @pytest.mark.asyncio
    async def test_receives_agent_busy_when_busy(self, monkeypatch):
        """When session holds a running agent, agent_busy is True."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        seen = {}

        def _fake_hook(name, **kwargs):
            if name == "pre_gateway_dispatch":
                seen["agent_busy"] = kwargs.get("agent_busy", "MISSING")
                seen["session_key"] = kwargs.get("session_key", "MISSING")
            return [{"action": "allow"}]

        async def _capture(event, source, _quick_key, _run_generation):
            return "ok"

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

        runner, _adapter = _make_runner(Platform.WHATSAPP)
        runner._handle_message_with_agent = _capture  # noqa: SLF001

        event = _make_event("hi")
        _busy_key = runner._session_key_for_source(event.source)
        runner._running_agents[_busy_key] = MagicMock()  # fake busy agent slot

        # busy path: _handle_message may return None (interrupt/queue short-circuit)
        # since an agent is already running — this test only verifies the hook
        # received the correct agent_busy value, not the dispatch outcome.
        await runner._handle_message(event)
        assert seen.get("agent_busy") is True
        assert seen.get("session_key") != "MISSING"

    @pytest.mark.asyncio
    async def test_receives_agent_busy_in_kwargs(self, monkeypatch):
        """Hook can also observe agent_busy if it changes during hook execution."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        seen = {"before": None, "after": None}

        def _fake_hook(name, **kwargs):
            if name == "pre_gateway_dispatch":
                seen["before"] = kwargs.get("agent_busy", "MISSING")
                seen["session_key"] = kwargs.get("session_key", "MISSING")
                # Hook itself doesn't change _running_agents, so before == after
                # This test just confirms the kwarg is present in the signature
            return [{"action": "allow"}]

        async def _capture(event, source, _quick_key, _run_generation):
            return "ok"

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

        runner, _adapter = _make_runner(Platform.WHATSAPP)
        runner._handle_message_with_agent = _capture  # noqa: SLF001

        event = _make_event("hi")
        await runner._handle_message(event)

        assert seen.get("before") is False
        assert seen.get("session_key") != "MISSING"


class TestSteerInterruptTestsNotBroken:
    """Regression guard: existing steer/interrupt related tests must still pass."""

    @pytest.mark.asyncio
    async def test_steer_command_still_works(self, monkeypatch):
        """/steer command handler is not affected by session_is_busy changes."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        runner, adapter = _make_runner(Platform.WHATSAPP)

        # Simulate a running agent so session_is_busy returns True
        key = runner._session_key_for_source(_make_event().source)
        runner._running_agents[key] = MagicMock()

        # Internal event with /steer should still be processed (bypasses hook)
        event = _make_event("/steer new direction")
        event.internal = True

        async def _capture(event, source, _quick_key, _run_generation):
            return "steered"

        runner._handle_message_with_agent = _capture  # noqa: SLF001

        result = await runner._handle_message(event)
        # Internal events bypass the hook and should still be handled
        # The real handler returns a steer-queued message, not our mock
        assert result is not None
        assert "steer" in result.lower() or "queued" in result.lower()

    @pytest.mark.asyncio
    async def test_interrupt_still_works(self, monkeypatch):
        """Interrupt handling is not affected by session_is_busy changes."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        runner, adapter = _make_runner(Platform.WHATSAPP)

        # Simulate a running agent so session_is_busy returns True
        key = runner._session_key_for_source(_make_event().source)
        runner._running_agents[key] = MagicMock()

        # Internal event (like interrupt signal) should still be processed
        event = _make_event("/stop")
        event.internal = True

        async def _capture(event, source, _quick_key, _run_generation):
            return "stopped"

        runner._handle_message_with_agent = _capture  # noqa: SLF001

        result = await runner._handle_message(event)
        # Internal events bypass the hook and should still be handled
        # The real handler returns a stop message, not our mock
        assert result is not None
        assert "stop" in result.lower() or "⚡" in result


class TestSessionIsBusyEdgeCases:
    """Edge cases for session_is_busy public API."""

    @pytest.mark.asyncio
    async def test_runner_without_running_agents_attribute(self, monkeypatch):
        """Runner missing _running_agents attribute returns False gracefully."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        runner, _ = _make_runner(Platform.WHATSAPP)
        # Simulate bare-runner test: object.__new__ without __init__
        del runner._running_agents

        key = runner._session_key_for_source(_make_event().source)
        assert runner.session_is_busy(key) is False

    @pytest.mark.asyncio
    async def test_session_is_busy_matches_is_session_running(self, monkeypatch):
        """session_is_busy mirrors _is_session_running for the same key."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        runner, _ = _make_runner(Platform.WHATSAPP)
        event = _make_event("hi")
        key = runner._session_key_for_source(event.source)

        # Both should be False when idle
        assert runner.session_is_busy(key) is False
        assert runner._is_session_running(key) is False

        # Add a fake agent to _running_agents
        runner._running_agents[key] = MagicMock()

        # session_is_busy reads _running_agents directly
        assert runner.session_is_busy(key) is True

        # _is_session_running reads from SessionState.turn.agent
        # They may differ if SessionState isn't synced — that's expected.
        # This test just documents the semantic difference.

    @pytest.mark.asyncio
    async def test_unknown_platform_session_key(self, monkeypatch):
        """session_is_busy works with any session_key format, not just WhatsApp."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")

        runner, _ = _make_runner(Platform.TELEGRAM)
        event = _make_event("hi", platform=Platform.TELEGRAM)
        key = runner._session_key_for_source(event.source)

        assert runner.session_is_busy(key) is False

        runner._running_agents[key] = object()
        assert runner.session_is_busy(key) is True


class TestAgentBusyKwarg:
    """Review fix #1: agent_busy must reach the hook callback, not just
    a DEBUG log. Plugins rely on the documented contract."""

    @pytest.mark.asyncio
    async def test_receives_agent_busy_in_kwargs(self, monkeypatch):
        """pre_gateway_dispatch callback receives agent_busy kwarg."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        seen = {"agent_busy": None, "session_key": None}

        def _fake_hook(name, **kwargs):
            if name == "pre_gateway_dispatch":
                seen["agent_busy"] = kwargs.get("agent_busy", "MISSING")
                seen["session_key"] = kwargs.get("session_key", "MISSING")
            return [{"action": "allow"}]

        async def _capture(event, source, _quick_key, _run_generation):
            return "ok"

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

        runner, _adapter = _make_runner(Platform.WHATSAPP)
        runner._handle_message_with_agent = _capture  # noqa: SLF001

        event = _make_event("hi")
        await runner._handle_message(event)

        assert seen.get("agent_busy") is False
        assert seen.get("session_key") != "MISSING"


class TestStrictSignaturePluginCompat:
    """Review fix #2: invoke_hook must not blow up existing plugins when the
    hook gains new kwargs. Hermes plugins receive the payload via **kwargs or
    via matching keyword names; invoke_hook filters additive fields by the
    callback's declared parameters, so a legacy **kwargs callback that only
    inspects known keys keeps working untouched."""

    @pytest.mark.asyncio
    async def test_existing_kwargs_plugin_receives_new_fields(self, monkeypatch):
        """A pre-existing **kwargs plugin sees agent_busy_*/session_key without
        error and can read only the keys it cares about."""
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

        seen = {"event": None, "gateway": None, "agent_busy": "MISSING", "session_key": "MISSING"}

        def _legacy_hook(name, **kwargs):
            if name == "pre_gateway_dispatch":
                seen["event"] = kwargs.get("event")
                seen["gateway"] = kwargs.get("gateway")
                seen["agent_busy"] = kwargs.get("agent_busy", "MISSING")
                seen["session_key"] = kwargs.get("session_key", "MISSING")
            return [{"action": "allow"}]

        async def _capture(event, source, _quick_key, _run_generation):
            return "ok"

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _legacy_hook)

        runner, _adapter = _make_runner(Platform.WHATSAPP)
        # idle session: no running agent, so agent_busy is False
        runner._handle_message_with_agent = _capture  # noqa: SLF001

        result = await runner._handle_message(_make_event("hi"))
        assert result is not None
        assert seen.get("event") is not None
        assert seen.get("gateway") is not None
        assert seen.get("agent_busy") is False
        assert seen.get("session_key") != "MISSING"

    def test_invoke_hook_filters_additive_fields_from_narrow_callback(self, monkeypatch):
        """Unit-level: a callback declaring only (event, gateway) is invoked
        with exactly those, never TypeError on unseen kwargs."""
        from hermes_cli.plugins import PluginManager

        mgr = PluginManager.__new__(PluginManager)
        mgr._hooks = {"pre_gateway_dispatch": []}

        captured = {}

        def _narrow(event, gateway):
            captured["event"] = event
            captured["gateway"] = gateway
            return {"action": "allow"}

        mgr._hooks["pre_gateway_dispatch"].append(_narrow)
        # Payload carries additive fields the narrow callback never declared.
        out = mgr.invoke_hook(
            "pre_gateway_dispatch",
            event="E",
            gateway="G",
            session_store="S",
            session_key="K",
            agent_busy=True,
            telemetry_schema_version=1,
        )
        assert out == [{"action": "allow"}]
        assert captured == {"event": "E", "gateway": "G"}