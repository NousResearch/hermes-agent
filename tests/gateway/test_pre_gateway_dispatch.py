"""Tests for the pre_gateway_dispatch plugin hook.

The hook allows plugins to intercept incoming messages before auth and
agent dispatch. It runs in _handle_message and acts on returned action
dicts: {"action": "skip"|"rewrite"|"allow"}.
"""

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


@pytest.mark.asyncio
async def test_hook_skip_short_circuits_dispatch(monkeypatch):
    """A plugin returning {'action': 'skip'} drops the message before auth."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "skip", "reason": "plugin-handled"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)

    result = await runner._handle_message(_make_event("hi"))

    assert result is None
    adapter.send.assert_not_awaited()
    runner.pairing_store.generate_code.assert_not_called()


@pytest.mark.asyncio
async def test_hook_rewrite_replaces_event_text(monkeypatch):
    """A plugin returning {'action': 'rewrite', 'text': ...} mutates event.text."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    seen_text = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "rewrite", "text": "REWRITTEN"}]
        return []

    async def _capture(event, source, _quick_key, _run_generation):
        seen_text["value"] = event.text
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    await runner._handle_message(_make_event("original"))

    assert seen_text.get("value") == "REWRITTEN"


@pytest.mark.asyncio
async def test_hook_allow_falls_through_to_auth(monkeypatch):
    """A plugin returning {'action': 'allow'} continues to normal dispatch."""
    _clear_auth_env(monkeypatch)
    # No allowed users set → auth fails → pairing flow triggers.
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "allow"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = "12345"

    result = await runner._handle_message(_make_event("hi"))

    # auth chain ran → pairing code was generated
    assert result is None
    runner.pairing_store.generate_code.assert_called_once()


@pytest.mark.asyncio
async def test_hook_exception_does_not_break_dispatch(monkeypatch):
    """A raising plugin hook does not break the gateway."""
    _clear_auth_env(monkeypatch)
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    def _fake_hook(name, **kwargs):
        raise RuntimeError("plugin blew up")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = None

    # Should not raise; falls through to auth chain.
    result = await runner._handle_message(_make_event("hi"))
    assert result is None


@pytest.mark.asyncio
async def test_internal_events_bypass_hook(monkeypatch):
    """Internal events (event.internal=True) skip the plugin hook entirely."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    called = {"count": 0}

    def _fake_hook(name, **kwargs):
        called["count"] += 1
        return [{"action": "skip"}]

    async def _capture(event, source, _quick_key, _run_generation):
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    event = _make_event("hi")
    event.internal = True

    # Even though the hook would say skip, internal events bypass it.
    await runner._handle_message(event)
    assert called["count"] == 0


@pytest.mark.asyncio
async def test_hook_reply_sends_and_short_circuits_before_auth(monkeypatch):
    """A plugin reply action sends directly and drops the message before auth."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [
                {
                    "action": "reply",
                    "text": "Plugin handled this.",
                    "reply_to": True,
                    "reason": "direct-response",
                }
            ]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = "12345"

    result = await runner._handle_message(_make_event("hi"))

    assert result is None
    adapter.send.assert_awaited_once_with(
        "15551234567@s.whatsapp.net",
        "Plugin handled this.",
        reply_to="m1",
    )
    runner.pairing_store.generate_code.assert_not_called()


@pytest.mark.asyncio
async def test_hook_reply_to_false_sends_without_reply_anchor(monkeypatch):
    """reply_to=False sends to the chat without threading to the source message."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "reply", "text": "No anchor", "reply_to": False}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = "12345"

    result = await runner._handle_message(_make_event("hi"))

    assert result is None
    adapter.send.assert_awaited_once_with(
        "15551234567@s.whatsapp.net",
        "No anchor",
        reply_to=None,
    )
    runner.pairing_store.generate_code.assert_not_called()


@pytest.mark.asyncio
async def test_hook_reply_rate_limited_does_not_send(monkeypatch):
    """Pre-auth direct replies reuse the pairing rate limit to avoid spam."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "reply", "text": "Too fast", "reply_to": True}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store._is_rate_limited.return_value = True

    result = await runner._handle_message(_make_event("hi"))

    assert result is None
    adapter.send.assert_not_awaited()
    runner.pairing_store._is_rate_limited.assert_called_once_with(
        "whatsapp", "15551234567@s.whatsapp.net"
    )
    runner.pairing_store._record_rate_limit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_text", [None, "", "   ", 123])
async def test_hook_reply_invalid_text_does_not_send_or_crash(monkeypatch, bad_text):
    """Invalid reply text fails closed and normal auth dispatch continues."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    reached_agent = {"value": False}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "reply", "text": bad_text, "reply_to": True}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)

    async def _capture(event, source, _quick_key, _run_generation):
        reached_agent["value"] = True
        return "ok"

    runner._handle_message_with_agent = _capture  # noqa: SLF001

    result = await runner._handle_message(_make_event("hi"))

    assert result == "ok"
    adapter.send.assert_not_awaited()
    runner.pairing_store.generate_code.assert_not_called()
    assert reached_agent["value"] is True


@pytest.mark.asyncio
async def test_hook_reply_missing_adapter_does_not_crash(monkeypatch):
    """A reply action with no platform adapter is skipped safely."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "reply", "text": "No adapter", "reply_to": True}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner.adapters = {}
    runner.pairing_store.generate_code.return_value = "12345"

    result = await runner._handle_message(_make_event("hi"))

    assert result is None
    runner.pairing_store.generate_code.assert_not_called()



@pytest.mark.asyncio
async def test_hook_reply_routes_through_source_profile_not_platform_default(monkeypatch):
    """A reply for a secondary-profile chat is delivered by that profile's
    adapter, not the platform default.

    Regression for the wrong-profile delivery flagged on #56616: the reply path
    must resolve the adapter via ``_adapter_for_source(source)`` (which honors
    ``source.profile`` and ``_profile_adapters``) rather than
    ``self.adapters.get(source.platform)``, which always returns the default
    profile's adapter.
    """
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "reply", "text": "Profile reply", "reply_to": True}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, default_adapter = _make_runner(Platform.WHATSAPP)
    # A distinct adapter owns the "secondary" profile on the same platform.
    profile_adapter = SimpleNamespace(send=AsyncMock())
    runner._profile_adapters = {"secondary": {Platform.WHATSAPP: profile_adapter}}

    event = _make_event("hi")
    event.source.profile = "secondary"

    result = await runner._handle_message(event)

    assert result is None
    profile_adapter.send.assert_awaited_once_with(
        "15551234567@s.whatsapp.net",
        "Profile reply",
        reply_to="m1",
    )
    default_adapter.send.assert_not_called()
