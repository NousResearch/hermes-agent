"""Tests for the pre_gateway_dispatch plugin hook.

The hook allows plugins to intercept incoming messages before auth and
agent dispatch. It runs in _handle_message and acts on returned action
dicts: {"action": "skip"|"rewrite"|"allow"}.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
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
async def test_hook_fires_without_session_store_attribute(monkeypatch):
    """A runner missing session_store still delivers the event to plugins.

    Regression: the hook kwargs read ``self.session_store`` directly, so a
    partially-initialized runner raised AttributeError inside the dispatch
    try-block — the hook never fired, and every message logged
    "pre_gateway_dispatch invocation failed: 'GatewayRunner' object has no
    attribute 'session_store'". Plugins must receive the event (with
    session_store=None) instead.
    """
    _clear_auth_env(monkeypatch)

    seen = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen["session_store"] = kwargs.get("session_store", "MISSING")
            return [{"action": "skip", "reason": "plugin-handled"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    del runner.session_store

    result = await runner._handle_message(_make_event("hi"))
    assert result is None
    # Hook actually fired (skip short-circuited before auth) with a None store.
    assert seen == {"session_store": None}
    adapter.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# Busy-session path (#77976)
# ---------------------------------------------------------------------------
# Regression: _handle_active_session_busy_message skipped pre_gateway_dispatch
# entirely, so plugin-published allowlists never authorized senders (silent
# drop) and hook-based media sanitization never ran on queued follow-ups
# (unsanitized media reached the model on replay). The busy path must run the
# same hook, before the auth gate, as the cold path (_handle_message).

def _make_busy_runner(platform=Platform.WHATSAPP, authorized=True):
    """Build a minimal GatewayRunner for the busy-session path.

    Uses busy_input_mode=queue so the handler terminates at the pending
    queue (no steer/interrupt machinery) while still exercising the
    pre_gateway_dispatch hook and the auth gate.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True)})
    runner._draining = False
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    runner._running_agents = {}
    runner._busy_ack_ts = {}
    runner._pending_messages = {}
    runner.session_store = None
    adapter = SimpleNamespace(
        _pending_messages={},
        send=AsyncMock(),
        _send_with_retry=AsyncMock(),
    )
    runner.adapters = {platform: adapter}
    runner._adapter_for_source = lambda source: adapter
    runner._peek_session_state = lambda session_key: None
    runner._is_user_authorized = lambda source: authorized
    return runner, adapter


@pytest.mark.asyncio
async def test_busy_path_runs_hook_before_auth(monkeypatch):
    """#77976: the busy path must run pre_gateway_dispatch BEFORE the auth
    gate so a plugin-published allowlist authorizes the sender (instead of
    the message being silently dropped as unauthorized)."""
    published = {"allowed": set()}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            published["allowed"].add(kwargs["event"].source.user_id)
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_busy_runner(authorized=False)
    # Static authz config denies everyone; only the plugin-published
    # allowlist can authorize this sender.
    runner._is_user_authorized = lambda source: source.user_id in published["allowed"]

    event = _make_event("follow-up while busy")
    sk = "agent:main:whatsapp:dm:15551234567"

    result = await runner._handle_active_session_busy_message(event, sk)

    # Hook ran → allowlist published → auth passes → queued, not dropped.
    assert event.source.user_id in published["allowed"]
    assert adapter._pending_messages.get(sk) is event
    assert result is True  # handled (queued; busy ack suppressed by cooldown)


@pytest.mark.asyncio
async def test_busy_path_hook_sanitizes_media(monkeypatch):
    """#77976: media sanitized by the hook on the busy path must not reach
    the queued event — and therefore cannot reach the model on replay."""
    def _fake_hook(name, **kwargs):
        event = kwargs["event"]
        event.media_urls = []
        event.media_types = []
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_busy_runner(authorized=True)
    event = _make_event("photo burst follow-up")
    event.message_type = MessageType.PHOTO
    event.media_urls = ["/tmp/photo.jpg"]
    event.media_types = ["image/jpeg"]

    sk = "agent:main:whatsapp:dm:15551234567"
    await runner._handle_active_session_busy_message(event, sk)

    queued = adapter._pending_messages.get(sk)
    assert queued is not None
    assert queued.media_urls == []
    assert queued.media_types == []


@pytest.mark.asyncio
async def test_busy_path_hook_skip_drops_message(monkeypatch):
    """#77976: a plugin ``skip`` on the busy path drops the message without
    consulting auth or queueing anything."""
    def _fake_hook(name, **kwargs):
        return [{"action": "skip", "reason": "plugin-handled"}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_busy_runner(authorized=False)
    auth_calls = {"count": 0}

    def _auth(source):
        auth_calls["count"] += 1
        return True

    runner._is_user_authorized = _auth

    event = _make_event("hi")
    sk = "agent:main:whatsapp:dm:15551234567"
    result = await runner._handle_active_session_busy_message(event, sk)

    assert result is True
    assert sk not in adapter._pending_messages
    # Hook skip short-circuits BEFORE the auth gate.
    assert auth_calls["count"] == 0


@pytest.mark.asyncio
async def test_busy_path_internal_events_bypass_hook(monkeypatch):
    """Internal (system-generated) events skip the hook on the busy path too,
    matching cold-path semantics."""
    called = {"count": 0}

    def _fake_hook(name, **kwargs):
        called["count"] += 1
        return [{"action": "skip"}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_busy_runner(authorized=True)
    event = _make_event("background completion")
    event.internal = True

    sk = "agent:main:whatsapp:dm:15551234567"
    result = await runner._handle_active_session_busy_message(event, sk)

    assert called["count"] == 0
    # Internal events fall through to the base adapter for silent queueing.
    assert result is False


@pytest.mark.asyncio
async def test_busy_path_hook_rewrite_flows_to_queued_event(monkeypatch):
    """#77976: a plugin ``rewrite`` on the busy path must reach the queued
    event (in-place mutation is observed by the adapter's pending slot)."""
    def _fake_hook(name, **kwargs):
        return [{"action": "rewrite", "text": "rewritten follow-up"}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_busy_runner(authorized=True)
    event = _make_event("original follow-up")

    sk = "agent:main:whatsapp:dm:15551234567"
    await runner._handle_active_session_busy_message(event, sk)

    queued = adapter._pending_messages.get(sk)
    assert queued is not None
    assert queued.text == "rewritten follow-up"
