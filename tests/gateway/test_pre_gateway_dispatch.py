"""Tests for the pre_gateway_dispatch plugin hook.

The hook allows plugins to intercept incoming messages before auth and
agent dispatch. It runs in _handle_message and acts on returned action
dicts: {"action": "skip"|"rewrite"|"allow"|"authorize"}.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOW_ALL_USERS",
        "DISCORD_ALLOW_BOTS",
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


def _make_discord_bot_event(*, channel_context: str | None = "untrusted history") -> MessageEvent:
    return MessageEvent(
        text="<@999> signed machine request",
        message_id="bot-m1",
        channel_context=channel_context,
        source=SessionSource(
            platform=Platform.DISCORD,
            user_id="123456",
            chat_id="654321",
            user_name="request-bot",
            chat_type="group",
            is_bot=True,
        ),
    )


def _make_hook_gate_runner():
    runner, _adapter = _make_runner(Platform.DISCORD)
    runner.adapters[Platform.DISCORD].config = PlatformConfig(
        enabled=True,
        extra={"allow_bots": "hook_mentions"}
    )
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._admit_bot_message_for_source = lambda source: True
    return runner


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


@pytest.mark.asyncio
async def test_hook_mentions_authorize_is_one_event_and_can_rewrite_clear_context(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "hook_mentions")
    results = iter([
        [{"action": "authorize", "text": "validated request", "clear_channel_context": True}],
        [],
    ])
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: next(results))
    runner = _make_hook_gate_runner()

    admitted = await runner._hm_admit_event(_make_discord_bot_event())
    assert admitted is not None
    event, _source, _internal = admitted
    assert event.text == "validated request"
    assert event.channel_context is None
    assert event._plugin_authorized is True

    # Authorization is a receipt on this event, not on the source or session.
    assert await runner._hm_admit_event(_make_discord_bot_event(channel_context=None)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook_result",
    [
        [],
        [None],
        [{"action": "allow"}],
        [{"action": "rewrite", "text": "not authorization"}],
        [{"action": "authorize", "text": 42}],
        [{"action": "authorize", "clear_channel_context": "yes"}],
        [
            {"action": "authorize", "text": 42},
            {"action": "authorize", "text": "must not win"},
        ],
    ],
)
async def test_hook_mentions_denies_without_valid_explicit_authorize(monkeypatch, hook_result):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "hook_mentions")
    # Even an explicit human-style allowlist entry cannot bypass this mode.
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "123456")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: hook_result)
    runner = _make_hook_gate_runner()

    assert await runner._hm_admit_event(_make_discord_bot_event()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook_result",
    [
        [
            {"action": "authorize", "text": "validated request"},
            {"action": "skip", "reason": "later callback timed out"},
        ],
        [
            {"action": "skip", "reason": "earlier callback timed out"},
            {"action": "authorize", "text": "validated request"},
        ],
    ],
)
async def test_hook_mentions_fail_closed_skip_dominates_authorize(monkeypatch, hook_result):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "hook_mentions")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: hook_result)
    runner = _make_hook_gate_runner()

    assert await runner._hm_admit_event(_make_discord_bot_event()) is None


@pytest.mark.asyncio
async def test_hook_invocation_exception_fails_closed_for_authorized_traffic(monkeypatch):
    _clear_auth_env(monkeypatch)

    def _raise(*_args, **_kwargs):
        raise TimeoutError("plugin timed out")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _raise)
    runner, _adapter = _make_runner(Platform.WHATSAPP)
    event = _make_event()

    assert runner._hm_pre_gateway_dispatch_hook(event, event.source) is None


@pytest.mark.asyncio
async def test_busy_session_runs_hook_before_authorization_or_queue_side_effects(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_a, **_kw: [{"action": "skip", "reason": "invalid signature"}],
    )
    runner = _make_hook_gate_runner()
    runner._is_user_authorized_for_source = MagicMock(return_value=True)

    handled = await runner._handle_active_session_busy_message(
        _make_discord_bot_event(), "agent:main:discord:group:654321",
    )

    assert handled is True
    runner._is_user_authorized_for_source.assert_not_called()
