"""Tests for the pre_gateway_dispatch plugin hook's ``respond`` action.

The hook allows plugins to intercept incoming messages before auth and agent
dispatch, acting on returned action dicts: {"action": "skip"|"respond"|
"rewrite"|"allow"}. ``skip`` drops silently and ``rewrite`` continues normal
dispatch with different text, but neither lets a plugin that fully handled a
message (canned answer, signed single-action command, rate-limit notice)
actually reply — a synchronous hook cannot ``await adapter.send()``, and
scheduling a fire-and-forget send would bypass the gateway's normal delivery
path (threading metadata, delivery ledger, error handling).

``respond`` hands the plugin's text to ``_hm_admit_event`` -> ``_handle_message``'s
existing return-value delivery path, the same one every other early reply in
that method already uses.
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
async def test_respond_returns_text_and_stops_dispatch(monkeypatch):
    """``respond`` hands the plugin's answer to the normal delivery path."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "respond", "text": "handled by plugin"}]
        return []

    # gateway/run_inbound.py's _hm_pre_gateway_dispatch_hook does a local
    # `from hermes_cli.lifecycle import invoke_hook`, so that is the name
    # that must be patched (hermes_cli.lifecycle.invoke_hook wraps
    # hermes_cli.plugins.invoke_hook with first-party observers).
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    result = await runner._handle_message(_make_event("anything"))

    assert result == "handled by plugin"
    # The agent never ran; the plugin's text is the turn's whole result.
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_respond_with_empty_text_falls_through(monkeypatch):
    """An empty ``respond`` must not swallow the message silently.

    Returning "" would be indistinguishable from "no reply" downstream, so the
    directive is ignored and normal dispatch continues — a plugin bug degrades
    to normal behaviour instead of dropping user messages.
    """
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "respond", "text": ""}]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    result = await runner._handle_message(_make_event("hi"))

    # Fell through to normal dispatch (unauthorized sender -> None), i.e. the
    # empty directive did not short-circuit with an empty answer.
    assert result is None


@pytest.mark.asyncio
async def test_skip_still_drops_with_no_reply(monkeypatch):
    """``skip`` is unaffected by the new action: still a silent drop."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "skip", "reason": "test"}]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    result = await runner._handle_message(_make_event("hi"))

    assert result is None
    adapter.send.assert_not_awaited()
