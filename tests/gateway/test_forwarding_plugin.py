"""Tests for the declarative cross-channel forwarding plugin.

The plugin registers a ``pre_gateway_dispatch`` hook that reads rules from
``$HERMES_HOME/forwarding-rules.json`` and relays matching inbound messages
(text + first image) to a target chat via the target platform's adapter.
``forward_only: true`` also drops the message so the agent does not reply.
"""

from __future__ import annotations

import asyncio
import json
import types
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
import plugins.forwarding as fwd


@pytest.fixture(autouse=True)
def _rules_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # reset the module cache between tests
    fwd._RULES_CACHE = None
    fwd._RULES_MTIME = -1.0
    return tmp_path


def _write_rules(home, rules):
    (home / "forwarding-rules.json").write_text(json.dumps({"rules": rules}), encoding="utf-8")


def _event(platform, chat_id, text="hello", media_urls=None):
    source = types.SimpleNamespace(platform=platform, chat_id=chat_id)
    return types.SimpleNamespace(source=source, text=text, media_urls=media_urls or [])


def _gateway_with(target_platform):
    adapter = types.SimpleNamespace(send=AsyncMock(), send_image=AsyncMock())
    gw = types.SimpleNamespace(adapters={target_platform: adapter})
    return gw, adapter


def test_no_rules_is_noop(_rules_file):
    gw, adapter = _gateway_with(Platform.TELEGRAM)
    ev = _event(Platform.WHATSAPP, "120@g.us")
    assert fwd._on_inbound(event=ev, gateway=gw) is None
    adapter.send.assert_not_called()


def test_matching_rule_relays_text(_rules_file):
    _write_rules(_rules_file, [
        {"from": {"platform": "whatsapp", "chat": "120@g.us"},
         "to": {"platform": "telegram", "chat": "-100999"}},
    ])
    gw, adapter = _gateway_with(Platform.TELEGRAM)

    async def run():
        res = fwd._on_inbound(event=_event(Platform.WHATSAPP, "120@g.us", "job alert"), gateway=gw)
        await asyncio.sleep(0)  # let the scheduled relay task run
        return res

    res = asyncio.run(run())
    assert res is None  # not forward_only → agent still processes
    adapter.send.assert_awaited_once_with("-100999", "job alert")


def test_forward_only_drops_message(_rules_file):
    _write_rules(_rules_file, [
        {"from": {"platform": "wecom", "chat": "src"},
         "to": {"platform": "whatsapp", "chat": "dst@g.us"},
         "forward_only": True},
    ])
    gw, adapter = _gateway_with(Platform.WHATSAPP)

    async def run():
        res = fwd._on_inbound(event=_event(Platform.WECOM, "src", "notice"), gateway=gw)
        await asyncio.sleep(0)
        return res

    res = asyncio.run(run())
    assert res == {"action": "skip", "reason": "forwarded"}
    adapter.send.assert_awaited_once_with("dst@g.us", "notice")


def test_prefix_and_media(_rules_file):
    _write_rules(_rules_file, [
        {"from": {"platform": "whatsapp", "chat": "a@g.us"},
         "to": {"platform": "telegram", "chat": "-1"},
         "media": True, "prefix": "[fwd] "},
    ])
    gw, adapter = _gateway_with(Platform.TELEGRAM)

    async def run():
        fwd._on_inbound(
            event=_event(Platform.WHATSAPP, "a@g.us", "pic", media_urls=["/tmp/x.jpg", "/tmp/y.jpg"]),
            gateway=gw,
        )
        await asyncio.sleep(0)

    asyncio.run(run())
    adapter.send.assert_awaited_once_with("-1", "[fwd] pic")
    adapter.send_image.assert_awaited_once_with("-1", "/tmp/x.jpg", caption=None)


def test_unconnected_target_is_safe(_rules_file):
    _write_rules(_rules_file, [
        {"from": {"platform": "whatsapp", "chat": "a@g.us"},
         "to": {"platform": "telegram", "chat": "-1"}},
    ])
    gw = types.SimpleNamespace(adapters={})  # telegram not connected

    async def run():
        return fwd._on_inbound(event=_event(Platform.WHATSAPP, "a@g.us"), gateway=gw)

    # must not raise even though the target adapter is missing
    assert asyncio.run(run()) is None


def test_non_matching_chat_ignored(_rules_file):
    _write_rules(_rules_file, [
        {"from": {"platform": "whatsapp", "chat": "a@g.us"},
         "to": {"platform": "telegram", "chat": "-1"}},
    ])
    gw, adapter = _gateway_with(Platform.TELEGRAM)
    res = fwd._on_inbound(event=_event(Platform.WHATSAPP, "OTHER@g.us"), gateway=gw)
    assert res is None
    adapter.send.assert_not_called()
