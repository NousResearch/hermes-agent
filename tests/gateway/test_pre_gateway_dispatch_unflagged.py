"""``pre_gateway_dispatch`` without a required gate: every combination keeps the first-result rule.

One to three ordinary plugins over nine kinds of result, plus a failed hook invocation, through the
gateway's real ``_handle_message``: 820 cases against the rule as documented before required gates
existed (the first recognised action decides; a callback that raises, returns ``None``, a non-dict
or an unknown action has no say). The file uses no ``fail_closed`` keyword, so it runs unchanged on
a tree without the feature and pins that an unflagged gateway behaves the same with it.
"""

import asyncio
import itertools

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

ORIGINAL, REWRITTEN = "private words 4711", "new"


def _raise(**_kwargs):
    raise RuntimeError("boom")


async def _araise(**_kwargs):
    await asyncio.sleep(0)
    raise RuntimeError("boom")


KINDS = {
    "none": lambda **_: None,
    "allow": lambda **_: {"action": "allow"},
    "skip": lambda **_: {"action": "skip", "reason": "r"},
    "rewrite": lambda **_: {"action": "rewrite", "text": REWRITTEN},
    "rewrite-without-text": lambda **_: {"action": "rewrite"},
    "unknown": lambda **_: {"action": "unknown"},
    "string": lambda **_: "skip",
    "raise": _raise,
    "async-raise": _araise,
}
# What the first callback of each deciding kind does to the message; the other kinds have no say.
DECIDES = {"allow": ORIGINAL, "skip": None, "rewrite": REWRITTEN, "rewrite-without-text": ORIGINAL}


def _expected(combo):
    return next((DECIDES[kind] for kind in combo if kind in DECIDES), ORIGINAL)


def _manager(monkeypatch):
    from hermes_cli import plugins

    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: mgr)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: mgr)
    return mgr


async def _dispatch():
    """The text the gateway goes on with after admission, or ``None`` when the message is dropped."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    runner.session_store = None
    runner._is_user_authorized_for_source = lambda _source: True
    runner._admit_bot_message_for_source = lambda _source: True
    runner._admit_bot_chain = lambda _event: True
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._hm_estop_gate = lambda event, _source, _internal: event.text  # first step after admission
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="777", user_name="t")
    return await runner._handle_message(
        MessageEvent(text=ORIGINAL, message_type=MessageType.TEXT, source=source, message_id="m1"))


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.asyncio
async def test_ordinary_plugins_keep_the_first_result_rule(monkeypatch, count):
    wrong = []
    for combo in itertools.product(KINDS, repeat=count):
        mgr = _manager(monkeypatch)
        for index, kind in enumerate(combo):
            PluginContext(PluginManifest(name=f"p{index}", source="user"), mgr).register_hook(
                "pre_gateway_dispatch", KINDS[kind])
        if (got := await _dispatch()) != _expected(combo):
            wrong.append((combo, got))
    assert wrong == []


@pytest.mark.asyncio
async def test_a_failed_invocation_dispatches_the_message(monkeypatch):
    from hermes_cli import lifecycle

    mgr = _manager(monkeypatch)
    PluginContext(PluginManifest(name="p0", source="user"), mgr).register_hook("pre_gateway_dispatch", KINDS["skip"])

    async def broken(_name, **_kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(lifecycle, "ainvoke_hook", broken)
    assert await _dispatch() == ORIGINAL
