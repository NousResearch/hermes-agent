"""Busy-path leak matrix: a plugin slash command never reaches the running turn's model context.

Companion to ``test_busy_plugin_command_reject.py`` (which covers the three busy input modes and the
underscored/hyphen form). This file widens the guarantee from "steer/queue/interrupt are not called
with that one argument string" to "NO sink that can carry text towards the model ever sees the
prompt", and pins the surrounding behaviour that must not regress:

* every model-facing sink (``_steer_running_agent``, ``_redirect_active_turn``,
  ``_queue_or_replace_pending_event``, ``_enqueue_fifo``, ``_hm_merge_pending_for_source`` and the
  receiver's own ``steer``/``redirect``/``interrupt``) is spied on with a unique sentinel; the
  rejection reply is the ONLY thing that leaves the runner.
* the interception precedes the Telegram grace-window queue and the photo-burst queue, so media and
  just-started turns cannot park the raw text for the next turn either.
* a raising plugin handler is never invoked mid-turn, a core command shadows a same-named plugin
  command, and an unavailable plugin registry degrades to the old fall-through instead of crashing.
* the idle path still dispatches the plugin handler (the rejection is busy-only).

Reverting the interception (or making ``_hm_busy_plugin_command_reject`` return ``None``) makes the
``test_busy_*`` cases here fail: the sentinel shows up in a sink, or the reply is ``None``. The last
test asserts that sensitivity directly, so the suite cannot silently lose it.
"""

from __future__ import annotations

import time

import pytest

from agent.i18n import t
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

KEY = "telegram:dm:user-1"
# A sentinel that must never appear outside the chat-visible rejection (which never echoes args).
LEAK = "SENTINELA-7f4c2a"


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, user_id="user-1", chat_id="chat-1",
        user_name="tester", chat_type="dm",
    )


class _Adapter:
    """Delivery-adapter stand-in holding the only state the busy queue paths touch."""

    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self.sent: list = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append(text)


class _Receiver:
    """Running-agent stand-in: records every direct injection into the live turn."""

    def __init__(self, can_redirect: bool = True) -> None:
        self._supports_active_turn_redirect = can_redirect
        self.steered = None
        self.redirected = None
        self.interrupted = None
        self._active_children: list = []

    def steer(self, text):
        self.steered = text
        return True

    def redirect(self, text):
        self.redirected = text
        return True

    def interrupt(self, text):
        self.interrupted = text


class _Sinks:
    """Every gateway-side place that can hand text to the model, spied on with the text it got."""

    def __init__(self) -> None:
        self.steer: list[str] = []
        self.redirect: list[str] = []
        self.queue: list[str] = []
        self.fifo: list[str] = []
        self.merge: list[str] = []

    @property
    def all(self) -> list[str]:
        return [*self.steer, *self.redirect, *self.queue, *self.fifo, *self.merge]

    def carrying(self, token: str) -> list[str]:
        return [text for text in self.all if token in (text or "")]


@pytest.fixture
def plugin_commands():
    """Register probe plugin commands in the live registry (``/priv`` itself may not be installed).

    Names are probe-specific so these tests never depend on which plugins the operator installed.
    """
    from hermes_cli import plugins

    table = plugins.get_plugin_commands()
    calls: list[str] = []

    def handler(args):
        calls.append(args)
        return "HANDLER RAN:" + args

    entry = {
        "handler": handler,
        "description": "probe private command",
        "plugin": "probe-busy-leak",
        "plugin_key": "probe-busy-leak",
        "args_hint": "",
        "argument_mode": None,
    }
    # Install over whatever the live registry holds, remembering the previous entries so an
    # already-installed plugin command (e.g. the operator's real /priv) survives the fixture.
    saved = {name: table.get(name) for name in ("priv", "priv-chat", "secret-hold")}
    for name in saved:
        table[name] = entry
    try:
        yield {"entry": entry, "calls": calls}
    finally:
        for name, previous in saved.items():
            if previous is None:
                table.pop(name, None)
            else:
                table[name] = previous


def _busy_runner(mode: str = "steer", *, can_redirect: bool = True, sinks: _Sinks | None = None,
                 started_ts: float = 0.0):
    runner = GatewayRunner(config=GatewayConfig())
    runner._busy_input_mode = mode
    runner._busy_text_mode = mode
    adapter = _Adapter()
    runner._delivery_adapter_for = lambda source: adapter  # type: ignore[method-assign]
    receiver = _Receiver(can_redirect=can_redirect)
    state = runner._session_state(KEY)
    state.turn.agent = receiver
    state.turn.started_ts = started_ts
    if sinks is not None:
        def record(bucket: list[str], text: str) -> None:
            bucket.append(text if isinstance(text, str) else repr(text))

        runner._steer_running_agent = (  # type: ignore[method-assign]
            lambda running_agent, text: record(sinks.steer, text) or True)
        runner._redirect_active_turn = (  # type: ignore[method-assign]
            lambda running_agent, text, session_key, event: record(sinks.redirect, text) or True)
        runner._queue_or_replace_pending_event = (  # type: ignore[method-assign]
            lambda session_key, event: record(sinks.queue, event.text))
        runner._enqueue_fifo = (  # type: ignore[method-assign]
            lambda session_key, queued, adapter_: record(sinks.fifo, queued.text))
        runner._hm_merge_pending_for_source = (  # type: ignore[method-assign]
            lambda source, quick_key, event, merge_text=False: record(sinks.merge, event.text))
    return runner, adapter, receiver


def _assert_zero_model_delivery(reply, adapter, receiver, sinks: _Sinks, *, registered: str) -> None:
    """The rejection is visible in chat and nothing else left the runner.

    *registered* is the name the plugin actually registered: the reply names that one (the typed
    spelling is normalized, ``/PRIV`` and ``/secret_hold`` answer as ``/priv`` / ``/secret-hold``).
    """
    assert reply and isinstance(reply, str)
    assert f"/{registered}" in reply
    assert "plugin" in reply.lower()
    assert LEAK not in reply, "the rejection must never echo the private argument"
    # No gateway-side sink carried the prompt.
    assert sinks.carrying(LEAK) == [], f"prompt reached a model-facing sink: {sinks.carrying(LEAK)}"
    # No direct injection into the live turn.
    assert receiver.steered is None
    assert receiver.redirected is None
    assert receiver.interrupted is None
    # Nothing parked for the next turn.
    assert adapter._pending_messages == {}
    # The path replies in-band; it does not also push an adapter send of its own.
    assert adapter.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "can_redirect"), [
    ("steer", True), ("queue", True), ("interrupt", True), ("interrupt", False),
])
async def test_busy_plugin_command_reaches_no_model_sink(plugin_commands, mode, can_redirect):
    """All busy modes (both interrupt flavours): the prompt never reaches a model-facing sink."""
    sinks = _Sinks()
    runner, adapter, receiver = _busy_runner(mode, can_redirect=can_redirect, sinks=sinks)
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text=f"/priv {LEAK}", source=source), source, KEY)

    _assert_zero_model_delivery(reply, adapter, receiver, sinks, registered="priv")
    assert plugin_commands["calls"] == [], "a plugin handler must never run beside a live turn"


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "registered"), [
    (f"/priv {LEAK}", "priv"),
    ("/priv", "priv"),                            # no arguments at all
    (f"  /priv   {LEAK}  ", "priv"),              # whitespace-padded
    (f"/PRIV {LEAK}", "priv"),                    # case-insensitive command resolution
    (f"/priv@hermes_probe_bot {LEAK}", "priv"),   # group-chat bot suffix
    (f"/secret_hold {LEAK}", "secret-hold"),      # underscored form of a hyphen registration
])
async def test_busy_plugin_command_variants_are_all_rejected(plugin_commands, text, registered):
    """Every spelling that resolves to a registered plugin command is intercepted."""
    sinks = _Sinks()
    runner, adapter, receiver = _busy_runner("steer", sinks=sinks)
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text=text, source=source), source, KEY)

    _assert_zero_model_delivery(reply, adapter, receiver, sinks, registered=registered)
    assert plugin_commands["calls"] == []


@pytest.mark.asyncio
async def test_busy_plugin_command_on_a_photo_is_not_queued(plugin_commands):
    """The interception precedes the photo-burst queue: a captioned photo cannot park the text."""
    sinks = _Sinks()
    runner, adapter, receiver = _busy_runner("interrupt", sinks=sinks)
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text=f"/priv {LEAK}", message_type=MessageType.PHOTO,
                     media_urls=["/tmp/probe.jpg"], media_types=["image/jpeg"], source=source),
        source, KEY)

    _assert_zero_model_delivery(reply, adapter, receiver, sinks, registered="priv")
    assert plugin_commands["calls"] == []


@pytest.mark.asyncio
async def test_busy_plugin_command_inside_the_telegram_grace_window_is_not_queued(plugin_commands):
    """A just-started Telegram turn queues follow-ups (grace window) — never a plugin command."""
    sinks = _Sinks()
    runner, adapter, receiver = _busy_runner(
        "queue", sinks=sinks, started_ts=time.time())
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text=f"/priv {LEAK}", source=source), source, KEY)

    _assert_zero_model_delivery(reply, adapter, receiver, sinks, registered="priv")
    assert plugin_commands["calls"] == []


@pytest.mark.asyncio
async def test_raising_plugin_handler_is_never_invoked_mid_turn(plugin_commands):
    """The core does not dispatch the handler, so a handler that raises cannot reach the busy path."""
    calls = []

    def boom(args):
        calls.append(args)
        raise RuntimeError("probe handler exploded")

    plugin_commands["entry"]["handler"] = boom
    sinks = _Sinks()
    runner, adapter, receiver = _busy_runner("steer", sinks=sinks)
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text=f"/priv {LEAK}", source=source), source, KEY)

    _assert_zero_model_delivery(reply, adapter, receiver, sinks, registered="priv")
    assert calls == []


@pytest.mark.asyncio
async def test_core_command_shadows_a_same_named_plugin_command(plugin_commands):
    """Precedence is the idle path's: a core CommandDef wins, with its own busy_policy text."""
    from hermes_cli import plugins

    table = plugins.get_plugin_commands()
    table["model"] = dict(plugin_commands["entry"])
    try:
        sinks = _Sinks()
        runner, adapter, receiver = _busy_runner("steer", sinks=sinks)
        source = _source()

        reply = await runner._hm_handle_running_session_message(
            MessageEvent(text=f"/model {LEAK}", source=source), source, KEY)

        assert reply, "the core command must still answer mid-turn"
        assert "plugin" not in reply.lower(), "a core command must not take the plugin reject text"
        assert LEAK not in reply
        assert sinks.carrying(LEAK) == []
        assert receiver.steered is None and receiver.redirected is None
        assert adapter._pending_messages == {}
        assert plugin_commands["calls"] == []
    finally:
        table.pop("model", None)


@pytest.mark.asyncio
async def test_unavailable_plugin_registry_degrades_to_the_old_fall_through(plugin_commands, monkeypatch):
    """With the plugin registry unimportable the busy path behaves as it did before the fix.

    The reject helper guards its import for exactly this case ("plugins unavailable in this
    install — keep the plain-text path"): the leak returns, the router does not crash. Pinning the
    degraded mode keeps the fix's blast radius honest — only the interception is new, the
    fall-through below it is untouched.
    """
    import sys

    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", None)  # `from ... import` now raises
    sinks = _Sinks()
    runner, adapter, _receiver = _busy_runner("steer", sinks=sinks)
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text=f"/priv {LEAK}", source=source), source, KEY)

    assert reply is None, "degraded mode falls through to the busy input mode"
    assert sinks.steer and sinks.steer[-1].endswith(LEAK)
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "can_redirect", "where"), [
    ("steer", True, "steer"),            # steer mode → _steer_running_agent
    ("interrupt", True, "redirect"),     # redirect-capable agent → _redirect_active_turn
    ("interrupt", False, "interrupt"),   # no redirect support → agent.interrupt()
])
async def test_plain_text_still_forwards_in_every_busy_mode(plugin_commands, mode, can_redirect, where):
    """Control case: the interception must not touch ordinary follow-ups in any mode."""
    sinks = _Sinks()
    runner, adapter, receiver = _busy_runner(mode, can_redirect=can_redirect, sinks=sinks)
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text=f"nota normal {LEAK}", source=source), source, KEY)

    assert reply is None
    # The injection point for this mode received the text (the spied sinks bypass the receiver's
    # own steer/redirect, so the interrupt fallback is read off the receiver itself).
    delivered = receiver.interrupted if where == "interrupt" else getattr(sinks, where)
    assert delivered, f"{mode} (redirect={can_redirect}) should carry plain text to {where}"
    assert LEAK in (delivered[-1] if isinstance(delivered, list) else delivered)
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_idle_path_still_dispatches_the_plugin_handler(plugin_commands):
    """The rejection is busy-only: with no live turn the plugin handler runs as before."""
    runner = GatewayRunner(config=GatewayConfig())
    adapter = _Adapter()
    runner._delivery_adapter_for = lambda source: adapter  # type: ignore[method-assign]
    source = _source()

    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
        MessageEvent(text=f"/priv {LEAK}", source=source), source, "priv")

    assert handled is True
    assert command == "priv"
    assert result == f"HANDLER RAN:{LEAK}"
    assert plugin_commands["calls"] == [LEAK]


def test_rejection_text_comes_from_the_catalog(plugin_commands):
    """The visible message is the i18n entry, not an inline English string."""
    runner = GatewayRunner(config=GatewayConfig())

    reply = runner._hm_busy_plugin_command_reject("priv")

    assert reply == t("gateway.busy.plugin_command_rejected", command="priv")
    assert LEAK not in (reply or "")


@pytest.mark.asyncio
async def test_disabling_the_interception_leaks_the_prompt(plugin_commands, monkeypatch):
    """Sensitivity guard: with the interception neutered, the prompt DOES reach the turn.

    This is the property every ``test_busy_*`` case above depends on. If a future refactor makes the
    rejection vacuous (e.g. the helper returns ``None`` and the fall-through is what answers), this
    test fails and says so.
    """
    sinks = _Sinks()
    runner, _adapter, _receiver = _busy_runner("steer", sinks=sinks)
    monkeypatch.setattr(
        runner, "_hm_busy_plugin_command_reject", lambda command: None, raising=False)
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text=f"/priv {LEAK}", source=source), source, KEY)

    assert reply is None, "without the interception the busy path returns the old None"
    assert sinks.carrying(LEAK), "sensitivity guard: the prompt must reach a sink once neutered"
