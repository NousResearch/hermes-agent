"""A plugin slash command typed mid-turn is rejected — it never steers/queues/interrupts.

``_hm_busy_slash_or_photo`` resolved commands against the CORE registry only
(``hermes_cli.commands.resolve_command``); plugin commands live in their own registry
(``hermes_cli.plugins._plugin_commands``), so ``/priv hola`` fell through to the busy input mode
(steer / queue / interrupt) and its raw argument text entered the running turn's model context
while the plugin handler never ran. The core now rejects such a command with a visible message and
sends nothing to the model; the handler stays the plugin's business, the mid-run policy is core's.

Reverting the interception makes ``test_plugin_command_mid_turn_*`` fail: the reply becomes None
(steer mode) or the text lands in ``_pending_messages`` / ``interrupt`` instead.
"""

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource

KEY = "telegram:dm:user-1"


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, user_id="user-1", chat_id="chat-1",
        user_name="tester", chat_type="dm",
    )


class _Adapter:
    """Delivery-adapter stand-in holding the only state the busy queue path touches."""

    def __init__(self) -> None:
        self._pending_messages = {}
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append(text)


class _Receiver:
    """Running-agent stand-in: records every way text could reach the model."""

    _supports_active_turn_redirect = True

    def __init__(self) -> None:
        self.steered = None
        self.redirected = None
        self.interrupted = None

    def steer(self, text):
        self.steered = text
        return True

    def redirect(self, text):
        self.redirected = text
        return True

    def interrupt(self, text):
        self.interrupted = text


@pytest.fixture
def plugin_commands():
    """Register probe plugin commands in the live plugin registry (``/priv`` itself may not exist)."""
    from hermes_cli import plugins

    table = plugins.get_plugin_commands()
    entry = {
        "handler": lambda args: "HANDLER RAN:" + args,
        "description": "probe private command",
        "plugin": "probe-priv",
        "plugin_key": "probe-priv",
        "args_hint": "",
        "argument_mode": None,
    }
    table["priv"] = entry
    table["priv-chat"] = entry
    try:
        yield entry
    finally:
        table.pop("priv", None)
        table.pop("priv-chat", None)


def _busy_runner(mode: str = "steer"):
    runner = GatewayRunner(config=GatewayConfig())
    runner._busy_input_mode = mode
    runner._busy_text_mode = mode
    adapter = _Adapter()
    runner._delivery_adapter_for = lambda source: adapter  # type: ignore[method-assign]
    return runner, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["steer", "queue", "interrupt"])
async def test_plugin_command_mid_turn_is_rejected_without_reaching_the_model(plugin_commands, mode):
    runner, adapter = _busy_runner(mode)
    receiver = _Receiver()
    runner._session_state(KEY).turn.agent = receiver
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text="/priv hola", source=source), source, KEY)

    # Visible rejection naming the command, and not an echo of the private argument.
    assert reply and "/priv" in reply and "plugin" in reply.lower()
    assert "hola" not in reply
    # Zero characters towards the model, and nothing parked for the next turn.
    assert receiver.steered is None
    assert receiver.redirected is None
    assert receiver.interrupted is None
    assert adapter._pending_messages == {}
    # The handler never ran mid-turn.
    assert "HANDLER RAN" not in reply


@pytest.mark.asyncio
async def test_plugin_command_underscore_form_matches_the_hyphen_registration(plugin_commands):
    """Telegram's underscored autocomplete form hits the same rejection (idle path normalizes too)."""
    runner, _adapter = _busy_runner("steer")
    receiver = _Receiver()
    runner._session_state(KEY).turn.agent = receiver
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text="/priv_chat hola", source=source), source, KEY)

    assert reply and "/priv-chat" in reply
    assert receiver.steered is None and receiver.redirected is None and receiver.interrupted is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "attr"), [("steer", "steered"), ("interrupt", "redirected")])
async def test_plain_text_still_reaches_the_running_turn(plugin_commands, mode, attr):
    """Plain text keeps its busy_input_mode behaviour: the fix intercepts commands only."""
    runner, adapter = _busy_runner(mode)
    receiver = _Receiver()
    runner._session_state(KEY).turn.agent = receiver
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text="hola que tal", source=source), source, KEY)

    assert reply is None
    delivered = getattr(receiver, attr)
    assert delivered is not None and delivered.endswith("hola que tal")
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_core_command_keeps_its_own_busy_policy(plugin_commands):
    """A core command with busy_policy="reject" keeps the core reject text, not the plugin one."""
    runner, _adapter = _busy_runner("steer")
    receiver = _Receiver()
    runner._session_state(KEY).turn.agent = receiver
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text="/model", source=source), source, KEY)

    # /model declares busy_handler="model" → its own mid-run reject text, never the plugin one.
    assert reply == "Agent is running — wait or /stop first, then switch models."
    assert "plugin" not in reply.lower()
    assert receiver.steered is None and receiver.redirected is None and receiver.interrupted is None


@pytest.mark.asyncio
async def test_unknown_command_still_falls_through(plugin_commands):
    """An unresolved name (no core CommandDef, no plugin handler) behaves exactly as before."""
    runner, adapter = _busy_runner("steer")
    receiver = _Receiver()
    runner._session_state(KEY).turn.agent = receiver
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text="/definitely-not-a-command hola", source=source), source, KEY)

    assert reply is None
    assert receiver.steered is not None and receiver.steered.endswith("hola")
    assert adapter._pending_messages == {}


def test_reject_helper_is_inert_for_plain_text_and_core_commands(plugin_commands):
    """The helper answers None unless the name resolves in the plugin registry."""
    runner = GatewayRunner(config=GatewayConfig())
    assert runner._hm_busy_plugin_command_reject(None) is None
    assert runner._hm_busy_plugin_command_reject("model") is None
    assert runner._hm_busy_plugin_command_reject("definitely-not-a-command") is None
    reject_text = runner._hm_busy_plugin_command_reject("priv")
    assert reject_text is not None and "/priv" in reject_text
