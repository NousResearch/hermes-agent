"""A user-defined quick command typed mid-turn is rejected — it never steers/queues/interrupts.

``_hm_busy_slash_or_photo`` resolved commands against the CORE registry only, so a
``quick_commands`` entry (``type: exec`` / ``type: alias``) resolved on the idle path and nowhere
else: mid-turn the raw ``/nightly hola`` text fell through to the busy input mode and entered the
running turn's model context while the exec handler never ran (the sibling leak of
``test_busy_plugin_command_reject.py``). The busy fast-path now consults the config dict too: an
``exec`` quick command is rejected with a visible message, an ``alias`` is expanded to its CORE
target and inherits that command's declared ``busy_policy``.

Reverting the interception makes the ``test_exec_quick_command_mid_turn_*`` and alias tests fail:
the reply becomes None and the text lands in ``steer`` / ``redirect`` / ``interrupt`` (or, for the
marker command, the shell handler runs).
"""

import shlex

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource

KEY = "telegram:dm:user-1"


def _source(user_id: str = "user-1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, user_id=user_id, chat_id="chat-1",
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

    def assert_untouched(self) -> None:
        assert self.steered is None
        assert self.redirected is None
        assert self.interrupted is None


def _busy_runner(quick_commands: dict, mode: str = "steer"):
    runner = GatewayRunner(config=GatewayConfig())
    runner.config.quick_commands = quick_commands
    runner._busy_input_mode = mode
    runner._busy_text_mode = mode
    adapter = _Adapter()
    runner._delivery_adapter_for = lambda source: adapter  # type: ignore[method-assign]
    receiver = _Receiver()
    runner._session_state(KEY).turn.agent = receiver
    return runner, adapter, receiver


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["steer", "queue", "interrupt"])
async def test_exec_quick_command_mid_turn_is_rejected_without_reaching_the_model(tmp_path, mode):
    """The shell command never runs and zero characters reach the model, in every busy mode."""
    marker = tmp_path / "exec-ran"
    runner, adapter, receiver = _busy_runner(
        {"nightly": {"type": "exec", "command": f"touch {shlex.quote(str(marker))}"}}, mode)
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text="/nightly hola", source=source), source, KEY)

    # Visible rejection naming the typed command; the argument (the operator's prompt) is not echoed.
    assert reply and "/nightly" in reply and "quick command" in reply.lower()
    assert "hola" not in reply
    # Zero characters towards the model, nothing parked for the next turn, handler never ran.
    receiver.assert_untouched()
    assert adapter._pending_messages == {}
    assert not marker.exists()


@pytest.mark.asyncio
async def test_alias_quick_command_mid_turn_inherits_the_target_busy_policy():
    """An alias is rewritten to its core target: /model's own mid-run reject text, not a quick one."""
    runner, adapter, receiver = _busy_runner({"switch": {"type": "alias", "target": "/model"}})
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text="/switch gpt-5", source=source), source, KEY)

    assert reply == "Agent is running — wait or /stop first, then switch models."
    assert "quick command" not in " ".join(str(reply).lower().split())
    receiver.assert_untouched()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_alias_quick_command_mid_turn_to_unknown_target_is_rejected():
    """Nothing to dispatch: the alias is rejected whole instead of leaking its raw text."""
    runner, adapter, receiver = _busy_runner({"ghost": {"type": "alias", "target": "/nope-nope"}})
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text="/ghost hola", source=source), source, KEY)

    assert reply and "/ghost" in reply and "/nope-nope" in reply
    assert "hola" not in reply
    receiver.assert_untouched()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_alias_quick_command_mid_turn_to_plugin_command_is_rejected():
    """A plugin target keeps the plugin rejection (its handler is not run beside a live turn)."""
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
    try:
        runner, _adapter, receiver = _busy_runner({"shh": {"type": "alias", "target": "/priv"}})
        source = _source()
        reply = await runner._hm_handle_running_session_message(
            MessageEvent(text="/shh hola", source=source), source, KEY)
    finally:
        table.pop("priv", None)

    assert reply and "/priv" in reply and "plugin" in reply.lower()
    assert "HANDLER RAN" not in reply
    assert "hola" not in reply
    receiver.assert_untouched()


@pytest.mark.asyncio
async def test_quick_command_mid_turn_respects_the_slash_access_gate(tmp_path):
    """A non-admin cannot run mid-turn an exec quick command the idle path would refuse."""
    marker = tmp_path / "exec-ran"
    runner, adapter, receiver = _busy_runner(
        {"nightly": {"type": "exec", "command": f"touch {shlex.quote(str(marker))}"}})
    runner.config.platforms = {
        Platform.TELEGRAM: PlatformConfig(enabled=True, extra={"allow_admin_from": ["admin-1"]})
    }
    source = _source("user-1")  # allowed to chat, not an admin

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text="/nightly hola", source=source), source, KEY)

    assert reply and "nightly" in reply and "admin" in reply.lower()
    assert not marker.exists()
    receiver.assert_untouched()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_plain_text_with_quick_commands_configured_still_reaches_the_running_turn():
    """Regression guard: the quick-command lookup intercepts commands only, never plain chat."""
    runner, adapter, receiver = _busy_runner({"nightly": {"type": "exec", "command": "echo hi"}})
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text="hola que tal", source=source), source, KEY)

    assert reply is None
    assert receiver.steered is not None and receiver.steered.endswith("hola que tal")
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_unsupported_quick_command_type_mid_turn_is_rejected():
    """A malformed entry is a visible rejection too — never a fall-through that leaks text."""
    runner, adapter, receiver = _busy_runner({"weird": {"type": "prompt", "command": "x"}})
    source = _source()

    reply = await runner._hm_handle_running_session_message(
        MessageEvent(text="/weird hola", source=source), source, KEY)

    assert reply and "/weird" in reply
    assert "hola" not in reply
    receiver.assert_untouched()
    assert adapter._pending_messages == {}
