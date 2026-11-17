"""The messaging gateway delivers an ``async_rewake`` shell hook's wake (``hook_wake``) from its idle
watcher, into the chat whose turn fired the hook, the way it delivers a background process's watch
match. The hook is an explicit opt-in, so ``display.background_process_notifications: off`` (which
silences background-process chatter) does not swallow it.
"""

import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.run_notifications import INTERNAL_NOTIFICATION_FOOTER
from gateway.session import SessionSource

KEY = "agent:main:telegram:dm:123"


class _Admitting(AsyncMock):
    async def _execute_mock_call(self, event, *args, **kwargs):
        result = await super()._execute_mock_call(event, *args, **kwargs)
        event._gateway_accepted = True
        return result


def _hook_wake(output: str) -> dict:
    """The event exactly as the producer stamps it inside the firing chat's turn."""
    from agent.shell_hooks import ShellHookSpec
    from agent.shell_hooks.rewake import _wake_event
    from gateway.session_context import clear_session_vars, get_session_env, set_session_vars
    tokens = set_session_vars(platform="telegram", chat_id="123", chat_type="dm", user_id="u1",
                              session_key=KEY, session_id="sess-1")
    try:
        spec = ShellHookSpec(event="post_llm_call", command="agent-bus ready", async_rewake=True)
        return {**_wake_event(spec, "sess-1", KEY, get_session_env), "output": output, "elapsed": 3600}
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
async def test_idle_drain_wakes_the_firing_chat_even_with_process_notifications_off(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text("display:\n  background_process_notifications: off\n", encoding="utf-8")
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=_Admitting())
    runner.adapters[Platform.TELEGRAM] = adapter
    runner.session_store._entries[KEY] = SimpleNamespace(origin=SessionSource(
        platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", user_id="u1"))
    completion_queue = queue.Queue()
    completion_queue.put(_hook_wake("2 new messages from lex"))

    await runner._drain_watch_notifications(completion_queue)

    adapter.handle_message.assert_awaited_once()
    synth = adapter.handle_message.await_args.args[0]
    assert synth.internal is True
    assert (synth.source.platform, synth.source.chat_id) == (Platform.TELEGRAM, "123")
    assert synth.text.startswith("[IMPORTANT: Shell hook wake.")
    assert "2 new messages from lex" in synth.text and synth.text.endswith(INTERNAL_NOTIFICATION_FOOTER)
    assert completion_queue.empty()


def test_a_local_session_gets_a_card_so_an_idle_classic_cli_opens_the_turn():
    """An idle classic CLI over the gateway only opens an admitted turn that carries a card; the
    card is a one-line title, the model still reads the full notice."""
    from gateway.run_notifications_ownership import _local_completion_card
    evt = _hook_wake("ping")
    assert _local_completion_card(evt) == {"display_kind": "internal_notification",
                                           "display_text": "Shell Hook Wake: agent-bus ready"}
    identity = GatewayRunner._completion_delivery_identity(evt)
    assert identity is not None and identity != GatewayRunner._completion_delivery_identity(_hook_wake("ping"))
