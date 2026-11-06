"""In-chat /restart names the running work before it starts the drain.

Parity with the CLI's live-work guard (``hermes_cli/gateway_lifecycle_guards``): the CLI refuses a
restart while turns/cron/delegations are live and offers ``--yes``; the gateway slash path used to
start draining unannounced, so a busy gateway looked hung for up to ``restart_after_turn_timeout``.

Thread routing is inherited, not re-implemented: the handler returns text and the adapter's idle
reply path already anchors it on the inbound event — ``_thread_metadata_for_event(event)`` is
captured BEFORE the handler runs (``gateway/platforms/base.py`` ``_process_message_background``),
so a /restart reply lands in the originating Telegram topic / Discord thread like any other reply.
The parametrized test below pins that, for both platforms.
"""

import pytest
from unittest.mock import MagicMock

from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m1",
    )


@pytest.fixture(autouse=True)
def _hermetic_restart_context(monkeypatch):
    for var in ("INVOCATION_ID", "XPC_SERVICE_NAME", "HERMES_S6_SUPERVISED_CHILD",
                "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)


@pytest.mark.asyncio
async def test_restart_with_live_work_names_it_and_does_not_restart():
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    runner._drain_work_counts = MagicMock(return_value=(2, 1, 0, 0))

    result = await runner._handle_message(_event("/restart"))

    assert result == f"{t('gateway.restart.blocked')}\n2 agent, 1 cron"
    assert not result.startswith("gateway.restart.blocked")  # catalog resolved, not the bare key
    assert "2 agent, 1 cron" in result  # the breakdown reached the text, not just kwargs
    assert result != t("gateway.restart.restarting")  # never the idle wording
    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_restart_force_bypasses_the_guard():
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    runner._drain_work_counts = MagicMock(return_value=(1, 0, 0, 0))

    await runner._handle_message(_event("/restart force"))

    runner.request_restart.assert_called_once()


@pytest.mark.asyncio
async def test_restart_with_no_live_work_still_restarts():
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    runner._drain_work_counts = MagicMock(return_value=(0, 0, 0, 0))

    result = await runner._handle_message(_event("/restart"))

    runner.request_restart.assert_called_once()
    assert result == t("gateway.restart.restarting")
    # No work is draining, so the reply must not claim any: no blocked template, no task noun.
    assert "task(s)" not in result


@pytest.mark.asyncio
async def test_zero_and_live_work_wording_differ():
    # The whole point of the guard: an idle /restart and a busy /restart must not read alike.
    idle_runner, _ = make_restart_runner()
    idle_runner.request_restart = MagicMock(return_value=True)
    idle_runner._drain_work_counts = MagicMock(return_value=(0, 0, 0, 0))
    idle = await idle_runner._handle_message(_event("/restart"))

    busy_runner, _ = make_restart_runner()
    busy_runner.request_restart = MagicMock(return_value=True)
    busy_runner._drain_work_counts = MagicMock(return_value=(4, 0, 0, 0))
    busy = await busy_runner._handle_message(_event("/restart"))

    assert idle != busy
    assert "4" in busy and "4" not in idle


# A Telegram forum topic and a Discord thread: both carry the command's thread on the source, and
# the reply is delivered with it (Telegram maps metadata["thread_id"] -> message_thread_id,
# Discord resolves the channel from it), so /restart needs no thread handling of its own.
@pytest.mark.parametrize("platform,chat_type,thread_id", [
    (Platform.TELEGRAM, "group", "2949"),
    (Platform.DISCORD, "thread", "1131"),
])
@pytest.mark.asyncio
async def test_restart_reply_stays_in_the_command_thread(platform, chat_type, thread_id):
    runner, adapter = make_restart_runner(platform=platform)
    runner.request_restart = MagicMock(return_value=True)
    runner._drain_work_counts = MagicMock(return_value=(1, 0, 0, 0))
    adapter.set_message_handler(runner._handle_message)
    adapter.config.typing_indicator = False  # delivery is the subject here, not the typing loop

    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(platform=platform, chat_type=chat_type, thread_id=thread_id),
        message_id="m1",
    )
    await adapter._process_message_background(event, build_session_key(event.source))

    # The reply the user sees is the blocked-work notice, sent to the command's thread.
    assert adapter.sent_calls[-1] == (
        event.source.chat_id,
        f"{t('gateway.restart.blocked')}\n1 agent",
        {"thread_id": thread_id, "notify": True},
    )
