"""``/stop``, ``/new`` and ``/reset`` bypass the active-session guard so they can cancel a
running turn. The cancel must belong to the session's authorized users: the runner's
admission check denies the sender inside ``_dispatch_inline_reply`` but the base adapter
then cancelled the task regardless, so any room member sharing a session key (default:
threads, ``thread_sessions_per_user: False``) could abort another user's turn."""

import asyncio

import pytest

from tests.waiting import async_wait_until

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

OWNER, STRANGER = "@owner:hs", "@mallory:hs"


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.MATRIX)
        self.sent: list[str] = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=f"out-{len(self.sent)}")


def _event(user_id: str, text: str) -> MessageEvent:
    # A thread in a shared room: the session key carries no user part, so owner and
    # stranger land on the same session.
    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:hs", chat_type="group",
                          user_id=user_id, thread_id="$thread")
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source,
                        message_id=f"{user_id}:{text}")


class _Runner:
    """Stands in for GatewayRunner: authorizes OWNER only, and the owner's turn blocks
    until cancelled. Records every event that reached the handler."""

    def __init__(self):
        self.seen: list[tuple[str, str]] = []
        self.turn_started = asyncio.Event()
        self.turn_cancelled = asyncio.Event()

    def authorized(self, user_id, chat_type=None, chat_id=None, **_kw) -> bool:
        return user_id == OWNER

    async def handle(self, event: MessageEvent):
        self.seen.append((event.source.user_id, event.text))
        if event.source.user_id != OWNER:
            return None  # what _hm_admit_event does for an unauthorized sender
        if event.text.startswith("/"):
            return f"ok {event.text}"
        self.turn_started.set()
        try:
            await asyncio.Event().wait()  # the turn runs until cancelled
        except asyncio.CancelledError:
            self.turn_cancelled.set()
            raise
        return "done"


async def _busy_session():
    adapter, runner = _Adapter(), _Runner()
    adapter.set_message_handler(runner.handle)
    adapter.set_authorization_check(runner.authorized)
    await adapter.handle_message(_event(OWNER, "write the report"))
    await asyncio.wait_for(runner.turn_started.wait(), 5)
    key = next(iter(adapter._session_tasks))
    return adapter, runner, key, adapter._session_tasks[key]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/new", "/reset"])
async def test_unauthorized_sender_cannot_cancel_another_users_turn(command):
    adapter, runner, key, task = await _busy_session()
    await adapter.handle_message(_event(STRANGER, command))
    # The runner still gets to answer (decline, pairing); the turn is untouched.
    await async_wait_until(lambda: (STRANGER, command) in runner.seen, timeout=5)
    assert not task.cancelled() and not runner.turn_cancelled.is_set()
    assert key in adapter._active_sessions  # the owner's turn is still the one holding the session
    task.cancel()


@pytest.mark.asyncio
async def test_authorized_sender_still_cancels_their_own_turn():
    adapter, runner, _key, task = await _busy_session()
    await adapter.handle_message(_event(OWNER, "/stop"))
    await asyncio.wait_for(runner.turn_cancelled.wait(), 5)
    assert task.cancelled() or task.done()
    assert adapter.sent[-1] == "ok /stop"
