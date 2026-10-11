"""``/stop``, ``/new`` and ``/reset`` bypass the active-session guard so they can cancel a
running turn. The base adapter dispatches them to the runner and then cancels the session task.
The cancel must follow the runner's decision, not precede or ignore it: the runner can refuse the
command at admission (sender not on the allowlist) or at slash access (``group_allow_admin_from``
gating), and in a shared session (thread keys carry no user part) a refused sender otherwise
aborts another user's turn, drops their debounced text and promotes a queued follow-up.

The runner marks ``event._interrupt_accepted`` in the stop/reset handlers, which only run past
both gates; the adapter cancels only on that mark. Adapters with no authorization check
registered have no runner verdict to wait for and keep the old unconditional cancel."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, TextDebounceState
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from tests.gateway.test_slash_access_dispatch import _make_event, _make_runner, _make_source
from tests.waiting import async_wait_until

OWNER, MEMBER, STRANGER = "@owner:hs", "@member:hs", "@mallory:hs"
HELD_TEXT = "and add the Q3 numbers"


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
    # A thread in a shared room: the session key carries no user part, so everyone in the
    # thread lands on the same session.
    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:hs", chat_type="group",
                          user_id=user_id, thread_id="$thread")
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source,
                        message_id=f"{user_id}:{text}")


class _Runner:
    """Stands in for GatewayRunner. OWNER and MEMBER pass admission, STRANGER does not.
    Only OWNER may run /stop, /new, /reset (MEMBER is a non-admin under slash gating), and
    the runner marks the event when it accepts the interrupt, like its real stop/reset
    handlers. The owner's turn blocks until cancelled."""

    def __init__(self, *, check=None):
        self.seen: list[tuple[str, str]] = []
        self.turn_started = asyncio.Event()
        self.turn_cancelled = asyncio.Event()
        self.check = check or (lambda user_id, *_a, **_kw: user_id in (OWNER, MEMBER))

    async def handle(self, event: MessageEvent):
        self.seen.append((event.source.user_id, event.text))
        if event.source.user_id not in (OWNER, MEMBER):
            return None  # _hm_admit_event drops an unauthorized sender
        if event.text.startswith("/"):
            if event.source.user_id != OWNER:
                return f"⛔ You can't run {event.text} here."  # _check_slash_access denial text
            event._interrupt_accepted = True
            return f"ok {event.text}"
        self.turn_started.set()
        try:
            await asyncio.Event().wait()  # the turn runs until cancelled
        except asyncio.CancelledError:
            self.turn_cancelled.set()
            raise
        return "done"


async def _busy_session(*, wired: bool = True, check=None):
    adapter, runner = _Adapter(), _Runner(check=check)
    adapter.set_message_handler(runner.handle)
    if wired:
        adapter.set_authorization_check(runner.check)
    await adapter.handle_message(_event(OWNER, "write the report"))
    await asyncio.wait_for(runner.turn_started.wait(), 5)
    key = next(iter(adapter._session_tasks))
    # The owner's follow-up is sitting in the busy-text debounce when the command arrives.
    adapter._text_debounce_store()[key] = TextDebounceState(
        event=_event(OWNER, HELD_TEXT), task=None, first_ts=0.0, last_ts=0.0)
    return adapter, runner, key, adapter._session_tasks[key]


def _assert_turn_untouched(adapter, runner, key, task):
    assert not task.cancelled() and not runner.turn_cancelled.is_set()
    assert key in adapter._active_sessions  # the owner's turn still holds the session
    held = adapter._text_debounce_store().get(key)
    pending = adapter._pending_messages.get(key)
    texts = [held.event.text if held else "", pending.text if pending else ""]
    assert any(HELD_TEXT in text for text in texts), texts  # the owner's text was not dropped


async def _send_and_settle(adapter, runner, user_id, command):
    await adapter.handle_message(_event(user_id, command))
    await async_wait_until(lambda: (user_id, command) in runner.seen, timeout=5)
    await asyncio.sleep(0)  # let the post-dispatch branch run


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/new", "/reset"])
async def test_sender_refused_at_admission_cannot_cancel_another_users_turn(command):
    adapter, runner, key, task = await _busy_session()
    await _send_and_settle(adapter, runner, STRANGER, command)
    _assert_turn_untouched(adapter, runner, key, task)
    task.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/new", "/reset"])
async def test_allowlisted_member_refused_by_slash_access_cannot_cancel_the_turn(command):
    adapter, runner, key, task = await _busy_session()
    await _send_and_settle(adapter, runner, MEMBER, command)
    assert adapter.sent[-1].startswith("⛔")  # the member is told no...
    _assert_turn_untouched(adapter, runner, key, task)  # ...and that is what happens
    task.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("check", [
    pytest.param(lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")), id="raising"),
    pytest.param(lambda *_a, **_kw: None, id="non-boolean"),
])
async def test_an_inconclusive_authorization_check_does_not_cancel(check):
    adapter, runner, key, task = await _busy_session(check=check)
    await _send_and_settle(adapter, runner, STRANGER, "/stop")
    _assert_turn_untouched(adapter, runner, key, task)
    task.cancel()


@pytest.mark.asyncio
async def test_sender_without_a_user_id_cannot_cancel_the_turn():
    # An intake that forwards a slash command with no sender id (some group paths do) must fail
    # closed: the runner drops it at admission, nothing is marked, the turn keeps running.
    adapter, runner, key, task = await _busy_session()
    await _send_and_settle(adapter, runner, "", "/stop")
    _assert_turn_untouched(adapter, runner, key, task)
    task.cancel()


@pytest.mark.asyncio
async def test_accepted_stop_cancels_the_turn_and_drops_the_held_text():
    adapter, runner, key, task = await _busy_session()
    await adapter.handle_message(_event(OWNER, "/stop"))
    await asyncio.wait_for(runner.turn_cancelled.wait(), 5)
    assert task.cancelled() or task.done()
    assert adapter.sent[-1] == "ok /stop"
    assert key not in adapter._text_debounce_store()
    assert (OWNER, HELD_TEXT) not in runner.seen


@pytest.mark.asyncio
async def test_adapter_without_an_authorization_check_keeps_the_unconditional_cancel():
    adapter, runner, _key, task = await _busy_session(wired=False)
    await adapter.handle_message(_event(STRANGER, "/stop"))
    await asyncio.wait_for(runner.turn_cancelled.wait(), 5)
    assert task.cancelled() or task.done()


# --- The real runner: the mark is set only past slash access. -----------------------------

_GATED = {"group_allow_admin_from": ["222"], "group_user_allowed_commands": []}


async def _runner_busy_stop(user_id: str):
    runner = _make_runner(platform_extra=_GATED)
    src = _make_source(user_id=user_id, chat_type="group")
    key = build_session_key(src)
    runner._running_agents[key] = object()
    runner._running_agents_ts[key] = 0
    runner._interrupt_and_clear_session = AsyncMock()
    event = _make_event("/stop", src)
    return runner, event, await runner._handle_message(event)


@pytest.mark.asyncio
async def test_runner_marks_an_admins_busy_stop_as_accepted():
    runner, event, _reply = await _runner_busy_stop("222")
    runner._interrupt_and_clear_session.assert_awaited_once()
    assert getattr(event, "_interrupt_accepted", False) is True


@pytest.mark.asyncio
async def test_runner_leaves_a_slash_denied_busy_stop_unmarked():
    runner, event, reply = await _runner_busy_stop("333")
    assert "⛔" in str(reply)
    runner._interrupt_and_clear_session.assert_not_awaited()
    assert getattr(event, "_interrupt_accepted", False) is False
