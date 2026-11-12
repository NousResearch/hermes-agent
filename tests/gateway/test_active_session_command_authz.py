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

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, TextDebounceState
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from gateway.session_ingress_burst import holds_busy_burst
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
    runner._session_state(key).turn.started_ts = 0
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


@pytest.mark.asyncio
async def test_a_pre_gateway_dispatch_rewrite_does_not_hide_an_accepted_stop(monkeypatch):
    """A plugin's ``rewrite`` hands the runner a copy of the event; the stop handler's verdict must
    still reach the event the adapter dispatched, or an accepted /stop never cancels the turn."""
    async def rewrite(name, **kwargs):
        return [{"action": "rewrite", "text": kwargs["event"].text}] if name == "pre_gateway_dispatch" else []
    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", rewrite)
    runner, event, _reply = await _runner_busy_stop("222")
    runner._interrupt_and_clear_session.assert_awaited_once()
    assert getattr(event, "_interrupt_accepted", False) is True


@pytest.mark.asyncio
async def test_accepted_stop_drops_a_held_burst_before_its_slow_reply(monkeypatch):
    """A non-serial adapter's held busy burst whose window ends while the /stop reply is still in
    flight: dropped at the runner's acceptance, it never commits past the stop's route fence."""
    from gateway.session_ingress_burst import holds_busy_burst
    from tests.gateway.test_busy_burst_serial_reader import _busy_owner, _signal
    adapter, source = _signal(monkeypatch)
    monkeypatch.setattr(adapter, "serial_receive_loop", False, raising=False)  # Telegram/Matrix-like
    owner = await _busy_owner(monkeypatch, adapter, source, busy_text_mode="queue", busy_text_timing=(0.3, 1.0))
    burst = asyncio.create_task(adapter.handle_message(MessageEvent(text="B", source=source, message_id="B")))
    await async_wait_until(lambda: holds_busy_burst(adapter, owner.key), timeout=5)

    async def slow_send(chat_id, content, reply_to=None, metadata=None):
        await async_wait_until(lambda: not holds_busy_burst(adapter, owner.key), timeout=5)
        return SendResult(success=True, message_id="sent")
    monkeypatch.setattr(adapter, "send", slow_send)
    await adapter.handle_message(MessageEvent(text="/stop", source=source, message_id="S"))
    assert await owner.settle(1) == []
    burst.cancel()


@pytest.mark.asyncio
async def test_held_text_runs_when_the_owner_finishes_during_a_refusal_reply():
    """The owner's turn ends while the adapter is still sending a refused member's /stop reply: no
    owner task is left to drain the held follow-up, so it must run now, not sit in the pending slot."""
    adapter, runner, key, task = await _busy_session()
    refuse = runner.handle

    async def handle(event):
        if event.text == "/stop":
            task.cancel()  # the owner's turn ends while the refusal is in flight
            await asyncio.wait({task})
        return await refuse(event)
    adapter.set_message_handler(handle)
    await adapter.handle_message(_event(MEMBER, "/stop"))
    await async_wait_until(lambda: (OWNER, HELD_TEXT) in runner.seen, timeout=5)
    adapter._session_tasks[key].cancel()


@pytest.mark.asyncio
async def test_overlapping_refused_commands_hand_the_session_back_to_its_owner():
    """Two refused /stops overlap and the first finishes first: the owner's guard comes back, not the
    first command's, so the owner's turn ending releases the session."""
    adapter, started, finish = _Adapter(), asyncio.Event(), asyncio.Event()
    gates = {"stop-1": asyncio.Event(), "stop-2": asyncio.Event()}

    async def handle(event):
        if event.text == "/stop":
            await gates[event.message_id].wait()
            return "⛔ You can't run /stop here."  # refused: nothing marks the event
        started.set()
        await finish.wait()
        return "done"
    adapter.set_message_handler(handle)
    adapter.set_authorization_check(lambda *_a, **_kw: True)
    await adapter.handle_message(_event(OWNER, "write the report"))
    await asyncio.wait_for(started.wait(), 5)
    key = next(iter(adapter._session_tasks))
    guards, commands = [adapter._active_sessions[key]], []
    for message_id in gates:
        stop = _event(MEMBER, "/stop")
        stop.message_id = message_id
        commands.append(asyncio.create_task(adapter.handle_message(stop)))
        await async_wait_until(lambda: adapter._active_sessions.get(key) is not guards[-1], timeout=5)
        guards.append(adapter._active_sessions[key])
    for command, gate in zip(commands, gates.values()):
        gate.set()
        await asyncio.wait_for(command, 5)
    assert adapter._active_sessions.get(key) is guards[0]
    finish.set()
    await async_wait_until(lambda: key not in adapter._active_sessions, timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("commands, release", [
    pytest.param([OWNER, MEMBER], [1, 0], id="refusal-during-an-accepted-stop"),
    pytest.param([OWNER, MEMBER, MEMBER], [1, 2, 0], id="refusal-over-a-settled-refusal"),
    pytest.param([MEMBER, MEMBER, MEMBER], [0, 1, 2], id="three-refusals"),
])
async def test_overlapping_commands_hand_the_session_to_whoever_still_settles_it(commands, release):
    """Each refusal hands the session to the earliest command still in flight, else to the owner: an
    accepted /stop's drain then releases it, and refusals alone leave the owner's guard."""
    adapter, runner, key, task = await _busy_session()
    gates = [asyncio.Event() for _ in commands]

    async def handle(event):
        if event.text == "/stop":
            await gates[int(event.message_id)].wait()
        return await runner.handle(event)
    adapter.set_message_handler(handle)
    guards, tasks = [adapter._active_sessions[key]], []
    for index, user_id in enumerate(commands):
        stop = _event(user_id, "/stop")
        stop.message_id = str(index)
        tasks.append(asyncio.create_task(adapter.handle_message(stop)))
        await async_wait_until(lambda: adapter._active_sessions.get(key) is not guards[-1], timeout=5)
        guards.append(adapter._active_sessions[key])
    for index in release:
        gates[index].set()
        await asyncio.wait_for(tasks[index], 5)
    if OWNER in commands:
        assert runner.turn_cancelled.is_set()
        await async_wait_until(lambda: key not in adapter._active_sessions, timeout=5)
    else:
        assert adapter._active_sessions.get(key) is guards[0]
        task.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", [MEMBER, OWNER], ids=["refused", "accepted"])
async def test_a_stop_before_the_owners_task_first_runs_leaves_no_stale_guard(sender):
    """The owner's task is scheduled but has not run when a /stop installs its guard: the task still
    takes the guard it was scheduled under, so a refusal hands that back and the owner's finish
    releases it, and an accepted stop's drain releases its own."""
    adapter, started, finish, gate = _Adapter(), asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handle(event):
        if event.text != "/stop":
            started.set()
            await finish.wait()
            return "done"
        await gate.wait()
        if event.source.user_id != OWNER:
            return "⛔ You can't run /stop here."
        event.accept_interrupt()
        return "ok /stop"

    async def release_once_started():
        await started.wait()
        gate.set()
    adapter.set_message_handler(handle)
    adapter.set_authorization_check(lambda *_a, **_kw: True)
    helper = asyncio.create_task(release_once_started())
    await adapter.handle_message(_event(OWNER, "write the report"))
    key = next(iter(adapter._session_tasks))
    await asyncio.wait_for(adapter.handle_message(_event(sender, "/stop")), 5)  # before the task's first step
    await helper
    finish.set()
    await async_wait_until(lambda: key not in adapter._active_sessions, timeout=5)


@pytest.mark.asyncio
async def test_a_drain_started_under_a_settled_refusal_releases_the_session():
    """Refusals A then B over the owner; A settles, the owner finishes into its pending slot while B
    still replies, so a drain takes the session: it must own the owner's guard, not settled A's."""
    adapter, started, finish, drained = _Adapter(), asyncio.Event(), asyncio.Event(), asyncio.Event()
    gates = {"stop-a": asyncio.Event(), "stop-b": asyncio.Event()}
    seen = []

    async def handle(event):
        seen.append(event.text)
        if event.text == "/stop":
            await gates[event.message_id].wait()
            return "⛔ You can't run /stop here."
        started.set()
        await (finish if event.text == "write the report" else drained).wait()
        return "done"
    adapter.set_message_handler(handle)
    adapter.set_authorization_check(lambda *_a, **_kw: True)
    await adapter.handle_message(_event(OWNER, "write the report"))
    await asyncio.wait_for(started.wait(), 5)
    key = next(iter(adapter._session_tasks))
    commands = {}
    for message_id in gates:
        before = adapter._active_sessions[key]
        stop = _event(MEMBER, "/stop")
        stop.message_id = message_id
        commands[message_id] = asyncio.create_task(adapter.handle_message(stop))
        await async_wait_until(lambda: adapter._active_sessions.get(key) is not before, timeout=5)
    gates["stop-a"].set()
    await asyncio.wait_for(commands["stop-a"], 5)
    adapter._pending_messages[key] = _event(OWNER, HELD_TEXT)
    finish.set()
    await async_wait_until(lambda: HELD_TEXT in seen, timeout=5)  # the drain is running
    gates["stop-b"].set()
    await asyncio.wait_for(commands["stop-b"], 5)
    drained.set()
    await async_wait_until(lambda: key not in adapter._active_sessions, timeout=5)


@pytest.mark.asyncio
async def test_a_message_during_a_refusal_after_the_owner_finished_keeps_the_held_burst(monkeypatch):
    """A shared Signal group: the owner's held burst B, a stranger's refused /stop still replying when
    the owner's turn ends, then the stranger's message Z. Z's stale-lock heal must not drop B: the
    replying command settles the session itself."""
    from plugins.platforms.signal.adapter import SignalAdapter
    from tests.gateway.test_busy_burst_serial_reader import _busy_owner
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+155****0001")
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={"account": "+155****9999",
                                                                "group_sessions_per_user": False}))
    monkeypatch.setattr(adapter, "serial_receive_loop", False, raising=False)

    def member(user_id):
        return adapter.build_source(chat_id="group:g1", chat_type="group", user_id=user_id)
    owner = await _busy_owner(monkeypatch, adapter, member("+155****0001"),
                              GatewayConfig(group_sessions_per_user=False), busy_text_mode="queue",
                              busy_text_timing=(5.0, 10.0))
    executed = owner.runner._handle_message_with_agent
    texts = []

    async def turn(event, *args):
        texts.append(event.text)
        return await executed(event, *args)
    monkeypatch.setattr(owner.runner, "_handle_message_with_agent", turn)
    burst = asyncio.create_task(adapter.handle_message(MessageEvent(text="B", source=member("+155****0001"),
                                                                    message_id="B")))
    await async_wait_until(lambda: holds_busy_burst(adapter, owner.key), timeout=5)
    replying, reply_gate, dispatch = asyncio.Event(), asyncio.Event(), adapter._message_handler

    async def gated(event):
        if event.text == "/stop":
            replying.set()
            await reply_gate.wait()
        return await dispatch(event)
    adapter.set_message_handler(gated)
    stop = asyncio.create_task(adapter.handle_message(MessageEvent(text="/stop", source=member("+155****0002"),
                                                                   message_id="S")))
    await asyncio.wait_for(replying.wait(), 5)
    settled = asyncio.create_task(owner.settle(2))  # the owner's turn A ends during the refusal
    await async_wait_until(lambda: adapter._session_task_is_stale(owner.key), timeout=5)
    await adapter.handle_message(MessageEvent(text="Z", source=member("+155****0002"), message_id="Z"))
    reply_gate.set()
    await asyncio.wait_for(stop, 5)
    await async_wait_until(lambda: "B" in texts, timeout=15, what="the held burst B to run")
    await asyncio.wait_for(settled, 15)
    burst.cancel()
