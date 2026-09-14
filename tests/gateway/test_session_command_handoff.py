"""Follow-ups belong to the command handoff, not the previous task's cleanup."""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_key


async def wait(event):
    await asyncio.wait_for(event.wait(), timeout=5)


class HandoffAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
        self._busy_text_mode = ""
        self.sent = []
        self.begin_first = asyncio.Event()
        self.begin_first.set()
        self.entered = asyncio.Event()
        self.finish = asyncio.Event()
        self.cleaning = asyncio.Event()
        self.finish_cleanup = asyncio.Event()
        self.followup_started = asyncio.Event()
        self.finish_followup = asyncio.Event()
        self.command_sends = {}
        self.first_task = None

    async def connect(self, **kwargs):
        raise AssertionError("Offline adapter must not connect")

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        if content in self.command_sends:
            entered, release = self.command_sends[content]
            entered.set()
            await wait(release)
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def _stop_typing_refresh(self, *args, **kwargs):
        if asyncio.current_task() is self.first_task:
            self.cleaning.set()
            await wait(self.finish_cleanup)
        await super()._stop_typing_refresh(*args, **kwargs)

    async def _process_message_background(self, event, session_key):
        if event.text == "first":
            await wait(self.begin_first)
        await super()._process_message_background(event, session_key)

    def event(self, text):
        return MessageEvent(text=text, message_id=text, source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="handoff", chat_type="dm",
        ))

    async def settle(self):
        for _ in range(10):
            tasks = list(self._background_tasks)
            if not tasks:
                return
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        raise AssertionError("Adapter tasks did not finish")

    async def cleanup(self, commands):
        for gate in (self.begin_first, self.finish, self.finish_cleanup, self.finish_followup):
            gate.set()
        for _, release in self.command_sends.values():
            release.set()
        await asyncio.wait_for(asyncio.gather(*commands, return_exceptions=True), 5)
        await self.settle()


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "command_exit"), [
    ("/stop", "reply"), ("/new", "reply"), ("/reset", "reply"),
    ("/stop", "cancel"), ("/stop", "raise"),
])
@pytest.mark.parametrize("arrival", ["scheduled", "handler", "cleanup", "finished", "cancel"])
async def test_command_owns_followups_until_handoff(command, command_exit, arrival, caplog):
    adapter = HandoffAdapter()
    key = build_session_key(adapter.event("first").source)
    ack_started, ack_release = asyncio.Event(), asyncio.Event()
    adapter.command_sends["command reply"] = (ack_started, ack_release)
    followup_admissions = []
    command_task = None

    async def handler(event):
        if event.text == command:
            if command_exit == "raise":
                ack_started.set()
                await wait(ack_release)
                raise RuntimeError("command failed")
            return "command reply"
        if event.text == "first":
            adapter.entered.set()
            await wait(adapter.finish)
            return None
        assert command_task is not None
        followup_admissions.append(command_task.done())
        adapter.followup_started.set()
        await wait(adapter.finish_followup)
        return "follow-up reply"

    adapter._message_handler = handler
    try:
        if arrival == "scheduled":
            adapter.begin_first.clear()
        await adapter.handle_message(adapter.event("first"))
        adapter.first_task = adapter._session_tasks[key]
        if arrival != "scheduled":
            await wait(adapter.entered)
        command_task = asyncio.create_task(adapter.handle_message(adapter.event(command)))
        await wait(ack_started)
        if arrival == "scheduled":
            adapter.begin_first.set()
            await wait(adapter.entered)
        if arrival in {"cleanup", "finished"}:
            adapter.finish.set()
            await wait(adapter.cleaning)
        if arrival == "finished":
            adapter.finish_cleanup.set()
            await asyncio.wait_for(adapter.first_task, 5)
        followup = adapter.event("next")
        await adapter.handle_message(followup)
        assert followup._gateway_accepted
        adapter.finish_cleanup.set()
        if arrival != "cancel" and not (arrival == "scheduled" and command_exit != "reply"):
            adapter.finish.set()
            await asyncio.wait_for(adapter.first_task, 5)
        if command_exit == "cancel":
            command_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await command_task
            adapter.finish.set()
        else:
            ack_release.set()
            await asyncio.wait_for(command_task, 5)
            if command_exit == "raise":
                adapter.finish.set()
                assert "command failed" in caplog.text
        adapter.finish_followup.set()
        await adapter.settle()

        assert followup_admissions == [True], "Follow-up ran before the command handed off"
        replies = [] if command_exit == "raise" else ["command reply"]
        assert adapter.sent == replies + ["follow-up reply"]
        assert key not in adapter._pending_messages
        assert key not in adapter._active_sessions
        assert key not in adapter._session_tasks
    finally:
        await adapter.cleanup([command_task] if command_task else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("old_turn_done", [False, True])
@pytest.mark.parametrize("newer_exit", ["reply", "raise", "cancel"])
async def test_overlapping_commands_preserve_followup_ownership(old_turn_done, newer_exit, caplog):
    adapter = HandoffAdapter()
    key = build_session_key(adapter.event("first").source)
    first_ack, release_first_ack = asyncio.Event(), asyncio.Event()
    next_ack, release_next_ack = asyncio.Event(), asyncio.Event()
    adapter.command_sends = {
        "stop reply": (first_ack, release_first_ack),
        "reset reply": (next_ack, release_next_ack),
    }
    commands = []
    admissions = []

    async def handler(event):
        if event.text == "/stop":
            return "stop reply"
        if event.text == "/reset":
            if newer_exit != "reply":
                next_ack.set()
                await wait(release_next_ack)
                raise RuntimeError("newer command failed")
            return "reset reply"
        if event.text == "first":
            adapter.entered.set()
            await wait(adapter.finish)
            return None
        owner = commands[1] if newer_exit == "reply" else commands[0]
        admissions.append(owner.done())
        adapter.followup_started.set()
        await wait(adapter.finish_followup)
        return "follow-up reply"

    adapter._message_handler = handler
    try:
        await adapter.handle_message(adapter.event("first"))
        adapter.first_task = adapter._session_tasks[key]
        await wait(adapter.entered)
        commands.append(asyncio.create_task(adapter.handle_message(adapter.event("/stop"))))
        await wait(first_ack)
        commands.append(asyncio.create_task(adapter.handle_message(adapter.event("/reset"))))
        await wait(next_ack)
        if old_turn_done:
            adapter.finish.set()
            adapter.finish_cleanup.set()
            await asyncio.wait_for(adapter.first_task, 5)
        await adapter.handle_message(adapter.event("next"))
        adapter.finish_cleanup.set()
        if newer_exit == "cancel":
            commands[1].cancel()
            with pytest.raises(asyncio.CancelledError):
                await commands[1]
        else:
            release_next_ack.set()
            await asyncio.wait_for(commands[1], 5)
            if newer_exit == "raise":
                assert "newer command failed" in caplog.text
        if newer_exit != "reply":
            release_first_ack.set()
            await asyncio.wait_for(commands[0], 5)
        await wait(adapter.followup_started)
        followup_task = adapter._session_tasks.get(key)
        assert followup_task is not None, "Follow-up lost its processing owner"
        release_first_ack.set()
        await asyncio.wait_for(commands[0], 5)
        assert not followup_task.cancelled()
        adapter.finish_followup.set()
        await adapter.settle()
        assert admissions == [True]
        replies = ["reset reply"] if newer_exit == "reply" else []
        assert adapter.sent == ["stop reply"] + replies + ["follow-up reply"]
        assert key not in adapter._active_sessions
        assert key not in adapter._session_tasks
    finally:
        await adapter.cleanup(commands)
