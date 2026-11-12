"""A busy follow-up's own post-delivery callbacks run after its reply on authority routes.

A message sent while its chat is busy is admitted to the session FIFO through
``dispatch_shared_busy``; the drain runs it and its observer sends the reply. The drain leaves the
turn's post-delivery slot on the adapter for that native waiter (``_take_post_delivery``), but the
observer never popped it, so the follow-up's temp-bubble cleanup
(``display.platforms.<p>.cleanup_progress``), background-review release and deferred /goal notices
never ran. The turn now carries its run generation on its own event, the drain hands it to the
observer's event with the crash marker, and the observer pops only that generation's slot.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run_turn_bubble_cleanup import register_bubble_cleanup
from gateway.session import SessionSource
from tests.gateway.test_shared_busy_receive_loop import Adapter, _statuses, _wait_for

KEY = "agent:main:telegram:group:-1001:7"
CHAT = "-1001"


class CleanupAdapter(BasePlatformAdapter):
    """Records every delete_message call (the temp-bubble cleanup's only effect)."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.deleted = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="1")

    async def delete_message(self, chat_id, message_id) -> bool:
        self.deleted.append(str(message_id))
        return True

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _event(text):
    return MessageEvent(text=text, source=SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT),
                        message_id=text)


def _register_cleanup(adapter, generation, bubble_ids, *, key=KEY, chat_id=CHAT):
    """The finished turn registers deletion of its progress bubbles, as ``_run_agent`` does."""
    register_bubble_cleanup({"final_response": "done"}, SimpleNamespace(
        _cleanup_adapter=adapter, _cleanup_msg_ids=list(bubble_ids), session_key=key,
        _cleanup_progress=True, source=SimpleNamespace(chat_id=chat_id), run_generation=generation))


@pytest.mark.asyncio
async def test_authority_hands_the_turn_generation_to_the_delivering_event():
    """On an authority route the turn executes on a copy of the inbound event restored from the FIFO
    row; the lifecycle that sends the reply holds the original (the marker adopter) and needs the
    turn's generation along with its crash marker."""
    from gateway.session_ingress import _hand_over_turn_marker
    adopter, executed = _event("inbound"), _event("inbound")
    executed._hermes_run_generation = 7
    executed._gateway_active_turn_session_key, executed._gateway_active_turn_token = KEY, "token"
    authority = SimpleNamespace(marker_adopters={"admission-1": adopter}, pending_deliveries={})
    await _hand_over_turn_marker(authority, "admission-1", executed)
    assert adopter._gateway_active_turn_token == "token"
    assert getattr(adopter, "_hermes_run_generation", None) == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("handed_over, expected", [(True, ["102"]), (False, [])])
async def test_busy_follow_up_observer_runs_only_its_own_turns_slot(handed_over, expected):
    """The observer sends the follow-up's reply, then runs that turn's slot. Without a generation of
    its own it pops nothing: it holds no session guard, and the slot may belong to a turn another
    task is still delivering."""
    from gateway.session_ingress import dispatch_shared_busy
    adapter = CleanupAdapter()

    async def handler(event):
        if handed_over:  # what _hand_over_turn_marker leaves on the observer's event
            event._hermes_run_generation = 2
        _register_cleanup(adapter, 2, ["102"])

    adapter._message_handler = handler
    await dispatch_shared_busy(adapter, _event("follow-up"), KEY)
    assert adapter.deleted == expected
    assert (KEY in adapter._post_delivery_callbacks) is not handed_over


class _AuthorityAdapter(Adapter):
    def __init__(self):
        super().__init__()
        self.deleted = []

    async def delete_message(self, chat_id, message_id):
        self.deleted.append(str(message_id))
        return True


@pytest.mark.asyncio
async def test_authority_busy_follow_up_deletes_its_own_progress_bubbles(monkeypatch):
    """The production shape: real GatewayRunner, adapter, SessionAuthority and drain over SQLite (as
    test_busy_followup_marker_handoff); only the agent turn is a stand-in that binds like
    ``_hmwa_prepare_turn`` and registers its bubble cleanup like ``_run_agent``. The follow-up is
    admitted while the first turn runs and starts only after the first message's task has finished,
    so the generation race of a follow-up that binds while that task still unwinds stays out of
    this test."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result

    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'fixture-user')
    runner = GatewayRunner(GatewayConfig())
    adapter = _AuthorityAdapter()
    adapter.gateway_runner = runner  # as GatewayRunner._create_adapter binds every adapter
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='busy-chat', chat_type='dm', user_id='fixture-user')
    key = adapter._event_session_key(MessageEvent(text='', source=source))
    gate, first_task_done = asyncio.Event(), asyncio.Event()

    async def turn(event, source, quick_key, run_generation):
        await runner._mark_durable_active_turn(event, quick_key)
        if event.text == 'two':
            await asyncio.wait_for(first_task_done.wait(), 5)
        event._hermes_run_generation = run_generation  # _hmwa_prepare_turn's two stamps
        runner._bind_adapter_run_generation(adapter, key, run_generation)
        if event.text == 'one':
            await gate.wait()
        _register_cleanup(adapter, run_generation, ['bubble-' + event.text], key=key, chat_id=source.chat_id)
        execution_result.get()['result'] = {'final_response': 'reply to ' + event.text, 'messages': []}
        return 'reply to ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    finish = adapter._finish_session_task

    def finish_and_signal(session_key, interrupt_event):
        finish(session_key, interrupt_event)
        first_task_done.set()
    adapter._finish_session_task = finish_and_signal

    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    await asyncio.wait_for(adapter.handle_message(MessageEvent(text='two', source=source, message_id='m2')), 5)
    gate.set()
    await _wait_for(lambda: 'reply to two' in adapter.sent and not adapter._background_tasks, 15)
    assert adapter.sent == ['reply to one', 'reply to two']
    assert sorted(adapter.deleted) == ['bubble-one', 'bubble-two']
    assert key not in adapter._post_delivery_callbacks, "the follow-up's slot was left behind"
