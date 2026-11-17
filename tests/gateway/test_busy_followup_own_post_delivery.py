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

@pytest.mark.asyncio
async def test_busy_follow_up_goal_notice_keeps_its_bubble_cleanup_without_adapter_guard(monkeypatch):
    """A queued authority turn's /goal notice must not erase its generated cleanup slot."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_ingress import _run_own_post_delivery

    runner = GatewayRunner(GatewayConfig())
    adapter = CleanupAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    event = _event("follow-up")
    event._hermes_run_generation = 7
    notices = []

    async def session(*_args, **_kwargs):
        return SimpleNamespace(session_id="goal-test")

    async def manager(*_args, **_kwargs):
        return SimpleNamespace(
            is_active=lambda: True,
            evaluate_after_turn=lambda *_a, **_kw: {"message": "goal reached"},
        )

    async def executor(fn):
        return fn()

    async def notice(_source, message):
        notices.append(message)

    async def no_loop(**_kwargs):
        return None

    monkeypatch.setattr(runner.async_session_store, "get_or_create_session", session)
    monkeypatch.setattr(runner, "_post_turn_manager", manager)
    monkeypatch.setattr(runner, "_run_in_executor_with_context", executor)
    monkeypatch.setattr(runner, "_send_goal_status_notice", notice)
    monkeypatch.setattr(runner, "_post_turn_loop_completion", no_loop)
    monkeypatch.setattr(runner, "_session_key_for_source", lambda _source: KEY)
    monkeypatch.setattr(runner, "_goal_notice_adapter", lambda _source: adapter)

    _register_cleanup(adapter, 7, ["102"])
    assert not adapter._active_sessions  # busy observers have no adapter task guard
    await runner._run_post_turn_hooks(
        agent_result={"final_response": "done"}, source=event.source,
        is_internal=False, event=event,
    )
    assert notices == []  # status follows the reply, not the agent hook
    await _run_own_post_delivery(adapter, KEY, event)
    assert adapter.deleted == ["102"]
    assert notices == ["goal reached"]
    assert KEY not in adapter._post_delivery_callbacks


@pytest.mark.asyncio
async def test_busy_follow_up_loop_notice_keeps_its_bubble_cleanup_without_adapter_guard(monkeypatch):
    """A /loop tick's verdict notice from a queued authority turn keeps the same generation, so
    the chained slot still runs both the notice and the bubble cleanup."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_ingress import _run_own_post_delivery

    runner = GatewayRunner(GatewayConfig())
    adapter = CleanupAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    event = _event("tick")
    event._hermes_run_generation = 7
    notices = []

    async def session(*_args, **_kwargs):
        return SimpleNamespace(session_id="loop-test")

    async def manager(_entry, label, *_a, **_kw):
        if label != "loop completion":
            return None
        return SimpleNamespace(state=SimpleNamespace(awaiting_response=True),
                               complete_tick=lambda _text: {"message": "loop done"})

    async def executor(fn, *args):
        return fn(*args)

    async def notice(_source, message):
        notices.append(message)

    monkeypatch.setattr(runner.async_session_store, "get_or_create_session", session)
    monkeypatch.setattr(runner, "_post_turn_manager", manager)
    monkeypatch.setattr(runner, "_run_in_executor_with_context", executor)
    monkeypatch.setattr(runner, "_send_goal_status_notice", notice)
    monkeypatch.setattr(runner, "_session_key_for_source", lambda _source: KEY)
    monkeypatch.setattr(runner, "_goal_notice_adapter", lambda _source: adapter)

    _register_cleanup(adapter, 7, ["103"])
    assert not adapter._active_sessions
    await runner._run_post_turn_hooks(
        agent_result={"final_response": "done"}, source=event.source,
        is_internal=False, event=event,
    )
    assert notices == []
    await _run_own_post_delivery(adapter, KEY, event)
    assert adapter.deleted == ["103"]
    assert notices == ["loop done"]
    assert KEY not in adapter._post_delivery_callbacks
