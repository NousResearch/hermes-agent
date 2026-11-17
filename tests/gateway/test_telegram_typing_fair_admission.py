"""Typing fairness and bounded progress-edit courtesy through the real adapter; no network."""
import asyncio
from itertools import pairwise
from types import SimpleNamespace

import pytest

import gateway.platforms.base as base_module
import plugins.platforms.telegram.adapter as adapter_module
import plugins.platforms.telegram.telegram_draft_budget as draft_budget
import plugins.platforms.telegram.telegram_typing_gate as gate_module
from gateway.config import PlatformConfig


class Clock:
    def __init__(self):
        self.now = 100.0

    def get_running_loop(self):
        return SimpleNamespace(time=lambda: self.now)

    def __getattr__(self, name):
        return getattr(asyncio, name)


class Bot:
    def __init__(self, clock):
        self.clock, self.actions, self.edits = clock, [], []
        self.entered = asyncio.Event()
        self.hold = False

    async def send_chat_action(self, **kwargs):
        self.actions.append((self.clock.now, kwargs['message_thread_id']))
        self.entered.set()
        if self.hold:
            await asyncio.Event().wait()
        return True

    async def edit_message_text(self, **kwargs):
        self.edits.append(kwargs)
        return SimpleNamespace(message_id=900)


@pytest.fixture
def setup(monkeypatch):
    clock = Clock()
    obj = adapter_module.TelegramAdapter(PlatformConfig(enabled=True, token='offline-test'))
    bot = Bot(clock)
    obj._bot, obj._rich_send_disabled = bot, True
    for module in (adapter_module, gate_module, draft_budget):
        monkeypatch.setattr(module, 'asyncio', clock)
    return obj, bot, clock


@pytest.mark.asyncio
@pytest.mark.parametrize('topics', [('11', '22'), ('11', '22', '33'), ('11', '22', '33', '44')])
async def test_stable_topic_order_cannot_starve_sibling_work(setup, topics):
    obj, bot, clock = setup
    for tick in range(40):
        clock.now = 100.0 + tick * 0.5
        for topic in topics:
            await obj.send_typing('4242', {'thread_id': topic})
    assert all(b[0] - a[0] >= 1.0 for a, b in pairwise(bot.actions))
    for topic in topics:
        times = [at for at, actual in bot.actions if actual == int(topic)]
        assert len(times) >= 3, (topic, bot.actions)
        assert max(b - a for a, b in pairwise(times)) < 5.0


@pytest.mark.asyncio
async def test_progress_edits_cannot_hide_typing_or_lose_final_edit(setup):
    obj, bot, clock = setup
    for tick in range(15):
        clock.now = 100.0 + tick
        assert (await obj.edit_message('4242', '900', f'progress {tick}')).success
        await obj.send_typing('4242', {'thread_id': '11'})
    assert len(bot.actions) >= 4
    assert max(b[0] - a[0] for a, b in pairwise(bot.actions)) < 5.0
    assert len(bot.edits) == 15, 'typing must not take the slot needed by preview edits'
    assert (await obj.edit_message('4242', '900', 'complete', finalize=True)).success
    assert bot.edits[-1]['text'] == 'complete'


@pytest.mark.asyncio
@pytest.mark.parametrize('recreate_same_topic', [False, True])
async def test_repeated_short_topics_cannot_take_every_turn(setup, recreate_same_topic):
    obj, bot, clock = setup
    await obj.send_typing('4242', {'thread_id': '11'})
    for tick in range(1, 41):
        clock.now = 100.0 + tick * 0.5
        if tick % 4 == 0:
            topic = '22' if recreate_same_topic else str(100 + tick)
            await obj.send_typing('4242', {'thread_id': topic})
            await obj._stop_typing_with_metadata('4242', {'thread_id': topic})
        await obj.send_typing('4242', {'thread_id': '11'})
    times = [at for at, topic in bot.actions if topic == 11]
    assert len(times) >= 4, bot.actions
    assert max(b - a for a, b in pairwise(times)) < 5.0


@pytest.mark.asyncio
@pytest.mark.parametrize('guard', ['typing_cooldown', 'flood', 'pause', 'draft_budget'])
async def test_progress_slot_priority_preserves_hard_guards(setup, guard):
    obj, bot, clock = setup
    if guard == 'typing_cooldown':
        obj._record_typing_cooldown('4242', OSError('temporary network failure'))
    elif guard == 'flood':
        obj._record_send_flood_cooldown('4242', 60.0)
    elif guard == 'pause':
        obj.pause_typing_for_chat('4242')
    else:
        for when in (100.0, 105.0):
            clock.now = when
            for _ in range(20):
                assert draft_budget.claim_ai_action(obj, '4242') == 0
    clock.now = 110.0
    obj._hold_chat_outbound_slot('4242')
    await obj.send_typing('4242', {'thread_id': '11'})
    assert not bot.actions


@pytest.mark.asyncio
async def test_cancelled_request_spends_admission_but_allows_later_refresh(setup):
    obj, bot, clock = setup
    bot.hold = True
    task = asyncio.create_task(obj.send_typing('4242', {'thread_id': '11'}))
    await asyncio.wait_for(bot.entered.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    bot.hold = False
    clock.now = 100.5
    await obj.send_typing('4242', {'thread_id': '11'})
    assert len(bot.actions) == 1, 'the previous API outcome is unknown'
    clock.now = 103.0
    obj._hold_chat_outbound_slot('4242')
    await obj.send_typing('4242', {'thread_id': '11'})
    assert len(bot.actions) == 2


@pytest.mark.asyncio
async def test_fast_refresh_checks_keep_the_network_timeout(monkeypatch):
    obj = adapter_module.TelegramAdapter(PlatformConfig(enabled=True, token='offline-test'))
    stop = asyncio.Event()
    budgets = []

    class TimerObserver:
        def timeout(self, delay):
            budgets.append(delay)
            return asyncio.timeout(delay)

        def __getattr__(self, name):
            return getattr(asyncio, name)

    async def sent(**_kwargs):
        stop.set()

    obj._bot = SimpleNamespace(send_chat_action=sent)
    monkeypatch.setattr(base_module, 'asyncio', TimerObserver())
    await asyncio.wait_for(obj._keep_typing('4242', interval=0.5,
        metadata={'thread_id': '11'}, stop_event=stop), timeout=2.0)
    assert budgets and budgets[0] == 1.5


@pytest.mark.asyncio
@pytest.mark.parametrize('chat_id, metadata', [
    ('4242', {'thread_id': 22}), (4242, {'thread_id': 22}),
    ('4242', {'message_thread_id': 22}),
])
async def test_paused_waiter_uses_the_original_pause_identity(setup, chat_id, metadata):
    from gateway.platforms.helpers import typing_pause_key

    obj, bot, clock = setup
    await obj.send_typing('4242', {'thread_id': '11'})
    await obj.send_typing(chat_id, metadata)
    # Same key written by the turn lifecycle; it intentionally retains metadata value types.
    obj._typing_paused.add(typing_pause_key(chat_id, metadata))
    clock.now = 102.0
    await obj.send_typing('4242', {'thread_id': '11'})
    assert bot.actions == [(100.0, 11), (102.0, 11)]
