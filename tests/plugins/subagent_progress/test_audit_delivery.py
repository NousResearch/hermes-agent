import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from test_supervision import supervised, review
from test_progress import setup, report


def event_for(ctx, text=None, **metadata):
    return SimpleNamespace(internal=True, text=text or ctx.wakes[-1][0], metadata={
        'hermes_plugin_injection': True, 'hermes_plugin_id': 'subagent-progress',
        'gateway_session_key': 'agent:main:telegram:dm:test-a',
        'gateway_session_id': 'parent-a', **metadata})


def test_wake_is_identifier_only_and_pinned_to_birth_parent(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    rid = report(plugin)['checkpoint_id']
    assert 'Input checked' not in ctx.wakes[0][0]
    assert str(rid) in ctx.wakes[0][0]
    assert ctx.wake_kwargs[0]['expected_session_id'] == 'parent-a'
    assert callable(ctx.wake_kwargs[0]['on_delivery'])
    assert plugin.context(session_id='parent-a', platform='telegram')['context'].count('Input checked') == 1


def test_reviewed_stale_terminal_and_normal_events(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    rid = report(plugin)['checkpoint_id']
    event = event_for(ctx)
    assert plugin.dispatch(event=event) is None
    assert review(plugin, sup, parent, rid)['success']
    assert plugin.dispatch(event=event)['action'] == 'skip'
    plugin.current_child = lambda: child
    report(plugin, completed='Second')
    latest = event_for(ctx)
    plugin.stop(child_subagent_id='sub-a', child_session_id='compressed', child_status='completed')
    assert plugin.dispatch(event=latest)['action'] == 'skip'
    event.internal = False
    assert plugin.dispatch(event=event) is None
    event.internal = True
    event.metadata['hermes_plugin_id'] = 'other-plugin'
    assert plugin.dispatch(event=event) is None


def test_alert_old_generation_drops_and_context_omits_stale(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    clock.now = 450
    sup.tick()
    event = event_for(ctx)
    rid = report(plugin)['checkpoint_id']
    assert review(plugin, sup, parent, rid)['success']
    assert plugin.dispatch(event=event)['action'] == 'skip'
    assert plugin.context(session_id='parent-a', platform='telegram') is None


def test_retry_is_bounded_separate_from_display_and_never_renews(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    callbacks = []
    def enqueue(content, role='tool', **kwargs):
        ctx.wakes.append((content, role, kwargs['session_key']))
        callbacks.append(kwargs['on_delivery'])
        kwargs['on_delivery'](False)
        return True
    ctx.inject_message = enqueue
    deadline = child._delegate_reviewed_deadline.deadline
    rid = report(plugin)['checkpoint_id']
    for _ in range(5):
        sup.tick()
    assert len(callbacks) == 3
    assert len(parent.notices) == 1
    assert child._delegate_reviewed_deadline.deadline == deadline
    with plugin.db() as db:
        receipt = json.loads(db.execute('SELECT receipt FROM wake_deliveries WHERE report=?', (rid,)).fetchone()[0])
    assert receipt['accepted'] is False and receipt['attempts'] == 3


def test_accepted_and_pending_wakes_are_not_retried(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    callbacks = []
    def enqueue(content, role='tool', **kwargs):
        callbacks.append(kwargs['on_delivery'])
        return True
    ctx.inject_message = enqueue
    report(plugin)
    sup.tick()
    assert len(callbacks) == 1
    callbacks[0](True)
    sup.tick()
    assert len(callbacks) == 1


def test_unscheduled_failure_retry_and_expiry_cancel(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    calls = []
    ctx.inject_message = lambda *a, **kw: calls.append(kw) or False
    report(plugin)
    sup.tick()
    assert len(calls) == 2
    clock.now = 701
    sup.tick()
    assert len(calls) == 2


def test_old_sdk_error_is_explicit(supervised, caplog):
    plugin, ctx, parent, child, sup, clock = supervised
    ctx.inject_message = lambda content, role='tool', session_key=None: True
    assert report(plugin)['success']
    assert 'expected_session_id' in caplog.text and 'upgrade' in caplog.text.lower()


def test_goal_and_notice_keep_decision_ahead_of_evidence(setup):
    plugin, ctx, parent, child = setup
    goal = 'Long original goal ' * 1000
    plugin.start(parent_session_id='p', child_session_id='c', child_subagent_id='s2', child_goal=goal)
    with plugin.db() as db:
        assert db.execute("SELECT goal FROM children WHERE subagent='s2'").fetchone()[0] == goal
    report(plugin, completed='c' * 1000, next_step='n' * 600, blocker='CRITICAL ' + 'b' * 590,
           needs_decision=True, evidence=['path' + 'e' * 496] * 6)
    notice = parent.notices[-1]
    assert len(notice) <= 2800
    assert 'CRITICAL' in notice and 'checkpoint #' in notice
    assert notice.index('CRITICAL') < notice.index('Evidence')
    assert '…' in notice


def test_unload_during_async_lookup_never_sends(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    async def lookup(route):
        sup.close()
        return SimpleNamespace(session_id='parent-a', origin=SimpleNamespace())
    adapter = SimpleNamespace(send=AsyncMock())
    runner = SimpleNamespace(async_session_store=SimpleNamespace(lookup_by_session_key=lookup),
        _is_user_authorized_for_source=lambda *a, **kw: True, _adapter_for_source=lambda s: adapter)
    owner = {'route': 'r', 'parent': 'parent-a'}
    payload = {'needs_decision': True, 'subagent_id': 'sub-a'}
    assert not asyncio.run(plugin.deliver_notice(runner, owner, payload, 7))
    adapter.send.assert_not_called()
    assert not ctx.wakes
