"""Cleanup handover with real Discord SDK serialization and offline HTTP only.

Unlike the broader wire suite, these tests keep the production pacing interval.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tests.test_discord_child_progress_wire import (
    DCP, GatewayTurnMixin, adapter, clients, event, relay, stop, turn, turns,
)


async def until(predicate, timeout=8):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.005)


async def cleanup_turn(t, task):
    tracking = asyncio.create_task(asyncio.sleep(60))
    t._ctx.session_key = None
    await GatewayTurnMixin._run_agent_cleanup_turn_tasks(
        SimpleNamespace(_draining=False), t._ctx, progress_task=task,
        log_task=None, interrupt_monitor=None, _notify_task=None,
        tracking_task=tracking, stream_task=None,
    )
    # Mirrors the actual turn's finally, including cancelled-before-start consumers.
    t.end_progress_turn()


@pytest_asyncio.fixture(autouse=True)
async def cleanup():
    assert DCP.EDIT_INTERVAL == 2.0
    yield
    for t in turns:
        await stop(t)
    for c in clients:
        await c.close()
    turns.clear()
    clients.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize('patch_seconds', [.2, 1.5])
async def test_cleanup_during_default_pacing_preserves_completion(patch_seconds):
    a, w = adapter()
    t = turn(a, current=lambda: True)
    cb = relay(t)
    original_edit = w.edit_message

    async def slow_edit(channel, mid, *, params):
        await asyncio.sleep(patch_seconds)
        return await original_edit(channel, mid, params=params)

    w.edit_message = slow_edit
    task = asyncio.create_task(t.send_progress_messages())
    await event(cb, 'subagent.start')
    await until(lambda: bool(w.sends))
    owner = t._child_progress
    await event(cb, 'subagent.complete', status='completed')
    await asyncio.sleep(.1)  # the real queue consumer has entered its pacing sleep
    await cleanup_turn(t, task)
    assert not owner._dead
    await until(lambda: '✅' in w.text())
    assert owner._cursor == len(owner._parts)
    assert len(w.sends) == 1 and len(w.edits) == 1
    assert not owner._native_live()


@pytest.mark.asyncio
@pytest.mark.parametrize('prior_failures', [0, 2])
async def test_cleanup_cancels_inflight_patch_with_bounded_idempotent_recovery(prior_failures):
    a, w = adapter()
    t = turn(a, current=lambda: True)
    cb = relay(t)
    original_edit = w.edit_message
    started = asyncio.Event()
    cancelled = asyncio.Event()
    attempts, active, peak = 0, 0, 0
    cancelled_at = None
    retry_started = None

    async def blocked_edit(channel, mid, *, params):
        nonlocal attempts, active, peak, cancelled_at, retry_started
        attempts += 1
        active += 1
        peak = max(peak, active)
        try:
            if attempts <= prior_failures:
                raise ConnectionResetError('typed retry before cleanup')
            # Simulate a landed PATCH whose acknowledgement remains in flight.
            result = await original_edit(channel, mid, params=params)
            if attempts == prior_failures + 1:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled_at = time.monotonic()
                    cancelled.set()
                    raise
            else:
                retry_started = time.monotonic()
            return result
        finally:
            active -= 1

    w.edit_message = blocked_edit
    task = asyncio.create_task(t.send_progress_messages())
    await event(cb, 'subagent.start')
    await until(lambda: bool(w.sends))
    owner = t._child_progress
    await event(cb, 'tool.started', 'read_file', 'PENDING.py', {'path': 'PENDING.py'})
    await asyncio.wait_for(started.wait(), 8)
    cursor = owner._cursor
    began = time.monotonic()
    await cleanup_turn(t, task)
    assert 2.8 <= time.monotonic() - began < 4
    assert cancelled.is_set() and active == 0
    assert owner._cursor == cursor  # unacknowledged PATCH never advances the cursor
    if prior_failures == 2:
        assert owner._dead and attempts == 3
        return
    assert not owner._dead and owner._transport_retries == 1
    assert owner._retry_at > cancelled_at
    # Duplicate end-turn notifications must not overlap the old or retained task.
    for _ in range(3):
        t.end_progress_turn()
    await event(cb, 'tool.started', 'read_file', 'LATER.py', {'path': 'LATER.py'})
    await event(cb, 'subagent.complete', status='completed')
    await until(lambda: owner._cursor == len(owner._parts))
    assert retry_started - cancelled_at >= DCP.EDIT_INTERVAL - .05
    assert 'PENDING.py' in w.text() and 'LATER.py' in w.text() and '✅' in w.text()
    assert w.text().count('PENDING.py') == 1
    assert len(w.sends) == 1 and peak == 1
    assert owner._transport_retries == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_stage', ['backoff', 'request_then_backoff'])
async def test_cleanup_preserves_real_429_deadline_and_pending_cursor(monkeypatch, cancel_stage):
    import gateway.delegated_child_progress as progress
    from tests.test_discord_child_progress_wire import discord

    a, w = adapter()
    t = turn(a, current=lambda: True)
    cb = relay(t)
    task = asyncio.create_task(t.send_progress_messages())
    await event(cb, 'subagent.start')
    await until(lambda: bool(w.sends))
    owner = t._child_progress
    await until(lambda: owner._cursor == 1)  # wire acceptance precedes adapter acknowledgement
    cursor = owner._cursor
    original_edit = w.edit_message
    entered_request, release_request, release_backoff = (asyncio.Event() for _ in range(3))
    attempts, sleep_count, clock_offset = 0, 0, 0

    async def rate_limited_once(channel, mid, *, params):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            entered_request.set()
            if cancel_stage == 'request_then_backoff':
                await release_request.wait()
            error = discord.HTTPException(SimpleNamespace(status=429, reason='rate limited'), 'slow down')
            error.retry_after = 30
            raise error
        return await original_edit(channel, mid, params=params)

    async def gated_sleep(delay):
        nonlocal sleep_count
        if delay > 25:
            sleep_count += 1
            await release_backoff.wait()
        else:
            await asyncio.sleep(delay)

    # Advance only this publisher's monotonic clock once the gate is released.
    # The event loop and real cleanup's three-second deadline remain unmodified.
    monkeypatch.setattr(progress, 'time', SimpleNamespace(monotonic=lambda: time.monotonic() + clock_offset))
    monkeypatch.setattr(progress, 'asyncio', SimpleNamespace(**{**vars(asyncio), 'sleep': gated_sleep}))
    w.edit_message = rate_limited_once
    await event(cb, 'tool.started', 'read_file', 'PENDING.py', {'path': 'PENDING.py'})
    await asyncio.wait_for(entered_request.wait(), 3)
    cleanup_task = None
    if cancel_stage == 'request_then_backoff':
        cleanup_task = asyncio.create_task(cleanup_turn(t, task))
        await asyncio.sleep(.05)  # cleanup starts with an actual request in flight
        release_request.set()
    await until(lambda: sleep_count == 1)
    retry_at = owner._retry_at
    assert 29 < retry_at - progress.time.monotonic() <= 30
    if cleanup_task is None:
        began = time.monotonic()
        await cleanup_turn(t, task)
        assert time.monotonic() - began < 1
    else:
        await cleanup_task
    await until(lambda: sleep_count == 2)
    assert not owner._dead and not owner._request_in_flight
    assert owner._cursor == cursor and owner._transport_retries == 1
    assert owner._retry_at == retry_at and attempts == 1
    assert 'PENDING.py' not in w.text()
    for _ in range(3):
        t.end_progress_turn()
    await asyncio.sleep(.05)
    assert sleep_count == 2 and attempts == 1
    clock_offset = 31
    release_backoff.set()
    await until(lambda: 'PENDING.py' in w.text())
    await event(cb, 'tool.started', 'read_file', 'LATER.py', {'path': 'LATER.py'})
    await event(cb, 'subagent.complete', status='completed')
    await until(lambda: owner._cursor == len(owner._parts))
    assert 'LATER.py' in w.text() and '✅' in w.text()
    assert len(w.sends) == 1 and owner._transport_retries == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('post_number', [1, 2])
async def test_cleanup_timeout_of_actual_post_fails_closed_without_replay(post_number):
    a, w = adapter()
    t = turn(a, current=lambda: True, verbose=True)
    cb = relay(t)
    original_send = w.send_message
    started, cancelled = asyncio.Event(), asyncio.Event()
    active = 0

    async def landed_post(channel, *, params):
        nonlocal active
        active += 1
        try:
            result = await original_send(channel, params=params)
            if len(w.sends) == post_number:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return result
        finally:
            active -= 1

    w.send_message = landed_post
    task = asyncio.create_task(t.send_progress_messages())
    await event(cb, 'subagent.start')
    if post_number == 2:
        await until(lambda: bool(w.sends))
        await event(cb, 'tool.started', 'terminal', 'large command', {'command': 'x' * 2200})
    await asyncio.wait_for(started.wait(), 5)
    owner = t._child_progress
    cursor = owner._cursor
    began = time.monotonic()
    await cleanup_turn(t, task)
    assert 2.8 <= time.monotonic() - began < 4
    assert cancelled.is_set() and active == 0
    assert owner._dead and owner._cursor == cursor
    attempts = len(w.attempts)
    await event(cb, 'tool.started', 'read_file', 'LATER.py', {'path': 'LATER.py'})
    await event(cb, 'subagent.complete', status='completed')
    t.end_progress_turn()
    await asyncio.sleep(.1)
    assert len(w.attempts) == attempts == post_number
    assert not owner._native_live()
    assert w.bodies() == '' or ('x' * 2200).startswith(w.bodies())
