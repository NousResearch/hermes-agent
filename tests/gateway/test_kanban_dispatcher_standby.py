"""Exercise gateway leadership through the real watcher and filesystem locks."""

import asyncio
import builtins
import logging
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway import kanban_watchers as watchers
from gateway import kanban_watchers_common as common
from hermes_cli import config, kanban_db as kb, kanban_db_dispatch as dispatch_module


class Clock:
    """Advance watcher sleeps explicitly, without patching the asyncio module."""

    def __init__(self):
        self.sleeps = {}

    async def sleep(self, delay):
        resume = asyncio.get_running_loop().create_future()
        queue = self.sleeps.setdefault(asyncio.current_task(), asyncio.Queue())
        await queue.put((delay, resume))
        await resume

    async def paused(self, task):
        queue = self.sleeps.setdefault(task, asyncio.Queue())
        waiter = asyncio.create_task(queue.get())
        try:
            done, _ = await asyncio.wait(
                (task, waiter), timeout=5, return_when=asyncio.FIRST_COMPLETED,
            )
            assert task not in done, "dispatcher exited instead of remaining on standby"
            assert waiter in done, "watcher did not reach a bounded sleep"
            return waiter.result()
        finally:
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
                "HERMES_KANBAN_DISPATCH_IN_GATEWAY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert kb.kanban_home() == home
    assert kb.kanban_db_path().is_relative_to(home)
    assert Path(watchers.__file__).resolve() == (
        Path(__file__).resolve().parents[2] / "gateway" / "kanban_watchers.py"
    )
    cfg = {"kanban": {"dispatch_interval_seconds": 2, "auto_decompose": False}}
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    monkeypatch.setattr(dispatch_module, "resolve_max_in_progress", lambda value: value)
    # Real empty scratch board: dispatch_once retains its transaction, reapers,
    # and per-board tick lock, but cannot spawn a worker without a task.
    kb.init_db()
    monkeypatch.setattr(kb, "list_boards", lambda **kw: [{"slug": "default"}])
    dispatch = Mock(wraps=dispatch_module.dispatch_once)
    reaper = Mock(return_value=[])
    monkeypatch.setattr(dispatch_module, "dispatch_once", dispatch)
    monkeypatch.setattr(dispatch_module, "reap_worker_zombies", reaper)
    handles = []

    def tracked_open(*args, **kwargs):
        handle = builtins.open(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(common, "open", tracked_open, raising=False)
    clock = Clock()
    monkeypatch.setattr(watchers, "asyncio", SimpleNamespace(
        sleep=clock.sleep, to_thread=asyncio.to_thread, CancelledError=asyncio.CancelledError,
        create_task=asyncio.create_task, shield=asyncio.shield,
    ))
    return SimpleNamespace(
        path=home / "kanban" / ".dispatcher.lock", clock=clock,
        dispatch=dispatch, reaper=reaper, handles=handles, cfg=cfg,
    )


def runner():
    instance = watchers.GatewayKanbanWatchersMixin()
    instance._running = True
    return instance


async def cancel(task, instance):
    task.cancel()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
    # Cleanup even on RED so a failed assertion never strands a real lock.
    instance._release_kanban_dispatcher_lock()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_dies", [False, True])
async def test_standby_takes_over_without_restart(harness, owner_dies, caplog):
    h = harness
    caplog.set_level(logging.INFO, logger="gateway.run")
    owner = None
    process = None
    if owner_dies:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "from gateway.kanban_watchers import _acquire_singleton_lock; "
            "import sys; h, state = _acquire_singleton_lock(sys.argv[1]); "
            "print(state, flush=True); sys.stdin.read()",
            str(h.path), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        )
        assert process.stdout is not None
        assert await asyncio.wait_for(process.stdout.readline(), 5) == b"held\n"
    else:
        owner, state = watchers._acquire_singleton_lock(h.path)
        assert state == "held"
    b = runner()
    task = asyncio.create_task(b._kanban_dispatcher_watcher())
    try:
        # Two full retry cadences, not merely the first failed acquisition.
        for _ in range(4):
            delay, resume = await h.clock.paused(task)
            assert 0 < delay <= 1
            assert not b._owns_kanban_dispatcher_lock()
            h.dispatch.assert_not_called()
            h.reaper.assert_not_called()
            assert all(handle.closed for handle in h.handles if handle is not owner)
            resume.set_result(None)
        _, resume = await h.clock.paused(task)
        if process is not None:
            assert sum("standing by" in record.message for record in caplog.records) == 1
            assert len(h.handles) >= 3
            process.terminate()
            await asyncio.wait_for(process.wait(), 5)
        else:
            watchers._release_singleton_lock(owner)
        resume.set_result(None)
        # Finish the current retry cadence, then the existing startup delay.
        for _ in range(4):
            delay, resume = await h.clock.paused(task)
            if b._owns_kanban_dispatcher_lock():
                assert delay == 5
                resume.set_result(None)
                break
            resume.set_result(None)
        else:
            pytest.fail("standby did not acquire the released lock")
        await h.clock.paused(task)  # first completed dispatch tick
        assert h.dispatch.call_count == 1
        assert b._owns_kanban_dispatcher_lock()

        c = runner()
        third = asyncio.create_task(c._kanban_dispatcher_watcher())
        try:
            await h.clock.paused(third)
            assert not c._owns_kanban_dispatcher_lock()
            assert h.dispatch.call_count == 1
        finally:
            await cancel(third, c)
    finally:
        await cancel(task, b)
        watchers._release_singleton_lock(owner)
        if process is not None and process.returncode is None:
            process.kill()
            await asyncio.wait_for(process.wait(), 5)
    assert all(handle.closed for handle in h.handles)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel", "running_false"])
async def test_waiting_shutdown_closes_every_attempt(harness, stop):
    h = harness
    owner, state = watchers._acquire_singleton_lock(h.path)
    assert state == "held"
    b = runner()
    task = asyncio.create_task(b._kanban_dispatcher_watcher())
    try:
        _, resume = await h.clock.paused(task)
        if stop == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        else:
            b._running = False
            resume.set_result(None)
            await asyncio.wait_for(task, 5)
        assert not b._owns_kanban_dispatcher_lock()
        assert all(handle.closed for handle in h.handles if handle is not owner)
        h.dispatch.assert_not_called()
        h.reaper.assert_not_called()
    finally:
        await cancel(task, b)
        watchers._release_singleton_lock(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["startup", "tick_sleep", "setup_error"])
async def test_leader_releases_lock_on_exit(harness, phase):
    h = harness
    b = runner()
    if phase == "setup_error":
        h.cfg["kanban"]["default_assignee"] = 123  # .strip() fails during setup
    task = asyncio.create_task(b._kanban_dispatcher_watcher())
    try:
        if phase == "setup_error":
            with pytest.raises(AttributeError):
                await asyncio.wait_for(task, 5)
        else:
            delay, resume = await h.clock.paused(task)
            assert delay == 5
            if phase == "tick_sleep":
                resume.set_result(None)
                await h.clock.paused(task)
                assert h.dispatch.call_count == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        assert not b._owns_kanban_dispatcher_lock()
        assert all(handle.closed for handle in h.handles)
        new, state = watchers._acquire_singleton_lock(h.path)
        try:
            assert state == "held"
        finally:
            watchers._release_singleton_lock(new)
    finally:
        await cancel(task, b)


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled_by", ["config", "env"])
async def test_disabled_never_attempts_leadership(harness, monkeypatch, disabled_by):
    h = harness
    if disabled_by == "config":
        h.cfg["kanban"]["dispatch_in_gateway"] = False
    else:
        monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "off")
    await asyncio.wait_for(runner()._kanban_dispatcher_watcher(), 5)
    assert h.handles == []
    h.dispatch.assert_not_called()
    h.reaper.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("work", ["reaper", "dispatch"])
async def test_cancel_does_not_release_while_service_thread_is_running(harness, monkeypatch, work):
    h = harness
    b = runner()
    entered = asyncio.Event()
    released = asyncio.Event()
    work_done = asyncio.Event()
    finish = threading.Event()
    loop = asyncio.get_running_loop()
    target = h.reaper if work == "reaper" else h.dispatch
    original = target._mock_wraps

    def blocking_work(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        try:
            assert finish.wait(5), "test did not release service thread"
            return original(*args, **kwargs) if original else []
        finally:
            loop.call_soon_threadsafe(work_done.set)

    target.side_effect = blocking_work
    real_release = b._release_kanban_dispatcher_lock

    def release():
        real_release()
        released.set()

    monkeypatch.setattr(b, "_release_kanban_dispatcher_lock", release)
    task = asyncio.create_task(b._kanban_dispatcher_watcher())
    try:
        _, resume = await h.clock.paused(task)
        resume.set_result(None)
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert b._owns_kanban_dispatcher_lock()
        other, state = watchers._acquire_singleton_lock(h.path)
        try:
            assert state == "contended"
        finally:
            watchers._release_singleton_lock(other)
        finish.set()
        await asyncio.wait_for(released.wait(), 5)
        assert not b._owns_kanban_dispatcher_lock()
        assert all(handle.closed for handle in h.handles)
    finally:
        finish.set()
        if entered.is_set():
            await asyncio.wait_for(work_done.wait(), 5)
        await cancel(task, b)


@pytest.mark.asyncio
async def test_unavailable_lock_never_dispatches_without_exclusion(harness, monkeypatch, caplog):
    h = harness
    b = runner()
    monkeypatch.setattr(watchers, "_acquire_singleton_lock", lambda path: (None, "unavailable"))
    task = asyncio.create_task(b._kanban_dispatcher_watcher())
    try:
        for _ in range(4):
            delay, resume = await h.clock.paused(task)
            assert delay <= 1
            h.dispatch.assert_not_called()
            h.reaper.assert_not_called()
            resume.set_result(None)
        assert not b._owns_kanban_dispatcher_lock()
        assert "lock unavailable" in caplog.text
    finally:
        await cancel(task, b)


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled_by", ["config", "env"])
async def test_standby_rechecks_disable_before_taking_leadership(harness, monkeypatch, disabled_by):
    h = harness
    owner, state = watchers._acquire_singleton_lock(h.path)
    assert state == "held"
    b = runner()
    task = asyncio.create_task(b._kanban_dispatcher_watcher())
    try:
        for _ in range(2):
            _, resume = await h.clock.paused(task)
            if disabled_by == "config":
                h.cfg["kanban"]["dispatch_in_gateway"] = False
            else:
                monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "off")
            watchers._release_singleton_lock(owner)
            owner = None
            resume.set_result(None)
        await asyncio.wait_for(task, 2)
        h.dispatch.assert_not_called()
        h.reaper.assert_not_called()
        assert not b._owns_kanban_dispatcher_lock()
    finally:
        await cancel(task, b)
        watchers._release_singleton_lock(owner)


@pytest.mark.asyncio
async def test_second_watcher_cannot_drop_existing_leadership(harness):
    h = harness
    b = runner()
    first = asyncio.create_task(b._kanban_dispatcher_watcher())
    try:
        await h.clock.paused(first)
        original_handle = b._kanban_dispatcher_lock_handle
        assert original_handle is not None
        await asyncio.wait_for(b._kanban_dispatcher_watcher(), 1)
        assert b._kanban_dispatcher_lock_handle is original_handle
        assert not original_handle.closed
        other, state = watchers._acquire_singleton_lock(h.path)
        try:
            assert state == "contended"
        finally:
            watchers._release_singleton_lock(other)
    finally:
        await cancel(first, b)


@pytest.mark.asyncio
async def test_many_contended_attempts_do_not_leak_descriptors(harness):
    import psutil

    if not hasattr(psutil.Process(), "num_fds"):
        pytest.skip("POSIX descriptor-count probe")
    h = harness
    owner, state = watchers._acquire_singleton_lock(h.path)
    assert state == "held"
    initial_fds = psutil.Process().num_fds()
    b = runner()
    task = asyncio.create_task(b._kanban_dispatcher_watcher())
    try:
        for _ in range(44):
            _, resume = await h.clock.paused(task)
            resume.set_result(None)
        await h.clock.paused(task)
        assert len(h.handles) >= 22
        assert psutil.Process().num_fds() <= initial_fds
        assert all(handle.closed for handle in h.handles if handle is not owner)
    finally:
        await cancel(task, b)
        watchers._release_singleton_lock(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw, expected", [
    ("invalid", 60), (-10, 1), (1.5, 1.5), (float("nan"), 60), (float("inf"), 60),
])
async def test_standby_retry_interval_is_finite_and_has_a_floor(harness, monkeypatch, raw, expected):
    h = harness
    h.cfg["kanban"]["dispatch_interval_seconds"] = raw
    owner, state = watchers._acquire_singleton_lock(h.path)
    assert state == "held"
    acquire = watchers._acquire_singleton_lock
    attempts = 0

    def bounded_acquire(path):
        nonlocal attempts
        attempts += 1
        assert attempts <= 3, "busy-loop: repeated lock attempts without sleeping"
        return acquire(path)

    monkeypatch.setattr(watchers, "_acquire_singleton_lock", bounded_acquire)
    b = runner()
    task = asyncio.create_task(b._kanban_dispatcher_watcher())
    try:
        slept = 0
        while slept < expected:
            delay, resume = await h.clock.paused(task)
            assert attempts == 1
            assert 0 < delay <= 1
            slept += delay
            resume.set_result(None)
        await h.clock.paused(task)
        assert attempts == 2, "standby never retried after the bounded interval"
        h.dispatch.assert_not_called()
        h.reaper.assert_not_called()
    finally:
        await cancel(task, b)
        watchers._release_singleton_lock(owner)
