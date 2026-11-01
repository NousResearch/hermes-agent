"""Managed worker replacement must fence old plugin loads before accepting a new one."""
from threading import Event

import pytest

from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest


def context(home):
    manifest = PluginManifest(name="worker-consumer", key="worker-consumer", path=str(home))
    return PluginContext(manifest, PluginManager(scope_key=str(home)))


def test_replacement_revokes_and_joins_old_worker_even_without_unload(tmp_path):
    old = context(tmp_path).worker_scope()
    entered = Event()

    def consume(stop):
        entered.set()
        stop.wait(5)

    thread = old.start_thread(consume, name="consumer")
    assert entered.wait(2)
    assert old.activate()
    assert old.is_active()
    try:
        new = context(tmp_path).worker_scope()
        assert not old.is_active()
        assert not thread.is_alive()
        assert not new.is_active()
        assert new.activate()
        assert new.is_active()
        assert new.close()
        assert not new.is_active()
    finally:
        old.close()


def test_closed_scope_cannot_be_reactivated_by_clearing_public_stop_event(tmp_path):
    scope = context(tmp_path).worker_scope()
    assert scope.activate()
    assert scope.close()
    scope.stop_event.clear()
    assert not scope.activate()
    assert not scope.is_active()
    with pytest.raises(RuntimeError, match="closed"):
        scope.start_thread(lambda stop: None, name="forbidden")


def test_uncooperative_worker_blocks_new_load_until_real_thread_exit(tmp_path):
    old = context(tmp_path).worker_scope(timeout=0.01)
    release, started = Event(), Event()

    def stuck(stop):
        started.set()
        release.wait(5)

    thread = old.start_thread(stuck, name="controlled-stuck-worker")
    assert started.wait(2)
    assert old.activate()
    new_context = context(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="have not stopped"):
            new_context.worker_scope(timeout=0.01)
        assert thread.is_alive()
        assert not old.is_active()
        assert not old.activate()
    finally:
        release.set()
        thread.join(2)
        assert old.close()
    fresh = new_context.worker_scope()
    assert fresh.activate()
    assert fresh.close()


def test_ordinary_unload_joins_workers_even_when_another_callback_raises(tmp_path):
    ctx = context(tmp_path)
    scope = ctx.worker_scope()
    thread = scope.start_thread(lambda stop: stop.wait(5), name="unload-worker")
    assert scope.activate()

    def broken():
        raise RuntimeError("controlled independent cleanup failure")

    ctx.on_unload(broken)
    ctx._manager.unload(ctx.plugin_id)
    assert not scope.is_active()
    assert not thread.is_alive()


def test_profile_and_plugin_slots_are_independent(tmp_path):
    first = context(tmp_path / "first").worker_scope()
    second = context(tmp_path / "second").worker_scope()
    assert first.activate() and second.activate()
    assert first.instance_id != second.instance_id
    replacement = context(tmp_path / "first").worker_scope()
    assert not first.is_active()
    assert second.is_active()
    assert replacement.activate()
    replacement.close()
    second.close()


def test_worker_home_is_bound_to_owner_not_calling_thread_profile(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "caller"))
    ctx = context(tmp_path / "owner")
    scope = ctx.worker_scope()
    observed, finished = [], Event()

    def observe(stop):
        observed.append(get_hermes_home())
        finished.set()

    scope.start_thread(observe, name="profile-probe")
    assert finished.wait(2)
    assert scope.close()
    assert observed == [ctx._manager.home_path]


def test_worker_scope_is_cached_per_context(tmp_path):
    ctx = context(tmp_path)
    first = ctx.worker_scope()
    assert first is ctx.worker_scope()
    assert first.activate()
    assert first.close()
    assert ctx.worker_scope() is first
    assert not first.activate()


def test_failed_thread_start_does_not_leave_an_unjoinable_worker(tmp_path, monkeypatch):
    import threading

    scope = context(tmp_path).worker_scope()

    def cannot_start(self):
        raise RuntimeError("controlled thread resource exhaustion")

    with monkeypatch.context() as m:
        m.setattr(threading.Thread, "start", cannot_start)
        with pytest.raises(RuntimeError, match="resource exhaustion"):
            scope.start_thread(lambda stop: None, name="cannot-start")
    assert scope.close()
    fresh = context(tmp_path).worker_scope()
    assert fresh.activate()
    assert fresh.close()


def test_concurrent_first_access_returns_one_cached_scope(tmp_path, monkeypatch):
    import threading
    from agent.memory_provider import spawn_context_thread
    from hermes_cli import plugin_workers

    ctx = context(tmp_path)
    first_acquired, second_entered, release = Event(), Event(), Event()
    real_acquire = plugin_workers.acquire_worker_scope
    results, errors = [], []
    calls = 0
    counter_lock = threading.Lock()

    def controlled_acquire(*args, **kwargs):
        nonlocal calls
        with counter_lock:
            calls += 1
            ordinal = calls
        scope = real_acquire(*args, **kwargs)
        if ordinal == 1:
            first_acquired.set()
            release.wait(10)
        else:
            second_entered.set()
        return scope

    def access():
        try:
            results.append(ctx.worker_scope())
        except Exception as exc:
            errors.append(type(exc).__name__)

    monkeypatch.setattr(plugin_workers, "acquire_worker_scope", controlled_acquire)
    first = spawn_context_thread(access, name="first-access")
    second = spawn_context_thread(access, name="second-access")
    first.start()
    try:
        assert first_acquired.wait(2)
        second.start()
        # Let the unfixed competing acquisition expose itself; the fixed path
        # stays behind the context lock until the first cache write completes.
        second_entered.wait(2)
    finally:
        release.set()
        first.join(2)
        if second.ident is not None:
            second.join(2)
    assert not errors
    assert len(results) == 2
    try:
        assert results[0] is results[1]
        assert calls == 1
    finally:
        for scope in results:
            scope.close()


@pytest.mark.parametrize("timeout", [0, -1, 31, True, float("nan"), float("inf"), "5"])
def test_invalid_timeout_does_not_replace_active_owner(tmp_path, timeout):
    first = context(tmp_path).worker_scope()
    assert first.activate()
    try:
        with pytest.raises(ValueError, match="timeout"):
            context(tmp_path).worker_scope(timeout=timeout)
        assert first.is_active()
    finally:
        first.close()
