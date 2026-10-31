"""Dependency publication drains the owning gateway through final delivery, without a time cap."""

import asyncio
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.dependency_generation_exit import dependency_generation_watcher
from gateway.run_shutdown import GatewayShutdownMixin
from tests.pm.generation_fixture import make_install
from tests.waiting import async_wait_until


def make_runner():
    runner = SimpleNamespace(
        _primary_profile_name="default",
        _running=True,
        _draining=False,
        _restart_task_started=False,
        _background_tasks=set(),
        adapters={},
        _profile_adapters={},
        session_runtime_descriptor={},
        _running_agent_count=lambda: 0,
        _active_cron_job_count=lambda: 0,
        _active_api_run_count=lambda: 0,
        _active_deferred_agent_worker_count=lambda: 0,
        _mark_api_runs_shutdown_requested=Mock(),
    )
    runner._active_work_count = MethodType(
        GatewayShutdownMixin._active_work_count, runner
    )
    runner.request_restart = Mock()
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery_owner", ["adapter", "replay"])
async def test_generation_change_closes_admission_but_waits_for_final_delivery(
    tmp_path, monkeypatch, delivery_owner
):
    install, runner = make_install(tmp_path, monkeypatch), make_runner()
    install.publish("new")
    release = asyncio.Event()
    pending = asyncio.create_task(release.wait())
    if delivery_owner == "adapter":
        runner._profile_adapters = {
            "work": {"slack": SimpleNamespace(_session_tasks={"turn": pending})}
        }
    else:
        runner._delivery_work = {pending: 1}
    watcher = asyncio.create_task(
        dependency_generation_watcher(
            runner, observer=install.observer, owned=True, poll_s=0
        )
    )
    runner._background_tasks.add(watcher)
    try:
        assert await async_wait_until(lambda: runner._draining, timeout=10)
        runner.request_restart.assert_not_called()
        release.set()
        await pending
        await asyncio.wait_for(watcher, timeout=10)
        runner.request_restart.assert_called_once_with(via_service=True)
    finally:
        pending.cancel()
        watcher.cancel()
        await asyncio.gather(pending, watcher, return_exceptions=True)


@pytest.mark.asyncio
async def test_active_work_then_restart_uses_latest_publication_and_no_forced_deadline(
    tmp_path, monkeypatch
):
    install, runner = make_install(tmp_path, monkeypatch), make_runner()
    runner.request_restart = MethodType(GatewayShutdownMixin.request_restart, runner)
    runner._await_active_work_before_restart = MethodType(
        GatewayShutdownMixin._await_active_work_before_restart, runner
    )
    runner.stop = AsyncMock()
    install.publish("new")
    busy = [True]
    runner._active_work_count = lambda: int(busy[0])
    probed = asyncio.Event()
    loop = asyncio.get_running_loop()
    original = install.observer.probe

    def probe():
        if runner._draining:
            loop.call_soon_threadsafe(probed.set)
        return original()

    monkeypatch.setattr(install.observer, "probe", probe)
    task = asyncio.create_task(
        dependency_generation_watcher(
            runner, observer=install.observer, owned=True, poll_s=0
        )
    )
    try:
        await asyncio.wait_for(probed.wait(), timeout=10)
        assert not runner._restart_task_started
        install.publish(
            "old"
        )  # rollback while active remains safe; still finish the owned drain
        busy[0] = False
        await asyncio.wait_for(task, timeout=10)
        await asyncio.wait_for(runner._restart_task, timeout=10)
        runner.stop.assert_awaited_once_with(
            restart=True, detached_restart=False, service_restart=True
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["manual", "operator-drain", "operator-restart", "unreadable"]
)
async def test_other_lifecycle_owners_and_indeterminate_state_are_not_overridden(
    tmp_path, monkeypatch, mode
):
    install, runner = make_install(tmp_path, monkeypatch), make_runner()
    install.publish("new")
    runner._draining = mode == "operator-drain"
    runner._restart_task_started = mode == "operator-restart"
    if mode == "unreadable":
        (install.state / "publication.json").write_text("{}", encoding="utf-8")
    await dependency_generation_watcher(
        runner, observer=install.observer, owned=mode != "manual", poll_s=0, max_polls=3
    )
    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_a_restarted_watcher_resumes_its_drain_after_a_status_write_failure(
    tmp_path, monkeypatch
):
    install, runner = make_install(tmp_path, monkeypatch), make_runner()
    install.publish("new")
    runner._mark_api_runs_shutdown_requested.side_effect = RuntimeError(
        "status store busy"
    )
    with pytest.raises(RuntimeError, match="status store busy"):
        await dependency_generation_watcher(
            runner, observer=install.observer, owned=True, poll_s=0, max_polls=3
        )
    assert runner._draining and runner._dependency_restart_pending
    runner._mark_api_runs_shutdown_requested.side_effect = None
    await dependency_generation_watcher(
        runner, observer=install.observer, owned=True, poll_s=0, max_polls=2
    )
    runner.request_restart.assert_called_once_with(via_service=True)


@pytest.mark.asyncio
async def test_operator_stop_wins_when_a_generation_probe_is_already_in_flight(
    tmp_path, monkeypatch
):
    install, runner = make_install(tmp_path, monkeypatch), make_runner()
    install.publish("new")
    runner._dependency_restart_pending = runner._draining = True
    entered, release = asyncio.Event(), threading.Event()
    loop, original = asyncio.get_running_loop(), install.observer.probe

    def probe():
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10)
        return original()

    monkeypatch.setattr(install.observer, "probe", probe)
    task = asyncio.create_task(
        dependency_generation_watcher(
            runner, observer=install.observer, owned=True, poll_s=0, max_polls=1
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        runner._running = False
        release.set()
        await asyncio.wait_for(task, timeout=10)
        runner.request_restart.assert_not_called()
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
