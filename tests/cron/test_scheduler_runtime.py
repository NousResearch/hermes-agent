"""Policy and full-lifetime scheduler runtime lifecycle tests."""
from __future__ import annotations

import threading
import time

import pytest

from cron.scheduler_lease import SchedulerOwnershipLease
from cron.scheduler_provider import CronScheduler
from cron.scheduler_runtime import (
    OwnedSchedulerRuntime,
    SchedulerOwnershipPolicy,
    reserve_active_scheduler_provider,
    scheduler_runtime_is_eligible,
)


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


class _Provider(CronScheduler):
    def __init__(self, *, return_immediately=False, ignore_stop=False):
        self.return_immediately = return_immediately
        self.ignore_stop = ignore_stop
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.release = threading.Event()
        self.stop_calls = 0

    @property
    def name(self):
        return "external"

    def start(self, stop_event, **_kwargs):
        self.started.set()
        if self.return_immediately:
            return
        if self.ignore_stop:
            self.release.wait()
        else:
            stop_event.wait()
        self.stopped.set()

    def stop(self):
        self.stop_calls += 1


@pytest.mark.parametrize(
    ("mode", "provider", "runtime", "gateway", "expected"),
    [
        ("gateway", "builtin", "gateway", False, True),
        ("gateway", "builtin", "desktop", False, False),
        ("desktop", "builtin", "gateway", False, False),
        ("desktop", "builtin", "desktop", True, True),
        ("auto", "builtin", "gateway", True, True),
        ("auto", "builtin", "desktop", False, True),
        ("auto", "builtin", "desktop", True, False),
        ("auto", "chronos", "gateway", False, True),
        ("auto", "chronos", "desktop", False, False),
        ("desktop", "chronos", "desktop", False, False),
        ("desktop", "chronos", "gateway", False, False),
    ],
)
def test_scheduler_runtime_eligibility_matrix(mode, provider, runtime, gateway, expected):
    policy = SchedulerOwnershipPolicy(mode, provider)
    assert scheduler_runtime_is_eligible(
        policy, runtime=runtime, same_home_gateway_running=gateway
    ) is expected


def _run_runtime(runtime):
    stop = threading.Event()
    thread = threading.Thread(target=runtime.run, args=(stop,), daemon=True)
    thread.start()
    return stop, thread


def test_external_start_returning_keeps_lease_until_supervisor_shutdown(
    tmp_path, monkeypatch
):
    provider = _Provider(return_immediately=True)
    policy = SchedulerOwnershipPolicy("gateway", "external")
    monkeypatch.setattr(
        "cron.scheduler_runtime.read_scheduler_ownership_policy_strict", lambda: policy
    )
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler_runtime_strict",
        lambda _name: provider,
    )
    runtime = OwnedSchedulerRuntime(
        "gateway", hermes_home=tmp_path, poll_interval=0.01, drain_timeout=0.1
    )
    stop, thread = _run_runtime(runtime)
    _wait_for(provider.started.is_set)
    thread.join(0.05)
    assert thread.is_alive()
    assert SchedulerOwnershipLease.try_acquire(
        hermes_home=tmp_path, owner="desktop", provider="builtin"
    ) is None
    stop.set()
    thread.join(2)
    assert not thread.is_alive()
    assert provider.stop_calls == 1


def test_desktop_auto_yields_and_gateway_retries_without_overlap(tmp_path, monkeypatch):
    policy = SchedulerOwnershipPolicy("auto", "builtin")
    monkeypatch.setattr(
        "cron.scheduler_runtime.read_scheduler_ownership_policy_strict", lambda: policy
    )
    desktop_provider = _Provider()
    gateway_provider = _Provider()
    providers = iter([desktop_provider, gateway_provider])
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler_runtime_strict",
        lambda _name: next(providers),
    )
    gateway_present = threading.Event()
    desktop = OwnedSchedulerRuntime(
        "desktop",
        hermes_home=tmp_path,
        gateway_is_running=gateway_present.is_set,
        poll_interval=0.01,
        drain_timeout=0.2,
    )
    desktop_stop, desktop_thread = _run_runtime(desktop)
    _wait_for(desktop_provider.started.is_set)

    gateway = OwnedSchedulerRuntime(
        "gateway", hermes_home=tmp_path, poll_interval=0.01, drain_timeout=0.2
    )
    gateway_stop, gateway_thread = _run_runtime(gateway)
    time.sleep(0.05)
    assert not gateway_provider.started.is_set()
    gateway_present.set()
    _wait_for(desktop_provider.stopped.is_set)
    _wait_for(gateway_provider.started.is_set)
    assert desktop.active_provider is None

    gateway_stop.set()
    desktop_stop.set()
    gateway_thread.join(2)
    desktop_thread.join(2)
    assert not gateway_thread.is_alive() and not desktop_thread.is_alive()


def test_explicit_owner_changes_handoff_both_directions(tmp_path, monkeypatch):
    state = {"policy": SchedulerOwnershipPolicy("gateway", "builtin")}
    gateway_first = _Provider()
    desktop_provider = _Provider()
    gateway_second = _Provider()
    providers = iter([gateway_first, desktop_provider, gateway_second])
    monkeypatch.setattr(
        "cron.scheduler_runtime.read_scheduler_ownership_policy_strict",
        lambda: state["policy"],
    )
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler_runtime_strict",
        lambda _name: next(providers),
    )

    gateway = OwnedSchedulerRuntime(
        "gateway", hermes_home=tmp_path, poll_interval=0.01, drain_timeout=0.2
    )
    gateway_stop, gateway_thread = _run_runtime(gateway)
    _wait_for(gateway_first.started.is_set)
    desktop = OwnedSchedulerRuntime(
        "desktop", hermes_home=tmp_path, poll_interval=0.01, drain_timeout=0.2
    )
    desktop_stop, desktop_thread = _run_runtime(desktop)

    state["policy"] = SchedulerOwnershipPolicy("desktop", "builtin")
    _wait_for(gateway_first.stopped.is_set)
    _wait_for(desktop_provider.started.is_set)
    assert gateway.active_provider is None

    state["policy"] = SchedulerOwnershipPolicy("gateway", "builtin")
    _wait_for(desktop_provider.stopped.is_set)
    _wait_for(gateway_second.started.is_set)
    assert desktop.active_provider is None

    gateway_stop.set()
    desktop_stop.set()
    gateway_thread.join(2)
    desktop_thread.join(2)
    assert not gateway_thread.is_alive() and not desktop_thread.is_alive()


def test_malformed_policy_stops_active_provider_before_release(tmp_path, monkeypatch):
    state: dict[str, SchedulerOwnershipPolicy | None] = {
        "policy": SchedulerOwnershipPolicy("gateway", "builtin")
    }
    provider = _Provider()
    monkeypatch.setattr(
        "cron.scheduler_runtime.read_scheduler_ownership_policy_strict",
        lambda: state["policy"],
    )
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler_runtime_strict",
        lambda _name: provider,
    )
    runtime = OwnedSchedulerRuntime(
        "gateway", hermes_home=tmp_path, poll_interval=0.01, drain_timeout=0.2
    )
    stop, thread = _run_runtime(runtime)
    _wait_for(provider.started.is_set)
    state["policy"] = None
    _wait_for(provider.stopped.is_set)
    _wait_for(lambda: runtime.active_provider is None)
    lease = SchedulerOwnershipLease.try_acquire(
        hermes_home=tmp_path, owner="desktop", provider="builtin"
    )
    assert lease is not None
    lease.release()
    stop.set()
    thread.join(2)


def test_hung_provider_retains_lease_until_it_really_exits(tmp_path, monkeypatch):
    policy = SchedulerOwnershipPolicy("gateway", "builtin")
    provider = _Provider(ignore_stop=True)
    monkeypatch.setattr(
        "cron.scheduler_runtime.read_scheduler_ownership_policy_strict", lambda: policy
    )
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler_runtime_strict",
        lambda _name: provider,
    )
    runtime = OwnedSchedulerRuntime(
        "gateway", hermes_home=tmp_path, poll_interval=0.01, drain_timeout=0.05
    )
    stop, thread = _run_runtime(runtime)
    _wait_for(provider.started.is_set)
    stop.set()
    time.sleep(0.1)
    assert thread.is_alive()
    assert SchedulerOwnershipLease.try_acquire(
        hermes_home=tmp_path, owner="desktop", provider="builtin"
    ) is None
    provider.release.set()
    thread.join(2)
    assert not thread.is_alive()


def test_inflight_jobs_hold_lease_after_provider_thread_drains(
    tmp_path, monkeypatch
):
    policy = SchedulerOwnershipPolicy("gateway", "builtin")
    provider = _Provider()
    running = {"jobs": {"job-1"}}
    monkeypatch.setattr(
        "cron.scheduler_runtime.read_scheduler_ownership_policy_strict", lambda: policy
    )
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler_runtime_strict",
        lambda _name: provider,
    )
    monkeypatch.setattr(
        "cron.scheduler.get_running_job_ids", lambda: set(running["jobs"])
    )
    runtime = OwnedSchedulerRuntime(
        "gateway", hermes_home=tmp_path, poll_interval=0.01, drain_timeout=0.05
    )
    stop, thread = _run_runtime(runtime)
    _wait_for(provider.started.is_set)
    stop.set()
    _wait_for(provider.stopped.is_set)
    time.sleep(0.08)
    assert thread.is_alive()
    assert SchedulerOwnershipLease.try_acquire(
        hermes_home=tmp_path, owner="desktop", provider="builtin"
    ) is None
    running["jobs"].clear()
    thread.join(2)
    assert not thread.is_alive()


def test_lease_is_held_before_provider_construction(tmp_path, monkeypatch):
    policy = SchedulerOwnershipPolicy("gateway", "builtin")
    observed = []
    monkeypatch.setattr(
        "cron.scheduler_runtime.read_scheduler_ownership_policy_strict", lambda: policy
    )

    def construct(_name):
        observed.append(
            SchedulerOwnershipLease.try_acquire(
                hermes_home=tmp_path, owner="desktop", provider="builtin"
            )
        )
        return _Provider(return_immediately=True)

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler_runtime_strict", construct
    )
    runtime = OwnedSchedulerRuntime(
        "gateway", hermes_home=tmp_path, poll_interval=0.01, drain_timeout=0.1
    )
    stop, thread = _run_runtime(runtime)
    _wait_for(lambda: bool(observed))
    assert observed == [None]
    stop.set()
    thread.join(2)


def test_named_provider_resolution_fails_closed(monkeypatch):
    from cron.scheduler_provider import (
        InProcessCronScheduler,
        resolve_cron_scheduler_runtime_strict,
    )

    monkeypatch.setattr("plugins.cron_providers.load_cron_scheduler", lambda _name: None)
    assert resolve_cron_scheduler_runtime_strict("missing") is None
    assert isinstance(resolve_cron_scheduler_runtime_strict("builtin"), InProcessCronScheduler)


def test_provider_policy_changes_restart_while_gateway_remains_eligible(
    tmp_path, monkeypatch
):
    state = {"policy": SchedulerOwnershipPolicy("gateway", "builtin")}
    builtin_first = _Provider()
    external = _Provider()
    builtin_second = _Provider()
    providers = iter((builtin_first, external, builtin_second))
    monkeypatch.setattr(
        "cron.scheduler_runtime.read_scheduler_ownership_policy_strict",
        lambda: state["policy"],
    )
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler_runtime_strict",
        lambda _name: next(providers),
    )
    runtime = OwnedSchedulerRuntime(
        "gateway", hermes_home=tmp_path, poll_interval=0.01, drain_timeout=0.2
    )
    stop, thread = _run_runtime(runtime)
    _wait_for(builtin_first.started.is_set)

    state["policy"] = SchedulerOwnershipPolicy("gateway", "chronos")
    _wait_for(builtin_first.stopped.is_set)
    _wait_for(external.started.is_set)

    state["policy"] = SchedulerOwnershipPolicy("gateway", "builtin")
    _wait_for(external.stopped.is_set)
    _wait_for(builtin_second.started.is_set)

    stop.set()
    thread.join(2)
    assert not thread.is_alive()


def test_callback_generation_uses_pinned_home_during_shutdown(tmp_path, monkeypatch):
    old_home = tmp_path / "old-home"
    new_home = tmp_path / "new-home"
    old_home.mkdir()
    new_home.mkdir()
    home_state = {"current": old_home}
    policy = SchedulerOwnershipPolicy("gateway", "chronos")
    provider = _Provider()
    monkeypatch.setattr(
        "cron.scheduler_runtime.read_scheduler_ownership_policy_strict", lambda: policy
    )
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler_runtime_strict",
        lambda _name: provider,
    )
    runtime = OwnedSchedulerRuntime(
        "gateway", poll_interval=0.01, drain_timeout=0.05
    )
    monkeypatch.setattr(runtime, "_current_home", lambda: home_state["current"])
    stop, thread = _run_runtime(runtime)
    _wait_for(provider.started.is_set)
    reservation = reserve_active_scheduler_provider(hermes_home=old_home)
    assert reservation is not None
    reservation.release()

    home_state["current"] = new_home
    stop.set()
    thread.join(2)

    assert not thread.is_alive()
    assert reserve_active_scheduler_provider(hermes_home=old_home) is None
    old_home_lease = SchedulerOwnershipLease.try_acquire(
        hermes_home=old_home, owner="desktop", provider="builtin"
    )
    assert old_home_lease is not None
    old_home_lease.release()


def test_callback_reservation_fences_drain_and_lease_release(tmp_path, monkeypatch):
    policy = SchedulerOwnershipPolicy("gateway", "chronos")
    provider = _Provider()
    monkeypatch.setattr(
        "cron.scheduler_runtime.read_scheduler_ownership_policy_strict", lambda: policy
    )
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler_runtime_strict",
        lambda _name: provider,
    )
    runtime = OwnedSchedulerRuntime(
        "gateway", hermes_home=tmp_path, poll_interval=0.01, drain_timeout=0.05
    )
    stop, thread = _run_runtime(runtime)
    _wait_for(provider.started.is_set)
    reservation = reserve_active_scheduler_provider(hermes_home=tmp_path)
    assert reservation is not None and reservation.provider is provider

    stop.set()
    _wait_for(provider.stopped.is_set)
    assert reserve_active_scheduler_provider(hermes_home=tmp_path) is None
    time.sleep(0.08)
    assert thread.is_alive()
    assert SchedulerOwnershipLease.try_acquire(
        hermes_home=tmp_path, owner="desktop", provider="builtin"
    ) is None

    reservation.release()
    thread.join(2)
    assert not thread.is_alive()
    lease = SchedulerOwnershipLease.try_acquire(
        hermes_home=tmp_path, owner="desktop", provider="builtin"
    )
    assert lease is not None
    lease.release()
