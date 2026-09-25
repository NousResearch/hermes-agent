"""Canonical Retry's unlocked I/O and final admission-withdrawal fences.

Real SessionAuthority, registry, service, registered RPC and SQL transactions;
only transport and coordinator liveness observations are inert. No startup.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as tasks, hosted_room_links as links
from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import SessionAuthority
from gateway.session_hosted_service import CanonicalHostedRoomService
from tests.gateway.test_canonical_peer_backoff_retry import (
    case, current, healthy, rpc, selector, tick,
)
from tests.gateway.peer_retry_fixtures import LocalRPC
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


def pending(c):
    response = rpc(c, "groups.state")
    assert "error" not in response, response
    return response["result"].get("driver_status", {}).get("pending_actions", [])


def snapshot(c):
    return (current(c), links.load_room_link(c.service.db_path, room_id="room", member_id="peer"),
            dict(c.service._peer_route_status), healthy(c), list(c.peer.staged), list(c.peer.dispatches))


def deferred(c):
    tick(c, 4)
    assert tasks.is_proven_nonadmission(current(c))
    assert any(a["kind"] == "retry" for a in pending(c))
    c.peer.mode = "repaired"


@contextmanager
def held_retry(c, *, rejected=False):
    entered, release = threading.Event(), threading.Event()
    def probe():
        entered.set()
        assert release.wait(20), "test did not release the inert probe"
        if rejected:
            raise PeerRunsHTTPError("revoked during probe", status_code=403,
                                    error_code="invalid_room_grant")
    c.peer.on_probe = probe
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(rpc, c, "groups.retry", selector(c))
        try:
            assert entered.wait(5), "registered Retry never reached the probe"
            yield pool, future, release
        finally:
            release.set()
            c.peer.on_probe = None
        future.result(timeout=5)


def withdrawal(c, gate):
    authority, service = c.authority, c.service
    runtime = service.runtime
    replacement = SessionAuthority(authority.runner, profile_id=authority.profile_id,
        instance_id="replacement", db=authority.db, epoch=authority.epoch)
    # Build the actual replacement before I/O/SQL is held. Its installation,
    # not its constructor's schema work, is the concurrent ownership change.
    installed = None
    if gate == "installed_service":
        installed = CanonicalHostedRoomService(authority, None)
        installed.local_profiles = service.local_profiles
        installed.runtime.clock = runtime.clock
        installed.runtime._thread = SimpleNamespace(is_alive=lambda: True)
    def registry():
        installed = SessionAuthorities(authority.profile_id)
        installed.add(authority.profile_id, replacement)
        authority.runner.session_authorities = installed
    def replace_runtime():
        cold = CanonicalHostedRoomService(authority, None)
        cold.runtime._thread = SimpleNamespace(is_alive=lambda: True)
        cold.runtime.clock = runtime.clock
        service.runtime = cold.runtime
    operations = {
        "drain": lambda: setattr(authority.runner, "_draining", True),
        "stopping": runtime._stop.set,
        "stopped": lambda: setattr(runtime, "_thread", None),
        "installed_service": lambda: setattr(authority, "hosted_room_service", installed),
        "registry": registry,
        "service_authority": lambda: setattr(service, "authority", replacement),
        "runtime": replace_runtime,
    }
    return operations[gate]


def test_retry_probe_leaves_policy_and_other_room_publication_available(case):
    c = case
    deferred(c)
    before = snapshot(c)
    service = c.service
    service.authorize_room(c.connection.actor.subject, "other", create=True)
    service.create_room(room_id="other", name="Healthy", members=service._room("room")["members"])
    local = LocalRPC()
    service.member_rpcs[("other", "healthy", "default", c.connection.actor.subject,
                         c.authority.profile_id)] = local
    with held_retry(c) as (pool, future, release):
        acquired = service._policy_lock.acquire(timeout=2)
        if acquired:
            service._policy_lock.release()
        assert acquired, "peer network probe holds the service-wide policy lock"
        def progress():
            reply = rpc(c, "groups.send", dict(room_id="other", event_id="other-request",
                        payload=dict(text="@healthy continue", thread_id="other-thread")))
            assert "error" not in reply, reply
            tick(c, 3)  # real prepare_room and canonical terminal publication
            return [e for e in service._events("other") if e["kind"] == "message.member"]
        other = pool.submit(progress).result(timeout=5)
        assert len(other) == 1
        assert not future.done() and not release.is_set()
        assert snapshot(c) == before
        release.set()
        assert future.result(timeout=5).get("result", {}).get("retried") is True
    c.now[0] += service.runtime.unavailable_retry_max_seconds + 1
    tick(c, 3)
    assert current(c)["status"] == "settled"
    assert current(c)["payload"] == before[0]["payload"]
    assert current(c)["execution_generation"] == before[0]["execution_generation"] + 1
    assert healthy(c) == before[3] and len(c.peer.dispatches) == 1
    assert [e for e in service._events("other") if e["kind"] == "message.member"] == other


@pytest.mark.parametrize("gate", ["drain", "stopping", "stopped", "registry", "service_authority"])
def test_withdrawn_owner_hides_retry_and_refuses_without_probe(case, gate):
    c = case
    deferred(c)
    withdraw = withdrawal(c, gate)
    before = snapshot(c)
    probes = []
    c.peer.on_probe = lambda: probes.append(True)
    runtime = c.service.runtime
    runtime._wake.clear()
    withdraw()
    offered = pending(c)
    result = rpc(c, "groups.retry", selector(c))
    assert "error" in result, result
    assert not any(a["kind"] == "retry" for a in offered)
    assert probes == [] and snapshot(c) == before
    assert not runtime._wake.is_set()


@pytest.mark.parametrize("gate", ["drain", "stopping", "stopped", "installed_service",
                                 "registry", "service_authority", "runtime"])
@pytest.mark.parametrize("rejected", [False, True], ids=["ready-probe", "rejected-probe"])
def test_withdrawal_during_probe_has_no_requeue_health_write_or_wakeup(case, gate, rejected):
    c = case
    deferred(c)
    withdraw = withdrawal(c, gate)
    before = snapshot(c)
    runtime = c.service.runtime
    runtime._wake.clear()
    with held_retry(c, rejected=rejected) as (_, future, release):
        withdraw()
        assert not any(a["kind"] == "retry" for a in pending(c))
        release.set()
        result = future.result(timeout=5)
    assert "error" in result, result
    assert snapshot(c) == before
    assert not runtime._wake.is_set()
    assert not c.service.runtime._wake.is_set()


@pytest.mark.parametrize("fence", ["route", "owner", "cancel", "lease"])
def test_unlocked_probe_keeps_exact_sql_fences(case, fence):
    c = case
    deferred(c)
    runtime = c.service.runtime
    runtime._wake.clear()
    def route():
        c.service.register_peer_route(room_id="room", member_id="peer",
            route=replace(c.route, trace_id="changed-during-io"), client=c.peer,
            target_url=c.peer.base_url, catalog=c.peer.catalog)
    modifications = {
        "route": route,
        "owner": lambda: c.authority.db._execute_write(lambda conn: conn.execute(
            "UPDATE state_meta SET value='bob' WHERE key='gateway.hosted.owner.v1:room'")),
        "cancel": lambda: tasks.cancel_task(c.service.db_path, c.original["identity"],
            cancel_id="during-io", expected_cancel_generation=0, clock=runtime.clock),
        "lease": lambda: c.now.__setitem__(0, c.now[0] + runtime.lease_ttl_seconds + 1),
    }
    with held_retry(c) as (pool, future, release):
        pool.submit(modifications[fence]).result(timeout=5)
        before = snapshot(c)
        runtime._wake.clear()  # route installation itself may wake; Retry must not
        release.set()
        result = future.result(timeout=5)
    assert "error" in result, result
    assert snapshot(c) == before
    assert not runtime._wake.is_set()


@pytest.mark.parametrize("gate", ["drain", "stopping", "installed_service"])
def test_admission_is_rechecked_inside_requeue_transaction(case, monkeypatch, gate):
    c = case
    deferred(c)
    withdraw = withdrawal(c, gate)
    before = snapshot(c)
    runtime = c.service.runtime
    runtime._wake.clear()
    entered, release = threading.Event(), threading.Event()
    real_requeue = tasks.requeue_deferred_task
    probes = []
    c.peer.on_probe = lambda: probes.append(True)
    def gated_requeue(*args, authorize, **kwargs):
        def after_begin(conn):
            assert conn.in_transaction
            entered.set()
            assert release.wait(10), "test did not release the SQL authorization"
            authorize(conn)
        return real_requeue(*args, authorize=after_begin, **kwargs)
    monkeypatch.setattr(tasks, "requeue_deferred_task", gated_requeue)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rpc, c, "groups.retry", selector(c))
        try:
            assert entered.wait(5), "Retry never reached the real transaction callback"
            assert probes == [True]
            withdraw()
        finally:
            release.set()
        result = future.result(timeout=5)
    assert "error" in result, result
    assert snapshot(c) == before
    assert not runtime._wake.is_set()
