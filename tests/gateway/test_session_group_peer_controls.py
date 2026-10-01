"""Peer Retry and Discard through the canonical RPC: on evidence only.

Real SessionAuthority, registry, canonical service, registered connection and SQL transactions,
with manual driver cycles. Only the member's gateway and the healthy local member are inert.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as tasks, hosted_room_links as links, hosted_rooms as rooms
from gateway import session_group_peer_controls as peer_controls
from gateway.hosted_room_peer import (
    GatewayRoomCatalog, HostedRoomGrantError, catalog_mapping, decode_room_grant, issue_room_grant)
from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tui_gateway.hosted_room_driver import HostedRoomRuntime
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute


class LocalRPC:
    def resolve_exact(self, **kwargs):
        return {"session_id": "local-session"}

    def resume(self, **kwargs):
        return {"session_id": "local-session"}

    def submit(self, **kwargs):
        kwargs["on_terminal"]({"status": "settled", "text": "reply from local"})
        return {"accepted": True}


class Peer:
    """The member's gateway, as the canonical service sees it through its client."""

    base_url = "http://127.0.0.1:8765"
    secret = b"inert-canonical-retry-secret-32bytes"

    def __init__(self, catalog, scope, db_path):
        self.catalog, self.scope, self.db_path = catalog, scope, db_path
        self.mode, self.revoked, self.on_probe, self.on_stop = "unavailable", False, None, None
        self.dispatches, self.stops, self.stop_status, self.retired = [], [], "cancelled", []

    def prepare(self, **kwargs):
        return {"session_id": "peer-session"}

    def probe(self, *, grant):
        try:
            decode_room_grant(self.secret, grant, permission="dispatch")
        except HostedRoomGrantError as exc:
            raise PeerRunsHTTPError("expired", status_code=401, error_code="invalid_room_grant") from exc
        if self.revoked:
            raise PeerRunsHTTPError("revoked", status_code=403, error_code="room_reauthorization_required")
        if self.on_probe:
            self.on_probe()
        return {"catalog": self.catalog.as_mapping(), **self.scope}

    def dispatch(self, *, dispatch, grant):
        self.dispatches.append(dispatch)
        if self.mode == "unavailable":
            raise PeerRunsHTTPError("connection refused", retryable=True, not_admitted=True)
        if self.mode == "refused-dispatch":
            raise PeerRunsHTTPError("not admitted", status_code=409, not_admitted=True)
        if self.mode == "ambiguous-dispatch":
            raise PeerRunsHTTPError("admission response lost", retryable=True, ambiguous=True)
        if self.mode == "accepted-then-lost":
            rooms.upsert_remote_run_receipt(self.db_path, record={
                **{k: dispatch[k] for k in ("room_id", "home_install_id", "authority_gateway_id",
                                            "authority_epoch", "member_id", "target_install_id",
                                            "target_profile", "task_id", "execution_generation")},
                "run_id": "run-accepted", "session_id": "peer-session"})
            return {"status": "accepted", "run_id": "run-accepted"}
        return {"status": "settled", "text": "reply from peer"}

    def history(self, **kwargs):
        if self.mode == "accepted-then-lost":
            raise PeerRunsHTTPError("status lost", retryable=True)
        return []

    def status(self, **kwargs):
        return {"active": False}

    def revoke_grant_exact(self, *, grant):
        self.retired.append(grant)
        return {"revoked": True}

    def bind_room_scope(self, **scope):
        self.bound_scope = scope

    def stop_receipt(self, *, task_id, execution_generation, grant):
        self.stops.append((task_id, execution_generation))
        if self.on_stop:
            self.on_stop()
        if rooms.remote_run_receipt(self.db_path, record={
                **self.bound_scope, "task_id": task_id, "execution_generation": execution_generation}) is None:
            return None
        if self.stop_status is None:
            raise PeerRunsHTTPError("unreachable", retryable=True)
        return {"run_id": "run-accepted", "status": self.stop_status}


@pytest.fixture
def case(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime threads and network are forbidden here")
    monkeypatch.setattr(HostedRoomRuntime, "start", forbidden)
    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", forbidden)
    with SessionDB(home / "state.db") as db:
        runner = SimpleNamespace(_draining=False, session_authorities=SessionAuthorities(home))
        authority = SessionAuthority(runner, db=db, profile_id=str(home), instance_id="fixture",
                                     epoch=begin_runtime_epoch(db, instance_id="fixture"))
        runner.session_authorities.add(home, authority)
        service = CanonicalHostedRoomService(authority, None)
        authority.hosted_room_service = service
        connection = AuthorityConnection(authority, SimpleNamespace(), {
            "user_id": "alice", "profile_id": str(home), "instance_id": "fixture"})
        service.authorize_room(connection.actor.subject, "room", create=True)
        service.local_profiles = lambda: ("default",)
        catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
            installation_id="peer-install", persistent_process=True, target_profile="default"))
        gateway = rooms.local_authority_gateway_id()
        scope = dict(room_id="room", home_install_id=gateway, authority_gateway_id=gateway,
                     authority_epoch=1, member_id="peer", target_profile="default")
        peer = Peer(catalog, scope, service.db_path)
        grant = issue_room_grant(peer.secret, grant_id="grant", **scope,
                                 target_install_id=catalog.installation_id,
                                 execution_policy_digest=catalog.execution_policy.policy_digest,
                                 ttl_seconds=3600)
        route = PeerMemberRoute(home_install_id=gateway, member_id="peer",
                                target_install_id=catalog.installation_id, target_profile="default",
                                capability_digest=catalog.catalog_digest,
                                execution_policy_digest=catalog.execution_policy.policy_digest,
                                cancellation_scope_id="cancel-room", trace_id="trace-room", grant=grant)
        service.create_room(room_id="room", name="Retry", members=[
            dict(member_id="peer", profile="default", handle="peer", target=dict(
                kind="peer", peer_id="peer-install", installation_id="peer-install", profile="default",
                capability_digest=catalog.catalog_digest)),
            dict(member_id="healthy", profile="default", handle="healthy")])
        service.register_peer_route(room_id="room", member_id="peer", route=route, client=peer,
                                    target_url=peer.base_url, catalog=catalog)
        service.member_rpcs[("room", "healthy", "default", connection.actor.subject, str(home))] = LocalRPC()
        now = [time.time()]
        service.runtime.clock = lambda: now[0]
        # The RPC requires a live coordinator: report one without starting any thread.
        service.runtime._thread = SimpleNamespace(is_alive=lambda: True)
        service.send(room_id="room", event_id="request",
                     payload=dict(thread_id="thread", text="Review this plan together"))
        original = tasks.list_tasks(db.db_path, room_id="room")[0]
        assert original["payload"]["target_member_id"] == "peer"
        yield SimpleNamespace(service=service, connection=connection, peer=peer, route=route, now=now,
                              original=original, authority=authority, home=home)
        service.runtime._thread = None


def tick(c, n=1):
    for _ in range(n):
        c.service.runtime._run_cycle()


def current(c):
    return tasks.get_task(c.service.db_path, c.original["identity"])


def rpc(c, method, params=None):
    return asyncio.run(c.connection.dispatch(dict(id=1, method=method, params=params or {"room_id": "room"})))


def selector(c):
    return dict(room_id="room", member_id="peer", task_id=c.original["identity"].task_id,
                execution_generation=current(c)["execution_generation"])


def actions(c):
    response = rpc(c, "groups.state")
    assert "error" not in response, response
    return response["result"].get("driver_status", {}).get("pending_actions", [])


def link(c):
    return next(l for l in links.load_room_links(c.service.db_path) if (l.room_id, l.member_id) == ("room", "peer"))


def healthy(c):
    return [e for e in c.service._events("room")
            if e["kind"] == "message.member" and e["payload"]["member_id"] == "healthy"]


def snapshot(c):
    return (current(c), links.load_room_links(c.service.db_path), dict(c.service._peer_route_status),
            healthy(c), len(c.peer.dispatches))


def deferred(c):
    tick(c, 4)
    assert tasks.is_proven_nonadmission(current(c)), current(c).get("result")
    assert any(a["kind"] == "retry" for a in actions(c))
    c.peer.mode = "repaired"


@pytest.mark.parametrize("failure", ["unavailable", "refused-dispatch"])
def test_retry_sends_a_proven_unreceived_turn_once_without_replaying_others(case, failure):
    c = case
    c.peer.mode = failure
    tick(c, 4)
    saved = current(c)
    assert saved["status"] == "deferred", saved.get("result")
    assert len(healthy(c)) == 1  # the room moved on to the next member
    events = c.service._events("room")
    assert any(e["kind"] == "turn.deferred" and e["payload"]["member_id"] == "peer" for e in events)
    assert not any(e["kind"] == "turn.failed" and e["payload"]["member_id"] == "peer" for e in events)
    exact = selector(c)
    assert dict(kind="retry", **{k: v for k, v in exact.items() if k != "room_id"}) in actions(c)
    before, sent = healthy(c), len(c.peer.dispatches)
    c.peer.mode = "repaired"
    reply = rpc(c, "groups.retry", exact)
    assert reply.get("result", {}).get("retried") is True, reply
    assert current(c)["status"] == "queued"
    assert current(c)["execution_generation"] == saved["execution_generation"]
    assert link(c).status == "ready"
    assert "error" in rpc(c, "groups.retry", exact)  # one proof, one requeue
    tick(c)
    assert len(c.peer.dispatches) == sent  # the member's backoff still applies
    c.now[0] += c.service.runtime.unavailable_retry_max_seconds + 1
    tick(c, 3)
    done = current(c)
    assert done["status"] == "settled", done.get("result")
    assert done["execution_generation"] == saved["execution_generation"] + 1
    assert done["payload"] == saved["payload"] == c.original["payload"]
    assert len(c.peer.dispatches) == sent + 1
    assert c.peer.dispatches[-1]["prompt"] == saved["payload"]["prompt"]
    assert healthy(c) == before
    assert "error" in rpc(c, "groups.retry", exact)


@pytest.mark.parametrize("state", ["running", "indeterminate", "deferred"])
def test_an_unknown_peer_attempt_never_gets_retry_or_a_blind_discard(case, state):
    c = case
    c.peer.mode = "ambiguous-dispatch"
    tick(c)
    assert current(c)["status"] == "running"
    if state != "running":
        c.now[0] += c.service.runtime.lease_ttl_seconds + 1
        binding = c.service.bindings()[0]
        lease = tasks.acquire_lease(c.service.db_path, room_id="room", gateway_id=binding.gateway_id,
                                    authority_epoch=binding.authority_epoch,
                                    process_generation="unknown-successor", ttl_seconds=600,
                                    clock=lambda: c.now[0])
        tasks.recover_room(c.service.db_path, lease, clock=lambda: c.now[0])
        assert current(c)["status"] == "indeterminate"
        if state == "deferred":
            tasks.defer_indeterminate_task(c.service.db_path, c.original["identity"], lease,
                                           expected_execution_generation=1, expected_cancel_generation=0,
                                           reason="member_unavailable", clock=lambda: c.now[0])
            assert current(c)["result"] == {"reason": "member_unavailable", "retryable": True}
    before = current(c)
    assert not any(a["member_id"] == "peer" for a in actions(c))
    assert rpc(c, "groups.retry", selector(c))["error"]["data"]["reason"] == "unknown_execution"
    forged = rpc(c, "groups.retry", {**selector(c), "nonadmission": True})
    assert forged["error"]["data"]["reason"] == "invalid_params"
    assert "error" in rpc(c, "groups.discard", selector(c))  # no receipt: nothing to stop
    assert current(c) == before
    assert len(c.peer.dispatches) == 1 and c.peer.stops == []


@pytest.mark.parametrize("fence", [
    "generation", "member", "foreign", "capability", "profile", "route", "grant", "expired",
    "reauthorization", "revoked", "owner", "epoch", "room_epoch", "wrong_target", "probe_scope",
    "cancel_race", "lease_race", "route_race", "owner_race", "epoch_race", "room_epoch_race",
    "reauthorization_race"])
def test_retry_rechecks_the_exact_current_authority(case, monkeypatch, fence):
    c = case
    tick(c, 4)
    assert current(c)["status"] == "deferred", current(c).get("result")
    exact = selector(c)
    assert any(a["kind"] == "retry" for a in actions(c))
    c.peer.mode = "repaired"

    def replace_route():
        c.service.register_peer_route(room_id="room", member_id="peer",
                                      route=replace(c.route, trace_id="replacement"), client=c.peer,
                                      target_url=c.peer.base_url, catalog=c.peer.catalog)

    def replace_grant():
        grant = issue_room_grant(c.peer.secret, grant_id="replacement", **c.peer.scope,
                                 target_install_id=c.peer.catalog.installation_id,
                                 execution_policy_digest=c.peer.catalog.execution_policy.policy_digest,
                                 ttl_seconds=3600)
        c.service.register_peer_route(room_id="room", member_id="peer", route=replace(c.route, grant=grant),
                                      client=c.peer, target_url=c.peer.base_url, catalog=c.peer.catalog)

    def change_room_epoch():
        rooms.claim_authority(c.service.db_path, room_id="room", expected_gateway_id=c.route.home_install_id,
                              expected_epoch=1, new_gateway_id=c.route.home_install_id, event_id="reclaim")

    def replace_owner():
        c.authority.db._execute_write(lambda conn: conn.execute(
            "UPDATE state_meta SET value='bob' WHERE key='gateway.hosted.owner.v1:room'"))

    def needs_reauthorization():
        links.mark_room_link_status(c.service.db_path, room_id="room", member_id="peer",
                                    status="needs_reauthorization")
    modifications = {
        "generation": lambda: exact.update(execution_generation=2),
        "member": lambda: exact.update(member_id="healthy"),
        "foreign": lambda: setattr(c.connection, "actor", replace(c.connection.actor, subject="bob")),
        "capability": lambda: setattr(c.connection, "actor", replace(
            c.connection.actor, capabilities=frozenset({"session:read"}))),
        "profile": lambda: setattr(c.connection, "actor", replace(c.connection.actor, profile_id="/foreign")),
        "route": replace_route,
        "grant": replace_grant,
        "expired": lambda: monkeypatch.setattr(time, "time", lambda: c.now[0] + 7200),
        "reauthorization": needs_reauthorization,
        "revoked": lambda: setattr(c.peer, "revoked", True),
        "owner": replace_owner,
        "epoch": lambda: begin_runtime_epoch(c.authority.db, instance_id="new-owner"),
        "room_epoch": change_room_epoch,
        "wrong_target": lambda: c.service.peer_routes.update({("room", "peer"): replace(
            c.route, target_install_id="foreign-install")}),
        "probe_scope": lambda: c.peer.scope.update(member_id="foreign-member"),
        "cancel_race": lambda: setattr(c.peer, "on_probe", lambda: tasks.cancel_task(
            c.service.db_path, c.original["identity"], cancel_id="race", expected_cancel_generation=0,
            clock=lambda: c.now[0])),
        "lease_race": lambda: setattr(c.peer, "on_probe", lambda: c.now.__setitem__(
            0, c.now[0] + c.service.runtime.lease_ttl_seconds + 1)),
        "route_race": lambda: setattr(c.peer, "on_probe", replace_route),
        "owner_race": lambda: setattr(c.peer, "on_probe", replace_owner),
        "epoch_race": lambda: setattr(c.peer, "on_probe", lambda: begin_runtime_epoch(
            c.authority.db, instance_id="new-owner")),
        "room_epoch_race": lambda: setattr(c.peer, "on_probe", change_room_epoch),
        "reauthorization_race": lambda: setattr(c.peer, "on_probe", needs_reauthorization),
    }
    modifications[fence]()
    before, replies = current(c), healthy(c)
    result = rpc(c, "groups.retry", exact)
    assert "error" in result, result
    after = current(c)
    if fence == "cancel_race":
        assert after["status"] == "cancelled"
        assert after["cancel_generation"] == before["cancel_generation"] + 1
    else:
        assert after == before
    assert after["execution_generation"] == 1
    assert len(c.peer.dispatches) == 1 and healthy(c) == replies
    # A failed Retry never marks the route ready; a refused grant marks it for reauthorization.
    expected = ("needs_reauthorization" if fence in {"expired", "revoked", "reauthorization", "reauthorization_race"}
                else "ready" if fence in {"route", "grant", "route_race"} else "unavailable")
    assert link(c).status == expected
    if fence in {"route", "grant", "expired", "reauthorization", "revoked", "wrong_target"}:
        assert not any(a["kind"] == "retry" for a in actions(c))


def test_a_turn_sent_on_an_unstored_route_can_never_be_retried(case):
    c = case
    key = ("room", "peer")
    c.service.peer_routes[key] = replace(c.route, target_install_id="foreign-install")
    tick(c)
    saved = current(c)
    assert tasks.is_proven_nonadmission(saved) and saved["result"]["nonadmission"]["retry_binding"] is None
    c.service.peer_routes[key] = c.route
    tick(c, 3)
    assert not any(a["kind"] == "retry" for a in actions(c))
    c.peer.mode = "repaired"
    assert "error" in rpc(c, "groups.retry", selector(c))
    assert current(c) == saved and len(c.peer.dispatches) == 1


def test_a_failed_publication_keeps_the_proof_for_the_next_scan(case):
    c = case
    publish = c.service.runtime.publish_terminal

    def fail_once(binding, task):
        c.service.runtime.publish_terminal = publish
        raise RuntimeError("publication unavailable after durable deferral")
    c.service.runtime.publish_terminal = fail_once
    tick(c)
    before = current(c)
    assert before["status"] == "deferred" and tasks.is_proven_nonadmission(before)
    assert not any(e["kind"] == "turn.deferred" for e in c.service._events("room"))
    tick(c, 4)  # the room's own rescan publishes it, once
    assert current(c) == before
    assert len([e for e in c.service._events("room") if e["kind"] == "turn.deferred"]) == 1
    assert len(healthy(c)) == 1
    c.peer.mode = "repaired"
    assert rpc(c, "groups.retry", selector(c)).get("result", {}).get("retried") is True
    c.now[0] += c.service.runtime.unavailable_retry_max_seconds + 1
    tick(c, 3)
    assert current(c)["status"] == "settled", current(c).get("result")
    assert len(healthy(c)) == 1 and len(c.peer.dispatches) == 2


@pytest.mark.parametrize("corruption", ["legacy", "boolean", "cancel", "run", "binding"])
def test_a_legacy_or_malformed_proof_is_held(case, corruption):
    c = case
    tick(c, 4)
    result = current(c)["result"]
    mutations = {
        "legacy": lambda: result.pop("nonadmission"),
        "boolean": lambda: result.update(nonadmission=True),
        "cancel": lambda: result["nonadmission"].update(cancel_generation=1),
        "run": lambda: result["nonadmission"].update(run_process_generation="foreign"),
        "binding": lambda: result["nonadmission"].update(retry_binding=None),
    }
    mutations[corruption]()
    c.authority.db._execute_write(lambda conn: conn.execute(
        "UPDATE hosted_room_driver_tasks SET result_json=? WHERE room_id=? AND task_id=?",
        (json.dumps(result), "room", c.original["identity"].task_id)))
    before = current(c)
    assert not any(a["kind"] == "retry" for a in actions(c))
    c.peer.mode = "repaired"
    assert "error" in rpc(c, "groups.retry", selector(c))
    assert current(c) == before and len(c.peer.dispatches) == 1


def test_a_restarted_service_keeps_the_proof_but_never_retries_on_its_own(case):
    c = case
    tick(c, 4)
    before, replies = current(c), healthy(c)
    old = c.service
    old.runtime._thread = None
    c.now[0] += old.runtime.lease_ttl_seconds + old.runtime.unavailable_retry_max_seconds + 1
    cold = CanonicalHostedRoomService(c.authority, None)
    c.authority.hosted_room_service = cold
    cold.local_profiles = old.local_profiles
    cold.member_rpcs = old.member_rpcs
    cold.peer_clients[("room", "peer")] = c.peer  # only the inert I/O boundary is shared
    cold.runtime.clock = lambda: c.now[0]
    cold.runtime._thread = SimpleNamespace(is_alive=lambda: True)
    c.service = cold
    try:
        tick(c, 3)
        assert current(c) == before
        assert len(c.peer.dispatches) == 1 and healthy(c) == replies
        assert any(a["kind"] == "retry" for a in actions(c))
        c.peer.mode = "repaired"
        assert rpc(c, "groups.retry", selector(c)).get("result", {}).get("retried") is True
        tick(c, 3)
        assert current(c)["status"] == "settled", current(c).get("result")
        assert current(c)["execution_generation"] == before["execution_generation"] + 1
        assert current(c)["payload"] == before["payload"]
        assert healthy(c) == replies and len(c.peer.dispatches) == 2
    finally:
        cold.runtime._thread = None


@contextmanager
def held_retry(c, *, rejected=False):
    entered, release = threading.Event(), threading.Event()

    def probe():
        entered.set()
        assert release.wait(20), "test did not release the probe"
        if rejected:
            raise PeerRunsHTTPError("revoked during probe", status_code=403, error_code="invalid_room_grant")
    c.peer.on_probe = probe
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(rpc, c, "groups.retry", selector(c))
        try:
            assert entered.wait(5), "Retry never reached the probe"
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
    installed = None
    if gate == "installed_service":
        installed = CanonicalHostedRoomService(authority, None)
        installed.local_profiles = service.local_profiles
        installed.runtime.clock = runtime.clock
        installed.runtime._thread = SimpleNamespace(is_alive=lambda: True)

    def registry():
        registered = SessionAuthorities(authority.profile_id)
        registered.add(authority.profile_id, replacement)
        authority.runner.session_authorities = registered

    def replace_runtime():
        cold = CanonicalHostedRoomService(authority, None)
        cold.runtime._thread = SimpleNamespace(is_alive=lambda: True)
        cold.runtime.clock = runtime.clock
        service.runtime = cold.runtime
    return {
        "drain": lambda: setattr(authority.runner, "_draining", True),
        "stopping": runtime._stop.set,
        "stopped": lambda: setattr(runtime, "_thread", None),
        "installed_service": lambda: setattr(authority, "hosted_room_service", installed),
        "registry": registry,
        "service_authority": lambda: setattr(service, "authority", replacement),
        "runtime": replace_runtime,
    }[gate]


def test_the_probe_holds_no_lock_other_rooms_need(case):
    c = case
    deferred(c)
    before = snapshot(c)
    service = c.service
    service.authorize_room(c.connection.actor.subject, "other", create=True)
    service.create_room(room_id="other", name="Healthy", members=service._room("room")["members"])
    service.member_rpcs[("other", "healthy", "default", c.connection.actor.subject,
                         c.authority.profile_id)] = LocalRPC()
    with held_retry(c) as (pool, future, release):
        acquired = service._policy_lock.acquire(timeout=2)
        if acquired:
            service._policy_lock.release()
        assert acquired, "the member probe holds the service-wide policy lock"

        def progress():
            reply = rpc(c, "groups.send", dict(room_id="other", event_id="other-request",
                                               payload=dict(text="@healthy continue", thread_id="other-thread")))
            assert "error" not in reply, reply
            tick(c, 3)
            return [e for e in service._events("other") if e["kind"] == "message.member"]
        other = pool.submit(progress).result(timeout=10)
        assert len(other) == 1
        assert not future.done()
        assert snapshot(c) == before
        release.set()
        assert future.result(timeout=5).get("result", {}).get("retried") is True
    c.now[0] += service.runtime.unavailable_retry_max_seconds + 1
    tick(c, 3)
    assert current(c)["status"] == "settled"
    assert current(c)["execution_generation"] == before[0]["execution_generation"] + 1


@pytest.mark.parametrize("gate", ["drain", "stopping", "stopped", "registry", "service_authority"])
def test_a_withdrawn_owner_hides_retry_and_refuses_it_without_a_probe(case, gate):
    c = case
    deferred(c)
    withdraw = withdrawal(c, gate)
    before = snapshot(c)
    probes = []
    c.peer.on_probe = lambda: probes.append(True)
    runtime = c.service.runtime
    runtime._wake.clear()
    withdraw()
    offered = actions(c)
    result = rpc(c, "groups.retry", selector(c))
    assert "error" in result, result
    assert not any(a["kind"] == "retry" for a in offered)
    assert probes == [] and snapshot(c) == before
    assert not runtime._wake.is_set()


@pytest.mark.parametrize("gate", ["drain", "stopping", "stopped", "installed_service", "registry",
                                  "service_authority", "runtime"])
@pytest.mark.parametrize("rejected", [False, True], ids=["ready-probe", "rejected-probe"])
def test_withdrawal_during_the_probe_writes_nothing_and_wakes_nothing(case, gate, rejected):
    c = case
    deferred(c)
    withdraw = withdrawal(c, gate)
    before = snapshot(c)
    runtime = c.service.runtime
    runtime._wake.clear()
    with held_retry(c, rejected=rejected) as (_, future, release):
        withdraw()
        assert not any(a["kind"] == "retry" for a in actions(c))
        release.set()
        result = future.result(timeout=5)
    assert "error" in result, result
    assert snapshot(c) == before
    assert not runtime._wake.is_set() and not c.service.runtime._wake.is_set()


@pytest.mark.parametrize("fence", ["route", "owner", "cancel", "lease"])
def test_changes_during_the_probe_are_caught_by_the_final_transaction(case, fence):
    c = case
    deferred(c)
    runtime = c.service.runtime
    modifications = {
        "route": lambda: c.service.register_peer_route(
            room_id="room", member_id="peer", route=replace(c.route, trace_id="changed-during-io"),
            client=c.peer, target_url=c.peer.base_url, catalog=c.peer.catalog),
        "owner": lambda: c.authority.db._execute_write(lambda conn: conn.execute(
            "UPDATE state_meta SET value='bob' WHERE key='gateway.hosted.owner.v1:room'")),
        "cancel": lambda: tasks.cancel_task(c.service.db_path, c.original["identity"], cancel_id="during-io",
                                            expected_cancel_generation=0, clock=runtime.clock),
        "lease": lambda: c.now.__setitem__(0, c.now[0] + runtime.lease_ttl_seconds + 1),
    }
    with held_retry(c) as (pool, future, release):
        pool.submit(modifications[fence]).result(timeout=5)
        before = snapshot(c)
        runtime._wake.clear()  # installing a route may wake the room itself; Retry must not
        release.set()
        result = future.result(timeout=5)
    assert "error" in result, result
    assert snapshot(c) == before
    assert not runtime._wake.is_set()


@pytest.mark.parametrize("gate", ["drain", "stopping", "installed_service"])
def test_admission_is_rechecked_inside_the_requeue_transaction(case, monkeypatch, gate):
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
            assert release.wait(10), "test did not release the transaction"
            authorize(conn)
        return real_requeue(*args, authorize=after_begin, **kwargs)
    monkeypatch.setattr(tasks, "requeue_deferred_task", gated_requeue)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rpc, c, "groups.retry", selector(c))
        try:
            assert entered.wait(5), "Retry never reached its transaction"
            assert probes == [True]
            withdraw()
        finally:
            release.set()
        result = future.result(timeout=5)
    assert "error" in result, result
    assert snapshot(c) == before
    assert not runtime._wake.is_set()


def unknown_with_receipt(c):
    """An accepted peer attempt whose observation was lost: unknown, with a durable receipt."""
    c.peer.mode = "accepted-then-lost"
    tick(c)
    assert current(c)["status"] == "running"
    c.now[0] += c.service.runtime.lease_ttl_seconds + 1
    tick(c)
    assert current(c)["status"] == "indeterminate"
    c.peer.mode = "repaired"


def test_discard_retires_only_proven_nonadmission_without_contacting_target(case):
    c = case
    tick(c, 4)
    exact = selector(c)
    assert current(c)['status'] == 'deferred' and tasks.is_proven_nonadmission(current(c))
    before = len(c.peer.dispatches)
    assert dict(kind='discard', **{k: v for k, v in exact.items() if k != 'room_id'}) in actions(c)
    from tui_gateway.contracts.groups_bot_relay import GroupsDiscardParams, GroupsDiscardResult
    GroupsDiscardParams.model_validate(exact)
    reply = rpc(c, 'groups.discard', exact)
    GroupsDiscardResult.model_validate(reply.get('result'))
    assert reply.get('result', {}).get('discarded') is True, reply
    assert current(c)['status'] == 'cancelled'
    assert c.peer.stops == [] and len(c.peer.dispatches) == before
    assert rpc(c, 'groups.discard', exact).get('result', {}).get('discarded') is True


def test_accepted_peer_discard_points_to_stop_and_exact_stop_remains_usable(case, monkeypatch):
    c = case
    unknown_with_receipt(c)
    exact = selector(c)
    before = current(c)
    assert not any(a['kind'] == 'discard' for a in actions(c))
    refused = rpc(c, 'groups.discard', exact)
    assert refused.get('error', {}).get('message') == 'peer_stop_required', refused
    assert current(c) == before and c.peer.stops == []
    def status(**kwargs):
        return {'active': not c.peer.stops, 'status': 'cancelled' if c.peer.stops else 'running',
                'task_id': exact['task_id'], 'execution_generation': exact['execution_generation']}
    monkeypatch.setattr(c.peer, 'status', status)
    stopped = rpc(c, 'groups.stop')
    assert 'result' in stopped, stopped
    assert current(c)['status'] == 'cancelled'
    assert c.peer.stops == [(exact['task_id'], exact['execution_generation'])]
    assert rpc(c, 'groups.stop')['result']['cancelled'] == 0


@pytest.mark.parametrize('outcome', ['still-stopping', 'unreachable', 'withdrawn'])
def test_discard_never_stops_or_erases_an_accepted_unknown_turn(case, outcome):
    c = case
    unknown_with_receipt(c)
    if outcome == 'still-stopping':
        c.peer.stop_status = 'stopping'
    elif outcome == 'unreachable':
        c.peer.stop_status = None
    else:
        c.authority.runner._draining = True
    before = current(c)
    assert 'error' in rpc(c, 'groups.discard', selector(c))
    assert current(c) == before and c.peer.stops == []
    assert not any(e['kind'] == 'turn.cancelled' for e in c.service._events('room'))


@pytest.mark.parametrize('stage', ['before_control', 'before_dispatch'])
@pytest.mark.parametrize('restart', [False, True])
def test_verified_renewal_preserves_deferred_retry_evidence_across_control_and_restart(case, monkeypatch, stage, restart):
    from gateway.session_group_peer_routes import CanonicalPeerClient
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    c = case
    tick(c, 4)
    original = current(c)
    proof = original['result']
    exact = selector(c)
    assert tasks.is_proven_nonadmission(original)
    c.peer.mode = 'repaired'
    if stage == 'before_dispatch':
        assert rpc(c, 'groups.retry', exact)['result']['retried']
    claims = decode_room_grant(c.peer.secret, c.route.grant, permission='status')
    now = time.time()
    renewed = issue_room_grant(c.peer.secret, grant_id='routine-renewal', **c.peer.scope,
        target_install_id=c.route.target_install_id, execution_policy_digest=c.route.execution_policy_digest,
        permissions=claims['permissions'], issued_at=now, ttl_seconds=claims['status_expires_at'] - now,
        status_expires_at=claims['status_expires_at'])
    binding = HostedRoomBinding('room', c.route.home_install_id, 1)
    tracked = CanonicalPeerClient(c.service, binding, ('room', 'peer'), c.route, c.peer)
    from gateway import hosted_room_peer
    def refresh(**kwargs):
        kwargs['on_issued'](renewed)
        return {'grant': renewed, 'catalog': c.peer.catalog.as_mapping()}
    monkeypatch.setattr(c.peer, 'refresh_grant', refresh, raising=False)
    monkeypatch.setattr(hosted_room_peer, 'room_grant_needs_dispatch_refresh',
                        lambda token, **kwargs: token == c.route.grant and kwargs.get('leeway_seconds') != 0)
    tracked.probe(grant=c.route.grant)  # the same automatic renewal path as the supervisor
    if stage == 'before_control':
        assert current(c)['result'] == proof  # the original evidence was not rewritten
    if restart:
        old = c.service
        c.now[0] += old.runtime.lease_ttl_seconds + 1
        old.runtime._thread = None
        c.service = CanonicalHostedRoomService(c.authority, None)
        c.authority.hosted_room_service = c.service
        c.service.runtime.clock = lambda: c.now[0]
        c.service.runtime._thread = SimpleNamespace(is_alive=lambda: True)
        c.service.peer_clients[('room', 'peer')] = c.peer
        c.service.member_rpcs = old.member_rpcs
        tick(c)  # the restarted supervisor reacquires the room lease before offering controls
    if stage == 'before_control':
        assert any(a['kind'] == 'retry' for a in actions(c))
        assert rpc(c, 'groups.retry', exact)['result']['retried']
    c.now[0] += c.service.runtime.unavailable_retry_max_seconds + 1
    tick(c, 3)
    assert current(c)['status'] == 'settled'
    assert len(healthy(c)) == 1
    assert c.peer.dispatches[-1]['task_id'] == exact['task_id']
    assert c.peer.dispatches[-1]['execution_generation'] == exact['execution_generation'] + 1
    c.service.runtime._thread = None


def test_manual_grant_replacement_requires_new_send_but_can_discard_proven_unaccepted_work(case):
    c = case
    tick(c, 4)
    exact = selector(c)
    replacement = issue_room_grant(c.peer.secret, grant_id='manual-reinvite', **c.peer.scope,
        target_install_id=c.route.target_install_id, execution_policy_digest=c.route.execution_policy_digest,
        ttl_seconds=3600)
    c.service.register_peer_route(room_id='room', member_id='peer', route=replace(c.route, grant=replacement),
                                  client=c.peer, target_url=c.peer.base_url, catalog=c.peer.catalog)
    assert 'error' in rpc(c, 'groups.retry', exact)
    assert any(a['kind'] == 'discard' for a in actions(c))
    assert rpc(c, 'groups.discard', exact)['result']['task']['status'] == 'cancelled'
    assert c.peer.stops == []
