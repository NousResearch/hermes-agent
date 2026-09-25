"""Canonical pending action -> authorized RPC -> manual driver ticks.

No coordinator, listener, model or native worker. The registered connection and
canonical service are real; only peer I/O and the healthy local RPC are inert.
"""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
import time

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms as rooms
from gateway.hosted_room_peer import (GatewayRoomCatalog, catalog_mapping,
    issue_room_grant, decode_room_grant)
from gateway.session_authority import SessionAuthority
from gateway.session_authorities import SessionAuthorities
from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tui_gateway.hosted_room_driver import HostedRoomRuntime
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tests.gateway.peer_retry_fixtures import LocalRPC, Peer


class Target(Peer):
    base_url = "http://127.0.0.1:8765"
    secret = b"inert-canonical-retry-secret-32bytes"

    def __init__(self, catalog, scope):
        super().__init__("unavailable")
        self.catalog, self.scope = catalog, scope
        self.revoked = False
        self.on_probe = None

    def probe(self, *, grant):
        decode_room_grant(self.secret, grant, permission="dispatch")
        if self.revoked:
            raise PeerRunsHTTPError("revoked", status_code=403, error_code="invalid_room_grant")
        if self.on_probe:
            self.on_probe()
        return {"catalog": self.catalog.as_mapping(), **self.scope}

    def revoke_grant_exact(self, *, grant):
        decode_room_grant(self.secret, grant, permission="status")
        return {"revoked": True}

    def dispatch(self, **kwargs):
        if self.mode == "refused-dispatch":
            self.dispatches.append(kwargs)
            raise PeerRunsHTTPError("not admitted", retryable=True, not_admitted=True)
        return super().dispatch(**kwargs)


@pytest.fixture
def case(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    def forbidden(*args, **kwargs):
        raise AssertionError("runtime/listener/network startup forbidden")
    monkeypatch.setattr(HostedRoomRuntime, "start", forbidden)
    monkeypatch.setattr("tui_gateway.hosted_room_peer_http._open_roomlink_url", forbidden)
    with SessionDB(home / "state.db") as db:
        runner = SimpleNamespace(_draining=False, session_authorities=SessionAuthorities(home))
        authority = SessionAuthority(runner, db=db, profile_id=str(home), instance_id="fixture",
            epoch=begin_runtime_epoch(db, instance_id="fixture"))
        runner.session_authorities.add(home, authority)
        service = CanonicalHostedRoomService(authority, None)
        authority.hosted_room_service = service
        connection = AuthorityConnection(authority, SimpleNamespace(), {"user_id": "alice",
            "profile_id": str(home), "instance_id": "fixture"})
        service.authorize_room(connection.actor.subject, "room", create=True)
        service.local_profiles = lambda: ("default",)
        catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(installation_id="peer-install",
            persistent_process=True, target_profile="reviewer", attachments=True))
        gateway = rooms.local_authority_gateway_id()
        scope = dict(room_id="room", home_install_id=gateway, authority_gateway_id=gateway,
                     authority_epoch=1, member_id="peer", target_profile="reviewer")
        peer = Target(catalog, scope)
        grant = issue_room_grant(peer.secret, grant_id="grant", **scope,
            target_install_id=catalog.installation_id,
            execution_policy_digest=catalog.execution_policy.policy_digest,
            permissions=("dispatch", "status", "stop", "attachment.stage"), ttl_seconds=3600)
        route = PeerMemberRoute(home_install_id=gateway, member_id="peer",
            target_install_id=catalog.installation_id, target_profile="reviewer",
            capability_digest=catalog.catalog_digest,
            execution_policy_digest=catalog.execution_policy.policy_digest,
            cancellation_scope_id="cancel", trace_id="trace", grant=grant, attachments=True)
        service.create_room(room_id="room", name="Retry", members=[
            dict(member_id="peer", profile="reviewer", handle="peer", target=dict(kind="peer",
                peer_id="peer-install", installation_id="peer-install", profile="reviewer",
                capability_digest=catalog.catalog_digest)),
            dict(member_id="healthy", profile="default", handle="healthy")])
        service.register_peer_route(room_id="room", member_id="peer", route=route, client=peer,
                                    target_url=peer.base_url, catalog=catalog)
        local = LocalRPC()
        # Keep canonical peer resolution and its controls, replacing only the
        # healthy member's external execution boundary.
        service.member_rpcs[("room", "healthy", "default", connection.actor.subject, str(home))] = local
        now = [time.time()]
        service.runtime.clock = lambda: now[0]
        # The RPC requires a live coordinator. Report that inert boundary without
        # starting any thread; manual _run_cycle still executes the real driver.
        service.runtime._thread = SimpleNamespace(is_alive=lambda: True)
        item = service.put_attachment(room_id="room", upload_id="file", kind="file",
            name="notes.txt", mime="text/plain", data=b"frozen input")
        manifest = [{k: item[k] for k in ("attachment_id", "name", "kind", "size", "mime")}]
        service.send(room_id="room", event_id="request", payload=dict(thread_id="thread",
            text="Review this file together", attachments=manifest))
        original = tasks.list_tasks(db.db_path, room_id="room")[0]
        c = SimpleNamespace(service=service, connection=connection, peer=peer, route=route,
            now=now, original=original, local=local, manifest=manifest, authority=authority)
        yield c
        service.runtime._thread = None
        assert not service.runtime.status()["running"]
        assert not service.runtime._room_threads


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
    response = rpc(c, "groups.state")  # canonical status lives in groups.state, not groups.status
    assert "error" not in response, response
    return response["result"]["driver_status"]["pending_actions"]


def healthy(c):
    return [e for e in c.service._events("room")
            if e["kind"] == "message.member" and e["payload"]["member_id"] == "healthy"]


@pytest.mark.parametrize("failure", ["unavailable", "refused-dispatch"])
def test_canonical_retry_recovers_frozen_peer_once_without_replaying_healthy(case, failure):
    c = case
    c.peer.mode = failure
    tick(c, 4)
    saved = current(c)
    assert saved["status"] == "deferred", saved.get("result")
    assert len(healthy(c)) == 1
    events = c.service._events("room")
    assert any(e["kind"] == "turn.deferred" and e["payload"]["member_id"] == "peer" for e in events)
    assert not any(e["kind"] == "turn.failed" and e["payload"]["member_id"] == "peer" for e in events)
    exact = selector(c)
    assert dict(kind="retry", **{k: v for k, v in exact.items() if k != "room_id"}) in actions(c)
    before = healthy(c)
    prior_dispatches = len(c.peer.dispatches)
    c.peer.mode = "repaired"
    reply = rpc(c, "groups.retry", exact)
    assert reply.get("result", {}).get("retried") is True, reply
    assert current(c)["status"] == "queued"
    assert current(c)["execution_generation"] == saved["execution_generation"]
    assert "error" in rpc(c, "groups.retry", exact)  # no silent second requeue
    tick(c)
    assert len(c.peer.staged) == 1
    c.now[0] += c.service.runtime.unavailable_retry_max_seconds + 1
    tick(c, 3)
    done = current(c)
    assert done["status"] == "settled", done.get("result")
    assert done["execution_generation"] == saved["execution_generation"] + 1
    assert done["payload"] == saved["payload"] == c.original["payload"]
    assert len(c.peer.dispatches) == prior_dispatches + 1
    assert c.peer.dispatches[-1]["dispatch"]["prompt"] == saved["payload"]["prompt"]
    assert healthy(c) == before
    assert "error" in rpc(c, "groups.retry", exact)


@pytest.mark.parametrize("state", ["running", "indeterminate", "deferred"])
def test_unknown_peer_never_gets_new_retry_rights(case, state):
    c = case
    c.peer.mode = "ambiguous-dispatch"
    tick(c)
    assert current(c)["status"] == "running"
    assert len(c.peer.dispatches) == 1
    if state != "running":
        c.now[0] += c.service.runtime.lease_ttl_seconds + 1
        binding = c.service.bindings()[0]
        lease = tasks.acquire_lease(c.service.db_path, room_id="room", gateway_id=binding.gateway_id,
            authority_epoch=binding.authority_epoch, process_generation="unknown-successor",
            ttl_seconds=600, clock=lambda: c.now[0])
        tasks.recover_room(c.service.db_path, lease, clock=lambda: c.now[0])
        assert current(c)["status"] == "indeterminate"
        if state == "deferred":
            tasks.defer_indeterminate_task(c.service.db_path, c.original["identity"], lease,
                expected_execution_generation=1, expected_cancel_generation=0,
                reason="member_unavailable", clock=lambda: c.now[0])
            assert current(c)["result"] == {"reason": "member_unavailable", "retryable": True}
    before = current(c)
    assert not any(a["kind"] == "retry" for a in actions(c))
    assert "error" in rpc(c, "groups.retry", selector(c))
    forged = rpc(c, "groups.retry", {**selector(c), "nonadmission": True})
    assert forged["error"]["data"]["reason"] == "invalid_params"
    assert "error" in rpc(c, "groups.discard", selector(c))  # blanket peer discard stays refused
    assert current(c) == before
    assert len(c.peer.dispatches) == 1


@pytest.mark.parametrize("fence", ["generation", "member", "foreign", "capability", "profile",
    "route", "expired", "reauthorization", "revoked", "quarantine", "work_closed", "owner", "epoch",
    "cancel_race", "lease_race", "route_race", "owner_race", "epoch_race", "work_closed_race",
    "reauthorization_race", "quarantine_race", "room_epoch", "room_epoch_race", "grant",
    "wrong_target", "probe_scope"])
def test_peer_retry_rechecks_exact_current_authority(case, monkeypatch, fence):
    from gateway import hosted_room_links as links
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
            permissions=("dispatch", "status", "stop", "attachment.stage"), ttl_seconds=3600)
        c.service.register_peer_route(room_id="room", member_id="peer", route=replace(c.route, grant=grant),
            client=c.peer, target_url=c.peer.base_url, catalog=c.peer.catalog)
    def change_room_epoch():
        rooms.claim_authority(c.service.db_path, room_id="room", expected_gateway_id=c.route.home_install_id,
            expected_epoch=1, new_gateway_id=c.route.home_install_id, event_id="reclaim")
    def replace_owner():
        c.authority.db._execute_write(lambda conn: conn.execute(
            "UPDATE state_meta SET value='bob' WHERE key='gateway.hosted.owner.v1:room'"))
    def replace_epoch():
        begin_runtime_epoch(c.authority.db, instance_id="new-owner")
    def close_work():
        c.service.begin_room_disband("room")
    modifications = {
        "generation": lambda: exact.update(execution_generation=2),
        "member": lambda: exact.update(member_id="healthy"),
        "foreign": lambda: setattr(c.connection, "actor", replace(c.connection.actor, subject="bob")),
        "capability": lambda: setattr(c.connection, "actor", replace(c.connection.actor,
            capabilities=frozenset({"session:read"}))),
        "profile": lambda: setattr(c.connection, "actor", replace(c.connection.actor, profile_id="/foreign")),
        "route": replace_route,
        "grant": replace_grant,
        "room_epoch": change_room_epoch,
        "room_epoch_race": lambda: setattr(c.peer, "on_probe", change_room_epoch),
        "wrong_target": lambda: c.service.peer_routes.update({("room", "peer"):
            replace(c.route, target_install_id="foreign-install")}),
        "probe_scope": lambda: c.peer.scope.update(member_id="foreign-member"),
        "expired": lambda: monkeypatch.setattr(time, "time", lambda: c.now[0] + 7200),
        "reauthorization": lambda: links.mark_room_link_status(c.service.db_path, room_id="room",
            member_id="peer", status="needs_reauthorization"),
        "revoked": lambda: setattr(c.peer, "revoked", True),
        "quarantine": lambda: c.authority.db._execute_write(lambda conn: conn.execute(
            "INSERT INTO hosted_room_quarantine VALUES (?, ?, ?)", ("room", "lost-authority", c.now[0]))),
        "work_closed": close_work,
        "owner": replace_owner,
        "epoch": replace_epoch,
        "cancel_race": lambda: setattr(c.peer, "on_probe", lambda: tasks.cancel_task(c.service.db_path,
            c.original["identity"], cancel_id="race", expected_cancel_generation=0, clock=lambda: c.now[0])),
        "lease_race": lambda: setattr(c.peer, "on_probe", lambda: c.now.__setitem__(0,
            c.now[0] + c.service.runtime.lease_ttl_seconds + 1)),
        "route_race": lambda: setattr(c.peer, "on_probe", replace_route),
        "owner_race": lambda: setattr(c.peer, "on_probe", replace_owner),
        "epoch_race": lambda: setattr(c.peer, "on_probe", replace_epoch),
        "work_closed_race": lambda: setattr(c.peer, "on_probe", close_work),
        "reauthorization_race": lambda: setattr(c.peer, "on_probe", lambda: links.mark_room_link_status(
            c.service.db_path, room_id="room", member_id="peer", status="needs_reauthorization")),
        "quarantine_race": lambda: setattr(c.peer, "on_probe", lambda: c.authority.db._execute_write(
            lambda conn: conn.execute("INSERT INTO hosted_room_quarantine VALUES (?, ?, ?)",
                                     ("room", "lost-authority", c.now[0])))),
    }
    modifications[fence]()
    before = current(c)
    replies = healthy(c)
    result = rpc(c, "groups.retry", exact)
    assert "error" in result, result
    after = current(c)
    if fence == "cancel_race":
        assert after["status"] == "cancelled"
        assert after["cancel_generation"] == before["cancel_generation"] + 1
    else:
        assert after == before
    assert after["execution_generation"] == before["execution_generation"] == 1
    assert c.peer.dispatches == []
    assert healthy(c) == replies
    if fence in {"lease_race", "reauthorization_race"}:
        route = links.load_room_link(c.service.db_path, room_id="room", member_id="peer")
        assert route.status == ("needs_reauthorization" if fence == "reauthorization_race" else "unavailable")
    if fence in {"route", "expired", "reauthorization", "work_closed", "capability", "revoked"}:
        assert not any(a["kind"] == "retry" for a in actions(c))


def test_throwing_publication_rescan_retains_proof_and_recovery(case):
    c = case
    publish = c.service.runtime.publish_terminal
    def fail_once(binding, task):
        c.service.runtime.publish_terminal = publish
        raise RuntimeError("publication unavailable after durable deferral")
    c.service.runtime.publish_terminal = fail_once
    tick(c)
    before = current(c)
    assert before["status"] == "deferred"
    assert tasks.is_proven_nonadmission(before)
    assert not any(e["kind"] == "turn.deferred" for e in c.service._events("room"))
    tick(c, 4)  # real prepare_room rescan, not a replay of the throwing callback
    assert current(c) == before
    assert len([e for e in c.service._events("room") if e["kind"] == "turn.deferred"]) == 1
    assert len(healthy(c)) == 1
    c.peer.mode = "repaired"
    assert rpc(c, "groups.retry", selector(c)).get("result", {}).get("retried") is True
    c.now[0] += c.service.runtime.unavailable_retry_max_seconds + 1
    tick(c, 3)
    assert current(c)["status"] == "settled", current(c).get("result")
    assert len(healthy(c)) == 1 and len(c.peer.dispatches) == 1


@pytest.mark.parametrize("corruption", ["legacy", "boolean", "cancel", "run", "binding"])
def test_legacy_or_malformed_deferred_proof_stays_held(case, corruption):
    import json
    c = case
    tick(c, 4)
    result = current(c)["result"]
    # Corrupt an actually produced durable row; never mint a positive proof flag
    # as test setup. Legacy shape is indistinguishable from unknown provenance.
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
    assert "error" in rpc(c, "groups.retry", selector(c))
    assert current(c) == before and c.peer.dispatches == []


def test_cold_canonical_service_keeps_proof_but_never_auto_retries(case):
    c = case
    tick(c, 4)
    before = current(c)
    replies = healthy(c)
    old = c.service
    old.runtime._thread = None
    c.now[0] += old.runtime.lease_ttl_seconds + old.runtime.unavailable_retry_max_seconds + 1
    cold = CanonicalHostedRoomService(c.authority, None)
    c.authority.hosted_room_service = cold
    cold.local_profiles = old.local_profiles
    cold.member_rpcs = old.member_rpcs
    cold.peer_clients[("room", "peer")] = c.peer  # only the inert I/O boundary
    cold.runtime.clock = lambda: c.now[0]
    cold.runtime._thread = SimpleNamespace(is_alive=lambda: True)
    c.service = cold
    try:
        tick(c, 3)
        assert current(c) == before
        assert c.peer.dispatches == [] and len(c.peer.staged) == 1
        assert healthy(c) == replies
        assert any(a["kind"] == "retry" for a in actions(c))
        c.peer.mode = "repaired"
        exact = selector(c)
        assert rpc(c, "groups.retry", exact).get("result", {}).get("retried") is True
        tick(c, 3)
        assert current(c)["status"] == "settled", current(c).get("result")
        assert current(c)["execution_generation"] == before["execution_generation"] + 1
        assert current(c)["payload"] == before["payload"]
        assert healthy(c) == replies and len(c.peer.dispatches) == 1
    finally:
        cold.runtime._thread = None
        assert not cold.runtime._room_threads
