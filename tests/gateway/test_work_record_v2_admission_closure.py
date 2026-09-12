"""V2 HTTP admission adversaries and deterministic grant/writer-lock races."""

import asyncio
import copy
import threading
import time
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_room_replica_ingress as ingress
from gateway import hosted_room_work_records as work
from gateway import hosted_rooms as rooms
from tests.gateway.test_api_server_room_replicas import HOME, TARGET, KEY, MEMBERS, setup  # noqa: F401
from tests.gateway.test_api_server_room_work_records import seed, invitation, deliver
from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR, transfer
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


async def enrolled_work(http, source, *, prefix_only=False):
    seed(source)
    receipt = dict(room_id="room", home_install_id=HOME, authority_gateway_id=HOME, authority_epoch=1,
        member_id="reviewer", target_install_id=TARGET, target_profile="default", task_id="task",
        execution_generation=7, run_id="original-run", session_id="original-session")
    rooms.upsert_remote_run_receipt(source, record=receipt)
    transfer(source)
    issued = await invitation(http, home_install_id=SUCCESSOR, authority_gateway_id=SUCCESSOR,
        authority_epoch=2, passive_only=True)
    entry = retirement.prepare_home_enrollment(source, room_id="room", target_install_id=TARGET,
        endpoint=str(http.make_url("/")), local_gateway_id=SUCCESSOR, secret=peer.gateway_room_grant_secret())
    response = await http.post("/v1/group-replicas/enroll", json={"enrollment": entry,
        **retirement.home_enrollment_history(source, enrollment_id=entry["enrollment_id"])},
        headers={"Authorization": f"Bearer {KEY}"})
    assert response.status == 200, await response.text()
    client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=10)
    page = rooms.read_events(source, room_id="room", replica_version=2, limit=1 if prefix_only else 100)
    await asyncio.to_thread(client.replicate_page, grant=issued["grant"], target_profile="default",
        room_id="room", room_name="Workshop", members=MEMBERS, page=page)
    return client, issued["grant"], work.capture(source, room_id="room", local_gateway_id=SUCCESSOR)


MUTATIONS = [
    ("version", True), ("version", 2.0), ("authority.epoch", 2.0), ("home_install_id", HOME),
    ("incompleteness", None), ("incompleteness", []), ("incompleteness", "prior_authority_work_unknown"),
    ("lineage_sha256", "0" * 64), ("extra", "not permitted"), ("tasks", {}),
    ("receipts.0.home_install_id", SUCCESSOR), ("receipts.0.authority_epoch", 2),
    ("receipts.0.room_id", "other"), ("receipts.0.member_id", "other"),
    ("receipts.0.target_install_id", "other"), ("receipts.0.target_profile", "other"),
    ("tasks.0.source_event_seq", 0), ("tasks.0.source_event_seq", 3),
    ("tasks.0.source_event_seq", True), ("tasks.0.profile", "other"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("path,value", MUTATIONS, ids=[p + '-' + str(v) for p, v in MUTATIONS])
async def test_v2_envelope_provenance_negatives_cannot_replace_target(setup, path, value):
    source, target, app = setup
    async with TestClient(TestServer(app)) as http:
        client, grant, record = await enrolled_work(http, source)
        await deliver(client, grant, record)
        bad = copy.deepcopy(record)
        node = bad
        keys = path.split('.')
        for key in keys[:-1]:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if value is None:
            node.pop(keys[-1])
        else:
            node[keys[-1]] = value
        bad["revision"] += 1
        bad["digest"] = work.digest({k: v for k, v in bad.items() if k not in {"revision", "digest"}})
        with pytest.raises(PeerRunsHTTPError) as error:
            await deliver(client, grant, bad)
        assert error.value.status_code in {401, 409}
        summary = replicas.replica_state(target, room_id="room")["work_records"]
        assert summary["digest"] == record["digest"]
        assert summary["receipts"] == record["receipts"]
        assert summary["task_origins"]["task"] == {"gateway_id": HOME, "epoch": 1}
        assert summary["source_loss_safe"] is False
        assert not rooms.list_rooms(target)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix_only", [False, True], ids=["anchor-before-claim", "claim-not-committed"])
async def test_v2_anchor_requires_actual_claim_in_committed_prefix(setup, prefix_only):
    source, target, app = setup
    async with TestClient(TestServer(app)) as http:
        client, grant, record = await enrolled_work(http, source, prefix_only=prefix_only)
        if not prefix_only:
            with rooms._transaction(source) as conn:
                record["history"] = {"seq": 1, "event_sha256": work.history_anchor(conn, "hosted_room_events", "room", 1)}
            record["digest"] = work.digest({k: v for k, v in record.items() if k not in {"revision", "digest"}})
        with pytest.raises(PeerRunsHTTPError):
            await deliver(client, grant, record)
        assert replicas.replica_state(target, room_id="room")["work_records"]["availability"] == "not_retained"


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["revocation", "expiry"])
async def test_v2_authorization_is_checked_inside_audited_writer_after_lock_wait(setup, monkeypatch, race):
    source, target, app = setup
    async with TestClient(TestServer(app)) as http:
        client, grant, record = await enrolled_work(http, source)
        claims = peer.decode_room_grant(peer.gateway_room_grant_secret(), grant, permission=work.PERMISSION)
        clock = [time.time()]
        monkeypatch.setattr(peer, "time", SimpleNamespace(time=lambda: clock[0]))
        monkeypatch.setattr(ingress, "time", SimpleNamespace(time=lambda: clock[0]))
        locked, waiting, release = threading.Event(), threading.Event(), threading.Event()
        owner = threading.local()
        attempting_sql, audited = threading.Event(), []
        def traced_connect(path):
            conn = rooms._connect(path)
            def trace(statement):
                if statement == "BEGIN IMMEDIATE":
                    assert locked.is_set() and not release.is_set()
                    attempting_sql.set()
            conn.set_trace_callback(trace)
            return conn
        original_replica_writer = replicas._transaction
        monkeypatch.setattr(replicas, "_transaction", partial(rooms.transaction, traced_connect, immediate=False))
        original_transaction, replica_transaction = rooms._transaction, replicas._replica_transaction
        authorizations = []
        authorize = ingress.authorize_granted_room

        def observe_authorize(**kwargs):
            # Called by actual work consumer, after replica audit acquired writer.
            assert audited == [True]
            authorizations.append(kwargs["authority"])
            callback = authorize(**kwargs)
            def checked(conn):
                assert conn.in_transaction
                return callback(conn)
            return checked
        monkeypatch.setattr(ingress, "authorize_granted_room", observe_authorize)

        @contextmanager
        def held_transaction(*args, **kwargs):
            with original_transaction(*args, **kwargs) as conn:
                yield conn
                if getattr(owner, "hold", False):
                    locked.set()
                    assert release.wait(8)
        monkeypatch.setattr(rooms, "_transaction", held_transaction)

        @contextmanager
        def waiting_transaction(*args, **kwargs):
            waiting.set()
            assert locked.wait(8)
            with replica_transaction(*args, **kwargs) as conn:
                audited.append(conn.in_transaction)
                yield conn
        monkeypatch.setattr(replicas, "_replica_transaction", waiting_transaction)

        def writer():
            owner.hold = True
            if race == "revocation":
                rooms.revoke_room_grant_scope(target, claims=claims, expires_at=claims["status_expires_at"])
            else:
                with held_transaction(target, immediate=True):
                    pass
        request = asyncio.create_task(deliver(client, grant, record))
        assert await asyncio.to_thread(waiting.wait, 8)  # HTTP auth already passed.
        holder = asyncio.create_task(asyncio.to_thread(writer))
        try:
            assert await asyncio.to_thread(locked.wait, 8)
            assert await asyncio.to_thread(attempting_sql.wait, 8)
            assert authorizations == []
            if race == "expiry":
                clock[0] = claims["status_expires_at"] + 1
        finally:
            release.set()
        await holder
        with pytest.raises(PeerRunsHTTPError) as error:
            await request
        assert error.value.status_code == 401
        assert authorizations == [record["authority"]]
        monkeypatch.setattr(rooms, "_transaction", original_transaction)
        monkeypatch.setattr(replicas, "_replica_transaction", replica_transaction)
        monkeypatch.setattr(replicas, "_transaction", original_replica_writer)
        assert replicas.replica_state(target, room_id="room")["work_records"]["availability"] == "not_retained"
