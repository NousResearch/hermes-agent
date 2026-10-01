"""Shared fixtures for passive Group Chat copies: a home store, a participant store, real grants.

``pair`` keeps both SQLite stores real and routes the publisher's HTTP requests straight into the
participant's production ingress, so only urllib I/O is replaced. ``api`` serves the participant's
real aiohttp routes for tests that need the HTTP layer too.
"""

import copy
import io
import json
import sqlite3
import time
import urllib.error
from types import SimpleNamespace

import pytest
from aiohttp import web

from gateway import hosted_room_driver as driver
from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_work_records as records
from gateway import hosted_rooms as rooms
from gateway.hosted_room_replica_ingress import ingest_granted_page

SECRET = b"passive-copy-test-secret-not-a-credential"
HOME = "install:home"
TARGET = "install:participant"
KEY = ("room", "reviewer")
API_KEY = "disposable-api-key-for-loopback-only"


def catalog(target=TARGET, profile="default"):
    return peer.GatewayRoomCatalog.from_mapping(peer.catalog_mapping(
        installation_id=target, target_profile=profile, persistent_process=True,
        execution_policy=peer.execution_policy_mapping(target_profile=profile, config={})))


def member(member_id, *, target=TARGET, profile="default"):
    return {"member_id": member_id, "handle": member_id, "profile": profile, "target": {
        "kind": "peer", "peer_id": f"peer-{member_id}", "installation_id": target,
        "profile": profile, "capability_digest": catalog(target, profile).catalog_digest}}


MEMBERS = [{"member_id": "writer", "handle": "writer", "profile": "default",
            "target": {"kind": "local", "profile": "default"}}, member("reviewer")]


def append(db, event_id, text=None, *, epoch=1, gateway=HOME):
    return rooms.append_event(
        db, room_id="room", event_id=event_id, kind="message.user", actor={"kind": "user", "id": "owner"},
        payload={"text": text or event_id}, authority_gateway_id=gateway, authority_epoch=epoch)


def grant(*, secret=SECRET, member_id="reviewer", target=TARGET, profile="default", permissions=("replicate",),
          now=None, **overrides):
    """A signed grant as the participant would mint it."""
    now = time.time() if now is None else now
    fields = dict(
        grant_id=f"grant-{member_id}", room_id="room", home_install_id=HOME, authority_gateway_id=HOME,
        authority_epoch=1, member_id=member_id, target_install_id=target, target_profile=profile,
        permissions=permissions, execution_policy_digest=catalog(target, profile).execution_policy.policy_digest,
        issued_at=now - 1, ttl_seconds=600, status_ttl_seconds=3600)
    fields.update(overrides)
    return peer.issue_room_grant(secret, **fields)


def reserve(target_db, token, *, now=None):
    """Record the participant-side reservation that the grant's issuer made when minting it."""
    claims = json.loads(peer._split_token(token)[0])
    rooms.reserve_peer_room(target_db, claims=claims, expires_at=claims["status_expires_at"], now=now)
    return claims


def save_link(db, *, member_id="reviewer", target=TARGET, profile="default", url="http://127.0.0.1:9876",
              **grant_overrides):
    """Register a member route on the home, as ``groups.peer.register`` stores it."""
    token = grant(member_id=member_id, target=target, profile=profile, **grant_overrides)
    link = links.make_stored_link(
        room_id="room", member_id=member_id, target_url=url, target_profile=profile, grant=token,
        catalog=catalog(target, profile), cancellation_scope_id="cancel", trace_id="trace")
    links.save_room_link(db, link)
    return link


def http_error(request, code, error_code="unavailable"):
    return urllib.error.HTTPError(request.full_url, code, "controlled failure", {},
                                  io.BytesIO(json.dumps({"error": {"code": error_code}}).encode()))


def _route_error(request, exc):
    """The status the participant's real routes answer for an ingress refusal."""
    from gateway import hosted_room_replicas as replicas
    if isinstance(exc, peer.HostedRoomGrantError):
        return http_error(request, 401, "invalid_room_grant")
    if isinstance(exc, (replicas.ReplicaCapacityError, records.WorkRecordCapacityError)):
        return http_error(request, 507, "storage_full")
    if isinstance(exc, records.WorkRecordPrefixError):
        return http_error(request, 409, "work_records_prefix")
    if isinstance(exc, replicas.ReplicaGapError):
        return http_error(request, 409, "room_replica_gap")
    if isinstance(exc, (records.WorkRecordError, rooms.HostedRoomError)):
        return http_error(request, 409, "invalid_room_replica")
    return None


class HTTP:
    """Replace urllib only: serialization, scoped auth and the participant's ingress stay real.

    History pages land in ``requests`` and task-evidence records in ``records``; ``before``,
    ``error``, ``lose_ack`` and ``reply_transform`` act on history, their ``record_*`` siblings on
    records. ``source``, when set, is checked to have no writer held across a request.
    """

    def __init__(self, target, source=None):
        self.target, self.source = target, source
        self.requests, self.records = [], []
        self.before = self.error = self.reply_transform = None
        self.record_before = self.record_error = self.record_reply_transform = None
        self.lose_ack = self.lose_record_ack = False

    def __call__(self, request, *, timeout, **kwargs):
        from gateway import hosted_room_proof as proof
        import urllib.parse
        assert 0 < timeout <= 3
        assert request.method == "POST"
        authorization = request.get_header('Authorization')
        state = None
        if authorization.startswith(proof.SCHEME):
            envelope = json.loads(proof._b64decode(authorization[len(proof.SCHEME):]))
            claims = json.loads(proof._b64decode(envelope['payload']))
            token, key, mac, _, plaintext = proof.verify_request(authorization, secret=SECRET,
                installation_id=claims['target_install_id'], method=request.method,
                path=urllib.parse.urlsplit(request.full_url).path, body=request.data,
                headers=dict(request.header_items()))
            state = key, mac
            # Fault hooks model the participant after transport verification/decryption.
            request = urllib.request.Request(request.full_url, method=request.method, data=plaintext,
                headers={**{k: v for k, v in request.header_items()
                            if k.lower() not in {'authorization', 'content-length'}},
                         'Authorization': 'HermesRoom ' + token})
        else:
            token = authorization.removeprefix('HermesRoom ')
        try:
            response = self._receive(request, token)
        except urllib.error.HTTPError as exc:
            if state is None:
                raise
            raw = exc.read()
            wire, nonce, signed = proof.seal_response(*state, exc.code, raw)
            headers = {proof.RESPONSE_HEADER: signed, proof.RESPONSE_NONCE_HEADER: nonce}
            raise urllib.error.HTTPError(exc.url, exc.code, exc.msg, headers, io.BytesIO(wire)) from exc
        headers = {}
        if state is not None:
            wire, nonce, signed = proof.seal_response(*state, 200, response.getvalue())
            response = io.BytesIO(wire)
            headers = {proof.RESPONSE_HEADER: signed, proof.RESPONSE_NONCE_HEADER: nonce}
        response.status, response.headers = 200, headers
        return response

    def _receive(self, request, token):
        if self.source is not None:
            # No home transaction may span HTTP.
            with sqlite3.connect(self.source, timeout=0.1) as conn:
                conn.execute("BEGIN IMMEDIATE")
        body = json.loads(request.data)
        if request.full_url.endswith("/v1/room-members/work-records"):
            self.records.append(copy.deepcopy(body["record"]))
            if self.record_before:
                self.record_before(request, body)
            if self.record_error:
                raise self.record_error(request) if callable(self.record_error) else self.record_error
            try:
                claims = peer.decode_room_grant(SECRET, token, permission=records.PERMISSION)
                result = records.ingest(self.target, record=body["record"], token=token, secret=SECRET,
                                        target_install_id=claims["target_install_id"],
                                        target_profile=claims["target_profile"])
            except Exception as exc:
                raise _route_error(request, exc) or exc
            if self.lose_record_ack:
                self.lose_record_ack = False
                raise TimeoutError("record reply lost after the participant committed")
            reply = {"object": "hermes.room_member.work_records", **result}
            if self.record_reply_transform:
                reply = self.record_reply_transform(reply)
            return io.BytesIO(json.dumps(reply).encode())
        assert request.full_url.endswith("/v1/room-members/replica")
        self.requests.append((request.full_url, copy.deepcopy(body)))
        if self.before:
            self.before(request, body)
        if self.error:
            code, error_code = self.error
            raise http_error(request, code, error_code)
        try:
            claims = peer.decode_room_grant(SECRET, token, permission="replicate")
            result = ingest_granted_page(
                self.target, token=token, secret=SECRET, target_install_id=claims["target_install_id"],
                target_profile=claims["target_profile"], **body)
        except Exception as exc:
            raise _route_error(request, exc) or exc
        if self.lose_ack:
            self.lose_ack = False
            raise TimeoutError("reply lost after the participant committed")
        if self.reply_transform:
            result = self.reply_transform(result)
        return io.BytesIO(json.dumps(result).encode())


@pytest.fixture
def pair(tmp_path, monkeypatch):
    """A home store with a registered ``replicate`` route, and the participant's store."""
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
    source, target = tmp_path / "home.db", tmp_path / "participant.db"
    rooms.create_room(source, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    append(source, "hello")
    link = save_link(source)
    reserve(target, link.grant)
    http = HTTP(target)
    monkeypatch.setattr("tui_gateway.hosted_room_peer_http._open_roomlink_url", http)
    return SimpleNamespace(source=source, target=target, link=link, http=http)


def add_route(pair, *, member_id="z-other", target=TARGET, **grant_overrides):
    """A second member route on the same Group Chat (another Bot on that participant, by default)."""
    members = rooms.room_state(pair.source, room_id="room")["members"] + [member(member_id, target=target)]
    with sqlite3.connect(pair.source) as conn:
        conn.execute("UPDATE hosted_rooms SET members_json=? WHERE room_id='room'", (json.dumps(members),))
    other = save_link(pair.source, member_id=member_id, target=target, **grant_overrides)
    reserve(pair.target, other.grant)
    return other


TASK = driver.TaskIdentity("room", "task", "thread", "turn")
EVIDENCE = ("replicate", records.PERMISSION)


def admit(source, task=TASK, *, member_id="reviewer", seq=1):
    return driver.admit_task(source, task, payload={
        "target_profile": "default", "target_member_id": member_id,
        "prompt": "PRIVATE_PROMPT /private/workspace", "source_event_seq": seq}, clock=lambda: 100)


def start(source, task=TASK):
    held = driver.acquire_lease(source, room_id="room", gateway_id=HOME, authority_epoch=1,
                                process_generation="process", ttl_seconds=30, clock=lambda: 100)
    return driver.start_task(source, task, held, expected_cancel_generation=0, clock=lambda: 100)


def add_task(source, suffix):
    """One more message and the task it starts, as a busy Group Chat produces them."""
    append(source, f"message-{suffix}")
    admit(source, driver.TaskIdentity("room", f"task-{suffix}", "thread", f"turn-{suffix}"),
          seq=rooms.room_state(source, room_id="room")["latest_seq"])


@pytest.fixture
def copying(pair, monkeypatch):
    """``pair`` whose route also carries task evidence, with a publisher for it."""
    from gateway.hosted_room_replication import HostedRoomReplicationPublisher
    pair.link = save_link(pair.source, permissions=EVIDENCE)
    reserve(pair.target, pair.link.grant)
    pair.http.source = pair.source
    pair.records = pair.http.records
    pair.pub = HostedRoomReplicationPublisher(pair.source)
    return pair


def both_routes_carry_evidence(pair):
    """Two member routes to one participant, both opted in to task evidence."""
    other = add_route(pair, permissions=EVIDENCE)
    pair.link = save_link(pair.source, permissions=EVIDENCE)
    reserve(pair.target, pair.link.grant)
    return other


@pytest.fixture
def api(tmp_path, monkeypatch):
    """The participant's real room-member HTTP routes over its own store, and a home store."""
    from gateway.config import PlatformConfig
    from gateway.platforms import api_server_room_grants as grants
    from gateway.platforms.api_server import APIServerAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    source, target = tmp_path / "home.db", tmp_path / "participant.db"
    monkeypatch.setattr(rooms, "default_db_path", lambda: target)
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: TARGET)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))
    app = web.Application()
    for method, path, handler in grants._http_routes(adapter):
        app.router.add_route(method, path, handler)
    rooms.create_room(source, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    append(source, "hello", "Café 日本語")
    return SimpleNamespace(source=source, target=target, app=app, adapter=adapter)


async def invite(http, **flags):
    """Mint a participant grant over the API-key invitation route."""
    body = dict(room_id="room", home_install_id=HOME, authority_gateway_id=HOME, authority_epoch=1,
                member_id="reviewer")
    body.update(flags)
    response = await http.post("/v1/room-members/invitations", json=body,
                               headers={"Authorization": f"Bearer {API_KEY}"})
    assert response.status == 201, await response.text()
    return await response.json()


# -- copy retirement ---------------------------------------------------------------------------

HOME_SECRET = b"home-only-retirement-test-secret-32-bytes"


def prepare(home_db, *, target=TARGET, endpoint="https://participant.example", **kwargs):
    """The room owner's ``groups.replication.prepare`` on the home."""
    from gateway import hosted_room_replica_retirement as retirement
    return retirement.prepare_home_enrollment(
        home_db, room_id="room", target_install_id=target, endpoint=endpoint, local_gateway_id=HOME,
        secret=kwargs.pop("secret", HOME_SECRET), **kwargs)


def enroll(target_db, enrollment, **kwargs):
    """The participant operator's ``groups.replication.enroll``."""
    from gateway import hosted_room_replica_retirement as retirement
    return retirement.enroll_target(target_db, enrollment=enrollment, target_install_id=TARGET, **kwargs)


def disband(home_db):
    return rooms.disband_room(home_db, room_id="room", expected_gateway_id=HOME, expected_epoch=1)


def notice(home_db, enrollment, loader=lambda: HOME_SECRET):
    from gateway import hosted_room_replica_retirement as retirement
    return retirement.materialize_notice(home_db, enrollment_id=enrollment["enrollment_id"], local_gateway_id=HOME,
                                         secret_loader=loader)


def retire(target_db, outgoing):
    from gateway import hosted_room_replica_retirement as retirement
    return retirement.retire_copy(target_db, payload=outgoing.payload(), value=outgoing.value,
                                  local_gateway_id=TARGET)
