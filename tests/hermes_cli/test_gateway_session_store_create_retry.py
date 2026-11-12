"""GatewaySessionStore.create under a busy gateway (#146359): a lost ``session.create`` answer is asked
again under the same ``request_id``, and a failed create holds that id for the next one, so the
session the owner may already have built is adopted rather than orphaned (scripted gateway)."""
from contextlib import asynccontextmanager

import pytest

from hermes_cli.gateway_client import AnswerLost, GatewayClientError, GatewayRPCError
from hermes_cli.gateway_connection import DISCONNECTED, NOT_SENT, GatewayConnection
from hermes_cli.gateway_session_store import GatewaySessionStore
from tests.hermes_cli.test_gateway_session_store import FakeClient

_DIAL_FAILED = GatewayClientError("Gateway connection failed; no local fallback")
_ID_MISMATCH = GatewayRPCError("invalid_params", data={"field": "request_id"})


@pytest.fixture
def store():
    client = FakeClient()

    @asynccontextmanager
    async def connect():
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    yield GatewaySessionStore(conn), client
    conn.close()


def _failing_creates(client, *errors):
    """Each ``session.create`` raises the next of ``errors`` (what the transport raised), then answers."""
    original, pending = client.rpc, list(errors)

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.create" and pending:
            client.calls.append((method, params))
            raise pending.pop(0)
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc


def _creates(client):
    return [params for method, params in client.calls if method == "session.create"]


def _create_ids(client):
    return [params["request_id"] for params in _creates(client)]


@pytest.mark.parametrize("lost", [TimeoutError(), AnswerLost(DISCONNECTED)], ids=["deadline", "hang_up"])
def test_a_lost_create_answer_is_asked_again_under_the_same_request_id(store, lost):
    """The owner keys a local session by its ``request_id`` and answers a repeat with that session. A
    fresh id on the retry builds a second agent and orphans the first session."""
    db, client = store
    _failing_creates(client, lost)
    assert db.create({}) == "local-new"
    first, second = _creates(client)
    assert first == second  # same request_id and the same frozen request, or the owner refuses it


@pytest.mark.parametrize("refusal", [GatewayRPCError("invalid_params"), GatewayClientError(NOT_SENT), _DIAL_FAILED],
                         ids=["owner_refused", "never_sent", "dial_failed"])
def test_a_refused_or_unsent_create_is_not_asked_again(store, refusal):
    """A refusal is the owner's answer, and a request that never left has nothing to recover."""
    db, client = store
    _failing_creates(client, refusal)
    with pytest.raises(GatewayClientError) as raised:
        db.create({})
    assert raised.value is refusal and len(_creates(client)) == 1


def test_the_next_create_after_two_lost_answers_reuses_their_request_id(store):
    """Both asks lost: the owner may still have created the session. The user's next prompt creates
    again (``_init_agent``) and must land on it rather than orphan it; a different request does not."""
    db, client = store
    _failing_creates(client, TimeoutError(), TimeoutError())
    with pytest.raises(AnswerLost):
        db.create({})
    assert db.create({}) == "local-new"
    db.create({"model": "other"})
    db.create({})  # answered above: a new create of the same request is a new session
    ids = _create_ids(client)
    assert ids[0] == ids[1] == ids[2] != ids[3] and ids[4] not in ids[:4]


@pytest.mark.parametrize("failures", [
    (TimeoutError(), GatewayClientError(NOT_SENT)), (TimeoutError(), _DIAL_FAILED),
    (TimeoutError(), GatewayRPCError("runtime_draining")), (TimeoutError(), KeyboardInterrupt()),
    (GatewayRPCError("storage_unavailable"),), (GatewayRPCError("request_failed"),),
    (GatewayRPCError("invalid_params"),), (KeyboardInterrupt(),),
], ids=["retry_never_sent", "retry_dial_failed", "retry_refused_draining", "retry_interrupted",
        "storage_after_commit", "internal_error", "fresh_refusal", "first_ask_interrupted"])
def test_a_failed_create_holds_its_request_id_for_the_next(store, failures):
    """Any failure may follow the owner's commit (a lost first ask before a retry that failed, a
    failure after the commit, Ctrl+C), and the user sees what actually failed. Holding an id the
    owner never used costs nothing: the next create simply asks under it."""
    db, client = store
    _failing_creates(client, *failures)
    with pytest.raises(type(failures[-1])) as raised:
        db.create({})
    assert raised.value is failures[-1]
    assert db.create({}) == "local-new"
    assert len(set(_create_ids(client))) == 1


@pytest.mark.parametrize("later", [GatewayRPCError("runtime_draining"), GatewayClientError(NOT_SENT),
                                   GatewayRPCError("invalid_params", data={"field": "cwd"})],
                         ids=["refused_draining", "never_sent", "refused_before_the_lookup"])
def test_only_an_answer_settles_a_held_request_id(store, later):
    """Held after two lost answers: a later ask under it that is refused for another reason (a
    draining owner, a cwd gone for now) or never sent says nothing about the first session, so the id
    stays held until an ask under it is answered."""
    db, client = store
    _failing_creates(client, TimeoutError(), TimeoutError(), later)
    with pytest.raises(AnswerLost):
        db.create({})
    with pytest.raises(GatewayClientError):
        db.create({})
    db.create({})  # answered: the session the lost asks may have created
    db.create({})  # a new create after that is a new session
    ids = _create_ids(client)
    assert ids[0] == ids[1] == ids[2] == ids[3] != ids[4]


@pytest.mark.parametrize("verdict", [_ID_MISMATCH, GatewayRPCError("admission_conflict"),
                                     GatewayRPCError("not_found")],
                         ids=["another_request", "another_key", "deleted"])
def test_the_owners_verdict_on_a_held_request_id_releases_it(store, verdict):
    """The owner refused the id itself (another frozen request, another launch key, its session
    deleted): every ask under it would be refused the same way, so the next create mints a new one."""
    db, client = store
    _failing_creates(client, TimeoutError(), TimeoutError(), verdict)
    with pytest.raises(AnswerLost):
        db.create({})
    with pytest.raises(GatewayRPCError):
        db.create({})
    db.create({})
    ids = _create_ids(client)
    assert ids[0] == ids[1] == ids[2] != ids[3]


def test_a_verdict_on_the_retry_after_a_lost_ask_holds_nothing(store):
    db, client = store
    _failing_creates(client, TimeoutError(), _ID_MISMATCH)
    with pytest.raises(GatewayRPCError):
        db.create({})
    db.create({})
    first, second, third = _create_ids(client)
    assert first == second != third


def test_a_verdict_on_another_request_keeps_the_held_one(store):
    db, client = store
    _failing_creates(client, TimeoutError(), TimeoutError(), _ID_MISMATCH)
    with pytest.raises(AnswerLost):
        db.create({})
    with pytest.raises(GatewayRPCError):
        db.create({"model": "other"})
    db.create({})
    ids = _create_ids(client)
    assert ids[0] == ids[1] == ids[3] != ids[2]


def test_resuming_a_session_settles_a_held_create(store):
    """Lost creates, then the user resumes a session: a later /new is a new session, never the one the
    lost asks made (which may be the very session resumed)."""
    db, client = store
    _failing_creates(client, TimeoutError(), TimeoutError())
    with pytest.raises(AnswerLost):
        db.create({})
    db.attach("local-old")
    db.create({})
    ids = _create_ids(client)
    assert ids[0] == ids[1] != ids[2]


def test_create_if_missing_asks_a_lost_answer_again_under_the_same_request_id(store, monkeypatch):
    db, client = store
    from hermes_cli import gateway_session_store as gss
    monkeypatch.setattr(gss, "session_store", lambda: db)
    _failing_creates(client, TimeoutError())
    assert gss.create_titled_session("Fresh Named") == "local-new"
    first, second = _creates(client)
    assert first == second and first["title"] == "Fresh Named"
