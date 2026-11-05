"""Contract cases for the admission reads and the API payload compaction."""

from __future__ import annotations

import base64
import hashlib

import pytest

from persistence import AuthorityRefusal, FenceStale, StaleFenceKind
from persistence.authority.runtime import Delete
from persistence.conversation.sessions import NewSession

SESSION = "sess-read"


def _seed(store, *session_ids: str, source: str = "contract") -> int:
    for sid in session_ids or (SESSION,):
        store.sessions.create(NewSession(session_id=sid, source=source))
    return store.runtime.begin_epoch(instance_id="instance-read")


def _admit(store, epoch, request_id, *, session_id=SESSION, principal_id="p", payload=None):
    return store.admissions.admit(epoch=epoch, principal_id=principal_id, session_id=session_id, request_id=request_id,
                                  payload=payload if payload is not None else {"text": request_id})[0]


def _settle_next(store, epoch, session_id=SESSION, outcome="completed"):
    claimed = store.admissions.claim(epoch=epoch, session_id=session_id)
    return store.admissions.settle(epoch=epoch, admission_id=claimed.admission_id, generation=claimed.generation, outcome=outcome)


def _delete(store, epoch, session_id):
    row = store.sessions.get(session_id)
    store.runtime.mutate(epoch, principal_id="p", session_id=session_id, request_id="delete-" + session_id, expected_revision=row.runtime_revision,
                         expected_generation=row.runtime_generation, command=Delete(), owner=None)


def test_retired_replay_answers_only_the_exact_retry_on_a_retired_target(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    assert store.admissions.retired_replay(epoch=epoch, principal_id="p", session_id=SESSION, request_id="r1", payload={"text": "r1"}) is None
    record = _admit(store, epoch, "r1")
    _settle_next(store, epoch)
    _delete(store, epoch, SESSION)
    replay = store.admissions.retired_replay(epoch=epoch, principal_id="p", session_id=SESSION, request_id="r1", payload={"text": "r1"})
    assert (replay.admission_id, replay.status, replay.outcome) == (record.admission_id, "terminal", "completed")
    with pytest.raises(AuthorityRefusal, match="^admission_conflict$"):
        store.admissions.retired_replay(epoch=epoch, principal_id="p", session_id=SESSION, request_id="r1", payload={"text": "other"})
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store.admissions.retired_replay(epoch=epoch, principal_id="p", session_id=SESSION, request_id="fresh", payload={"text": "r1"})
    with pytest.raises(FenceStale) as stale:
        store.admissions.retired_replay(epoch=epoch + 1, principal_id="p", session_id=SESSION, request_id="r1", payload={"text": "r1"})
    assert stale.value.kind is StaleFenceKind.STALE_EPOCH
    with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
        store.admissions.retired_replay(epoch=epoch, principal_id="", session_id=SESSION, request_id="r1", payload={"text": "r1"})
    # CONTRACT: admit runs the same read first, so the exact retry replays with created False.
    assert store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id="r1", payload={"text": "r1"}) == (replay, False)


def test_pending_targets_and_has_pending_follow_status_and_principal(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store, "a", "b", "c")
    _admit(store, epoch, "r1", session_id="a", principal_id="api")
    _admit(store, epoch, "r2", session_id="b")
    _admit(store, epoch, "r3", session_id="c")
    _settle_next(store, epoch, "c")
    assert store.admissions.pending_targets(statuses=frozenset({"queued"})) == frozenset({"a", "b"})
    assert store.admissions.pending_targets(statuses=frozenset({"queued"}), principal_id="api") == frozenset({"a"})
    assert store.admissions.pending_targets(statuses=frozenset({"terminal"})) == frozenset({"c"})
    assert store.admissions.has_pending(statuses=frozenset({"queued", "started"}))
    assert store.admissions.has_pending(session_ids=("c", "absent"))
    assert not store.admissions.has_pending(session_ids=("c",), statuses=frozenset({"queued", "started"}))
    assert not store.admissions.has_pending(session_ids=("absent",))


def test_ledger_busy_counts_pending_admissions_and_live_workers(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    assert not store.admissions.ledger_busy(SESSION, SESSION)
    _admit(store, epoch, "r1")
    assert store.admissions.ledger_busy(SESSION, "elsewhere")
    _settle_next(store, epoch)
    assert not store.admissions.ledger_busy(SESSION, "elsewhere")


def test_webhook_reads_select_signed_ingress_rows(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store, source="webhook")
    ingress = {"native_text_v1": {"source": {"platform": "webhook", "chat_id": "hook-1"}, "text": "event"}}
    automation = {"native_text_v1": {"source": {"platform": "webhook", "chat_id": "hook-1"}, "automation": {"kind": "watch"}}}
    first = _admit(store, epoch, "delivery-1", payload=ingress)
    _admit(store, epoch, "watch-1", payload=automation)
    assert store.admissions.webhook_ingress_payloads(chat_id="hook-1") == [ingress]
    assert store.admissions.webhook_ingress_payloads(chat_id="hook-2") == []
    assert store.admissions.unfinalized_webhook_admissions() == []
    _settle_next(store, epoch)
    assert store.admissions.unfinalized_webhook_admissions() == [first.admission_id]
    store.sessions.end(SESSION, "webhook_complete")
    assert store.admissions.unfinalized_webhook_admissions() == []


def test_terminal_ids_reads_live_rows_and_tombstones(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store, "live", "gone")
    queued = _admit(store, epoch, "q", session_id="live")
    retired = _admit(store, epoch, "g", session_id="gone")
    _settle_next(store, epoch, "gone")
    _delete(store, epoch, "gone")
    settled = _admit(store, epoch, "s", session_id="live")
    store.admissions.cancel(epoch=epoch, admission_id=settled.admission_id)
    assert store.admissions.terminal_ids([queued.admission_id, retired.admission_id, settled.admission_id, "absent"]) == frozenset({retired.admission_id, settled.admission_id})


def test_compact_api_payloads_rewrites_the_settled_candidates(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    pixels = b"committed image bytes"
    digest = hashlib.sha256(pixels).hexdigest()
    look = {"type": "text", "text": "look"}
    image = {"type": "image_url", "image_url": {"url": "DATA:image/png;base64," + base64.b64encode(pixels).decode()}}
    other = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    inline = {"text": [look, image, other], "api_turn_v1": {"media": [{"sha256": digest}], "history": None}}
    history = {"text": "t", "api_turn_v1": {"history": [{"role": "user", "content": "old"}]}}
    responses = {"text": "t", "api_turn_v1": {"history": [{"role": "user", "content": "old"}]}}
    ids = [_admit(store, epoch, rid, principal_id="api", payload=p).admission_id for rid, p in (("chat-1", inline), ("chat-2", history), ("responses:3", responses))]
    queued = _admit(store, epoch, "chat-4", principal_id="api", payload=history)
    for _ in ids:
        _settle_next(store, epoch)
    assert store.admissions.api_compaction_candidates() == ids[:2]
    # CONTRACT: each given row still a candidate is rewritten in the one write; a responses: history, a queued row and an unknown id stay as they are.
    assert store.admissions.compact_api_payloads(admission_ids=[*ids, queued.admission_id, "absent"]) == 2
    retained = {"type": "image_url", "image_url": {"url": "hermes-retained:" + digest}}
    assert store.admissions.get(admission_id=ids[0]).payload == {**inline, "text": [look, retained, other]}
    assert store.admissions.get(admission_id=ids[1]).payload == {**history, "api_turn_v1": {"history": None}}
    assert store.admissions.get(admission_id=ids[2]).payload == responses
    assert store.admissions.get(admission_id=queued.admission_id).payload == history
    # CONTRACT: an inline image this admission never committed keeps the row a candidate; a second pass writes nothing.
    assert store.admissions.api_compaction_candidates() == ids[:1]
    assert store.admissions.compact_api_payloads(admission_ids=ids) == 0
    # CONTRACT: the digest is untouched, so the exact retry still replays.
    assert store.admissions.admit(epoch=epoch, principal_id="api", session_id=SESSION, request_id="chat-2", payload=history)[1] is False
