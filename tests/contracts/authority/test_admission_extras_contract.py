"""Contract cases for admit's order of checks and its typed extras."""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal, FenceStale
from persistence.authority.admissions import ActiveReplace, ApiTargetClaim, CarrierRewind, GateRefusal, QueuedReplacement, RetryRewriteRefused
from session_rules.automation import FOLDED_COMPLETION_TEXT
from persistence.authority.vocabulary import ADMISSION_QUEUE_MAX
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession

SESSION = "sess-extra"
GATE = GateRefusal("runtime_draining", field="owner", detail="drain started")


def _seed(store, *session_ids: str) -> int:
    for sid in session_ids or (SESSION,):
        store.sessions.create(NewSession(session_id=sid, source="contract"))
    return store.runtime.begin_epoch(instance_id="instance-extra")


def _admit(store, epoch: int, request_id: str, *, session_id: str = SESSION, principal_id: str = "p", **extras):
    return store.admissions.admit(epoch=epoch, principal_id=principal_id, session_id=session_id, request_id=request_id, payload={"text": request_id}, **extras)


def test_a_gate_refusal_raises_and_writes_nothing(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    # CONTRACT: no gate admits.
    first, created = _admit(store, epoch, "r0", gate=None)
    assert created and first.status == "queued"
    session = store.sessions.get(SESSION)
    # CONTRACT: a gate refusal raises with its reason, field and detail and writes nothing: no row, none of the typed extras after it, no revision bump.
    with pytest.raises(AuthorityRefusal) as refused:
        _admit(store, epoch, "r1", gate=GATE, replaces=QueuedReplacement(first.admission_id))
    assert (refused.value.reason, refused.value.field, refused.value.detail) == ("runtime_draining", "owner", "drain started")
    assert store.admissions.list(session_id=SESSION, pending_only=False) == [first]
    assert store.admissions.find(principal_id="p", session_id=SESSION, request_id="r1") is None
    assert store.sessions.get(SESSION) == session


def test_gate_runs_after_the_replay_and_the_queue_cap(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    first, _ = _admit(store, epoch, "r0")
    # CONTRACT: an exact live replay bypasses the gate.
    assert _admit(store, epoch, "r0", gate=GATE) == (first, False)
    with pytest.raises(AuthorityRefusal) as refused:
        _admit(store, epoch, "r1", gate=GATE)
    assert (refused.value.reason, refused.value.field, refused.value.detail) == ("runtime_draining", "owner", "drain started")
    for n in range(1, ADMISSION_QUEUE_MAX):
        _admit(store, epoch, f"r{n}")
    # CONTRACT: the queue cap refuses before the gate.
    with pytest.raises(AuthorityRefusal, match="^queue_full$"):
        _admit(store, epoch, "overflow", gate=GATE)
    assert len(store.admissions.list(session_id=SESSION)) == ADMISSION_QUEUE_MAX


def test_replaces_cancels_the_queued_row_and_binds_the_replay(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    old, _ = _admit(store, epoch, "tick-1")
    new, created = _admit(store, epoch, "tick-2", replaces=QueuedReplacement(old.admission_id))
    assert created and store.admissions.get(admission_id=old.admission_id).outcome == "cancelled"
    # CONTRACT: the exact replay answers only the request that retired that row.
    assert _admit(store, epoch, "tick-2", replaces=QueuedReplacement(old.admission_id)) == (new, False)
    with pytest.raises(AuthorityRefusal, match="^admission_conflict$"):
        _admit(store, epoch, "tick-2", replaces=QueuedReplacement("someone-else"))
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    with pytest.raises(AuthorityRefusal, match="^admission_started$"):
        _admit(store, epoch, "tick-3", replaces=QueuedReplacement(claimed.admission_id))
    assert [r.request_id for r in store.admissions.list(session_id=SESSION)] == ["tick-2"]


def test_supersedes_requires_the_queued_row_on_this_target(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store, SESSION, "sess-other")
    elsewhere, _ = _admit(store, epoch, "tick", session_id="sess-other")
    with pytest.raises(FenceStale):
        _admit(store, epoch, "tick-new", supersedes=elsewhere.admission_id)
    tick, _ = _admit(store, epoch, "tick")
    fresh, created = _admit(store, epoch, "tick-new", supersedes=tick.admission_id)
    assert created and store.admissions.get(admission_id=tick.admission_id).outcome == "cancelled"
    assert [r.admission_id for r in store.admissions.list(session_id=SESSION)] == [fresh.admission_id]


def test_a_fold_absorbs_the_queued_tail_completion_and_keeps_only_its_identity(backend, scope_a):
    from persistence.authority.admissions import QueuedFold

    store = backend.open(scope_a)
    epoch = _seed(store)

    def admit(request_id, **extras):
        payload = {"text": request_id, "local_automation_v1": {"owner": SESSION, "identities": [["completion", request_id]]}}
        return store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id=request_id, payload=payload, **extras)
    done, _ = admit("done-1")
    tick, _ = admit("tick-1")
    # CONTRACT: a queued row behind the folded one that is not in skip_ids refuses stale_generation and writes nothing.
    with pytest.raises(FenceStale):
        admit("batch-1", fold=QueuedFold(done.admission_id, frozenset()))
    assert store.admissions.get(admission_id=done.admission_id).status == "queued"
    # CONTRACT: with every later queued row in skip_ids the folded row is cancelled and keeps only its own identity: its text becomes the folded marker and its descriptor drops its identities.
    batch, created = admit("batch-1", fold=QueuedFold(done.admission_id, frozenset({tick.admission_id})))
    folded = store.admissions.get(admission_id=done.admission_id)
    assert created and (folded.status, folded.outcome) == ("terminal", "cancelled")
    assert folded.payload == {"text": FOLDED_COMPLETION_TEXT, "local_automation_v1": {"owner": SESSION}}
    # CONTRACT: an exact replay runs no fold; a fold of a row no longer queued refuses stale_generation.
    assert admit("batch-1", fold=QueuedFold(done.admission_id, frozenset({tick.admission_id}))) == (batch, False)
    with pytest.raises(FenceStale):
        admit("batch-2", fold=QueuedFold(done.admission_id, frozenset({tick.admission_id, batch.admission_id})))


def test_api_target_spends_one_request_id_for_one_target(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store, SESSION, "sess-other")
    claim = ApiTargetClaim(settings={"model": "m"}, client_key=True)
    _admit(store, epoch, "idem-1", principal_id="api", api_target=claim)
    with pytest.raises(AuthorityRefusal, match="^admission_conflict$"):
        _admit(store, epoch, "idem-1", principal_id="api", session_id="sess-other", api_target=claim)
    assert store.admissions.list(session_id="sess-other") == []
    record, created = _admit(store, epoch, "idem-2", principal_id="api", session_id="sess-other", api_target=claim)
    assert created and record.principal_id == "api"


def test_active_replace_rewrites_the_transcript_inside_the_admission(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    store.messages.append(SESSION, NewMessage(role="user", content="first"))
    store.messages.append(SESSION, NewMessage(role="assistant", content="answer"))
    pinned = tuple(store.messages.active_message_ids(SESSION))
    rewrite = ActiveReplace(SESSION, messages=({"role": "user", "content": "first"},), expected_active_ids=pinned)
    _record, created = _admit(store, epoch, "retry", rewrite=rewrite)
    assert created and [r.content for r in store.messages.read_transcript(SESSION)] == ["first"]
    # CONTRACT: a transcript that moved since the plan refuses both the rewrite and the admission.
    store.messages.append(SESSION, NewMessage(role="assistant", content="later"))
    stale = ActiveReplace(SESSION, messages=(), expected_active_ids=pinned)
    with pytest.raises(RetryRewriteRefused):
        _admit(store, epoch, "retry-2", rewrite=stale)
    assert [r.request_id for r in store.admissions.list(session_id=SESSION)] == ["retry"]
    assert [r.content for r in store.messages.read_transcript(SESSION)] == ["first", "later"]
    # CONTRACT: the gate refuses before the rewrite runs.
    with pytest.raises(AuthorityRefusal, match="^runtime_draining$"):
        _admit(store, epoch, "retry-3", gate=GATE, rewrite=ActiveReplace(SESSION, messages=(), expected_active_ids=tuple(store.messages.active_message_ids(SESSION))))
    assert len(store.messages.read_transcript(SESSION)) == 2


def test_carrier_rewind_on_a_moved_transcript_is_refused(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    target = store.messages.append(SESSION, NewMessage(role="user", content="ask"))
    with pytest.raises(RetryRewriteRefused):
        _admit(store, epoch, "retry", rewrite=CarrierRewind(SESSION, target_row_id=int(target), expected_active_ids=(int(target) + 1,), expected_target_content="ask"))
    assert store.admissions.list(session_id=SESSION) == []
    assert [r.content for r in store.messages.read_transcript(SESSION)] == ["ask"]
