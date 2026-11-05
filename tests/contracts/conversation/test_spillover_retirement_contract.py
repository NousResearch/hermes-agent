"""Contract cases for the spillover retirement candidates the deletion fence records: every delete path (``SessionRepository.delete`` and ``delete_many``, ``SessionMaintenance.prune_sessions``, ``SessionRuntimeRepository.mutate(Delete())``) records, inside its own transaction, one candidate per spillover archive the doomed sessions' rows name.

Seeding uses public protocol operations only; the candidates are read back through ``AdmissionRepository.retired_candidates("spillover")``. The verbs leave the collection to the caller, so nothing here unlinks or drops a candidate.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from persistence import SessionId
from persistence.authority.runtime import Delete
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, SessionLineageKind
from persistence.housekeeping.maintenance import PruneCriteria
from tests.contracts._scopes import MEDIA_ROOT

PREFIX = "gateway.spillover.retired.v1."

# One row per pointer form: a host path, structured content, a Windows path in the API content and in structured content (JSON-escaped), a sandbox pointer and the bare word (neither names an archive), and two names in one row, the second with a trailing full stop the name rule keeps.
POINTERS = (
    NewMessage(role="user", content="Saved to /home/u/.hermes/cache/spillover/out-1.txt"),
    NewMessage(role="tool", content=[{"type": "text", "text": "see ~/.hermes/cache/spillover/out-2.txt"}], tool_call_id="c1", tool_name="terminal"),
    NewMessage(role="assistant", content="short", api_content="full text in C:\\Users\\u\\.hermes\\cache\\spillover\\out-3.txt"),
    NewMessage(role="tool", content=[{"type": "text", "text": "C:\\Users\\u\\cache\\spillover\\out-4.txt"}], tool_call_id="c2", tool_name="terminal"),
    NewMessage(role="user", content="/tmp/hermes-results/sandbox.txt and the word spillover alone"),
    NewMessage(role="assistant", content="spillover/a.txt then spillover/b.log."),
)


def _session(store, sid: str, *messages: NewMessage, **fields) -> None:
    store.sessions.create(NewSession(session_id=SessionId(sid), source="cli", **fields))
    for message in messages:
        store.messages.append(SessionId(sid), message)


def _candidates(store) -> list[tuple[str, object]]:
    return [(key[len(PREFIX):], value) for key, value in store.admissions.retired_candidates("spillover")]


def test_a_delete_records_every_archive_its_rows_and_its_delegates_rows_name(backend, scope_a):
    store = backend.open(scope_a)
    _session(store, "sp-a", *POINTERS)
    _session(store, "sp-a-child", NewMessage(role="user", content="spillover/child.txt"),
             parent_session_id=SessionId("sp-a"), lineage_kind=SessionLineageKind.DELEGATE)
    _session(store, "sp-keep", NewMessage(role="user", content="spillover/keep.txt"))
    assert _candidates(store) == []
    assert set(store.sessions.delete(SessionId("sp-a"), media_root=MEDIA_ROOT)) == {"sp-a", "sp-a-child"}
    # CONTRACT: one candidate per named archive, its value the name, from the root and the delegate child the delete takes; a surviving session's archive and the non-archive pointers record nothing.
    assert _candidates(store) == [("a.txt", "a.txt"), ("b.log.", "b.log."), ("child.txt", "child.txt"), ("out-1.txt", "out-1.txt"),
                                  ("out-2.txt", "out-2.txt"), ("out-3.txt", "out-3.txt"), ("out-4.txt", "out-4.txt")]
    _session(store, "sp-again", NewMessage(role="user", content="spillover/out-1.txt again"))
    store.sessions.delete(SessionId("sp-again"), media_root=MEDIA_ROOT)
    # CONTRACT: a later delete naming a recorded archive keeps the one candidate.
    assert len(_candidates(store)) == 7


def test_a_bulk_delete_records_the_archives_of_the_sessions_it_removes(backend, scope_a):
    store = backend.open(scope_a)
    _session(store, "sp-m1", NewMessage(role="user", content="spillover/many-1.txt"))
    _session(store, "sp-m2", NewMessage(role="assistant", content="short", api_content="spillover/many-2.txt"))
    _session(store, "sp-m3", NewMessage(role="user", content="spillover/many-3.txt"))
    assert store.sessions.delete_many([SessionId("sp-m1"), SessionId("sp-m2")], media_root=MEDIA_ROOT).deleted == 2
    # CONTRACT: the removed sessions' archives, the content and API-content pointers alike; the kept session's stays unrecorded.
    assert _candidates(store) == [("many-1.txt", "many-1.txt"), ("many-2.txt", "many-2.txt")]


def test_a_prune_records_the_archives_of_the_ended_sessions_it_removes(backend, scope_a):
    store = backend.open(scope_a)
    _session(store, "sp-old", NewMessage(role="user", content="spillover/old.txt"))
    _session(store, "sp-open", NewMessage(role="user", content="spillover/open.txt"))
    store.sessions.end(SessionId("sp-old"), "done")
    criteria = PruneCriteria(last_active_before=(datetime.now(UTC) + timedelta(minutes=5)).timestamp())
    assert store.maintenance.prune_sessions(criteria=criteria, media_root=MEDIA_ROOT).deleted_ids == (SessionId("sp-old"),)
    # CONTRACT: only the pruned session's archive; the open session is no prune candidate.
    assert _candidates(store) == [("old.txt", "old.txt")]


def test_a_runtime_delete_records_the_archives_of_its_targets(backend, scope_a):
    store = backend.open(scope_a)
    _session(store, "sp-x", NewMessage(role="user", content="spillover/mut.txt"))
    epoch = store.runtime.begin_epoch(instance_id="instance-spillover")
    row = store.sessions.get(SessionId("sp-x"))
    outcome = store.runtime.mutate(epoch, principal_id="p", session_id="sp-x", request_id="req-del", expected_revision=row.runtime_revision,
                                   expected_generation=row.runtime_generation, command=Delete(), media_root=MEDIA_ROOT, owner=None)
    assert outcome.gone is not None and outcome.gone.session_ids == ("sp-x",)
    # CONTRACT: the runtime delete records the same candidates inside its write.
    assert _candidates(store) == [("mut.txt", "mut.txt")]
