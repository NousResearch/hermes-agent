"""Contract cases for a rename through a compressed conversation: the title lands on the live tip of the compression chain, a committed write advances every segment's revision, a clear clears the whole lineage under the title rules, and a rotating compress moves the name to the new segment."""

from __future__ import annotations

import pytest

from persistence import CapabilityUnsupported, FenceStale, StaleFenceKind
from persistence.authority import runtime as rt
from persistence.authority.bindings import local_session_id
from persistence.conversation.sessions import CANONICAL_BOT_CHAT_TITLE, END_REASON_COMPRESSION, NewSession, SessionQuery, TitleSource

PRINCIPAL = "principal-a"


def _compressed(store, root_title=None):
    store.sessions.create(NewSession(session_id="chat", source="desktop"))
    if root_title:
        store.sessions.set_title("chat", root_title)
    store.sessions.end("chat", END_REASON_COMPRESSION)
    store.sessions.create(NewSession(session_id="chat-2", source="desktop", parent_session_id="chat"))
    store.sessions.set_auto_title("chat-2", "Plugin para sessoes", source=TitleSource.LLM)
    return store.runtime.begin_epoch(instance_id="owner")


def _rename(store, epoch, sid, title, request_id, *, expected_revision=None):
    if expected_revision is None:
        expected_revision = store.sessions.get(sid).runtime_revision
    return store.runtime.mutate(epoch, principal_id=PRINCIPAL, session_id=sid, request_id=request_id, expected_revision=expected_revision,
                                command=rt.Rename(title), owner=None).result


def _listed_title(store):
    [entry] = [entry for entry in store.sessions.list_sessions(SessionQuery()) if "chat" in (entry.lineage_ids or (entry.session.session_id,))]
    return entry.session.title


def test_rename_through_an_ended_segment_lands_on_the_tip(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _compressed(store)
    receipt = _rename(store, epoch, "chat", "SESSION TALK", "r1")
    # CONTRACT: the title lands on the live tip, and the receipt reads it there.
    assert receipt["title"] == store.sessions.get("chat-2").title == "SESSION TALK"
    assert store.sessions.get("chat").title is None
    assert _listed_title(store) == "SESSION TALK"
    # CONTRACT: the receipt's revision is the addressed row's, so the next fence on that id holds.
    assert receipt["revision"] == store.sessions.get("chat").runtime_revision == 1
    assert _rename(store, epoch, "chat", "Second name", "r2")["title"] == "Second name"
    assert store.sessions.get("chat-2").title == "Second name"


def test_a_name_held_by_an_ended_ancestor_moves_to_the_tip(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _compressed(store, root_title="Old name")
    _rename(store, epoch, "chat", "Old name", "r1")
    assert (store.sessions.get("chat-2").title, store.sessions.get("chat").title) == ("Old name", None)
    # CONTRACT: a rename sent to the tip itself writes the tip.
    assert _rename(store, epoch, "chat-2", "Tip name", "r2")["title"] == "Tip name"
    assert store.sessions.get("chat-2").title == "Tip name"


def test_a_tip_rename_makes_an_older_copy_of_any_segment_stale(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _compressed(store)
    stale = store.sessions.get("chat").runtime_revision
    _rename(store, epoch, "chat-2", "B name", "b1")
    # CONTRACT: a committed rename advances every segment's revision, so a rename through an older copy of the ended segment refuses.
    with pytest.raises(FenceStale) as refused:
        _rename(store, epoch, "chat", "A name", "a1", expected_revision=stale)
    assert refused.value.kind is StaleFenceKind.REVISION_CONFLICT
    assert store.sessions.get("chat-2").title == "B name"


def test_clearing_through_an_ended_segment_clears_the_listed_name(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _compressed(store, root_title="Old name")
    store.sessions.set_title("chat-2", "")
    # CONTRACT: an untitled tip lists under an ancestor's title, so a clear clears every row of the lineage.
    assert _rename(store, epoch, "chat", "", "c1")["title"] is None
    assert (store.sessions.get("chat").title, store.sessions.get("chat-2").title) == (None, None)
    assert _listed_title(store) is None


def test_a_name_freed_from_another_chat_advances_that_chats_revision(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id="ghost", source="desktop"))
    store.sessions.set_title("ghost", "Shared name")
    store.sessions.end("ghost", "user_exit")
    epoch = _compressed(store)
    before = store.sessions.get("ghost").runtime_revision
    _rename(store, epoch, "chat", "Shared name", "g1")
    # CONTRACT: the row the uniqueness rule freed the name from is affected: its revision advances.
    assert (store.sessions.get("ghost").title, store.sessions.get("chat-2").title) == (None, "Shared name")
    assert store.sessions.get("ghost").runtime_revision > before


def test_clearing_through_a_canonical_bot_chat_ancestor_keeps_its_name(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _compressed(store, root_title=CANONICAL_BOT_CHAT_TITLE)
    store.sessions.set_hidden("chat", True)
    store.sessions.set_title("chat-2", "")
    # CONTRACT: the lineage clear writes each row under the title rules, so the hidden canonical Bot Chat's name refuses and nothing commits.
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        _rename(store, epoch, "chat", "", "b1")
    assert store.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE) is not None


def test_a_rotating_compress_moves_the_name_to_the_new_segment(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="owner")
    sid = str(local_session_id(store.owner_profile, "p", "create-1"))
    store.sessions.create(NewSession(session_id=sid, source="tui"))
    route = "agent:main:local:" + sid
    entry = {"session_key": route, "session_id": sid, "created_at": "2026-10-27T09:00:00", "updated_at": "2026-10-27T09:00:00"}
    store.runtime.seed_local_receipt(sid, {"session_id": sid, "profile_id": store.owner_profile, "principal_id": "p", "request_id": "create-1", "route": route,
                                           "entry": entry, "lineage": [sid], "policy": {"source": "tui", "model": "m1", "config_json": '{"model": {"provider": "p1"}}', "cwd": "/w"}})
    _rename(store, epoch, sid, "Named chat", "name")
    row = store.sessions.get(sid)
    fence = dict(principal_id="p", session_id=sid, request_id="c", expected_revision=row.runtime_revision, expected_generation=row.runtime_generation, command=rt.Compress())
    snapshot = store.runtime.prepare_mutation(epoch, **fence)
    values = {"in_place": False, "messages": [{"role": "user", "content": "hi"}], "summary": "s"}
    compress = dict(fence, prepared=rt.PreparedMutation(snapshot, values), owner=None)
    if getattr(backend, "refuses_rotating_publication", False):
        # Reference limit: past its fence the reference refuses the publication itself (module docstring of mutation_commands).
        with pytest.raises(CapabilityUnsupported):
            store.runtime.mutate(epoch, **compress)
        pytest.skip("backend does not declare rotating publications: the fence ran, the publication refused CapabilityUnsupported")
    child = store.runtime.mutate(epoch, **compress).result["target_session_id"]
    # CONTRACT: a rotation moves the conversation's name onto its live segment.
    assert (store.sessions.get(child).title, store.sessions.get(sid).title) == ("Named chat", None)
    assert store.sessions.get_by_title("Named chat").session_id == child
