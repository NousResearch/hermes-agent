"""Contract cases for the local lineage ids of a session listing: ``SessionListEntry.lineage_ids`` on the row that represents a local conversation (``persistence/conversation/sessions.py``).

A local conversation's creation receipt lists its physical segments; a listing without children leaves out every segment before the latest reset and names the whole lineage on the row of that segment, after its compression projection, receipt ids first. Conversations are built through the binding verbs; receipts the verbs cannot write go in through the ``runtime.seed_local_receipt`` seam. Each case pins one behaviour on both backends.
"""

from __future__ import annotations

from persistence import SessionId
from persistence.authority.bindings import local_session_id
from persistence.conversation.sessions import (
    END_REASON_COMPRESSION,
    END_REASON_SESSION_RESET,
    NewSession,
    SessionLineageKind,
    SessionOrder,
    SessionQuery,
    SessionWindow,
)
from tests.contracts._scopes import MEDIA_ROOT
from tests.contracts.authority.test_bindings_contract import begin, make_binding

PRINCIPAL = "user-1"
WINDOW = SessionWindow(limit=50)


def _local(store, epoch, request_id):
    sid = SessionId(local_session_id("default", PRINCIPAL, request_id))
    binding = make_binding(sid, route="route-" + request_id, request_id=request_id)
    store.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    return sid, binding


def _reset(store, epoch, parent, child, binding):
    store.bindings.reset_target(epoch=epoch, parent_session_id=SessionId(parent), entry={**dict(binding.entry), "session_id": child})


def _step(store, owner, parent, child, binding):
    """A compression step past the receipt's tip: the child carries the route, owner and principal, the parent ends by compression."""
    store.sessions.create(NewSession(session_id=SessionId(child), source="cli", parent_session_id=SessionId(parent),
                                     session_key=binding.route, chat_id=owner, user_id=PRINCIPAL))
    store.sessions.end(SessionId(parent), END_REASON_COMPRESSION)


def _compress(store, epoch, owner, parent, child, binding):
    _step(store, owner, parent, child, binding)
    store.bindings.adopt_untracked_compression(epoch=epoch, session_id=owner, expected=store.bindings.receipt(owner))


def _ended_by_reset(store, *session_ids):
    for sid in session_ids:
        store.sessions.create(NewSession(session_id=SessionId(sid), source="cli"))
        store.sessions.end(SessionId(sid), END_REASON_SESSION_RESET)


def _listed(store, query=SessionQuery(), window=WINDOW):
    """``{id: (lineage_root_id, lineage_ids)}`` of the listing."""
    return {e.session.session_id: (e.lineage_root_id, e.lineage_ids) for e in store.sessions.list_sessions(query, window)}


def test_a_reset_lineage_is_listed_once_naming_every_segment(backend, scope_a):
    # CONTRACT: segments start at the creation id and after each session_reset; every start but the last leaves the listing and the last carries the receipt lineage.
    store = backend.open(scope_a)
    epoch = begin(store)
    owner, binding = _local(store, epoch, "req-r")
    _reset(store, epoch, owner, "r-1", binding)
    _reset(store, epoch, "r-1", "r-2", binding)
    for query in (SessionQuery(), SessionQuery(order=SessionOrder.LAST_ACTIVE), SessionQuery(project_compression_tips=False)):
        assert _listed(store, query) == {"r-2": (None, (owner, "r-1", "r-2"))}


def test_a_listing_with_children_lists_every_segment_without_lineage_ids(backend, scope_a):
    # CONTRACT: only a listing without children reads the local lineages.
    store = backend.open(scope_a)
    epoch = begin(store)
    owner, binding = _local(store, epoch, "req-r")
    _reset(store, epoch, owner, "r-1", binding)
    assert _listed(store, SessionQuery(include_children=True)) == {owner: (None, ()), "r-1": (None, ())}


def test_a_compressed_local_chat_carries_its_lineage_on_the_projected_row(backend, scope_a):
    # CONTRACT: the head is found by the projected row's compression root, and the receipt lineage leads the chain ids; unprojected, the root row carries the lineage.
    store = backend.open(scope_a)
    epoch = begin(store)
    owner, binding = _local(store, epoch, "req-c")
    _compress(store, epoch, owner, owner, "c-1", binding)
    assert _listed(store) == {"c-1": (owner, (owner, "c-1"))}
    assert _listed(store, SessionQuery(project_compression_tips=False)) == {owner: (None, (owner, "c-1"))}


def test_compression_after_a_reset_merges_the_chain_after_the_receipt(backend, scope_a):
    # CONTRACT: the chain ids are the receipt lineage followed by the row's chain ids, each once: a step the receipt has not adopted joins at the end; the head is the reset segment the chain is rooted at.
    store = backend.open(scope_a)
    epoch = begin(store)
    owner, binding = _local(store, epoch, "req-m")
    _compress(store, epoch, owner, owner, "m-1", binding)
    _reset(store, epoch, "m-1", "m-2", binding)
    _compress(store, epoch, owner, "m-2", "m-3", binding)
    _step(store, owner, "m-3", "m-4", binding)
    lineage = (owner, "m-1", "m-2", "m-3", "m-4")
    assert _listed(store) == {"m-4": ("m-2", lineage)}
    assert _listed(store, SessionQuery(order=SessionOrder.LAST_ACTIVE)) == {"m-4": ("m-2", lineage)}
    assert _listed(store, SessionQuery(project_compression_tips=False)) == {"m-2": (None, (owner, "m-1", "m-2", "m-3"))}


def test_two_lineages_and_a_plain_local_chat_keep_their_own_ids(backend, scope_a):
    # CONTRACT: each receipt maps its own head; a receipt without a lineage of two or more segments maps nothing.
    store = backend.open(scope_a)
    epoch = begin(store)
    first, first_binding = _local(store, epoch, "req-a")
    _reset(store, epoch, first, "a-1", first_binding)
    second, second_binding = _local(store, epoch, "req-b")
    _compress(store, epoch, second, second, "b-1", second_binding)
    solo, _binding = _local(store, epoch, "req-solo")
    assert _listed(store) == {"a-1": (None, (first, "a-1")), "b-1": (second, (second, "b-1")), solo: (None, ())}


def test_branch_and_delegate_children_of_a_head_carry_no_lineage_ids(backend, scope_a):
    # CONTRACT: only the head row of a receipt is stamped; a branch or delegate child of it is its own conversation.
    store = backend.open(scope_a)
    epoch = begin(store)
    owner, binding = _local(store, epoch, "req-r")
    _reset(store, epoch, owner, "r-1", binding)
    store.sessions.create(NewSession(session_id=SessionId("r-branch"), source="cli", parent_session_id=SessionId("r-1"), lineage_kind=SessionLineageKind.BRANCH))
    store.sessions.create(NewSession(session_id=SessionId("r-delegate"), source="cli", parent_session_id=SessionId("r-1"), lineage_kind=SessionLineageKind.DELEGATE))
    assert _listed(store) == {"r-1": (None, (owner, "r-1")), "r-branch": (None, ())}
    assert _listed(store, SessionQuery(include_subagents=True)) == {"r-1": (None, (owner, "r-1")), "r-branch": (None, ()), "r-delegate": (None, ())}


def test_a_pinned_head_is_back_filled_with_its_lineage_and_a_pinned_earlier_segment_is_not(backend, scope_a):
    # CONTRACT: the pinned back-fill leaves out earlier segments too, and the back-filled head is stamped.
    store = backend.open(scope_a)
    epoch = begin(store)
    owner, binding = _local(store, epoch, "req-r")
    _reset(store, epoch, owner, "r-1", binding)
    store.sessions.set_pinned(SessionId("r-1"), True)
    store.sessions.set_pinned(owner, True)
    store.sessions.create(NewSession(session_id=SessionId("newer"), source="cli"))
    assert _listed(store, SessionQuery(include_pinned=True), SessionWindow(limit=1)) == {"newer": (None, ()), "r-1": (None, (owner, "r-1"))}


def test_an_archived_head_keeps_its_lineage_in_the_archived_listings(backend, scope_a):
    # CONTRACT: the lineage stamp does not depend on the archive filter.
    store = backend.open(scope_a)
    epoch = begin(store)
    owner, binding = _local(store, epoch, "req-q")
    _reset(store, epoch, owner, "q-1", binding)
    store.sessions.set_archived(SessionId("q-1"), True)
    assert _listed(store) == {}
    assert _listed(store, SessionQuery(archived_only=True)) == {"q-1": (None, (owner, "q-1"))}
    assert _listed(store, SessionQuery(include_archived=True)) == {"q-1": (None, (owner, "q-1"))}


def test_id_search_stamps_the_head_and_never_finds_an_earlier_segment(backend, scope_a):
    # CONTRACT: the id search is a listing without children, so it stamps the head and leaves earlier segments out.
    store = backend.open(scope_a)
    epoch = begin(store)
    owner, binding = _local(store, epoch, "req-r")
    _reset(store, epoch, owner, "r-1", binding)
    assert [(e.session.session_id, e.lineage_ids) for e in store.sessions.search_by_id("r-1")] == [("r-1", (owner, "r-1"))]
    assert store.sessions.search_by_id(owner) == ()


def test_a_summary_carries_no_local_lineage(backend, scope_a):
    # CONTRACT: the one-row read is the rich row alone, never stamped.
    store = backend.open(scope_a)
    epoch = begin(store)
    owner, binding = _local(store, epoch, "req-r")
    _reset(store, epoch, owner, "r-1", binding)
    for sid in (owner, SessionId("r-1")):
        entry = store.sessions.summary(sid)
        assert (entry.lineage_root_id, entry.lineage_ids) == (None, ()), sid


def test_a_deleted_lineage_leaves_no_row_naming_it(backend, scope_a):
    # CONTRACT: with the listing above, deleting one segment deletes the conversation, and no remaining row carries its ids.
    store = backend.open(scope_a)
    epoch = begin(store)
    owner, binding = _local(store, epoch, "req-x")
    _reset(store, epoch, owner, "x-1", binding)
    assert set(store.sessions.delete(SessionId("x-1"), media_root=MEDIA_ROOT)) == {owner, "x-1"}
    assert _listed(store) == {}
    assert _listed(store, SessionQuery(include_children=True)) == {}


def test_a_receipt_naming_a_missing_segment(backend, scope_a):
    # CONTRACT: a missing row has no end_reason, so the segment after it is no start; a missing latest start hides the earlier ones with nothing stamped.
    store = backend.open(scope_a)
    _ended_by_reset(store, "g-0", "h-0")
    store.sessions.create(NewSession(session_id=SessionId("h-2"), source="cli"))
    store.runtime.seed_local_receipt("g-0", {"session_id": "g-0", "lineage": ["g-0", "g-ghost"]})
    store.runtime.seed_local_receipt("h-0", {"session_id": "h-0", "lineage": ["h-0", "h-ghost", "h-2"]})
    assert _listed(store) == {"h-2": (None, ())}


def test_receipts_with_a_non_string_segment_or_one_segment_map_nothing(backend, scope_a):
    # CONTRACT: a lineage of fewer than two segments or with a non-string segment is skipped, so its rows stay listed unstamped.
    store = backend.open(scope_a)
    _ended_by_reset(store, "e-0", "s-0")
    store.runtime.seed_local_receipt("e-0", {"session_id": "e-0", "lineage": ["e-0", 5]})
    store.runtime.seed_local_receipt("s-0", {"session_id": "s-0", "lineage": ["s-0"]})
    assert _listed(store) == {"e-0": (None, ()), "s-0": (None, ())}


def test_a_receipt_with_a_non_string_owner_maps_nothing(backend, scope_a):
    # CONTRACT: a receipt whose ``session_id`` is not a string is skipped, so no segment leaves the listing and none is stamped.
    store = backend.open(scope_a)
    _ended_by_reset(store, "n-0")
    store.sessions.create(NewSession(session_id=SessionId("n-1"), source="cli"))
    store.runtime.seed_local_receipt("n-0", {"session_id": 7, "lineage": ["n-0", "n-1"]})
    assert _listed(store) == {"n-0": (None, ()), "n-1": (None, ())}
