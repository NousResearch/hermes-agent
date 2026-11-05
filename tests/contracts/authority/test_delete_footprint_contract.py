"""Contract cases for ``SessionRuntimeRepository.delete_footprint`` and the target walk ``mutate`` runs for ``Delete``.

The footprint cases fetch the operation first, so a backend that has not built it skips as staged; the ``mutate`` cases run on every backend. Children are compared as sets: the read defines no order.
"""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal, SessionId
from persistence.authority import runtime as rt
from persistence.authority.admissions import GateRefusal
from persistence.authority.bindings import local_session_id
from persistence.conversation.sessions import NewSession
from tests.contracts._staged import operation
from tests.contracts.authority.test_bindings_contract import begin, make_binding


def _chain(store, root: str, tip: str) -> None:
    store.sessions.create(NewSession(session_id=SessionId(root), source="cli"))
    store.sessions.create(NewSession(session_id=SessionId(tip), source="cli", parent_session_id=SessionId(root)))
    store.sessions.end(SessionId(root), "compression")


def _local_with_segment(store, epoch: int) -> SessionId:
    """A reserved local conversation whose target a reset moved to ``f-seg-2``."""
    sid = local_session_id("default", "user-1", "req-F")
    binding = make_binding(sid, route="route-req-F", request_id="req-F")
    store.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    store.bindings.reset_target(epoch=epoch, parent_session_id=sid, entry={**dict(binding.entry), "session_id": "f-seg-2"})
    return SessionId(sid)


def _bad_receipt(store, sid: str) -> None:
    """A row whose local receipt names an id its identity does not derive."""
    store.sessions.create(NewSession(session_id=SessionId(sid), source="cli"))
    route = "route-bad"
    entry = {"session_key": route, "session_id": sid, "created_at": "2026-09-16T10:00:00+00:00", "updated_at": "2026-09-16T10:00:00+00:00"}
    store.runtime.seed_local_receipt(sid, {"session_id": sid, "profile_id": "default", "principal_id": "user-1", "request_id": "req-bad",
                                           "route": route, "entry": entry, "policy": {"source": "cli", "model": "model-a", "cwd": "/workspace"}})


def _delete(store, epoch: int, sid: str, request_id: str, owner=None, **kwargs) -> rt.MutationOutcome:
    row = store.sessions.get(SessionId(sid))
    revision, generation = (row.runtime_revision, row.runtime_generation) if row is not None else (0, 0)
    return store.runtime.mutate(epoch, principal_id="p", session_id=sid, request_id=request_id, expected_revision=revision,
                                expected_generation=generation, command=rt.Delete(), owner=owner, **kwargs)


def _footprint(result: rt.DeleteFootprint) -> tuple[tuple[str, ...], set[str]]:
    return tuple(result.targets), set(result.children)


# ── delete_footprint ──────────────────────────────────────────────────────


def test_footprint_of_a_lone_row_and_a_missing_row(backend, scope_a):
    store = backend.open(scope_a)
    footprint = operation(store, "runtime", "delete_footprint")
    store.sessions.create(NewSession(session_id=SessionId("f-lone"), source="cli"))
    # CONTRACT: a lone row targets itself and has no children; a missing row answers the id alone.
    assert _footprint(footprint(SessionId("f-lone"), owner=None)) == (("f-lone",), set())
    assert _footprint(footprint(SessionId("f-missing"), owner=None)) == (("f-missing",), set())


def test_footprint_of_a_compressed_chain_from_either_end(backend, scope_a):
    store = backend.open(scope_a)
    footprint = operation(store, "runtime", "delete_footprint")
    _chain(store, "f-root", "f-tip")
    # CONTRACT: the walk runs both ways along the chain; the id comes first, the rest sorted; the tip is the root's child.
    assert _footprint(footprint(SessionId("f-root"), owner=None)) == (("f-root", "f-tip"), {"f-tip"})
    assert _footprint(footprint(SessionId("f-tip"), owner=None)) == (("f-tip", "f-root"), {"f-tip"})


def test_footprint_takes_delegate_subtrees_and_lists_branch_children(backend, scope_a):
    store = backend.open(scope_a)
    footprint = operation(store, "runtime", "delete_footprint")
    store.sessions.create(NewSession(session_id=SessionId("f-parent"), source="cli"))
    store.sessions.create(NewSession(session_id=SessionId("f-branch"), source="cli", parent_session_id=SessionId("f-parent"), model_config={"_branched_from": "f-parent"}))
    store.sessions.create(NewSession(session_id=SessionId("f-del"), source="cli", parent_session_id=SessionId("f-parent"), model_config={"_delegate_from": "f-parent"}))
    store.sessions.create(NewSession(session_id=SessionId("f-del-kid"), source="cli", parent_session_id=SessionId("f-del"), model_config={"_delegate_from": "f-del"}))
    # CONTRACT: the delegate subtree is a target; a branch child is not, and shows among the children.
    assert _footprint(footprint(SessionId("f-parent"), owner=None)) == (("f-parent", "f-del", "f-del-kid"), {"f-branch", "f-del", "f-del-kid"})
    assert _footprint(footprint(SessionId("f-del"), owner=None)) == (("f-del", "f-del-kid"), {"f-del-kid"})
    assert _footprint(footprint(SessionId("f-branch"), owner=None)) == (("f-branch",), set())


def test_footprint_of_a_local_conversation_takes_every_segment(backend, scope_a):
    store = backend.open(scope_a)
    footprint = operation(store, "runtime", "delete_footprint")
    owner = _local_with_segment(store, begin(store))
    # CONTRACT: the owner and any segment each reach the whole conversation.
    assert _footprint(footprint(owner, owner=None)) == ((owner, "f-seg-2"), {"f-seg-2"})
    assert _footprint(footprint(SessionId("f-seg-2"), owner=None)) == (("f-seg-2", owner), {"f-seg-2"})


def test_footprint_refuses_a_receipt_whose_lineage_does_not_validate(backend, scope_a):
    store = backend.open(scope_a)
    footprint = operation(store, "runtime", "delete_footprint")
    _bad_receipt(store, "f-bad")
    # CONTRACT: the walk's lineage check refuses rather than guessing the conversation.
    with pytest.raises(AuthorityRefusal, match="^storage_unavailable$"):
        footprint(SessionId("f-bad"), owner=None)


def test_footprint_targets_are_what_delete_deletes(backend, scope_a):
    store = backend.open(scope_a)
    footprint = operation(store, "runtime", "delete_footprint")
    epoch = begin(store)
    _chain(store, "e-root", "e-tip")
    store.sessions.create(NewSession(session_id=SessionId("e-branch"), source="cli", parent_session_id=SessionId("e-tip"), model_config={"_branched_from": "e-tip"}))
    owner = _local_with_segment(store, epoch)
    chain, local = footprint(SessionId("e-tip"), owner=None), footprint(SessionId("f-seg-2"), owner=None)
    assert _footprint(chain) == (("e-tip", "e-root"), {"e-branch", "e-tip"})
    # CONTRACT: the footprint's targets are exactly the ids mutate(Delete) deletes, in its order.
    assert _delete(store, epoch, "e-tip", "del-e").result["deleted_ids"] == list(chain.targets)
    assert _delete(store, epoch, "f-seg-2", "del-local").result["deleted_ids"] == list(local.targets) == ["f-seg-2", owner]


# ── mutate(Delete) over the same walk ─────────────────────────────────────


def test_delete_takes_a_compressed_chain_from_either_end(backend, scope_a):
    store = backend.open(scope_a)
    epoch = begin(store)
    _chain(store, "d-root", "d-tip")
    _chain(store, "e-root", "e-tip")
    store.sessions.create(NewSession(session_id=SessionId("e-branch"), source="cli", parent_session_id=SessionId("e-tip"), model_config={"_branched_from": "e-tip"}))
    # CONTRACT: deleting either end deletes the whole chain; a branch child of it survives, detached.
    assert _delete(store, epoch, "d-root", "del-d").result["deleted_ids"] == ["d-root", "d-tip"]
    assert _delete(store, epoch, "e-tip", "del-e").result["deleted_ids"] == ["e-tip", "e-root"]
    assert [store.sessions.get(SessionId(sid)) for sid in ("d-root", "d-tip", "e-root", "e-tip")] == [None] * 4
    assert store.sessions.get(SessionId("e-branch")).parent_session_id is None


def test_delete_of_a_local_segment_takes_the_conversation(backend, scope_a):
    store = backend.open(scope_a)
    epoch = begin(store)
    owner = _local_with_segment(store, epoch)
    # CONTRACT: a segment deletes as its whole local conversation.
    assert _delete(store, epoch, "f-seg-2", "del-local").result["deleted_ids"] == ["f-seg-2", owner]
    assert store.sessions.get(owner) is None and store.sessions.get(SessionId("f-seg-2")) is None


def test_delete_of_a_missing_row_refuses_not_found(backend, scope_a):
    store = backend.open(scope_a)
    epoch = begin(store)
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        _delete(store, epoch, "d-missing", "del-missing")


def test_delete_walks_its_targets_before_the_live_guard(backend, scope_a):
    store = backend.open(scope_a)
    epoch = begin(store)
    _bad_receipt(store, "f-bad")
    store.sessions.create(NewSession(session_id=SessionId("f-lone"), source="cli"))
    guard = GateRefusal("runtime_coordination_required", field="live")
    # CONTRACT: the walk's lineage refusal comes before the live guard's, and without a guard too.
    for kwargs in ({"live_guard": guard}, {}):
        with pytest.raises(AuthorityRefusal, match="^storage_unavailable$"):
            _delete(store, epoch, "f-bad", "del-bad", **kwargs)
    with pytest.raises(AuthorityRefusal) as refused:
        _delete(store, epoch, "f-lone", "del-lone", live_guard=guard)
    assert (refused.value.reason, refused.value.field) == ("runtime_coordination_required", "live")
    assert store.sessions.get(SessionId("f-bad")) is not None and store.sessions.get(SessionId("f-lone")) is not None
