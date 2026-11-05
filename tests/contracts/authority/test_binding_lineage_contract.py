"""Contract cases for the local lineage verbs of ``SessionBindingRepository``.

A local session's receipt names its current target and the lineage of its physical segments; ``/new`` (``reset_target``) and compression move both. Each case pins one verb's behaviour through the public verbs; corrupt receipts are seeded through the reference meta space and gate on failpoints.
"""

from __future__ import annotations

import json


from persistence import SessionId
from persistence.authority.bindings import local_session_id
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from tests.contracts._reference._authority import authority_state
from tests.contracts.authority.test_bindings_contract import assert_refusal, begin, make_binding
from tests.contracts.conftest import require_failpoints

PRINCIPAL = "user-1"


def _local(store, request_id: str = "req-L"):
    """A freshly reserved local session with a derived id; returns ``(epoch, session_id, binding)``."""
    epoch = begin(store)
    sid = local_session_id("default", PRINCIPAL, request_id)
    binding = make_binding(sid, route="route-" + request_id, request_id=request_id)
    store.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    return epoch, SessionId(sid), binding


def _reset(store, epoch, parent: str, child: str, binding) -> None:
    store.bindings.reset_target(epoch=epoch, parent_session_id=SessionId(parent), entry={**dict(binding.entry), "session_id": child})


def _untracked_step(store, parent: str, child: str, owner: str, route: str) -> None:
    """A compression step a receipt-unaware writer published: the child copies the route columns and the parent ends by compression."""
    store.sessions.create(NewSession(session_id=SessionId(child), source="cli", parent_session_id=SessionId(parent),
                                     session_key=route, chat_id=owner, user_id=PRINCIPAL))
    store.sessions.end(SessionId(parent), "compression")


def test_reset_target_moves_the_receipt_and_route_to_the_child(backend, scope_a):
    # CONTRACT: ``reset_target``: a child row resets from the parent, the parent ends session_reset, the receipt's entry and lineage and the route move to the child.
    store_a = backend.open(scope_a)
    epoch, sid, binding = _local(store_a)
    _reset(store_a, epoch, sid, "seg-2", binding)
    receipt = store_a.bindings.receipt(sid)
    assert (receipt.current_target, receipt.lineage) == ("seg-2", (sid, "seg-2"))
    parent, child = store_a.sessions.get(sid), store_a.sessions.get(SessionId("seg-2"))
    assert (parent.end_reason, child.parent_session_id, child.session_key, child.chat_id) == ("session_reset", sid, binding.route, sid)
    assert json.loads(store_a.routing.load_routing_entries()[binding.route])["session_id"] == "seg-2"
    assert store_a.bindings.validate_lineage(receipt) == "seg-2"


def test_reset_target_refusals(backend, scope_a):
    # CONTRACT: a missing parent or one ended by compression is admission_conflict; a parent without an owner receipt is storage_unavailable; a parent that is not the receipt's tip is admission_conflict.
    store_a = backend.open(scope_a)
    epoch, sid, binding = _local(store_a)
    assert_refusal(lambda: _reset(store_a, epoch, "missing", "x", binding), "admission_conflict")
    store_a.sessions.create(NewSession(session_id=SessionId("plain"), source="cli", chat_id="nobody"))
    assert_refusal(lambda: _reset(store_a, epoch, "plain", "x", binding), "storage_unavailable")
    _reset(store_a, epoch, sid, "seg-2", binding)
    assert_refusal(lambda: _reset(store_a, epoch, sid, "seg-3", binding), "admission_conflict")
    assert store_a.sessions.get(SessionId("seg-3")) is None


def test_lineage_owner_and_conversation_ids_follow_the_receipt(backend, scope_a):
    # CONTRACT: a segment carries its owner as chat_id and is in the owner's lineage; anything else owns itself.
    store_a = backend.open(scope_a)
    epoch, sid, binding = _local(store_a)
    _reset(store_a, epoch, sid, "seg-2", binding)
    assert store_a.bindings.lineage_owner(SessionId("seg-2")) == sid
    assert store_a.bindings.lineage_owner(sid) == sid
    store_a.sessions.create(NewSession(session_id=SessionId("copied"), source="cli", chat_id=sid))
    assert store_a.bindings.lineage_owner(SessionId("copied")) == "copied"
    assert store_a.bindings.conversation_ids(SessionId("seg-2")) == [sid, "seg-2"]
    assert store_a.bindings.conversation_ids(SessionId("copied")) == ["copied"]


def test_marks_ids_and_lineages(backend, scope_a):
    # CONTRACT: ``binding_marks`` reads both markers and the lineage owner only when asked and marked; ``local_session_ids`` lists receipt ids; ``list_lineages`` skips a receipt without a list lineage.
    store_a = backend.open(scope_a)
    epoch, sid, binding = _local(store_a)
    marks = store_a.bindings.binding_marks(sid, lineage_of=sid)
    assert (marks.policy, marks.legacy, marks.lineage_owner) == (True, False, sid)
    unmarked = store_a.bindings.binding_marks(SessionId("other"), lineage_of=sid)
    assert (unmarked.policy, unmarked.legacy, unmarked.lineage_owner) == (False, False, None)
    assert store_a.bindings.local_session_ids() == [sid]
    assert store_a.bindings.list_lineages() == []
    _reset(store_a, epoch, sid, "seg-2", binding)
    lineages = store_a.bindings.list_lineages()
    assert [(item.owner, item.lineage) for item in lineages] == [(sid, (sid, "seg-2"))]


def test_validate_lineage_refuses_receipts_the_rows_do_not_support(backend, scope_a):
    # CONTRACT: ``validate_lineage``: the id must be the local identity, the lineage must run from the root to the target through real hops, and the tip must have no compression step.
    store_a = backend.open(scope_a)
    epoch, sid, binding = _local(store_a)
    receipt = store_a.bindings.receipt(sid)
    assert store_a.bindings.validate_lineage(receipt) == sid
    store_a.bindings.reserve(epoch=epoch, binding=make_binding("not-derived", route="r-nd"), owner=None)
    assert_refusal(lambda: store_a.bindings.validate_lineage(store_a.bindings.receipt(SessionId("not-derived"))), "storage_unavailable")
    from dataclasses import replace

    for broken in (replace(receipt, lineage=()), replace(receipt, lineage=(sid, sid)), replace(receipt, lineage=(SessionId("x"),)),
                   replace(receipt, lineage=(sid, SessionId("seg-ghost")), entry={**dict(receipt.entry), "session_id": "seg-ghost"})):
        assert_refusal(lambda broken=broken: store_a.bindings.validate_lineage(broken), "storage_unavailable")
    _untracked_step(store_a, sid, "cmp-1", sid, binding.route)
    assert_refusal(lambda: store_a.bindings.validate_lineage(store_a.bindings.receipt(sid)), "storage_unavailable")


def test_adopt_untracked_compression_appends_the_steps_once(backend, scope_a):
    # CONTRACT: ``adopt_untracked_compression``: steps carrying the receipt's route, owner and principal are appended and the receipt, route and target move to the last; a stale ``expected`` or nothing adoptable writes nothing.
    store_a = backend.open(scope_a)
    epoch, sid, binding = _local(store_a)
    receipt = store_a.bindings.receipt(sid)
    assert store_a.bindings.compression_adoptable(receipt) is False
    _untracked_step(store_a, sid, "cmp-1", sid, binding.route)
    _untracked_step(store_a, "cmp-1", "cmp-2", sid, binding.route)
    assert store_a.bindings.compression_adoptable(receipt) is True
    from dataclasses import replace

    store_a.bindings.adopt_untracked_compression(epoch=epoch, session_id=sid, expected=replace(receipt, route="stale"))
    assert store_a.bindings.receipt(sid) == receipt
    store_a.bindings.adopt_untracked_compression(epoch=epoch, session_id=sid, expected=receipt)
    adopted = store_a.bindings.receipt(sid)
    assert (adopted.current_target, adopted.lineage) == ("cmp-2", (sid, "cmp-1", "cmp-2"))
    assert json.loads(store_a.routing.load_routing_entries()[binding.route])["session_id"] == "cmp-2"
    assert store_a.bindings.validate_lineage(adopted) == "cmp-2"
    assert store_a.bindings.compression_adoptable(adopted) is False


def test_compression_steps_of_another_identity_are_not_adoptable(backend, scope_a):
    store_a = backend.open(scope_a)
    _epoch, sid, binding = _local(store_a)
    _untracked_step(store_a, sid, "cmp-x", "someone", binding.route)
    assert store_a.bindings.compression_adoptable(store_a.bindings.receipt(sid)) is False


def test_end_idle_stamps_only_an_idle_open_target(backend, scope_a):
    # CONTRACT: ``end_idle``: 0 for an ended row; 1 for the idle open target, stamped with the reason.
    store_a = backend.open(scope_a)
    epoch, sid, _binding = _local(store_a)
    assert store_a.bindings.end_idle(epoch=epoch, session_id=sid, target_id=sid, reason="acp_idle") == 1
    assert store_a.sessions.get(sid).end_reason == "acp_idle"
    assert store_a.bindings.end_idle(epoch=epoch, session_id=sid, target_id=sid, reason="acp_idle") == 0
    assert store_a.bindings.end_idle(epoch=epoch, session_id=sid, target_id=SessionId("missing"), reason="x") == 0


def test_target_has_history_counts_messages_and_children(backend, scope_a):
    # CONTRACT: a transcript row or a child session is history.
    store_a = backend.open(scope_a)
    for sid in ("h-empty", "h-msg", "h-parent"):
        store_a.sessions.create(NewSession(session_id=SessionId(sid), source="cli"))
    store_a.messages.append(SessionId("h-msg"), NewMessage(role="user", content="hi"))
    store_a.sessions.create(NewSession(session_id=SessionId("h-child"), source="cli", parent_session_id=SessionId("h-parent")))
    assert [store_a.bindings.target_has_history(sid) for sid in ("h-empty", "h-msg", "h-parent", "nowhere")] == [False, True, True, False]


def test_record_peer_with_ancestors_hands_a_moved_conversation_to_a_messaging_lane(backend, scope_a):
    # CONTRACT: ``record_session_peer(include_compression_ancestors=True)``: when the caller says the source names a messaging platform, a receipt whose current target the stamp moved is retired in the same write, its earlier segments take the lane's identity and the title follows the head; otherwise the receipt stays.
    store_a = backend.open(scope_a)
    epoch, sid, binding = _local(store_a)
    _reset(store_a, epoch, sid, "seg-2", binding)
    store_a.sessions.set_title(sid, "Named chat")
    peer = {"source": "telegram", "session_key": "agent:main:telegram:dm:1", "chat_id": "1", "include_compression_ancestors": True}
    store_a.routing.record_session_peer("seg-2", **peer)
    assert store_a.bindings.receipt(sid).current_target == "seg-2"
    assert store_a.sessions.get(sid).session_key == binding.route
    store_a.routing.record_session_peer("seg-2", **peer, messaging_lane=True)
    assert_refusal(lambda: store_a.bindings.receipt(sid), "storage_unavailable")
    owner_row = store_a.sessions.get(sid)
    assert (owner_row.session_key, owner_row.source, owner_row.chat_id) == ("agent:main:telegram:dm:1", "telegram", "1")
    assert store_a.sessions.get(SessionId("seg-2")).title == "Named chat"


def test_release_handed_off_receipts_hands_the_conversation_to_the_lane(backend, scope_a):
    # CONTRACT: ``release_handed_off_receipts``: a receipt whose current target row carries another route and a lane source is retired, its earlier segments take the lane's identity, and the title follows the head; other receipts stay.
    store_a = backend.open(scope_a)
    epoch, sid, binding = _local(store_a)
    _reset(store_a, epoch, sid, "seg-2", binding)
    _keep_epoch, other, _other_binding = _local(store_a, "req-other")
    store_a.sessions.set_title(sid, "Named chat")
    require_failpoints(backend)
    with authority_state(backend, scope_a).transaction() as staged:
        staged.scope["sessions"]["seg-2"].update(session_key="agent:main:telegram:dm:1", source="telegram", chat_id="1")
    epoch = begin(store_a, "instance-b")
    assert store_a.bindings.release_handed_off_receipts(epoch=epoch, lanes=frozenset({"discord"})) == {}
    handed = store_a.bindings.release_handed_off_receipts(epoch=epoch, lanes=frozenset({"telegram"}))
    assert handed == {sid: ("seg-2", "telegram")}
    assert_refusal(lambda: store_a.bindings.receipt(sid), "storage_unavailable")
    assert store_a.bindings.receipt(other).session_id == other
    owner_row = store_a.sessions.get(sid)
    assert (owner_row.session_key, owner_row.source, owner_row.chat_id) == ("agent:main:telegram:dm:1", "telegram", "1")
    assert store_a.sessions.get(SessionId("seg-2")).title == "Named chat"
