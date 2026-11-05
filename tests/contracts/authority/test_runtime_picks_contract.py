"""Contract cases for the runtime picks, the policy refresh and the viewer closes."""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal, FenceStale, StaleFenceKind
from persistence.authority.bindings import local_session_id
from persistence.authority.runtime import ChatPickParams, DeferredPick, ReasoningPick, SessionRuntimeState, Sidebar, chat_pick_key, mutation_receipt_key
from persistence.conversation.sessions import NewSession

SESSION = "s-pick"


def _seed(store, session_id: str = SESSION) -> int:
    store.sessions.create(NewSession(session_id=session_id, source="cli"))
    return store.runtime.begin_epoch(instance_id="instance-pick")


def _defer(store, epoch, request_id, operation, payload, session_id=SESSION):
    key, digest = mutation_receipt_key("p", session_id, request_id, operation=operation, payload=payload, expected_revision=0)
    return store.runtime.defer_model_pick(epoch, key=key, digest=digest, session_id=session_id, pick=DeferredPick(operation, "p", request_id, payload))


def test_a_deferred_model_pick_receipt_names_the_shown_model(backend, scope_a):
    # CONTRACT: a model pick's receipt names ``shown`` when the pick carries one (the configured default a queued reset adopts), else the payload's model; an exact retry returns the stored receipt and the slot keeps the payload.
    store = backend.open(scope_a)
    epoch = _seed(store)
    key, digest = mutation_receipt_key("p", SESSION, "m-reset", operation="model", payload={"model": "reset"}, expected_revision=0)

    def defer(shown):
        return store.runtime.defer_model_pick(epoch, key=key, digest=digest, session_id=SESSION,
                                              pick=DeferredPick("model", "p", "m-reset", {"model": "reset"}, shown=shown))
    receipt = defer("gpt-default")
    assert receipt["model"] == "gpt-default"
    assert defer("another-default") == receipt
    assert store.runtime.deferred_pick(SESSION).pick["payload"] == {"model": "reset"}


def test_defer_model_pick_keeps_the_latest_pick_and_its_receipt(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    receipt = _defer(store, epoch, "m1", "model", {"model": "gpt"})
    assert receipt == {"session_id": SESSION, "operation": "model", "status": "deferred", "model": "gpt", "revision": 0, "execution_generation": 0}
    # CONTRACT: an exact retry returns the stored receipt; the receipt is a mutation receipt.
    assert _defer(store, epoch, "m1", "model", {"model": "gpt"}) == receipt
    assert store.runtime.has_mutation_receipt("p", SESSION, "m1")
    level = _defer(store, epoch, "r1", "reasoning", {"effort": "high"})
    assert level["effort"] == "high"
    slot = store.runtime.deferred_pick(SESSION)
    assert slot.pick == {"principal_id": "p", "request_id": "m1", "payload": {"model": "gpt"}, "application_id": "deferred-pick:m1",
                         "reasoning": {"principal_id": "p", "request_id": "r1", "payload": {"effort": "high"}, "application_id": "deferred-pick:r1"}}
    # CONTRACT: a later model pick without a level keeps the older level; one with a level drops it.
    _defer(store, epoch, "m2", "model", {"model": "claude"})
    assert store.runtime.deferred_pick(SESSION).pick["reasoning"]["request_id"] == "r1"
    _defer(store, epoch, "m3", "model", {"model": "claude", "reasoning": "low"})
    assert "reasoning" not in store.runtime.deferred_pick(SESSION).pick
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        _defer(store, epoch, "m4", "model", {"model": "x"}, session_id="s-absent")
    with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
        _defer(store, epoch, "m5", "model", {"model": ""})
    with pytest.raises(FenceStale):
        _defer(store, epoch + 1, "m6", "model", {"model": "x"})


def test_forget_and_mark_asked_compare_the_stored_value(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    assert store.runtime.deferred_pick(SESSION) is None
    _defer(store, epoch, "m1", "model", {"model": "gpt"})
    slot = store.runtime.deferred_pick(SESSION)
    store.runtime.mark_deferred_pick_asked(SESSION, stored='{"stale":1}', confirm="tok")
    with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
        store.runtime.mark_deferred_pick_asked(SESSION, stored="not json", confirm="tok")
    assert store.runtime.deferred_pick(SESSION) == slot
    store.runtime.mark_deferred_pick_asked(SESSION, stored=slot.stored, confirm="tok")
    asked = store.runtime.deferred_pick(SESSION)
    assert asked.pick == {**slot.pick, "awaiting_confirm": "tok"}
    store.runtime.forget_deferred_pick(SESSION, stored=slot.stored)
    assert store.runtime.deferred_pick(SESSION) == asked
    store.runtime.forget_deferred_pick(SESSION, stored=asked.stored)
    assert store.runtime.deferred_pick(SESSION) is None


def test_chat_pick_prepare_and_commit_share_the_receipt(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    params = ChatPickParams(request_id="pick-1", operation="model", payload={"model": "gpt"}, expected_revision=0, expected_generation=0)
    key, digest = chat_pick_key("p", SESSION, params)
    assert store.runtime.prepare_chat_pick(epoch, key=key, digest=digest, session_id=SESSION, params=params) == SessionRuntimeState(revision=0, generation=0)
    committed = store.runtime.commit_chat_pick(epoch, key=key, digest=digest, session_id=SESSION, params=params, result={"model": "gpt"})
    assert committed == {"model": "gpt", "revision": 1}
    # CONTRACT: an exact retry answers the stored receipt from both verbs, past the moved revision.
    assert store.runtime.prepare_chat_pick(epoch, key=key, digest=digest, session_id=SESSION, params=params) == committed
    assert store.runtime.commit_chat_pick(epoch, key=key, digest=digest, session_id=SESSION, params=params, result={"model": "x"}) == committed
    other = ChatPickParams(request_id="pick-1", operation="model", payload={"model": "other"}, expected_revision=1, expected_generation=0)
    with pytest.raises(AuthorityRefusal, match="^admission_conflict$"):
        store.runtime.prepare_chat_pick(epoch, key=key, digest=chat_pick_key("p", SESSION, other)[1], session_id=SESSION, params=other)
    late = ChatPickParams(request_id="pick-2", operation="model", payload={"model": "x"}, expected_revision=0, expected_generation=0)
    with pytest.raises(FenceStale) as stale:
        store.runtime.prepare_chat_pick(epoch, key=chat_pick_key("p", SESSION, late)[0], digest=chat_pick_key("p", SESSION, late)[1], session_id=SESSION, params=late)
    assert stale.value.kind is StaleFenceKind.REVISION_CONFLICT
    with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
        chat_pick_key("p", SESSION, ChatPickParams(request_id="x", operation="model", payload={}, expected_revision=-1, expected_generation=0))


def test_viewer_close_records_are_epoch_fenced_compare_and_delete(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    assert store.runtime.viewer_close(SESSION) is None and store.runtime.viewer_closes() == []
    store.runtime.store_viewer_close(epoch, session_id=SESSION, record='{"op":"a","reason":"tui_close"}')
    store.runtime.store_viewer_close(epoch, session_id="s-other", record="b")
    assert store.runtime.viewer_close(SESSION) == '{"op":"a","reason":"tui_close"}'
    assert store.runtime.viewer_closes() == [("s-other", "b"), (SESSION, '{"op":"a","reason":"tui_close"}')]
    assert store.runtime.forget_viewer_close(epoch, session_id=SESSION, record="not-it") is False
    with pytest.raises(FenceStale):
        store.runtime.forget_viewer_close(epoch + 1, session_id=SESSION, record='{"op":"a","reason":"tui_close"}')
    with pytest.raises(FenceStale):
        store.runtime.store_viewer_close(epoch + 1, session_id=SESSION, record="c")
    assert store.runtime.forget_viewer_close(epoch, session_id=SESSION, record='{"op":"a","reason":"tui_close"}') is True
    assert store.runtime.viewer_close(SESSION) is None


def _local(store, request_id: str = "create-1") -> str:
    sid = str(local_session_id(store.owner_profile, "p", request_id))
    store.sessions.create(NewSession(session_id=sid, source="tui"))
    store.runtime.seed_local_receipt(sid, {"session_id": sid, "profile_id": store.owner_profile, "principal_id": "p", "request_id": request_id, "route": "agent:main:local:" + sid,
                                    "entry": {"session_id": sid}, "lineage": [sid], "policy": {"model": "m1", "config_json": "{}", "cwd": "/w"}})
    return sid


def test_prepare_mutation_returns_the_local_snapshot_or_the_replay(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-pick")
    sid = _local(store)
    snapshot = store.runtime.prepare_mutation(epoch, principal_id="p", session_id=sid, request_id="r", expected_revision=0, command=ReasoningPick("high"))
    assert (snapshot.target, snapshot.target_revision, snapshot.receipt["policy"]["model"]) == (sid, 0, "m1")
    # CONTRACT: without a local receipt the snapshot refuses runtime_coordination_required.
    store.sessions.create(NewSession(session_id="s-plain", source="cli"))
    with pytest.raises(AuthorityRefusal, match="^runtime_coordination_required$"):
        store.runtime.prepare_mutation(epoch, principal_id="p", session_id="s-plain", request_id="r", expected_revision=0, command=ReasoningPick("high"))
    store.runtime.mutate(epoch, principal_id="p", session_id="s-plain", request_id="flags", expected_revision=0, command=Sidebar(pinned=True), owner=None)
    replay = store.runtime.prepare_mutation(epoch, principal_id="p", session_id="s-plain", request_id="flags", expected_revision=0, command=Sidebar(pinned=True))
    assert replay.replayed and replay.result["pinned"] is True
    # CONTRACT: a busy session refuses the snapshot (require_target_advanceable).
    store.admissions.admit(epoch=epoch, principal_id="p", session_id=sid, request_id="turn", payload={"text": "t"})
    store.admissions.claim(epoch=epoch, session_id=sid)
    with pytest.raises(AuthorityRefusal, match="^session_busy$"):
        store.runtime.prepare_mutation(epoch, principal_id="p", session_id=sid, request_id="r", expected_revision=0, command=ReasoningPick("high"))


def test_commit_refreshed_policy_compares_the_policy_and_the_generation(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-pick")
    sid = _local(store)
    old = {"model": "m1", "config_json": "{}", "cwd": "/w"}
    new = {**old, "model": "m2"}
    with pytest.raises(FenceStale) as moved:
        store.runtime.commit_refreshed_policy(epoch, session_id=sid, old={**old, "model": "x"}, new=new, generation=0)
    assert moved.value.kind is StaleFenceKind.REVISION_CONFLICT
    with pytest.raises(FenceStale):
        store.runtime.commit_refreshed_policy(epoch, session_id=sid, old=old, new=new, generation=3)
    assert store.runtime.commit_refreshed_policy(epoch, session_id=sid, old=old, new=new, generation=0) == 1
    assert store.sessions.get(sid).runtime_generation == 1
    assert store.runtime.prepare_mutation(epoch, principal_id="p", session_id=sid, request_id="r", expected_revision=0, command=ReasoningPick(None)).receipt["policy"] == new
    # CONTRACT: queued input makes the session not idle.
    store.admissions.admit(epoch=epoch, principal_id="p", session_id=sid, request_id="q", payload={"text": "q"})
    with pytest.raises(AuthorityRefusal, match="^session_busy$"):
        store.runtime.commit_refreshed_policy(epoch, session_id=sid, old=new, new=old, generation=1)


def test_policy_picks_read_committed_non_deferred_receipts_and_the_receipt_picks(backend, scope_a):
    from persistence.authority.runtime import LocalSnapshot, PreparedMutation

    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-pick")
    sid = _local(store)
    assert store.runtime.policy_picks(sid) == set()
    _defer(store, epoch, "deferred", "model", {"model": "gpt"}, session_id=sid)
    assert store.runtime.policy_picks(sid) == set()
    snapshot = store.runtime.prepare_mutation(epoch, principal_id="p", session_id=sid, request_id="lvl", expected_revision=0, command=ReasoningPick("high"))
    assert isinstance(snapshot, LocalSnapshot)
    policy = {**snapshot.receipt["policy"], "reasoning": "high"}
    outcome = store.runtime.mutate(epoch, principal_id="p", session_id=sid, request_id="lvl", expected_revision=0, command=ReasoningPick("high"),
                                   prepared=PreparedMutation(snapshot, {"policy": policy}), owner=None)
    assert outcome.result["effort"] == "high" and outcome.result["execution_generation"] == 1
    assert store.runtime.policy_picks(sid) == {"reasoning"}
