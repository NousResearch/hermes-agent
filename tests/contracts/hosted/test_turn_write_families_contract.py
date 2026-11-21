"""The claimed-turn fence on every write family a turn makes: transcript appends (single and batched), transcript
replacement, an in-place row rewrite (the prune rewrite), the session updates, usage accounting, the compression
continuation and a delegation's completion. Under the turn's current fence each lands; once the session's next claim
or another owner's takeover superseded the turn, each is refused with ``TurnFenceSuperseded`` and leaves the store
exactly as it was: no row, counter, lock move or ledger change of a partly applied write. Requires authority_fences
(``require_capability`` in the claim helper).

The observation is the whole conversation surface the families touch, read outside the turn: a write that landed
any of its parts before the refusal changes it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable, NamedTuple

import pytest

from persistence import DelegationId, SessionId, TurnFenceSuperseded
from persistence.authority.coordination import OwnerStamp
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import HeldMessage, NewMessage, PrunedRowChange, ReplaceMode, TranscriptRead, TranscriptView
from persistence.conversation.sessions import SessionQuery, SessionWindow
from persistence.conversation.usage import TokenCounts
from tests.contracts.hosted.test_turn_fence_contract import SESSION, _claimed, _elsewhere, _next_claim, _take_over, _turn

CALL = {"id": "call-1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
CHILD = SessionId("turn-family-child")
DELEGATION = DelegationId("turn-family-d1")


def _msg(content: str, i: int, role: str = "user", **kwargs: Any) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=datetime(2026, 9, 11, 10, 0, i, tzinfo=UTC).timestamp(), **kwargs)


class _Family(NamedTuple):
    """One write family: ``prepare`` runs inside the turn while its fence holds (a lock the write needs), ``write`` is
    the family's write; both take the store and the seeded tool-result row id."""

    prepare: Callable[[Any, Any], None]
    write: Callable[[Any, Any], Any]


def _nothing(store: Any, result: Any) -> None:
    return None


def _pruned(result: Any) -> PrunedRowChange:
    original = HeldMessage(role="tool", content="long output", tool_call_id="call-1", row_id=result, persisted=True)
    return PrunedRowChange(original, _msg("[pruned]", 2, "tool", tool_call_id="call-1", tool_name="terminal"),
                           _msg("long output", 2, "tool", tool_call_id="call-1", tool_name="terminal"))


def _lock_compression(store: Any, result: Any) -> None:
    assert store.coordination.try_acquire_compression_lock(session_id=SESSION, holder="compressor", ttl_seconds=60.0)


def _publish_child(store: Any, result: Any) -> Any:
    return store.lineage.publish_compression_child(
        SESSION, CompressionChild(session_id=CHILD, source="contract"), [_msg("summary", 9), _msg("tail", 10, "assistant")],
        compression_lock_holder="compressor", require_compression_lease=True, require_lease_refresh=True)


def _complete_delegation(store: Any, result: Any) -> None:
    store.coordination.record_delegation_completion(
        delegation_id=DELEGATION, state="completed", completed_at=1_800_000_001.0,
        event={"delegation_id": DELEGATION, "status": "completed"}, result={"status": "completed"})


_FAMILIES: dict[str, _Family] = {
    "append": _Family(_nothing, lambda store, result: store.messages.append(SESSION, _msg("late reply", 5, "assistant"))),
    "append_batch": _Family(_nothing, lambda store, result: store.messages.append_batch(
        SESSION, [_msg("late one", 5, "assistant"), _msg("late two", 6), _msg("late three", 7, "assistant")])),
    "replace_transcript": _Family(_nothing, lambda store, result: store.messages.replace_transcript(
        SESSION, [_msg("replaced one", 5), _msg("replaced two", 6, "assistant")], ReplaceMode.DESTRUCTIVE)),
    "rewrite_pruned_rows": _Family(_nothing, lambda store, result: store.messages.rewrite_pruned_rows(
        SESSION, [_pruned(result)], model_config_patch={"_prune_anchor": 3})),
    "set_title": _Family(_nothing, lambda store, result: store.sessions.set_title(SESSION, "late title")),
    "update_cwd": _Family(_nothing, lambda store, result: store.sessions.update_cwd(SESSION, "/late")),
    "patch_model_config": _Family(_nothing, lambda store, result: store.sessions.patch_model_config(SESSION, {"late": True})),
    "end": _Family(_nothing, lambda store, result: store.sessions.end(SESSION, "cli_close")),
    "auxiliary_usage": _Family(_nothing, lambda store, result: store.usage.record_auxiliary_usage(
        SESSION, "vision", model="model-a", tokens=TokenCounts(input_tokens=11, output_tokens=7), estimated_cost_usd=0.5)),
    "billing_route": _Family(_nothing, lambda store, result: store.usage.set_billing_route(
        SESSION, provider="late-provider", base_url="https://late.example")),
    "compression_continuation": _Family(_lock_compression, _publish_child),
    "delegation_completion": _Family(_nothing, _complete_delegation),
}


def _seeded(backend, scope):
    """A claimed session holding a tool turn and a dispatched delegation; returns the store, epoch, claim and the
    tool-result row id."""
    store, epoch, claimed = _claimed(backend, scope)
    store.messages.append(SESSION, _msg("run it", 0))
    store.messages.append(SESSION, _msg("", 1, "assistant", tool_calls=(CALL,)))
    result = store.messages.append(SESSION, _msg("long output", 2, "tool", tool_call_id="call-1", tool_name="terminal"))
    store.coordination.record_delegation_dispatch(
        delegation_id=DELEGATION, origin_session="route-1", origin_ui_session_id="ui-1", parent_session_id=SESSION,
        dispatched_at=1_800_000_000.0, owner=OwnerStamp(pid=4242, started_at=777), task={"goal": "g"})
    return store, epoch, claimed, result


def _observe(store: Any) -> tuple[Any, ...]:
    """Everything the families write, read as another reader would: outside the turn."""
    def read() -> tuple[Any, ...]:
        everything = TranscriptRead(view=TranscriptView.ALL)
        listed = store.sessions.list_sessions(SessionQuery(include_children=True), SessionWindow(limit=100))
        delegation = store.coordination.get_delegation(delegation_id=DELEGATION)
        return (store.sessions.get(SESSION), store.sessions.get(CHILD),
                tuple((r.id, r.role, r.content, r.visibility) for r in store.messages.read_transcript(SESSION, everything)),
                tuple((r.id, r.content) for r in store.messages.read_transcript(CHILD, everything)),
                store.messages.count_all(), sorted(str(entry.session.session_id) for entry in listed),
                store.lineage.tip(SESSION), store.coordination.get_compression_lock_holder(session_id=SESSION),
                dict(store.usage.auxiliary_usage_by_task(SESSION)), store.usage.recent_route(SESSION),
                None if delegation is None else (delegation.state, delegation.completed_at))
    return _elsewhere(read)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_each_turn_write_family_lands_under_the_current_fence(backend_name: str, backend, scope_a, family: str) -> None:
    """While the turn holds its lease, each family's write lands as it would outside a turn."""
    store, _epoch, claimed, result = _seeded(backend, scope_a)
    prepare, write = _FAMILIES[family]
    with _turn(scope_a, claimed):
        prepare(store, result)
        before = _observe(store)
        write(store, result)
    assert _observe(store) != before


@pytest.mark.parametrize("supersession", ["next_claim", "takeover"])
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_each_turn_write_family_is_refused_once_superseded_and_leaves_no_partial_state(
        backend_name: str, backend, scope_a, family: str, supersession: str) -> None:
    """After the session's next claim, or another owner's takeover, each family's write of the superseded turn
    raises TurnFenceSuperseded and the store reads exactly as before the attempt: none of the write's rows, counters,
    session fields, lock moves or ledger changes landed."""
    store, epoch, claimed, result = _seeded(backend, scope_a)
    prepare, write = _FAMILIES[family]
    with _turn(scope_a, claimed):
        prepare(store, result)
        if supersession == "next_claim":
            _next_claim(store, epoch, claimed)
        else:
            _take_over(store)
        before = _observe(store)
        with pytest.raises(TurnFenceSuperseded) as refused:
            write(store, result)
    assert refused.value.session_id == SESSION
    assert _observe(store) == before
