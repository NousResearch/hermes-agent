"""Contract suite for the worker session family (``persistence.workers.sessions.WorkerSessionVerbs``).

Each verb is driven through ``worker_executions`` by a registered worker on its own session, and each write is read back through the owning capability's public reads (sessions, messages, usage), which proves it reached the tables every other reader uses. The receipt transaction itself is pinned in ``test_worker_receipt_contract.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from persistence import AuthorityEpoch, SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession, SessionLifecycleStatus, SessionLineageKind, TitleSource
from persistence.core.errors import CompressionClosed, TurnLeaseLost
from persistence.workers.sessions import RowAnnotation, WorkerActivity, WorkerAppendResult, WorkerBillingRoute, WorkerTitleResult
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts._reference.workers.receipt import WRITE_FAILPOINT
from tests.contracts.conftest import require_failpoints
from tests.contracts.workers._support import SESSION, assistant, begin_epoch, ident, last_sequence, refused, register, seed, user

OTHER = SessionId("sess-other")
STAMP = datetime(2026, 9, 25, 12, 0, tzinfo=UTC).timestamp()
REFUSED = "Title unavailable or invalid"


class _Worker:
    """A registered worker on :data:`SESSION` with its own sequence counter."""

    def __init__(self, backend, scope, *, session: NewSession | None = None) -> None:
        self.store = backend.open(scope)
        if session is not None:
            self.store.sessions.create(session)
            self.epoch = begin_epoch(backend, scope)
        else:
            self.epoch = seed(backend, scope)
        register(backend, scope, self.epoch)
        self.backend, self.scope, self._next = backend, scope, iter(range(1, 1000))

    def args(self, sequence: int | None = None) -> dict:
        return ident(self.epoch, sequence=next(self._next) if sequence is None else sequence)

    def __getattr__(self, verb: str):
        repository = self.store.worker_executions

        def call(*, sequence: int | None = None, **kwargs):
            return getattr(repository, verb)(**self.args(sequence), **kwargs)
        return call


def _msg(role: str, content, **extra) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=STAMP, **extra)


# ── payload checks (the handler's, after the receipt fences) ─────────────────


def test_session_payload_checks_refuse_malformed_values_before_the_sequence_advances(backend, scope_a):
    worker = _Worker(backend, scope_a)
    for activity in (WorkerActivity(True, None, None), WorkerActivity(float("nan"), None, None), WorkerActivity("1", None, None), WorkerActivity(None, 3, None)):
        with refused("invalid_params"):
            worker.touch_session_activity(sequence=1, activity=activity)
    with refused("invalid_params"):
        worker.set_session_billing_route(sequence=1, route=WorkerBillingRoute(provider=1, base_url=None, billing_mode=None))
    with refused("invalid_params"):
        worker.create_session(sequence=1, source="contract", model_config={"_reset_from": "x"})
    with refused("invalid_params"):
        worker.patch_session_sidecars(sequence=1, patch={"_branched_from": "x"})
    for pin in ("ab", 7, ["a" * 257], [""], {"tools": []}, {"version": 1, "tools": []}, {"version": "v", "tools": {}},
                {"version": "v", "tools": [], "extra": 1}, {"version": "v", "tools": [], "delegation_backend": "b" * 65},
                {"version": "v", "tools": ["x" * (4 * 1024 * 1024)]}):
        with refused("invalid_params"):
            worker.set_session_tool_pin(sequence=1, tool_pin=pin)
    with refused("invalid_params"):
        worker.set_message_api_content(sequence=1, row_id=0, content="x", api_content="y")
    with refused("invalid_params"):
        worker.message_role(sequence=1, row_id="1")
    assert last_sequence(backend, scope_a, worker.epoch) == 0
    worker.set_session_tool_pin(sequence=1, tool_pin=["a"] * 4097)
    worker.set_session_tool_pin(sequence=2, tool_pin={"version": "v", "tools": [{"name": "t"}], "delegation_backend": "local"})
    assert worker.store.sessions.get(SESSION).tool_pin == {"version": "v", "tools": [{"name": "t"}], "delegation_backend": "local"}


def test_the_epoch_fence_precedes_an_invalid_payload(backend, scope_a):
    worker = _Worker(backend, scope_a)
    with refused("stale_epoch"):
        worker.store.worker_executions.next_session_title(**(worker.args(1) | {"epoch": AuthorityEpoch(0)}), base_title=None)
    with refused("invalid_params"):
        worker.next_session_title(sequence=1, base_title=None)


def test_row_annotations_round_trip_the_wire():
    wire = {"_row_id": 4, "timestamp": 1.5, "_db_row_snapshot": "s4", "_canonical_row": {"content": "won", "_content_only": True},
            "message_uid": "u4", "_tool_call_uids": {"c": "t4"}, "_tool_call_uid": None}
    annotation = RowAnnotation.from_wire(wire)
    assert (annotation.row_id, annotation.db_row_snapshot, annotation.tool_call_uids) == (4, "s4", {"c": "t4"})
    assert annotation.as_wire() == wire
    assert RowAnnotation.from_wire({"_row_id": 6}).as_wire() == {key: (6 if key == "_row_id" else None) for key in wire}


# ── titles ───────────────────────────────────────────────────────────────────


def test_title_precedence_collision_and_replay(backend, scope_a):
    worker = _Worker(backend, scope_a)
    store = worker.store
    store.sessions.create(NewSession(session_id=OTHER, source="contract"))
    store.sessions.set_title(OTHER, "Taken")
    assert worker.set_session_title(title="  Mine\u200b ", source=TitleSource.USER) == WorkerTitleResult(value=True)
    assert (store.sessions.get(SESSION).title, store.sessions.get(SESSION).title_source) == ("Mine", TitleSource.USER)
    # An automatic title never replaces a user one: not applied, not refused.
    assert worker.set_session_title(title="Auto", source=TitleSource.LLM) == WorkerTitleResult(value=False)
    collision = worker.args()
    taken = store.worker_executions.set_session_title(**collision, title="Taken", source=TitleSource.USER)
    assert taken == WorkerTitleResult(value=False, error=REFUSED, title_candidate="Taken")
    # The refusal is a receipted result: the exact retry replays it, another payload conflicts.
    assert store.worker_executions.set_session_title(**collision, title="Taken", source=TitleSource.USER) == taken
    with refused("admission_conflict"):
        store.worker_executions.set_session_title(**collision, title="Other", source=TitleSource.USER)
    assert worker.set_session_title(title="x" * 101, source=TitleSource.USER) == WorkerTitleResult(value=False, error=REFUSED, title_candidate="x" * 101)
    assert store.sessions.get(SESSION).title == "Mine"
    assert worker.set_session_title(title="", source=TitleSource.USER) == WorkerTitleResult(value=True)
    assert (store.sessions.get(SESSION).title, store.sessions.get(SESSION).title_source) == (None, None)


def test_a_compression_ancestor_title_transfers_to_the_worker_session(backend, scope_a):
    store = backend.open(scope_a)
    parent = SessionId("sess-parent")
    store.sessions.create(NewSession(session_id=parent, source="contract"))
    store.sessions.set_title(parent, "Thread")
    store.sessions.end(parent, END_REASON_COMPRESSION)
    worker = _Worker(backend, scope_a, session=NewSession(session_id=SESSION, source="contract", parent_session_id=parent))
    assert worker.set_session_title(title="Thread", source=TitleSource.USER) == WorkerTitleResult(value=True)
    assert (store.sessions.get(parent).title, store.sessions.get(SESSION).title) == (None, "Thread")


def test_title_source_needs_a_title(backend, scope_a):
    worker = _Worker(backend, scope_a)
    assert worker.set_session_title_source(source=TitleSource.LLM) is False
    worker.store.sessions.set_title(SESSION, "Named")
    assert worker.set_session_title_source(source=TitleSource.LLM) is True
    assert worker.store.sessions.get(SESSION).title_source is TitleSource.LLM


def test_next_title_is_authorized_by_the_current_title_or_a_refused_candidate(backend, scope_a):
    worker = _Worker(backend, scope_a)
    store = worker.store
    store.sessions.create(NewSession(session_id=OTHER, source="contract"))
    store.sessions.set_title(OTHER, "Held")
    with refused("permission_denied"):
        worker.next_session_title(sequence=1, base_title="Held")
    # A refused set_session_title receipt of this running worker authorizes its candidate.
    worker.set_session_title(sequence=1, title="Held", source=TitleSource.USER)
    assert worker.next_session_title(sequence=2, base_title="Held") == "Held #2"
    store.sessions.create(NewSession(session_id=SessionId("sess-third"), source="contract"))
    store.sessions.set_title(SessionId("sess-third"), "MINE #4")
    assert worker.set_session_title(sequence=3, title="Mine", source=TitleSource.USER) == WorkerTitleResult(value=True)
    # Numbering reads every "<base> #n" title, ASCII case-insensitively on the prefix.
    assert worker.next_session_title(sequence=4, base_title="Mine") == "Mine #5"


# ── activity, billing, prompt, tools, sidecars ───────────────────────────────


def test_activity_moves_forward_only_and_clear_keeps_the_stamp(backend, scope_a):
    worker = _Worker(backend, scope_a)
    stamp = datetime(2026, 9, 25, 10, 0, tzinfo=UTC)
    worker.touch_session_activity(activity=WorkerActivity(ts=stamp.timestamp(), description="tool: x", provenance="tool"))
    worker.touch_session_activity(activity=WorkerActivity(ts=(stamp - timedelta(hours=1)).timestamp(), description="old", provenance="model"))
    activity = worker.store.sessions.get(SESSION)
    assert (activity.last_activity_at, activity.last_activity_description) == (stamp.timestamp(), "tool: x")
    worker.clear_session_activity()
    cleared = worker.store.sessions.get(SESSION)
    assert (cleared.last_activity_at, cleared.last_activity_description, cleared.last_activity_provenance) == (stamp.timestamp(), "", "unknown")


def test_billing_route_replaces_the_route_and_keeps_the_stored_prompt(backend, scope_a):
    # Route writers never touch the frozen prompt snapshot; only session.prompt replaces it.
    worker = _Worker(backend, scope_a)
    worker.set_session_system_prompt(system_prompt="cached")
    worker.set_session_billing_route(route=WorkerBillingRoute(provider="p", base_url="https://b", billing_mode="metered"))
    worker.set_session_billing_route(route=WorkerBillingRoute(provider="q", base_url=None, billing_mode=None))
    route = worker.store.sessions.get(SESSION)
    assert (route.billing_provider, route.billing_base_url, route.billing_mode) == ("q", None, "metered")
    assert route.system_prompt == "cached"


def test_prompt_tools_and_sidecars_land_on_the_session(backend, scope_a):
    worker = _Worker(backend, scope_a)
    store = worker.store
    worker.set_session_system_prompt(system_prompt="be brief")
    pin = {"version": "v1", "tools": [{"type": "function", "function": {"name": "terminal"}}]}
    worker.set_session_tool_pin(tool_pin=pin)
    worker.patch_session_sidecars(patch={"_usage_anchor": {"input": 3}, "_proactive_prune_rearm_tokens": 9, "_anthropic_rejected_thinking": ["f"]})
    worker.patch_session_sidecars(patch={"_proactive_prune_rearm_tokens": None})
    snapshot = store.sessions.get(SESSION)
    assert (snapshot.system_prompt, snapshot.tool_pin) == ("be brief", pin)
    assert store.sessions.get_model_config_value(SESSION, "_usage_anchor") == {"input": 3}
    assert store.sessions.get_model_config_value(SESSION, "_anthropic_rejected_thinking") == ["f"]
    assert store.sessions.get_model_config_value(SESSION, "_proactive_prune_rearm_tokens") is None
    worker.set_session_tool_pin(tool_pin=None)
    worker.set_session_system_prompt(system_prompt=None)
    assert (store.sessions.get(SESSION).tool_pin, store.sessions.get(SESSION).system_prompt) == (None, None)


# ── create, end, lifecycle, api content, roles ───────────────────────────────


def test_create_fills_unset_facets_and_refuses_foreign_identity(backend, scope_a):
    worker = _Worker(backend, scope_a, session=NewSession(session_id=SESSION, source="contract", session_key="peer"))
    with refused("permission_denied"):
        worker.create_session(sequence=1, source="cli")
    with refused("permission_denied"):
        worker.create_session(sequence=1, source="contract", session_key="other")
    with refused("invalid_params"):
        worker.create_session(sequence=1, model="m1")
    assert worker.create_session(sequence=1, source="contract", session_key="peer", model="m1", model_config={"max_tokens": 5}) == SESSION
    worker.create_session(sequence=2, source="contract", model="m2")
    record = worker.store.sessions.get(SESSION)
    assert (record.model, record.session_key, dict(record.model_config).get("max_tokens")) == ("m1", "peer", 5)


def test_create_merges_constructor_keys_beside_a_reserved_child_marker(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SessionId("sess-parent"), source="contract"))
    worker = _Worker(backend, scope_a, session=NewSession(session_id=SESSION, source="subagent", parent_session_id=SessionId("sess-parent"),
                                                          lineage_kind=SessionLineageKind.DELEGATE))
    worker.create_session(source="subagent", model_config={"max_iterations": 7})
    config = dict(store.sessions.get(SESSION).model_config)
    assert (config.get("_delegate_from"), config.get("max_iterations")) == ("sess-parent", 7)


def test_create_compares_the_origin_as_stored(backend, scope_a):
    origin = {"platform": "telegram", "chat_id": "7"}
    worker = _Worker(backend, scope_a, session=NewSession(session_id=SESSION, source="contract", origin=origin))
    with refused("permission_denied"):
        worker.create_session(sequence=1, source="contract", origin_json=json.dumps(origin, sort_keys=True, indent=1))
    assert worker.create_session(sequence=1, source="contract", origin_json=json.dumps(origin)) == SESSION


def test_end_keeps_the_first_reason_and_lifecycle_reads_the_last_row(backend, scope_a):
    worker = _Worker(backend, scope_a)
    store = worker.store
    assert worker.session_lifecycle() is SessionLifecycleStatus.EMPTY
    store.messages.append(SESSION, _msg("assistant", "", tool_calls=({"id": "c"},)))
    assert worker.session_lifecycle() is SessionLifecycleStatus.INTERRUPTED
    store.messages.append(SESSION, _msg("assistant", "done"))
    assert worker.session_lifecycle() is SessionLifecycleStatus.COMPLETE
    worker.end_session(end_reason="user_exit")
    worker.end_session(end_reason="other")
    assert store.sessions.get(SESSION).end_reason == "user_exit"
    with refused("invalid_params"):
        worker.end_session(end_reason="")


def test_api_content_lands_on_the_latest_matching_user_row_only(backend, scope_a):
    worker = _Worker(backend, scope_a)
    store = worker.store
    store.messages.append(SESSION, _msg("user", "first"))
    store.messages.append(SESSION, _msg("user", "second"))
    assert worker.set_latest_user_api_content(content="first", api_content="sent-1") == 0
    assert worker.set_latest_user_api_content(content="second", api_content="sent-2") == 1
    assert [r.api_content for r in store.messages.read_transcript(SESSION)] == [None, "sent-2"]
    # Structured content matches on the stored encoding after the payload's sorted-key round trip.
    store.messages.append(SESSION, _msg("user", [{"z": 1, "a": 2}]))
    assert worker.set_latest_user_api_content(content=[{"z": 1, "a": 2}], api_content="x") == 0
    store.messages.append(SESSION, _msg("user", [{"a": 2, "z": 1}]))
    assert worker.set_latest_user_api_content(content=[{"z": 1, "a": 2}], api_content="y") == 1


def test_row_addressed_user_edits_touch_only_an_active_user_row_of_this_session(backend, scope_a):
    worker = _Worker(backend, scope_a)
    store = worker.store
    store.sessions.create(NewSession(session_id=OTHER, source="contract"))
    first = store.messages.append(SESSION, _msg("user", "first"))
    reply = store.messages.append(SESSION, _msg("assistant", "reply"))
    foreign = store.messages.append(OTHER, _msg("user", "theirs"))
    assert worker.set_message_api_content(row_id=first, content="first", api_content="sent") == 1
    assert worker.set_message_api_content(row_id=first, content="changed", api_content="sent") == 0
    assert worker.set_message_api_content(row_id=reply, content="reply", api_content="sent") == 0
    assert worker.set_message_api_content(row_id=foreign, content="theirs", api_content="sent") == 0
    assert worker.set_user_message_content(row_id=first, content="edited") == 1
    assert worker.set_user_message_content(row_id=reply, content="edited") == 0
    assert worker.set_user_message_content(row_id=foreign, content="edited") == 0
    assert [(r.content, r.api_content) for r in store.messages.read_transcript(SESSION)] == [("edited", "sent"), ("reply", None)]
    assert [r.content for r in store.messages.read_transcript(OTHER)] == ["theirs"]


def test_a_user_rewrite_with_display_metadata_replaces_it_and_clears_the_sent_copy(backend, scope_a):
    worker = _Worker(backend, scope_a)
    store = worker.store
    store.sessions.create(NewSession(session_id=OTHER, source="contract"))
    typed = store.messages.append(SESSION, _msg("user", "typed", api_content="sent", display_metadata={"old": 1}))
    plain = store.messages.append(SESSION, _msg("user", "plain", api_content="sent", display_metadata={"keep": 1}))
    foreign = store.messages.append(OTHER, _msg("user", "theirs", api_content="sent", display_metadata={"keep": 1}))
    withheld = {"withheld_prompt": "typed"}
    assert worker.set_user_message_content(row_id=typed, content="[withheld]", display_metadata=withheld) == 1
    assert worker.set_user_message_content(row_id=plain, content="edited") == 1
    assert worker.set_user_message_content(row_id=foreign, content="[withheld]", display_metadata=withheld) == 0
    assert [(r.content, r.api_content, r.display_metadata) for r in store.messages.read_transcript(SESSION)] == [
        ("[withheld]", None, withheld), ("edited", "sent", {"keep": 1})]
    assert [(r.content, r.api_content, r.display_metadata) for r in store.messages.read_transcript(OTHER)] == [("theirs", "sent", {"keep": 1})]


def test_a_user_rewrite_as_sent_takes_row_id_content_and_an_optional_display_metadata(backend, scope_a):
    worker = _Worker(backend, scope_a)
    row = worker.store.messages.append(SESSION, _msg("user", "typed", api_content="sent"))
    for payload in ({"row_id": row, "content": "x", "display_metadata": {"a": 1}, "api_content": "y"},
                    {"row_id": row, "display_metadata": {"a": 1}}, {"content": "x", "display_metadata": {"a": 1}},
                    {"row_id": 0, "content": "x", "display_metadata": {"a": 1}}, {"row_id": str(row), "content": "x", "display_metadata": {"a": 1}}):
        with refused("invalid_params"):
            worker.persist_operation(sequence=1, operation="session.user_content", payload=payload)
    assert last_sequence(backend, scope_a, worker.epoch) == 0
    result = worker.persist_operation(sequence=1, operation="session.user_content", payload={"row_id": row, "content": "x", "display_metadata": {"a": 1}})
    assert result["value"] == 1
    assert [(r.content, r.api_content, r.display_metadata) for r in worker.store.messages.read_transcript(SESSION)] == [("x", None, {"a": 1})]


def test_tail_and_row_roles_read_active_rows_of_this_session(backend, scope_a):
    worker = _Worker(backend, scope_a)
    store = worker.store
    store.sessions.create(NewSession(session_id=OTHER, source="contract"))
    assert worker.latest_conversation_role(sequence=1) is None
    first = store.messages.append(SESSION, _msg("user", "q"))
    store.messages.append(SESSION, _msg("system", "note"))
    foreign = store.messages.append(OTHER, _msg("assistant", "theirs"))
    assert worker.latest_conversation_role(sequence=2) == "user"
    roles = (worker.message_role(sequence=3, row_id=first), worker.message_role(sequence=4, row_id=foreign), worker.message_role(sequence=5, row_id=999))
    assert roles == ("user", None, None)
    # A read-only receipt replays the answer recorded then.
    store.messages.append(SESSION, _msg("assistant", "a"))
    assert worker.latest_conversation_role(sequence=2) == "user"
    assert worker.latest_conversation_role(sequence=6) == "assistant"


# ── transcript batch ─────────────────────────────────────────────────────────


def test_transcript_batch_appends_once_and_repairs_or_adopts(backend, scope_a):
    worker = _Worker(backend, scope_a)
    store = worker.store
    first = worker.args()
    batch = [user("q", timestamp=STAMP), assistant("", timestamp=STAMP)]
    result = store.worker_executions.append_transcript(**first, messages=batch)
    assert isinstance(result, WorkerAppendResult) and result.count == 2
    assert store.worker_executions.append_transcript(**first, messages=batch) == result
    # The worker takes the row ids, ownership versions and durable uids the store stamped.
    stored = store.messages.read_transcript(SESSION)
    assert [a.row_id for a in result.annotations] == [r.id for r in stored]
    assert [a.message_uid for a in result.annotations] == [r.message_uid for r in stored] and None not in [r.message_uid for r in stored]
    blank = result.annotations[1].row_id
    repaired = worker.append_transcript(messages=[assistant("a", _row_id=blank, timestamp=STAMP)])
    assert (repaired.count, repaired.annotations[0].row_id, repaired.annotations[0].canonical_row) == (0, blank, None)
    adopted = worker.append_transcript(messages=[assistant("b", _row_id=blank, timestamp=STAMP)])
    assert (adopted.count, adopted.annotations[0].row_id, adopted.annotations[0].canonical_row) == (0, blank, {"content": "a", "_content_only": True})
    assert adopted.annotations[0].db_row_snapshot == repaired.annotations[0].db_row_snapshot != result.annotations[1].db_row_snapshot
    assert [r.content for r in store.messages.read_transcript(SESSION)] == ["q", "a"]


def test_transcript_rows_carry_the_worker_wire_fields(backend, scope_a):
    worker = _Worker(backend, scope_a)
    row = {"role": "tool", "content": "out", "tool_call_id": "call-1", "tool_name": "terminal", "timestamp": STAMP, "display_kind": "tool",
           "_compressed_summary": False, "finish_reason": "stop", "data_use": ["d"], "_merged_turn_prefix": "p", "_absorbed_row_ids": [1]}
    assert worker.append_transcript(messages=[row]).count == 1
    stored = worker.store.messages.read_transcript(SESSION)[0]
    assert (stored.role, stored.tool_call_id, stored.tool_name, stored.timestamp) == ("tool", "call-1", "terminal", STAMP)
    with refused("invalid_params"):
        worker.append_transcript(messages=[dict(row, unknown=1)])
    with refused("invalid_params"):
        worker.append_transcript(messages=[dict(row, _compressed_summary_has_user_turn=True)])
    with refused("invalid_params"):
        worker.append_transcript(messages=[user("x")] * 1001)


def test_transcript_refusals_commit_nothing(backend, scope_a):
    worker = _Worker(backend, scope_a)
    store = worker.store
    store.sessions.create(NewSession(session_id=OTHER, source="contract"))
    foreign = store.messages.append(OTHER, _msg("assistant", "theirs"))
    with refused("permission_denied"):
        worker.append_transcript(sequence=1, messages=[assistant("x", _row_id=foreign)])
    assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="live", ttl_seconds=300.0) is True
    with pytest.raises(TurnLeaseLost):
        worker.append_transcript(sequence=1, messages=[user("late")], turn_lease_holder="gone")
    with pytest.raises(TurnLeaseLost):
        worker.append_transcript(sequence=1, messages=[], turn_lease_holder="gone")
    assert store.messages.read_transcript(SESSION) == ()
    assert worker.append_transcript(sequence=1, messages=[user("ok")], turn_lease_holder="live").count == 1
    store.sessions.end(SESSION, END_REASON_COMPRESSION)
    with pytest.raises(CompressionClosed):
        worker.append_transcript(sequence=2, messages=[user("closed")])
    with pytest.raises(CompressionClosed):
        worker.append_transcript(sequence=2, messages=[])
    assert [r.content for r in store.messages.read_transcript(SESSION)] == ["ok"]
    assert last_sequence(backend, scope_a, worker.epoch) == 1


def test_transcript_rows_keep_the_stored_content_encoding(backend, scope_a):
    worker = _Worker(backend, scope_a)
    worker.append_transcript(messages=[assistant("before\ud800after"), assistant([{"type": "text", "text": "x"}])])
    assert [r.content for r in worker.store.messages.read_transcript(SESSION)] == ["before\ufffdafter", [{"type": "text", "text": "x"}]]


def test_a_crash_between_the_session_write_and_its_receipt_publishes_nothing(backend, scope_a):
    require_failpoints(backend)
    worker = _Worker(backend, scope_a)
    authority_state(backend, scope_a).failpoints.arm(WRITE_FAILPOINT)
    with pytest.raises(InjectedFailure):
        worker.set_session_title(sequence=1, title="Half", source=TitleSource.USER)
    assert worker.store.sessions.get(SESSION).title is None
    assert worker.set_session_title(sequence=1, title="Half", source=TitleSource.USER) == WorkerTitleResult(value=True)
