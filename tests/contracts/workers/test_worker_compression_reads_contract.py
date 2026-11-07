"""Contract suite for the worker's receipted reads (``persistence.workers.compression_reads.WorkerCompressionReads``).

Read results are compared with the non-worker verbs that own the state (sessions, prompt snapshot, billing route, cooldown, lineage counters and reads, ``read_conversation``) for the same target; authorization, fence and replay cases check refusals and recorded results.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from persistence import AuthorityEpoch, SessionId
from persistence.conversation.messages import ConversationRequest, NewMessage, TranscriptView
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession, SessionLineageKind
from persistence.workers.compression_reads import WorkerLineageView
from persistence.workers.executions import WorkerExecutionId
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts._reference.workers.receipt import WRITE_FAILPOINT
from tests.contracts.conftest import require_capability, require_failpoints
from tests.contracts.workers._support import SESSION, begin_epoch, ident, last_sequence, refused, register, require_workers

ROOT = SessionId("sess-root")
CHILD = SessionId("sess-worker-cont")
FOREIGN = SessionId("sess-foreign")
T1 = 2_000_000_000.0
FLAGS = ("include_ancestors", "include_inactive", "repair_alternation", "include_row_ids", "include_compacted")


def _reader(backend, scope):
    """A worker on SESSION, a delegate of ROOT, with an unrelated FOREIGN session beside them."""
    require_capability(backend, "transactional_claims")
    require_workers(backend)
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=ROOT, source="contract"))
    store.sessions.create(NewSession(session_id=SESSION, source=" contract ", parent_session_id=ROOT, lineage_kind=SessionLineageKind.DELEGATE))
    store.sessions.create(NewSession(session_id=FOREIGN, source="contract"))
    epoch = begin_epoch(backend, scope)
    register(backend, scope, epoch)
    return store, epoch


def _continue(store):
    """Compress SESSION into CHILD: the one forward compression chain from the worker's session."""
    store.sessions.end(SESSION, END_REASON_COMPRESSION)
    store.sessions.create(NewSession(session_id=CHILD, source="contract", parent_session_id=SESSION))


def _call(store, epoch, sequence, verb, **kwargs):
    return getattr(store.worker_executions, verb)(**ident(epoch, sequence=sequence), **kwargs)


def _msg(role, content):
    return NewMessage(role=role, content=content, timestamp=datetime.now(UTC).timestamp())


# ── compression_session_context ──────────────────────────────────────────────


@pytest.mark.parametrize("until", [1790639758.6100938, 1790639758.6100934])
def test_the_context_carries_epoch_times_exactly(backend, scope_a, until):
    store, epoch = _reader(backend, scope_a)
    store.messages.record_compression_cooldown(SESSION, until, "precise")
    context = _call(store, epoch, 1, "compression_session_context", target_session_id=SESSION)
    assert (context["compression_failure_cooldown_until"], context["compression_failure_error"]) == (until, "precise")
    assert context["started_at"] == store.sessions.get(SESSION).started_at
    store.messages.clear_compression_cooldown(SESSION)
    assert _call(store, epoch, 2, "compression_session_context", target_session_id=SESSION)["compression_failure_cooldown_until"] is None


def test_session_context_is_the_session_row_and_replays_the_recorded_row(backend, scope_a):
    store, epoch = _reader(backend, scope_a)
    sessions, lineage = store.sessions, store.lineage
    pin = {"version": "v1", "tools": [{"name": "t1"}, {"name": "t2"}]}
    store.usage.set_billing_route(SESSION, provider="prov", base_url="https://b", billing_mode="m")
    sessions.update_system_prompt(SESSION, "prefix\n")
    sessions.update_tool_pin(SESSION, pin)
    sessions.set_title(SESSION, "worker title")
    sessions.patch_model_config(SESSION, {"k": "v"})
    store.messages.record_compression_cooldown(SESSION, T1, "boom")
    lineage.set_compression_fallback_streak(SESSION, 2)
    lineage.set_compression_ineffective_count(SESSION, 3)
    lineage.set_compression_recovery_deadline(SESSION, T1 + 100)
    context = _call(store, epoch, 1, "compression_session_context", target_session_id=SESSION)
    record = sessions.get(SESSION)
    assert (context["id"], context["source"], context["title"], context["parent_session_id"]) == (SESSION, record.source, "worker title", ROOT)
    assert (context["system_prompt"], json.loads(context["tool_names"])) == ("prefix\n", pin)
    assert (context["billing_provider"], context["billing_base_url"], context["billing_mode"]) == ("prov", "https://b", "m")
    assert json.loads(context["model_config"]) == {"_delegate_from": ROOT, "k": "v"}
    assert (context["compression_failure_cooldown_until"], context["compression_failure_error"]) == (T1, "boom")
    assert (context["compression_fallback_streak"], context["compression_ineffective_count"], context["compression_recovery_deadline"]) == (2, 3, T1 + 100)
    # An ancestor is readable too; its row is its own.
    assert _call(store, epoch, 2, "compression_session_context", target_session_id=ROOT)["id"] == ROOT
    # An exact duplicate replays the row recorded at the time, not a fresh read.
    sessions.set_title(SESSION, "renamed")
    assert _call(store, epoch, 1, "compression_session_context", target_session_id=SESSION) == context
    with refused("admission_conflict"):
        _call(store, epoch, 1, "compression_session_context", target_session_id=ROOT)
    assert last_sequence(backend, scope_a, epoch) == 2


def test_reads_authorize_ancestors_and_the_compression_chain_only(backend, scope_a):
    store, epoch = _reader(backend, scope_a)
    _continue(store)
    for verb in ("compression_session_context", "compression_lineage_view"):
        with refused("permission_denied"):
            _call(store, epoch, 1, verb, target_session_id=FOREIGN)
    with refused("permission_denied"):
        _call(store, epoch, 1, "compression_history", target_session_id=FOREIGN, **dict.fromkeys(FLAGS, False))
    # Refusals record nothing: sequence 1 is still free for a readable target.
    assert _call(store, epoch, 1, "compression_session_context", target_session_id=CHILD)["id"] == CHILD
    assert last_sequence(backend, scope_a, epoch) == 1


# ── compression_lineage_view ──────────────────────────────────────────────────


def test_lineage_view_equals_the_lineage_reads_for_the_target(backend, scope_a):
    store, epoch = _reader(backend, scope_a)
    _continue(store)
    lineage = store.lineage
    for sequence, target in enumerate((SESSION, ROOT, CHILD), start=1):
        view = _call(store, epoch, sequence, "compression_lineage_view", target_session_id=target)
        assert set(view.readable_ids) == {ROOT, SESSION, CHILD}
        assert view == WorkerLineageView(readable_ids=view.readable_ids, lineage=lineage.lineage(target), root=lineage.conversation_root(target),
                                         tip=lineage.tip(target), resume=lineage.resolve_resume_target(target),
                                         identity=(lineage.is_fork_child(target), store.sessions.get(target).source.strip()))
    view = _call(store, epoch, 1, "compression_lineage_view", target_session_id=SESSION)
    assert (view.identity, view.tip) == ((True, "contract"), CHILD)


def test_lineage_view_replays_the_recorded_view(backend, scope_a):
    store, epoch = _reader(backend, scope_a)
    before = _call(store, epoch, 1, "compression_lineage_view", target_session_id=SESSION)
    assert before.tip == SESSION
    _continue(store)
    assert _call(store, epoch, 1, "compression_lineage_view", target_session_id=SESSION) == before
    assert _call(store, epoch, 2, "compression_lineage_view", target_session_id=SESSION).tip == CHILD


# ── compression_history ───────────────────────────────────────────────────────


_REQUESTS = (
    dict.fromkeys(FLAGS, False),
    dict(dict.fromkeys(FLAGS, False), include_inactive=True, include_row_ids=True),
    dict(dict.fromkeys(FLAGS, False), include_compacted=True, include_ancestors=True),
    dict(dict.fromkeys(FLAGS, False), repair_alternation=True, include_ancestors=True, include_row_ids=True),
)


def _expected(store, target, flags):
    view = TranscriptView.ALL if flags["include_inactive"] else TranscriptView.DISPLAY if flags["include_compacted"] else TranscriptView.ACTIVE
    request = ConversationRequest(view=view, include_lineage=flags["include_ancestors"], repair_alternation=flags["repair_alternation"],
                                  include_row_ids=flags["include_row_ids"])
    return tuple(m.as_openai_message() for m in store.messages.read_conversation(target, request))


def test_history_equals_read_conversation_and_replays_the_recorded_messages(backend, scope_a):
    store, epoch = _reader(backend, scope_a)
    store.messages.append(SESSION, _msg("user", "hello"))
    store.messages.append(SESSION, _msg("assistant", "hi"))
    _continue(store)
    store.messages.append(CHILD, _msg("user", "again"))
    sequence = 0
    for target in (SESSION, CHILD):
        for flags in _REQUESTS:
            sequence += 1
            assert _call(store, epoch, sequence, "compression_history", target_session_id=target, **flags) == _expected(store, target, flags)
    assert [m["content"] for m in _expected(store, SESSION, _REQUESTS[0])] == ["hello", "hi"]
    # Sequence 5 read CHILD's active transcript; a later append does not reach its replay.
    recorded = _expected(store, CHILD, _REQUESTS[0])
    store.messages.append(CHILD, _msg("assistant", "late"))
    assert _call(store, epoch, 5, "compression_history", target_session_id=CHILD, **_REQUESTS[0]) == recorded
    with refused("admission_conflict"):
        _call(store, epoch, 1, "compression_history", target_session_id=SESSION, **_REQUESTS[1])
    with refused("invalid_params"):
        _call(store, epoch, 9, "compression_history", target_session_id=SESSION, **dict(_REQUESTS[0], include_row_ids=1))


# ── fences and crash window ───────────────────────────────────────────────────


def test_reads_fence_and_validate_before_reading(backend, scope_a):
    store, epoch = _reader(backend, scope_a)
    with refused("stale_epoch"):
        store.worker_executions.compression_session_context(**ident(AuthorityEpoch(epoch + 1)), target_session_id=SESSION)
    with refused("not_found"):
        store.worker_executions.compression_lineage_view(**ident(epoch, execution_id=WorkerExecutionId("ghost")), target_session_id=SESSION)
    with refused("invalid_params"):  # not the next sequence
        _call(store, epoch, 2, "compression_history", target_session_id=SESSION, **_REQUESTS[0])
    with refused("invalid_params"):
        _call(store, epoch, 1, "compression_session_context", target_session_id=SessionId(""))
    _call(store, epoch, 1, "finish")
    with refused("stale_generation"):
        _call(store, epoch, 2, "compression_lineage_view", target_session_id=SESSION)


def test_reads_leave_the_revision_alone(backend, scope_a):
    store, epoch = _reader(backend, scope_a)
    before = store.sessions.get(SESSION).runtime_revision
    _call(store, epoch, 1, "compression_session_context", target_session_id=SESSION)
    _call(store, epoch, 2, "compression_lineage_view", target_session_id=SESSION)
    _call(store, epoch, 3, "compression_history", target_session_id=SESSION, **_REQUESTS[0])
    assert store.sessions.get(SESSION).runtime_revision == before


def test_a_read_that_fails_before_its_receipt_leaves_the_sequence_free(backend, scope_a):
    require_failpoints(backend)
    store, epoch = _reader(backend, scope_a)
    authority_state(backend, scope_a).failpoints.arm(WRITE_FAILPOINT)
    with pytest.raises(InjectedFailure):
        _call(store, epoch, 1, "compression_session_context", target_session_id=SESSION)
    store.sessions.set_title(SESSION, "after the crash")
    # The retry is a fresh read, not a replay of the aborted one.
    assert _call(store, epoch, 1, "compression_session_context", target_session_id=SESSION)["title"] == "after the crash"
