"""Contract cases for the listing row (``persistence.conversation.sessions.SessionRepository.list_sessions`` / ``persistence.conversation.sessions.SessionRepository.summary``, record ``SessionListEntry``).

Every case seeds through public protocol operations only (sessions, messages, usage) and observes through the listing row. The rules are the ``persistence/conversation/sessions.py`` docstrings: the row's record carries the counters ``transcript_totals`` reports, the accounting ``read_session_usage`` reports and the prompt snapshot, ``last_active`` is the listing recency key, and ``summary`` is the unprojected listing row of one id.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, SessionListEntry, SessionOrder, SessionQuery, SessionWindow
from persistence.conversation.usage import TokenCounts, UsageDelta
from tests.contracts.conversation.test_usage import accounting

T0 = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)


def _new(session_id: str, source: str = "cli", **kwargs) -> NewSession:
    return NewSession(session_id=SessionId(session_id), source=source, **kwargs)


def _msg(content: str, i: int, role: str = "user", tool_calls: tuple = ()) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=(T0 + timedelta(seconds=i)).timestamp(), tool_calls=tool_calls)


def _at(i: int) -> float:
    return (T0 + timedelta(seconds=i)).timestamp()


def _seed_rich(backend, scope, sid: str):
    """One session with a transcript (a tool-calling assistant turn), recorded usage, a system prompt snapshot and a model config."""
    store = backend.open(scope)
    store.sessions.create(_new(sid, model_config={"temperature": 0.2}))
    messages = backend.open(scope).messages
    messages.append(SessionId(sid), _msg("hello there", 1))
    messages.append(
        SessionId(sid),
        _msg("", 2, role="assistant", tool_calls=({"id": "c1", "type": "function"}, {"id": "c2", "type": "function"})),
    )
    messages.append(SessionId(sid), _msg("done", 3, role="assistant"))
    store.usage.queue_usage(
        SessionId(sid),
        UsageDelta(
            tokens=TokenCounts(input_tokens=120, output_tokens=30, cache_read_tokens=7, cache_write_tokens=5, reasoning_tokens=11),
            api_call_count=2, estimated_cost_usd=0.25, actual_cost_usd=0.2,
        ),
    )
    store.sessions.update_system_prompt(SessionId(sid), "You are the pin.")


def _listed(backend, scope, query: SessionQuery = SessionQuery(), limit: int = 100) -> list[SessionListEntry]:
    store = backend.open(scope)
    return list(store.sessions.list_sessions(query, SessionWindow(limit=limit)))


def test_empty_store_shape(backend, scope_a):
    store_a = backend.open(scope_a)
    assert store_a.sessions.list_sessions(SessionQuery(), SessionWindow(limit=10)) == ()
    assert store_a.sessions.summary(SessionId("nobody")) is None


def test_summary_borrows_every_field_from_its_owning_family(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_rich(backend, scope_a, "rich-1")
    sid = SessionId("rich-1")
    got = store_a.sessions.summary(sid)
    assert got is not None
    assert got.session == store_a.sessions.get(sid)
    row = got.session
    assert (row.message_count, row.tool_call_count) == (3, 2)
    usage = accounting(store_a, sid)
    assert usage is not None
    assert (row.input_tokens, row.output_tokens, row.cache_read_tokens, row.cache_write_tokens, row.reasoning_tokens) == (
        usage.tokens.input_tokens, usage.tokens.output_tokens, usage.tokens.cache_read_tokens,
        usage.tokens.cache_write_tokens, usage.tokens.reasoning_tokens,
    ) == (120, 30, 7, 5, 11)
    assert (row.api_call_count, row.estimated_cost_usd, row.actual_cost_usd) == (2, pytest.approx(0.25), pytest.approx(0.2))
    assert got.last_active == _at(3)  # newest transcript row, any visibility
    assert got.lineage_root_id is None and got.preview == "hello there"
    assert row.system_prompt == "You are the pin." and dict(row.model_config) == {"temperature": 0.2}
    # The same row read through the listing carries the same numbers.
    [listed] = _listed(backend, scope_a)
    assert listed == got


def test_summary_counts_follow_the_counters_through_a_deactivation(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_rich(backend, scope_a, "rich-d")
    sid = SessionId("rich-d")
    transcript = store_a.messages.read_transcript(sid)
    assert store_a.messages.deactivate_message(sid, transcript[1].id)
    got = store_a.sessions.summary(sid)
    assert got is not None
    # CONTRACT: the row carries the transcript counters, which a deactivation leaves where the inserts put them.
    assert (got.session.message_count, got.session.tool_call_count) == (3, 2)


def test_never_recorded_facets_read_as_zero_none_and_started_at(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("bare-1"))
    got = store_a.sessions.summary(SessionId("bare-1"))
    assert got is not None
    row = got.session
    assert (row.message_count, row.tool_call_count, row.api_call_count) == (0, 0, 0)
    assert (row.input_tokens, row.output_tokens, row.cache_read_tokens, row.cache_write_tokens, row.reasoning_tokens) == (0, 0, 0, 0, 0)
    assert (row.estimated_cost_usd, row.actual_cost_usd) == (None, None)
    assert got.last_active == row.started_at
    assert got.preview == "" and got.unread is False
    assert (row.system_prompt, dict(row.model_config)) == (None, {})


def test_summary_is_the_unprojected_listing_row(backend, scope_a):
    store_a = backend.open(scope_a)
    for i in range(7):
        store_a.sessions.create(_new(f"row-{i}", source="api_server" if i % 2 else "cli"))
        store_a.messages.append(SessionId(f"row-{i}"), _msg(f"m{i}", 10 + i))
    store_a.sessions.set_pinned(SessionId("row-1"), True)
    store_a.sessions.set_title(SessionId("row-4"), "Bot Chat")
    store_a.sessions.set_hidden(SessionId("row-5"), True)
    # The oldest row gets the newest turn: LAST_ACTIVE order differs from STARTED_AT order.
    store_a.messages.append(SessionId("row-0"), _msg("late", 99))
    by_recency = _listed(backend, scope_a, SessionQuery(order=SessionOrder.LAST_ACTIVE))
    assert [str(s.session.session_id) for s in by_recency][:2] == ["row-0", "row-6"]
    assert by_recency[0].last_active == _at(99)
    queries = [
        SessionQuery(),
        SessionQuery(order=SessionOrder.LAST_ACTIVE, include_pinned=True),
        SessionQuery(sources=("api_server",), order=SessionOrder.LAST_ACTIVE, include_pinned=True),
        SessionQuery(search_text="bot chat", include_hidden=True, order=SessionOrder.LAST_ACTIVE),
        SessionQuery(include_hidden=True),
    ]
    for query in queries:
        for offset in range(0, 9, 3):
            for entry in store_a.sessions.list_sessions(query, SessionWindow(limit=3, offset=offset)):
                assert entry == store_a.sessions.summary(entry.session.session_id), query
    # The unpinned window plus the pinned back-fill; the pinned row has its own counters.
    first = store_a.sessions.list_sessions(SessionQuery(order=SessionOrder.LAST_ACTIVE, include_pinned=True), SessionWindow(limit=2))
    pinned = [s for s in first if s.session.pinned]
    assert [s.session.session_id for s in pinned] == [SessionId("row-1")] and pinned[0].session.message_count == 1


def test_compression_root_projects_to_tip_with_lineage_root(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("root-1"))
    store_a.messages.append(SessionId("root-1"), _msg("first", 1))
    store_a.usage.queue_usage(
        SessionId("root-1"), UsageDelta(tokens=TokenCounts(input_tokens=900), api_call_count=9, estimated_cost_usd=0.9)
    )
    store_a.sessions.end(SessionId("root-1"), "compression")
    store_a.sessions.create(_new("tip-1", parent_session_id=SessionId("root-1")))
    for i in (5, 6):
        store_a.messages.append(SessionId("tip-1"), _msg(f"t{i}", i))
    store_a.usage.queue_usage(
        SessionId("tip-1"), UsageDelta(tokens=TokenCounts(input_tokens=10), api_call_count=1, estimated_cost_usd=0.01)
    )
    store_a.sessions.update_system_prompt(SessionId("tip-1"), "tip prompt")
    [projected] = _listed(backend, scope_a, SessionQuery(order=SessionOrder.LAST_ACTIVE))
    assert projected.session.session_id == SessionId("tip-1")
    assert projected.lineage_root_id == SessionId("root-1") and projected.continuation_kind == "compression"
    assert projected.lineage_ids == (SessionId("root-1"), SessionId("tip-1"))
    # The hybrid row: transcript counters, recency and the prompt are the tip's; the usage accumulators stay the root row's.
    row = projected.session
    assert row.message_count == 2 and row.system_prompt == "tip prompt"
    assert projected.last_active == _at(6)
    assert (row.input_tokens, row.api_call_count, row.estimated_cost_usd) == (900, 9, pytest.approx(0.9))
    # The single-session read never projects.
    root = store_a.sessions.summary(SessionId("root-1"))
    tip = store_a.sessions.summary(SessionId("tip-1"))
    assert root is not None and root.lineage_root_id is None and (root.session.message_count, root.session.api_call_count) == (1, 9)
    assert tip is not None and tip.lineage_root_id is None and (tip.session.message_count, tip.session.api_call_count) == (2, 1)
    # Without projection no listed row carries a root (which rows list is the listing contract's question, not this one's).
    unprojected = _listed(backend, scope_a, SessionQuery(include_children=True, project_compression_tips=False))
    assert unprojected and all(s.lineage_root_id is None for s in unprojected)


def test_summary_admits_archived_hidden_and_children_but_stays_in_scope(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    _seed_rich(backend, scope_a, "wide-1")
    store_a.sessions.set_archived(SessionId("wide-1"), True)
    store_a.sessions.set_hidden(SessionId("wide-1"), True)
    assert _listed(backend, scope_a) == []
    got = store_a.sessions.summary(SessionId("wide-1"))
    assert got is not None and got.session.archived and got.session.hidden and got.session.message_count == 3
    assert store_b.sessions.summary(SessionId("wide-1")) is None
    assert _listed(backend, scope_b, SessionQuery(include_archived=True, include_hidden=True)) == []


def test_summary_compact_rows_leave_the_prompt_unread(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_rich(backend, scope_a, "compact-1")
    full = store_a.sessions.summary(SessionId("compact-1"))
    compact = store_a.sessions.summary(SessionId("compact-1"), compact_rows=True)
    assert full is not None and compact is not None
    # CONTRACT: a compact row marks itself and does not read the prompt snapshot; every other column is the full row's.
    assert (full.compact, compact.compact) == (False, True)
    assert compact.session.system_prompt is None and compact.session.system_prompt_hash is None
    assert compact.session.message_count == full.session.message_count == 3 and compact.preview == full.preview
    [listed] = store_a.sessions.list_sessions(SessionQuery(), SessionWindow(limit=5), compact_rows=True)
    assert listed == compact
