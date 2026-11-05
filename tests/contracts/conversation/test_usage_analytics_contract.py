"""Contract suite for ``UsageAnalytics`` (persistence/conversation/usage.py): the dashboard usage reads and the insights reads.

Seeds go through public verbs (``sessions.create``, ``usage.queue_usage`` then ``flush_usage``, ``usage.record_auxiliary_usage``, ``messages.append``) and the session row-dict seam (``started_at``, one NULL ``tool_call_count`` cell). Where the read defines no total order the rows compare as a multiset.
"""

from __future__ import annotations

import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Optional

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta

C = 1_750_000_000.0  # the window bound every case reads at
FAR = C + 10 * 86400  # a bound after every seeded session

CLI_CALLS = r'[{"id": "k1", "type": "function", "function": {"name": "skill_view", "arguments": "{\"name\": \"alpha\"}"}}, {"id": "k2", "type": "function", "function": {"name": "terminal", "arguments": "{\"command\": \"ls\"}"}}]'
LOCAL_CALLS = r'[{"id": "l1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a\"}"}}]'
TG_CALLS = r'[{"id": "t1", "type": "function", "function": {"name": "terminal", "arguments": "{\"command\": \"hermes skill_manage list\"}"}}]'


def _reads(store: Any, *names: str) -> list[Any]:
    return [getattr(store.usage_analytics, name) for name in names]


def _bag(rows: list[dict[str, Any]]) -> Counter:
    return Counter(tuple(sorted(row.items())) for row in rows)


def _session(backend: Any, scope: Any, store: Any, session_id: str, started_at: float, *, source: str = "cli", model: Optional[str] = None) -> None:
    store.sessions.create(NewSession(SessionId(session_id), source, model=model))
    backend.scope_store(scope)["sessions"][session_id]["started_at"] = datetime.fromtimestamp(started_at, UTC)


def _delta(store: Any, session_id: str, model: str, provider: str, *, base_url: Optional[str] = None, estimated: Optional[float] = None,
           actual: Optional[float] = None, cost_status: Optional[str] = None, cost_source: Optional[str] = None, **tokens: int) -> None:
    store.usage.queue_usage(SessionId(session_id), UsageDelta(
        tokens=TokenCounts(**tokens), api_call_count=1, estimated_cost_usd=estimated, actual_cost_usd=actual,
        cost_status=cost_status, cost_source=cost_source, route=BillingRoute(model, provider, base_url)))
    assert store.usage.flush_usage()


def _call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _seed_sessions(backend: Any, scope: Any, store: Any) -> None:
    """Session rows on both sides of ``C`` over two local days: models ``m-a`` on two providers, ``m-b``, unset and ``''``."""
    _session(backend, scope, store, "s-before", C - 10)
    _delta(store, "s-before", "m-a", "p-1", input_tokens=100, output_tokens=10)
    _session(backend, scope, store, "s-equal", C)
    _delta(store, "s-equal", "m-a", "p-1", input_tokens=1, output_tokens=1)
    _session(backend, scope, store, "s-a1", C + 100)
    _delta(store, "s-a1", "m-a", "p-1", estimated=0.5, input_tokens=10, output_tokens=5, cache_read_tokens=2, reasoning_tokens=3)
    store.messages.append(SessionId("s-a1"), NewMessage("assistant", "", C + 101, tool_calls=(_call("c1", "terminal", "{}"), _call("c2", "read_file", "{}"))))
    _session(backend, scope, store, "s-a2", C + 200)
    _delta(store, "s-a2", "m-a", "p-2", estimated=0.25, input_tokens=20, output_tokens=10)
    backend.scope_store(scope)["sessions"]["s-a2"]["tool_call_count"] = None
    _session(backend, scope, store, "s-b", C + 86400)
    _delta(store, "s-b", "m-b", "p-1", actual=0.4, cost_status="final", cost_source="api", input_tokens=7, output_tokens=3)
    _session(backend, scope, store, "s-none", C + 300)
    _session(backend, scope, store, "s-empty", C + 400, model="")


def _seed_calls(backend: Any, scope: Any, store: Any) -> None:
    """Main-loop and auxiliary attribution rows on sessions at and after ``C``."""
    _session(backend, scope, store, "u-old", C)
    _delta(store, "u-old", "m-a", "p-1", input_tokens=1000, output_tokens=1)
    store.usage.record_auxiliary_usage(SessionId("u-old"), "vision", model="m-v", billing_provider="p-v", tokens=TokenCounts(input_tokens=500))
    _session(backend, scope, store, "u-1", C + 10)
    _delta(store, "u-1", "m-a", "p-1", base_url="https://a", estimated=0.5, input_tokens=10, output_tokens=5, cache_read_tokens=1, reasoning_tokens=2)
    _delta(store, "u-1", "m-a", "p-1", base_url="https://b", estimated=0.25, input_tokens=4, output_tokens=2)
    _session(backend, scope, store, "u-2", C + 20)
    _delta(store, "u-2", "m-a", "p-1", input_tokens=30, output_tokens=10)
    _session(backend, scope, store, "u-3", C + 30)
    _delta(store, "u-3", "m-a", "p-2", input_tokens=1, output_tokens=1)
    store.usage.record_auxiliary_usage(SessionId("u-1"), "vision", model="m-v", billing_provider="p-v", tokens=TokenCounts(input_tokens=50, output_tokens=5), estimated_cost_usd=0.1)
    store.usage.record_auxiliary_usage(SessionId("u-2"), "vision", model="m-v", billing_provider="p-v", tokens=TokenCounts(input_tokens=20, output_tokens=2, cache_read_tokens=4, reasoning_tokens=1))
    store.usage.record_auxiliary_usage(SessionId("u-1"), "compression", model="m-a", billing_provider="p-1", tokens=TokenCounts(input_tokens=3, output_tokens=1))
    store.usage.record_auxiliary_usage(SessionId("u-3"), "title_generation")


def _seed_messages(backend: Any, scope: Any, store: Any) -> None:
    """Transcripts of a ``cli``, a ``local`` (created by ``cli``), a ``telegram`` and an older session."""
    def append(session_id: str, role: str, content: str, at: float, **fields: Any) -> None:
        store.messages.append(SessionId(session_id), NewMessage(role, content, at, **fields))

    _session(backend, scope, store, "m-old", C - 1)
    append("m-old", "assistant", "", C - 1, tool_calls=(_call("o1", "skill_view", '{"name": "old"}'),))
    _session(backend, scope, store, "m-cli", C)
    append("m-cli", "user", "hi", C + 1)
    append("m-cli", "assistant", "", C + 2, tool_calls=(_call("k1", "skill_view", '{"name": "alpha"}'), _call("k2", "terminal", '{"command": "ls"}')))
    append("m-cli", "tool", "ok", C + 3, tool_name="terminal", tool_call_id="k2")
    append("m-cli", "tool", "ok", C + 4, tool_call_id="k1")
    append("m-cli", "assistant", "done", C + 5)
    store.sessions.create(NewSession(SessionId("m-local"), "local"))
    store.sessions.create(NewSession(SessionId("m-local"), "cli"))
    backend.scope_store(scope)["sessions"]["m-local"]["started_at"] = datetime.fromtimestamp(C + 5, UTC)
    append("m-local", "user", "q", C + 6)
    append("m-local", "assistant", "", C + 7, tool_calls=(_call("l1", "read_file", '{"path": "a"}'),))
    append("m-local", "tool", "x", C + 8, tool_name="read_file", tool_call_id="l1")
    _session(backend, scope, store, "m-tg", C + 10, source="telegram")
    append("m-tg", "assistant", "", C + 11, tool_calls=(_call("t1", "terminal", '{"command": "hermes skill_manage list"}'),))
    append("m-tg", "tool", "x", C + 12, tool_name="terminal", tool_call_id="t1")


# ── dashboard reads over session rows ─────────────────────────────────────────


def test_daily_usage_groups_window_sessions_by_local_day(backend, scope_a):
    store = backend.open(scope_a)
    (daily_usage,) = _reads(store, "daily_usage")
    _seed_sessions(backend, scope_a, store)
    first, second = (time.strftime("%Y-%m-%d", time.localtime(C + offset)) for offset in (100, 86400))
    # CONTRACT: sessions started strictly after the bound, one row per local calendar day of started_at, ordered by day; a
    # day whose sessions carry no actual cost reads actual_cost 0, unset models and NULL tool-call cells still count.
    assert daily_usage(started_after=C) == [
        {"day": first, "input_tokens": 30, "output_tokens": 15, "cache_read_tokens": 2, "reasoning_tokens": 3,
         "estimated_cost": 0.75, "actual_cost": 0, "sessions": 4, "api_calls": 2},
        {"day": second, "input_tokens": 7, "output_tokens": 3, "cache_read_tokens": 0, "reasoning_tokens": 0,
         "estimated_cost": 0.0, "actual_cost": 0.4, "sessions": 1, "api_calls": 1},
    ]
    # CONTRACT: an empty window reads no rows.
    assert daily_usage(started_after=FAR) == []


def test_daily_usage_rows_are_fresh_mutable_dicts(backend, scope_a):
    store = backend.open(scope_a)
    (daily_usage,) = _reads(store, "daily_usage")
    _seed_sessions(backend, scope_a, store)
    rows = daily_usage(started_after=C)
    rows[0]["sessions"] = -1
    # CONTRACT: callers edit rows in place; an edit never reaches the next read.
    assert daily_usage(started_after=C)[0]["sessions"] == 4


def test_session_usage_by_model_keeps_empty_model_drops_unset(backend, scope_a):
    store = backend.open(scope_a)
    (by_model,) = _reads(store, "session_usage_by_model")
    _seed_sessions(backend, scope_a, store)
    # CONTRACT: one row per stored model, '' included and unset excluded, providers merged, ordered by input plus output
    # tokens descending (the volumes here are distinct, so the order is total); an all-unset cost reads 0.
    assert by_model(started_after=C) == [
        {"model": "m-a", "input_tokens": 30, "output_tokens": 15, "estimated_cost": 0.75, "sessions": 2, "api_calls": 2},
        {"model": "m-b", "input_tokens": 7, "output_tokens": 3, "estimated_cost": 0.0, "sessions": 1, "api_calls": 1},
        {"model": "", "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0, "sessions": 1, "api_calls": 0},
    ]
    # CONTRACT: an empty window reads no rows.
    assert by_model(started_after=FAR) == []


def test_session_usage_totals_and_the_model_rows_filter(backend, scope_a):
    store = backend.open(scope_a)
    (totals,) = _reads(store, "session_usage_totals")
    _seed_sessions(backend, scope_a, store)
    expected = {"total_input": 37, "total_output": 18, "total_cache_read": 2, "total_reasoning": 3, "total_estimated_cost": 0.75,
                "total_actual_cost": 0.4, "total_sessions": 5, "total_api_calls": 3}
    # CONTRACT: one row over every window session.
    assert totals(started_after=C) == expected
    # CONTRACT: model_rows_only drops sessions whose model is unset or ''.
    assert totals(started_after=C, model_rows_only=True) == {**expected, "total_sessions": 3}
    empty = {"total_input": None, "total_output": None, "total_cache_read": None, "total_reasoning": None,
             "total_estimated_cost": 0, "total_actual_cost": 0, "total_sessions": 0, "total_api_calls": None}
    # CONTRACT: over an empty window the sums are None, the cost sums 0 and the count 0.
    assert totals(started_after=FAR) == empty
    assert totals(started_after=FAR, model_rows_only=True) == empty


def test_session_usage_by_route_splits_providers(backend, scope_a):
    store = backend.open(scope_a)
    (by_route,) = _reads(store, "session_usage_by_route")
    _seed_sessions(backend, scope_a, store)
    # CONTRACT: one row per (model, billing_provider) of window sessions with a non-empty model, ordered by input plus output
    # tokens descending (distinct volumes here); tool_calls sums the stored cells and reads None where every cell is NULL.
    assert by_route(started_after=C) == [
        {"model": "m-a", "billing_provider": "p-2", "input_tokens": 20, "output_tokens": 10, "cache_read_tokens": 0, "reasoning_tokens": 0,
         "estimated_cost": 0.25, "actual_cost": 0, "sessions": 1, "api_calls": 1, "tool_calls": None, "last_used_at": C + 200,
         "avg_tokens_per_session": 30.0},
        {"model": "m-a", "billing_provider": "p-1", "input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 2, "reasoning_tokens": 3,
         "estimated_cost": 0.5, "actual_cost": 0, "sessions": 1, "api_calls": 1, "tool_calls": 2, "last_used_at": C + 100,
         "avg_tokens_per_session": 15.0},
        {"model": "m-b", "billing_provider": "p-1", "input_tokens": 7, "output_tokens": 3, "cache_read_tokens": 0, "reasoning_tokens": 0,
         "estimated_cost": 0.0, "actual_cost": 0.4, "sessions": 1, "api_calls": 1, "tool_calls": 0, "last_used_at": C + 86400,
         "avg_tokens_per_session": 10.0},
    ]
    # CONTRACT: an empty window reads no rows.
    assert by_route(started_after=FAR) == []


def test_session_tool_calls_by_route_counts_null_cells_as_zero(backend, scope_a):
    store = backend.open(scope_a)
    (tool_calls,) = _reads(store, "session_tool_calls_by_route")
    _seed_sessions(backend, scope_a, store)
    # CONTRACT: the same groups as session_usage_by_route, unordered, with a NULL tool-call cell summed as 0.
    assert _bag(tool_calls(started_after=C)) == _bag([
        {"model": "m-a", "billing_provider": "p-1", "tool_calls": 2},
        {"model": "m-a", "billing_provider": "p-2", "tool_calls": 0},
        {"model": "m-b", "billing_provider": "p-1", "tool_calls": 0},
    ])
    # CONTRACT: an empty window reads no rows.
    assert tool_calls(started_after=FAR) == []


# ── dashboard reads over per-call usage rows ──────────────────────────────────


def _pop_last_used(rows: list[dict[str, Any]], before: float, after: float) -> list[dict[str, Any]]:
    for row in rows:
        # CONTRACT: last_used_at is the latest write of the group's rows.
        assert before <= row.pop("last_used_at") <= after
    return rows


def test_call_usage_by_route_reads_main_loop_rows_only(backend, scope_a):
    store = backend.open(scope_a)
    (call_usage,) = _reads(store, "call_usage_by_route")
    before = time.time()
    _seed_calls(backend, scope_a, store)
    after = time.time()
    rows = _pop_last_used(call_usage(started_after=C), before, after)
    # CONTRACT: main-loop rows (no task) of sessions started after the bound, grouped by (model, billing_provider) across base
    # URLs, ordered by input plus output tokens descending (distinct volumes here); sessions counts distinct sessions and
    # avg_tokens_per_session averages per usage row; auxiliary rows and the session started at the bound are excluded.
    assert rows == [
        {"model": "m-a", "billing_provider": "p-1", "input_tokens": 44, "output_tokens": 17, "cache_read_tokens": 1, "reasoning_tokens": 2,
         "estimated_cost": 0.75, "actual_cost": 0.0, "sessions": 2, "api_calls": 3, "avg_tokens_per_session": pytest.approx(61 / 3)},
        {"model": "m-a", "billing_provider": "p-2", "input_tokens": 1, "output_tokens": 1, "cache_read_tokens": 0, "reasoning_tokens": 0,
         "estimated_cost": 0.0, "actual_cost": 0.0, "sessions": 1, "api_calls": 1, "avg_tokens_per_session": 2.0},
    ]
    # CONTRACT: an empty window reads no rows.
    assert call_usage(started_after=FAR) == []


def test_per_call_rows_carry_actual_cost_and_cost_labels(backend, scope_a):
    store = backend.open(scope_a)
    call_usage, model_usage = _reads(store, "call_usage_by_route", "insight_model_usage")
    _session(backend, scope_a, store, "c-1", C + 5)
    _delta(store, "c-1", "m-a", "p-1", actual=0.4, cost_status="final", cost_source="api", input_tokens=1, output_tokens=1)
    _delta(store, "c-1", "m-a", "p-1", actual=0.1, input_tokens=1, output_tokens=1)
    # CONTRACT: a usage row sums its actual cost; an unlabelled call keeps the stored cost status and source.
    (row,) = call_usage(started_after=C)
    assert row["actual_cost"] == pytest.approx(0.5)
    (usage,) = model_usage(started_since=C)
    assert (usage["actual_cost_usd"], usage["cost_status"], usage["cost_source"]) == (pytest.approx(0.5), "final", "api")


def test_auxiliary_usage_by_model_task_reads_task_rows_only(backend, scope_a):
    store = backend.open(scope_a)
    (auxiliary,) = _reads(store, "auxiliary_usage_by_model_task")
    before = time.time()
    _seed_calls(backend, scope_a, store)
    after = time.time()
    rows = _pop_last_used(auxiliary(started_after=C), before, after)
    # CONTRACT: rows with a task grouped by (model, task, billing_provider), ordered by input plus output tokens descending
    # (distinct volumes here); sessions counts distinct sessions; an auxiliary call without a route reads model 'unknown'
    # and provider ''; main-loop rows and the session started at the bound are excluded.
    assert rows == [
        {"model": "m-v", "task": "vision", "billing_provider": "p-v", "input_tokens": 70, "output_tokens": 7, "cache_read_tokens": 4,
         "reasoning_tokens": 1, "estimated_cost": 0.1, "sessions": 2, "api_calls": 2},
        {"model": "m-a", "task": "compression", "billing_provider": "p-1", "input_tokens": 3, "output_tokens": 1, "cache_read_tokens": 0,
         "reasoning_tokens": 0, "estimated_cost": 0.0, "sessions": 1, "api_calls": 1},
        {"model": "unknown", "task": "title_generation", "billing_provider": "", "input_tokens": 0, "output_tokens": 0,
         "cache_read_tokens": 0, "reasoning_tokens": 0, "estimated_cost": 0.0, "sessions": 1, "api_calls": 1},
    ]
    # CONTRACT: an empty window reads no rows.
    assert auxiliary(started_after=FAR) == []


def _usage_row(session_id: str, model: str, provider: str, input_tokens: int, output_tokens: int, *, base_url: str = "",
               cache_read: int = 0, reasoning: int = 0, estimated: float = 0.0) -> dict[str, Any]:
    return {"session_id": session_id, "model": model, "billing_provider": provider, "billing_base_url": base_url, "billing_mode": "",
            "api_call_count": 1, "input_tokens": input_tokens, "output_tokens": output_tokens, "cache_read_tokens": cache_read,
            "cache_write_tokens": 0, "reasoning_tokens": reasoning, "estimated_cost_usd": estimated, "actual_cost_usd": 0.0,
            "cost_status": None, "cost_source": None}


def test_insight_model_usage_reads_every_row_at_or_after_the_bound(backend, scope_a):
    store = backend.open(scope_a)
    (model_usage,) = _reads(store, "insight_model_usage")
    _seed_calls(backend, scope_a, store)
    later = [
        _usage_row("u-1", "m-a", "p-1", 10, 5, base_url="https://a", cache_read=1, reasoning=2, estimated=0.5),
        _usage_row("u-1", "m-a", "p-1", 4, 2, base_url="https://b", estimated=0.25),
        _usage_row("u-1", "m-v", "p-v", 50, 5, estimated=0.1),
        _usage_row("u-1", "m-a", "p-1", 3, 1),
        _usage_row("u-2", "m-a", "p-1", 30, 10),
        _usage_row("u-2", "m-v", "p-v", 20, 2, cache_read=4, reasoning=1),
        _usage_row("u-3", "m-a", "p-2", 1, 1),
        _usage_row("u-3", "unknown", "", 0, 0),
    ]
    # CONTRACT: every usage row, main loop and auxiliary alike, of sessions started at or after the bound (inclusive),
    # unordered, with unset route fields as ''.
    assert _bag(model_usage(started_since=C)) == _bag([*later, _usage_row("u-old", "m-a", "p-1", 1000, 1), _usage_row("u-old", "m-v", "p-v", 500, 0)])
    assert _bag(model_usage(started_since=C + 10)) == _bag(later)
    # CONTRACT: an empty window reads no rows.
    assert model_usage(started_since=FAR) == []


# ── insights reads over sessions and messages ─────────────────────────────────


def _insight_session(session_id: str, started_at: float, **cells: Any) -> dict[str, Any]:
    return {"id": session_id, "source": "cli", "model": None, "started_at": started_at, "ended_at": None, "message_count": 0, "tool_call_count": 0,
            "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "billing_provider": None,
            "billing_base_url": None, "billing_mode": None, "estimated_cost_usd": None, "actual_cost_usd": None, "cost_status": None,
            "cost_source": None, "api_call_count": 0, **cells}


def test_insight_sessions_newest_first_from_the_bound(backend, scope_a):
    store = backend.open(scope_a)
    (sessions,) = _reads(store, "insight_sessions")
    _seed_sessions(backend, scope_a, store)
    # CONTRACT: sessions started at or after the bound (inclusive), newest first (distinct starts here), with the stored
    # cells: a NULL tool-call cell reads None and an unaccounted session keeps unset costs.
    assert sessions(started_since=C) == [
        _insight_session("s-b", C + 86400, model="m-b", input_tokens=7, output_tokens=3, billing_provider="p-1", estimated_cost_usd=0.0,
                         actual_cost_usd=0.4, cost_status="final", cost_source="api", api_call_count=1),
        _insight_session("s-empty", C + 400, model=""),
        _insight_session("s-none", C + 300),
        _insight_session("s-a2", C + 200, model="m-a", tool_call_count=None, input_tokens=20, output_tokens=10, billing_provider="p-2",
                         estimated_cost_usd=0.25, api_call_count=1),
        _insight_session("s-a1", C + 100, model="m-a", message_count=1, tool_call_count=2, input_tokens=10, output_tokens=5,
                         cache_read_tokens=2, billing_provider="p-1", estimated_cost_usd=0.5, api_call_count=1),
        _insight_session("s-equal", C, model="m-a", input_tokens=1, output_tokens=1, billing_provider="p-1", estimated_cost_usd=0.0,
                         api_call_count=1),
    ]
    # CONTRACT: an empty window reads no rows.
    assert sessions(started_since=FAR) == []


def test_insight_source_filter_includes_local_rows_created_by_the_surface(backend, scope_a):
    store = backend.open(scope_a)
    (sessions,) = _reads(store, "insight_sessions")
    _seed_messages(backend, scope_a, store)
    ids = lambda source: [row["id"] for row in sessions(started_since=C, source=source)]
    # CONTRACT: no source (or '') reads every window session, newest first.
    assert ids(None) == ids("") == ["m-tg", "m-local", "m-cli"]
    # CONTRACT: a source reads its own sessions plus the 'local' sessions that surface created.
    assert ids("cli") == ["m-local", "m-cli"]
    # CONTRACT: 'local' matches the stored source itself.
    assert ids("local") == ["m-local"]
    assert ids("telegram") == ["m-tg"]
    assert ids("nope") == []
    local = sessions(started_since=C, source="cli")[0]
    # CONTRACT: the row keeps its stored source and counts every transcript row and tool call.
    assert (local["source"], local["message_count"], local["tool_call_count"]) == ("local", 3, 1)


def test_insight_tool_result_counts_need_a_tool_name(backend, scope_a):
    store = backend.open(scope_a)
    (counts,) = _reads(store, "insight_tool_result_counts")
    _seed_messages(backend, scope_a, store)
    # CONTRACT: tool rows with a tool_name, counted per (session, tool), unordered; a tool row without a name and the
    # session started before the bound are excluded.
    assert _bag(counts(started_since=C)) == _bag([
        {"session_id": "m-cli", "tool_name": "terminal", "count": 1},
        {"session_id": "m-local", "tool_name": "read_file", "count": 1},
        {"session_id": "m-tg", "tool_name": "terminal", "count": 1},
    ])
    # CONTRACT: the source filter applies, local surface arm included.
    assert _bag(counts(started_since=C, source="cli")) == _bag([
        {"session_id": "m-cli", "tool_name": "terminal", "count": 1},
        {"session_id": "m-local", "tool_name": "read_file", "count": 1},
    ])
    # CONTRACT: the bound is inclusive on the session's start: a later bound drops the session started at C.
    assert _bag(counts(started_since=C + 1)) == _bag([
        {"session_id": "m-local", "tool_name": "read_file", "count": 1},
        {"session_id": "m-tg", "tool_name": "terminal", "count": 1},
    ])
    assert counts(started_since=C, source="nope") == []


def test_insight_tool_call_payloads_are_the_stored_text(backend, scope_a):
    store = backend.open(scope_a)
    (payloads,) = _reads(store, "insight_tool_call_payloads")
    _seed_messages(backend, scope_a, store)
    # CONTRACT: every assistant row with tool calls, unordered, as its stored JSON text; rows without calls are excluded.
    assert _bag(payloads(started_since=C)) == _bag([
        {"session_id": "m-cli", "tool_calls": CLI_CALLS},
        {"session_id": "m-local", "tool_calls": LOCAL_CALLS},
        {"session_id": "m-tg", "tool_calls": TG_CALLS},
    ])
    assert payloads(started_since=C, source="telegram") == [{"session_id": "m-tg", "tool_calls": TG_CALLS}]
    assert payloads(started_since=FAR) == []


def test_insight_skill_call_payloads_match_the_text_anywhere(backend, scope_a):
    store = backend.open(scope_a)
    (skills,) = _reads(store, "insight_skill_call_payloads")
    _seed_messages(backend, scope_a, store)
    # CONTRACT: assistant rows whose tool_calls text contains skill_view or skill_manage anywhere (a name inside a terminal
    # call's arguments included), with the row timestamp, unordered.
    assert _bag(skills(started_since=C)) == _bag([
        {"tool_calls": CLI_CALLS, "timestamp": C + 2},
        {"tool_calls": TG_CALLS, "timestamp": C + 11},
    ])
    assert skills(started_since=C, source="cli") == [{"tool_calls": CLI_CALLS, "timestamp": C + 2}]
    assert skills(started_since=C, source="local") == []


def test_insight_message_stats_over_rows_and_over_none(backend, scope_a):
    store = backend.open(scope_a)
    (stats,) = _reads(store, "insight_message_stats")
    _seed_messages(backend, scope_a, store)
    # CONTRACT: one row counting every message of the window's sessions and its user, assistant and tool rows.
    assert stats(started_since=C) == {"total_messages": 10, "user_messages": 2, "assistant_messages": 4, "tool_messages": 4}
    assert stats(started_since=C, source="cli") == {"total_messages": 8, "user_messages": 2, "assistant_messages": 3, "tool_messages": 3}
    assert stats(started_since=C, source="telegram") == {"total_messages": 2, "user_messages": 0, "assistant_messages": 1, "tool_messages": 1}
    none = {"total_messages": 0, "user_messages": None, "assistant_messages": None, "tool_messages": None}
    # CONTRACT: over no rows the count is 0 and the per-role sums None.
    assert stats(started_since=C, source="nope") == none
    assert stats(started_since=FAR) == none
