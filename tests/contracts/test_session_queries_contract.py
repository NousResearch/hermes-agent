"""Contract cases for session ownership, message counts, cron summaries, session filters, pruning filters and text search; related session cases live in ``tests/contracts/conversation/test_sessions.py``.

Cases run on every registered backend. Setup uses public protocol operations except for ``_seed_text_corpus``, which calls the backends' ``seed_session`` and ``seed_message`` fixture helpers; observations use the contract surface. The rules are documented in the ``persistence`` package.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, UTC

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, SessionOrder, SessionQuery, SessionWindow
from persistence.housekeeping.maintenance import PruneCriteria
from persistence.conversation.search import SearchSort
from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta

from tests.contracts._scopes import MEDIA_ROOT


T0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)


def _new(session_id: str, source: str = "cli", **kwargs) -> NewSession:
    return NewSession(session_id=SessionId(session_id), source=source, **kwargs)


def _msg(content: str, i: int, role: str = "user") -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=(T0 + timedelta(seconds=i)).timestamp())


# ── persistence.conversation.sessions.SessionRepository.owns_session ───────────────────────────────────────────────────────────


def test_owns_session_is_existence_in_scope_whatever_the_visibility(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(_new("own-1"))
    assert store_a.sessions.owns_session(SessionId("own-1")) is True
    store_a.sessions.set_hidden(SessionId("own-1"), True)
    store_a.sessions.end(SessionId("own-1"), "cli_close")
    assert store_a.sessions.owns_session(SessionId("own-1")) is True
    # Unknown ids and other scopes' rows are not owned.
    assert store_a.sessions.owns_session(SessionId("nobody")) is False
    assert store_b.sessions.owns_session(SessionId("own-1")) is False


# ── persistence.conversation.messages.MessageRepository.count_all ─────────────────────────────────────────────────────


def test_count_all_is_the_store_wide_row_census(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    assert store_a.messages.count_all() == 0
    for sid in ("cnt-1", "cnt-2"):
        store_a.sessions.create(_new(sid))
    for i in range(3):
        store_a.messages.append(SessionId("cnt-1"), _msg(f"one {i}", i))
    store_a.messages.append(SessionId("cnt-2"), _msg("two", 10))
    assert store_a.messages.count_all() == 4
    # Compaction archives the old rows and appends the replacement: archived rows keep counting.
    store_a.messages.compact(SessionId("cnt-1"), [_msg("SUMMARY", 20, role="assistant")])
    assert store_a.sessions.get(SessionId("cnt-1")).message_count == 1
    assert store_a.messages.count_all() == 5
    # A deleted session's rows stop counting; other scopes never count.
    assert store_a.sessions.delete(SessionId("cnt-2"), media_root=MEDIA_ROOT) == (SessionId("cnt-2"),)
    assert store_a.messages.count_all() == 4
    assert store_b.messages.count_all() == 0


# ── cron runs carry the listing entry of each run ────────────────────────────


def test_cron_run_entries_carry_the_summary_of_each_run(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("cron_job7_20260921_120000")  # the pre-unified cron_{job}_{YYYYmmdd_HHMMSS} run id
    store_a.sessions.create(_new(str(sid), source="cron", model_config={"temperature": 0.1}))
    store_a.messages.append(sid, _msg("run the report", 1))
    store_a.messages.append(
        sid, NewMessage(role="assistant", content="", timestamp=(T0 + timedelta(seconds=2)).timestamp(), tool_calls=({"id": "c1", "type": "function"},))
    )
    store_a.usage.queue_usage(
        sid, UsageDelta(tokens=TokenCounts(input_tokens=40, output_tokens=9), api_call_count=1, estimated_cost_usd=0.5)
    )
    # The cron listing does not drain the accounting queue; a caller that wants the totals flushes first.
    assert store_a.usage.flush_usage() is True
    [run] = store_a.sessions.list_cron_job_runs("job7", SessionWindow(limit=5))
    # CONTRACT: the run's entry IS the listing entry `summary` computes, with the job it belongs to (a legacy id names no execution).
    expected = store_a.sessions.summary(sid)
    assert expected is not None
    assert run.entry == expected
    assert (run.cron_job_id, run.cron_execution_id) == ("job7", None)
    row = run.entry.session
    assert (row.message_count, row.tool_call_count) == (2, 1)
    assert (row.input_tokens, row.output_tokens, row.api_call_count) == (40, 9, 1)
    assert row.estimated_cost_usd == pytest.approx(0.5)
    assert run.entry.lineage_root_id is None and run.entry.unread is False


# ── SessionQuery.search_text ───────────────────────────────────────────────


def _seed_search(backend, scope):
    store = backend.open(scope)
    s = backend.open(scope).sessions
    s.create(_new("s-alpha"))
    s.set_title(SessionId("s-alpha"), "Deploy AN-94 rollout")
    s.create(_new("s-beta"))
    s.set_title(SessionId("s-beta"), "weekly notes")
    s.create(_new("zz-needle-id"))
    s.create(_new("root-x"))
    s.set_title(SessionId("root-x"), "Kraken hunt")
    store.messages.append(SessionId("root-x"), _msg("start", 1))
    s.end(SessionId("root-x"), "compression")
    s.create(_new("tip-x", parent_session_id=SessionId("root-x")))
    store.messages.append(SessionId("tip-x"), _msg("continued", 2))


def _search_ids(backend, scope, text, order=SessionOrder.LAST_ACTIVE) -> set[str]:
    store = backend.open(scope)
    entries = store.sessions.list_sessions(SessionQuery(search_text=text, order=order), SessionWindow(limit=50))
    return {str(e.session.session_id) for e in entries}


@pytest.mark.parametrize("order", [SessionOrder.LAST_ACTIVE, SessionOrder.STARTED_AT])
def test_search_text_matches_title_id_and_compact_title_over_the_chain(backend, scope_a, scope_b, order):
    store_a = backend.open(scope_a)
    _seed_search(backend, scope_a)
    everything = _search_ids(backend, scope_a, None, order)
    assert everything == {"s-alpha", "s-beta", "zz-needle-id", "tip-x"}
    # Title substring, case-insensitive.
    assert _search_ids(backend, scope_a, "ROLLOUT", order) == (everything if order is SessionOrder.STARTED_AT else {"s-alpha"})
    # The punctuation-stripped arm: `an94` finds `AN-94`; the typed form finds it too.
    assert _search_ids(backend, scope_a, "an94", order) == (everything if order is SessionOrder.STARTED_AT else {"s-alpha"})
    assert _search_ids(backend, scope_a, "AN-94", order) == (everything if order is SessionOrder.STARTED_AT else {"s-alpha"})
    # Id substring on an untitled row.
    assert _search_ids(backend, scope_a, "needle", order) == (everything if order is SessionOrder.STARTED_AT else {"zz-needle-id"})
    # Chain arm: the root's title and the root's id both surface the listed tip.
    assert _search_ids(backend, scope_a, "kraken", order) == (everything if order is SessionOrder.STARTED_AT else {"tip-x"})
    assert _search_ids(backend, scope_a, "root-x", order) == (everything if order is SessionOrder.STARTED_AT else {"tip-x"})
    # A blank needle restricts nothing; pattern characters are literal text.
    assert _search_ids(backend, scope_a, "   ", order) == everything
    assert _search_ids(backend, scope_a, "%", order) == (everything if order is SessionOrder.STARTED_AT else set())
    assert _search_ids(backend, scope_a, "s_alpha", order) == (everything if order is SessionOrder.STARTED_AT else set())
    # Combines by AND with the other fields, and never reads another scope.
    entries = store_a.sessions.list_sessions(
        SessionQuery(search_text="s-", exclude_sources=("cli",), order=order), SessionWindow(limit=50)
    )
    assert entries == ()
    assert _search_ids(backend, scope_b, "alpha", order) == set()


# ── persistence.conversation.sessions.SessionRepository.search_by_id ──────────────────────────────────────────────────


def test_search_by_id_ranks_exact_then_prefix_then_substring(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    s = backend.open(scope_a).sessions
    # Recency is set by the newest transcript row; the substring match is the most recent, the exact one the oldest.
    for i, sid in enumerate(["abc123", "abc123-later", "x-abc123-mid", "x-abc123-new", "zzz"]):
        s.create(_new(sid))
        store_a.messages.append(SessionId(sid), _msg(f"m {sid}", 10 * (i + 1)))
    s.set_archived(SessionId("x-abc123-mid"), True)
    got = [str(r.session.session_id) for r in s.search_by_id("  ABC123 ")]
    # Rank 0, rank 1, then the substring matches in last-active order (newest first); archived rows are admitted by default.
    assert got == ["abc123", "abc123-later", "x-abc123-new", "x-abc123-mid"]
    assert [str(r.session.session_id) for r in s.search_by_id("abc123", limit=2)] == ["abc123", "abc123-later"]
    # include_archived=False narrows admission to unarchived rows.
    narrowed = s.search_by_id("abc123", include_archived=False)
    assert "x-abc123-mid" not in {str(r.session.session_id) for r in narrowed}
    # Rows are the listing's entries.
    [first] = s.search_by_id("abc123", limit=1)
    assert first.session.message_count == 1
    # Blank needles, pattern characters, other scopes and a non-positive limit find nothing.
    assert s.search_by_id("   ") == ()
    assert s.search_by_id("%") == ()
    assert store_b.sessions.search_by_id("abc123") == ()
    assert s.search_by_id("abc123", limit=0) == ()


def test_search_by_id_ranks_only_the_limit_times_4_newest_matches(backend, scope_a):
    """The source's candidate window (``limit * 4`` most recently active matches) is the contract's bound."""
    store_a = backend.open(scope_a)
    s = backend.open(scope_a).sessions
    for i, sid in enumerate(["win", "x-win-1", "x-win-2", "x-win-3", "x-win-4"]):
        s.create(_new(sid))
        store_a.messages.append(SessionId(sid), _msg(f"m {sid}", 10 * (i + 1)))
    # limit=1: the window holds the four newest matches, so the older exact match is not ranked.
    assert [str(r.session.session_id) for r in s.search_by_id("win", limit=1)] == ["x-win-4"]
    # limit=2: the window (8) reaches the exact match, which ranks first.
    assert [str(r.session.session_id) for r in s.search_by_id("win", limit=2)] == ["win", "x-win-4"]


def test_search_by_id_resolves_an_old_compression_root_to_its_tip(backend, scope_a):
    store_a = backend.open(scope_a)
    s = backend.open(scope_a).sessions
    s.create(_new("old-root-7"))
    store_a.messages.append(SessionId("old-root-7"), _msg("start", 1))
    s.end(SessionId("old-root-7"), "compression")
    s.create(_new("fresh-tip-7", parent_session_id=SessionId("old-root-7")))
    store_a.messages.append(SessionId("fresh-tip-7"), _msg("go on", 2))
    [hit] = s.search_by_id("old-root-7")
    assert str(hit.session.session_id) == "fresh-tip-7" and hit.lineage_root_id == SessionId("old-root-7")


# ── PruneCriteria accounting filters ───────────────────────────────────────


def _seed_accounting(backend, scope):
    s, m, u = backend.open(scope).sessions, backend.open(scope).messages, backend.open(scope).usage
    for sid in ("acc-1", "acc-2", "acc-3"):
        s.create(_new(sid))
    m.append(SessionId("acc-1"), _msg("hi", 1))
    m.append(
        SessionId("acc-1"),
        NewMessage(role="assistant", content="", timestamp=(T0 + timedelta(seconds=2)).timestamp(), tool_calls=({"id": "a", "type": "function"}, {"id": "b", "type": "function"})),
    )
    u.queue_usage(
        SessionId("acc-1"),
        UsageDelta(tokens=TokenCounts(input_tokens=100, output_tokens=50), api_call_count=1, estimated_cost_usd=0.5, route=BillingRoute(model="m1", billing_provider="OpenRouter")),
    )
    m.append(SessionId("acc-2"), _msg("yo", 3))
    u.queue_usage(
        SessionId("acc-2"),
        UsageDelta(tokens=TokenCounts(input_tokens=10), api_call_count=1, estimated_cost_usd=0.1, actual_cost_usd=2.0, route=BillingRoute(model="m2", billing_provider="anthropic")),
    )
    m.append(SessionId("acc-3"), _msg("plain", 4))
    for sid in ("acc-1", "acc-2", "acc-3"):
        s.end(SessionId(sid), "cli_close")
    assert u.flush_usage() is True


def _prune_ids(backend, scope, criteria: PruneCriteria) -> set[str]:
    store = backend.open(scope)
    rows = store.maintenance.list_prune_candidates(criteria=criteria)
    ids = {str(r.session_id) for r in rows}
    assert store.maintenance.count_prune_matches(criteria=criteria) == len(ids)
    return ids


@pytest.mark.parametrize(
    "criteria, expected",
    [
        (PruneCriteria(provider="openrouter"), {"acc-1"}),
        (PruneCriteria(provider="ANTHROPIC"), {"acc-2"}),
        (PruneCriteria(provider=""), {"acc-1", "acc-2", "acc-3"}),
        (PruneCriteria(min_tokens=150), {"acc-1"}),
        (PruneCriteria(max_tokens=10), {"acc-2", "acc-3"}),
        (PruneCriteria(min_cost=1.0), {"acc-2"}),
        (PruneCriteria(max_cost=0.5), {"acc-1", "acc-3"}),
        (PruneCriteria(min_tool_calls=2), {"acc-1"}),
        (PruneCriteria(max_tool_calls=0), {"acc-2", "acc-3"}),
        (PruneCriteria(provider="openrouter", max_tokens=149), set()),
    ],
)
def test_prune_accounting_filters_narrow_candidates_and_counts(backend, scope_a, criteria, expected):
    _seed_accounting(backend, scope_a)
    assert _prune_ids(backend, scope_a, criteria) == expected


def test_prune_with_an_accounting_filter_deletes_exactly_its_candidates(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_accounting(backend, scope_a)
    assert store_a.maintenance.prune_sessions(None, criteria=PruneCriteria(provider="anthropic"), media_root=MEDIA_ROOT).deleted_ids == (SessionId("acc-2"),)
    assert store_a.sessions.get(SessionId("acc-2")) is None
    assert store_a.sessions.get(SessionId("acc-1")) is not None
    assert store_a.sessions.get(SessionId("acc-3")) is not None


# ── persistence.conversation.search.SessionSearch.search_messages ─────────────────────────────────────────────────────


def _seed_text_corpus(backend, scope) -> dict[str, object]:
    """Distinct timestamps so NEWEST has a total order; both sessions start now."""
    store = backend.open(scope)
    now = datetime.now(UTC).timestamp()
    store.search.seed_session("txt-1", source="cli", model="m", started_at=now)
    store.search.seed_session("txt-cron", source="cron", model="m", started_at=now)
    seed = backend.open(scope).search.seed_message

    def at(i: int) -> float:
        return (T0 + timedelta(seconds=i)).timestamp()

    return {
        "fox": seed("txt-1", "user", "the quick brown fox", timestamp=at(1)),
        "deploy": seed("txt-1", "assistant", "deploy chat-send now via my-app.config.ts", timestamp=at(2)),
        "needle": seed("txt-1", "user", "unrelated haystack needle", timestamp=at(3)),
        "todo": seed("txt-1", "assistant", "TODO fix the quick parser", timestamp=at(4)),
        "cron": seed("txt-cron", "user", "cron quick digest", timestamp=at(5)),
    }


def _text_ids(backend, scope, text, *, sort=SearchSort.NEWEST, **kwargs) -> list:
    store = backend.open(scope)
    return [hit.message_id for hit in store.search.search_messages(text, sort=sort, **kwargs)]


@pytest.mark.parametrize(
    "text, names",
    [
        ("quick", {"fox", "todo", "cron"}),
        ('"quick brown"', {"fox"}),
        ("TODO: fix", {"todo"}),
        ("fox OR needle", {"fox", "needle"}),
        ("quick NOT parser", {"fox", "cron"}),
        ("hay*", {"needle"}),
        ("chat-send", {"deploy"}),
        ("quick NOT brown fox", {"todo", "cron"}),
        ("quick NOT parser AND fox", {"fox"}),
        ("digest OR brown NOT parser", {"cron", "fox"}),
    ],
)
def test_search_messages_reads_the_store_query_syntax(backend, scope_a, text, names):
    ids = _seed_text_corpus(backend, scope_a)
    got = _text_ids(backend, scope_a, text)
    assert set(got) == {ids[n] for n in names}
    assert len(got) == len(set(got))


@pytest.mark.parametrize("text", ["", "   ", ":::", "OR", "AND NOT", '""', "(^~)"])
def test_search_messages_with_nothing_searchable_is_empty(backend, scope_a, text):
    store_a = backend.open(scope_a)
    _seed_text_corpus(backend, scope_a)
    assert store_a.search.search_messages(text, limit=10) == ()


@pytest.mark.parametrize(
    "text",
    ["a OR NOT b", "NEAR(quick fox)", "col:val quick", "-quick", "quick.", "quick_brown", '"quick" "', "quick*fox*", "((((", "\x00quick", "\x000\x00", "\x009999\x00 quick"],
)
def test_search_messages_never_raises_on_user_input(backend, scope_a, text):
    store_a = backend.open(scope_a)
    ids = _seed_text_corpus(backend, scope_a)
    hits = store_a.search.search_messages(text, limit=10)
    assert {hit.message_id for hit in hits} <= set(ids.values())


def test_search_messages_filters_and_paging(backend, scope_a):
    ids = _seed_text_corpus(backend, scope_a)
    # Paging one hit at a time under NEWEST loses and duplicates nothing and keeps timestamp order.
    assert [_text_ids(backend, scope_a, "quick", limit=1, offset=i)[0] for i in range(3)] == [ids["cron"], ids["todo"], ids["fox"]]
    assert _text_ids(backend, scope_a, "quick", limit=1, offset=3) == []
    assert _text_ids(backend, scope_a, "quick", sort=SearchSort.OLDEST, limit=2) + _text_ids(backend, scope_a, "quick", sort=SearchSort.OLDEST, limit=2, offset=2) == [ids["fox"], ids["todo"], ids["cron"]]
    assert set(_text_ids(backend, scope_a, "quick", role_filter=["user"])) == {ids["fox"], ids["cron"]}
    assert set(_text_ids(backend, scope_a, "quick", exclude_sources=["cron"])) == {ids["fox"], ids["todo"]}


def test_session_start_window_bounds_search(backend, scope_a):
    ids = _seed_text_corpus(backend, scope_a)
    now = int(datetime.now(UTC).timestamp())
    everything = {ids["fox"], ids["todo"], ids["cron"]}
    assert set(_text_ids(backend, scope_a, "quick", after_ts=now - 3600, before_ts=now + 3600)) == everything
    assert set(_text_ids(backend, scope_a, "quick", role_filter=["tool", "user"], after_ts=now - 3600)) == {ids["fox"], ids["cron"]}
    for window in ({"after_ts": now + 3600}, {"before_ts": now - 3600}):
        assert _text_ids(backend, scope_a, "quick", **window) == []
