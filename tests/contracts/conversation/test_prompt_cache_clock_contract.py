"""Contract cases for the session's prompt-cache clock: the ``UsageDelta.prompt_cache_clock`` write through ``queue_usage``, the ``SessionRecord`` cell, and the two usage reads ``prompt_cache_clock`` and ``prompt_cache_generation``.

Sessions are minted through ``SessionRepository.create`` and transcripts written through ``MessageRepository``; archived-compacted rows come from ``compact`` and rewound rows from ``deactivate_message``, so the generation is observed over real visibility states.
"""

from __future__ import annotations

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import SESSION_ROW_COLUMNS, NewSession
from persistence.conversation.usage import PromptCacheClockRecord, TokenCounts, UsageDelta

SID = SessionId("s-cache-clock")
CLOCK_A = '{"tier":"5m","sent_at":100.0}'
CLOCK_B = '{"tier":"1h","sent_at":200.0}'


def _session(backend, scope, session_id: SessionId = SID):
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=session_id, source="cli"))
    return store


def test_fresh_session_reads_no_clock_no_fallback_and_generation_zero(backend, scope_a):
    store = _session(backend, scope_a)
    # CONTRACT: a known session with no clock, no cache usage and no archived rows reads the empty clock record.
    assert store.usage.prompt_cache_clock(SID) == PromptCacheClockRecord(clock=None, last_assistant_at=None, generation=0)
    assert store.usage.prompt_cache_generation(SID) == 0
    assert store.sessions.get(SID).prompt_cache_clock is None


def test_unknown_session_reads_none_and_generation_zero(backend, scope_a):
    store = backend.open(scope_a)
    # CONTRACT: an unknown session has no clock record (a typed None, as the other usage reads), and counts no archived rows.
    assert store.usage.prompt_cache_clock(SessionId("s-nope")) is None
    assert store.usage.prompt_cache_generation(SessionId("s-nope")) == 0


def test_clock_round_trips_through_queue_usage_and_the_session_record(backend, scope_a):
    store = _session(backend, scope_a)
    store.usage.queue_usage(SID, UsageDelta(tokens=TokenCounts(input_tokens=3), api_call_count=1, prompt_cache_clock=CLOCK_A))
    # CONTRACT: the delta's clock is stored as sent and read back by the clock read and the session record.
    assert store.usage.prompt_cache_clock(SID).clock == CLOCK_A
    record = store.sessions.get(SID)
    assert record.prompt_cache_clock == CLOCK_A
    # CONTRACT: the row dict carries the clock as the column after ``transcript_revision``.
    assert SESSION_ROW_COLUMNS[-2:] == ("transcript_revision", "prompt_cache_clock")
    assert record.as_session_row()["prompt_cache_clock"] == CLOCK_A


def test_deltas_with_different_clocks_apply_in_order_and_keep_the_last(backend, scope_a):
    store = _session(backend, scope_a)
    for clock in (CLOCK_A, CLOCK_B):
        store.usage.queue_usage(SID, UsageDelta(tokens=TokenCounts(input_tokens=2), api_call_count=1, prompt_cache_clock=clock))
    # CONTRACT: deltas carrying different clocks apply in order, never merged; the last applied clock is stored and every counter still sums.
    assert store.usage.prompt_cache_clock(SID).clock == CLOCK_B
    record = store.sessions.get(SID)
    assert (record.prompt_cache_clock, record.input_tokens, record.api_call_count) == (CLOCK_B, 4, 2)


def test_a_delta_without_a_clock_keeps_the_stored_clock(backend, scope_a):
    store = _session(backend, scope_a)
    store.usage.queue_usage(SID, UsageDelta(api_call_count=1, prompt_cache_clock=CLOCK_A))
    store.usage.queue_usage(SID, UsageDelta(tokens=TokenCounts(output_tokens=5), api_call_count=1))
    # CONTRACT: a delta with no clock leaves the stored clock alone.
    assert store.usage.prompt_cache_clock(SID).clock == CLOCK_A
    assert store.sessions.get(SID).output_tokens == 5


def _assistant_rows(store) -> None:
    store.messages.append(SID, NewMessage("user", "hello", timestamp=100.0))
    store.messages.append(SID, NewMessage("assistant", "first", timestamp=150.0))
    store.messages.append(SID, NewMessage("assistant", "second", timestamp=200.0))
    store.messages.append(SID, NewMessage("user", "later", timestamp=300.0))


def test_resume_fallback_is_absent_while_cache_counters_are_zero(backend, scope_a):
    store = _session(backend, scope_a)
    _assistant_rows(store)
    store.usage.queue_usage(SID, UsageDelta(tokens=TokenCounts(input_tokens=10, output_tokens=4), api_call_count=1))
    # CONTRACT: with no cache read or write accounted, the newest assistant row is not offered as a fallback.
    assert store.usage.prompt_cache_clock(SID).last_assistant_at is None


@pytest.mark.parametrize("tokens", [TokenCounts(cache_read_tokens=1), TokenCounts(cache_write_tokens=1)], ids=["read", "write"])
def test_resume_fallback_is_the_newest_active_assistant_row_once_the_cache_was_used(backend, scope_a, tokens):
    store = _session(backend, scope_a)
    _assistant_rows(store)
    rewound = store.messages.append(SID, NewMessage("assistant", "rewound", timestamp=400.0))
    assert store.messages.deactivate_message(SID, rewound)
    store.usage.queue_usage(SID, UsageDelta(tokens=tokens, api_call_count=1))
    clock = store.usage.prompt_cache_clock(SID)
    # CONTRACT: once cache read plus write tokens are non-zero, the fallback is the newest active assistant row's time; user and inactive rows never count.
    assert clock.last_assistant_at == pytest.approx(200.0)
    assert clock.clock is None


def test_generation_counts_archived_compacted_rows_only(backend, scope_a):
    store = _session(backend, scope_a)
    _assistant_rows(store)
    rewound = store.messages.append(SID, NewMessage("assistant", "rewound", timestamp=400.0))
    assert store.messages.deactivate_message(SID, rewound)
    # CONTRACT: a rewound row is inactive but not compacted, so it does not move the generation.
    assert store.usage.prompt_cache_generation(SID) == 0
    store.messages.compact(SID, [NewMessage("user", "summary", timestamp=500.0)])
    # CONTRACT: compaction archives the four active rows; the generation counts them, through both reads.
    assert store.usage.prompt_cache_generation(SID) == 4
    assert store.usage.prompt_cache_clock(SID).generation == 4
    store.messages.compact(SID, [NewMessage("user", "summary two", timestamp=600.0)])
    # CONTRACT: the generation only grows: a second compaction adds the summary row it archived.
    assert store.usage.prompt_cache_generation(SID) == 5


def test_clocks_are_scope_isolated(backend, scope_a, scope_b):
    store_a = _session(backend, scope_a)
    store_b = _session(backend, scope_b)
    store_a.usage.queue_usage(SID, UsageDelta(api_call_count=1, prompt_cache_clock=CLOCK_A))
    # CONTRACT: one profile's clock never reaches another profile's session of the same id.
    assert store_b.usage.prompt_cache_clock(SID).clock is None
    assert store_a.usage.prompt_cache_clock(SID).clock == CLOCK_A
