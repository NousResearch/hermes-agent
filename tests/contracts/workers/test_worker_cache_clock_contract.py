"""Contract cases for the prompt-cache clock a worker's ``usage.main`` receipt carries: the owner stores it with its durable context generation stamped in, a clock that is not a JSON object is stored as sent, and a report without a clock keeps the stored one."""

from __future__ import annotations

import json

from persistence.conversation.messages import NewMessage
from tests.contracts.workers._support import SESSION, ident, last_sequence, refused, register, seed

CLOCK = json.dumps({"tier": "5m", "sent_at": 100.0, "generation": 0})


def _worker_after_compaction(backend, scope):
    """A registered worker on :data:`SESSION` whose transcript had three rows archived by a compaction before registration."""
    epoch = seed(backend, scope)
    store = backend.open(scope)
    for i in range(3):
        store.messages.append(SESSION, NewMessage("user" if i % 2 == 0 else "assistant", f"turn {i}", timestamp=100.0 + i))
    store.messages.compact(SESSION, [NewMessage("user", "summary", timestamp=200.0)])
    assert store.usage.prompt_cache_generation(SESSION) == 3
    register(backend, scope, epoch)
    return backend.open(scope), epoch


def test_usage_main_accepts_the_clock_and_stamps_the_owner_generation(backend, scope_a):
    store, epoch = _worker_after_compaction(backend, scope_a)
    store.worker_executions.record_usage(**ident(epoch), input_tokens=1, api_call_count=1, prompt_cache_clock=CLOCK)
    stored = store.usage.prompt_cache_clock(SESSION).clock
    # CONTRACT: the receipt is accepted and the stored clock is the worker's event with ``generation`` set to the owner's count at commit.
    assert json.loads(stored) == {"tier": "5m", "sent_at": 100.0, "generation": 3}
    assert store.sessions.get(SESSION).prompt_cache_clock == stored
    assert last_sequence(backend, scope_a, epoch) == 1
    # CONTRACT: the receipt digest covers the payload as sent, so the same report replays to the stored result.
    store.worker_executions.record_usage(**ident(epoch), input_tokens=1, api_call_count=1, prompt_cache_clock=CLOCK)
    assert store.sessions.get(SESSION).input_tokens == 1


def test_usage_main_stores_a_clock_that_is_not_a_json_object_as_sent(backend, scope_a):
    store, epoch = _worker_after_compaction(backend, scope_a)
    for sequence, clock in enumerate(("not json", "[1, 2]"), start=1):
        store.worker_executions.record_usage(**ident(epoch, sequence=sequence), input_tokens=1, prompt_cache_clock=clock)
        # CONTRACT: a clock the owner cannot read as a JSON object is stored unchanged.
        assert store.usage.prompt_cache_clock(SESSION).clock == clock


def test_usage_main_without_a_clock_keeps_the_stored_clock(backend, scope_a):
    store, epoch = _worker_after_compaction(backend, scope_a)
    store.worker_executions.record_usage(**ident(epoch), input_tokens=1, prompt_cache_clock=CLOCK)
    stored = store.usage.prompt_cache_clock(SESSION).clock
    store.worker_executions.record_usage(**ident(epoch, sequence=2), input_tokens=1, prompt_cache_clock=None)
    store.worker_executions.record_usage(**ident(epoch, sequence=3), input_tokens=1)
    # CONTRACT: a report with no clock, or a ``None`` clock, leaves the stored clock alone.
    assert store.usage.prompt_cache_clock(SESSION).clock == stored


def test_usage_main_refuses_an_oversized_or_non_text_clock(backend, scope_a):
    store, epoch = _worker_after_compaction(backend, scope_a)
    for bad in ("x" * 4097, 7):
        with refused("invalid_params"):
            store.worker_executions.record_usage(**ident(epoch), input_tokens=1, prompt_cache_clock=bad)
    # CONTRACT: the clock is bounded as every text key of the report; a refusal commits nothing.
    assert last_sequence(backend, scope_a, epoch) == 0
    assert store.usage.prompt_cache_clock(SESSION).clock is None
