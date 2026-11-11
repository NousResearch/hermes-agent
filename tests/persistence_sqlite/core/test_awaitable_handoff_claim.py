"""Coroutines awaiting the same session's handoff claim through ``store.aio`` interleave at each await; the claim's
compare-and-set keeps exactly one winner."""

import asyncio

import pytest

from persistence.conversation.sessions import NewSession


@pytest.mark.asyncio
async def test_concurrent_claim_handoff_single_winner(backend, scope_a):
    store = backend.open(scope_a)
    sid = "s-handoff"
    await store.aio.sessions.create(NewSession(session_id=sid, source="test"))
    await store.aio.coordination.request_handoff(session_id=sid, platform="telegram")

    results = await asyncio.gather(*(store.aio.coordination.claim_handoff(session_id=sid) for _ in range(20)))

    assert sum(results) == 1, f"exactly one claim must win, got {sum(results)}"
