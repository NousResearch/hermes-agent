"""Regression tests: text debounce batching must never merge messages from
different authors.

Incident 2026-08-20 (WhatsApp group, shared session / group_sessions_per_user
false): user A sent a message; seconds later user B sent a short reply inside the
debounce window. Both landed on the same batch key, were concatenated, and the
gateway ingested the combined text under user A's tag — the agent then
addressed A about a line B wrote. Batching keys on the SESSION, so shared
group sessions make cross-author collisions routine.
"""

import asyncio

import pytest

from gateway.platforms.helpers import TextBatchAggregator, batch_authors_match
from gateway.platforms.base import MessageEvent
from gateway.session import Platform, SessionSource


def _event(text: str, user_id: str, user_name: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="120363000000000000@g.us",
            chat_type="group",
            user_id=user_id,
            user_name=user_name,
        ),
    )


def test_batch_authors_match_same_and_different():
    a1 = _event("hola", "111@lid", "Alice")
    a2 = _event("sigo yo", "111@lid", "Alice")
    b = _event("quick interjection", "222@lid", "Bruna")
    assert batch_authors_match(a1, a2)
    assert not batch_authors_match(a1, b)


def test_batch_authors_match_no_ids_falls_back_to_name():
    x = _event("a", None, "OnlyName")
    y = _event("b", None, "OnlyName")
    z = _event("c", None, "Other")
    assert batch_authors_match(x, y)
    assert not batch_authors_match(x, z)


@pytest.mark.asyncio
async def test_aggregator_never_merges_across_authors():
    """The exact incident: two authors, one shared session key."""
    dispatched = []

    async def handler(event):
        dispatched.append((event.source.user_name, event.text))

    agg = TextBatchAggregator(handler=handler, batch_delay=0.05, split_delay=0.05)
    key = "agent:main:whatsapp:group:120363000000000000@g.us"

    agg.enqueue(_event("first message line one", "111@lid", "Alice"), key)
    agg.enqueue(_event("first message line two", "111@lid", "Alice"), key)
    agg.enqueue(_event("quick interjection", "222@lid", "Bruna"), key)
    await asyncio.sleep(0.3)

    assert len(dispatched) == 2, dispatched
    by_author = dict(dispatched)
    assert by_author["Alice"] == (
        "first message line one\nfirst message line two"
    )
    assert by_author["Bruna"] == "quick interjection"


@pytest.mark.asyncio
async def test_aggregator_still_merges_same_author():
    dispatched = []

    async def handler(event):
        dispatched.append(event.text)

    agg = TextBatchAggregator(handler=handler, batch_delay=0.05, split_delay=0.05)
    agg.enqueue(_event("linea 1", "111@lid", "Alice"), "k")
    agg.enqueue(_event("linea 2", "111@lid", "Alice"), "k")
    await asyncio.sleep(0.2)
    assert dispatched == ["linea 1\nlinea 2"]


@pytest.mark.asyncio
async def test_whatsapp_adapter_enqueue_flushes_on_author_change():
    """Same guard in the WhatsApp adapter's own implementation."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.05
    adapter._text_batch_split_delay_seconds = 0.05
    dispatched = []

    async def fake_handle(event):
        dispatched.append((event.source.user_name, event.text))

    adapter.handle_message = fake_handle  # type: ignore[method-assign]
    adapter._text_batch_key = lambda e: "sharedkey"  # type: ignore[method-assign]

    adapter._enqueue_text_event(_event("text from alice", "111@lid", "Alice"))
    adapter._enqueue_text_event(_event("text from bruna", "222@lid", "Bruna"))
    await asyncio.sleep(0.3)

    assert ("Alice", "text from alice") in dispatched
    assert ("Bruna", "text from bruna") in dispatched
    assert len(dispatched) == 2
