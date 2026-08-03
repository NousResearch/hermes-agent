"""Yuanbao per-turn state cleanup: RecallGuard tracking dicts + member cache TTL.

Covers the salvage of PRs #23383 / #23384:

* ``_processing_msg_ids`` / ``_processing_msg_texts`` must be cleared when a
  turn finishes (they previously leaked forever, letting RecallGuard match a
  recall against an already-finished turn).
* The cleanup must pop ONLY when the finishing event's msg_id is truthy AND
  still owns the entry.  An id-less event (internal/synthetic message, push
  without msg_id) never wrote an entry, so it must never erase one either —
  the entry it sees belongs to a concurrently-queued id-bearing message whose
  drain task still needs it.
* ``_member_cache`` entries past ``MEMBER_CACHE_TTL_S`` must actually be
  evicted on read (the dict shrinks), while fresh entries survive.
"""
import asyncio
import time
from types import SimpleNamespace

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.yuanbao import MessageSender, YuanbaoAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _OutboundStub:
    async def start_slow_notifier(self, chat_id):  # noqa: ANN001
        pass

    def cancel_slow_notifier(self, chat_id):  # noqa: ANN001
        pass


def _bare_adapter():
    """YuanbaoAdapter instance without running its heavy __init__."""
    adapter = object.__new__(YuanbaoAdapter)
    adapter._outbound = _OutboundStub()
    adapter._processing_msg_ids = {}
    adapter._processing_msg_texts = {}
    return adapter


def _event(message_id):
    return SimpleNamespace(
        source=SimpleNamespace(chat_id="chat-1"),
        message_id=message_id,
    )


def _run_turn(monkeypatch, adapter, event, session_key, during_turn=None):
    """Run the yuanbao _process_message_background wrapper with the base
    class processing stubbed out (optionally mutating state mid-turn)."""

    async def _base_stub(self, ev, sk):  # noqa: ANN001
        if during_turn is not None:
            during_turn()

    monkeypatch.setattr(
        BasePlatformAdapter, "_process_message_background", _base_stub
    )
    asyncio.run(
        YuanbaoAdapter._process_message_background(adapter, event, session_key)
    )


# ---------------------------------------------------------------------------
# _processing_msg_ids / _processing_msg_texts cleanup (PR #23383)
# ---------------------------------------------------------------------------

def test_tracking_entries_cleared_after_normal_turn(monkeypatch):
    """A turn whose msg_id still owns the tracking entry clears it on exit."""
    adapter = _bare_adapter()
    sk = "yuanbao:group:G:user:U"
    # _dispatch_inbound_event wrote these before handle_message.
    adapter._processing_msg_ids[sk] = "m1"
    adapter._processing_msg_texts[sk] = "hello"

    _run_turn(monkeypatch, adapter, _event("m1"), sk)

    assert sk not in adapter._processing_msg_ids
    assert sk not in adapter._processing_msg_texts


def test_idless_event_must_not_erase_drain_tasks_entry(monkeypatch):
    """An id-less outer event finishing must NOT pop the tracking entry a
    concurrently-dispatched id-bearing message (queued as pending, to be
    handled by a drain task) wrote during the outer turn."""
    adapter = _bare_adapter()
    sk = "yuanbao:group:G:user:U"

    def _pending_message_arrives():
        # Simulates _dispatch_inbound_event for msg "m2" arriving while the
        # id-less event is still processing: it writes tracking state, then
        # handle_message routes it to _pending_messages for the drain task.
        adapter._processing_msg_ids[sk] = "m2"
        adapter._processing_msg_texts[sk] = "recallable text"

    _run_turn(
        monkeypatch, adapter, _event(None), sk,
        during_turn=_pending_message_arrives,
    )

    # The drain task for "m2" still needs these for RecallGuard matching.
    assert adapter._processing_msg_ids.get(sk) == "m2"
    assert adapter._processing_msg_texts.get(sk) == "recallable text"


def test_overwritten_entry_not_erased_by_outdated_turn(monkeypatch):
    """If a newer message already overwrote the entry, the older finishing
    turn must leave it alone (drain task owns it)."""
    adapter = _bare_adapter()
    sk = "yuanbao:group:G:user:U"
    adapter._processing_msg_ids[sk] = "m1"
    adapter._processing_msg_texts[sk] = "first"

    def _newer_message_arrives():
        adapter._processing_msg_ids[sk] = "m2"
        adapter._processing_msg_texts[sk] = "second"

    _run_turn(
        monkeypatch, adapter, _event("m1"), sk,
        during_turn=_newer_message_arrives,
    )

    assert adapter._processing_msg_ids.get(sk) == "m2"
    assert adapter._processing_msg_texts.get(sk) == "second"


# ---------------------------------------------------------------------------
# _member_cache TTL eviction (PR #23384)
# ---------------------------------------------------------------------------

def _bare_sender(adapter_stub):
    sender = object.__new__(MessageSender)
    sender._adapter = adapter_stub
    return sender


def test_member_cache_expired_entry_is_evicted():
    """Reading an expired entry must delete it — the cache dict shrinks."""
    now = time.time()
    adapter = SimpleNamespace(
        MEMBER_CACHE_TTL_S=300.0,
        _member_cache={
            "g-stale": (now - 301.0, [{"nickname": "bob", "user_id": "u1"}]),
        },
    )
    sender = _bare_sender(adapter)

    body = sender._build_msg_body_with_mentions("hi @bob", "g-stale")

    # Expired ⇒ no member data ⇒ plain text body, and the key is GONE.
    assert body == [{"msg_type": "TIMTextElem", "msg_content": {"text": "hi @bob"}}]
    assert "g-stale" not in adapter._member_cache
    assert len(adapter._member_cache) == 0


def test_member_cache_fresh_entry_survives_read():
    """A fresh entry is used for mention resolution and stays cached."""
    now = time.time()
    members = [{"nickname": "bob", "user_id": "u1"}]
    adapter = SimpleNamespace(
        MEMBER_CACHE_TTL_S=300.0,
        _member_cache={"g-fresh": (now - 10.0, members)},
    )
    sender = _bare_sender(adapter)

    body = sender._build_msg_body_with_mentions("hi @bob", "g-fresh")

    assert "g-fresh" in adapter._member_cache
    # Fresh members were actually used: an @mention element is present.
    assert any(el.get("msg_type") == "TIMCustomElem" for el in body)
