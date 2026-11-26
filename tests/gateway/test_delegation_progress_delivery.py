"""Progress events must never touch the durable completion path.

``delegation_progress`` rides the same shared queue as ``async_delegation`` results. These
tests pin the separation: its own type name, no delivery claim/row, no parent-turn wake, and
no fall-through into the process-completion formatter that CLI/TUI drains use.
"""

import asyncio
import queue

import pytest

from gateway.config import Platform
from gateway.run import _drain_gateway_watch_events
from gateway.delegation_progress import deliver_delegation_progress
from gateway.run_notifications import GatewayNotificationsMixin, _process_event_turn_origin
from gateway.session import SessionSource
from tools import async_delegation as ad
from tools.process_registry_notifications import format_process_notification


def _evt(**over):
    evt = {
        "type": ad.DELEGATION_PROGRESS_EVENT,
        "delegation_id": "d1",
        "session_key": "agent:main:telegram:dm:42",
        "parent_session_id": "sess-42",
        "goal": "Investigate the flaky retry test",
        "minutes": 6,
        "done": 2,
        "total": 4,
        "detail": "browser_exec",
        "text": "⏳ Background task \"Investigate the flaky retry test\" — 6 min elapsed, 2/4 subtasks done. "
                "Latest: browser_exec",
    }
    evt.update(over)
    return evt


# ── Naming / completion-path isolation ──────────────────────────────────────
def test_progress_type_is_not_a_completion_alias():
    evt = _evt()

    assert not evt["type"].startswith("async_delegation")  # every claim/dedup branch keys on that prefix
    assert not ad.is_interim_delegation_event(evt)
    assert ad.claim_event_delivery(evt, "test") == ""  # non-durable: no ledger row claimed
    assert GatewayNotificationsMixin._completion_delivery_identity(evt) is None
    assert _process_event_turn_origin(evt) == "process_notification"


def test_process_notification_formatter_never_renders_progress():
    """A CLI/TUI drain must drop it, not turn it into a phantom 'process exited' notice."""
    assert format_process_notification(_evt()) is None


def test_gateway_watch_drain_leaves_progress_for_the_delegation_watcher():
    q = queue.Queue()
    q.put(_evt())

    assert _drain_gateway_watch_events(q) == []
    assert q.qsize() == 1  # requeued, not swallowed by the watch drain
    assert q.get_nowait()["type"] == ad.DELEGATION_PROGRESS_EVENT


# ── Delivery ────────────────────────────────────────────────────────────────
class _Stub:
    def __init__(self, source):
        self.source = source
        self.sent = []

    def _build_process_event_source(self, _evt):
        return self.source

    async def _send_watcher_message(self, platform, chat_id, thread_id, text, watcher):
        self.sent.append((platform, chat_id, thread_id, text, watcher))


def _source(platform=Platform.TELEGRAM):
    return SessionSource(platform=platform, chat_id="42", chat_type="dm", thread_id="7")


def test_progress_is_sent_to_the_originating_chat_without_claiming_the_result(monkeypatch):
    monkeypatch.setattr(ad, "claim_completion_delivery",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("progress must not claim")))
    monkeypatch.setattr(ad, "complete_event_delivery",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("progress must not settle")))
    stub = _Stub(_source())

    asyncio.run(deliver_delegation_progress(stub, _evt()))

    assert len(stub.sent) == 1
    platform, chat_id, thread_id, text, watcher = stub.sent[0]
    assert (platform, chat_id, thread_id) == ("telegram", "42", "7")
    assert "2/4" in text
    assert watcher["type"] == ad.DELEGATION_PROGRESS_EVENT


def test_progress_skips_the_api_server_route_that_needs_a_delivery_row():
    stub = _Stub(_source(Platform.API_SERVER))

    asyncio.run(deliver_delegation_progress(stub, _evt()))

    assert stub.sent == []


def test_progress_with_no_resolvable_route_is_dropped_quietly():
    stub = _Stub(None)

    asyncio.run(deliver_delegation_progress(stub, _evt()))

    assert stub.sent == []


def test_empty_text_is_not_sent():
    stub = _Stub(_source())

    asyncio.run(deliver_delegation_progress(stub, _evt(text="")))

    assert stub.sent == []
