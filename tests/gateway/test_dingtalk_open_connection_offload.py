"""#144992 — the DingTalk Stream handshake must not block the gateway event loop."""

import threading
import time
from types import SimpleNamespace

from plugins.platforms.dingtalk.adapter import _offload_open_connection


def _client(fetch):
    client = SimpleNamespace(open_connection=fetch)
    _offload_open_connection(client)
    return client


def test_fast_handshake_returns_the_ticket_on_the_first_call():
    client = _client(lambda: {"endpoint": "wss://x", "ticket": "t"})
    assert client.open_connection() == {"endpoint": "wss://x", "ticket": "t"}


def test_stalled_handshake_returns_none_quickly_and_is_reused_not_respawned():
    release, calls = threading.Event(), []

    def fetch():
        calls.append(1)
        release.wait(30)
        return {"ticket": "late"}

    client = _client(fetch)
    started = time.monotonic()
    assert client.open_connection() is None
    assert client.open_connection() is None
    # Two attempts, bounded wait each: the loop is never held for the stalled request.
    assert time.monotonic() - started < 2.0
    assert len(calls) == 1

    release.set()
    deadline = time.monotonic() + 5
    ticket = None
    while ticket is None and time.monotonic() < deadline:
        ticket = client.open_connection()
    assert ticket == {"ticket": "late"}
    assert len(calls) == 1


def test_each_reconnect_after_a_ticket_fetches_a_fresh_one():
    tickets = iter([{"ticket": "a"}, {"ticket": "b"}])
    client = _client(lambda: next(tickets))
    assert client.open_connection() == {"ticket": "a"}
    assert client.open_connection() == {"ticket": "b"}


def test_failed_handshake_passes_none_through():
    client = _client(lambda: None)
    assert client.open_connection() is None
