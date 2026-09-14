"""Regression: the timeout watchdog must not release the client's FDs.

Salvaged from PR #72260 (@necoweb3 / dsad), adapted to the re-armable
no-progress watchdog introduced in PR #99660.

``close()`` from a thread that does not own the in-flight httpx connection
releases the raw TLS fd while the owner's OpenSSL BIO still caches that
integer. The kernel can hand the same integer to the next ``open()`` in the
process — a SessionDB or kanban.db handle — and the owner's unwinding TLS
flush then writes an application-data record into that database file
(#29507 / #67142 / #70773). ``shutdown()`` is FD-safe from any thread;
``close()`` is not, so it belongs to the owning thread.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.auxiliary_client import _CodexCompletionsAdapter


def _adapter_with_recording_client(stream):
    """Build an adapter whose client records (action, thread) events.

    The nested ``_client._transport._pool._connections`` shape is what
    ``force_close_tcp_sockets`` traverses.
    """
    events = []

    class _Sock:
        def shutdown(self, how):
            events.append(("shutdown", threading.get_ident()))

        def close(self):
            events.append(("sock.close", threading.get_ident()))

    sock = _Sock()

    class _Conn:
        def __init__(self):
            self._connection = self
            self._network_stream = self

        def get_extra_info(self, name):
            return sock if name == "socket" else None

    class _Client:
        def __init__(self):
            self._transport = SimpleNamespace(
                _pool=SimpleNamespace(_connections=[_Conn()])
            )

    class _LeafClient:
        def __init__(self):
            self._client = _Client()
            self.base_url = "https://chatgpt.com/backend-api/codex"
            self.responses = SimpleNamespace(create=lambda **kw: stream)

        def close(self):
            events.append(("client.close", threading.get_ident()))

    return _CodexCompletionsAdapter(_LeafClient(), "gpt-5.5"), events


class TestCodexAuxiliaryTimeoutFdOwnership:
    def test_stalled_stream_timeout_shuts_down_from_timer_and_closes_from_owner(self):
        """The watchdog Timer fires on a stalled stream: it may only
        shutdown(); the real close() must land on the owning thread in the
        adapter's ``finally``."""

        def _stalled():
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                time.sleep(0.02)
                yield SimpleNamespace(type="response.in_progress")

        adapter, events = _adapter_with_recording_client(_stalled())
        owner_tid = threading.get_ident()

        def _consume(stream, *, model, on_event):
            del model
            for event in stream:
                on_event(event)
            return SimpleNamespace(output=[], usage=None)

        with (
            patch("agent.auxiliary_client._AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS", 0.3),
            patch("agent.auxiliary_client._evict_cached_client_instance"),
            patch("agent.codex_runtime._consume_codex_event_stream", _consume),
            pytest.raises(TimeoutError),
        ):
            adapter.create(
                messages=[{"role": "user", "content": "summarize"}],
                timeout=300,
            )

        # Give the daemon Timer thread a beat to finish its callback.
        time.sleep(0.2)
        actions = [a for a, _ in events]
        # Stranger thread (Timer) only shut the sockets down.
        shutdown_tids = {tid for a, tid in events if a == "shutdown"}
        assert "shutdown" in actions, events
        assert owner_tid not in shutdown_tids, "shutdown ran on owner thread"
        # close() from a stranger thread is the corruption vector — banned.
        stranger_closes = [
            (a, tid) for a, tid in events
            if a in {"client.close", "sock.close"} and tid != owner_tid
        ]
        assert not stranger_closes, f"stranger-thread FD release: {stranger_closes}"
        # The owning thread released the FDs on unwind.
        assert ("client.close", owner_tid) in events, events

    def test_owner_thread_deadline_hit_closes_directly(self):
        """When the OWNING thread detects the deadline in _check_cancelled,
        it may close() directly — no shutdown-only detour required."""

        def _one_keepalive_then_block():
            yield SimpleNamespace(type="response.in_progress")
            time.sleep(1.0)  # past the patched window; owner detects on next event
            yield SimpleNamespace(type="response.in_progress")

        adapter, events = _adapter_with_recording_client(_one_keepalive_then_block())
        owner_tid = threading.get_ident()

        def _consume(stream, *, model, on_event):
            del model
            for event in stream:
                on_event(event)
            return SimpleNamespace(output=[], usage=None)

        # Cancel the watchdog race by making the Timer window huge relative
        # to the owner's per-event check: patch the timer to never fire.
        class _NeverTimer:
            def __init__(self, *_a, **_k):
                self.daemon = True

            def start(self):
                pass

            def cancel(self):
                pass

        with (
            patch("agent.auxiliary_client._AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS", 0.3),
            patch("agent.auxiliary_client._evict_cached_client_instance"),
            patch("agent.auxiliary_client.threading.Timer", _NeverTimer),
            patch("agent.codex_runtime._consume_codex_event_stream", _consume),
            pytest.raises(TimeoutError),
        ):
            adapter.create(
                messages=[{"role": "user", "content": "summarize"}],
                timeout=300,
            )

        stranger = [(a, t) for a, t in events if t != owner_tid]
        assert not stranger, f"non-owner activity: {stranger}"
        assert ("client.close", owner_tid) in events, events
