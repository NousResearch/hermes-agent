"""Regressions for the auxiliary-client timeout close path.

The Codex Responses adapter enforces ``total_timeout`` with a
``threading.Timer`` whose callback runs on a stranger thread relative to any
in-flight worker using the process-shared OpenAI client. That callback used to
call ``client.close()`` directly — releasing TLS socket FDs while an owning
worker's SSL BIO still cached the raw integer. The kernel recycled the FD into
the next ``open()`` (e.g. the kanban dispatcher opening ``kanban.db``), and
the worker's delayed unwind flushed a 24-byte TLS application-data record over
SQLite header bytes 5..28 — the same corruption class as #29507/#70773.

The fix routes the timer callback through the FD-safe abort primitive
(``force_close_tcp_sockets``: shutdown-only, no FD release) instead of
``close()``, then evicts the cached client entry (#23432) so the next
auxiliary call does not reuse the aborted transport.

The tests below run at object granularity: no network, no real sockets.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Shared fakes: mimic the httpcore-1 layout force_close_tcp_sockets walks.
# ---------------------------------------------------------------------------


class _FakeSocket:
    """Records shutdown/close calls without touching real FDs."""

    def __init__(self):
        self.shutdown_calls = 0
        self.close_calls = 0

    def shutdown(self, _how):
        self.shutdown_calls += 1

    def close(self):
        self.close_calls += 1


def _build_fake_client(sock):
    """Build a fake shared OpenAI client with one pooled socket."""
    stream = SimpleNamespace(_sock=sock)
    http11 = SimpleNamespace(_network_stream=stream)
    pool_entry = SimpleNamespace(_connection=http11)
    pool = SimpleNamespace(_connections=[pool_entry])
    transport = SimpleNamespace(_transport=SimpleNamespace(_pool=pool)) if False else None
    # Keep the nesting shape identical to tests/run_agent/test_tls_fd_recycle_corruption.py:
    # client -> _client -> _transport -> _pool -> _connections[*] -> _connection -> _network_stream -> _sock
    http_client = SimpleNamespace(_transport=SimpleNamespace(_pool=pool))
    return SimpleNamespace(
        _client=http_client,
        responses=_FakeResponses(),
    )


class _FakeResponses:
    """Responses API stub whose stream can be made to hang."""

    def __init__(self, hang_seconds: float = 30.0):
        self._hang_seconds = hang_seconds
        self.stream_kwargs = None

    def create(self, **kwargs):
        self.stream_kwargs = kwargs
        return _FakeHangingStream(self._hang_seconds)


class _FakeHangingStream:
    """Event stream that blocks like a stalled provider connection."""

    def __init__(self, hang_seconds: float):
        self._hang_seconds = hang_seconds
        self.closed = False

    def __iter__(self):
        deadline = time.monotonic() + self._hang_seconds
        while time.monotonic() < deadline:
            time.sleep(0.02)
            yield SimpleNamespace(type="response.in_progress")
        yield SimpleNamespace(type="response.failed")

    def close(self):
        self.closed = True

    def get_final_response(self):  # pragma: no cover — consumer never gets here on timeout
        return None


def _make_adapter(client):
    from agent.auxiliary_client import _CodexCompletionsAdapter

    return _CodexCompletionsAdapter(client, "gpt-5.5")


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    """Poll predicate until truthy or timeout (timer thread may lag the raise)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# Test 1: the timer thread must NOT call client.close().
# ---------------------------------------------------------------------------


def test_timeout_timer_never_closes_shared_client():
    """Timeout fires on a hung stream → sockets shut down, FDs NOT released.

    This is the smoking-gun guarantee for the auxiliary timeout path. If a
    future refactor reintroduces ``self._client.close()`` in the timer
    callback, the kanban-db TLS-header corruption (#29507 class) re-opens.
    """
    sock = _FakeSocket()
    client = _build_fake_client(sock)
    client.close = lambda: setattr(sock, "close_calls", sock.close_calls + 1000)

    adapter = _make_adapter(client)

    started = time.monotonic()
    try:
        adapter.create(messages=[{"role": "user", "content": "summarize"}], timeout=0.15)
    except TimeoutError:
        pass
    elapsed = time.monotonic() - started

    # The timer enforced the timeout quickly — not the stream's 30s hang.
    assert elapsed < 5.0, f"timeout took {elapsed:.2f}s; timer did not fire"
    # The sweep may run on the timer thread slightly after the owner raises
    # (the owner's own deadline check can win the race); wait for it.
    assert _wait_for(lambda: sock.shutdown_calls >= 1), (
        "socket abort (shutdown) must run so the timeout lands"
    )
    # And critically: the FD was NEVER released.
    assert sock.close_calls == 0, (
        "client.close() must NEVER run from the timeout timer thread — releasing "
        "the FD here is the race that wrote TLS bytes into kanban.db"
    )


# ---------------------------------------------------------------------------
# Test 2: cache eviction still runs after the socket abort.
# ---------------------------------------------------------------------------


def test_timeout_still_evicts_cached_client_instance(monkeypatch):
    """After the abort, the cached wrapper must be dropped (#23432 contract).

    Otherwise the next auxiliary call reuses the aborted transport and fails
    fast with a connection error until restart.
    """
    import agent.auxiliary_client as ac

    evicted = []
    monkeypatch.setattr(ac, "_evict_cached_client_instance", lambda c: evicted.append(c))

    sock = _FakeSocket()
    client = _build_fake_client(sock)
    adapter = _make_adapter(client)

    try:
        adapter.create(messages=[{"role": "user", "content": "summarize"}], timeout=0.15)
    except TimeoutError:
        pass

    assert evicted == [client], "cache eviction must still fire after the socket abort"


# ---------------------------------------------------------------------------
# Test 3: explicit owner cancellation keeps its existing safe behaviour.
# ---------------------------------------------------------------------------


def test_owner_cancelled_attempt_does_not_touch_shared_client_sockets():
    """Owner hard-cancelled → neither close nor socket sweep may run.

    The pre-existing contract (comment: 'The request owner already
    hard-cancelled this attempt... closing/evicting it here would disrupt
    unrelated sessions') must survive the refactor untouched.
    """
    sock = _FakeSocket()
    client = _build_fake_client(sock)
    client.close = lambda: setattr(sock, "close_calls", sock.close_calls + 1000)

    # Install interrupt protection with an already-cancelled source so the
    # adapter's attempt-local decision object resolves "cancelled".
    import agent.auxiliary_client as ac

    cancelled_event = threading.Event()
    cancelled_event.set()
    with ac.aux_interrupt_protection(active=True, cancel_event=cancelled_event):
        adapter = _make_adapter(client)
        try:
            adapter.create(messages=[{"role": "user", "content": "summarize"}], timeout=0.15)
        except BaseException:
            # AuxiliaryExplicitCancellation deliberately derives from
            # BaseException (frozen host-cancel signal); swallow anything.
            pass  # any raise is fine; we only care about side effects

    # No FD release, no shutdown — the cancelled-attempt branch returns early.
    assert sock.close_calls == 0
    assert sock.shutdown_calls == 0


# ---------------------------------------------------------------------------
# Test 4: end-to-end at object granularity — the exact reporter timeline.
# ---------------------------------------------------------------------------


def test_fd_recycle_window_closed_in_auxiliary_timeout_path():
    """Stranger-thread timer + hung worker → FD survives until owner unwinds.

    Simulates: worker thread blocked consuming the event stream, timer fires
    on its own thread, then the worker unwinds. With the fix, the FD was
    never released by the stranger, so no recycle into another open() file
    can happen.
    """
    fd_released = {"yes": False}

    class _OwnedSocket(_FakeSocket):
        def close(self):
            fd_released["yes"] = True

    sock = _OwnedSocket()
    client = _build_fake_client(sock)
    adapter = _make_adapter(client)

    errors = []

    def worker():
        try:
            adapter.create(messages=[{"role": "user", "content": "summarize"}], timeout=0.15)
        except TimeoutError:
            pass
        except Exception as exc:  # pragma: no cover — surface unexpected failures
            errors.append(exc)

    t = threading.Thread(target=worker, name="simulated-aux-worker")
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), "worker hung past join timeout"
    assert not errors, f"unexpected worker errors: {errors!r}"

    assert fd_released["yes"] is False, (
        "the auxiliary timeout path released the socket FD from the timer "
        "thread — this is exactly the #29507/#70773 FD-recycle race"
    )
    assert sock.shutdown_calls >= 1
