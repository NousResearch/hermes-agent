"""Regressions for the post-terminal Codex drain closing its stream from a stranger thread.

``run_codex_stream`` drains the SSE stream on a ``codex-post-terminal-drain`` daemon thread after
``final`` is assembled. When the drain budget expired, the owner thread used to ``close()`` the raw
stream and the Relay-managed wrapper while the drain thread was still blocked in ``SSL_read`` — the
FD-recycle race of #29507: ``close()`` releases the FD, the kernel recycles the integer into the next
``open()``, and the drain thread's TLS layer flushes a record into that unrelated file (it clobbered
a SQLite header).

Two invariants, both proven red on the unfixed code:

* ``test_close_ownership_contract`` — whichever way the drain ends (within budget, at the budget, never,
  no socket found, the wrapper owning its raw stream, a failing socket shutdown, a raising logging
  filter, an interrupt on the owner thread during the wait or inside ``Thread.start``, a thread that
  never launched), every stream is closed exactly once, only by the thread
  that iterated it, the socket is only ever ``shutdown()``, never ``close()``d, and the client's
  release/close (worker release, or agent close of the shared client) waits for that last reader too.
* ``test_real_httpx_stream_shutdown_wakes_reader_without_releasing_fd`` — on the real httpx/httpcore
  stream shape, the owner-side wake-up finds the socket and unblocks a reader without releasing its FD.
"""
from __future__ import annotations

import itertools
import logging
import socket as _socket
import threading
from types import SimpleNamespace

import pytest

import agent.codex_runtime as codex_runtime
from agent import relay_llm
from agent.client_lifecycle import ClientLifecycleMixin

_DRAIN_THREAD = "codex-post-terminal-drain"


class _FakeSocket:
    """Records shutdown/close/settimeout calls with the calling thread's name; touches no FD."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def settimeout(self, _value):
        self.calls.append(("settimeout", threading.current_thread().name))

    def shutdown(self, how):
        assert how == _socket.SHUT_RDWR
        self.calls.append(("shutdown", threading.current_thread().name))

    def close(self):  # pragma: no cover — must never run; asserted below
        self.calls.append(("close", threading.current_thread().name))

    def names(self):
        return [name for name, _thread in self.calls]


class _RaisingShutdownSocket(_FakeSocket):
    def shutdown(self, how):
        super().shutdown(how)
        raise RuntimeError("shutdown failed")


class _FakeNetworkStream:
    def __init__(self, sock):
        self._socket = sock

    def get_extra_info(self, key):
        return self._socket if key == "socket" else None


_CLOSE_SEQ = itertools.count()
_CLOSE_SEQ_LOCK = threading.Lock()


class _CloseRecorder:
    """Records the thread of every ``close()`` and a global sequence number (cross-thread ordering)."""

    def __init__(self):
        self.closes: list[str] = []
        self.order: list[int] = []

    def close(self):
        with _CLOSE_SEQ_LOCK:
            self.order.append(next(_CLOSE_SEQ))
            self.closes.append(threading.current_thread().name)


class _HeldOpenRawStream(_CloseRecorder):
    """Yields a completed response, then blocks (like an SSL_read) until ``release`` is set."""

    def __init__(self, release: threading.Event, sock: _FakeSocket | None):
        super().__init__()
        self._release = release
        self.blocked = threading.Event()
        extensions = {"network_stream": _FakeNetworkStream(sock)} if sock is not None else {}
        self.response = SimpleNamespace(extensions=extensions, is_closed=False)
        message = SimpleNamespace(type="message", status="completed",
                                  content=[SimpleNamespace(type="output_text", text="All done.")])
        usage = SimpleNamespace(input_tokens=10, output_tokens=6, total_tokens=16)
        self._events = [
            SimpleNamespace(type="response.output_item.done", item=message),
            SimpleNamespace(type="response.completed",
                            response=SimpleNamespace(status="completed", usage=usage, id="resp_drain_fd")),
        ]

    def __iter__(self):
        yield from self._events
        self.blocked.set()
        assert self._release.wait(5.0), "test never released the blocked drain"


class _ManagedWrapper(_CloseRecorder):
    """Stand-in for Relay's ``ManagedLlmStream``: a distinct object wrapping the raw stream."""

    final_response = None
    instances: list["_ManagedWrapper"] = []

    def __init__(self, request, stream_factory, *, on_stream_created=None, **_kwargs):
        super().__init__()
        self.raw = stream_factory(request)
        on_stream_created(self.raw)
        self._iter = iter(self.raw)
        _ManagedWrapper.instances.append(self)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


class _FaithfulWrapper(_ManagedWrapper):
    """Like Relay's real wrapper: ``close()`` closes the raw stream it owns and marks its response closed."""

    def close(self):
        super().close()
        self.raw.close()
        self.raw.response.is_closed = True


class _RaisingHandoffLogFilter(logging.Filter):
    """Raises on every hand-off log record (the WARNING and its DEBUG fallback alike)."""

    def filter(self, record):
        message = record.getMessage()
        if "post-terminal drain" in message or "stream remained open" in message:
            raise RuntimeError("logging filter failed")
        return True


class _InterruptingEvent(threading.Event):
    """``wait(timeout)`` raises like Ctrl-C landing on the owner thread during the drain budget."""

    def wait(self, timeout=None):
        if timeout is not None:
            raise KeyboardInterrupt
        return super().wait(timeout)


class _InterruptedStartThread(threading.Thread):
    """``start()`` launches the thread and then raises, like Ctrl-C landing inside ``Thread.start``."""

    def start(self):
        super().start()
        raise KeyboardInterrupt


class _FailingStartThread(threading.Thread):
    """``start()`` fails before any thread exists (CPython's ``RuntimeError: can't start new thread``)."""

    def start(self):
        raise RuntimeError("can't start new thread")


class _FakeClient(_CloseRecorder):
    """The request-local OpenAI client: ``close()`` here is the pool close that would release FDs."""

    base_url = "https://example.invalid/backend-api/codex"

    def __init__(self, raw):
        super().__init__()
        self.responses = SimpleNamespace(create=lambda **_kw: raw)


class _FakeAgent(ClientLifecycleMixin):
    """Real client lifecycle (hold / deferred release); everything else stubbed."""

    provider = "openai-codex"
    base_url = "https://example.invalid/backend-api/codex"
    model = "gpt-5-codex"
    session_id = "test-session"
    is_subagent = False
    _fallback_index = 0
    show_commentary = False
    interim_assistant_callback = None
    _current_api_request_id = None
    _interrupt_requested = False

    def _fire_stream_delta(self, _text): pass
    def _fire_reasoning_delta(self, *_a): pass
    def _fire_streamed_codex_commentary(self, *_a): pass
    def _touch_activity(self, _label): pass
    def _client_log_context(self): return "provider=test"
    def _abort_request_openai_client(self, *_a, **_k): pass


def _run(monkeypatch, *, budget: float, release: threading.Event, sock: _FakeSocket | None,
         wrapper_cls=_ManagedWrapper, expect_raise=None, implicit_client: bool = False, hold_raises=None):
    _ManagedWrapper.instances.clear()
    raw = _HeldOpenRawStream(release, sock)
    client, agent = _FakeClient(raw), _FakeAgent()
    if hold_raises is not None:
        # An interrupt on the owner thread before the drain thread is started (the hold is the last
        # statement before ``start()``): no reader can exist, so the owner must still own the close.
        def _interrupted_hold(_client):
            raise hold_raises
        agent._hold_openai_client = _interrupted_hold
    monkeypatch.setattr(relay_llm, "stream", wrapper_cls)
    monkeypatch.setattr(codex_runtime, "_stream_drain_timeout", lambda: budget)
    request = {"model": "gpt-5-codex", "input": [{"role": "user", "content": "Ping"}], "store": False}
    if implicit_client:
        agent.client = client  # the summary drain passes no client: the shared primary client is used
    client_arg = None if implicit_client else client
    if expect_raise is not None:
        with pytest.raises(expect_raise):
            codex_runtime.run_codex_stream(agent, request, client=client_arg)
    else:
        final = codex_runtime.run_codex_stream(agent, request, client=client_arg)
        assert final.status == "completed" and final.id == "resp_drain_fd"
    (wrapper,) = _ManagedWrapper.instances
    if implicit_client:
        # What agent close does right after a max-iteration summary (run_agent._drop_shared_client).
        agent._close_openai_client(client, reason="agent_close", shared=True)
    else:
        # What the request worker does right after run_codex_stream returns: release/close the request client.
        agent._close_request_openai_client(client, reason="request_complete")
    return raw, wrapper, client


def _drain_threads():
    return [t for t in threading.enumerate() if t.name == _DRAIN_THREAD]


def _join_drain_threads():
    for thread in _drain_threads():
        thread.join(5.0)
        assert not thread.is_alive()


@pytest.fixture(autouse=True)
def _no_lingering_drain_threads():
    yield
    _join_drain_threads()


# Scenario → budget, socket factory, wrapper class, expected exception, hand-off expected?
_HANDOFF = {
    # the drain outlives the budget: the owner thread shuts the socket down and hands the close over
    "timeout": dict(budget=0.05, sock=_FakeSocket, wrapper=_ManagedWrapper, raises=None, handoff=True),
    # same, streaming on the agent's shared client (no client argument, as the max-iteration summary does):
    # agent close must not close that client's pool while the drain thread still reads
    "implicit_client": dict(budget=0.05, sock=_FakeSocket, wrapper=_ManagedWrapper, raises=None, handoff=True,
                            implicit=True),
    # no socket under the stream: still hand off (a lingering daemon thread beats corruption) + WARNING
    "no_socket": dict(budget=0.05, sock=lambda: None, wrapper=_ManagedWrapper, raises=None, handoff=True),
    # a wrapper that closes its own raw stream (ManagedLlmStream does) must not have it closed again
    "wrapper_owns_raw": dict(budget=0.05, sock=_FakeSocket, wrapper=_FaithfulWrapper, raises=None, handoff=True),
    # failures after the hand-off never return the close to the owner thread
    "shutdown_raises": dict(budget=0.05, sock=_RaisingShutdownSocket, wrapper=_ManagedWrapper, raises=None, handoff=True),
    "logging_raises": dict(budget=0.05, sock=_FakeSocket, wrapper=_ManagedWrapper, raises=None, handoff=True),
    # an interrupt on the owner thread during the budget, or inside Thread.start() after the thread launched:
    # hand off, then re-raise
    "interrupt_during_wait": dict(budget=0.05, sock=_FakeSocket, wrapper=_ManagedWrapper, raises=KeyboardInterrupt, handoff=True),
    "interrupt_in_thread_start": dict(budget=0.05, sock=_FakeSocket, wrapper=_ManagedWrapper, raises=KeyboardInterrupt, handoff=True),
    # Thread.start() fails before any thread exists: no reader, so the owner still closes (as before)
    "thread_start_fails": dict(budget=0.05, sock=_FakeSocket, wrapper=_ManagedWrapper, raises=RuntimeError, handoff=False),
    # an interrupt before start() is even attempted (raised by the hold): no thread, the owner closes
    "interrupt_before_start": dict(budget=0.05, sock=_FakeSocket, wrapper=_ManagedWrapper, raises=KeyboardInterrupt,
                                   handoff=False, hold_raises=KeyboardInterrupt),
    # the drain ends inside the budget: no shutdown, the owner closes once (the wrapper owns the raw stream)
    "within_budget": dict(budget=5.0, sock=_FakeSocket, wrapper=_ManagedWrapper, raises=None, handoff=False),
    # budget 0: no drain thread at all, the owner closes once
    "budget_zero": dict(budget=0.0, sock=_FakeSocket, wrapper=_ManagedWrapper, raises=None, handoff=False),
}


@pytest.mark.parametrize("scenario", list(_HANDOFF) + ["race"])
def test_close_ownership_contract(monkeypatch, caplog, scenario):
    """Exactly one close per stream, only by the thread that iterated it; the socket is never closed."""
    owner = threading.current_thread().name
    if scenario == "race":
        # The drain ends exactly as the budget expires (50 tries): whichever side wins, exactly one close per
        # stream. A socket shutdown is the witness of a hand-off — then ONLY the drain thread may close.
        budget = 0.02
        for _ in range(50):
            release, sock = threading.Event(), _FakeSocket()
            timer = threading.Timer(budget, release.set)
            timer.start()
            try:
                raw, wrapper, client = _run(monkeypatch, budget=budget, release=release, sock=sock)
            finally:
                timer.join(5.0)
                release.set()
            _join_drain_threads()
            assert len(wrapper.closes) == 1 and len(client.closes) == 1
            if "shutdown" in sock.names():
                assert sock.names() == ["settimeout", "shutdown"]
                assert wrapper.closes == [_DRAIN_THREAD] and raw.closes == [_DRAIN_THREAD]
                # The client's pool close follows the drain thread's stream closes: on the drain thread when
                # the worker's release found the hold, on the owner thread when the drain had already let go.
                assert client.order[0] > max(wrapper.order[0], raw.order[0])
            else:
                assert wrapper.closes == [owner] and raw.closes == [] and client.closes == [owner]
            assert "close" not in sock.names()
        return

    spec = _HANDOFF[scenario]
    release, sock = threading.Event(), spec["sock"]()
    started: list = []
    if scenario == "budget_zero":
        real_thread = threading.Thread

        def _spy_thread(*args, **kwargs):
            started.append(kwargs.get("name"))
            return real_thread(*args, **kwargs)

        monkeypatch.setattr(codex_runtime.threading, "Thread", _spy_thread)
    if scenario == "within_budget":
        release.set()  # the provider closes right after the terminal event
    if scenario in ("interrupt_during_wait", "interrupt_in_thread_start", "thread_start_fails"):
        thread_cls = {"interrupt_during_wait": threading.Thread, "interrupt_in_thread_start": _InterruptedStartThread,
                      "thread_start_fails": _FailingStartThread}[scenario]
        event_cls = _InterruptingEvent if scenario == "interrupt_during_wait" else threading.Event
        proxy = SimpleNamespace(Thread=thread_cls, Lock=threading.Lock, Event=event_cls)
        monkeypatch.setattr(codex_runtime, "threading", proxy)
    log_filter = _RaisingHandoffLogFilter()
    if scenario == "logging_raises":
        codex_runtime.logger.addFilter(log_filter)
    try:
        # DEBUG enabled so the fallback diagnostics emit records too (a raising filter hits them as well).
        with caplog.at_level(logging.DEBUG, logger="agent.codex_runtime"):
            raw, wrapper, client = _run(monkeypatch, budget=spec["budget"], release=release, sock=sock,
                                        wrapper_cls=spec["wrapper"], expect_raise=spec["raises"],
                                        implicit_client=spec.get("implicit", False),
                                        hold_raises=spec.get("hold_raises"))
        if spec["handoff"]:
            # The iteration is still blocked: the owner thread closed nothing and woke the reader FD-safely;
            # the worker's client release is deferred (a pool close would release the reader's FD too).
            assert raw.blocked.wait(2.0)
            assert raw.closes == [] and wrapper.closes == [], (
                "close() from the owner thread while the drain thread iterates is the #29507 FD-recycle race")
            assert client.closes == [], "the request client was closed under the drain thread's reader"
            if sock is not None:
                assert sock.names() == ["settimeout", "shutdown"]
                assert {thread for _name, thread in sock.calls} == {owner}
            else:
                assert "stream remained open" in caplog.text and "drain thread will close" in caplog.text
    finally:
        codex_runtime.logger.removeFilter(log_filter)
        release.set()
    _join_drain_threads()
    if spec["handoff"]:
        assert wrapper.closes == [_DRAIN_THREAD]
        assert raw.closes == [_DRAIN_THREAD]  # closed once: by the drain thread, or by the wrapper on that thread
        assert client.closes == [_DRAIN_THREAD]  # the deferred worker release ran after the last read
    else:
        assert wrapper.closes == [owner]
        assert raw.closes == []  # unchanged: the managed wrapper owns the raw stream on this path
        assert client.closes == [owner]  # no reader outlived the call: the worker's release is immediate
        if scenario == "budget_zero":
            assert _DRAIN_THREAD not in started
    if scenario in ("thread_start_fails", "interrupt_before_start"):
        assert not raw.blocked.is_set()  # no drain thread ever iterated past the terminal event
    if sock is not None:
        assert "close" not in sock.names()
        if not spec["handoff"]:
            assert sock.calls == []


def test_real_httpx_stream_shutdown_wakes_reader_without_releasing_fd():
    """On the real httpx/httpcore stream shape the owner-side wake-up finds the socket, unblocks a reader
    stuck in ``recv`` and leaves the FD open for the reader to close (no network, no TLS)."""
    import httpx
    from httpcore._backends.sync import SyncStream

    ours, theirs = _socket.socketpair()
    try:
        response = httpx.Response(200, extensions={"network_stream": SyncStream(ours)})
        entering, woke = threading.Event(), threading.Event()
        received = {}

        def _reader():
            try:
                entering.set()  # event-based sync (AGENTS.md flake policy): the next statement blocks in recv
                received["data"] = ours.recv(1)
            except OSError as exc:  # pragma: no cover — a shut-down socket returns b"" rather than raising
                received["error"] = exc
            woke.set()

        reader = threading.Thread(target=_reader, name="blocked-reader", daemon=True)
        reader.start()
        assert entering.wait(2.0), "reader must reach recv"
        assert not woke.is_set(), "nothing was sent: the reader is blocked in recv"

        assert codex_runtime._shutdown_stream_socket(SimpleNamespace(response=response)) is True
        assert woke.wait(2.0), "SHUT_RDWR must wake the reader"
        assert received.get("data") == b"" and "error" not in received
        assert ours.fileno() >= 0, "the FD must still be owned by the reader (shutdown, never close)"
        reader.join(2.0)
    finally:
        ours.close()
        theirs.close()
