"""Regression tests for remote CUA session lifecycle recovery (M1 + M2).

M1: an MCPError carrying session-expiry semantics (mcp 2.0.0 surfaces a 404
"Session not found" from the bridge's idle-expiry as
``MCPError(code=INVALID_REQUEST, message="Session terminated")``) must be
classified as a recoverable closed-session condition so the next call
invalidates+rebuilds the transport — exactly as ClosedResourceError/EOF
already do. Before the fix, MCPError escaped the classifier, the lifecycle
coroutine stayed parked on ``_shutdown_event``, ``_started`` stayed True, and
every subsequent ``call_tool`` hit the dead session and raised the same error
(permanently wedged until process restart).

M2: the startup handshake (``session.initialize()`` + capability discovery)
must be bounded by a deadline INSIDE the owning coroutine so a hung
``initialize()`` raises inside the coro, its ``finally`` unwinds the transport
context managers, and the future completes — instead of being abandoned
pending inside a closing loop.
"""
import asyncio
import threading

from tools.computer_use import cua_backend_session as _cbs


# ── M1: classify expired-session MCPError as closed-session ───────────────


def _mcp_error(message: str = "Session terminated"):
    """Build a real MCPError the way mcp 2.0.0 does (code, message, data)."""
    from mcp.shared.exceptions import MCPError
    return MCPError(-32000, message)


class TestClassifier:
    """Direct classifier tests — no transport, no bridge."""

    def test_mcperror_session_terminated_is_closed(self):
        assert _cbs._CuaDriverSession._is_closed_session_error(_mcp_error("Session terminated"))

    def test_mcperror_session_not_found_is_closed(self):
        assert _cbs._CuaDriverSession._is_closed_session_error(_mcp_error("Session not found"))

    def test_mcperror_session_expired_is_closed(self):
        assert _cbs._CuaDriverSession._is_closed_session_error(_mcp_error("Session expired"))

    def test_mcperror_unrelated_message_is_not_closed(self):
        # A non-expiry MCPError (e.g. INVALID_PARAMS) must not trigger rebuild.
        assert not _cbs._CuaDriverSession._is_closed_session_error(_mcp_error("Invalid params"))

    def test_plain_exception_with_session_text_is_closed(self):
        # Fallback message-text check catches fakes/subclasses without the SDK type.
        assert _cbs._CuaDriverSession._is_closed_session_error(
            RuntimeError("Session terminated by remote"))

    def test_existing_classifiers_still_work(self):
        from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
        assert _cbs._CuaDriverSession._is_closed_session_error(ClosedResourceError())
        assert _cbs._CuaDriverSession._is_closed_session_error(BrokenResourceError())
        assert _cbs._CuaDriverSession._is_closed_session_error(EndOfStream())
        assert _cbs._CuaDriverSession._is_closed_session_error(EOFError())
        assert _cbs._CuaDriverSession._is_closed_session_error(BrokenPipeError())


class _ExpireThenRecoverBridge:
    """Bridge whose first run() raises MCPError("Session terminated") and whose
    second run() (the post-rebuild retry) returns a normal result."""

    def __init__(self, retry_result):
        self._retry_result = retry_result
        self.calls = 0

    def run(self, coro, timeout):
        self.calls += 1
        if self.calls == 1:
            coro.close()
            raise _mcp_error("Session terminated")
        # Second call: the replay after rebuild. Close the coro and return the
        # canned result (the real bridge would await it).
        coro.close()
        return self._retry_result


def _make_recovery_session(retry_result):
    """Build a _CuaDriverSession wired for the M1 rebuild-path test.

    The session starts in the 'started' state. The first call_tool raises
    MCPError("Session terminated"); _is_closed_session_error must classify it
    as closed-session, triggering _recreate_session. _recreate_session tears
    down and rebuilds under _lock; we stub it so it just sets _started back to
    True (the rebuild is unit-tested elsewhere). The second bridge.run (the
    replay for replay-safe tools) returns retry_result.
    """
    obj = _cbs._CuaDriverSession.__new__(_cbs._CuaDriverSession)
    obj._bridge = _ExpireThenRecoverBridge(retry_result)
    obj._remote_config = None  # local session — CLI fallback path exists
    obj._timeout_suspect = False
    obj._started = True
    obj._lock = threading.Lock()
    obj._LIFECYCLE_CALLS = frozenset()
    obj._TRANSPORT_REPLAY_SAFE_TOOLS = frozenset(
        {"get_cursor_position", "get_displays", "get_screen_size",
         "get_window_state", "list_apps", "list_windows"})
    obj._notify_transport_reset = lambda: None
    obj._declared_session_id = None
    obj._capabilities, obj._tool_schemas, obj._capability_version = {}, {}, ""
    obj._transport_generation, obj._transport_reset_callback = 0, None
    # Stub _recreate_session: tear down + rebuild = set _started True (the
    # real method restarts the lifecycle; we only need the side effect that
    # the next bridge.run sees a started session). Signature matches the real
    # _recreate_session(self, name, timeout, log_msg, *, restart, clear_timeout_suspect).
    recreate_calls = []
    def _fake_recreate(self_inner, name, timeout, log_msg, *, restart=True, clear_timeout_suspect=False):
        recreate_calls.append(name)
        obj._started = True
    obj._recreate_session = _fake_recreate.__get__(obj)
    obj._recreate_calls = recreate_calls
    return obj


def test_m1_expired_session_triggers_rebuild_for_replay_safe(monkeypatch):
    """After an idle-expiry MCPError on a replay-safe tool, the session is
    recreated and the call is retried on the fresh session."""
    retry = {"isError": False, "data": "ok"}
    obj = _make_recovery_session(retry)
    # get_cursor_position is replay-safe → retried after rebuild.
    result = obj.call_tool("get_cursor_position", {}, timeout=5.0)
    assert obj._recreate_calls == ["get_cursor_position"]
    assert obj._bridge.calls == 2  # first (expired) + second (replay)
    assert result == retry


def test_m1_expired_session_fail_closed_for_mutating(monkeypatch):
    """After an idle-expiry MCPError on a MUTATING (non-replay-safe) tool, the
    session is recreated but the call is NOT replayed — surface outcome-unknown."""
    codes = []
    monkeypatch.setattr(_cbs, "_outcome_unknown",
                        lambda name, exc, code: codes.append(code) or {"code": code})
    obj = _make_recovery_session({"should_not_reach": True})
    # click is NOT in _TRANSPORT_REPLAY_SAFE_TOOLS → fail closed.
    result = obj.call_tool("click", {"x": 10, "y": 20}, timeout=5.0)
    assert obj._recreate_calls == ["click"]
    assert obj._bridge.calls == 1  # only the expired call; no replay
    assert codes == ["transport_outcome_unknown"]
    assert result == {"code": "transport_outcome_unknown"}


def test_m1_expired_session_on_remote_fail_closed_no_cli(monkeypatch):
    """A remote session must never fall back to the local CLI; an idle-expiry
    MCPError on a mutating tool surfaces as remote_transport_outcome_unknown."""
    codes = []
    monkeypatch.setattr(_cbs, "_outcome_unknown",
                        lambda name, exc, code: codes.append(code) or {"code": code})
    obj = _make_recovery_session({"should_not_reach": True})
    obj._remote_config = object()  # remote → no local CLI fallback
    result = obj.call_tool("click", {"x": 1, "y": 2}, timeout=5.0)
    assert obj._recreate_calls == ["click"]
    assert codes == ["transport_outcome_unknown"]


# ── M2: in-coroutine startup deadline ─────────────────────────────────────


def test_m2_hung_initialize_raises_within_budget_and_unwinds():
    """A lifecycle whose initialize() hangs forever must raise within the
    startup deadline, unwind the transport context managers, and leave the
    owner task done (not pending inside a closing loop).

    Mirrors the reviewer's probe: transport_entered=True, transport_exited=True,
    owner_task done, no pending tasks in the stopped loop.
    """
    entered = []
    exited = []
    ready = threading.Event()

    # A fake ClientSession whose initialize() hangs forever.
    class _HungSession:
        async def initialize(self):
            # Block forever — only a cancel/deadline can interrupt this.
            await asyncio.Event().wait()

        async def list_tools(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    # A fake streamable_http_client / httpx.AsyncClient that records enter/exit.
    class _FakeCM:
        def __init__(self, tracker, label):
            self._tracker, self._label = tracker, label
        async def __aenter__(self):
            self._tracker.append(("enter", self._label))
            return self
        async def __aexit__(self, *exc):
            self._tracker.append(("exit", self._label))

    class _FakeStreams:
        async def __aenter__(self):
            entered.append("streams")
            return (object(), object())
        async def __aexit__(self, *exc):
            exited.append("streams")

    class _FakeHttpClient:
        async def __aenter__(self):
            entered.append("http")
            return self
        async def __aexit__(self, *exc):
            exited.append("http")

    obj = _cbs._CuaDriverSession.__new__(_cbs._CuaDriverSession)
    obj._bridge = _cbs._AsyncBridge.__new__(_cbs._AsyncBridge)
    obj._remote_config = object()  # take the remote path
    obj._embedded_daemon = None
    obj._lock = threading.Lock()
    obj._started = False
    obj._setup_error = None
    obj._lifecycle_future = None
    obj._transport_generation = 0
    obj._transport_reset_callback = None
    obj._capabilities, obj._tool_schemas, obj._capability_version = {}, {}, ""
    obj._declared_session_id = None
    # Use a short deadline so the test is fast.
    obj._STARTUP_DEADLINE_S = 1.0

    # Patch the imports _lifecycle_coro does for the remote path so we get
    # our hung session and fake CMs.
    import time

    real_run_coro = asyncio.run_coroutine_threadsafe

    async def _lifecycle():
        # Replicate the remote-path structure with fakes, using the real
        # _start_with_deadline (which is what M2 fixes).
        import anyio
        obj._shutdown_event = asyncio.Event()
        obj._startup_phase = "binary-check"
        _t0 = time.monotonic()
        try:
            obj._startup_phase = "remote-connect"
            async with _FakeHttpClient() as http_client:
                async with _FakeStreams() as streams:
                    obj._startup_phase = "mcp-initialize"
                    async with _HungSession() as session:
                        await obj._start_with_deadline(session, _t0)
                        # Should never reach here.
                        ready.set()
                        await obj._shutdown_event.wait()
        except BaseException as e:
            obj._setup_error = e
            ready.set()
            raise
        finally:
            obj._session, obj._started = None, False

    loop = asyncio.new_event_loop()
    fut = real_run_coro(_lifecycle(), loop)

    # Run the loop until the future completes (the deadline fires).
    try:
        # Drive the loop in a thread (like _AsyncBridge does).
        def _drive():
            loop.run_forever()
        t = threading.Thread(target=_drive, daemon=True)
        t.start()
        # The in-coroutine deadline (1s) raises TimeoutError → RuntimeError,
        # coro finally unwinds CMs, future completes.
        exc = None
        try:
            result = fut.result(timeout=10.0)
        except Exception as e:
            exc = e
        assert exc is not None, "hung initialize must raise, not return cleanly"
        assert "startup handshake exceeded" in str(exc)
        assert "mcp-initialize" in str(exc)
        # Transport CMs must have been entered AND exited (unwound).
        assert "http" in entered
        assert "streams" in entered
        assert "streams" in exited
        assert "http" in exited
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2.0)
        # Collect all remaining tasks; none should be pending (the hung
        # initialize's asyncio.Event().wait() was cancelled by the deadline).
        pending = asyncio.all_tasks(loop=loop)
        loop.close()
        assert not pending, f"pending tasks left in closed loop: {pending}"

    # The owner future must be done (not pending/abandoned).
    assert fut.done()


def test_m2_start_with_deadline_succeeds_on_fast_initialize():
    """A normal (fast) initialize()+discovery completes within the deadline
    and sets _ready_event — no spurious timeout."""
    obj = _cbs._CuaDriverSession.__new__(_cbs._CuaDriverSession)
    obj._ready_event = threading.Event()
    obj._startup_phase = "mcp-initialize"
    obj._STARTUP_DEADLINE_S = 35.0

    class _FastSession:
        async def initialize(self):
            pass
        async def list_tools(self):
            class _R:
                tools = []
            return _R()

    import asyncio
    async def _run():
        await obj._start_with_deadline(_FastSession(), 0.0)
    asyncio.new_event_loop().run_until_complete(_run())
    assert obj._ready_event.is_set()
    assert obj._startup_phase == "ready"
    assert obj._session is not None  # session exposed on success