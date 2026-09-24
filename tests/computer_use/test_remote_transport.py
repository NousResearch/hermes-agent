"""Pure unit tests for the remote CUA client transport (no network).

Contracts covered:
- The remote transport consumes the streamable_http_client yield by INDEX
  (streams[0], streams[1]), not fixed-arity unpacking — so it tolerates both
  the mcp 2.x 2-tuple (read, write) and the 1.x 3-tuple (read, write,
  get_session_id).  A regression to ``read, write = streams`` or
  ``read, write, _ = streams`` is caught by running the real lifecycle coro
  against a fake transport yielding each arity.
- The remote AsyncClient kwargs never leak the bearer token to env proxies
  (trust_env=False, proxy=None) and keep long tool calls alive (read=None).
- The seeded mcp-protocol-version header matches the SDK's current
  LATEST_HANDSHAKE_VERSION (derived the same way production derives it), so it
  can't desync when the SDK is upgraded.
- Remote sessions never fall back to the LOCAL cua-driver CLI: a transient transport
  failure on a remote session surfaces as a fail-closed outcome-unknown envelope.
- Local sessions keep the CLI fallback (backward compat).
"""
import asyncio
import contextlib

from tools.computer_use import cua_backend_session as _cbs


def _make_remote_session(monkeypatch, *, stream_arity):
    """Build a _CuaDriverSession whose _lifecycle_coro runs the remote branch
    against a fake streamable_http_client yielding *stream_arity* streams.

    The fake ClientSession.initialize() signals the ready event; the shutdown
    event is pre-set so the lifecycle coro exits cleanly after initialize.
    """
    obj = _cbs._CuaDriverSession.__new__(_cbs._CuaDriverSession)
    obj._bridge = _FakeBridge()
    obj._remote_config = _FakeRemoteConfig()
    obj._timeout_suspect = False
    obj._started = True
    obj._LIFECYCLE_CALLS = frozenset()
    obj._TRANSPORT_REPLAY_SAFE_TOOLS = frozenset()
    obj._notify_transport_reset = lambda: None
    obj._capabilities = {}
    obj._tool_schemas = {}
    obj._capability_version = ""
    obj._lock = __import__("threading").Lock()
    obj._transport_generation = 0
    obj._transport_reset_callback = None
    obj._ready_event = __import__("threading").Event()
    obj._setup_error = None
    obj._session = None
    obj._embedded_daemon = None
    obj._declared_session_id = None
    obj._lifecycle_future = None

    # Pre-create the shutdown event on the fake bridge's loop so the coro can
    # set it immediately after initialize completes.
    loop = obj._bridge._loop
    shutdown_event = asyncio.Event()

    class _FakeRead:
        pass

    class _FakeWrite:
        pass

    fake_streams = tuple(_FakeRead() for _ in range(stream_arity))

    @contextlib.asynccontextmanager
    async def _fake_streamable_http_client(url, **kwargs):
        yield fake_streams

    @contextlib.asynccontextmanager
    async def _fake_client_session(read, write):
        # Signal ready, then let the coro see the pre-set shutdown event.
        obj._ready_event.set()
        shutdown_event.set()
        yield _FakeSession()

    class _FakeSession:
        async def initialize(self):
            pass

        async def list_tools(self):
            class _R:
                tools = []
            return _R()

    # Patch the imports the lifecycle coro does locally.
    import mcp
    import mcp.client.streamable_http as _shttp

    monkeypatch.setattr(_shttp, "streamable_http_client", _fake_streamable_http_client)
    monkeypatch.setattr(mcp, "ClientSession", _fake_client_session)

    # _core is an _OriginProxy; patch the SDK symbols it resolves.
    import tools.mcp_tool as _mt
    monkeypatch.setattr(_mt, "sdk_httpx", lambda: _FakeHttpx(), raising=False)

    # The coro creates the shutdown event on the loop; we need it to be our
    # pre-set one so it exits immediately.  Patch asyncio.Event on the coro's
    # module to return our pre-set event.
    obj._shutdown_event = shutdown_event

    return obj


class _FakeRemoteConfig:
    url = "http://localhost:8765/mcp"
    token = "t" * 40


class _FakeHttpx:
    class AsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    class Timeout:
        def __init__(self, *a, **kw):
            pass


class _FakeBridge:
    def __init__(self):
        self._loop = asyncio.new_event_loop()

    def start(self):
        pass

    def stop(self):
        self._loop.close()

    def run(self, coro, timeout=None):
        return self._loop.run_until_complete(coro)


def test_remote_transport_consumes_2tuple_streams(monkeypatch):
    """The lifecycle coro must handle the mcp 2.x 2-tuple (read, write).

    Uses indexed access (streams[0], streams[1]); if someone reintroduces
    ``read, write, _ = streams`` this fails on the 2-tuple.
    """
    obj = _make_remote_session(monkeypatch, stream_arity=2)
    obj._bridge._loop.run_until_complete(_run_lifecycle_briefly(obj))


def test_remote_transport_consumes_3tuple_streams(monkeypatch):
    """The lifecycle coro must also handle the mcp 1.x 3-tuple.

    If someone reintroduces ``read, write = streams`` this fails on the
    3-tuple (too many values to unpack).
    """
    obj = _make_remote_session(monkeypatch, stream_arity=3)
    obj._bridge._loop.run_until_complete(_run_lifecycle_briefly(obj))


async def _run_lifecycle_briefly(obj):
    """Drive _lifecycle_coro just past the stream consumption + initialize."""
    obj._shutdown_event = asyncio.Event()
    # Run the coro; it will set _ready_event after initialize, then wait on
    # _shutdown_event.  Set the shutdown event from a concurrent task so the
    # coro exits cleanly.
    async def _set_shutdown():
        # Wait until ready is signalled (initialize done), then shut down.
        while not obj._ready_event.is_set():
            await asyncio.sleep(0.01)
        obj._shutdown_event.set()

    await asyncio.wait_for(
        asyncio.gather(obj._lifecycle_coro(), _set_shutdown()),
        timeout=10.0,
    )


def test_client_kwargs():
    kwargs = _cbs._remote_http_client_kwargs("sekrit-token")
    assert kwargs["trust_env"] is False  # env proxies must not receive the bearer token
    assert kwargs["proxy"] is None
    assert kwargs["follow_redirects"] is False
    assert "Bearer" in kwargs["headers"]["Authorization"]
    # The protocol-version header must match the SDK's current
    # LATEST_HANDSHAKE_VERSION — derived the same way production code derives
    # it (via tools.mcp_tool), never a hardcoded date that desyncs on SDK
    # upgrade.
    from tools.mcp_tool_common import _core
    expected_version = _core.LATEST_HANDSHAKE_VERSION
    assert kwargs["headers"]["mcp-protocol-version"] == expected_version, (
        f"mcp-protocol-version header {kwargs['headers']['mcp-protocol-version']!r} "
        f"does not match SDK LATEST_HANDSHAKE_VERSION {expected_version!r}"
    )
    # Long tool calls (screenshots, UI waits) must not hit httpx2's 5s default read timeout.
    assert kwargs["timeout"].read is None


class _TransientBridge:
    """Bridge whose run() raises immediately with a transient daemon error."""

    def run(self, coro, timeout):
        coro.close()  # the real bridge would await it; close keeps the loop warning-free
        raise RuntimeError("[Errno 35] Resource temporarily unavailable")


def _make_session(remote_config):
    """Build a _CuaDriverSession without __init__ with only the attrs _call_tool reads."""
    obj = _cbs._CuaDriverSession.__new__(_cbs._CuaDriverSession)
    obj._bridge = _TransientBridge()
    obj._remote_config = remote_config
    obj._timeout_suspect = False
    obj._started = True
    obj._LIFECYCLE_CALLS = frozenset()  # get_window_state is a plain tool call here
    # Replay-safe set: only these tools reach the CLI-fallback branch after a transient error.
    obj._TRANSPORT_REPLAY_SAFE_TOOLS = frozenset(
        {"get_cursor_position", "get_displays", "get_screen_size",
         "get_window_state", "list_apps", "list_windows"})
    obj._notify_transport_reset = lambda: None
    return obj


def test_remote_cli_fallback_disabled_for_remote(monkeypatch):
    codes = []

    def _fake_outcome(name, exc, code):
        codes.append(code)
        return {"outcome": code}

    monkeypatch.setattr(_cbs, "_outcome_unknown", _fake_outcome)
    obj = _make_session(remote_config=object())
    obj._call_tool_via_cli = lambda name, args, timeout: (_ for _ in ()).throw(
        AssertionError("must not spawn local CLI for remote sessions"))

    result = obj.call_tool("get_window_state", {"window": "front"}, timeout=5.0)

    assert codes == ["remote_transport_outcome_unknown"]
    assert result == {"outcome": "remote_transport_outcome_unknown"}


def test_remote_transport_fallback_used_for_local(monkeypatch):
    monkeypatch.setattr(_cbs, "_outcome_unknown",
                        lambda name, exc, code: (_ for _ in ()).throw(
                            AssertionError("local path must not fail closed here")))
    obj = _make_session(remote_config=None)
    obj._call_tool_via_cli = lambda name, args, timeout: {"ok": True, "fallback": True}

    result = obj.call_tool("get_window_state", {"window": "front"}, timeout=5.0)

    assert result == {"ok": True, "fallback": True}