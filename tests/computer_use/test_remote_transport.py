"""Pure unit tests for the remote CUA client transport (no network).

Contracts covered:
- mcp 2.x streamable_http_client yields a 2-tuple of streams (the indexed access in
  cua_backend_session._lifecycle_coro relies on it; a fixed-arity unpack fails).
- The remote AsyncClient kwargs never leak the bearer token to env proxies
  (trust_env=False, proxy=None) and keep long tool calls alive (read=None).
- Remote sessions never fall back to the LOCAL cua-driver CLI: a transient transport
  failure on a remote session surfaces as a fail-closed outcome-unknown envelope.
- Local sessions keep the CLI fallback (backward compat).
"""
import inspect

from tools.computer_use import cua_backend_session as _cbs


def test_remote_transport_arity_documented():
    # Documents the mcp 2.x contract the indexed access relies on: a 2-tuple of streams.
    from mcp.client.streamable_http import streamable_http_client

    src = inspect.getsource(streamable_http_client)
    assert "yield read_stream, write_stream" in src


def test_client_kwargs():
    kwargs = _cbs._remote_http_client_kwargs("sekrit-token")
    assert kwargs["trust_env"] is False  # env proxies must not receive the bearer token
    assert kwargs["proxy"] is None
    assert kwargs["follow_redirects"] is False
    assert "Bearer" in kwargs["headers"]["Authorization"]
    assert kwargs["headers"]["mcp-protocol-version"] == "2025-03-26"
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