"""A killed/exited cua-driver surfaces as MCPError(CONNECTION_CLOSED); it must trigger a reconnect."""

import pytest

mcp_exceptions = pytest.importorskip("mcp.shared.exceptions")
mcp_types = pytest.importorskip("mcp.types")

from tools.computer_use.cua_backend_session import _CuaDriverSession


def test_mcp_connection_closed_is_recoverable():
    exc = mcp_exceptions.MCPError(code=mcp_types.CONNECTION_CLOSED, message="Connection closed")
    assert _CuaDriverSession._is_closed_session_error(exc) is True


def test_other_mcp_errors_are_not_reconnect_triggers():
    exc = mcp_exceptions.MCPError(code=-32602, message="Invalid params")
    assert _CuaDriverSession._is_closed_session_error(exc) is False


def test_existing_stream_errors_still_recoverable():
    assert _CuaDriverSession._is_closed_session_error(BrokenPipeError()) is True
    assert _CuaDriverSession._is_closed_session_error(ValueError("x")) is False
