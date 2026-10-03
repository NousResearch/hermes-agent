"""Regression tests for stale provider sockets in agent-owned client pools."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.agent_runtime_helpers import cleanup_dead_connections


class _ProbeSocket:
    def __init__(self, *, dead):
        self.dead = dead

    def setblocking(self, _enabled):
        pass

    def recv(self, _size, _flags):
        if self.dead:
            return b""
        raise BlockingIOError


def _client_with_socket(sock):
    stream = SimpleNamespace(_sock=sock)
    http11 = SimpleNamespace(_network_stream=stream)
    pool_entry = SimpleNamespace(_connection=http11)
    pool = SimpleNamespace(_connections=[pool_entry])
    transport = SimpleNamespace(_pool=pool)
    http_client = SimpleNamespace(_transport=transport)
    return SimpleNamespace(_client=http_client)


def test_cleanup_retires_dead_cached_request_client():
    """A half-closed request pool must not be hidden by a healthy primary."""
    primary = _client_with_socket(_ProbeSocket(dead=False))
    request = _client_with_socket(_ProbeSocket(dead=True))
    agent = SimpleNamespace(
        client=primary,
        _request_client_cache={"client": request},
        _replace_primary_openai_client=MagicMock(),
        _close_cached_request_openai_client=MagicMock(),
    )

    assert cleanup_dead_connections(agent) is True
    agent._replace_primary_openai_client.assert_not_called()
    agent._close_cached_request_openai_client.assert_called_once_with(
        reason="dead_connection_cleanup"
    )


def test_cleanup_rebuilds_dead_primary_client():
    """The existing primary-client recovery remains the cleanup path."""
    primary = _client_with_socket(_ProbeSocket(dead=True))
    request = _client_with_socket(_ProbeSocket(dead=False))
    agent = SimpleNamespace(
        client=primary,
        _request_client_cache={"client": request},
        _replace_primary_openai_client=MagicMock(),
        _close_cached_request_openai_client=MagicMock(),
    )

    assert cleanup_dead_connections(agent) is True
    agent._replace_primary_openai_client.assert_called_once_with(
        reason="dead_connection_cleanup"
    )
    agent._close_cached_request_openai_client.assert_not_called()
