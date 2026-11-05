"""Pre-turn dead-connection sweep over an agent's OpenAI client pools (primary + cached request
client). Sibling of ``agent.agent_runtime_helpers``, which owns the pool-socket walk."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger("run_agent")  # origin module's logger name: log records / caplog filters unchanged


def _socket_is_dead(sock) -> bool:
    """Probe socket health with a non-blocking recv peek."""
    import socket as _socket
    try:
        sock.setblocking(False)
        return sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT) == b""
    except BlockingIOError:
        return False  # no data available: socket is healthy
    except OSError:
        return True
    finally:
        with contextlib.suppress(OSError):
            sock.setblocking(True)


def _count_dead_pool_sockets(client: Any) -> int:
    """Return the number of sockets in ``client``'s pool that have reached EOF (CLOSE-WAIT)."""
    from agent.agent_runtime_helpers import _iter_pool_sockets
    return sum(1 for sock in _iter_pool_sockets(client) if _socket_is_dead(sock))


def cleanup_dead_connections(agent) -> bool:
    """Detect and clean up dead TCP connections owned by an agent.

    Long-lived chat-completion turns use a cached per-request OpenAI client
    for sequential calls, while ``agent.client`` remains the primary client
    used for runtime rebuilds and provider management.  Inspect both pools:
    a provider FIN on the cached client otherwise stays in CLOSE-WAIT until
    session teardown and can be handed back to the next request.

    Dead sockets on the primary client rebuild that client.  Dead sockets on
    the request cache go through the cache's ownership-aware teardown hook so
    an in-flight worker is aborted safely and an idle client is closed by its
    owner.  Returns True if dead connections were found and cleaned up.
    """
    primary = getattr(agent, "client", None)
    cache = getattr(agent, "_request_client_cache", None)
    request_client = cache.get("client") if isinstance(cache, dict) else None

    clients: list[tuple[str, Any]] = []
    if primary is not None:
        clients.append(("primary", primary))
    if request_client is not None and request_client is not primary:
        clients.append(("request", request_client))
    if not clients:
        return False

    dead_clients: list[tuple[str, Any, int]] = []
    for label, client in clients:
        try:
            dead_count = _count_dead_pool_sockets(client)
            if dead_count:
                dead_clients.append((label, client, dead_count))
        except Exception:
            # Best-effort probe over private httpcore internals: one client's failure must not
            # skip the other's check or abort the turn.
            logger.debug("Dead connection check error for %s client", label, exc_info=True)

    if not dead_clients:
        return False

    total_dead = sum(count for _label, _client, count in dead_clients)
    labels = ", ".join(label for label, _client, _count in dead_clients)
    logger.warning(
        "Found %d dead connection(s) in %s client pool(s) — cleaning up",
        total_dead,
        labels,
    )

    for label, _client, _count in dead_clients:
        try:
            if label == "primary":
                agent._replace_primary_openai_client(reason="dead_connection_cleanup")
            else:
                # This hook clears the cache and chooses close() vs socket
                # shutdown based on whether another worker owns the client.
                agent._close_cached_request_openai_client(reason="dead_connection_cleanup")
        except Exception:
            # Cleaning one pool must not stop the other pool's cleanup.
            logger.debug("Dead connection cleanup error for %s client", label, exc_info=True)
    return True
