"""Port readiness checks used before restarting the gateway's API server."""

from __future__ import annotations

from hermes_platform.host.facts import os_family


def _gw():
    from hermes_cli import gateway
    return gateway


def _windows_tcp_port_free(host: str | None, port: int) -> bool:
    socket = _gw().socket
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    except socket.gaierror:
        return True  # The replacement will report the invalid listen address when it binds.
    for family, socktype, proto, _, address in addresses:
        try:
            with socket.socket(family, socktype, proto) as probe:
                # Windows can silently drop SYNs to a closed port. Exclusive bind also detects
                # wildcard listeners and sockets opened with SO_REUSEADDR without accepting traffic.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(address)
        except OSError:
            return False
    return True


def _wait_for_tcp_port_free(host: str | None, port: int, *, timeout: float = 10.0) -> bool:
    """Wait for the listen address to be available before starting its replacement.

    Windows uses an exclusive bind probe: a closed port may time out instead of refusing TCP.
    POSIX uses connection refusal, allowing the API server's bind retry to handle TIME_WAIT.
    A connect timeout remains busy, since a live listener can have a full accept queue.
    """
    gw = _gw()
    deadline = gw.time.monotonic() + timeout
    while gw.time.monotonic() < deadline:
        if os_family() == "win32":
            if _windows_tcp_port_free(host, port):
                return True
        else:
            try:
                with gw.socket.create_connection((host, port), timeout=0.2):
                    pass
            except ConnectionRefusedError:
                return True
            except TimeoutError:
                pass
            except OSError:
                return True  # The replacement's bind reports an invalid/unreachable address.
        gw.time.sleep(0.1)
    return False


def _wait_for_api_server_port_free(*, timeout: float = 10.0) -> bool:
    """Wait on the configured API server address only while that platform is enabled."""
    from gateway.config import Platform
    from gateway.platforms.api_server import listen_address

    gw = _gw()
    pconfig = gw.load_gateway_config().platforms.get(Platform.API_SERVER)
    if pconfig is None or not pconfig.enabled:
        return True
    host, port = listen_address(pconfig.extra or {})
    freed = gw._wait_for_tcp_port_free(host, port, timeout=timeout)
    if not freed:
        print(f"⚠ {host}:{port} still unavailable — new api_server may fail to bind")
    return freed
