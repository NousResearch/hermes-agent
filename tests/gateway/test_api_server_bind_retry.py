"""t_b9436ad2: an api_server bind that races a draining predecessor waits instead of parking fatal.

``connect()`` retries EADDRINUSE while the address is recoverable and parks at the deadline — a
real port conflict stays a fatal config error the reconnect watcher must not feed on (#52132).
These units pin the *contract* (wait-window bounds around the supervisor's stop budget, and the
accept/refuse probe the loop keys on) without pinning the loop's shape: t_33f99ff1 widened the
attempt-count model to a time budget riding out a SIGKILLed predecessor's TIME_WAIT, and the
assertions must not churn with it.
"""
import socket

from gateway.platforms import api_server
from gateway.platforms.api_server import _port_accepting


def _bind_idle_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_port_accepting_true_while_a_predecessor_still_listens():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)  # both probes connect without accept(); a backlog of 1 starves the second
    try:
        port = listener.getsockname()[1]
        assert _port_accepting("127.0.0.1", port) is True
        # A wildcard-bound server is probed over loopback; a listener there is still "held".
        assert _port_accepting("0.0.0.0", port) is True
    finally:
        listener.close()


def test_port_accepting_false_once_the_port_refuses():
    # Closed socket: nothing accepts. The loop reads that as "no live predecessor" — TIME_WAIT
    # or a real conflict — and switches to its slow ride-out / deadline park instead of
    # fast-retrying a socket that cannot come back.
    assert _port_accepting("127.0.0.1", _bind_idle_port(), timeout=0.2) is False


def test_retry_window_outlives_a_capped_drain_without_burning_forever():
    # The bind retry must cover how long a draining (or SIGKILL-reaped, TIME_WAIT-lingering)
    # predecessor can still keep the port from a fresh boot, without stretching a genuinely
    # doomed boot indefinitely. Accepts either the attempt-count or the time-budget model.
    budget = getattr(api_server, "_BIND_RETRY_BUDGET_S", None)
    if budget is None:  # attempt-count model: capped-backoff sum
        budget = sum(min(0.5 * (n + 1), api_server._BIND_RETRY_SLEEP_CAP_S)
                     for n in range(api_server._BIND_ATTEMPTS - 1))
    assert 30 <= budget <= 120, (
        "a draining predecessor must be wait-able, not park-able on first EADDRINUSE, "
        "yet a fresh boot must not hang for minutes per doomed bind"
    )
