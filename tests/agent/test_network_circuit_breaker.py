import errno
import time

import pytest


def test_host_connectivity_errors_are_classified_through_exception_chain():
    from agent.network_circuit_breaker import is_host_connectivity_error

    low = OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address")
    wrapped = RuntimeError("provider wrapper")
    wrapped.__cause__ = low

    assert is_host_connectivity_error(wrapped) is True
    assert is_host_connectivity_error(OSError(errno.EHOSTUNREACH, "No route to host")) is True
    assert is_host_connectivity_error(TimeoutError("ordinary timeout")) is False


def test_breaker_opens_after_repeated_host_connectivity_errors(monkeypatch):
    from agent.network_circuit_breaker import NetworkCircuitBreaker, NetworkCircuitOpen

    now = {"t": 1000.0}
    monkeypatch.setattr(time, "time", lambda: now["t"])
    breaker = NetworkCircuitBreaker(threshold=2, cooldown_seconds=30)

    breaker.before_request("provider")
    breaker.record_failure(OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address"), surface="provider")
    breaker.before_request("provider")
    breaker.record_failure(OSError(errno.EHOSTUNREACH, "No route to host"), surface="provider")

    with pytest.raises(NetworkCircuitOpen):
        breaker.before_request("provider")

    now["t"] += 31
    breaker.before_request("provider")


def test_parse_netstat_counts_and_pressure_threshold():
    from agent.network_circuit_breaker import parse_netstat_state_counts, socket_pressure_is_high

    output = """
tcp4 0 0 192.168.1.2.50001 1.1.1.1.443 TIME_WAIT
tcp4 0 0 192.168.1.2.50002 1.1.1.1.443 TIME_WAIT
tcp4 0 0 192.168.1.2.50003 1.1.1.1.443 ESTABLISHED
"""
    counts = parse_netstat_state_counts(output)
    assert counts["TIME_WAIT"] == 2
    assert counts["ESTABLISHED"] == 1
    assert socket_pressure_is_high({"TIME_WAIT": 12000}, port_range_size=16384, threshold_ratio=0.70)
    assert not socket_pressure_is_high({"TIME_WAIT": 100}, port_range_size=16384, threshold_ratio=0.70)
