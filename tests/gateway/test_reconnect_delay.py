from gateway.run import _reconnect_backoff


def test_reconnect_delay_stays_bounded_after_long_outage():
    assert [_reconnect_backoff(attempt) for attempt in range(1, 7)] == [30, 60, 120, 240, 300, 300]
    assert _reconnect_backoff(10**6) == _reconnect_backoff(6)
