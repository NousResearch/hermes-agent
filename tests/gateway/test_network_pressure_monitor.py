def test_gateway_socket_pressure_monitor_opens_global_breaker(monkeypatch):
    import gateway.run as run
    from agent.network_circuit_breaker import get_global_network_breaker, NetworkCircuitOpen

    breaker = get_global_network_breaker()
    breaker.reset()
    monkeypatch.setenv("HERMES_GATEWAY_SOCKET_PRESSURE_THRESHOLD", "0.70")
    monkeypatch.setattr(run, "_socket_state_counts", lambda: {"TIME_WAIT": 12000})
    monkeypatch.setattr(run, "_ephemeral_port_range_size", lambda: 16384)

    assert run._gateway_socket_pressure_check() is True
    try:
        try:
            breaker.before_request("provider")
        except NetworkCircuitOpen:
            pass
        else:
            raise AssertionError("global network breaker did not open")
    finally:
        breaker.reset()
