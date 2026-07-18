def test_a2a_is_a_single_listener_port_binding_platform():
    from gateway.run import _PORT_BINDING_PLATFORM_VALUES

    assert "a2a" in _PORT_BINDING_PLATFORM_VALUES
