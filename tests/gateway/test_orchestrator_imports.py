def test_orchestrator_package_exports_phase_one_two_api():
    import gateway.orchestrator as orchestrator

    assert hasattr(orchestrator, "run_doctor")
    assert hasattr(orchestrator, "run_lanes")
    assert hasattr(orchestrator, "synthesize")
