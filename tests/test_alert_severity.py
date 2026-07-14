from gateway.alert_severity import classify_severity


def test_service_down_is_urgent():
    assert classify_severity(source="10.55.0.53:8090", alert_type="ServiceDown") == "urgent"


def test_security_incident_is_urgent():
    assert classify_severity(source="opnsense", alert_type="HighAttackVolume") == "urgent"


def test_high_system_load_is_batched():
    assert classify_severity(source="prometheus", alert_type="HighSystemLoad") == "batched"


def test_high_cpu_is_batched():
    assert classify_severity(source="glance", alert_type="HighCPUUsage") == "batched"


def test_disk_space_low_is_batched():
    assert classify_severity(source="jellyfin", alert_type="DiskSpaceLow") == "batched"


def test_unknown_alert_type_defaults_to_batched():
    # Fail toward not-interrupting for anything not explicitly classified urgent —
    # matches the design's "batched unless proven urgent" default (spec §4.2).
    assert classify_severity(source="anything", alert_type="SomeNewAlertType") == "batched"
