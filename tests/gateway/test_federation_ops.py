"""Phase 22 tests: federation ops layer (health monitoring + lost-contact SOS)."""
import time

from gateway.federation.federation_ops import (
    HealthMonitor,
    LostContactSOS,
    collect_local_health,
    HEALTH_CRITICAL,
)


def test_health_monitor_basic():
    hm = HealthMonitor(device_id="A", offline_threshold_s=30.0)
    revived = hm.update_from_heartbeat("B", {
        "hostname": "mac-b", "gateway_up": True, "federation_connected": True,
        "cpu_load": 0.5, "cpu_cores": 8, "memory_gb": 16.0,
        "disk_free_gb": 100.0, "hermes_version": "v0.19.1",
    })
    assert revived is False
    h = hm.get_health("B")
    assert h is not None
    assert h.gateway_up is True
    assert hm.compute_health_score("B") > 0.9
    print("✅ test_health_monitor_basic")


def test_death_and_revival():
    hm = HealthMonitor(device_id="A", offline_threshold_s=30.0)
    hm.update_from_heartbeat("B", {"gateway_up": True})
    assert hm.mark_failed("B") is False
    assert hm.mark_failed("B") is False
    died = hm.mark_failed("B")
    assert died is True
    h = hm.get_health("B")
    assert h is not None
    assert h.level == HEALTH_CRITICAL
    revived = hm.update_from_heartbeat("B", {"gateway_up": True})
    assert revived is True
    types = [a["type"] for a in hm.get_alerts()]
    assert "lost_contact" in types
    assert "recovery" in types
    print("✅ test_death_and_revival")


def test_sos_escalation():
    hm = HealthMonitor(device_id="A", offline_threshold_s=30.0)
    hm.update_from_heartbeat("B", {"gateway_up": True})
    hm.mark_failed("B")
    hm.mark_failed("B")
    hm.mark_failed("B")
    status = hm.get_health("B")
    assert status is not None
    status.last_heartbeat_at = time.time() - 400  # > critical window
    sos = LostContactSOS(device_id="A", health=hm)
    alerts = sos.update()
    assert len(alerts) == 1
    assert alerts[0].alert_type == "lost_contact"
    assert "critical" in alerts[0].message
    print("✅ test_sos_escalation")


def test_collect_local_health():
    h = collect_local_health(hermes_version="v0.19.1")
    assert "cpu_load" in h
    assert "cpu_cores" in h
    assert "disk_free_gb" in h
    assert h["hermes_version"] == "v0.19.1"
    print("✅ test_collect_local_health")


if __name__ == "__main__":
    test_health_monitor_basic()
    test_death_and_revival()
    test_sos_escalation()
    test_collect_local_health()
    print("\n✅ Phase 22 ops tests passed")
