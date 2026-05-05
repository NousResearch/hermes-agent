"""Tests for router-monitor anticipation candidate generation."""

from datetime import datetime, timezone

from agent.anticipation import AnticipationPermission
from agent.anticipation_loops import build_router_monitor_candidates

NOW = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)


def base_snapshot(**overrides):
    snapshot = {
        "collected_at": NOW.isoformat(),
        "monitoring": {
            "quarantine_dir_exists": True,
            "cron_entries": [
                "* * * * * /root/quarantine/quarantine-v3.sh run",
                "*/5 * * * * /root/quarantine/gl-ngx-watchdog.sh",
                "*/5 * * * * /root/quarantine/cron-guard.sh",
            ],
            "cron_log_last_at": NOW.isoformat(),
            "segfault_count_7d": 0,
        },
        "counts": {"trusted": 36, "quarantined": 2, "blocked": 1, "active_dhcp": 20},
        "unknown_devices": [],
    }
    snapshot.update(overrides)
    return snapshot


def test_router_monitor_candidates_detect_monitoring_degradation():
    snapshot = base_snapshot(
        monitoring={
            "quarantine_dir_exists": True,
            "cron_entries": ["0 9 * * 0 /root/quarantine/quarantine-v3.sh digest"],
            "cron_log_last_at": "2026-05-05T16:30:00+00:00",
            "segfault_count_7d": 0,
        }
    )

    candidates = build_router_monitor_candidates(snapshot, now=NOW)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.loop_id == "router_monitor"
    assert candidate.proposed_permission is AnticipationPermission.ASK_TO_EXECUTE
    assert candidate.confidence >= 0.85
    assert "monitoring may be degraded" in candidate.title.lower()
    assert "minutely quarantine cron job is missing" in candidate.body
    assert "cron log is stale" in candidate.body
    assert "I did not change router state" in candidate.body
    assert candidate.dedupe_key.startswith("router_monitor:health:")


def test_router_monitor_candidates_silent_log_one_time_randomized_mac():
    snapshot = base_snapshot(
        unknown_devices=[
            {
                "mac": "7a:11:22:33:44:55",
                "active": False,
                "locally_administered": True,
                "seen_count_72h": 1,
                "hostnames": [],
                "traffic_summary": "no meaningful traffic",
            }
        ]
    )

    candidates = build_router_monitor_candidates(snapshot, now=NOW)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.proposed_permission is AnticipationPermission.SILENT_LOG
    assert candidate.confidence >= 0.7
    assert "one-time randomized" in candidate.title.lower()
    assert "7a:11:22" in candidate.body
    assert "No Telegram-worthy interruption" in candidate.body


def test_router_monitor_candidates_suggest_recurring_likely_phone():
    snapshot = base_snapshot(
        unknown_devices=[
            {
                "mac": "7a:11:22:33:44:55",
                "active": False,
                "locally_administered": True,
                "seen_count_72h": 4,
                "hostnames": ["Pixel-Guest"],
                "traffic_summary": "Google connectivity checks",
            }
        ]
    )

    candidates = build_router_monitor_candidates(snapshot, now=NOW)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.proposed_permission is AnticipationPermission.SUGGEST
    assert candidate.confidence >= 0.78
    assert "recurring randomized" in candidate.title.lower()
    assert "Pixel-Guest" in candidate.body
    assert "correlate it against guests" in candidate.body


def test_router_monitor_candidates_suggest_recurring_normal_unknown():
    snapshot = base_snapshot(
        unknown_devices=[
            {
                "mac": "00:11:22:33:44:55",
                "active": False,
                "locally_administered": False,
                "seen_count_72h": 4,
                "hostnames": ["esp-sensor"],
                "traffic_summary": "periodic DHCP only",
            }
        ]
    )

    candidates = build_router_monitor_candidates(snapshot, now=NOW)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.proposed_permission is AnticipationPermission.SUGGEST
    assert "recurring unknown" in candidate.title.lower()
    assert "esp-sensor" in candidate.body


def test_router_monitor_candidates_parse_string_booleans_without_escalating_false():
    snapshot = base_snapshot(
        unknown_devices=[
            {
                "mac": "7a:11:22:33:44:55",
                "active": "false",
                "locally_administered": "true",
                "seen_count_72h": "1",
            }
        ]
    )

    candidates = build_router_monitor_candidates(snapshot, now=NOW)

    assert len(candidates) == 1
    assert candidates[0].proposed_permission is AnticipationPermission.SILENT_LOG


def test_router_monitor_candidates_sanitize_snapshot_strings():
    snapshot = base_snapshot(
        unknown_devices=[
            {
                "mac": "00:11:22:33:44:55",
                "active": True,
                "locally_administered": False,
                "seen_count_72h": 2,
                "ips": ["192.168.8.240\x1b[31m"],
                "hostnames": ["evil\x1b[2Jhost\nnext"],
                "traffic_summary": "scan\x1b[99m" + ("x" * 400),
            }
        ]
    )

    candidates = build_router_monitor_candidates(snapshot, now=NOW)

    body = candidates[0].body
    assert "\x1b" not in body
    assert "\nnext" not in body
    assert len(body) < 900


def test_router_monitor_candidates_ask_to_execute_for_active_unknown():
    snapshot = base_snapshot(
        unknown_devices=[
            {
                "mac": "00:11:22:33:44:55",
                "active": True,
                "locally_administered": False,
                "seen_count_72h": 2,
                "ips": ["192.168.8.240"],
                "hostnames": ["unknown-laptop"],
                "traffic_summary": "active LAN traffic",
            }
        ]
    )

    candidates = build_router_monitor_candidates(snapshot, now=NOW)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.proposed_permission is AnticipationPermission.ASK_TO_EXECUTE
    assert candidate.confidence >= 0.86
    assert "active unknown" in candidate.title.lower()
    assert "192.168.8.240" in candidate.body
    assert "review router status before approving or blocking" in candidate.body
    assert "I did not block anything" in candidate.body


def test_router_monitor_candidates_order_health_before_device_noise():
    snapshot = base_snapshot(
        monitoring={
            "quarantine_dir_exists": False,
            "cron_entries": [],
            "cron_log_last_at": "2026-05-05T17:55:00+00:00",
        },
        unknown_devices=[
            {
                "mac": "7a:11:22:33:44:55",
                "active": False,
                "locally_administered": True,
                "seen_count_72h": 1,
            }
        ],
    )

    candidates = build_router_monitor_candidates(snapshot, now=NOW)

    assert len(candidates) == 2
    assert candidates[0].proposed_permission is AnticipationPermission.ASK_TO_EXECUTE
    assert "monitoring" in candidates[0].title.lower()
