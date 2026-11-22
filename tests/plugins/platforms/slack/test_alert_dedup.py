from pathlib import Path

from plugins.platforms.slack.alert_dedup import AlertFingerprintStore, parse_alerts


ALERT = {
    "text": ":fire: 1 alert(s) firing.",
    "attachments": [{
        "text": (
            "*Firing*: Kubernetes node down: 192.168.1.171\n\n"
            "  *Summary:* Kubernetes node down: 192.168.1.171:9100\n"
            "  *Description:* exporter is absent.\n\n"
            "  :rotating_light: <https://grafana.example/alerting/grafana/rule/view?"
            "orgId=1|Alert>\n"
            "  <https://grafana.example/alerting/silence/new?"
            "alertmanager=grafana&amp;matcher=__alert_rule_uid__%3Ddfmcp1iues5c0a"
            "&amp;matcher=alert_type%3Dnode-down&amp;matcher=node_instance%3D"
            "192.168.1.171%3A9100&amp;matcher=severity%3Dcritical|Silence>"
        )
    }],
}


def test_parse_alert_uses_stable_matchers_and_not_summary():
    alerts = parse_alerts(ALERT, channel_id="C123", message_ts="10.20")
    assert len(alerts) == 1
    assert alerts[0].fingerprint == (
        "dfmcp1iues5c0a|alert_type=node-down|node_instance=192.168.1.171:9100|severity=critical"
    )
    assert alerts[0].summary == "Kubernetes node down: 192.168.1.171:9100"


def test_store_returns_duplicate_and_persists_original(tmp_path: Path):
    store = AlertFingerprintStore(tmp_path / "alerts.json")
    assert store.observe(ALERT, channel_id="C123", message_ts="10.20") == []
    duplicates = store.observe(ALERT, channel_id="C123", message_ts="10.21")
    assert len(duplicates) == 1
    assert duplicates[0].original_permalink.endswith("/p1020")
    assert duplicates[0].current_permalink.endswith("/p1021")


def test_resolved_alert_starts_new_lifecycle(tmp_path: Path):
    store = AlertFingerprintStore(tmp_path / "alerts.json")
    store.observe(ALERT, channel_id="C123", message_ts="10.20")
    resolved = {**ALERT, "attachments": [{"text": ALERT["attachments"][0]["text"].replace("*Firing*:", "*Resolved*:")}]}
    assert store.observe(resolved, channel_id="C123", message_ts="10.21") == []
    assert store.observe(ALERT, channel_id="C123", message_ts="10.22") == []
