import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from hermes_cli.route_advisory import classify_route_advisory, format_route_advisory


def test_classify_route_advisory_normalizes_and_logs(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    payload = {
        "route_id": "business-growth",
        "route_name": "Business-Growth",
        "owner": "business-growth",
        "profile": "business-growth",
        "prompt_class": "business",
        "action": "Route to Business-Growth (requires approval)",
        "blocked": False,
        "blocked_actions": ["unapproved_account_access"],
        "requires_approval": True,
        "reason": "Requires future approval.",
        "confidence": 5,
        "advisory_mode": True,
        "auto_execute": False,
        "is_live": True,
    }

    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        assert args == ["/bin/hermes-route-test", "--json", "--stdin"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("hermes_cli.route_advisory.subprocess.run", fake_run)

    confidential_text = "Use Bearer abcdefghijklmnopqrstuvwxyz123456 for confidential BMI radio registration notes"
    advisory = classify_route_advisory(
        confidential_text,
        surface="cron:create",
        command="/bin/hermes-route-test",
    )

    assert confidential_text not in captured["args"]
    assert captured["kwargs"]["input"] == confidential_text

    assert advisory["route_id"] == "business-growth"
    assert advisory["profile"] == "business-growth"
    assert advisory["advisory_mode"] is True
    assert advisory["auto_execute"] is False
    assert advisory["requires_approval"] is True
    assert advisory["blocked_actions"] == ["unapproved_account_access"]

    log_path = home / "logs" / "routing_decisions.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["surface"] == "cron:create"
    assert records[0]["route_id"] == "business-growth"
    assert records[0]["prompt_sha256"]
    assert records[0]["prompt_length"] == len(confidential_text)
    assert "prompt_preview" not in records[0]
    assert "confidential BMI radio registration notes" not in json.dumps(records[0])


def test_classify_route_advisory_timeout_falls_back_to_macos(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, timeout=1.0)

    monkeypatch.setattr("hermes_cli.route_advisory.subprocess.run", fake_run)

    advisory = classify_route_advisory(
        "What should I work on next?",
        surface="gateway:telegram",
        command="/bin/hermes-route-test",
    )

    assert advisory["route_id"] == "main-hermes"
    assert advisory["profile"] == "macos"
    assert advisory["auto_execute"] is False
    assert advisory["error"] == "hermes-route timeout"
    assert (home / "logs" / "routing_decisions.jsonl").exists()


def test_format_route_advisory_is_advisory_only():
    text = format_route_advisory({
        "route_id": "design-media",
        "profile": "design-media",
        "confidence": 3,
        "requires_approval": True,
        "blocked_actions": ["raw_client_data_upload"],
    })

    assert "design-media -> profile design-media" in text
    assert "advisory-only, no auto-switch" in text
    assert "approval required" in text
