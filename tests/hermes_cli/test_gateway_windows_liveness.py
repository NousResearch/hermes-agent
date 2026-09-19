"""Regression coverage for the Windows gateway Scheduled-Task liveness consumer."""

from datetime import datetime, timezone
import json

from hermes_cli import gateway_windows, gateway_windows_liveness as liveness


def _heartbeat(home, *, age=181, pid=42):
    path = home / "state" / "gateway.heartbeat"
    path.parent.mkdir()
    path.write_text(json.dumps({"pid": pid, "updated_at": datetime.fromtimestamp(1_000 - age, timezone.utc).isoformat()}))


def test_inspect_distinguishes_dead_from_a_stalled_live_process(tmp_path, monkeypatch):
    _heartbeat(tmp_path)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
    assert liveness.inspect(tmp_path, now=1_000) == "dead"
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
    assert liveness.inspect(tmp_path, now=1_000) == "stalled"


def test_inspect_fails_open_for_missing_signals_and_a_fresh_planned_stop(tmp_path):
    assert liveness.inspect(tmp_path, now=1_000) == "unknown"
    _heartbeat(tmp_path)
    (tmp_path / ".gateway-planned-stop.json").write_text(
        json.dumps({"written_at": datetime.fromtimestamp(990, timezone.utc).isoformat()})
    )
    assert liveness.inspect(tmp_path, now=1_000) == "fenced"


def test_dead_gateway_alert_and_scheduled_recovery_are_deduplicated(tmp_path, monkeypatch):
    _heartbeat(tmp_path)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
    calls = []
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: calls.append(args) or (0, "", ""))
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    assert liveness.run(tmp_path, now=1_000) == "dead"
    assert liveness.run(tmp_path, now=1_001) == "dead"
    assert calls == [["/Run", "/TN", "Hermes_Gateway_alice"]]
    assert json.loads((tmp_path / "state" / "gateway-needs-attention.json").read_text())["kind"] == "dead"


def test_liveness_task_wires_a_minute_trigger_to_the_hidden_consumer(tmp_path):
    xml = gateway_windows._build_scheduled_task_xml("Hermes_Gateway_Liveness", tmp_path / "watch.vbs", None, liveness=True)
    assert "<Interval>PT1M</Interval>" in xml
    assert "watch.vbs" in xml
