import json
from gateway.debrief_queue import queue_for_debrief, QUEUE_PATH


def test_queue_for_debrief_appends_entry(tmp_path, monkeypatch):
    fake_queue = tmp_path / "debrief_queue.jsonl"
    monkeypatch.setattr("gateway.debrief_queue.QUEUE_PATH", fake_queue)
    queue_for_debrief(source="glance", alert_type="HighCPUUsage", message="CPU 85%")
    lines = fake_queue.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["source"] == "glance"
    assert entry["alert_type"] == "HighCPUUsage"
    assert entry["message"] == "CPU 85%"
    assert "queued_at" in entry
