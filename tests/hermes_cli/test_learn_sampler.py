from __future__ import annotations

import json
from datetime import timedelta

from hermes_time import now as hermes_now


def _read_events(home):
    events_file = home / "learn" / "events.jsonl"
    if not events_file.exists():
        return []
    return [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_sample_once_persists_redacted_metadata_only(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import sampler, state

    state.start(mode="learn")

    record = sampler.sample_once(
        collector=lambda: {
            "process_name": "chrome.exe",
            "window_title": "Inbox from owner@example.com https://example.com/path?token=secret",
            "timestamp": hermes_now().isoformat(),
            "idle": False,
            "idle_seconds": 0,
            "duration_seconds": 61,
        }
    )

    assert record is not None
    assert record["process_name"] == "chrome.exe"
    assert record["domain"] == "example.com"
    assert record["category"] == "browser"
    assert record["idle"] is False
    assert record["duration_seconds"] == 61
    assert "owner@example.com" not in json.dumps(record)
    assert "token=secret" not in json.dumps(record)
    assert "https://example.com/path" not in json.dumps(record)

    persisted = _read_events(home)
    assert persisted == [record]


def test_sample_once_respects_denylist_and_allowlist(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import sampler, state

    state.start(mode="learn")
    state.update_config(denylist=["slack.exe"])

    denied = sampler.sample_once(
        collector=lambda: {
            "process_name": "slack.exe",
            "window_title": "Acme internal chat",
            "timestamp": hermes_now().isoformat(),
        }
    )
    assert denied is None
    assert _read_events(home) == []

    state.update_config(allowlist=["code.exe"], denylist=[])
    skipped = sampler.sample_once(
        collector=lambda: {
            "process_name": "notepad.exe",
            "window_title": "scratch",
            "timestamp": hermes_now().isoformat(),
        }
    )
    kept = sampler.sample_once(
        collector=lambda: {
            "process_name": "code.exe",
            "window_title": "hermes-agent",
            "timestamp": hermes_now().isoformat(),
        }
    )

    assert skipped is None
    assert kept is not None
    assert [event["process_name"] for event in _read_events(home)] == ["code.exe"]


def test_sample_once_prunes_events_outside_retention_window(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import sampler, state

    state.start(mode="learn")
    state.update_config(retention_days=7)

    learn_dir = home / "learn"
    learn_dir.mkdir(parents=True, exist_ok=True)
    old_timestamp = (hermes_now() - timedelta(days=10)).isoformat()
    (learn_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "timestamp": old_timestamp,
                "process_name": "code.exe",
                "window_title": "old",
                "category": "development",
                "duration_seconds": 30,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sampler.sample_once(
        collector=lambda: {
            "process_name": "code.exe",
            "window_title": "new",
            "timestamp": hermes_now().isoformat(),
            "duration_seconds": 30,
        }
    )

    events = _read_events(home)
    assert len(events) == 1
    assert events[0]["window_title"] == "new"


def test_sample_once_infers_duration_from_previous_sample(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import sampler, state

    state.start(mode="learn")

    sampler.sample_once(
        collector=lambda: {
            "process_name": "code.exe",
            "window_title": "hermes-agent",
            "timestamp": "2026-06-23T12:00:00+00:00",
        }
    )
    second = sampler.sample_once(
        collector=lambda: {
            "process_name": "code.exe",
            "window_title": "hermes-agent",
            "timestamp": "2026-06-23T12:01:15+00:00",
        }
    )

    assert second is not None
    assert second["duration_seconds"] == 75
