"""Tests for voice benchmark JSONL telemetry."""

from __future__ import annotations

import json

from gateway import voice_bench


def test_append_event_redacts_and_renames_text_fields(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))

    voice_bench.append_event(
        {
            "turn_id": "voice-1",
            "stage": "stt",
            "platform": "telegram",
            "chat_id": "123",
            "transcript": "use api_key=sk-test-secret and continue",
        }
    )

    item = json.loads(path.read_text(encoding="utf-8"))
    assert "transcript" not in item
    assert item["transcript_chars"] == 39
    assert item["transcript_preview"] == "use api_key=[REDACTED] and continue"


def test_format_recent_displays_safe_previews(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))
    voice_bench.append_event(
        {
            "turn_id": "voice-1",
            "stage": "stt",
            "platform": "telegram",
            "chat_id": "123",
            "elapsed_ms": 220,
            "transcript": "Hermes, cât face 2 plus 3?",
        }
    )
    voice_bench.append_event(
        {
            "turn_id": "voice-1",
            "stage": "agent",
            "platform": "telegram",
            "chat_id": "123",
            "elapsed_ms": 1200,
            "response": "5",
        }
    )

    output = voice_bench.format_recent("telegram", "123", limit=1)

    assert "total=1420ms" in output
    assert "heard: Hermes, cât face 2 plus 3?" in output
    assert "reply: 5" in output


def test_recent_events_ignores_malformed_rows_and_filters_chat(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))
    path.write_text(
        "\n".join(
            [
                "{not-json",
                json.dumps({"turn_id": "wrong", "platform": "telegram", "chat_id": "999"}),
                json.dumps({"turn_id": "ok", "platform": "telegram", "chat_id": "123"}),
            ]
        ),
        encoding="utf-8",
    )

    events = voice_bench.recent_events(platform="telegram", chat_id="123")

    assert [event["turn_id"] for event in events] == ["ok"]


def test_format_recent_reports_unavailable_on_read_failure(monkeypatch):
    class BrokenPath:
        def exists(self):
            return True

        def read_text(self, **_kwargs):
            raise OSError("permission denied")

    monkeypatch.setattr(voice_bench, "bench_path", lambda: BrokenPath())

    assert voice_bench.format_recent("telegram", "123") == (
        "Voice bench unavailable: telemetry read failed."
    )
