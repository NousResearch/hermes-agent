"""Tests for voice benchmark JSONL telemetry."""

from __future__ import annotations

import json
import stat

from gateway import voice_bench


def test_append_event_redacts_and_renames_text_fields(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))
    monkeypatch.setenv("HERMES_VOICE_BENCH_SYNC", "1")

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
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_append_event_redacts_standalone_github_tokens(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))
    monkeypatch.setenv("HERMES_VOICE_BENCH_SYNC", "1")
    token = "ghp_abcdefghijklmnopqrstuvwxyz123456"

    voice_bench.append_event(
        {
            "turn_id": "voice-token",
            "stage": "agent",
            "platform": "telegram",
            "chat_id": "123",
            "response": f"standalone token {token}",
        }
    )

    item = json.loads(path.read_text(encoding="utf-8"))
    assert token not in item["response_preview"]
    assert "[REDACTED]" in item["response_preview"]


def test_append_event_compacts_large_telemetry_file(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))
    monkeypatch.setenv("HERMES_VOICE_BENCH_SYNC", "1")
    monkeypatch.setattr(voice_bench, "MAX_FILE_BYTES", 20)
    monkeypatch.setattr(voice_bench, "MAX_EVENTS", 2)
    path.write_text(
        "\n".join(
            json.dumps({"turn_id": f"old-{idx}", "stage": "stt"})
            for idx in range(4)
        )
        + "\n",
        encoding="utf-8",
    )

    voice_bench.append_event(
        {"turn_id": "new", "stage": "stt", "platform": "telegram", "chat_id": "123"}
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["turn_id"] for row in rows] == ["old-3", "new"]


def test_append_event_serializes_unknown_values(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))
    monkeypatch.setenv("HERMES_VOICE_BENCH_SYNC", "1")

    voice_bench.append_event(
        {
            "turn_id": "voice-object",
            "stage": "agent",
            "platform": "telegram",
            "chat_id": "123",
            "extra": object(),
        }
    )

    item = json.loads(path.read_text(encoding="utf-8"))
    assert item["turn_id"] == "voice-object"


def test_append_event_redacts_arbitrary_secret_fields(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))
    monkeypatch.setenv("HERMES_VOICE_BENCH_SYNC", "1")

    voice_bench.append_event(
        {
            "turn_id": "voice-secret-field",
            "stage": "agent",
            "platform": "telegram",
            "chat_id": "123",
            "api_key": "sk-test-secret",
            "metadata": {"authorization": "bearer abcdefghijklmnop"},
            "note": "standalone github_pat_abcdefghijklmnopqrstuvwxyz",
        }
    )

    item = json.loads(path.read_text(encoding="utf-8"))
    assert item["api_key"] == "[REDACTED]"
    assert item["metadata"]["authorization"] == "[REDACTED]"
    assert "github_pat_abcdefghijklmnopqrstuvwxyz" not in item["note"]


def test_format_recent_displays_safe_previews(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))
    monkeypatch.setenv("HERMES_VOICE_BENCH_SYNC", "1")
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


def test_format_recent_redacts_legacy_raw_text_fields(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))
    rows = [
        {
            "turn_id": "voice-legacy",
            "stage": "stt",
            "platform": "telegram",
            "chat_id": "123",
            "elapsed_ms": 100,
            "transcript": "token=ghp_legacysecretvalue",
        },
        {
            "turn_id": "voice-legacy",
            "stage": "agent",
            "platform": "telegram",
            "chat_id": "123",
            "elapsed_ms": 100,
            "response": "authorization: bearer abcdefghijklmnop",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    output = voice_bench.format_recent("telegram", "123", limit=1)

    assert "ghp_legacysecretvalue" not in output
    assert "abcdefghijklmnop" not in output
    assert "token=[REDACTED]" in output
    assert "authorization=[REDACTED]" in output


def test_format_recent_aggregates_duplicate_stage_timings(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))
    rows = [
        {"turn_id": "voice-dup", "stage": "stt", "platform": "telegram", "chat_id": "123", "elapsed_ms": 100},
        {"turn_id": "voice-dup", "stage": "stt", "platform": "telegram", "chat_id": "123", "elapsed_ms": 150},
        {"turn_id": "voice-dup", "stage": "agent", "platform": "telegram", "chat_id": "123", "elapsed_ms": 200},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    output = voice_bench.format_recent("telegram", "123", limit=1)

    assert "total=450ms" in output
    assert "stt=250ms" in output



def test_format_recent_includes_modular_benchmark_fields(tmp_path, monkeypatch):
    path = tmp_path / "voice_bench.jsonl"
    monkeypatch.setenv("HERMES_VOICE_BENCH_PATH", str(path))
    monkeypatch.setenv("HERMES_VOICE_BENCH_SYNC", "1")
    voice_bench.append_event({
        "turn_id": "voice-modular",
        "stage": "stt",
        "platform": "telegram",
        "chat_id": "123",
        "mode": "modular",
        "stt_provider": "deepgram",
        "tts_provider": "cartesia",
        "stt_model": "nova-3",
        "elapsed_ms": 120,
        "transcript": "hello",
    })
    voice_bench.append_event({
        "turn_id": "voice-modular",
        "stage": "brain",
        "platform": "telegram",
        "chat_id": "123",
        "elapsed_ms": 80,
    })
    voice_bench.append_event({
        "turn_id": "voice-modular",
        "stage": "tts",
        "platform": "telegram",
        "chat_id": "123",
        "tts_model": "sonic-2",
        "first_audio_ms": 310,
        "elapsed_ms": 190,
        "error": "authorization bearer abcdefghijklmnop",
    })

    row_text = path.read_text(encoding="utf-8")
    assert "abcdefghijklmnop" not in row_text
    output = voice_bench.format_recent("telegram", "123", limit=1)
    assert "mode=modular" in output
    assert "stt_provider=deepgram" in output
    assert "tts_provider=cartesia" in output
    assert "first_audio=310ms" in output
    assert "brain=80ms" in output

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

        def open(self, *_args, **_kwargs):
            raise OSError("permission denied")

    monkeypatch.setattr(voice_bench, "bench_path", lambda: BrokenPath())

    assert voice_bench.format_recent("telegram", "123") == (
        "Voice bench unavailable: telemetry read failed."
    )


def test_format_recent_reports_unavailable_on_write_failure(monkeypatch):
    monkeypatch.setattr(voice_bench, "_LAST_WRITE_ERROR", "disk full")

    assert voice_bench.format_recent("telegram", "123") == (
        "Voice bench unavailable: telemetry write failed."
    )
