import json
from pathlib import Path

from hermes_cli.cockpit_quota import build_quota_recommendation, get_quota_recommendation_payload


def test_build_quota_recommendation_thresholds():
    assert build_quota_recommendation(39_999) is None

    heads_up = build_quota_recommendation(40_000)
    assert heads_up is not None
    assert heads_up["level"] == "heads_up"
    assert heads_up["threshold"] == 40_000
    assert heads_up["dedupe_key"] == "quota-session-reset-heads_up-40000"

    recommend = build_quota_recommendation(70_000)
    assert recommend is not None
    assert recommend["level"] == "recommend"
    assert "fresh session" in recommend["message"].lower()

    strong = build_quota_recommendation(100_000)
    assert strong is not None
    assert strong["level"] == "strong"
    assert "expensive" in strong["message"].lower()

    urgent = build_quota_recommendation(130_000)
    assert urgent is not None
    assert urgent["level"] == "urgent"
    assert "very expensive" in urgent["message"].lower()


def test_get_quota_recommendation_payload_uses_highest_recent_prompt_tokens(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    sessions_file = sessions_dir / "sessions.json"
    sessions_file.write_text(json.dumps({
        "low": {
            "session_id": "low-session",
            "updated_at": "2026-05-14T10:00:00",
            "platform": "discord",
            "display_name": "low",
            "last_prompt_tokens": 12_000,
        },
        "high": {
            "session_id": "high-session",
            "updated_at": "2026-05-14T12:00:00",
            "platform": "discord",
            "display_name": "Personal Assistant / #hermes",
            "last_prompt_tokens": 108_865,
        },
    }))

    payload = get_quota_recommendation_payload(hermes_home=tmp_path)

    assert payload["schema_version"] == 1
    assert payload["recommendation"] is not None
    assert payload["recommendation"]["level"] == "strong"
    assert payload["recommendation"]["prompt_tokens"] == 108_865
    assert payload["recommendation"]["session_id"] == "high-session"
    assert payload["recommendation"]["platform"] == "discord"
    assert payload["recommendation"]["display_name"] == "Personal Assistant / #hermes"
    assert payload["read_only"] is True
    assert payload["actions_enabled"] is False


def test_get_quota_recommendation_payload_returns_none_under_threshold(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "sessions.json").write_text(json.dumps({
        "quiet": {
            "session_id": "quiet-session",
            "updated_at": "2026-05-14T12:00:00",
            "platform": "discord",
            "display_name": "quiet",
            "last_prompt_tokens": 39_999,
        }
    }))

    payload = get_quota_recommendation_payload(hermes_home=tmp_path)

    assert payload["recommendation"] is None
    assert payload["max_prompt_tokens"] == 39_999
