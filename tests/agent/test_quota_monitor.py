import json
import time


def test_record_rate_limit_accepts_inner_openai_error_dict(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_QUOTA_STATE_PATH", str(tmp_path / "quota_state.json"))

    from agent.quota_monitor import read_quota_state, record_rate_limit

    reset_at = int(time.time()) + 3600
    record_rate_limit(
        provider="openai-codex",
        model="gpt-5.5",
        status_code=429,
        error_body={
            "type": "usage_limit_reached",
            "message": "The usage limit has been reached",
            "plan_type": "plus",
            "resets_at": reset_at,
            "resets_in_seconds": 3600,
        },
    )

    entry = read_quota_state()["openai-codex"]
    assert entry["error_type"] == "usage_limit_reached"
    assert entry["plan_type"] == "plus"
    assert entry["resets_at"] == reset_at
    assert entry["resets_in_seconds"] == 3600


def test_record_rate_limit_accepts_wrapped_error_json_string(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_QUOTA_STATE_PATH", str(tmp_path / "quota_state.json"))

    from agent.quota_monitor import read_quota_state, record_rate_limit

    reset_at = int(time.time()) + 7200
    record_rate_limit(
        provider="openai-codex",
        model="gpt-5.5",
        status_code=429,
        error_body=json.dumps(
            {
                "error": {
                    "type": "usage_limit_reached",
                    "plan_type": "plus",
                    "resets_at": reset_at,
                    "resets_in_seconds": 7200,
                }
            }
        ),
    )

    entry = read_quota_state()["openai-codex"]
    assert entry["error_type"] == "usage_limit_reached"
    assert entry["plan_type"] == "plus"
    assert entry["resets_at"] == reset_at
    assert entry["resets_in_seconds"] == 7200


def test_format_quota_report_includes_reset_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_QUOTA_STATE_PATH", str(tmp_path / "quota_state.json"))

    from agent.quota_monitor import (
        format_quota_report,
        record_fallback_notification,
        record_rate_limit,
    )

    record_fallback_notification("openai-codex", "custom:kimi-coding-shim")
    record_rate_limit(
        provider="openai-codex",
        model="gpt-5.5",
        status_code=429,
        error_body={
            "type": "usage_limit_reached",
            "plan_type": "plus",
            "resets_at": int(time.time()) + 3600,
        },
    )

    report = format_quota_report()
    assert "**openai-codex** (plus) — usage_limit_reached" in report
    assert "resets in" in report
    assert "fell back to custom:kimi-coding-shim" in report


def test_get_quota_summary_uses_resets_in_seconds_when_resets_at_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_QUOTA_STATE_PATH", str(tmp_path / "quota_state.json"))

    from agent.quota_monitor import get_quota_summary_for_provider, record_rate_limit

    record_rate_limit(
        provider="openai-codex",
        model="gpt-5.5",
        status_code=429,
        error_body={
            "type": "usage_limit_reached",
            "plan_type": "plus",
            "resets_in_seconds": 3600,
        },
    )

    summary = get_quota_summary_for_provider("openai-codex")
    assert summary is not None
    assert summary.startswith("openai-codex quota resets in")
