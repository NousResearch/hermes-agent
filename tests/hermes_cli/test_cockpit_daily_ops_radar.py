"""Tests for BIF-526 Cockpit Daily Ops Radar endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


RADAR_JOB_ID = "a82830911bcd"


@pytest.fixture()
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.config import get_hermes_home
    from hermes_cli import web_server

    home = Path(get_hermes_home())
    out_dir = home / "cron" / "output" / RADAR_JOB_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "2026-05-14_08-15-24.md").write_text(
        "\n".join(
            [
                "# Cron Job: Hermes Agent Daily Ops Radar",
                "",
                f"**Job ID:** {RADAR_JOB_ID}",
                "**Run Time:** 2026-05-14 08:15:24",
                "**Mode:** no_agent (script)",
                "",
                "---",
                "",
                "Hermes daily ops radar: 7 relevant upstream change(s) in 12 commit(s) behind origin/main.",
                "Repo: /Users/marco/.hermes/hermes-agent",
                "💬 Chat/gateway: abc123456 fix(gateway): keep reconnect loop alive; def987654 feat(slack): token=shh-secret-value must redact",
                "🧭 provider/model routing/breaking: fedcba987 docs: safe provider note (+2 more)",
                "Top commits to review:",
                "• abc123456 fix(gateway): keep reconnect loop alive — gateway/platforms/qqbot/adapter.py, tests/gateway/test_qqbot.py",
                "• def987654 feat(slack): support !cmd with API_KEY=raw-secret — cli.py",
                "Compare: git -C /Users/marco/.hermes/hermes-agent log --oneline main..origin/main",
                "State key: ghp_thisShouldNotLeak1234567890",
            ]
        ),
        encoding="utf-8",
    )
    jobs_path = home / "cron" / "jobs.json"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": RADAR_JOB_ID,
                        "name": "Hermes Agent Daily Ops Radar",
                        "script": "hermes_daily_ops_radar.py",
                        "enabled": True,
                        "state": "scheduled",
                        "schedule_display": "30 6 * * *",
                        "next_run_at": "2026-05-15T06:30:00-04:00",
                        "last_run_at": "2026-05-14T08:15:24.569620-04:00",
                        "last_status": "ok",
                        "last_error": "raw-secret-error-should-not-leak",
                        "origin": {"chat_id": "1503821368045863027", "chat_name": "#hermes"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return TestClient(web_server.app)


@pytest.fixture()
def auth_headers():
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN

    return {_SESSION_HEADER_NAME: _SESSION_TOKEN}


def _json_text(value) -> str:
    return json.dumps(value, sort_keys=True)


def test_daily_ops_radar_requires_dashboard_session_token(client):
    response = client.get("/api/cockpit/daily-ops-radar")

    assert response.status_code == 401


def test_daily_ops_radar_returns_latest_read_only_summary(client, auth_headers):
    response = client.get("/api/cockpit/daily-ops-radar", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["read_only"] is True
    assert body["actions_enabled"] is False
    assert body["upgrade"]["prepare_review"]["enabled"] is False
    assert body["upgrade"]["prepare_review"]["mutates"] is False
    assert body["job"]["id"] == RADAR_JOB_ID
    assert body["job"]["script"] == "hermes_daily_ops_radar.py"
    assert body["job"]["status"] == "ok"
    assert body["job"]["next_run_at"] == "2026-05-15T06:30:00-04:00"
    assert body["summary"]["last_run"] == "2026-05-14 08:15:24"
    assert body["summary"]["behind_count"] == 12
    assert body["summary"]["relevant_change_count"] == 7
    assert body["summary"]["compare_command"] == "git log --oneline main..origin/main"
    assert body["upgrade"]["recommendation"]["question"] == "Should we upgrade?"
    assert body["upgrade"]["recommendation"]["label"] == "Prepare review first"
    assert body["upgrade"]["recommendation"]["risk_level"] in {"medium", "high"}
    assert "Do not upgrade blindly" in body["upgrade"]["recommendation"]["rationale"]
    assert body["upgrade"]["recommendation"]["freshness"] == "fresh"
    assert body["upgrade"]["recommendation"]["basis"] == "2026-05-14 08:15:24"
    assert [c["label"] for c in body["summary"]["categories"]] == [
        "Chat/gateway",
        "provider/model routing/breaking",
    ]
    assert body["summary"]["top_commits"][0]["sha"] == "abc123456"
    assert body["summary"]["upgrade_brief"]["question"] == "What major improvements would we get?"
    assert body["upgrade"]["brief"] == body["summary"]["upgrade_brief"]
    assert "More reliable messaging" in body["upgrade"]["brief"]["headline"]
    assert "Better model/provider routing" in body["upgrade"]["brief"]["headline"]
    assert "Why this matters" in body["upgrade"]["brief"]["why_this_matters"]


def test_daily_ops_radar_extracts_plain_language_upgrade_brief_from_markdown():
    from hermes_cli.cockpit_daily_ops import parse_daily_ops_radar_markdown

    summary = parse_daily_ops_radar_markdown(
        "\n".join(
            [
                "Hermes daily ops radar: 9 relevant upstream change(s) in 14 commit(s) behind origin/main.",
                "💬 Chat/gateway: abc123456 fix(gateway): keep reconnect loop alive; def987654 feat(slack): improve queued busy-message delivery",
                "🧪 tests/build: fedcba987 test(cockpit): cover upgrade radar layout (+2 more)",
                "Top commits to review:",
                "• abc123456 fix(gateway): keep reconnect loop alive — gateway/platforms/slack.py",
            ]
        )
    )

    brief = summary["upgrade_brief"]
    assert brief["question"] == "What major improvements would we get?"
    assert [group["label"] for group in brief["groups"]] == ["Chat/gateway", "tests/build"]
    assert "More reliable messaging and gateway behavior" in brief["groups"][0]["summary"]
    assert "keep reconnect loop alive" in brief["groups"][0]["summary"]
    assert "abc123456" not in _json_text(brief)
    assert "— gateway/platforms" not in _json_text(brief)


def test_daily_ops_radar_sanitizes_secrets_and_huge_raw_logs(client, auth_headers):
    response = client.get("/api/cockpit/daily-ops-radar", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    text = _json_text(body)
    assert "shh-secret-value" not in text
    assert "raw-secret" not in text
    assert "ghp_thisShouldNotLeak" not in text
    assert "1503821368045863027" not in text
    assert "last_error" not in text
    assert len(body["raw_excerpt"]) <= 2400
    assert "State key:" not in body["raw_excerpt"]
    assert "/Users/" not in text
    assert "/opt/" not in text
    assert "/tmp/" not in text
    assert "git -C" not in text
    assert "shh-secret-value" not in _json_text(body["upgrade"]["brief"])
    assert "API_KEY=raw-secret" not in _json_text(body["upgrade"]["brief"])
    assert "/Users/" not in _json_text(body["upgrade"]["brief"])


def test_daily_ops_radar_endpoint_does_not_mutate_upgrade_or_cron_state(client, auth_headers, monkeypatch):
    import subprocess

    def fail_run(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("daily ops radar endpoint must not invoke subprocess")

    monkeypatch.setattr(subprocess, "run", fail_run)
    response = client.get("/api/cockpit/daily-ops-radar", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["upgrade"]["prepare_review"]["status"] == "disabled_no_safe_preflight_endpoint"
    assert body["upgrade"]["prepare_review"]["method"] == "GET"
    assert body["upgrade"]["recommendation"]["label"] == "Prepare review first"
    assert body["upgrade"]["brief"]["question"] == "What major improvements would we get?"


def test_upgrade_recommendation_waits_when_radar_missing_or_stale():
    from hermes_cli.cockpit_daily_ops import RADAR_STALE_AFTER_SECONDS, build_upgrade_recommendation

    missing = build_upgrade_recommendation({}, {"status": "ok"}, "cron_output_missing", now=1000)
    assert missing["label"] == "Wait"
    assert missing["risk_level"] == "unknown"
    assert "not trustworthy" in missing["rationale"]

    stale = build_upgrade_recommendation(
        {"behind_count": 1, "relevant_change_count": 0, "categories": [], "source_mtime": 1000},
        {"status": "ok"},
        "cron_output_latest_markdown",
        now=1000 + RADAR_STALE_AFTER_SECONDS + 1,
    )
    assert stale["label"] == "Wait"
    assert stale["freshness"] == "stale"


def test_upgrade_recommendation_high_or_risky_changes_prepare_review_first():
    from hermes_cli.cockpit_daily_ops import build_upgrade_recommendation

    current_live_shape = build_upgrade_recommendation(
        {
            "last_run": "2026-05-14 09:00:00",
            "behind_count": 347,
            "relevant_change_count": 269,
            "categories": [{"label": "provider/model routing"}],
            "source_mtime": 1000,
        },
        {"status": "ok"},
        "cron_output_latest_markdown",
        now=1200,
    )
    assert current_live_shape["label"] == "Prepare review first"
    assert current_live_shape["risk_level"] == "high"
    assert "Do not upgrade blindly" in current_live_shape["rationale"]
    assert "347 behind / 269 relevant" in current_live_shape["signals"]

    risky_low_count = build_upgrade_recommendation(
        {
            "behind_count": 2,
            "relevant_change_count": 1,
            "categories": [{"label": "security"}],
            "source_mtime": 1000,
        },
        {"status": "ok"},
        "cron_output_latest_markdown",
        now=1200,
    )
    assert risky_low_count["label"] == "Prepare review first"
    assert risky_low_count["risk_level"] == "medium"


def test_daily_ops_radar_profile_home_falls_back_to_default_root(monkeypatch, tmp_path):
    from hermes_cli.cockpit_daily_ops import get_daily_ops_radar_payload

    root = tmp_path / "hermes-root"
    profile_home = root / "profiles" / "cockpit"
    out_dir = root / "cron" / "output" / RADAR_JOB_ID
    profile_home.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    (out_dir / "2026-05-14_08-15-24.md").write_text(
        "\n".join(
            [
                "# Cron Job: Hermes Agent Daily Ops Radar",
                "**Job ID:** a82830911bcd",
                "**Run Time:** 2026-05-14 08:15:24",
                "Hermes daily ops radar: 2 relevant upstream change(s) in 4 commit(s) behind origin/main.",
                "💬 Chat/gateway: abc123456 fix(gateway): keep reconnect loop alive",
            ]
        ),
        encoding="utf-8",
    )
    (root / "cron" / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": RADAR_JOB_ID,
                        "name": "Hermes Agent Daily Ops Radar",
                        "script": "hermes_daily_ops_radar.py",
                        "enabled": True,
                        "state": "scheduled",
                        "last_status": "ok",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = get_daily_ops_radar_payload()

    assert payload["source"] == "cron_output_latest_markdown"
    assert payload["job"]["id"] == RADAR_JOB_ID
    assert payload["summary"]["behind_count"] == 4
    assert payload["summary"]["relevant_change_count"] == 2


def test_upgrade_recommendation_low_clean_changes_says_upgrade_now_but_only_review():
    from hermes_cli.cockpit_daily_ops import build_upgrade_recommendation

    recommendation = build_upgrade_recommendation(
        {
            "behind_count": 3,
            "relevant_change_count": 1,
            "categories": [{"label": "docs"}],
            "source_mtime": 1000,
        },
        {"status": "ok"},
        "cron_output_latest_markdown",
        now=1200,
    )

    assert recommendation["label"] == "Upgrade now"
    assert recommendation["risk_level"] == "low"
    assert "safe to prepare upgrade review" in recommendation["rationale"]
