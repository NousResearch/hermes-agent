from pathlib import Path

import pytest

from hermes_cli.ops_social_status import read_social_platform_history, read_social_platform_status, write_manual_social_platform_status


def test_social_platform_status_defaults_when_missing(tmp_path):
    result = read_social_platform_status(tmp_path / "missing.json")

    assert result["ok"] is True
    assert result["mode"] == "local_read_only"
    assert result["warning"] is None
    platforms = {item["platform"]: item for item in result["platforms"]}
    assert platforms["YouTube"]["published"] == "Needs sync"
    assert platforms["TikTok"]["scheduled"] == "0"
    assert platforms["TikTok"]["status"] == "blocked"


def test_social_platform_status_merges_local_snapshot(tmp_path):
    path = tmp_path / "social-platform-status.json"
    path.write_text(
        """
        {
          "updated_at": "2026-05-26T09:30:00+08:00",
          "source": "manual-read-only-sync",
          "platforms": [
            {
              "platform": "YouTube",
              "published": 12,
              "scheduled": 3,
              "issues_private": "1 private review item",
              "status": "ok",
              "source": "YouTube local cache"
            },
            {
              "platform": "Bluesky",
              "published": 4,
              "scheduled": 0,
              "issues_private": "None",
              "status": "ok"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    result = read_social_platform_status(path)

    assert result["updated_at"] == "2026-05-26T09:30:00+08:00"
    platforms = {item["platform"]: item for item in result["platforms"]}
    assert platforms["YouTube"]["published"] == "12"
    assert platforms["YouTube"]["scheduled"] == "3"
    assert platforms["YouTube"]["issues_private"] == "1 private review item"
    assert platforms["Facebook"]["published"] == "Needs sync"
    assert platforms["Bluesky"]["published"] == "4"


def test_social_platform_status_invalid_json_fails_closed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    result = read_social_platform_status(path)

    assert result["ok"] is True
    assert result["warning"]
    assert {item["platform"] for item in result["platforms"]} >= {"YouTube", "Facebook", "Instagram", "TikTok"}


def test_write_manual_social_platform_status_writes_local_snapshot(tmp_path):
    path = tmp_path / "state" / "ops-center" / "social-platform-status.json"
    result = write_manual_social_platform_status(
        {
            "source": "manual-test",
            "platforms": [
                {
                    "platform": "YouTube",
                    "published": "12",
                    "scheduled": "3",
                    "issues_private": "1 private review item",
                    "readiness": "Manual count only.",
                    "source": "Manual dashboard entry",
                    "status": "ok",
                    "last_checked_at": "2026-05-26T19:00:00+08:00",
                }
            ],
        },
        path,
        history_path=tmp_path / "history.jsonl",
    )

    assert path.exists()
    assert result["ok"] is True
    assert result["source"] == "manual-test"
    platforms = {item["platform"]: item for item in result["platforms"]}
    assert platforms["YouTube"]["published"] == "12"
    assert platforms["YouTube"]["scheduled"] == "3"
    assert platforms["YouTube"]["status"] == "ok"
    assert platforms["YouTube"]["last_checked_at"] == "2026-05-26T19:00:00+08:00"
    assert platforms["Facebook"]["published"] == "Needs sync"


def test_write_manual_social_platform_status_rejects_invalid_payload(tmp_path):
    with pytest.raises(ValueError, match="platforms"):
        write_manual_social_platform_status({"source": "bad"}, tmp_path / "out.json")


def test_write_manual_social_platform_status_normalizes_unknown_status(tmp_path):
    result = write_manual_social_platform_status(
        {"platforms": [{"platform": "Instagram", "status": "mystery", "last_checked_at": "2026-05-18T00:00:00+00:00"}]},
        tmp_path / "social.json",
        history_path=tmp_path / "history.jsonl",
    )

    platforms = {item["platform"]: item for item in result["platforms"]}
    assert platforms["Instagram"]["status"] == "needs_review"
    assert platforms["Instagram"]["last_checked_at"] == "2026-05-18T00:00:00+00:00"


def test_write_manual_social_platform_status_appends_history(tmp_path):
    status_path = tmp_path / "social.json"
    history_path = tmp_path / "history.jsonl"

    write_manual_social_platform_status(
        {"source": "first", "platforms": [{"platform": "YouTube", "published": 1, "scheduled": 0, "status": "ok"}]},
        status_path,
        history_path=history_path,
    )
    write_manual_social_platform_status(
        {"source": "second", "platforms": [{"platform": "YouTube", "published": 2, "scheduled": 1, "status": "needs_review"}]},
        status_path,
        history_path=history_path,
    )

    history = read_social_platform_history(history_path, limit=5)

    assert history["ok"] is True
    assert len(history["events"]) == 2
    assert history["events"][0]["source"] == "second"
    assert history["events"][0]["status_counts"]["needs_review"] == 1
    assert history["events"][1]["source"] == "first"


def test_read_social_platform_history_skips_bad_jsonl_lines(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"source":"good","platform_count":1}\nnot-json\n', encoding="utf-8")

    history = read_social_platform_history(path)

    assert len(history["events"]) == 1
    assert history["events"][0]["source"] == "good"
