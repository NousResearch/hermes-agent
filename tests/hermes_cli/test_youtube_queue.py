"""Tests for the local YouTube queue/readiness control plane."""

import csv
import io
import json

import pytest

from hermes_cli import youtube_queue as yq


ALL_SCRIPTURE_CHECKS = {
    "video_file": True,
    "thumbnail": True,
    "title": True,
    "description": True,
    "captions": True,
    "sources_or_scripture_refs": True,
    "fact_check": False,
    "human_approval": True,
}

ALL_NEWSLISH_CHECKS = {
    "video_file": True,
    "thumbnail": True,
    "title": True,
    "description": True,
    "captions": True,
    "sources_or_scripture_refs": True,
    "fact_check": True,
    "human_approval": True,
}


def asset_paths(tmp_path):
    paths = {
        "video": tmp_path / "video.mp4",
        "thumbnail": tmp_path / "thumb.png",
        "captions": tmp_path / "captions.vtt",
    }
    for path in paths.values():
        path.write_text("fixture", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def make_scripture_item(tmp_path, **overrides):
    data = {
        "channel_id": "scripturedepth",
        "title": "Psalm 23 hope in one minute",
        "description": "A concise ScriptureDepth description.",
        "format": "short",
        "tags": ["Bible", "Psalm 23", "Shorts"],
        "source_refs": ["Psalm 23"],
        "asset_paths": asset_paths(tmp_path),
        "checks": dict(ALL_SCRIPTURE_CHECKS),
        "review_status": "approved",
        "risk": "low",
    }
    data.update(overrides)
    return yq.create_item(data)


def test_publish_readiness_blocks_empty_assets_even_when_checks_true(_isolate_hermes_home, tmp_path):
    item = make_scripture_item(tmp_path, asset_paths={"video": None, "thumbnail": None, "captions": None})

    readiness = yq.publish_readiness(item["id"])

    assert readiness["ready"] is False
    assert "asset_paths.video is required" in readiness["blockers"]
    assert "asset_paths.thumbnail is required" in readiness["blockers"]
    assert "asset_paths.captions is required" in readiness["blockers"]
    assert readiness["publish_enabled"] is False


def test_checks_reject_truthy_strings(_isolate_hermes_home):
    with pytest.raises(ValueError, match="checks.title must be a boolean"):
        yq.create_item({
            "channel_id": "newslish",
            "title": "Truthy string bypass",
            "checks": {"title": "false"},
        })


def test_manifest_import_rejects_non_boolean_json_checks(_isolate_hermes_home):
    manifest = json.dumps({
        "items": [
            {
                "channel_id": "newslish",
                "title": "Bad checks manifest",
                "checks": {"title": "false"},
            }
        ]
    })

    result = yq.import_manifest(manifest, "json")

    assert result["created_count"] == 0
    assert result["error_count"] == 1
    assert "checks.title must be a boolean" in result["errors"][0]["error"]


def test_newslish_source_refs_must_be_urls(_isolate_hermes_home, tmp_path):
    item = yq.create_item({
        "channel_id": "newslish",
        "title": "News update in one minute",
        "description": "A concise Newslish description.",
        "format": "short",
        "tags": ["news", "explainer"],
        "source_refs": ["not-a-url"],
        "asset_paths": asset_paths(tmp_path),
        "checks": dict(ALL_NEWSLISH_CHECKS),
        "review_status": "approved",
        "risk": "low",
    })

    readiness = yq.publish_readiness(item["id"])

    assert readiness["ready"] is False
    assert "Newslish source_refs must be http(s) URLs" in readiness["blockers"]
    assert "Newslish requires source URLs" in readiness["blockers"]


def test_schedule_requires_iso_datetime_and_valid_timezone(_isolate_hermes_home, tmp_path):
    item = make_scripture_item(tmp_path, 
        visibility="scheduled",
        scheduled_for="tomorrow maybe",
        timezone="Mars/Olympus",
    )

    readiness = yq.publish_readiness(item["id"])

    assert readiness["ready"] is False
    assert "scheduled_for must be an ISO datetime" in readiness["blockers"]
    assert "timezone must be a valid IANA timezone" in readiness["blockers"]


def test_ready_transition_is_structurally_gated(_isolate_hermes_home, tmp_path):
    paths = asset_paths(tmp_path)
    item = make_scripture_item(tmp_path, asset_paths={"video": None, "thumbnail": paths["thumbnail"], "captions": paths["captions"]})

    with pytest.raises(ValueError, match="asset_paths.video is required"):
        yq.patch_item(item["id"], {"status": "ready"})


def test_publish_plan_remains_dry_run_for_ready_item(_isolate_hermes_home, tmp_path):
    paths = asset_paths(tmp_path)
    item = make_scripture_item(tmp_path, asset_paths=paths)

    readiness = yq.publish_readiness(item["id"])
    plan = yq.publish_plan(item["id"])

    assert readiness["ready"] is True
    assert plan["publish_enabled"] is False
    assert plan["youtube_api_call_allowed"] is False
    assert plan["side_effects"] == []
    assert plan["payload_preview"]["video_path"] == paths["video"]


def test_publish_readiness_distinguishes_missing_and_nonexistent_asset_paths(_isolate_hermes_home, tmp_path):
    missing_video = tmp_path / "missing.mp4"
    item = make_scripture_item(tmp_path, asset_paths={"video": str(missing_video), "thumbnail": None, "captions": str(tmp_path / "captions.vtt")})

    readiness = yq.publish_readiness(item["id"])

    assert f"asset_paths.video does not exist: {missing_video}" in readiness["blockers"]
    assert "asset_paths.thumbnail is required" in readiness["blockers"]
    assert readiness["publish_enabled"] is False


def test_manifest_templates_round_trip_csv_and_json(_isolate_hermes_home):
    csv_template = yq.manifest_template("csv")["content"]
    json_template = yq.manifest_template("json")["content"]

    csv_result = yq.import_manifest(csv_template, "csv")
    json_result = yq.import_manifest(
        json_template
        .replace("Psalm 23 hope in one minute", "Psalm 23 hope in one minute JSON")
        .replace("What changed today in one minute", "What changed today in one minute JSON"),
        "json",
    )

    assert csv_result["created_count"] == 2, csv_result
    assert csv_result["error_count"] == 0
    assert json_result["created_count"] == 2, json_result
    assert json_result["error_count"] == 0


def test_manifest_import_rejects_unknown_json_and_csv_fields(_isolate_hermes_home):
    bad_json = json.dumps({"items": [{"channel_id": "newslish", "title": "Bad unknown", "surprise": "nope"}]})
    with pytest.raises(ValueError, match="unknown manifest field"):
        yq.import_manifest(bad_json, "json")

    bad_csv = "channel_id,title,surprise\nnewslish,Bad unknown,nope\n"
    with pytest.raises(ValueError, match="unknown manifest field"):
        yq.import_manifest(bad_csv, "csv")


def test_manifest_import_reports_duplicate_ids_and_titles(_isolate_hermes_home):
    manifest = json.dumps({
        "items": [
            {"id": "dup-id", "channel_id": "scripturedepth", "title": "Duplicate title"},
            {"id": "dup-id", "channel_id": "scripturedepth", "title": "Different title"},
            {"channel_id": "scripturedepth", "title": "Duplicate title"},
        ]
    })

    result = yq.import_manifest(manifest, "json")

    assert result["created_count"] == 1
    assert result["error_count"] == 2
    assert "duplicate queue item id" in result["errors"][0]["error"]
    assert "duplicate queue title" in result["errors"][1]["error"]


def test_audit_records_create_patch_archive_and_import(_isolate_hermes_home, tmp_path):
    item = make_scripture_item(tmp_path, title="Audit smoke")
    yq.patch_item(item["id"], {"notes": "patched"})
    yq.archive_item(item["id"])

    template = yq.manifest_template("csv")["content"]
    reader = csv.DictReader(io.StringIO(template))
    row = next(reader)
    row["title"] = "Audit import"
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerow(row)
    yq.import_manifest(output.getvalue(), "csv")

    actions = [event["action"] for event in yq.read_audit(limit=20)["events"]]

    assert "queue.create" in actions
    assert "queue.patch" in actions
    assert actions.count("queue.patch") >= 2  # explicit patch plus archive via patch_item
    assert "queue.batch_import" in actions
