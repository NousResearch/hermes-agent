from __future__ import annotations

from hermes_cli import kanban_db as kb


def test_archive_verification_blocks_exact_leftover_path(tmp_path):
    intake = tmp_path / "MOVE"
    archive = tmp_path / "ARCHIVE" / "batch-001"
    intake.mkdir()
    archive.mkdir(parents=True)

    leftover = intake / "source-a.mp4"
    leftover.write_text("raw", encoding="utf-8")
    (archive / "source-a.mp4").write_text("archived", encoding="utf-8")
    (archive / "README.md").write_text(
        "source split\nreview gate\npreserved subfolder artifacts\n"
        "final synthesis/review artifacts\n",
        encoding="utf-8",
    )

    result = kb.verify_archive_move_contract(
        source_dir=intake,
        archive_dir=archive,
        expected_items=["source-a.mp4"],
        require_empty_source=True,
    )

    assert result["ok"] is False
    assert {issue["code"] for issue in result["issues"]} == {"source_not_empty"}
    assert result["issues"][0]["path"] == str(leftover)


def test_deploy_archive_metadata_rejects_noncanonical_invisible_zip_path():
    result = kb.verify_deploy_artifact_contract(
        artifact_path=r"\\d\\packages\\deploy.zip",
        required_entries=["manifest.json"],
    )

    codes = {issue["code"] for issue in result["issues"]}
    assert result["ok"] is False
    assert "noncanonical_path" in codes
    assert "host_visible_missing" in codes
