"""Tests for the Google Drive pipeline plugin package."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.google_drive_pipeline import register
from plugins.google_drive_pipeline.cli import google_drive_pipeline_command
from plugins.google_drive_pipeline.models import PublishArtifactRequest
from plugins.google_drive_pipeline.pipeline import GoogleDrivePipeline
from plugins.google_drive_pipeline.store import GoogleDrivePipelineStore


def test_register_adds_cli_only():
    mgr = PluginManager()
    manifest = PluginManifest(name="google_drive_pipeline")
    ctx = PluginContext(manifest, mgr)

    register(ctx)

    assert "google-drive-pipeline" in mgr._cli_commands
    entry = mgr._cli_commands["google-drive-pipeline"]
    assert entry["plugin"] == "google_drive_pipeline"
    assert callable(entry["setup_fn"])
    assert callable(entry["handler_fn"])


def test_store_persists_publish_record(tmp_path):
    store_path = tmp_path / "drive-store.json"
    store = GoogleDrivePipelineStore(store_path)
    store.upsert_record(
        "rec-1",
        {
            "source_key": "sha:folder:file",
            "source_path": "/tmp/report.pdf",
            "file_id": "file-1",
            "folder_cache_key": "root:Reports",
            "folder": {"folder_id": "folder-1", "name": "Reports"},
        },
    )

    reloaded = GoogleDrivePipelineStore(store_path)
    record = reloaded.get_record("rec-1")

    assert record is not None
    assert record["file_id"] == "file-1"
    assert reloaded.get_record_by_source_key("sha:folder:file")["record_id"] == "rec-1"
    assert reloaded.get_cached_folder("root:Reports")["folder_id"] == "folder-1"


def test_publish_artifact_reuses_existing_record(monkeypatch, tmp_path):
    file_path = tmp_path / "report.pdf"
    file_path.write_text("hello")
    store = GoogleDrivePipelineStore(tmp_path / "drive-store.json")
    pipeline = GoogleDrivePipeline(store=store)

    def fake_resolve_folder(**kwargs):
        from plugins.google_drive_pipeline.models import DriveFolderRef
        return DriveFolderRef(folder_id="folder-1", name="Reports")

    def fake_find_existing_file(folder_id, name):
        return {"id": "file-1", "name": name, "webViewLink": "https://drive/file-1"}

    monkeypatch.setattr(pipeline, "resolve_folder", fake_resolve_folder)
    monkeypatch.setattr(pipeline, "_find_existing_file", fake_find_existing_file)
    monkeypatch.setattr(
        pipeline,
        "_refresh_existing_file",
        lambda file_id: {
            "id": file_id,
            "name": "report.pdf",
            "webViewLink": "https://drive/file-1",
        },
    )

    first = pipeline.publish_artifact(
        PublishArtifactRequest(
            file_path=file_path,
            folder_name="Reports",
            duplicate_policy="reuse",
        )
    )
    second = pipeline.publish_artifact(
        PublishArtifactRequest(
            file_path=file_path,
            folder_name="Reports",
            duplicate_policy="reuse",
        )
    )

    assert first.file_id == "file-1"
    assert second.file_id == "file-1"
    assert second.record_id == first.record_id


def test_publish_artifact_versions_on_duplicate(monkeypatch, tmp_path):
    file_path = tmp_path / "report.pdf"
    file_path.write_text("hello")
    store = GoogleDrivePipelineStore(tmp_path / "drive-store.json")
    pipeline = GoogleDrivePipeline(store=store)

    def fake_resolve_folder(**kwargs):
        from plugins.google_drive_pipeline.models import DriveFolderRef
        return DriveFolderRef(folder_id="folder-1", name="Reports")

    monkeypatch.setattr(pipeline, "resolve_folder", fake_resolve_folder)
    monkeypatch.setattr(
        pipeline,
        "_find_existing_file",
        lambda folder_id, name: {"id": "existing-1", "name": name, "webViewLink": "https://drive/existing-1"},
    )
    monkeypatch.setattr(
        pipeline,
        "_run_google_api",
        lambda args: {
            "id": "uploaded-1",
            "name": args[args.index("--name") + 1],
            "webViewLink": "https://drive/uploaded-1",
        },
    )

    result = pipeline.publish_artifact(
        PublishArtifactRequest(
            file_path=file_path,
            folder_name="Reports",
            duplicate_policy="version",
        )
    )

    assert result.file_id == "uploaded-1"
    assert result.file_name.startswith("report-")
    assert result.file_name.endswith(".pdf")


def test_publish_artifact_preserves_parent_folder_on_created_folder(monkeypatch, tmp_path):
    file_path = tmp_path / "report.pdf"
    file_path.write_text("hello")
    store = GoogleDrivePipelineStore(tmp_path / "drive-store.json")
    pipeline = GoogleDrivePipeline(store=store)

    def fake_run_google_api(args):
        if args[:2] == ["drive", "search"]:
            return []
        if args[:2] == ["drive", "create-folder"]:
            return {
                "id": "folder-1",
                "name": "Reports",
                "webViewLink": "https://drive/folder-1",
            }
        if args[:2] == ["drive", "upload"]:
            return {
                "id": "file-1",
                "name": "report.pdf",
                "webViewLink": "https://drive/file-1",
            }
        raise AssertionError(args)

    monkeypatch.setattr(pipeline, "_run_google_api", fake_run_google_api)
    monkeypatch.setattr(pipeline, "_find_existing_file", lambda folder_id, name: None)

    result = pipeline.publish_artifact(
        PublishArtifactRequest(
            file_path=file_path,
            folder_name="Reports",
            parent_folder_id="parent-123",
            create_missing_folder=True,
        )
    )

    assert result.folder.parent_folder_id == "parent-123"
    assert store.get_cached_folder("parent-123:Reports")["parent_folder_id"] == "parent-123"


def test_publish_artifact_reuse_reuploads_when_cached_record_is_stale(monkeypatch, tmp_path):
    file_path = tmp_path / "report.pdf"
    file_path.write_text("hello")
    store = GoogleDrivePipelineStore(tmp_path / "drive-store.json")
    pipeline = GoogleDrivePipeline(store=store)

    def fake_resolve_folder(**kwargs):
        from plugins.google_drive_pipeline.models import DriveFolderRef
        return DriveFolderRef(folder_id="folder-1", name="Reports")

    monkeypatch.setattr(pipeline, "resolve_folder", fake_resolve_folder)

    uploaded = {"count": 0}

    def fake_run_google_api(args):
        if args[:2] == ["drive", "upload"]:
            uploaded["count"] += 1
            return {
                "id": f"uploaded-{uploaded['count']}",
                "name": args[args.index("--name") + 1],
                "webViewLink": f"https://drive/uploaded-{uploaded['count']}",
            }
        raise AssertionError(args)

    def fake_refresh_existing_file(file_id):
        return None

    monkeypatch.setattr(pipeline, "_run_google_api", fake_run_google_api)
    monkeypatch.setattr(pipeline, "_find_existing_file", lambda folder_id, name: None)
    monkeypatch.setattr(pipeline, "_refresh_existing_file", fake_refresh_existing_file)

    first = pipeline.publish_artifact(
        PublishArtifactRequest(
            file_path=file_path,
            folder_name="Reports",
            duplicate_policy="reuse",
        )
    )
    second = pipeline.publish_artifact(
        PublishArtifactRequest(
            file_path=file_path,
            folder_name="Reports",
            duplicate_policy="reuse",
        )
    )

    assert first.file_id == "uploaded-1"
    assert second.file_id == "uploaded-2"
    assert uploaded["count"] == 2


def test_publish_artifact_dry_run_returns_planned_result(monkeypatch, tmp_path):
    file_path = tmp_path / "report.pdf"
    file_path.write_text("hello")
    store = GoogleDrivePipelineStore(tmp_path / "drive-store.json")
    pipeline = GoogleDrivePipeline(store=store)

    def fake_resolve_folder(**kwargs):
        from plugins.google_drive_pipeline.models import DriveFolderRef
        return DriveFolderRef(folder_id="folder-1", name="Reports")

    monkeypatch.setattr(pipeline, "resolve_folder", fake_resolve_folder)
    monkeypatch.setattr(pipeline, "_find_existing_file", lambda folder_id, name: None)

    result = pipeline.publish_artifact(
        PublishArtifactRequest(
            file_path=file_path,
            folder_name="Reports",
            dry_run=True,
        )
    )

    assert result.action == "planned"
    assert result.file_id == ""
    assert result.file_name == "report.pdf"


def test_resolve_folder_by_id(monkeypatch, tmp_path):
    store = GoogleDrivePipelineStore(tmp_path / "drive-store.json")
    pipeline = GoogleDrivePipeline(store=store)

    monkeypatch.setattr(
        pipeline,
        "_run_google_api",
        lambda args: {
            "id": "folder-1",
            "name": "Reports",
            "mimeType": "application/vnd.google-apps.folder",
            "parents": ["parent-123"],
            "webViewLink": "https://drive/folder-1",
        },
    )

    folder = pipeline.resolve_folder(folder_id="folder-1")

    assert folder.folder_id == "folder-1"
    assert folder.parent_folder_id == "parent-123"


def test_cli_publish_validates_share_args(tmp_path, capsys):
    rc = google_drive_pipeline_command(
        Namespace(
            google_drive_pipeline_action="publish",
            file_path=str(tmp_path / "report.pdf"),
            drive_name="",
            folder_id="",
            folder_name="Reports",
            parent_folder_id="",
            create_missing_folder=False,
            duplicate_policy="version",
            share_type="user",
            share_role="reader",
            share_email="",
            share_domain="",
            notify=False,
            dry_run=True,
            store_path=str(tmp_path / "drive-store.json"),
        )
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "email is required" in captured.out


def test_cli_show_returns_error_for_unknown_record(tmp_path, capsys):
    rc = google_drive_pipeline_command(
        Namespace(
            google_drive_pipeline_action="show",
            record_id="missing",
            store_path=str(tmp_path / "drive-store.json"),
        )
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "Unknown record_id" in captured.out
