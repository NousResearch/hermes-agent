"""CLI commands for the Google Drive artifact pipeline plugin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plugins.google_drive_pipeline.models import PublishArtifactRequest, SharePolicy
from plugins.google_drive_pipeline.pipeline import GoogleDrivePipeline, GoogleDrivePipelineError
from plugins.google_drive_pipeline.store import (
    GoogleDrivePipelineStore,
    resolve_google_drive_pipeline_store_path,
)


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="google_drive_pipeline_action")

    resolve_p = subs.add_parser("resolve-folder", help="Resolve a Drive folder by id or exact name")
    resolve_p.add_argument("--folder-id", default="")
    resolve_p.add_argument("--folder-name", default="")
    resolve_p.add_argument("--parent-folder-id", default="")
    resolve_p.add_argument("--create-missing", action="store_true")
    resolve_p.add_argument("--store-path", default="")

    publish_p = subs.add_parser("publish", help="Publish a local artifact into Drive")
    publish_p.add_argument("file_path")
    publish_p.add_argument("--drive-name", default="")
    publish_p.add_argument("--folder-id", default="")
    publish_p.add_argument("--folder-name", default="")
    publish_p.add_argument("--parent-folder-id", default="")
    publish_p.add_argument("--create-missing-folder", action="store_true")
    publish_p.add_argument(
        "--duplicate-policy",
        default="version",
        choices=["fail", "reuse", "version"],
    )
    publish_p.add_argument("--share-type", default="")
    publish_p.add_argument("--share-role", default="reader")
    publish_p.add_argument("--share-email", default="")
    publish_p.add_argument("--share-domain", default="")
    publish_p.add_argument("--notify", action="store_true")
    publish_p.add_argument("--dry-run", action="store_true")
    publish_p.add_argument("--store-path", default="")

    list_p = subs.add_parser("list", aliases=["ls"], help="List recent publish records")
    list_p.add_argument("--limit", type=int, default=20)
    list_p.add_argument("--store-path", default="")

    show_p = subs.add_parser("show", help="Show a stored publish record")
    show_p.add_argument("record_id")
    show_p.add_argument("--store-path", default="")

    subparser.set_defaults(func=google_drive_pipeline_command)


def google_drive_pipeline_command(args: argparse.Namespace) -> int:
    action = getattr(args, "google_drive_pipeline_action", None)
    if not action:
        print("Usage: hermes google-drive-pipeline {resolve-folder|publish|list|show}")
        return 2

    try:
        if action == "resolve-folder":
            _cmd_resolve_folder(args)
        elif action == "publish":
            _cmd_publish(args)
        elif action in ("list", "ls"):
            _cmd_list(args)
        elif action == "show":
            _cmd_show(args)
        else:
            print(f"Unknown google-drive-pipeline action: {action}")
            return 2
        return 0
    except GoogleDrivePipelineError as exc:
        print(str(exc))
        return 1


def _store(args: argparse.Namespace) -> GoogleDrivePipelineStore:
    return GoogleDrivePipelineStore(
        resolve_google_drive_pipeline_store_path(getattr(args, "store_path", None))
    )


def _pipeline(args: argparse.Namespace) -> GoogleDrivePipeline:
    return GoogleDrivePipeline(store=_store(args))


def _build_share_policy(args: argparse.Namespace) -> SharePolicy | None:
    share_type = str(getattr(args, "share_type", "") or "").strip()
    if not share_type:
        return None
    try:
        return SharePolicy(
            access_type=share_type,
            role=str(getattr(args, "share_role", "") or "reader").strip() or "reader",
            email=str(getattr(args, "share_email", "") or "").strip() or None,
            domain=str(getattr(args, "share_domain", "") or "").strip() or None,
            notify=bool(getattr(args, "notify", False)),
        )
    except ValueError as exc:
        raise GoogleDrivePipelineError(str(exc)) from exc


def _cmd_resolve_folder(args: argparse.Namespace) -> None:
    folder = _pipeline(args).resolve_folder(
        folder_id=str(args.folder_id or "").strip() or None,
        folder_name=str(args.folder_name or "").strip() or None,
        parent_folder_id=str(args.parent_folder_id or "").strip() or None,
        create_missing=bool(args.create_missing),
    )
    print(json.dumps(folder.to_dict(), indent=2, ensure_ascii=False))


def _cmd_publish(args: argparse.Namespace) -> None:
    request = PublishArtifactRequest(
        file_path=Path(args.file_path),
        drive_name=str(args.drive_name or "").strip() or None,
        folder_id=str(args.folder_id or "").strip() or None,
        folder_name=str(args.folder_name or "").strip() or None,
        parent_folder_id=str(args.parent_folder_id or "").strip() or None,
        create_missing_folder=bool(args.create_missing_folder),
        duplicate_policy=str(args.duplicate_policy or "version"),
        share_policy=_build_share_policy(args),
        dry_run=bool(args.dry_run),
    )
    result = _pipeline(args).publish_artifact(request)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def _cmd_list(args: argparse.Namespace) -> None:
    records = list(_store(args).list_records().values())
    records = [record for record in records if record.get("kind") != "folder_cache"]
    records.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    print(json.dumps(records[: max(0, int(args.limit))], indent=2, ensure_ascii=False))


def _cmd_show(args: argparse.Namespace) -> None:
    record = _store(args).get_record(args.record_id)
    if record is None:
        raise GoogleDrivePipelineError(f"Unknown record_id: {args.record_id}")
    print(json.dumps(record, indent=2, ensure_ascii=False))
