"""Pipeline orchestration for Google Drive artifact publishing."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plugins.google_drive_pipeline.models import (
    DriveFolderRef,
    PublishArtifactRequest,
    PublishArtifactResult,
    SharePolicy,
)
from plugins.google_drive_pipeline.store import GoogleDrivePipelineStore


DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


class GoogleDrivePipelineError(RuntimeError):
    """Base class for Drive pipeline failures."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _escape_drive_query(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


def _normalize_string(value: Any) -> str:
    return str(value or "").strip()


def _build_source_key(source_sha256: str, folder_id: str, desired_name: str) -> str:
    return f"{source_sha256}:{folder_id}:{desired_name}"


def _drive_query_for_exact_name(name: str, parent_folder_id: str | None = None) -> str:
    clauses = [
        f"mimeType = '{DRIVE_FOLDER_MIME_TYPE}'",
        "trashed = false",
        f"name = '{_escape_drive_query(name)}'",
    ]
    if parent_folder_id:
        clauses.append(f"'{_escape_drive_query(parent_folder_id)}' in parents")
    return " and ".join(clauses)


def _file_query_for_name(name: str, parent_folder_id: str) -> str:
    clauses = [
        "trashed = false",
        f"name = '{_escape_drive_query(name)}'",
        f"'{_escape_drive_query(parent_folder_id)}' in parents",
    ]
    return " and ".join(clauses)


@dataclass
class GoogleDrivePipeline:
    store: GoogleDrivePipelineStore
    python_executable: str = sys.executable
    script_path: Path | None = None

    def __post_init__(self) -> None:
        if self.script_path is None:
            self.script_path = (
                Path(__file__).resolve().parents[2]
                / "skills/productivity/google-workspace/scripts/google_api.py"
            )

    def resolve_folder(
        self,
        *,
        folder_id: str | None = None,
        folder_name: str | None = None,
        parent_folder_id: str | None = None,
        create_missing: bool = False,
    ) -> DriveFolderRef:
        explicit_folder_id = _normalize_string(folder_id)
        explicit_folder_name = _normalize_string(folder_name)
        explicit_parent = _normalize_string(parent_folder_id)

        if explicit_folder_id:
            payload = self._run_google_api(["drive", "get", explicit_folder_id])
            if payload.get("mimeType") != DRIVE_FOLDER_MIME_TYPE:
                raise GoogleDrivePipelineError(f"Drive item is not a folder: {explicit_folder_id}")
            return DriveFolderRef.from_dict(payload)

        if not explicit_folder_name:
            raise GoogleDrivePipelineError("folder_name is required when folder_id is not provided.")

        cache_key = f"{explicit_parent}:{explicit_folder_name}"
        cached_folder = self.store.get_cached_folder(cache_key)
        if isinstance(cached_folder, dict):
            return DriveFolderRef.from_dict(cached_folder)

        results = self._run_google_api(
            [
                "drive",
                "search",
                _drive_query_for_exact_name(explicit_folder_name, explicit_parent or None),
                "--raw-query",
                "--max",
                "10",
            ]
        )
        if not isinstance(results, list):
            raise GoogleDrivePipelineError("Drive search returned unexpected payload.")

        folder_hits = [
            item for item in results if item.get("mimeType") == DRIVE_FOLDER_MIME_TYPE
        ]
        if folder_hits:
            folder = DriveFolderRef.from_dict(folder_hits[0])
            self.store.upsert_record(
                f"folder-cache:{cache_key}",
                {
                    "folder_cache_key": cache_key,
                    "folder": folder.to_dict(),
                    "source_key": "",
                    "kind": "folder_cache",
                },
            )
            return folder

        if not create_missing:
            raise GoogleDrivePipelineError(
                f"No Drive folder matched '{explicit_folder_name}'."
            )

        args = ["drive", "create-folder", explicit_folder_name]
        if explicit_parent:
            args.extend(["--parent", explicit_parent])
        payload = self._run_google_api(args)
        folder = DriveFolderRef.from_dict(
            {
                **payload,
                "parent_folder_id": explicit_parent or None,
                "existed": False,
            }
        )
        self.store.upsert_record(
            f"folder-cache:{cache_key}",
            {
                "folder_cache_key": cache_key,
                "folder": folder.to_dict(),
                "source_key": "",
                "kind": "folder_cache",
            },
        )
        return folder

    def publish_artifact(self, request: PublishArtifactRequest) -> PublishArtifactResult:
        if not request.dry_run and not request.file_path.exists():
            raise GoogleDrivePipelineError(f"Local file does not exist: {request.file_path}")

        folder = self.resolve_folder(
            folder_id=request.folder_id,
            folder_name=request.folder_name,
            parent_folder_id=request.parent_folder_id,
            create_missing=request.create_missing_folder,
        )
        source_sha256 = self._compute_sha256(request.file_path) if request.file_path.exists() else ""
        source_key = _build_source_key(source_sha256, folder.folder_id, request.desired_name)

        if request.duplicate_policy == "reuse":
            existing_record = self.store.get_record_by_source_key(source_key)
            if existing_record:
                result = PublishArtifactResult.from_dict(existing_record)
                if not request.dry_run and result.file_id:
                    refreshed = self._refresh_existing_file(result.file_id)
                    if refreshed is not None:
                        result.file_name = str(refreshed.get("name") or result.file_name)
                        result.web_view_link = str(
                            refreshed.get("webViewLink") or result.web_view_link
                        )
                    else:
                        existing_record = None
                if existing_record is not None:
                    if request.share_policy:
                        sharing = self._apply_share_policy(
                            result.file_id,
                            request.share_policy,
                            request.dry_run,
                        )
                        result.sharing = sharing
                    return self._persist_result(result, source_key)

        existing_file = self._find_existing_file(folder.folder_id, request.desired_name)
        final_name = request.desired_name
        action = "uploaded"

        if existing_file:
            if request.duplicate_policy == "fail":
                raise GoogleDrivePipelineError(
                    f"A Drive file named '{request.desired_name}' already exists in folder {folder.folder_id}."
                )
            if request.duplicate_policy == "reuse":
                action = "reused"
                result = PublishArtifactResult(
                    record_id=f"gdpl_{uuid.uuid4().hex[:12]}",
                    source_path=str(request.file_path),
                    source_sha256=source_sha256,
                    folder=folder,
                    file_id=str(existing_file["id"]),
                    file_name=str(existing_file["name"]),
                    web_view_link=str(existing_file.get("webViewLink") or ""),
                    duplicate_policy=request.duplicate_policy,
                    action=action,
                    created_at=_utc_now(),
                )
                if request.share_policy:
                    result.sharing = self._apply_share_policy(result.file_id, request.share_policy, request.dry_run)
                return self._persist_result(result, source_key)
            if request.duplicate_policy == "version":
                final_name = self._versioned_name(request.desired_name)

        if request.dry_run:
            result = PublishArtifactResult(
                record_id=f"gdpl_{uuid.uuid4().hex[:12]}",
                source_path=str(request.file_path),
                source_sha256=source_sha256,
                folder=folder,
                file_id="",
                file_name=final_name,
                web_view_link="",
                duplicate_policy=request.duplicate_policy,
                action="planned",
                created_at=_utc_now(),
            )
            if request.share_policy:
                result.sharing = {
                    "status": "planned",
                    **request.share_policy.to_dict(),
                }
            return result

        payload = self._run_google_api(
            [
                "drive",
                "upload",
                str(request.file_path),
                "--name",
                final_name,
                "--parent",
                folder.folder_id,
            ]
        )
        result = PublishArtifactResult(
            record_id=f"gdpl_{uuid.uuid4().hex[:12]}",
            source_path=str(request.file_path),
            source_sha256=source_sha256,
            folder=folder,
            file_id=str(payload.get("id") or ""),
            file_name=str(payload.get("name") or final_name),
            web_view_link=str(payload.get("webViewLink") or ""),
            duplicate_policy=request.duplicate_policy,
            action=action,
            created_at=_utc_now(),
        )
        if request.share_policy:
            result.sharing = self._apply_share_policy(result.file_id, request.share_policy, False)
        return self._persist_result(result, source_key)

    def _persist_result(self, result: PublishArtifactResult, source_key: str) -> PublishArtifactResult:
        payload = result.to_dict()
        payload["source_key"] = source_key
        payload["folder_cache_key"] = f"{result.folder.parent_folder_id or ''}:{result.folder.name}"
        payload["folder"] = result.folder.to_dict()
        saved = self.store.upsert_record(result.record_id, payload)
        return PublishArtifactResult.from_dict(saved)

    def _apply_share_policy(
        self,
        file_id: str,
        share_policy: SharePolicy,
        dry_run: bool,
    ) -> dict[str, Any]:
        if dry_run:
            return {"status": "planned", **share_policy.to_dict()}
        return self._run_google_api(share_policy.to_cli_args(file_id))

    def _find_existing_file(self, folder_id: str, name: str) -> dict[str, Any] | None:
        results = self._run_google_api(
            [
                "drive",
                "search",
                _file_query_for_name(name, folder_id),
                "--raw-query",
                "--max",
                "10",
            ]
        )
        if not isinstance(results, list):
            return None
        for item in results:
            if item.get("mimeType") != DRIVE_FOLDER_MIME_TYPE:
                return item
        return None

    def _refresh_existing_file(self, file_id: str) -> dict[str, Any] | None:
        try:
            payload = self._run_google_api(["drive", "get", file_id])
        except GoogleDrivePipelineError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _versioned_name(self, name: str) -> str:
        stamp = _utc_now().strftime("%Y%m%d-%H%M%S")
        path = Path(name)
        if path.suffix:
            return f"{path.stem}-{stamp}{path.suffix}"
        return f"{name}-{stamp}"

    def _compute_sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _run_google_api(self, args: list[str]) -> dict[str, Any] | list[Any]:
        cmd = [self.python_executable, str(self.script_path), *args]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error = (result.stderr or result.stdout or "Unknown google_api failure").strip()
            raise GoogleDrivePipelineError(error)
        stdout = result.stdout.strip()
        if not stdout:
            return {}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise GoogleDrivePipelineError(
                f"google_api.py returned non-JSON output: {stdout}"
            ) from exc
