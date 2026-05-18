"""Normalized models for the Google Drive artifact pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


DuplicatePolicy = Literal["fail", "reuse", "version"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


@dataclass
class DriveFolderRef:
    folder_id: str
    name: str
    web_view_link: str | None = None
    parent_folder_id: str | None = None
    existed: bool = True

    def __post_init__(self) -> None:
        self.folder_id = str(self.folder_id).strip()
        self.name = str(self.name).strip()
        if not self.folder_id:
            raise ValueError("DriveFolderRef.folder_id is required.")
        if not self.name:
            raise ValueError("DriveFolderRef.name is required.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DriveFolderRef":
        parents = payload.get("parents") or []
        parent_folder_id = None
        if isinstance(parents, list) and parents:
            parent_folder_id = str(parents[0]).strip() or None
        return cls(
            folder_id=payload.get("folder_id") or payload.get("id") or "",
            name=payload.get("name") or "",
            web_view_link=payload.get("web_view_link") or payload.get("webViewLink"),
            parent_folder_id=payload.get("parent_folder_id") or parent_folder_id,
            existed=bool(payload.get("existed", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "folder_id": self.folder_id,
                "name": self.name,
                "web_view_link": self.web_view_link,
                "parent_folder_id": self.parent_folder_id,
                "existed": self.existed,
            }
        )


@dataclass
class SharePolicy:
    access_type: Literal["user", "group", "domain", "anyone"] = "user"
    role: str = "reader"
    email: str | None = None
    domain: str | None = None
    notify: bool = False

    def __post_init__(self) -> None:
        if self.access_type in ("user", "group") and not _clean_optional(self.email):
            raise ValueError(f"email is required for share access_type={self.access_type}.")
        if self.access_type == "domain" and not _clean_optional(self.domain):
            raise ValueError("domain is required for share access_type=domain.")
        self.email = _clean_optional(self.email)
        self.domain = _clean_optional(self.domain)

    def to_cli_args(self, file_id: str) -> list[str]:
        args = [
            "drive",
            "share",
            file_id,
            "--type",
            self.access_type,
            "--role",
            self.role,
        ]
        if self.email:
            args.extend(["--email", self.email])
        if self.domain:
            args.extend(["--domain", self.domain])
        if self.notify:
            args.append("--notify")
        return args

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "access_type": self.access_type,
                "role": self.role,
                "email": self.email,
                "domain": self.domain,
                "notify": self.notify,
            }
        )


def _clean_optional(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


@dataclass
class PublishArtifactRequest:
    file_path: Path
    drive_name: str | None = None
    folder_id: str | None = None
    folder_name: str | None = None
    parent_folder_id: str | None = None
    create_missing_folder: bool = False
    duplicate_policy: DuplicatePolicy = "version"
    share_policy: SharePolicy | None = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path).expanduser()
        if self.duplicate_policy not in ("fail", "reuse", "version"):
            raise ValueError("duplicate_policy must be fail, reuse, or version.")
        if not self.folder_id and not self.folder_name:
            raise ValueError("folder_id or folder_name is required.")

    @property
    def desired_name(self) -> str:
        return (self.drive_name or self.file_path.name).strip()


@dataclass
class PublishArtifactResult:
    record_id: str
    source_path: str
    source_sha256: str
    folder: DriveFolderRef
    file_id: str
    file_name: str
    web_view_link: str
    duplicate_policy: DuplicatePolicy
    action: Literal["uploaded", "reused", "planned"]
    sharing: dict[str, Any] | None = None
    created_at: datetime | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PublishArtifactResult":
        return cls(
            record_id=str(payload.get("record_id") or "").strip(),
            source_path=str(payload.get("source_path") or "").strip(),
            source_sha256=str(payload.get("source_sha256") or "").strip(),
            folder=DriveFolderRef.from_dict(payload.get("folder") or {}),
            file_id=str(payload.get("file_id") or "").strip(),
            file_name=str(payload.get("file_name") or "").strip(),
            web_view_link=str(payload.get("web_view_link") or payload.get("webViewLink") or "").strip(),
            duplicate_policy=payload.get("duplicate_policy") or "version",
            action=payload.get("action") or "uploaded",
            sharing=payload.get("sharing"),
            created_at=_parse_datetime(payload.get("created_at")),
        )

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "record_id": self.record_id,
                "source_path": self.source_path,
                "source_sha256": self.source_sha256,
                "folder": self.folder.to_dict(),
                "file_id": self.file_id,
                "file_name": self.file_name,
                "web_view_link": self.web_view_link,
                "duplicate_policy": self.duplicate_policy,
                "action": self.action,
                "sharing": self.sharing,
                "created_at": _serialize_datetime(self.created_at) or _utc_now_iso(),
            }
        )
