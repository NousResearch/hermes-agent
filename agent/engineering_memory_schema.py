"""Schema and Markdown codec for the shared engineering-memory ledger.

The record format is deliberately boring: structured frontmatter is the
machine contract and the body is bounded explanatory evidence.  This module
does not decide whether a record is safe or approved; that belongs to the
curator and explicit review workflow.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from agent.skill_utils import parse_frontmatter

SCHEMA_NAME = "agent_engineering_record_v1"
ALLOWED_AGENTS = frozenset({"hermes", "claude", "codex", "shared"})
ALLOWED_STATUSES = frozenset(
    {"candidate", "quarantined", "needs_review", "approved", "superseded", "rejected", "withdrawn"}
)
AUTHORITY = "source-index"
CLASSIFICATION = "diagnostic-only"
MAX_TITLE_CHARS = 240
MAX_SUMMARY_CHARS = 2_000
MAX_BODY_CHARS = 20_000
MAX_LIST_ITEMS = 64
MAX_ITEM_CHARS = 512

_FIELDS = frozenset(
    {
        "record_id",
        "schema_name",
        "sync_owned",
        "classification",
        "authority",
        "status",
        "title",
        "summary",
        "agent",
        "client_model",
        "repository",
        "workspace",
        "branch_or_ref",
        "verified_head",
        "memory_area",
        "task_type",
        "component",
        "symptoms",
        "tags",
        "observed_at",
        "verified_at",
        "expires_at",
        "source_label",
        "source_digest",
        "evidence_refs",
        "related_record_ids",
        "supersedes",
        "conflict_set",
        "threat_scan",
        "secret_scan",
        "review",
        "body",
    }
)


class EngineeringMemorySchemaError(ValueError):
    """Raised when a record cannot be represented safely by the v1 schema."""


def _required_text(mapping: Mapping[str, Any], field_name: str, *, max_chars: int | None = None) -> str:
    value = mapping.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise EngineeringMemorySchemaError(f"{field_name}: required non-empty string")
    value = value.strip()
    if max_chars is not None and len(value) > max_chars:
        raise EngineeringMemorySchemaError(f"{field_name}: exceeds {max_chars} characters")
    if "\x00" in value:
        raise EngineeringMemorySchemaError(f"{field_name}: NUL is not allowed")
    return value


def _optional_text(mapping: Mapping[str, Any], field_name: str, *, max_chars: int = MAX_ITEM_CHARS) -> str | None:
    value = mapping.get(field_name)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise EngineeringMemorySchemaError(f"{field_name}: expected string or null")
    value = value.strip()
    if len(value) > max_chars:
        raise EngineeringMemorySchemaError(f"{field_name}: exceeds {max_chars} characters")
    return value or None


def _list_text(mapping: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    value = mapping.get(field_name, [])
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise EngineeringMemorySchemaError(f"{field_name}: expected a list")
    if len(value) > MAX_LIST_ITEMS:
        raise EngineeringMemorySchemaError(f"{field_name}: exceeds {MAX_LIST_ITEMS} items")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise EngineeringMemorySchemaError(f"{field_name}: items must be non-empty strings")
        item = item.strip()
        if len(item) > MAX_ITEM_CHARS:
            raise EngineeringMemorySchemaError(f"{field_name}: item exceeds {MAX_ITEM_CHARS} characters")
        result.append(item)
    return tuple(sorted(set(result)))


def _timestamp(mapping: Mapping[str, Any], field_name: str, *, required: bool) -> datetime | None:
    value = mapping.get(field_name)
    if value is None or value == "":
        if required:
            raise EngineeringMemorySchemaError(f"{field_name}: required timestamp")
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EngineeringMemorySchemaError(f"{field_name}: invalid timestamp") from exc
    else:
        raise EngineeringMemorySchemaError(f"{field_name}: expected ISO timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _metadata(mapping: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    value = mapping.get(field_name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EngineeringMemorySchemaError(f"{field_name}: expected mapping")
    # Metadata is intentionally shallow and JSON-safe.  Arbitrary nested text
    # is not allowed to become an unbounded transport for untrusted content.
    try:
        encoded = json.dumps(dict(value), sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise EngineeringMemorySchemaError(f"{field_name}: must be JSON-safe") from exc
    if len(encoded) > 4_000:
        raise EngineeringMemorySchemaError(f"{field_name}: exceeds 4000 characters")
    return dict(value)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if value else None


@dataclass(frozen=True)
class EngineeringMemoryRecord:
    record_id: str
    title: str
    summary: str
    agent: str
    client_model: str | None
    repository: str
    workspace: str | None
    branch_or_ref: str | None
    verified_head: str | None
    memory_area: str
    task_type: str
    component: str
    symptoms: tuple[str, ...]
    tags: tuple[str, ...]
    observed_at: datetime
    verified_at: datetime | None
    expires_at: datetime | None
    source_label: str
    source_digest: str
    evidence_refs: tuple[str, ...]
    related_record_ids: tuple[str, ...]
    supersedes: str | None
    conflict_set: tuple[str, ...]
    status: str = "candidate"
    threat_scan: dict[str, Any] = field(default_factory=dict)
    secret_scan: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, source_path: Path | None = None) -> "EngineeringMemoryRecord":
        unknown = set(mapping) - _FIELDS
        if unknown:
            raise EngineeringMemorySchemaError(f"unknown fields: {', '.join(sorted(unknown))}")
        if mapping.get("schema_name") != SCHEMA_NAME:
            raise EngineeringMemorySchemaError("schema_name: unsupported schema")
        if mapping.get("sync_owned") is not True:
            raise EngineeringMemorySchemaError("sync_owned: must be true")
        if mapping.get("classification") != CLASSIFICATION:
            raise EngineeringMemorySchemaError("classification: unsupported classification")
        if mapping.get("authority") != AUTHORITY:
            raise EngineeringMemorySchemaError("authority: unsupported authority")

        record_id = _required_text(mapping, "record_id", max_chars=160)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}", record_id):
            raise EngineeringMemorySchemaError("record_id: contains unsafe characters")
        status = mapping.get("status", "candidate")
        if status not in ALLOWED_STATUSES:
            raise EngineeringMemorySchemaError("status: unsupported status")
        agent = _required_text(mapping, "agent", max_chars=32)
        if agent not in ALLOWED_AGENTS:
            raise EngineeringMemorySchemaError("agent: unsupported agent")
        body = mapping.get("body", "")
        if not isinstance(body, str):
            raise EngineeringMemorySchemaError("body: expected string")
        body = body.strip()
        if len(body) > MAX_BODY_CHARS:
            raise EngineeringMemorySchemaError(f"body: exceeds {MAX_BODY_CHARS} characters")
        if "\x00" in body:
            raise EngineeringMemorySchemaError("body: NUL is not allowed")

        return cls(
            record_id=record_id,
            title=_required_text(mapping, "title", max_chars=MAX_TITLE_CHARS),
            summary=_required_text(mapping, "summary", max_chars=MAX_SUMMARY_CHARS),
            agent=agent,
            client_model=_optional_text(mapping, "client_model"),
            repository=_required_text(mapping, "repository"),
            workspace=_optional_text(mapping, "workspace"),
            branch_or_ref=_optional_text(mapping, "branch_or_ref"),
            verified_head=_optional_text(mapping, "verified_head"),
            memory_area=_required_text(mapping, "memory_area", max_chars=MAX_ITEM_CHARS),
            task_type=_required_text(mapping, "task_type", max_chars=MAX_ITEM_CHARS),
            component=_required_text(mapping, "component", max_chars=MAX_ITEM_CHARS),
            symptoms=_list_text(mapping, "symptoms"),
            tags=_list_text(mapping, "tags"),
            observed_at=_timestamp(mapping, "observed_at", required=True),
            verified_at=_timestamp(mapping, "verified_at", required=False),
            expires_at=_timestamp(mapping, "expires_at", required=False),
            source_label=_required_text(mapping, "source_label"),
            source_digest=_required_text(mapping, "source_digest"),
            evidence_refs=_list_text(mapping, "evidence_refs"),
            related_record_ids=_list_text(mapping, "related_record_ids"),
            supersedes=_optional_text(mapping, "supersedes", max_chars=160),
            conflict_set=_list_text(mapping, "conflict_set"),
            status=status,
            threat_scan=_metadata(mapping, "threat_scan"),
            secret_scan=_metadata(mapping, "secret_scan"),
            review=_metadata(mapping, "review"),
            body=body,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "schema_name": SCHEMA_NAME,
            "sync_owned": True,
            "classification": CLASSIFICATION,
            "authority": AUTHORITY,
            "status": self.status,
            "title": self.title,
            "summary": self.summary,
            "agent": self.agent,
            "client_model": self.client_model,
            "repository": self.repository,
            "workspace": self.workspace,
            "branch_or_ref": self.branch_or_ref,
            "verified_head": self.verified_head,
            "memory_area": self.memory_area,
            "task_type": self.task_type,
            "component": self.component,
            "symptoms": list(self.symptoms),
            "tags": list(self.tags),
            "observed_at": _iso(self.observed_at),
            "verified_at": _iso(self.verified_at),
            "expires_at": _iso(self.expires_at),
            "source_label": self.source_label,
            "source_digest": self.source_digest,
            "evidence_refs": list(self.evidence_refs),
            "related_record_ids": list(self.related_record_ids),
            "supersedes": self.supersedes,
            "conflict_set": list(self.conflict_set),
            "threat_scan": self.threat_scan,
            "secret_scan": self.secret_scan,
            "review": self.review,
            "body": self.body,
        }

    def canonical_payload(self) -> bytes:
        value = self.to_mapping()
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def canonical_record_digest(self) -> str:
        return hashlib.sha256(self.canonical_payload()).hexdigest()


def canonical_record_digest(mapping: Mapping[str, Any]) -> str:
    return EngineeringMemoryRecord.from_mapping(mapping).canonical_record_digest()


def parse_markdown_record(text: str, *, source_path: Path | None = None) -> EngineeringMemoryRecord:
    frontmatter, body = parse_frontmatter(text)
    if not frontmatter:
        location = f" in {source_path}" if source_path else ""
        raise EngineeringMemorySchemaError(f"frontmatter{location}: required")
    value = dict(frontmatter)
    value["body"] = body.strip()
    return EngineeringMemoryRecord.from_mapping(value, source_path=source_path)


def render_markdown_record(record: EngineeringMemoryRecord) -> str:
    mapping = record.to_mapping()
    body = mapping.pop("body")
    # SafeDump emits plain data only; the parser still validates every field on read.
    frontmatter = yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True, default_flow_style=False).rstrip()
    return f"---\n{frontmatter}\n---\n\n{body}\n"
