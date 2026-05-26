"""Normalized models for the Teams meeting pipeline plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


ArtifactType = Literal["transcript", "recording", "call_record"]


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
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


def _clean_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


@dataclass
class GraphSubscription:
    subscription_id: str
    resource: str
    change_type: str
    notification_url: str
    expiration_datetime: datetime
    client_state: str | None = None
    latest_renewal_at: datetime | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        if not self.subscription_id.strip():
            raise ValueError("GraphSubscription.subscription_id is required.")
        if not self.resource.strip():
            raise ValueError("GraphSubscription.resource is required.")
        if not self.change_type.strip():
            raise ValueError("GraphSubscription.change_type is required.")
        if not self.notification_url.strip():
            raise ValueError("GraphSubscription.notification_url is required.")
        self.expiration_datetime = _parse_datetime(self.expiration_datetime)
        self.latest_renewal_at = _parse_datetime(self.latest_renewal_at)
        if self.expiration_datetime is None:
            raise ValueError("GraphSubscription.expiration_datetime is required.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphSubscription":
        return cls(
            subscription_id=str(payload.get("subscription_id") or payload.get("id") or "").strip(),
            resource=str(payload.get("resource") or "").strip(),
            change_type=str(payload.get("change_type") or payload.get("changeType") or "").strip(),
            notification_url=str(
                payload.get("notification_url") or payload.get("notificationUrl") or ""
            ).strip(),
            expiration_datetime=payload.get("expiration_datetime")
            or payload.get("expirationDateTime"),
            client_state=payload.get("client_state") or payload.get("clientState"),
            latest_renewal_at=payload.get("latest_renewal_at") or payload.get("latestRenewalAt"),
            status=payload.get("status"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "subscription_id": self.subscription_id,
                "resource": self.resource,
                "change_type": self.change_type,
                "notification_url": self.notification_url,
                "expiration_datetime": _serialize_datetime(self.expiration_datetime),
                "client_state": self.client_state,
                "latest_renewal_at": _serialize_datetime(self.latest_renewal_at),
                "status": self.status,
            }
        )


@dataclass
class TeamsMeetingRef:
    meeting_id: str
    organizer_user_id: str | None = None
    join_web_url: str | None = None
    calendar_event_id: str | None = None
    thread_id: str | None = None
    tenant_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.meeting_id.strip():
            raise ValueError("TeamsMeetingRef.meeting_id is required.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamsMeetingRef":
        return cls(
            meeting_id=str(payload.get("meeting_id") or payload.get("id") or "").strip(),
            organizer_user_id=payload.get("organizer_user_id") or payload.get("organizerUserId"),
            join_web_url=payload.get("join_web_url") or payload.get("joinWebUrl"),
            calendar_event_id=payload.get("calendar_event_id") or payload.get("calendarEventId"),
            thread_id=payload.get("thread_id") or payload.get("threadId"),
            tenant_id=payload.get("tenant_id") or payload.get("tenantId"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "meeting_id": self.meeting_id,
                "organizer_user_id": self.organizer_user_id,
                "join_web_url": self.join_web_url,
                "calendar_event_id": self.calendar_event_id,
                "thread_id": self.thread_id,
                "tenant_id": self.tenant_id,
                "metadata": self.metadata or None,
            }
        )


@dataclass
class MeetingArtifact:
    artifact_type: ArtifactType
    artifact_id: str
    display_name: str | None = None
    content_type: str | None = None
    source_url: str | None = None
    download_url: str | None = None
    created_at: datetime | None = None
    available_at: datetime | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.artifact_type not in ("transcript", "recording", "call_record"):
            raise ValueError(
                "MeetingArtifact.artifact_type must be transcript, recording, or call_record."
            )
        if not self.artifact_id.strip():
            raise ValueError("MeetingArtifact.artifact_id is required.")
        self.created_at = _parse_datetime(self.created_at)
        self.available_at = _parse_datetime(self.available_at)
        if self.size_bytes is not None:
            self.size_bytes = int(self.size_bytes)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MeetingArtifact":
        return cls(
            artifact_type=payload.get("artifact_type") or payload.get("artifactType"),
            artifact_id=str(payload.get("artifact_id") or payload.get("id") or "").strip(),
            display_name=payload.get("display_name")
            or payload.get("displayName")
            or payload.get("name"),
            content_type=payload.get("content_type") or payload.get("contentType"),
            source_url=payload.get("source_url") or payload.get("sourceUrl") or payload.get("webUrl"),
            download_url=payload.get("download_url")
            or payload.get("downloadUrl")
            or payload.get("@microsoft.graph.downloadUrl"),
            created_at=payload.get("created_at") or payload.get("createdDateTime"),
            available_at=payload.get("available_at")
            or payload.get("availableDateTime")
            or payload.get("lastModifiedDateTime"),
            size_bytes=payload.get("size_bytes") or payload.get("size"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "artifact_type": self.artifact_type,
                "artifact_id": self.artifact_id,
                "display_name": self.display_name,
                "content_type": self.content_type,
                "source_url": self.source_url,
                "download_url": self.download_url,
                "created_at": _serialize_datetime(self.created_at),
                "available_at": _serialize_datetime(self.available_at),
                "size_bytes": self.size_bytes,
                "metadata": self.metadata or None,
            }
        )


@dataclass
class TeamsMeetingSummaryPayload:
    meeting_ref: TeamsMeetingRef
    title: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    participants: list[str] = field(default_factory=list)
    transcript_text: str | None = None
    summary: str | None = None
    key_decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    call_metrics: dict[str, Any] = field(default_factory=dict)
    source_artifacts: list[MeetingArtifact] = field(default_factory=list)
    confidence: str | None = None
    confidence_notes: str | None = None
    notion_target: str | None = None
    linear_target: str | None = None
    teams_target: str | None = None

    def __post_init__(self) -> None:
        self.start_time = _parse_datetime(self.start_time)
        self.end_time = _parse_datetime(self.end_time)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamsMeetingSummaryPayload":
        return cls(
            meeting_ref=TeamsMeetingRef.from_dict(payload["meeting_ref"]),
            title=payload.get("title"),
            start_time=payload.get("start_time") or payload.get("startTime"),
            end_time=payload.get("end_time") or payload.get("endTime"),
            participants=list(payload.get("participants") or []),
            transcript_text=payload.get("transcript_text") or payload.get("transcriptText"),
            summary=payload.get("summary"),
            key_decisions=list(payload.get("key_decisions") or payload.get("keyDecisions") or []),
            action_items=list(payload.get("action_items") or payload.get("actionItems") or []),
            risks=list(payload.get("risks") or []),
            call_metrics=dict(payload.get("call_metrics") or payload.get("callMetrics") or {}),
            source_artifacts=[
                MeetingArtifact.from_dict(item) for item in payload.get("source_artifacts", [])
            ],
            confidence=payload.get("confidence"),
            confidence_notes=payload.get("confidence_notes") or payload.get("confidenceNotes"),
            notion_target=payload.get("notion_target") or payload.get("notionTarget"),
            linear_target=payload.get("linear_target") or payload.get("linearTarget"),
            teams_target=payload.get("teams_target") or payload.get("teamsTarget"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "meeting_ref": self.meeting_ref.to_dict(),
                "title": self.title,
                "start_time": _serialize_datetime(self.start_time),
                "end_time": _serialize_datetime(self.end_time),
                "participants": self.participants or None,
                "transcript_text": self.transcript_text,
                "summary": self.summary,
                "key_decisions": self.key_decisions or None,
                "action_items": self.action_items or None,
                "risks": self.risks or None,
                "call_metrics": self.call_metrics or None,
                "source_artifacts": [artifact.to_dict() for artifact in self.source_artifacts]
                or None,
                "confidence": self.confidence,
                "confidence_notes": self.confidence_notes,
                "notion_target": self.notion_target,
                "linear_target": self.linear_target,
                "teams_target": self.teams_target,
            }
        )


@dataclass
class TeamsMeetingPipelineJob:
    job_id: str
    event_id: str
    source_event_type: str
    dedupe_key: str
    status: str
    retry_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    meeting_ref: TeamsMeetingRef | None = None
    selected_artifact_strategy: str | None = None
    summary_payload: TeamsMeetingSummaryPayload | None = None
    error_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("TeamsMeetingPipelineJob.job_id is required.")
        if not self.event_id.strip():
            raise ValueError("TeamsMeetingPipelineJob.event_id is required.")
        if not self.source_event_type.strip():
            raise ValueError("TeamsMeetingPipelineJob.source_event_type is required.")
        if not self.dedupe_key.strip():
            raise ValueError("TeamsMeetingPipelineJob.dedupe_key is required.")
        if not self.status.strip():
            raise ValueError("TeamsMeetingPipelineJob.status is required.")
        self.retry_count = int(self.retry_count)
        self.created_at = _parse_datetime(self.created_at)
        self.updated_at = _parse_datetime(self.updated_at)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamsMeetingPipelineJob":
        meeting_ref_payload = payload.get("meeting_ref") or payload.get("meetingRef")
        summary_payload = payload.get("summary_payload") or payload.get("summaryPayload")
        return cls(
            job_id=str(payload.get("job_id") or payload.get("jobId") or "").strip(),
            event_id=str(payload.get("event_id") or payload.get("eventId") or "").strip(),
            source_event_type=str(
                payload.get("source_event_type") or payload.get("sourceEventType") or ""
            ).strip(),
            dedupe_key=str(payload.get("dedupe_key") or payload.get("dedupeKey") or "").strip(),
            status=str(payload.get("status") or "").strip(),
            retry_count=payload.get("retry_count") or payload.get("retryCount") or 0,
            created_at=payload.get("created_at") or payload.get("createdAt"),
            updated_at=payload.get("updated_at") or payload.get("updatedAt"),
            meeting_ref=TeamsMeetingRef.from_dict(meeting_ref_payload) if meeting_ref_payload else None,
            selected_artifact_strategy=payload.get("selected_artifact_strategy")
            or payload.get("selectedArtifactStrategy"),
            summary_payload=TeamsMeetingSummaryPayload.from_dict(summary_payload)
            if summary_payload
            else None,
            error_info=dict(payload.get("error_info") or payload.get("errorInfo") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "job_id": self.job_id,
                "event_id": self.event_id,
                "source_event_type": self.source_event_type,
                "dedupe_key": self.dedupe_key,
                "status": self.status,
                "retry_count": self.retry_count,
                "created_at": _serialize_datetime(self.created_at),
                "updated_at": _serialize_datetime(self.updated_at),
                "meeting_ref": self.meeting_ref.to_dict() if self.meeting_ref else None,
                "selected_artifact_strategy": self.selected_artifact_strategy,
                "summary_payload": self.summary_payload.to_dict() if self.summary_payload else None,
                "error_info": self.error_info or None,
            }
        )


__all__ = [
    "ArtifactType",
    "GraphSubscription",
    "MeetingArtifact",
    "TeamsMeetingPipelineJob",
    "TeamsMeetingRef",
    "TeamsMeetingSummaryPayload",
]
