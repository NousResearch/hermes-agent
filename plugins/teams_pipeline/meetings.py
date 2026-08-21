"""Graph-backed Teams meeting helpers for the plugin runtime."""

from __future__ import annotations

import base64
import binascii
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from plugins.teams_pipeline.models import MeetingArtifact, TeamsMeetingRef
from tools.microsoft_graph_client import MicrosoftGraphAPIError, MicrosoftGraphClient

# Graph uses both slash keys (users/{id}/...) and quoted keys (users('{id}')/...).
_USERS_MEETING_RE = re.compile(
    r"(?i)(?:^|/)users(?:\('([^']+)'\)|/([^/'()]+))/onlineMeetings(?:\('([^']+)'\)|/([^/'?]+))"
)
_COMM_MEETING_RE = re.compile(
    r"(?i)(?:^|/)communications/onlineMeetings(?:\('([^']+)'\)|/([^/'?]+))"
)
_TRANSCRIPT_RE = re.compile(r"(?i)/transcripts(?:\('([^']+)'\)|/([^/'?]+))")
_RECORDING_RE = re.compile(r"(?i)/recordings(?:\('([^']+)'\)|/([^/'?]+))")
_RESOURCE_SENTINELS = frozenset(
    {
        "getalltranscripts",
        "getallrecordings",
        "transcripts",
        "recordings",
    }
)


class TeamsMeetingError(RuntimeError):
    """Base class for Teams meeting pipeline failures."""


class TeamsMeetingNotFoundError(TeamsMeetingError):
    """Raised when the meeting cannot be resolved from Graph."""


class TeamsMeetingArtifactNotFoundError(TeamsMeetingError):
    """Raised when a transcript or recording cannot be found."""


class TeamsMeetingPermissionError(TeamsMeetingError):
    """Raised when Graph access is denied for the requested resource."""


def parse_graph_meeting_resource(resource: str) -> dict[str, str | None]:
    """Parse organizer, meeting, and artifact ids from a Graph resource or @odata.id."""

    text = str(resource or "").strip()
    organizer_user_id: str | None = None
    meeting_id: str | None = None
    transcript_id: str | None = None
    recording_id: str | None = None

    users_match = _USERS_MEETING_RE.search(text)
    if users_match:
        organizer_user_id = unquote(users_match.group(1) or users_match.group(2) or "").strip() or None
        meeting_id = unquote(users_match.group(3) or users_match.group(4) or "").strip() or None

    if not meeting_id:
        comm_match = _COMM_MEETING_RE.search(text)
        if comm_match:
            meeting_id = unquote(comm_match.group(1) or comm_match.group(2) or "").strip() or None

    if meeting_id and meeting_id.lower() in _RESOURCE_SENTINELS:
        meeting_id = None

    transcript_match = _TRANSCRIPT_RE.search(text)
    if transcript_match:
        transcript_id = unquote(transcript_match.group(1) or transcript_match.group(2) or "").strip() or None

    recording_match = _RECORDING_RE.search(text)
    if recording_match:
        recording_id = unquote(recording_match.group(1) or recording_match.group(2) or "").strip() or None

    return {
        "organizer_user_id": organizer_user_id,
        "meeting_id": meeting_id,
        "transcript_id": transcript_id,
        "recording_id": recording_id,
    }


def looks_like_transcript_id(value: str, *, odata_type: str | None = None) -> bool:
    """True when a Graph id is a callTranscript artifact rather than an onlineMeeting."""

    if "calltranscript" in str(odata_type or "").lower():
        return True
    text = str(value or "")
    if "transcript" in text.lower():
        return True
    return "transcript" in _decoded_id_hint(text)


def _decoded_id_hint(value: str) -> str:
    """Best-effort base64 decode of a Graph id for artifact-marker sniffing.

    Graph transcript ids are base64url blobs whose *decoded* payload carries a
    ``...-TranscriptV2`` suffix while the encoded form contains no readable
    marker (this is exactly the id shape from getAllTranscripts
    ``resourceData.id``). Returns lowercase decoded text, or "" when the value
    does not decode.
    """

    stripped = value.strip()
    if len(stripped) < 16:
        return ""
    padded = stripped + "=" * (-len(stripped) % 4)
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            return decoder(padded).decode("utf-8", "ignore").lower()
        except (binascii.Error, ValueError):
            continue
    return ""


def _meeting_path(meeting_ref: TeamsMeetingRef | str) -> str:
    if isinstance(meeting_ref, TeamsMeetingRef):
        meeting_id = meeting_ref.meeting_id
        organizer_user_id = meeting_ref.organizer_user_id
    else:
        meeting_id = str(meeting_ref)
        organizer_user_id = None
    encoded_meeting_id = quote(meeting_id, safe="")
    if organizer_user_id:
        return (
            f"/users/{quote(organizer_user_id, safe='')}/onlineMeetings/{encoded_meeting_id}"
        )
    return f"/communications/onlineMeetings/{encoded_meeting_id}"


def _wrap_graph_error(exc: MicrosoftGraphAPIError, *, missing_message: str) -> TeamsMeetingError:
    if exc.status_code in {401, 403}:
        return TeamsMeetingPermissionError(str(exc))
    if exc.status_code == 404:
        return TeamsMeetingNotFoundError(missing_message)
    return TeamsMeetingError(str(exc))


def _parse_organizer_user_id(payload: dict[str, Any]) -> str | None:
    organizer = payload.get("organizer")
    if not isinstance(organizer, dict):
        return None
    identity = organizer.get("identity")
    if not isinstance(identity, dict):
        return None
    user = identity.get("user")
    if not isinstance(user, dict):
        return None
    return user.get("id")


def _parse_thread_id(payload: dict[str, Any]) -> str | None:
    chat = payload.get("chatInfo")
    if isinstance(chat, dict):
        thread_id = chat.get("threadId")
        if thread_id:
            return str(thread_id)
    return payload.get("threadId")


def _normalize_meeting_ref(
    payload: dict[str, Any],
    *,
    tenant_id: str | None = None,
    organizer_user_id: str | None = None,
) -> TeamsMeetingRef:
    metadata = {
        key: payload.get(key)
        for key in ("subject", "startDateTime", "endDateTime", "createdDateTime")
        if payload.get(key) is not None
    }
    participants = payload.get("participants")
    if participants is not None:
        metadata["participants"] = participants
    return TeamsMeetingRef(
        meeting_id=str(payload.get("id") or "").strip(),
        organizer_user_id=organizer_user_id or _parse_organizer_user_id(payload),
        join_web_url=payload.get("joinWebUrl"),
        calendar_event_id=payload.get("calendarEventId"),
        thread_id=_parse_thread_id(payload),
        tenant_id=tenant_id or payload.get("tenantId"),
        metadata=metadata,
    )


def _normalize_artifact(
    artifact_type: str,
    payload: dict[str, Any],
    *,
    default_source_url: str | None = None,
) -> MeetingArtifact:
    metadata = dict(payload)
    download_url = (
        payload.get("@microsoft.graph.downloadUrl")
        or payload.get("downloadUrl")
        or payload.get("recordingContentUrl")
        or payload.get("transcriptContentUrl")
    )
    source_url = payload.get("webUrl") or payload.get("contentUrl") or default_source_url
    return MeetingArtifact(
        artifact_type=artifact_type,  # type: ignore[arg-type]
        artifact_id=str(payload.get("id") or "").strip(),
        display_name=payload.get("displayName") or payload.get("name"),
        content_type=payload.get("contentType") or payload.get("fileMimeType"),
        source_url=source_url,
        download_url=download_url,
        created_at=payload.get("createdDateTime"),
        available_at=payload.get("lastModifiedDateTime") or payload.get("meetingEndDateTime"),
        size_bytes=payload.get("size"),
        metadata=metadata,
    )


def _transcript_sort_key(artifact: MeetingArtifact) -> tuple[int, int, str]:
    status = str(artifact.metadata.get("status") or "").lower()
    has_download = int(bool(artifact.download_url or artifact.source_url))
    is_completed = int(status in {"available", "completed", "succeeded"})
    timestamp = ""
    if artifact.available_at is not None:
        timestamp = artifact.available_at.isoformat()
    elif artifact.created_at is not None:
        timestamp = artifact.created_at.isoformat()
    return (is_completed, has_download, timestamp)


def _recording_download_path(meeting_ref: TeamsMeetingRef, artifact: MeetingArtifact) -> str:
    if artifact.download_url:
        return artifact.download_url
    return f"{_meeting_path(meeting_ref)}/recordings/{quote(artifact.artifact_id, safe='')}/content"


def _transcript_download_path(meeting_ref: TeamsMeetingRef, artifact: MeetingArtifact) -> str:
    if artifact.download_url:
        return artifact.download_url
    return f"{_meeting_path(meeting_ref)}/transcripts/{quote(artifact.artifact_id, safe='')}/content"


async def resolve_meeting_reference(
    client: MicrosoftGraphClient,
    *,
    meeting_id: str | None = None,
    join_web_url: str | None = None,
    tenant_id: str | None = None,
    organizer_user_id: str | None = None,
) -> TeamsMeetingRef:
    if meeting_id and looks_like_transcript_id(meeting_id):
        if join_web_url:
            meeting_id = None
        else:
            raise TeamsMeetingError(
                "Refusing to GET /communications/onlineMeetings/{id} with a transcript id. "
                "Graph v1.0 does not support that id format; use the organizer-scoped meeting "
                "id from the notification @odata.id, or a join URL."
            )
    if meeting_id:
        try:
            payload = await client.get_json(
                _meeting_path(
                    TeamsMeetingRef(meeting_id=meeting_id, organizer_user_id=organizer_user_id)
                )
            )
        except MicrosoftGraphAPIError as exc:
            raise _wrap_graph_error(exc, missing_message=f"Teams meeting not found: {meeting_id}") from exc
        if not isinstance(payload, dict) or not payload.get("id"):
            raise TeamsMeetingNotFoundError(f"Teams meeting not found: {meeting_id}")
        return _normalize_meeting_ref(
            payload,
            tenant_id=tenant_id,
            organizer_user_id=organizer_user_id,
        )

    if join_web_url:
        escaped_join_url = join_web_url.replace("'", "''")
        lookup_path = "/communications/onlineMeetings"
        if organizer_user_id:
            lookup_path = f"/users/{quote(organizer_user_id, safe='')}/onlineMeetings"
        try:
            payload = await client.get_json(
                lookup_path,
                params={"$filter": f"JoinWebUrl eq '{escaped_join_url}'"},
            )
        except MicrosoftGraphAPIError as exc:
            raise _wrap_graph_error(
                exc,
                missing_message=f"Teams meeting not found for join URL: {join_web_url}",
            ) from exc
        candidates = payload.get("value") if isinstance(payload, dict) else None
        if not isinstance(candidates, list) or not candidates:
            raise TeamsMeetingNotFoundError(f"Teams meeting not found for join URL: {join_web_url}")
        return _normalize_meeting_ref(
            candidates[0],
            tenant_id=tenant_id,
            organizer_user_id=organizer_user_id,
        )

    raise ValueError("Either meeting_id or join_web_url is required.")


async def list_transcript_artifacts(
    client: MicrosoftGraphClient,
    meeting_ref: TeamsMeetingRef,
) -> list[MeetingArtifact]:
    try:
        payloads = await client.collect_paginated(f"{_meeting_path(meeting_ref)}/transcripts")
    except MicrosoftGraphAPIError as exc:
        raise _wrap_graph_error(
            exc,
            missing_message=f"No transcripts found for Teams meeting {meeting_ref.meeting_id}",
        ) from exc
    return [_normalize_artifact("transcript", payload) for payload in payloads if isinstance(payload, dict)]


def select_preferred_transcript(candidates: list[MeetingArtifact]) -> MeetingArtifact | None:
    transcripts = [candidate for candidate in candidates if candidate.artifact_type == "transcript"]
    if not transcripts:
        return None
    return sorted(transcripts, key=_transcript_sort_key, reverse=True)[0]


async def download_transcript_text(
    client: MicrosoftGraphClient,
    meeting_ref: TeamsMeetingRef,
    transcript: MeetingArtifact,
    *,
    encoding: str = "utf-8",
) -> str:
    suffix = Path(transcript.display_name or "transcript.vtt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(prefix="teams-transcript-", suffix=suffix, delete=False) as handle:
        destination = Path(handle.name)
    try:
        await client.download_to_file(_transcript_download_path(meeting_ref, transcript), destination)
        text = destination.read_text(encoding=encoding).strip()
    except MicrosoftGraphAPIError as exc:
        raise _wrap_graph_error(
            exc,
            missing_message=(
                f"Transcript {transcript.artifact_id} not found for meeting {meeting_ref.meeting_id}"
            ),
        ) from exc
    finally:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass

    if not text:
        raise TeamsMeetingArtifactNotFoundError(
            f"Transcript {transcript.artifact_id} for meeting {meeting_ref.meeting_id} was empty."
        )
    return text


async def fetch_preferred_transcript_text(
    client: MicrosoftGraphClient,
    meeting_ref: TeamsMeetingRef,
) -> tuple[MeetingArtifact | None, str | None]:
    transcripts = await list_transcript_artifacts(client, meeting_ref)
    transcript = select_preferred_transcript(transcripts)
    if transcript is None:
        return None, None
    try:
        return transcript, await download_transcript_text(client, meeting_ref, transcript)
    except TeamsMeetingArtifactNotFoundError:
        return None, None


async def list_recording_artifacts(
    client: MicrosoftGraphClient,
    meeting_ref: TeamsMeetingRef,
) -> list[MeetingArtifact]:
    try:
        payloads = await client.collect_paginated(f"{_meeting_path(meeting_ref)}/recordings")
    except MicrosoftGraphAPIError as exc:
        raise _wrap_graph_error(
            exc,
            missing_message=f"No recordings found for Teams meeting {meeting_ref.meeting_id}",
        ) from exc
    return [_normalize_artifact("recording", payload) for payload in payloads if isinstance(payload, dict)]


async def download_recording_artifact(
    client: MicrosoftGraphClient,
    meeting_ref: TeamsMeetingRef,
    recording: MeetingArtifact,
    destination: str | Path,
) -> dict[str, Any]:
    destination_path = Path(destination)
    try:
        result = await client.download_to_file(
            _recording_download_path(meeting_ref, recording),
            destination_path,
        )
    except MicrosoftGraphAPIError as exc:
        raise _wrap_graph_error(
            exc,
            missing_message=f"Recording {recording.artifact_id} not found for meeting {meeting_ref.meeting_id}",
        ) from exc
    return {
        "artifact": recording.to_dict(),
        "path": str(destination_path),
        "size_bytes": result.get("size_bytes") or recording.size_bytes,
        "content_type": result.get("content_type") or recording.content_type,
    }


async def fetch_call_record_artifact(
    client: MicrosoftGraphClient,
    *,
    call_record_id: str,
    allow_permission_errors: bool = True,
) -> MeetingArtifact | None:
    try:
        payload = await client.get_json(f"/communications/callRecords/{quote(call_record_id, safe='')}")
    except MicrosoftGraphAPIError as exc:
        if exc.status_code in {401, 403} and allow_permission_errors:
            return None
        if exc.status_code == 404:
            return None
        raise _wrap_graph_error(exc, missing_message=f"Call record not found: {call_record_id}") from exc

    if not isinstance(payload, dict) or not payload.get("id"):
        return None

    metrics = {
        "version": payload.get("version"),
        "modalities": payload.get("modalities"),
        "participant_count": len(payload.get("participants") or []),
        "organizer": _parse_organizer_user_id(payload),
    }
    sessions = payload.get("sessions") or []
    if sessions:
        metrics["session_count"] = len(sessions)

    return MeetingArtifact(
        artifact_type="call_record",
        artifact_id=str(payload["id"]),
        display_name=payload.get("type") or "call_record",
        source_url=payload.get("webUrl"),
        created_at=payload.get("startDateTime"),
        available_at=payload.get("endDateTime"),
        metadata={"call_record": payload, "metrics": metrics},
    )


async def enrich_meeting_with_call_record(
    client: MicrosoftGraphClient,
    meeting_ref: TeamsMeetingRef,
    *,
    call_record_id: str | None = None,
    allow_permission_errors: bool = True,
) -> MeetingArtifact | None:
    resolved_call_record_id = call_record_id or meeting_ref.metadata.get("call_record_id")
    if not resolved_call_record_id:
        return None
    return await fetch_call_record_artifact(
        client,
        call_record_id=str(resolved_call_record_id),
        allow_permission_errors=allow_permission_errors,
    )
