"""Graph-backed Teams meeting helpers for the plugin runtime."""

from __future__ import annotations

import base64
import binascii
import re
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Awaitable
from urllib.parse import quote, unquote

from plugins.teams_pipeline.models import MeetingArtifact, TeamsMeetingRef
from tools.microsoft_graph_client import MicrosoftGraphAPIError, MicrosoftGraphClient

# Graph uses both slash keys (users/{id}/...) and quoted keys (users('{id}')/...),
# so every segment pattern has a quoted group and a slash group.
_USERS_MEETING_RE = re.compile(r"(?i)(?:^|/)users(?:\('([^']+)'\)|/([^/'()]+))/onlineMeetings(?:\('([^']+)'\)|/([^/'?]+))")
_COMM_MEETING_RE = re.compile(r"(?i)(?:^|/)communications/onlineMeetings(?:\('([^']+)'\)|/([^/'?]+))")
_TRANSCRIPT_RE = re.compile(r"(?i)/transcripts(?:\('([^']+)'\)|/([^/'?]+))")
_RECORDING_RE = re.compile(r"(?i)/recordings(?:\('([^']+)'\)|/([^/'?]+))")
# Collection function names that the meeting regex can capture in place of a real meeting id.
_RESOURCE_SENTINELS = frozenset({"getalltranscripts", "getallrecordings", "transcripts", "recordings"})


class TeamsMeetingError(RuntimeError): """Base class for Teams meeting pipeline failures."""
class TeamsMeetingNotFoundError(TeamsMeetingError): """Raised when the meeting cannot be resolved from Graph."""
class TeamsMeetingArtifactNotFoundError(TeamsMeetingError): """Raised when a transcript or recording cannot be found."""
class TeamsMeetingPermissionError(TeamsMeetingError): """Raised when Graph access is denied for the requested resource."""


def _match_id(match: re.Match[str] | None, *groups: int) -> str | None:
    """Unquoted, stripped value of the first non-empty capture group, or None."""
    if match is None:
        return None
    return unquote(next((match.group(g) for g in groups if match.group(g)), "")).strip() or None


def parse_graph_meeting_resource(resource: str) -> dict[str, str | None]:
    """Parse organizer, meeting, and artifact ids from a Graph resource or @odata.id."""
    text = str(resource or "").strip()
    users_match = _USERS_MEETING_RE.search(text)
    meeting_id = _match_id(users_match, 3, 4) or _match_id(_COMM_MEETING_RE.search(text), 1, 2)
    if meeting_id and meeting_id.lower() in _RESOURCE_SENTINELS:
        meeting_id = None
    return {
        "organizer_user_id": _match_id(users_match, 1, 2),
        "meeting_id": meeting_id,
        "transcript_id": _match_id(_TRANSCRIPT_RE.search(text), 1, 2),
        "recording_id": _match_id(_RECORDING_RE.search(text), 1, 2)}


def looks_like_transcript_id(value: str, *, odata_type: str | None = None) -> bool:
    """True when a Graph id is a callTranscript artifact rather than an onlineMeeting."""
    if "calltranscript" in str(odata_type or "").lower():
        return True
    return "transcript" in str(value or "").lower() or "transcript" in _decoded_id_hint(str(value or ""))


def _decoded_id_hint(value: str) -> str:
    """Best-effort base64 decode of a Graph id: getAllTranscripts ``resourceData.id`` blobs only carry
    their ``-TranscriptV2`` marker in the *decoded* payload. Lowercase decoded text, or "" if undecodable."""
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


def _meetings_collection(organizer_user_id: str | None) -> str:
    """Organizer-scoped collection when the organizer is known (required for app-only Graph access)."""
    return f"/users/{quote(organizer_user_id, safe='')}/onlineMeetings" if organizer_user_id else "/communications/onlineMeetings"


def _meeting_path(meeting_ref: TeamsMeetingRef) -> str:
    return f"{_meetings_collection(meeting_ref.organizer_user_id)}/{quote(meeting_ref.meeting_id, safe='')}"


def _wrap_graph_error(exc: MicrosoftGraphAPIError, *, missing_message: str) -> TeamsMeetingError:
    if exc.status_code in {401, 403}:
        return TeamsMeetingPermissionError(str(exc))
    return TeamsMeetingNotFoundError(missing_message) if exc.status_code == 404 else TeamsMeetingError(str(exc))


async def _graph(awaitable: Awaitable[Any], *, missing_message: str) -> Any:
    """Await a Graph client call, translating MicrosoftGraphAPIError into TeamsMeetingError subclasses."""
    try:
        return await awaitable
    except MicrosoftGraphAPIError as exc:
        raise _wrap_graph_error(exc, missing_message=missing_message) from exc


def _parse_organizer_user_id(payload: dict[str, Any]) -> str | None:
    organizer = payload.get("organizer")
    identity = organizer.get("identity") if isinstance(organizer, dict) else None
    user = identity.get("user") if isinstance(identity, dict) else None
    return user.get("id") if isinstance(user, dict) else None


def _normalize_meeting_ref(payload: dict[str, Any], *, tenant_id: str | None = None, organizer_user_id: str | None = None) -> TeamsMeetingRef:
    metadata = {key: payload.get(key) for key in ("subject", "startDateTime", "endDateTime", "createdDateTime", "participants")
                if payload.get(key) is not None}
    chat = payload.get("chatInfo")
    thread_id = str(chat["threadId"]) if isinstance(chat, dict) and chat.get("threadId") else payload.get("threadId")
    return TeamsMeetingRef(
        meeting_id=str(payload.get("id") or "").strip(),
        organizer_user_id=organizer_user_id or _parse_organizer_user_id(payload),
        join_web_url=payload.get("joinWebUrl"), calendar_event_id=payload.get("calendarEventId"),
        thread_id=thread_id, tenant_id=tenant_id or payload.get("tenantId"), metadata=metadata)


def _normalize_artifact(artifact_type: str, payload: dict[str, Any]) -> MeetingArtifact:
    return MeetingArtifact(
        artifact_type=artifact_type,  # type: ignore[arg-type]
        artifact_id=str(payload.get("id") or "").strip(),
        display_name=payload.get("displayName") or payload.get("name"),
        content_type=payload.get("contentType") or payload.get("fileMimeType"),
        source_url=payload.get("webUrl") or payload.get("contentUrl"),
        download_url=(payload.get("@microsoft.graph.downloadUrl") or payload.get("downloadUrl")
                      or payload.get("recordingContentUrl") or payload.get("transcriptContentUrl")),
        created_at=payload.get("createdDateTime"),
        available_at=payload.get("lastModifiedDateTime") or payload.get("meetingEndDateTime"),
        size_bytes=payload.get("size"), metadata=dict(payload))


def _transcript_sort_key(artifact: MeetingArtifact) -> tuple[int, int, str]:
    """Prefer completed, downloadable, most recent transcripts (in that priority order)."""
    status = str(artifact.metadata.get("status") or "").lower()
    has_download = int(bool(artifact.download_url or artifact.source_url))
    is_completed = int(status in {"available", "completed", "succeeded"})
    stamp = artifact.available_at or artifact.created_at
    return (is_completed, has_download, stamp.isoformat() if stamp is not None else "")


async def resolve_meeting_reference(
    client: MicrosoftGraphClient, *, meeting_id: str | None = None, join_web_url: str | None = None,
    tenant_id: str | None = None, organizer_user_id: str | None = None) -> TeamsMeetingRef:
    if meeting_id and looks_like_transcript_id(meeting_id):
        if not join_web_url:
            raise TeamsMeetingError("Refusing to GET /communications/onlineMeetings/{id} with a transcript id. "
                                    "Graph v1.0 does not support that id format; use the organizer-scoped meeting "
                                    "id from the notification @odata.id, or a join URL.")
        meeting_id = None
    collection = _meetings_collection(organizer_user_id)
    if meeting_id:
        missing = f"Teams meeting not found: {meeting_id}"
        path = f"{collection}/{quote(meeting_id, safe='')}"
        payload = await _graph(client.get_json(path), missing_message=missing)
        if not isinstance(payload, dict) or not payload.get("id"):
            raise TeamsMeetingNotFoundError(missing)
    elif join_web_url:
        missing = f"Teams meeting not found for join URL: {join_web_url}"
        params = {"$filter": f"JoinWebUrl eq '{join_web_url.replace(chr(39), chr(39) * 2)}'"}
        listing = await _graph(client.get_json(collection, params=params), missing_message=missing)
        candidates = listing.get("value") if isinstance(listing, dict) else None
        if not isinstance(candidates, list) or not candidates:
            raise TeamsMeetingNotFoundError(missing)
        payload = candidates[0]
    else:
        raise ValueError("Either meeting_id or join_web_url is required.")
    return _normalize_meeting_ref(payload, tenant_id=tenant_id, organizer_user_id=organizer_user_id)


async def _list_artifacts(client: MicrosoftGraphClient, meeting_ref: TeamsMeetingRef, *, artifact_type: str) -> list[MeetingArtifact]:
    collection = f"{artifact_type}s"
    payloads = await _graph(
        client.collect_paginated(f"{_meeting_path(meeting_ref)}/{collection}"),
        missing_message=f"No {collection} found for Teams meeting {meeting_ref.meeting_id}")
    return [_normalize_artifact(artifact_type, payload) for payload in payloads if isinstance(payload, dict)]


list_transcript_artifacts = partial(_list_artifacts, artifact_type="transcript")
list_recording_artifacts = partial(_list_artifacts, artifact_type="recording")


async def _download_artifact(
    client: MicrosoftGraphClient, meeting_ref: TeamsMeetingRef, artifact: MeetingArtifact, destination: Path, *,
    kind: str, **download_kwargs: Any) -> dict[str, Any]:
    path = artifact.download_url or f"{_meeting_path(meeting_ref)}/{kind}s/{quote(artifact.artifact_id, safe='')}/content"
    return await _graph(
        client.download_to_file(path, destination, **download_kwargs),
        missing_message=f"{kind.capitalize()} {artifact.artifact_id} not found for meeting {meeting_ref.meeting_id}")


def select_preferred_transcript(candidates: list[MeetingArtifact]) -> MeetingArtifact | None:
    transcripts = [candidate for candidate in candidates if candidate.artifact_type == "transcript"]
    return max(transcripts, key=_transcript_sort_key) if transcripts else None


async def download_transcript_text(
    client: MicrosoftGraphClient, meeting_ref: TeamsMeetingRef, transcript: MeetingArtifact, *, encoding: str = "utf-8") -> str:
    suffix = Path(transcript.display_name or "transcript.vtt").suffix or ".txt"
    with tempfile.TemporaryDirectory(prefix="teams-transcript-", ignore_cleanup_errors=True) as tmp_dir:
        destination = Path(tmp_dir) / f"transcript{suffix}"
        # Graph's transcript /content endpoint rejects JSON content negotiation.
        await _download_artifact(client, meeting_ref, transcript, destination, kind="transcript", headers={"Accept": "text/vtt"})
        text = destination.read_text(encoding=encoding).strip()
    if not text:
        raise TeamsMeetingArtifactNotFoundError(f"Transcript {transcript.artifact_id} for meeting {meeting_ref.meeting_id} was empty.")
    return text


async def fetch_preferred_transcript_text(client: MicrosoftGraphClient, meeting_ref: TeamsMeetingRef) -> tuple[MeetingArtifact | None, str | None]:
    transcript = select_preferred_transcript(await list_transcript_artifacts(client, meeting_ref))
    if transcript is None:
        return None, None
    try:
        return transcript, await download_transcript_text(client, meeting_ref, transcript)
    except TeamsMeetingArtifactNotFoundError:
        return None, None


async def download_recording_artifact(
    client: MicrosoftGraphClient, meeting_ref: TeamsMeetingRef, recording: MeetingArtifact, destination: str | Path) -> dict[str, Any]:
    destination_path = Path(destination)
    result = await _download_artifact(client, meeting_ref, recording, destination_path, kind="recording")
    return {
        "artifact": recording.to_dict(), "path": str(destination_path),
        "size_bytes": result.get("size_bytes") or recording.size_bytes,
        "content_type": result.get("content_type") or recording.content_type}


async def enrich_meeting_with_call_record(
    client: MicrosoftGraphClient, meeting_ref: TeamsMeetingRef, *, call_record_id: str | None = None, allow_permission_errors: bool = True
) -> MeetingArtifact | None:
    """Call records need the extra CallRecords.Read.All scope, so denial is optional-soft (None)."""
    if not (call_record_id := str(call_record_id or meeting_ref.metadata.get("call_record_id") or "")):
        return None
    try:
        payload = await client.get_json(f"/communications/callRecords/{quote(call_record_id, safe='')}")
    except MicrosoftGraphAPIError as exc:
        if exc.status_code == 404 or (exc.status_code in {401, 403} and allow_permission_errors):
            return None
        raise _wrap_graph_error(exc, missing_message=f"Call record not found: {call_record_id}") from exc
    if not isinstance(payload, dict) or not payload.get("id"):
        return None
    metrics = {"version": payload.get("version"), "modalities": payload.get("modalities"),
               "participant_count": len(payload.get("participants") or []), "organizer": _parse_organizer_user_id(payload)}
    if sessions := payload.get("sessions"):
        metrics["session_count"] = len(sessions)
    return MeetingArtifact(
        artifact_type="call_record", artifact_id=str(payload["id"]), display_name=payload.get("type") or "call_record",
        source_url=payload.get("webUrl"), created_at=payload.get("startDateTime"), available_at=payload.get("endDateTime"),
        metadata={"call_record": payload, "metrics": metrics})


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

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
# ---- END PLUGIN-COMPAT ----
