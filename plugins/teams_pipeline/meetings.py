"""Graph-backed Teams meeting helpers for the plugin runtime."""

from __future__ import annotations

import datetime
import logging
import os
import re
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

logger = logging.getLogger(__name__)

# Configurable calendar search organizers — comma-separated UUIDs via env var.
# Avoids hardcoding tenant-specific IDs in upstream source.
_DEFAULT_CALENDAR_ORGANIZERS = os.environ.get(
    "HERMES_TEAMS_CALENDAR_ORGANIZERS", ""
)
_KNOWN_CALENDAR_ORGANIZERS = [
    uid.strip()
    for uid in _DEFAULT_CALENDAR_ORGANIZERS.split(",")
    if uid.strip()
]

from plugins.teams_pipeline.models import MeetingArtifact, TeamsMeetingRef
from tools.microsoft_graph_client import MicrosoftGraphAPIError, MicrosoftGraphClient

# ---------------------------------------------------------------------------
# Recap URL constants — parsed from Teams meeting recap URLs
# ---------------------------------------------------------------------------
_RECAP_URL_PATTERN = re.compile(
    r"teams\.microsoft\.com/l/meetingrecap",
    re.IGNORECASE,
)
_RECAP_PARAM_RE = re.compile(r"(threadId|callId|organizerId|tenantId|driveId|driveItemId|iCalUid|fileUrl)=([^&]*)")

# UUID pattern for call record ID detection
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_uuid(value: str) -> bool:
    """Check if a string looks like a UUID (call record ID format)."""
    return bool(_UUID_RE.match(value))


def _resolve_short_meet_url(short_url: str, *, follow_redirects: int = 3) -> str:
    """Resolve a short teams.microsoft.com/meet/ URL to its full meetup-join URL.

    Short meet URLs (e.g. https://teams.microsoft.com/meet/123456?p=passcode)
    use a numeric meeting ID that the Graph API cannot resolve directly.
    This function follows the HTTP redirect chain to obtain the full
    meetup-join URL which contains the GUID that Graph API understands.

    Args:
        short_url: The short meet URL to resolve.
        follow_redirects: Maximum number of redirects to follow.

    Returns:
        The resolved full meetup-join URL, or the original URL if resolution fails.
    """
    if '/meet/' not in short_url and '/meetup-join/' not in short_url:
        return short_url

    # Already a full URL — no need to resolve
    if '/meetup-join/' in short_url:
        return short_url

    current_url = short_url
    for _ in range(follow_redirects):
        try:
            req = urllib.request.Request(current_url, method='HEAD')
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
            response = urllib.request.urlopen(req, timeout=10)
            final_url = response.url
            # Check if we got a meetup-join URL
            if '/meetup-join/' in final_url:
                return final_url
            # If it's still a short URL or launcher page, try GET instead
            if '/meet/' in final_url or 'launcher' in final_url:
                req = urllib.request.Request(current_url, method='GET')
                req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
                response = urllib.request.urlopen(req, timeout=10)
                final_url = response.url
                if '/meetup-join/' in final_url:
                    return final_url
            return final_url
        except urllib.error.HTTPError as exc:
            # Some servers return 302 with Location header on HEAD
            location = exc.headers.get('Location', '') or exc.headers.get('location', '')
            if location and '/meetup-join/' in location:
                return location
            if location:
                current_url = location
                continue
            break
        except Exception:
            break
    return short_url


def _is_short_meet_url(url: str) -> bool:
    """Check if a URL is a short teams.microsoft.com/meet/ URL with numeric ID."""
    return bool(re.search(r'teams\.(microsoft|live)\.com/meet/\d+', url))


def _is_join_web_url(url: str) -> bool:
    """Check if a URL is a Teams meetup-join URL."""
    return '/meetup-join/' in url


def _extract_numeric_meeting_id(url: str) -> str | None:
    """Extract the numeric meeting ID from a short meet URL.

    Example: https://teams.microsoft.com/meet/290295479785902?p=SA9t6Fp9 -> 290295479785902
    Also handles: https://teams.live.com/meet/9322317401797?p=...
    """
    match = re.search(r'/meet/(\d+)', url)
    return match.group(1) if match else None


async def _resolve_short_meet_from_call_records(
    client: MicrosoftGraphClient,
    numeric_id: str,
    *,
    tenant_id: str | None = None,
) -> TeamsMeetingRef | None:
    """Try to resolve a short meet URL by searching call records.

    Call records may store the short meet URL in their joinWebUrl field.
    If found, we can extract the organizer and resolve through Graph API.
    """
    try:
        if hasattr(client, "collect_paginated"):
            candidates = await client.collect_paginated("/communications/callRecords")
        else:
            payload = await client.get_json("/communications/callRecords")
            candidates = payload.get("value") if isinstance(payload, dict) else None
    except MicrosoftGraphAPIError:
        return None

    if not isinstance(candidates, list):
        return None

    for call_record in candidates:
        if not isinstance(call_record, dict):
            continue
        join_web_url = str(call_record.get("joinWebUrl") or "").strip()
        if not join_web_url:
            continue
        decoded_join = unquote(join_web_url)
        if numeric_id not in decoded_join and numeric_id not in join_web_url:
            continue

        organizer_user_id = _parse_organizer_user_id(call_record)
        try:
            return await _resolve_meeting_from_join_url(
                client,
                join_web_url=join_web_url,
                organizer_user_id=organizer_user_id,
                tenant_id=tenant_id,
            )
        except TeamsMeetingNotFoundError:
            if organizer_user_id:
                metadata = {
                    key: call_record.get(key)
                    for key in ("subject", "startDateTime", "endDateTime", "createdDateTime", "participants")
                    if call_record.get(key) is not None
                }
                return TeamsMeetingRef(
                    meeting_id=str(call_record.get("id") or "").strip(),
                    organizer_user_id=organizer_user_id,
                    join_web_url=join_web_url,
                    tenant_id=tenant_id or call_record.get("tenantId"),
                    metadata=metadata,
                )
            continue

    return None


async def _resolve_short_meet_from_calendar(
    client: MicrosoftGraphClient,
    numeric_id: str,
    *,
    organizer_user_id: str | None = None,
    tenant_id: str | None = None,
) -> TeamsMeetingRef | None:
    """Try to resolve a short meet URL by searching the organizer's calendar.

    Calendar events store the full meetup-join URL in onlineMeeting.joinUrl.
    When we have a short meet URL with a numeric ID, we search the calendar
    for events that have this numeric ID in their body or join URL.
    """
    # Search the last 14 days to next 7 days
    now = datetime.datetime.now(datetime.timezone.utc)
    start = (now - datetime.timedelta(days=14)).strftime("%Y-%m-%dT00:00:00Z")
    end = (now + datetime.timedelta(days=7)).strftime("%Y-%m-%dT23:59:59Z")

    # Try the provided organizer first, then fall back to searching common organizers
    search_users = []
    if organizer_user_id:
        search_users.append(organizer_user_id)

    # Add known organizers from environment configuration
    for org in _KNOWN_CALENDAR_ORGANIZERS:
        if org not in search_users:
            search_users.append(org)

    for user_id in search_users:
        try:
            payload = await client.get_json(
                f"/users/{quote(user_id, safe='')}/calendarView",
                params={
                    "startDateTime": start,
                    "endDateTime": end,
                    "$top": 50,
                },
            )
            events = payload.get("value", []) if isinstance(payload, dict) else []

            for event in events:
                # Check onlineMeeting joinUrl
                online = event.get("onlineMeeting", {}) or {}
                join_url = online.get("joinUrl", "") or ""

                # Also check body/bodyPreview for the numeric ID
                body_preview = str(event.get("bodyPreview", "") or "")
                body_content = str(event.get("body", {}).get("content", "") or "")

                # Match by numeric ID in join URL or body
                if (
                    numeric_id in join_url
                    or numeric_id in body_preview
                    or numeric_id in body_content
                ):
                    # Found the event — use the full joinUrl from onlineMeeting
                    if join_url:
                        return await _resolve_meeting_from_join_url(
                            client,
                            join_web_url=join_url,
                            organizer_user_id=user_id,
                            tenant_id=tenant_id,
                        )

        except MicrosoftGraphAPIError:
            # User not accessible, skip
            continue

    return None


class TeamsMeetingError(RuntimeError):
    """Base class for Teams meeting pipeline failures."""


class TeamsMeetingNotFoundError(TeamsMeetingError):
    """Raised when the meeting cannot be resolved from Graph."""


class TeamsMeetingArtifactNotFoundError(TeamsMeetingError):
    """Raised when a transcript or recording cannot be found."""


class TeamsMeetingPermissionError(TeamsMeetingError):
    """Raised when Graph access is denied for the requested resource."""


def _meeting_vtc_id(meeting_id: str) -> str:
    return re.sub(r"\s+", "", meeting_id).strip()


def _meeting_path(meeting_ref: TeamsMeetingRef | str) -> str:
    if isinstance(meeting_ref, TeamsMeetingRef):
        meeting_id = meeting_ref.meeting_id
        organizer_user_id = meeting_ref.organizer_user_id
    else:
        meeting_id = str(meeting_ref)
        organizer_user_id = None

    if organizer_user_id:
        return f"/users/{quote(organizer_user_id, safe='')}/onlineMeetings/{quote(meeting_id, safe='')}"
    return f"/communications/onlineMeetings/{quote(meeting_id, safe='')}"


def _organizer_drive_recordings_path(organizer_user_id: str) -> str:
    return f"/users/{quote(organizer_user_id, safe='')}/drive/root/children"


def _organizer_drive_search_path(organizer_user_id: str, query: str) -> str:
    return f"/users/{quote(organizer_user_id, safe='')}/drive/search(q='{query}')"


async def _list_artifacts_from_organizer_drive(
    client: MicrosoftGraphClient,
    organizer_user_id: str,
    artifact_type: str,
    *,
    start_datetime: str | None = None,
    end_datetime: str | None = None,
) -> list[MeetingArtifact]:
    """List recording/transcript files from the organizer's OneDrive Recordings folder.

    Teams stores meeting recordings and transcripts in the organizer's
    OneDrive ``Recordings`` folder (non-channel meetings) or in SharePoint
    (channel meetings). This function navigates to the Recordings folder
    and returns matching files.
    """
    artifacts: list[MeetingArtifact] = []

    # Navigate to the Recordings folder
    try:
        children_path = _organizer_drive_recordings_path(organizer_user_id)
        root_items = await client.collect_paginated(children_path)
        recordings_folder = None
        for item in root_items:
            if isinstance(item, dict) and item.get("name") == "Recordings" and "folder" in item:
                recordings_folder = item
                break
        if recordings_folder is None:
            return []

        folder_id = recordings_folder.get("id", "")
        recordings_children_path = (
            f"/users/{quote(organizer_user_id, safe='')}/drive/items/{quote(folder_id, safe='')}/children"
        )
        files = await client.collect_paginated(recordings_children_path)
    except MicrosoftGraphAPIError:
        return []

    for item in files:
        if not isinstance(item, dict):
            continue
        if "folder" in item:
            continue
        name = str(item.get("name", "")).lower()
        # Filter by type
        if artifact_type == "recording":
            if not any(ext in name for ext in (".mp4", ".mov", ".avi", ".mkv", "recording")):
                continue
        else:
            if not any(ext in name for ext in (".vtt", ".txt", ".docx", "transcript")):
                continue

        # Filter by date range if provided
        created = item.get("createdDateTime", "")
        modified = item.get("lastModifiedDateTime", "")
        if start_datetime and created and created < start_datetime:
            continue
        # Transcripts are uploaded to OneDrive AFTER the meeting ends
        # (Teams needs 5-30 min to process). Don't filter by end_datetime
        # for transcripts — only recordings should use the end time cutoff.
        if artifact_type == "recording" and end_datetime and created and created > end_datetime:
            continue

        file_size = item.get("size", 0)
        download_url = item.get("@microsoft.graph.downloadUrl")
        file_name = item.get("name", artifact_type)
        artifact_id = item.get("id", file_name)
        parent_ref = item.get("parentReference", {})
        drive_id = parent_ref.get("driveId", "")

        artifacts.append(
            MeetingArtifact(
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                display_name=file_name,
                source_url=item.get("webUrl"),
                download_url=download_url,
                created_at=created or modified,
                available_at=modified or created,
                size_bytes=file_size,
                content_type=item.get("file", {}).get("mimeType", ""),
                metadata={
                    "drive_id": drive_id,
                    "parent_reference": parent_ref,
                    "item": item,
                },
            )
        )

    return artifacts


def _wrap_graph_error(exc: MicrosoftGraphAPIError, *, missing_message: str) -> TeamsMeetingError:
    if exc.status_code in (401, 403):
        return TeamsMeetingPermissionError(str(exc))
    if exc.status_code == 404:
        return TeamsMeetingNotFoundError(missing_message)
    return TeamsMeetingError(str(exc))


def _parse_organizer_user_id(payload: dict[str, Any]) -> str | None:
    # v1.0 format: organizer.identity.user.id or organizer.user.id
    organizer = payload.get("organizer")
    if isinstance(organizer, dict):
        identity = organizer.get("identity")
        if isinstance(identity, dict):
            user = identity.get("user")
            if isinstance(user, dict) and user.get("id"):
                return user.get("id")

        user = organizer.get("user")
        if isinstance(user, dict) and user.get("id"):
            return user.get("id")

    # organizer_v2 format (direct call record fetch): organizer_v2.id
    organizer_v2 = payload.get("organizer_v2")
    if isinstance(organizer_v2, dict):
        oid = organizer_v2.get("id")
        if oid and isinstance(oid, str) and oid.strip():
            return oid.strip()

    # Beta format: participants.organizer.identity.user.id
    participants = payload.get("participants")
    if isinstance(participants, dict):
        org_participant = participants.get("organizer")
        if isinstance(org_participant, dict):
            identity = org_participant.get("identity")
            if isinstance(identity, dict):
                user = identity.get("user")
                if isinstance(user, dict) and user.get("id"):
                    return user.get("id")

    return None


def _parse_thread_id(payload: dict[str, Any]) -> str | None:
    chat = payload.get("chatInfo")
    if isinstance(chat, dict):
        thread_id = chat.get("threadId")
        if thread_id:
            return str(thread_id)
    return payload.get("threadId")


def _normalize_meeting_ref(payload: dict[str, Any], *, tenant_id: str | None = None) -> TeamsMeetingRef:
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
        organizer_user_id=_parse_organizer_user_id(payload),
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


def _extract_meeting_token(join_web_url: str) -> str | None:
    for pattern in (
        r"/meet/([^/?#]+)",
        r"/chat/([^/?#]+)",
        r"/meetup-join/([^/?#]+)",
    ):
        match = re.search(pattern, join_web_url)
        if match:
            token = match.group(1).strip()
            if token:
                return token
    return None


def _pending_join_meeting_ref(join_web_url: str, *, tenant_id: str | None = None) -> TeamsMeetingRef:
    meeting_id = _extract_meeting_token(join_web_url) or join_web_url
    return TeamsMeetingRef(
        meeting_id=meeting_id,
        join_web_url=join_web_url,
        tenant_id=tenant_id,
        metadata={"pending_resolution": True, "join_web_url": join_web_url},
    )


# ---------------------------------------------------------------------------
# Recap URL parser — extracts identifiers from Teams meeting recap URLs
# ---------------------------------------------------------------------------

def parse_recap_url(url: str) -> dict[str, str] | None:
    """Parse a Teams meeting recap URL and extract structured identifiers.

    Meeting recap URLs have the format:
    https://teams.microsoft.com/l/meetingrecap?threadId=...&callId=...&organizerId=...

    Returns a dict with keys: threadId, callId, organizerId, tenantId,
    driveId, driveItemId, iCalUid, fileUrl — or None if not a recap URL.
    """
    if not _RECAP_URL_PATTERN.search(url):
        return None

    result: dict[str, str] = {}
    for match in _RECAP_PARAM_RE.finditer(url):
        key = match.group(1)
        value = unquote(match.group(2))
        result[key] = value

    if not result:
        return None

    # Extract VTC GUID from threadId
    thread_id = result.get("threadId", "")
    vtc_match = re.search(r"meeting_([A-Za-z0-9_\-]+)@thread", thread_id)
    if vtc_match:
        try:
            import base64 as _b64
            result["vtcGuid"] = _b64.b64decode(vtc_match.group(1) + "==").decode("utf-8", errors="replace")
        except Exception:
            result["vtcGuid"] = vtc_match.group(1)

    return result


# ---------------------------------------------------------------------------
# Direct call record resolution — fetch by ID (bypasses paginated scan)
# ---------------------------------------------------------------------------

async def _resolve_meeting_from_call_record_id(
    client: MicrosoftGraphClient,
    call_record_id: str,
    *,
    tenant_id: str | None = None,
) -> TeamsMeetingRef | None:
    """Resolve a meeting reference by directly fetching a call record by ID.

    This is the PRIMARY fix for the short-meet-URL issue: when the pipeline
    receives a callRecords webhook notification, the call record ID in the
    notification can be used to fetch the full call record, which contains
    the joinWebUrl, organizer info, and other metadata needed to resolve
    the online meeting and fetch transcripts.
    """
    import base64 as _b64

    try:
        record = await client.get_json(f"/communications/callRecords/{call_record_id}")
    except MicrosoftGraphAPIError:
        return None

    if not isinstance(record, dict):
        return None

    join_web_url = str(record.get("joinWebUrl") or "").strip()
    if not join_web_url:
        return None

    # Extract organizer from call record
    organizer_user_id = _parse_organizer_user_id(record)

    # Extract VTC GUID from joinWebUrl for direct meeting lookup
    vtc_guid = None
    decoded_url = unquote(join_web_url)
    vtc_match = re.search(r"meeting_([A-Za-z0-9_\-]+)@thread", decoded_url)
    if vtc_match:
        try:
            vtc_guid = _b64.b64decode(vtc_match.group(1) + "==").decode("utf-8", errors="replace")
        except Exception:
            vtc_guid = vtc_match.group(1)

    # Build metadata from call record
    metadata: dict[str, Any] = {
        "source": "call_record_direct",
        "call_record_id": call_record_id,
    }
    for key in ("subject", "startDateTime", "endDateTime", "createdDateTime", "participants"):
        if record.get(key) is not None:
            metadata[key] = record[key]

    # Try VTC GUID lookup first (most reliable for app-only auth)
    if vtc_guid:
        try:
            payload = await client.get_json(
                "/communications/onlineMeetings",
                params={"$filter": f"VideoTeleconferenceId eq '{vtc_guid}'"},
            )
            candidates = payload.get("value") if isinstance(payload, dict) else None
            if isinstance(candidates, list) and candidates:
                return _normalize_meeting_ref(candidates[0], tenant_id=tenant_id)
        except MicrosoftGraphAPIError:
            pass  # VTC lookup failed, fall through

    # Try JoinWebUrl filter on /users/{id}/onlineMeetings
    if organizer_user_id:
        escaped_url = join_web_url.replace("'", "''")
        try:
            payload = await client.get_json(
                f"/users/{quote(organizer_user_id, safe='')}/onlineMeetings",
                params={"$filter": f"JoinWebUrl eq '{escaped_url}'"},
            )
            candidates = payload.get("value") if isinstance(payload, dict) else None
            if isinstance(candidates, list) and candidates:
                return _normalize_meeting_ref(candidates[0], tenant_id=tenant_id)
        except MicrosoftGraphAPIError:
            pass  # User-scoped lookup failed, fall through

    # Last resort: return ref with call record metadata (OneDrive fallback path)
    if organizer_user_id:
        return TeamsMeetingRef(
            meeting_id=call_record_id,
            organizer_user_id=organizer_user_id,
            join_web_url=join_web_url,
            tenant_id=tenant_id or record.get("tenantId"),
            metadata=metadata,
        )

    return None


# ---------------------------------------------------------------------------
# Meeting chat transcript event lookup
# ---------------------------------------------------------------------------

async def _lookup_transcript_event_in_chat(
    client: MicrosoftGraphClient,
    chat_id: str,
) -> dict[str, Any] | None:
    """Look for a callTranscriptEventMessageDetail in the meeting chat.

    Teams posts a system event message when the transcript is ready.
    This confirms the transcript exists even if the Graph API lookup fails.
    """
    try:
        payload = await client.get_json(f"/chats/{quote(chat_id, safe='')}/messages")
        messages = payload.get("value") if isinstance(payload, dict) else None
    except MicrosoftGraphAPIError:
        return None

    if not isinstance(messages, list):
        return None

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        event_detail = msg.get("eventDetail") or {}
        if event_detail.get("@odata.type") == "#microsoft.graph.callTranscriptEventMessageDetail":
            return {
                "callId": event_detail.get("callId"),
                "transcriptICalUid": event_detail.get("callTranscriptICalUid"),
                "organizer": event_detail.get("meetingOrganizer"),
                "messageId": msg.get("id"),
                "createdDateTime": msg.get("createdDateTime"),
            }

    return None


async def _resolve_meeting_from_call_records(
    client: MicrosoftGraphClient,
    *,
    meeting_token: str,
    tenant_id: str | None = None,
) -> TeamsMeetingRef | None:
    try:
        if hasattr(client, "collect_paginated"):
            candidates = await client.collect_paginated("/communications/callRecords")
        else:
            payload = await client.get_json("/communications/callRecords")
            candidates = payload.get("value") if isinstance(payload, dict) else None
    except MicrosoftGraphAPIError as exc:
        if exc.status_code in (401, 403):
            raise _wrap_graph_error(exc, missing_message=f"Call record not found: {meeting_token}") from exc
        if exc.status_code in (400, 404):
            return None
        raise

    if not isinstance(candidates, list):
        return None

    for call_record in candidates:
        if not isinstance(call_record, dict):
            continue
        join_web_url = str(call_record.get("joinWebUrl") or "").strip()
        if not join_web_url:
            continue
        if meeting_token not in join_web_url and meeting_token not in unquote(join_web_url):
            continue

        organizer_user_id = _parse_organizer_user_id(call_record)
        try:
            return await _resolve_meeting_from_join_url(
                client,
                join_web_url=join_web_url,
                organizer_user_id=organizer_user_id,
                tenant_id=tenant_id,
            )
        except TeamsMeetingNotFoundError:
            # OnlineMeeting record may be gone for old meetings.
            # Return a meeting ref from the callRecord metadata so that
            # downstream code (OneDrive fallback) can still access the
            # organizer's drive.
            if organizer_user_id:
                metadata = {
                    key: call_record.get(key)
                    for key in ("subject", "startDateTime", "endDateTime", "createdDateTime", "participants")
                    if call_record.get(key) is not None
                }
                return TeamsMeetingRef(
                    meeting_id=str(call_record.get("id") or "").strip(),
                    organizer_user_id=organizer_user_id,
                    join_web_url=join_web_url,
                    tenant_id=tenant_id or call_record.get("tenantId"),
                    metadata=metadata,
                )
            continue

    return None


def _extract_vtc_id_from_join_url(join_web_url: str) -> str | None:
    """Extract VideoTeleconferenceId from a Teams join URL.

    Handles formats:
    - /meetup-join/19:meeting_<base64>@thread.v2/...
    - /meet/<token>
    - /chat/<token>
    """
    from urllib.parse import unquote
    decoded = unquote(join_web_url)
    # Try meeting token from meetup-join URL
    match = re.search(r"/meetup-join/([^/?#]+)", decoded)
    if match:
        token = match.group(1)
        # Token format: 19:meeting_<base64>@thread.v2
        # The VTC ID is the base64 part after "meeting_"
        inner = re.search(r"meeting_([A-Za-z0-9_\-]+)", token)
        if inner:
            return inner.group(1)
        # Fallback: use the whole token (minus thread suffix) as VTC
        clean = re.sub(r"@.*$", "", token).strip()
        if clean and clean != token:
            return clean
    # Try /meet/<token> or /chat/<token>
    for pattern in (r"/meet/([^/?#]+)", r"/chat/([^/?#]+)"):
        match = re.search(pattern, decoded)
        if match:
            token = match.group(1).strip()
            if token:
                return token
    return None


async def _resolve_meeting_from_join_url(
    client: MicrosoftGraphClient,
    *,
    join_web_url: str,
    organizer_user_id: str | None = None,
    tenant_id: str | None = None,
) -> TeamsMeetingRef:
    escaped_join_url = join_web_url.replace("'", "''")

    # Strategy 1: VideoTeleconferenceId lookup (works with app-only auth)
    vtc_id = _extract_vtc_id_from_join_url(join_web_url)
    if vtc_id:
        try:
            vtc_payload = await client.get_json(
                "/communications/onlineMeetings",
                params={"$filter": f"VideoTeleconferenceId eq '{vtc_id}'"},
            )
            vtc_candidates = vtc_payload.get("value") if isinstance(vtc_payload, dict) else None
            if isinstance(vtc_candidates, list) and vtc_candidates:
                return _normalize_meeting_ref(vtc_candidates[0], tenant_id=tenant_id)
        except MicrosoftGraphAPIError:
            pass  # VTC lookup failed, try next strategy

    # Strategy 2: JoinWebUrl filter on /communications/onlineMeetings (delegated auth only)
    try:
        payload = await client.get_json(
            "/communications/onlineMeetings",
            params={"$filter": f"JoinWebUrl eq '{escaped_join_url}'"},
        )
        candidates = payload.get("value") if isinstance(payload, dict) else None
        if isinstance(candidates, list) and candidates:
            return _normalize_meeting_ref(candidates[0], tenant_id=tenant_id)
    except MicrosoftGraphAPIError as exc:
        if exc.status_code in (401, 403):
            raise _wrap_graph_error(exc, missing_message=f"Teams meeting not found for join URL: {join_web_url}") from exc
        if exc.status_code in (400, 404):
            # JoinWebUrl filter not supported with app-only auth on /communications
            # Try user-scoped path as last resort
            if organizer_user_id:
                organizer_id = quote(organizer_user_id, safe="")
                try:
                    user_payload = await client.get_json(
                        f"/users/{organizer_id}/onlineMeetings",
                        params={"$filter": f"JoinWebUrl eq '{escaped_join_url}'"},
                    )
                    user_candidates = user_payload.get("value") if isinstance(user_payload, dict) else None
                    if isinstance(user_candidates, list) and user_candidates:
                        return _normalize_meeting_ref(user_candidates[0], tenant_id=tenant_id)
                except MicrosoftGraphAPIError:
                    pass
            raise TeamsMeetingNotFoundError(f"Teams meeting not found for join URL: {join_web_url}") from exc
        raise

    # Strategy 3: JoinWebUrl filter on /users/{id}/onlineMeetings (requires app access policy)
    if organizer_user_id:
        organizer_id = quote(organizer_user_id, safe="")
        try:
            payload = await client.get_json(
                f"/users/{organizer_id}/onlineMeetings",
                params={"$filter": f"JoinWebUrl eq '{escaped_join_url}'"},
            )
            candidates = payload.get("value") if isinstance(payload, dict) else None
            if isinstance(candidates, list) and candidates:
                return _normalize_meeting_ref(candidates[0], tenant_id=tenant_id)
        except MicrosoftGraphAPIError as exc:
            if exc.status_code in (401, 403):
                raise _wrap_graph_error(exc, missing_message=f"Teams meeting not found for join URL: {join_web_url}") from exc
            if exc.status_code in (400, 404):
                raise TeamsMeetingNotFoundError(f"Teams meeting not found for join URL: {join_web_url}") from exc
            raise

    raise TeamsMeetingNotFoundError(f"Teams meeting not found for join URL: {join_web_url}")


async def resolve_meeting_reference(
    client: MicrosoftGraphClient,
    *,
    meeting_id: str | None = None,
    join_web_url: str | None = None,
    tenant_id: str | None = None,
    recap_url: str | None = None,
) -> TeamsMeetingRef:
    # ENHANCEMENT (RCA-20260518): Recap URL resolution.
    # When a recap URL is provided, extract all identifiers and use them
    # to resolve the meeting directly via call record ID.
    if recap_url:
        recap_data = parse_recap_url(recap_url)
        if recap_data:
            logger.info("Parsed recap URL: callId=%s organizerId=%s", recap_data.get("callId"), recap_data.get("organizerId"))
            call_id = recap_data.get("callId")
            if call_id:
                direct_ref = await _resolve_meeting_from_call_record_id(
                    client, call_id,
                    tenant_id=recap_data.get("tenantId") or tenant_id,
                )
                if direct_ref is not None:
                    logger.info("Resolved meeting from recap URL: %s", call_id)
                    return direct_ref
            # Fallback: use VTC GUID from recap URL
            vtc_guid = recap_data.get("vtcGuid")
            if vtc_guid:
                try:
                    payload = await client.get_json(
                        "/communications/onlineMeetings",
                        params={"$filter": f"VideoTeleconferenceId eq '{vtc_guid}'"},
                    )
                    candidates = payload.get("value") if isinstance(payload, dict) else None
                    if isinstance(candidates, list) and candidates:
                        return _normalize_meeting_ref(candidates[0], tenant_id=tenant_id)
                except MicrosoftGraphAPIError:
                    pass
            # Last resort: build pending ref from recap data
            organizer_id = recap_data.get("organizerId")
            thread_id = recap_data.get("threadId")
            return TeamsMeetingRef(
                meeting_id=call_id or recap_url,
                organizer_user_id=organizer_id,
                join_web_url=recap_data.get("fileUrl") or recap_url,
                tenant_id=recap_data.get("tenantId") or tenant_id,
                metadata={
                    "source": "recap_url",
                    "pending_resolution": True,
                    "recap_data": recap_data,
                    "thread_id": thread_id,
                },
            )
        # recap_url did not match recap format — check if it is a short meet URL
        # or join URL and delegate to the join_web_url resolution path below.
        if _is_short_meet_url(recap_url) or _is_join_web_url(recap_url):
            logger.info("recap_url is a meet/join URL, routing to join_web_url path: %s", recap_url)
            join_web_url = recap_url
        else:
            logger.info("recap_url did not parse as recap or meet URL: %s", recap_url)

    if meeting_id:
        try:
            payload = await client.get_json(_meeting_path(meeting_id))
            if isinstance(payload, dict) and payload.get("id"):
                return _normalize_meeting_ref(payload, tenant_id=tenant_id)
        except MicrosoftGraphAPIError as exc:
            if exc.status_code in (401, 403):
                # CsApplicationAccessPolicy 403: the organizer's meetings are
                # not accessible to this app.  Fall through to call-record /
                # OneDrive fallbacks instead of failing immediately.
                if "No application access policy" in str(exc):
                    logger.debug(
                        "Skipping onlineMeetings lookup — CsApplicationAccessPolicy "
                        "not granted for meeting %s", meeting_id,
                    )
                else:
                    raise _wrap_graph_error(
                        exc, missing_message=f"Teams meeting not found: {meeting_id}",
                    ) from exc
            elif exc.status_code not in (400, 404):
                raise

        # ENHANCEMENT (RCA-20260518): Direct call record resolution.
        # When meeting_id is a UUID (call record ID), not a base64-encoded
        # meeting token, try fetching the call record directly.  This is the
        # PRIMARY fix for short-meet-URL meetings where the webhook fires
        # with a call record ID that the onlineMeetings endpoint can't resolve.
        import uuid as _uuid
        if _is_uuid(meeting_id):
            logger.debug(
                "Meeting ID looks like a call record UUID — trying direct resolution: %s",
                meeting_id,
            )
            direct_ref = await _resolve_meeting_from_call_record_id(
                client, meeting_id, tenant_id=tenant_id,
            )
            if direct_ref is not None:
                logger.info(
                    "Resolved meeting via direct call record lookup: %s → %s",
                    meeting_id, direct_ref.meeting_id,
                )
                return direct_ref

        vtc_id = _meeting_vtc_id(meeting_id)
        if vtc_id and vtc_id != meeting_id:
            try:
                payload = await client.get_json(
                    "/communications/onlineMeetings",
                    params={"$filter": f"VideoTeleconferenceId eq '{vtc_id}'"},
                )
                candidates = payload.get("value") if isinstance(payload, dict) else None
                if isinstance(candidates, list) and candidates:
                    return _normalize_meeting_ref(candidates[0], tenant_id=tenant_id)
            except MicrosoftGraphAPIError as exc:
                if exc.status_code in (401, 403):
                    if "No application access policy" in str(exc):
                        logger.debug(
                            "Skipping VTC filter — CsApplicationAccessPolicy "
                            "not granted for meeting %s", meeting_id,
                        )
                    else:
                        raise _wrap_graph_error(
                            exc, missing_message=f"Teams meeting not found: {meeting_id}",
                        ) from exc
                elif exc.status_code not in (400, 404):
                    raise

        call_record_artifact = await fetch_call_record_artifact(
            client,
            call_record_id=meeting_id,
            allow_permission_errors=False,
        )
        call_record = (call_record_artifact.metadata or {}).get("call_record") if call_record_artifact else None
        if isinstance(call_record, dict):
            join_from_call_record = str(call_record.get("joinWebUrl") or "").strip() or None
            organizer_user_id = _parse_organizer_user_id(call_record)
            if join_from_call_record:
                try:
                    return await _resolve_meeting_from_join_url(
                        client,
                        join_web_url=join_from_call_record,
                        organizer_user_id=organizer_user_id,
                        tenant_id=tenant_id,
                    )
                except TeamsMeetingNotFoundError:
                    # OnlineMeeting record may be deleted after meeting ends.
                    # Return a ref with organizer_user_id so OneDrive fallback works.
                    if organizer_user_id:
                        metadata = {
                            key: call_record.get(key)
                            for key in ("subject", "startDateTime", "endDateTime", "createdDateTime", "participants")
                            if call_record.get(key) is not None
                        }
                        return TeamsMeetingRef(
                            meeting_id=meeting_id,
                            organizer_user_id=organizer_user_id,
                            join_web_url=join_from_call_record,
                            tenant_id=tenant_id or call_record.get("tenantId"),
                            metadata=metadata,
                        )

        bridged_ref = await _resolve_meeting_from_call_records(
            client,
            meeting_token=meeting_id,
            tenant_id=tenant_id,
        )
        if bridged_ref is not None:
            return bridged_ref

        # Last resort: if we have organizer from call record, return pending ref
        if isinstance(call_record, dict):
            organizer_user_id = _parse_organizer_user_id(call_record)
            if organizer_user_id:
                return TeamsMeetingRef(
                    meeting_id=meeting_id,
                    organizer_user_id=organizer_user_id,
                    join_web_url=str(call_record.get("joinWebUrl") or "").strip() or None,
                    tenant_id=tenant_id,
                    metadata={"pending_resolution": True, "source": "call_record"},
                )

        raise TeamsMeetingNotFoundError(f"Teams meeting not found: {meeting_id}")

    if join_web_url:
        # Save original URL for short meet URL detection (before any transformation)
        original_url = join_web_url
        is_short = _is_short_meet_url(join_web_url)
        numeric_id = _extract_numeric_meeting_id(join_web_url) if is_short else None

        # For short meet URLs, skip HTTP redirect resolution (goes to launcher page)
        # and go straight to Graph API resolution + calendar/call records fallback
        if not is_short:
            resolved_url = _resolve_short_meet_url(join_web_url)
            if resolved_url != join_web_url:
                join_web_url = resolved_url

        try:
            return await _resolve_meeting_from_join_url(client, join_web_url=join_web_url, tenant_id=tenant_id)
        except TeamsMeetingNotFoundError:
            # For short meet URLs, try multiple resolution strategies
            if is_short and numeric_id:
                # Strategy 1: Search call records for matching numeric ID
                short_ref = await _resolve_short_meet_from_call_records(
                    client, numeric_id, tenant_id=tenant_id
                )
                if short_ref is not None:
                    return short_ref

                # Strategy 2: Search calendar for matching numeric ID
                calendar_ref = await _resolve_short_meet_from_calendar(
                    client, numeric_id, tenant_id=tenant_id
                )
                if calendar_ref is not None:
                    return calendar_ref

            # Fallback: try token-based call record search
            bridged_ref = await _resolve_meeting_from_call_records(
                client,
                meeting_token=_extract_meeting_token(original_url) or original_url,
                tenant_id=tenant_id,
            )
            if bridged_ref is not None:
                return bridged_ref
            return _pending_join_meeting_ref(original_url, tenant_id=tenant_id)

    raise ValueError(
        f"Cannot resolve meeting reference. "
        f"recap_url='{recap_url}' is not a recognized Teams meeting URL. "
        f"Supported formats: recap URL (/l/meetingrecap?...&callId=), "
        f"short meet URL (/meet/NUMERIC_ID), or join URL (/l/meetup-join/...)"
    )


async def _resolve_artifact_meeting_ref(
    client: MicrosoftGraphClient,
    meeting_ref: TeamsMeetingRef,
) -> TeamsMeetingRef | None:
    if meeting_ref.metadata.get("pending_resolution") and meeting_ref.join_web_url:
        try:
            return await _resolve_meeting_from_join_url(
                client,
                join_web_url=meeting_ref.join_web_url,
                tenant_id=meeting_ref.tenant_id,
            )
        except TeamsMeetingNotFoundError:
            return None
    return meeting_ref


async def _get_beta_graph_client(client: MicrosoftGraphClient) -> MicrosoftGraphClient:
    """Create a beta version of the Graph client for endpoints not available in v1.0."""
    from tools.microsoft_graph_auth import MicrosoftGraphTokenProvider
    return MicrosoftGraphClient(
        token_provider=client.token_provider,
        base_url="https://graph.microsoft.com/beta",
        timeout=client.timeout,
        max_retries=client.max_retries,
    )


async def list_transcript_artifacts(
    client: MicrosoftGraphClient,
    meeting_ref: TeamsMeetingRef,
) -> list[MeetingArtifact]:
    resolved_meeting_ref = await _resolve_artifact_meeting_ref(client, meeting_ref)
    if resolved_meeting_ref is None:
        return []
    meeting_ref = resolved_meeting_ref

    # Try v1.0 first (faster, more stable)
    try:
        payloads = await client.collect_paginated(f"{_meeting_path(meeting_ref)}/transcripts")
        if payloads:
            return [_normalize_artifact("transcript", payload) for payload in payloads if isinstance(payload, dict)]
    except MicrosoftGraphAPIError as exc:
        if exc.status_code not in (400, 401, 403, 404):
            raise _wrap_graph_error(
                exc,
                missing_message=f"No transcripts found for Teams meeting {meeting_ref.meeting_id}",
            ) from exc

    # Fallback: try beta endpoint (required for some meeting types)
    if meeting_ref.organizer_user_id:
        try:
            beta_client = await _get_beta_graph_client(client)
            beta_payloads = await beta_client.collect_paginated(
                f"{_meeting_path(meeting_ref)}/transcripts"
            )
            if beta_payloads:
                return [_normalize_artifact("transcript", payload) for payload in beta_payloads if isinstance(payload, dict)]
        except MicrosoftGraphAPIError:
            pass  # Beta also failed, try OneDrive

    # Fallback: search organizer's OneDrive for transcript files
    if meeting_ref.organizer_user_id:
        meta = meeting_ref.metadata or {}
        drive_artifacts = await _list_artifacts_from_organizer_drive(
            client,
            meeting_ref.organizer_user_id,
            "transcript",
            start_datetime=str(meta.get("startDateTime", "")),
            end_datetime=str(meta.get("endDateTime", "")),
        )
        if drive_artifacts:
            return drive_artifacts

    return []


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
    resolved_meeting_ref: TeamsMeetingRef | None = None,
    encoding: str = "utf-8",
) -> str:
    download_ref = resolved_meeting_ref or meeting_ref
    suffix = Path(transcript.display_name or "transcript.vtt").suffix or ".txt"

    # Try direct content URL from beta endpoint first (has Accept: text/vtt requirement)
    content_url = (transcript.metadata or {}).get("transcriptContentUrl")
    if content_url:
        import httpx
        token = await client.token_provider.get_access_token()
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.get(content_url, headers={
                "Authorization": f"Bearer {token}",
                "Accept": "text/vtt",
            })
            if resp.status_code == 200:
                text = resp.text.strip()
                if text:
                    return text
            # Fall through to standard download if direct URL fails

    # Standard download via Graph client
    with tempfile.NamedTemporaryFile(prefix="teams-transcript-", suffix=suffix, delete=False) as handle:
        destination = Path(handle.name)
    try:
        await client.download_to_file(_transcript_download_path(download_ref, transcript), destination)
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
    resolved_meeting_ref = await _resolve_artifact_meeting_ref(client, meeting_ref)
    if resolved_meeting_ref is None:
        return None, None

    try:
        transcripts = await list_transcript_artifacts(client, resolved_meeting_ref)
    except TeamsMeetingNotFoundError:
        return None, None

    transcript = select_preferred_transcript(transcripts)
    if transcript is None:
        return None, None
    try:
        return transcript, await download_transcript_text(
            client,
            resolved_meeting_ref,
            transcript,
            resolved_meeting_ref=resolved_meeting_ref,
        )
    except TeamsMeetingArtifactNotFoundError:
        return None, None


async def list_recording_artifacts(
    client: MicrosoftGraphClient,
    meeting_ref: TeamsMeetingRef,
) -> list[MeetingArtifact]:
    resolved_meeting_ref = await _resolve_artifact_meeting_ref(client, meeting_ref)
    if resolved_meeting_ref is None:
        return []
    meeting_ref = resolved_meeting_ref

    # Try v1.0 first
    try:
        payloads = await client.collect_paginated(f"{_meeting_path(meeting_ref)}/recordings")
        if payloads:
            return [_normalize_artifact("recording", payload) for payload in payloads if isinstance(payload, dict)]
    except MicrosoftGraphAPIError as exc:
        if exc.status_code not in (400, 401, 403, 404):
            raise _wrap_graph_error(
                exc,
                missing_message=f"No recordings found for Teams meeting {meeting_ref.meeting_id}",
            ) from exc

    # Fallback: try beta endpoint
    if meeting_ref.organizer_user_id:
        try:
            beta_client = await _get_beta_graph_client(client)
            beta_payloads = await beta_client.collect_paginated(
                f"{_meeting_path(meeting_ref)}/recordings"
            )
            if beta_payloads:
                return [_normalize_artifact("recording", payload) for payload in beta_payloads if isinstance(payload, dict)]
        except MicrosoftGraphAPIError:
            pass

    # Fallback: search organizer's OneDrive
    if meeting_ref.organizer_user_id:
        meta = meeting_ref.metadata or {}
        drive_artifacts = await _list_artifacts_from_organizer_drive(
            client,
            meeting_ref.organizer_user_id,
            "recording",
            start_datetime=str(meta.get("startDateTime", "")),
            end_datetime=str(meta.get("endDateTime", "")),
        )
        if drive_artifacts:
            return drive_artifacts

    return []


async def download_recording_artifact(
    client: MicrosoftGraphClient,
    meeting_ref: TeamsMeetingRef,
    recording: MeetingArtifact,
    destination: str | Path,
    *,
    resolved_meeting_ref: TeamsMeetingRef | None = None,
) -> dict[str, Any]:
    download_ref = resolved_meeting_ref or meeting_ref
    destination_path = Path(destination)
    try:
        result = await client.download_to_file(
            _recording_download_path(download_ref, recording),
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
        if exc.status_code in (401, 403) and allow_permission_errors:
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
