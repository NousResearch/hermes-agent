from __future__ import annotations

from pathlib import Path

import pytest

from plugins.teams_pipeline.meetings import (
    TeamsMeetingArtifactNotFoundError,
    TeamsMeetingError,
    TeamsMeetingNotFoundError,
    download_recording_artifact,
    download_transcript_text,
    fetch_preferred_transcript_text,
    resolve_meeting_reference,
)
from plugins.teams_pipeline.models import MeetingArtifact, TeamsMeetingRef
from tools.microsoft_graph_client import MicrosoftGraphAPIError


class FakeGraphClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def get_json(self, path, *, params=None):
        self.calls.append((path, params))
        return self.payload


@pytest.mark.anyio
async def test_join_url_can_use_organizer_scoped_graph_lookup():
    client = FakeGraphClient({"value": [{"id": "meeting-1", "joinWebUrl": "https://teams.microsoft.com/meet/code"}]})

    meeting = await resolve_meeting_reference(
        client,
        join_web_url="https://teams.microsoft.com/meet/code",
        organizer_user_id="organizer-1",
    )

    assert meeting.meeting_id == "meeting-1"
    assert meeting.organizer_user_id == "organizer-1"
    assert client.calls == [
        (
            "/users/organizer-1/onlineMeetings",
            {"$filter": "JoinWebUrl eq 'https://teams.microsoft.com/meet/code'"},
        )
    ]


@pytest.mark.anyio
async def test_transcript_download_requests_graph_vtt_content():
    class FakeDownloadClient:
        def __init__(self):
            self.calls = []

        async def download_to_file(self, path, destination, *, headers=None):
            self.calls.append((path, headers))
            Path(destination).write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<v Speaker>Hello</v>\n",
                encoding="utf-8",
            )
            return {"content_type": "text/vtt"}

    client = FakeDownloadClient()
    meeting = TeamsMeetingRef(
        meeting_id="meeting-1",
        organizer_user_id="organizer-1",
    )
    transcript = MeetingArtifact(
        artifact_type="transcript",
        artifact_id="transcript-1",
        display_name="transcript.vtt",
    )

    text = await download_transcript_text(client, meeting, transcript)

    assert text.startswith("WEBVTT")
    assert client.calls == [
        (
            "/users/organizer-1/onlineMeetings/meeting-1/transcripts/transcript-1/content",
            {"Accept": "text/vtt"},
        )
    ]



class Artifact404Client:
    def __init__(self, *, transcripts=None):
        self.transcripts = transcripts or []

    async def collect_paginated(self, path, *, params=None, headers=None):
        return self.transcripts if path.endswith("/transcripts") else []

    async def download_to_file(self, path, destination, *, headers=None):
        raise MicrosoftGraphAPIError(404, "GET", path, "Not found")


@pytest.mark.anyio
async def test_transcript_content_404_is_artifact_missing_and_allows_fallback():
    payload = {
        "id": "tx-404",
        "displayName": "meeting.vtt",
        "status": "running",
        "lastModifiedDateTime": "2026-05-01T00:00:00Z",
    }
    client = Artifact404Client(transcripts=[payload])
    meeting = TeamsMeetingRef(meeting_id="meeting-1", organizer_user_id="organizer-1")

    artifact, text = await fetch_preferred_transcript_text(client, meeting)

    assert artifact is None
    assert text is None


@pytest.mark.anyio
async def test_recording_content_404_uses_artifact_error_not_meeting_error(tmp_path):
    client = Artifact404Client()
    meeting = TeamsMeetingRef(meeting_id="meeting-1", organizer_user_id="organizer-1")
    recording = MeetingArtifact(
        artifact_type="recording",
        artifact_id="rec-404",
        display_name="recording.mp4",
    )

    with pytest.raises(TeamsMeetingArtifactNotFoundError):
        await download_recording_artifact(client, meeting, recording, tmp_path / "recording.mp4")


@pytest.mark.anyio
async def test_artifact_non_404_still_uses_normal_graph_error_wrapping(tmp_path):
    class ErrorClient:
        async def download_to_file(self, path, destination, *, headers=None):
            raise MicrosoftGraphAPIError(500, "GET", path, "Internal Server Error")

    meeting = TeamsMeetingRef(meeting_id="meeting-1", organizer_user_id="organizer-1")
    transcript = MeetingArtifact(
        artifact_type="transcript",
        artifact_id="tx-500",
        display_name="transcript.vtt",
    )

    with pytest.raises(TeamsMeetingError) as exc_info:
        await download_transcript_text(ErrorClient(), meeting, transcript)

    assert not isinstance(exc_info.value, (TeamsMeetingArtifactNotFoundError, TeamsMeetingNotFoundError))
