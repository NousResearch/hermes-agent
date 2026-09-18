from __future__ import annotations

from pathlib import Path

import pytest

from plugins.teams_pipeline.meetings import (
    download_transcript_text,
    resolve_meeting_reference,
)
from plugins.teams_pipeline.models import MeetingArtifact, TeamsMeetingRef


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
