from __future__ import annotations

import pytest

from plugins.teams_pipeline.meetings import resolve_meeting_reference


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