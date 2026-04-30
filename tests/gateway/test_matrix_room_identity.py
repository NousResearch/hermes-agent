"""Tests for Matrix room identity resolution.

Uses sanitized example.org identifiers only. Do not add real homeservers, room IDs,
or person/project names to this upstream-facing test module.
"""

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.matrix import MatrixAdapter


ROOM_ID = "!roomid:example.org"
SENDER_ID = "@user:example.org"
BOT_ID = "@bot:example.org"


class FakeStateEvent:
    def __init__(self, **values):
        self.__dict__.update(values)


class FakeStateStore:
    async def get_members(self, room_id):
        assert str(room_id) == ROOM_ID
        return {
            SENDER_ID: {"membership": "join"},
            BOT_ID: {"membership": "join"},
        }


class FakeClient:
    def __init__(self, *, direct_rooms=None, room_name=None):
        self.direct_rooms = direct_rooms or []
        self.room_name = room_name
        self.state_store = FakeStateStore()

    async def get_account_data(self, event_type):
        assert event_type == "m.direct"
        return {SENDER_ID: self.direct_rooms}

    async def get_state_event(self, room_id, event_type):
        assert str(room_id) == ROOM_ID
        event_type = str(event_type)
        if event_type == "m.room.name":
            return FakeStateEvent(name=self.room_name) if self.room_name else None
        if event_type == "m.room.canonical_alias":
            return FakeStateEvent(alias="#example-project:example.org")
        return None


def make_adapter(fake_client):
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": BOT_ID,
                "require_mention": False,
            },
        )
    )
    adapter._client = fake_client
    adapter._joined_rooms = {ROOM_ID}
    adapter._dm_rooms = {}
    adapter._require_mention = False
    return adapter


@pytest.mark.asyncio
async def test_named_two_member_room_is_not_dm():
    adapter = make_adapter(FakeClient(direct_rooms=[], room_name="Project - Example"))

    identity = await adapter._resolve_room_identity(ROOM_ID)

    assert identity.room_name == "Project - Example"
    assert identity.joined_member_count == 2
    assert identity.is_direct_account_data is False
    assert identity.chat_type == "group"
    assert await adapter._is_dm_room(ROOM_ID) is False


@pytest.mark.asyncio
async def test_two_member_room_without_m_direct_is_not_dm():
    adapter = make_adapter(FakeClient(direct_rooms=[], room_name=None))

    identity = await adapter._resolve_room_identity(ROOM_ID)

    assert identity.joined_member_count == 2
    assert identity.is_direct_account_data is False
    assert identity.chat_type == "group"


@pytest.mark.asyncio
async def test_m_direct_room_without_name_is_dm():
    adapter = make_adapter(FakeClient(direct_rooms=[ROOM_ID], room_name=None))

    identity = await adapter._resolve_room_identity(ROOM_ID)

    assert identity.is_direct_account_data is True
    assert identity.chat_type == "dm"


@pytest.mark.asyncio
async def test_named_m_direct_room_surfaces_conflict_but_uses_named_room_context():
    adapter = make_adapter(FakeClient(direct_rooms=[ROOM_ID], room_name="Project - Example"))

    identity = await adapter._resolve_room_identity(ROOM_ID)

    assert identity.is_direct_account_data is True
    assert identity.has_direct_name_conflict is True
    assert identity.chat_type == "group"
    assert identity.display_name == "Project - Example"


@pytest.mark.asyncio
async def test_get_chat_info_reuses_room_identity_metadata():
    adapter = make_adapter(FakeClient(direct_rooms=[], room_name="Project - Example"))

    info = await adapter.get_chat_info(ROOM_ID)

    assert info == {"name": "Project - Example", "type": "group"}


@pytest.mark.asyncio
async def test_resolve_message_context_passes_room_name_to_source():
    adapter = make_adapter(FakeClient(direct_rooms=[], room_name="Project - Example"))
    adapter._background_read_receipt = lambda room_id, event_id: None

    async def fake_display_name(room_id, sender):
        assert room_id == ROOM_ID
        assert sender == SENDER_ID
        return "Example User"

    adapter._get_display_name = fake_display_name

    ctx = await adapter._resolve_message_context(
        ROOM_ID,
        SENDER_ID,
        "$eventid",
        "hello",
        {"body": "hello"},
        {},
    )

    assert ctx is not None
    _body, is_dm, chat_type, _thread_id, _display_name, source = ctx
    assert is_dm is False
    assert chat_type == "group"
    assert source.chat_id == ROOM_ID
    assert source.chat_name == "Project - Example"
    assert source.description == "group: Project - Example, thread: $eventid"
