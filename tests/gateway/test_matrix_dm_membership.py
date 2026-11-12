"""Matrix DM classification from joined-member state."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from tests.gateway.matrix_helpers import _make_adapter


class TestMatrixDmDetection:
    def setup_method(self):
        self.adapter = _make_adapter()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "members,is_direct,expected_type,expected_count",
        [
            (["@bot:ex.org", "@alice:ex.org"], False, "dm", 2),
            (["@bot:ex.org", "@alice:ex.org"], True, "dm", 2),
            (["@bot:ex.org"], True, "room", 1),
            (["@alice:ex.org", "@bob:ex.org"], True, "room", 2),
            (["@bot:ex.org", "@alice:ex.org", "@bob:ex.org"], True, "room", 3),
            (None, True, "room", None),
        ],
    )
    async def test_only_bot_and_one_joined_person_bypass_room_gate(
        self, members, is_direct, expected_type, expected_count
    ):
        room_id = "!room:ex.org"
        self.adapter._user_id = "@bot:ex.org"
        self.adapter._dm_rooms = {room_id: is_direct}
        self.adapter._allowed_room_ids = {"!allowed:ex.org"}
        self.adapter._client = MagicMock()
        self.adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
        self.adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)
        self.adapter._client.state_store.get_members = AsyncMock(return_value=members)

        identity = await self.adapter._resolve_room_identity(room_id)
        allowed = await self.adapter._is_allowed_matrix_room_event(room_id)

        assert (identity.chat_type, identity.joined_member_count,
                identity.is_direct_account_data, allowed) == (
            expected_type, expected_count, is_direct, expected_type == "dm"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("members,accepted", [
        (["@bot:ex.org", "@alice:ex.org"], True),
        (["@bot:ex.org", "@alice:ex.org", "@bob:ex.org"], False),
    ])
    async def test_only_two_person_room_bypasses_mention_gate(self, members, accepted):
        room_id = "!room:ex.org"
        self.adapter._user_id = "@bot:ex.org"
        self.adapter._dm_rooms = {room_id: True}
        self.adapter._client = MagicMock()
        self.adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
        self.adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)
        self.adapter._client.state_store.get_members = AsyncMock(return_value=members)
        self.adapter._get_display_name = AsyncMock(return_value="Alice")
        self.adapter._background_read_receipt = MagicMock()
        self.adapter._require_mention = True

        context = await self.adapter._resolve_message_context(
            room_id, "@alice:ex.org", "$event", "hello", {"body": "hello"}, {}
        )

        assert (context is not None) is accepted

    @pytest.mark.asyncio
    async def test_joined_members_api_supplies_bot_membership_when_store_is_empty(self):
        room_id = "!room:ex.org"
        self.adapter._user_id = "@bot:ex.org"
        self.adapter._client = MagicMock()
        self.adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
        self.adapter._client.state_store.has_full_member_list = AsyncMock(return_value=False)
        self.adapter._client.state_store.get_members = AsyncMock(return_value=None)
        self.adapter._client.get_joined_members = AsyncMock(return_value={
            "@bot:ex.org": {}, "@alice:ex.org": {},
        })

        identity = await self.adapter._resolve_room_identity(room_id)

        assert (identity.chat_type, identity.joined_member_count) == ("dm", 2)
        self.adapter._client.get_joined_members.assert_any_await(room_id)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("api_fails,expected_count", [(False, 3), (True, None)])
    async def test_partial_member_cache_cannot_bypass_room_gate(self, api_fails, expected_count):
        room_id = "!room:ex.org"
        self.adapter._user_id = "@bot:ex.org"
        self.adapter._allowed_room_ids = {"!allowed:ex.org"}
        self.adapter._client = MagicMock()
        self.adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
        self.adapter._client.state_store.has_full_member_list = AsyncMock(return_value=False)
        self.adapter._client.state_store.get_members = AsyncMock(
            return_value=["@bot:ex.org", "@alice:ex.org"]
        )
        self.adapter._client.get_joined_members = AsyncMock(
            side_effect=Exception("unavailable") if api_fails else None,
            return_value={"@bot:ex.org": {}, "@alice:ex.org": {}, "@bob:ex.org": {}},
        )

        identity = await self.adapter._resolve_room_identity(room_id)
        allowed = await self.adapter._is_allowed_matrix_room_event(room_id)

        assert (identity.chat_type, identity.joined_member_count, allowed) == (
            "room", expected_count, False,
        )
        self.adapter._client.state_store.get_members.assert_not_awaited()
        self.adapter._client.get_joined_members.assert_any_await(room_id)

    @pytest.mark.asyncio
    async def test_invited_user_does_not_change_joined_member_classification(self):
        from plugins.platforms.matrix.adapter import Membership

        room_id = "!room:ex.org"
        self.adapter._user_id = "@bot:ex.org"
        self.adapter._client = MagicMock()
        self.adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
        self.adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)

        async def get_members(room, *, memberships=(Membership.JOIN, Membership.INVITE)):
            users = {"@bot:ex.org", "@alice:ex.org"}
            if Membership.INVITE in memberships:
                users.add("@invited:ex.org")
            return users

        self.adapter._client.state_store.get_members = AsyncMock(side_effect=get_members)

        identity = await self.adapter._resolve_room_identity(room_id)

        assert (identity.chat_type, identity.joined_member_count) == ("dm", 2)
        self.adapter._client.state_store.get_members.assert_awaited_once_with(
            room_id, memberships=(Membership.JOIN,)
        )

    @pytest.mark.asyncio
    async def test_password_login_adopts_canonical_user_id_for_dm_membership(self):
        """A configured MXID that differs in case from the server's still finds DMs."""
        self.adapter._access_token = ""
        self.adapter._password = "secret"
        self.adapter._configured_user_id = "@Bot:ex.org"
        client = MagicMock()

        async def login(**kwargs):
            client.mxid = "@bot:ex.org"
            return MagicMock(user_id="@bot:ex.org", device_id="DEV")

        client.login = login
        assert await self.adapter._connect_authenticate(client, MagicMock()) is True

        client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
        client.state_store.has_full_member_list = AsyncMock(return_value=True)
        client.state_store.get_members = AsyncMock(return_value=["@bot:ex.org", "@alice:ex.org"])
        self.adapter._client = client

        identity = await self.adapter._resolve_room_identity("!dm:ex.org")

        assert (self.adapter._user_id, identity.chat_type, identity.joined_member_count) == (
            "@bot:ex.org", "dm", 2,
        )
