"""Native Teams mentions and processing reactions carried from #67750."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import ProcessingOutcome
from tests.gateway.test_teams import TeamsAdapter, _teams_mod, _make_config


class _FakeTeamsAccount:
    def __init__(self, *, id, name, aad_object_id=None):
        self.id = id
        self.name = name
        self.aad_object_id = aad_object_id


class _FakeTeamsMessageActivityInput:
    def __init__(self):
        self.text = ""
        self.mentions = []

    def add_mention(self, account):
        self.mentions.append(account)
        self.text += f"<at>{account.name}</at>"

    def add_text(self, text):
        self.text += text


class TestTeamsSend:

    @pytest.mark.anyio
    async def test_send_calls_app_send(self):
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_result = MagicMock()
        mock_result.id = "msg-123"
        mock_app = MagicMock()
        mock_app.send = AsyncMock(return_value=mock_result)
        adapter._app = mock_app

        result = await adapter.send("conv-id", "Hello")
        assert result.success is True
        assert result.message_id == "msg-123"
        mock_app.send.assert_awaited_once_with("conv-id", "Hello")

    @pytest.mark.anyio
    async def test_send_resolves_native_mention_directive(self, monkeypatch):
        monkeypatch.setattr(_teams_mod, "Account", _FakeTeamsAccount)
        monkeypatch.setattr(
            _teams_mod,
            "MessageActivityInput",
            _FakeTeamsMessageActivityInput,
        )

        member_client = SimpleNamespace(
            get_all=AsyncMock(
                return_value=[
                    SimpleNamespace(
                        id="member-1",
                        name="Jason Treadwell",
                        aad_object_id="aad-1",
                        user_principal_name="jason@example.com",
                        email="jason@example.com",
                    )
                ]
            )
        )
        mock_result = SimpleNamespace(id="msg-mention")
        mock_app = SimpleNamespace(
            api=SimpleNamespace(
                conversations=SimpleNamespace(
                    members=lambda _chat_id: member_client,
                )
            ),
            send=AsyncMock(return_value=mock_result),
        )
        adapter = TeamsAdapter(_make_config())
        adapter._app = mock_app

        result = await adapter.send(
            "conv-id",
            "[[mention:Jason Treadwell]] Hi Jason",
        )

        assert result.success is True
        outbound = mock_app.send.await_args.args[1]
        assert outbound.text == "<at>Jason Treadwell</at> Hi Jason"
        assert outbound.mentions[0].id == "member-1"
        member_client.get_all.assert_awaited_once()

    @pytest.mark.anyio
    async def test_send_resolves_mentions_with_flattened_sdk_member_api(self, monkeypatch):
        """Current Teams SDKs expose get_members and may disable members()."""
        monkeypatch.setattr(_teams_mod, "Account", _FakeTeamsAccount)
        monkeypatch.setattr(
            _teams_mod,
            "MessageActivityInput",
            _FakeTeamsMessageActivityInput,
        )

        conversations = SimpleNamespace(
            get_members=AsyncMock(
                return_value=[
                    SimpleNamespace(
                        id="member-1",
                        name="Jason Treadwell",
                        aad_object_id="aad-1",
                        user_principal_name="jason@example.com",
                        email="jason@example.com",
                    )
                ]
            ),
            members=None,
        )
        mock_app = SimpleNamespace(
            api=SimpleNamespace(conversations=conversations),
            send=AsyncMock(return_value=SimpleNamespace(id="msg-mention")),
        )
        adapter = TeamsAdapter(_make_config())
        adapter._app = mock_app

        result = await adapter.send(
            "conv-id",
            "[[mention:Jason Treadwell]] Hi Jason",
        )

        assert result.success is True
        conversations.get_members.assert_awaited_once_with("conv-id")
        outbound = mock_app.send.await_args.args[1]
        assert outbound.mentions[0].id == "member-1"

    @pytest.mark.anyio
    async def test_standalone_send_binds_deferred_sdk_models_for_mentions(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(_teams_mod, "Account", None)
        monkeypatch.setattr(_teams_mod, "MessageActivityInput", None)

        def bind_sdk_models():
            _teams_mod.Account = _FakeTeamsAccount
            _teams_mod.MessageActivityInput = _FakeTeamsMessageActivityInput
            return True

        monkeypatch.setattr(
            _teams_mod,
            "check_teams_requirements",
            bind_sdk_models,
        )
        conversations = SimpleNamespace(
            get_members=AsyncMock(
                return_value=[
                    SimpleNamespace(
                        id="member-1",
                        name="Jason Treadwell",
                        aad_object_id="aad-1",
                    )
                ]
            ),
        )
        mock_app = SimpleNamespace(
            api=SimpleNamespace(conversations=conversations),
            send=AsyncMock(return_value=SimpleNamespace(id="msg-mention")),
        )
        adapter = TeamsAdapter(_make_config())
        adapter._app = mock_app

        result = await adapter.send(
            "conv-id",
            "[[mention:Jason Treadwell]] Hi Jason",
        )

        assert result.success is True
        outbound = mock_app.send.await_args.args[1]
        assert outbound.mentions[0].name == "Jason Treadwell"

    @pytest.mark.anyio
    async def test_send_rejects_ambiguous_mention(self):
        member_client = SimpleNamespace(
            get_all=AsyncMock(
                return_value=[
                    SimpleNamespace(id="1", name="Joel Taylor"),
                    SimpleNamespace(id="2", name="Joel Smith"),
                ]
            )
        )
        mock_app = SimpleNamespace(
            api=SimpleNamespace(
                conversations=SimpleNamespace(
                    members=lambda _chat_id: member_client,
                )
            ),
            send=AsyncMock(),
        )
        adapter = TeamsAdapter(_make_config())
        adapter._app = mock_app

        result = await adapter.send("conv-id", "[[mention:Joel]] Hello")

        assert result.success is False
        assert "Could not uniquely resolve" in result.error
        mock_app.send.assert_not_awaited()


class TestTeamsReactions:
    def _adapter(self):
        reactions = SimpleNamespace(
            add=AsyncMock(),
            delete=AsyncMock(),
        )
        adapter = TeamsAdapter(_make_config())
        adapter._app = SimpleNamespace(
            api=SimpleNamespace(reactions=reactions),
        )
        return adapter, reactions

    @pytest.mark.anyio
    async def test_add_and_remove_reaction_maps_emoji(self):
        adapter, reactions = self._adapter()

        added = await adapter.add_reaction("conv-id", "👀", "msg-1")
        removed = await adapter.remove_reaction("conv-id", "msg-1")

        assert added == {
            "success": True,
            "message_id": "msg-1",
            "reaction_id": "1f440_eyes",
        }
        assert removed["success"] is True
        reactions.add.assert_awaited_once_with(
            "conv-id",
            "msg-1",
            "1f440_eyes",
        )
        reactions.delete.assert_awaited_once_with(
            "conv-id",
            "msg-1",
            "1f440_eyes",
        )

    @pytest.mark.anyio
    async def test_add_reaction_rejects_unknown_unicode(self):
        adapter, reactions = self._adapter()

        result = await adapter.add_reaction("conv-id", "😺", "msg-1")

        assert result["success"] is False
        assert "unsupported Teams emoji" in result["error"]
        reactions.add.assert_not_awaited()

    @pytest.mark.anyio
    async def test_processing_success_swaps_eyes_for_check(self):
        adapter, reactions = self._adapter()
        event = MagicMock()
        event.source.chat_id = "conv-id"
        event.message_id = "msg-1"

        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

        assert [entry.args for entry in reactions.add.await_args_list] == [
            ("conv-id", "msg-1", "1f440_eyes"),
            ("conv-id", "msg-1", "2705_whiteheavycheckmark"),
        ]
        reactions.delete.assert_awaited_once_with(
            "conv-id",
            "msg-1",
            "1f440_eyes",
        )

    @pytest.mark.anyio
    async def test_processing_reactions_can_be_disabled_from_platform_config(self):
        config = GatewayConfig.from_dict({
            "platforms": {
                "teams": {
                    "enabled": True,
                    "extra": {"reactions": False},
                },
            },
        })
        adapter = TeamsAdapter(config.platforms[Platform("teams")])
        reactions = SimpleNamespace(
            add=AsyncMock(),
            delete=AsyncMock(),
        )
        adapter._app = SimpleNamespace(
            api=SimpleNamespace(reactions=reactions),
        )
        event = MagicMock()
        event.source.chat_id = "conv-id"
        event.message_id = "msg-1"

        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

        reactions.add.assert_not_awaited()
        reactions.delete.assert_not_awaited()
