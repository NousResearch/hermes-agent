"""Carried exact Teams reply-context behavior, independent of upstream adapter tests."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from tests.gateway.test_teams import TeamsAdapter, _teams_mod, _make_config

@pytest.mark.anyio
async def test_send_records_only_confirmed_exact_outbound_context(monkeypatch):
    record = MagicMock()
    monkeypatch.setattr(_teams_mod, "_record_outbound_reply_context", record)
    adapter = TeamsAdapter(_make_config())
    adapter._app = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id="root-message-1")))
    result = await adapter.send("19:channel@thread.tacv2", "Daily brief part 2: decisions")
    assert result.success is True
    record.assert_called_once_with("19:channel@thread.tacv2", "root-message-1", "Daily brief part 2: decisions")
    record.reset_mock()
    adapter._app.send.return_value = SimpleNamespace(id=None)
    await adapter.send("19:channel@thread.tacv2", "Unconfirmed delivery")
    record.assert_not_called()


class TestTeamsMessageHandling:
    def _make_activity(
        self,
        *,
        text="Hello",
        from_id="user-123",
        from_aad_id="aad-456",
        from_name="Test User",
        conversation_id="19:abc@thread.v2",
        conversation_type="personal",
        tenant_id="tenant-789",
        activity_id="activity-001",
        attachments=None,
    ):
        activity = MagicMock()
        activity.text = text
        activity.id = activity_id
        activity.from_ = MagicMock()
        activity.from_.id = from_id
        activity.from_.aad_object_id = from_aad_id
        activity.from_.name = from_name
        activity.conversation = MagicMock()
        activity.conversation.id = conversation_id
        activity.conversation.conversation_type = conversation_type
        activity.conversation.name = "Test Chat"
        activity.conversation.tenant_id = tenant_id
        activity.attachments = attachments or []
        return activity

    def _make_ctx(self, activity):
        ctx = MagicMock()
        ctx.activity = activity
        return ctx

    @pytest.mark.anyio
    async def test_personal_message_creates_dm_event(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()

        activity = self._make_activity(conversation_type="personal")
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_type == "dm"

    @pytest.mark.anyio
    async def test_group_message_creates_group_event(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()

        activity = self._make_activity(conversation_type="groupChat")
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_type == "group"

    @pytest.mark.anyio
    async def test_channel_reply_uses_root_message_as_thread_id(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()

        activity = self._make_activity(
            conversation_type="channel",
            activity_id="reply-message-1",
        )
        activity.reply_to_id = "cron-root-message-1"
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.source.thread_id == "cron-root-message-1"
        assert event.reply_to_message_id == "cron-root-message-1"
        assert event.source.message_id == "reply-message-1"

    @pytest.mark.anyio
    async def test_channel_message_without_reply_id_does_not_use_cron_context(self, monkeypatch):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()
        cron_context = MagicMock()
        monkeypatch.setattr(
            TeamsAdapter,
            "_cron_reply_context",
            staticmethod(cron_context),
        )

        activity = self._make_activity(
            conversation_type="channel",
            activity_id="user-reply-1",
        )
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.source.thread_id is None
        assert event.reply_to_message_id is None
        assert event.reply_to_text is None
        cron_context.assert_not_called()

    @pytest.mark.anyio
    async def test_channel_message_uses_messageid_conversation_as_thread_id(self, monkeypatch):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()
        monkeypatch.setattr(
            TeamsAdapter,
            "_cron_reply_context",
            staticmethod(lambda _conversation_id, _thread_id: {
                "thread_id": _thread_id,
                "content": "New-thread cron context",
            }),
        )

        activity = self._make_activity(
            conversation_id="19:channel@thread.tacv2;messageid=1780267076971",
            conversation_type="channel",
            activity_id="user-reply-2",
        )
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.source.thread_id == "1780267076971"
        assert event.reply_to_text == "New-thread cron context"

    @pytest.mark.anyio
    async def test_cached_cron_context_skips_graph_fallback(self, monkeypatch):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id",
            client_secret="secret",
            tenant_id="tenant",
            fetch_reply_context=True,
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()
        adapter._fetch_parent_message_text = AsyncMock(return_value="Graph body")
        monkeypatch.setattr(
            TeamsAdapter,
            "_cron_reply_context",
            staticmethod(lambda _conversation_id, _thread_id: {
                "thread_id": "cron-root-message-1",
                "content": "Cached cron body",
            }),
        )

        activity = self._make_activity(
            conversation_type="channel",
            activity_id="reply-message-1",
        )
        activity.reply_to_id = "cron-root-message-1"
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.reply_to_text == "Cached cron body"
        adapter._fetch_parent_message_text.assert_not_awaited()

    @pytest.mark.anyio
    async def test_channel_reply_fetches_uncached_parent_from_graph(self, monkeypatch):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id",
            client_secret="secret",
            tenant_id="tenant",
            fetch_reply_context=True,
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()
        graph_client = SimpleNamespace(get_json=AsyncMock(return_value={
            "id": "root-message-1",
            "body": {
                "contentType": "html",
                "content": (
                    "<p><strong>TaskOps Morning Pack — 2/2 — Decisions Needed</strong></p>"
                    "<ul><li><strong>Approve</strong> item one.</li></ul>"
                    "<script>discard me</script>"
                ),
            },
        }))
        adapter._reply_context_graph_client = graph_client
        monkeypatch.setattr(
            TeamsAdapter,
            "_cron_reply_context",
            staticmethod(lambda _conversation_id, _thread_id: None),
        )

        activity = self._make_activity(
            conversation_id="19:channel@thread.tacv2;messageid=root-message-1",
            conversation_type="channel",
            activity_id="reply-message-1",
        )
        activity.reply_to_id = "root-message-1"
        activity.channel_data = {
            "team": {
                "id": "19:bot-framework-team@thread.tacv2",
                "aadGroupId": "team-1",
            },
            "channel": {"id": "19:channel@thread.tacv2"},
        }
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.reply_to_text == (
            "TaskOps Morning Pack — 2/2 — Decisions Needed\n- Approve item one."
        )
        graph_client.get_json.assert_awaited_once_with(
            "/teams/team-1/channels/19%3Achannel%40thread.tacv2/messages/root-message-1"
        )

    @pytest.mark.anyio
    async def test_channel_reply_continues_when_graph_fallback_fails(self, monkeypatch):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id",
            client_secret="secret",
            tenant_id="tenant",
            fetch_reply_context=True,
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()
        adapter._reply_context_graph_client = SimpleNamespace(
            get_json=AsyncMock(side_effect=RuntimeError("Graph unavailable")),
        )
        monkeypatch.setattr(
            TeamsAdapter,
            "_cron_reply_context",
            staticmethod(lambda _conversation_id, _thread_id: None),
        )

        activity = self._make_activity(
            conversation_id="19:channel@thread.tacv2",
            conversation_type="channel",
            activity_id="reply-message-1",
        )
        activity.reply_to_id = "root-message-1"
        activity.channel_data = {
            "team": {"id": "team-1"},
            "channel": {"id": "19:channel@thread.tacv2"},
        }
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.reply_to_message_id == "root-message-1"
        assert event.reply_to_text is None
