"""Tests for the Microsoft Teams platform adapter plugin."""

import sys
import types
from enum import Enum
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.teams_pipeline.models import TeamsMeetingRef, TeamsMeetingSummaryPayload
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


# ---------------------------------------------------------------------------
# SDK Mock — install in sys.modules before importing the adapter
# ---------------------------------------------------------------------------

def _ensure_teams_mock():
    """Install a teams SDK mock in sys.modules if the real package isn't present."""
    if "microsoft_teams" in sys.modules and hasattr(sys.modules["microsoft_teams"], "__file__"):
        return

    # Build the module hierarchy
    microsoft_teams = types.ModuleType("microsoft_teams")
    microsoft_teams_apps = types.ModuleType("microsoft_teams.apps")
    microsoft_teams_api = types.ModuleType("microsoft_teams.api")
    microsoft_teams_api_activities = types.ModuleType("microsoft_teams.api.activities")
    microsoft_teams_api_activities_typing = types.ModuleType("microsoft_teams.api.activities.typing")
    microsoft_teams_api_activities_invoke = types.ModuleType("microsoft_teams.api.activities.invoke")
    microsoft_teams_api_activities_invoke_adaptive_card = types.ModuleType(
        "microsoft_teams.api.activities.invoke.adaptive_card"
    )
    microsoft_teams_common = types.ModuleType("microsoft_teams.common")
    microsoft_teams_common_http = types.ModuleType("microsoft_teams.common.http")
    microsoft_teams_common_http_client = types.ModuleType("microsoft_teams.common.http.client")
    microsoft_teams_api_models = types.ModuleType("microsoft_teams.api.models")
    microsoft_teams_api_models_adaptive_card = types.ModuleType("microsoft_teams.api.models.adaptive_card")
    microsoft_teams_api_models_invoke_response = types.ModuleType("microsoft_teams.api.models.invoke_response")
    microsoft_teams_cards = types.ModuleType("microsoft_teams.cards")
    microsoft_teams_apps_http = types.ModuleType("microsoft_teams.apps.http")
    microsoft_teams_apps_http_adapter = types.ModuleType("microsoft_teams.apps.http.adapter")

    # App class mock
    class MockApp:
        def __init__(self, **kwargs):
            self._client_id = kwargs.get("client_id")
            self.server = MagicMock()
            self.server.handle_request = AsyncMock(return_value={"status": 200, "body": None})
            self.credentials = MagicMock()
            self.credentials.client_id = self._client_id

        @property
        def id(self):
            return self._client_id

        def on_message(self, func):
            self._message_handler = func
            return func

        def on_card_action(self, func):
            self._card_action_handler = func
            return func

        def on_message_reaction(self, func):
            self._message_reaction_handler = func
            return func

        def on_file_consent(self, func):
            self._file_consent_handler = func
            return func

        async def initialize(self):
            pass

        async def send(self, conversation_id, activity):
            result = MagicMock()
            result.id = "sent-activity-id"
            return result

        async def start(self, port=3978):
            pass

        async def stop(self):
            pass

    microsoft_teams_apps.App = MockApp
    microsoft_teams_apps.ActivityContext = MagicMock
    microsoft_teams_common_http_client.ClientOptions = MagicMock

    # MessageActivity mock
    microsoft_teams_api.MessageActivity = MagicMock
    microsoft_teams_api.ConversationReference = MagicMock
    microsoft_teams_api.MessageActivityInput = MagicMock
    microsoft_teams_api.Attachment = MagicMock

    # TypingActivityInput mock
    class MockTypingActivityInput:
        pass

    microsoft_teams_api_activities_typing.TypingActivityInput = MockTypingActivityInput

    # Adaptive card invoke activity mock
    microsoft_teams_api_activities_invoke_adaptive_card.AdaptiveCardInvokeActivity = MagicMock

    # Adaptive card response mocks
    microsoft_teams_api_models_adaptive_card.AdaptiveCardActionCardResponse = MagicMock
    microsoft_teams_api_models_adaptive_card.AdaptiveCardActionMessageResponse = MagicMock

    # Invoke response mocks
    class MockInvokeResponse:
        def __init__(self, status=200, body=None):
            self.status = status
            self.body = body

    microsoft_teams_api_models_invoke_response.InvokeResponse = MockInvokeResponse
    microsoft_teams_api_models_invoke_response.AdaptiveCardInvokeResponse = MagicMock

    # Cards mocks
    class MockAdaptiveCard:
        def with_version(self, v):
            return self

        def with_body(self, body):
            return self

        def with_actions(self, actions):
            return self

    microsoft_teams_cards.AdaptiveCard = MockAdaptiveCard
    microsoft_teams_cards.ExecuteAction = MagicMock
    microsoft_teams_cards.TextBlock = MagicMock

    # HttpRequest TypedDict mock
    def HttpRequest(body=None, headers=None):
        return {"body": body, "headers": headers}

    # HttpResponse TypedDict mock
    HttpResponse = dict
    HttpMethod = str
    from typing import Callable
    HttpRouteHandler = Callable

    microsoft_teams_apps_http_adapter.HttpRequest = HttpRequest
    microsoft_teams_apps_http_adapter.HttpResponse = HttpResponse
    microsoft_teams_apps_http_adapter.HttpMethod = HttpMethod
    microsoft_teams_apps_http_adapter.HttpRouteHandler = HttpRouteHandler

    # Wire the hierarchy
    for name, mod in {
        "microsoft_teams": microsoft_teams,
        "microsoft_teams.apps": microsoft_teams_apps,
        "microsoft_teams.api": microsoft_teams_api,
        "microsoft_teams.api.activities": microsoft_teams_api_activities,
        "microsoft_teams.api.activities.typing": microsoft_teams_api_activities_typing,
        "microsoft_teams.api.activities.invoke": microsoft_teams_api_activities_invoke,
        "microsoft_teams.api.activities.invoke.adaptive_card": microsoft_teams_api_activities_invoke_adaptive_card,
        "microsoft_teams.common": microsoft_teams_common,
        "microsoft_teams.common.http": microsoft_teams_common_http,
        "microsoft_teams.common.http.client": microsoft_teams_common_http_client,
        "microsoft_teams.api.models": microsoft_teams_api_models,
        "microsoft_teams.api.models.adaptive_card": microsoft_teams_api_models_adaptive_card,
        "microsoft_teams.api.models.invoke_response": microsoft_teams_api_models_invoke_response,
        "microsoft_teams.cards": microsoft_teams_cards,
        "microsoft_teams.apps.http": microsoft_teams_apps_http,
        "microsoft_teams.apps.http.adapter": microsoft_teams_apps_http_adapter,
    }.items():
        sys.modules.setdefault(name, mod)


_ensure_teams_mock()

# Load plugins/platforms/teams/adapter.py under a unique module name
# (plugin_adapter_teams) so it cannot collide with sibling plugin adapters.
_teams_mod = load_plugin_adapter("teams")

_teams_mod.AIOHTTP_AVAILABLE = True
# SDK import is deferred (#62935); bind mocked symbols the same way connect()
# does, but skip the real lazy-installer so collection does not pip-install
# microsoft-teams-apps.


def _bind_mock_sdk(feature, importer, target_globals, **kwargs):
    target_globals.update(importer())
    return True


with patch("tools.lazy_deps.ensure_and_bind", _bind_mock_sdk):
    assert _teams_mod.check_teams_requirements() is True
_teams_mod.TEAMS_SDK_AVAILABLE = True

# Ensure SDK symbols that were None (import failed on Python <3.12) are
# replaced with the mocked versions so runtime calls don't silently no-op.
import sys as _sys
_mt = _sys.modules.get("microsoft_teams.api.activities.typing")
if _mt and _teams_mod.TypingActivityInput is None:
    _teams_mod.TypingActivityInput = _mt.TypingActivityInput

TeamsAdapter = _teams_mod.TeamsAdapter
from plugins.platforms.teams.summary_writer import TeamsSummaryWriter  # noqa: E402
check_requirements = _teams_mod.check_requirements
check_teams_requirements = _teams_mod.check_teams_requirements
validate_config = _teams_mod.validate_config
register = _teams_mod.register


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**extra):
    return PlatformConfig(enabled=True, extra=extra)


# ---------------------------------------------------------------------------
# Tests: Requirements
# ---------------------------------------------------------------------------

class TestTeamsRequirements:





    def test_validate_config_with_env(self, monkeypatch):
        monkeypatch.setenv("TEAMS_CLIENT_ID", "test-id")
        monkeypatch.setenv("TEAMS_CLIENT_SECRET", "test-secret")
        monkeypatch.setenv("TEAMS_TENANT_ID", "test-tenant")
        assert validate_config(_make_config()) is True

    def test_validate_config_from_extra(self, monkeypatch):
        monkeypatch.delenv("TEAMS_CLIENT_ID", raising=False)
        monkeypatch.delenv("TEAMS_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("TEAMS_TENANT_ID", raising=False)
        cfg = _make_config(client_id="id", client_secret="secret", tenant_id="tenant")
        assert validate_config(cfg) is True


# ---------------------------------------------------------------------------
# Tests: Adapter Init
# ---------------------------------------------------------------------------

class TestTeamsAdapterInit:
    def test_reads_config_from_extra(self):
        config = _make_config(
            client_id="cfg-id",
            client_secret="cfg-secret",
            tenant_id="cfg-tenant",
        )
        adapter = TeamsAdapter(config)
        assert adapter._client_id == "cfg-id"
        assert adapter._client_secret == "cfg-secret"
        assert adapter._tenant_id == "cfg-tenant"


    def test_custom_port_from_env(self, monkeypatch):
        monkeypatch.setenv("TEAMS_PORT", "5000")
        adapter = TeamsAdapter(_make_config(client_id="id", client_secret="secret", tenant_id="tenant"))
        assert adapter._port == 5000

    def test_invalid_port_from_extra_falls_back_to_default(self):
        adapter = TeamsAdapter(
            _make_config(client_id="id", client_secret="secret", tenant_id="tenant", port="abc")
        )
        assert adapter._port == 3978


# ---------------------------------------------------------------------------
# Tests: Plugin registration
# ---------------------------------------------------------------------------

class TestTeamsPluginRegistration:



    def test_register_splits_passive_probe_from_active_installer(self):
        # check_fn is the PASSIVE probe (status displays call it freely);
        # the ACTIVE lazy-installer rides on ensure_deps_fn, which
        # create_adapter() invokes when the passive probe fails (#79812).
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["check_fn"] is check_requirements
        assert kwargs["ensure_deps_fn"] is check_teams_requirements

    def test_register_auth_env_vars(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["allowed_users_env"] == "TEAMS_ALLOWED_USERS"
        assert kwargs["allow_all_env"] == "TEAMS_ALLOW_ALL_USERS"


# ---------------------------------------------------------------------------
# Tests: Interactive setup (import fix regression — #18325 / #19173)
# ---------------------------------------------------------------------------

class TestTeamsInteractiveSetup:
    def test_interactive_setup_persists_credentials(self, tmp_path, monkeypatch):
        """Regression for #19173: interactive_setup must import prompt helpers
        from hermes_cli.cli_output (not hermes_cli.config) and persist
        credentials to .env without crashing.
        """
        hermes_home = tmp_path / "hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        import hermes_cli.cli_output as cli_output_mod

        answers = iter(["client-id", "client-secret", "tenant-id", "aad-1, aad-2"])
        monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(answers))
        monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: True)
        monkeypatch.setattr(cli_output_mod, "print_info", lambda *_a, **_kw: None)
        monkeypatch.setattr(cli_output_mod, "print_success", lambda *_a, **_kw: None)
        monkeypatch.setattr(cli_output_mod, "print_warning", lambda *_a, **_kw: None)

        _teams_mod.interactive_setup()

        env_text = (hermes_home / ".env").read_text(encoding="utf-8")
        assert "TEAMS_CLIENT_ID=client-id" in env_text
        assert "TEAMS_TENANT_ID=tenant-id" in env_text

class TestTeamsConnect:
    @pytest.mark.anyio
    async def test_connect_fails_without_sdk(self, monkeypatch):
        monkeypatch.setattr(_teams_mod, "TEAMS_SDK_AVAILABLE", False)
        monkeypatch.setattr(_teams_mod, "App", None)
        monkeypatch.setattr(_teams_mod, "ClientOptions", None)
        # Simulate the SDK being unavailable AND not installable (offline /
        # locked-down env): the lazy-installer can't rebind the globals, so
        # App stays None and connect() must fail without calling it.
        monkeypatch.setattr(
            "tools.lazy_deps.ensure_and_bind",
            lambda *_a, **_k: False,
        )
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        result = await adapter.connect()
        assert result is False

    @pytest.mark.anyio
    async def test_connect_fails_when_namespace_exists_but_app_unbound(self, monkeypatch):
        """find_spec('microsoft_teams') can be true from sibling packages
        without microsoft-teams-apps. connect() must not call App() while
        it is still None — that was ``'NoneType' object is not callable``.
        """
        monkeypatch.setattr(_teams_mod, "TEAMS_SDK_AVAILABLE", True)
        monkeypatch.setattr(_teams_mod, "App", None)
        monkeypatch.setattr(_teams_mod, "ClientOptions", None)
        monkeypatch.setattr(_teams_mod, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr(
            "tools.lazy_deps.ensure_and_bind",
            lambda *_a, **_k: False,
        )
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        result = await adapter.connect()
        assert result is False
        assert adapter._app is None


# ---------------------------------------------------------------------------
# Tests: Send
# ---------------------------------------------------------------------------



def _make_summary_payload():
    return TeamsMeetingSummaryPayload(
        meeting_ref=TeamsMeetingRef(meeting_id="meeting-123"),
        title="Weekly Sync",
        summary="Discussed launch readiness.",
        key_decisions=["Proceed with staged rollout."],
        action_items=["Send launch checklist."],
        risks=["QA sign-off still pending."],
    )


class TestTeamsSummaryWriter:

    @pytest.mark.anyio
    async def test_graph_delivery_posts_to_channel(self):
        graph_client = SimpleNamespace(
            post_json=AsyncMock(return_value={"id": "msg-123", "webUrl": "https://teams.example/messages/123"})
        )
        writer = TeamsSummaryWriter(graph_client=graph_client)
        payload = _make_summary_payload()

        result = await writer.write_summary(
            payload,
            {
                "delivery_mode": "graph",
                "team_id": "team-1",
                "channel_id": "channel-1",
            },
        )

        assert result["target_type"] == "channel"
        assert result["message_id"] == "msg-123"
        graph_client.post_json.assert_awaited_once()
        path = graph_client.post_json.await_args.args[0]
        body = graph_client.post_json.await_args.kwargs["json_body"]
        assert path == "/teams/team-1/channels/channel-1/messages"
        assert body["body"]["contentType"] == "html"
        assert "Weekly Sync" in body["body"]["content"]


# ---------------------------------------------------------------------------
# Tests: Message Handling
# ---------------------------------------------------------------------------

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
    async def test_channel_message_stashes_graph_file_target(self):
        from plugins.platforms.teams.graph_files import GraphFileTarget

        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()

        activity = self._make_activity(
            conversation_id="19:chan@thread.tacv2", conversation_type="channel")
        activity.channel_data = {
            "team": {"aadGroupId": "team-guid", "id": "19:team@thread.skype"},
            "channel": {"id": "19:chan@thread.tacv2"},
        }
        await adapter._on_message(self._make_ctx(activity))

        assert adapter._graph_file_targets["19:chan@thread.tacv2"] == GraphFileTarget(
            conversation_type="channel",
            team_id="team-guid",
            channel_id="19:chan@thread.tacv2",
        )

    @pytest.mark.anyio
    async def test_aad_user_route_survives_conversation_changes(self, monkeypatch):
        from gateway.profile_routing import parse_profile_routes
        from gateway.run import GatewayRunner

        routes = parse_profile_routes([
            {"name": "owner", "platform": "teams", "user_id": "aad-456", "profile": "owner"},
            {"name": "other", "platform": "teams", "user_id": "aad-789", "profile": "other"},
        ])
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = SimpleNamespace(multiplex_profiles=True, profile_routes=routes)
        monkeypatch.setattr(
            "gateway.run._multiplex_profile_homes",
            lambda _config: [("owner", None), ("other", None)],
        )

        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter.gateway_runner = runner
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()

        for activity_id, conversation_id, conversation_type, user_id in (
            ("activity-group", "19:shared@thread.v2", "groupChat", "aad-456"),
            ("activity-channel", "19:channel@thread.v2", "channel", "aad-456"),
            ("activity-dm", "19:dm@thread.v2", "personal", "aad-456"),
            ("activity-other", "19:shared@thread.v2", "groupChat", "aad-789"),
        ):
            await adapter._on_message(self._make_ctx(self._make_activity(
                activity_id=activity_id,
                conversation_id=conversation_id,
                conversation_type=conversation_type,
                from_aad_id=user_id,
            )))

        sources = [call.args[0].source for call in adapter.handle_message.await_args_list]
        assert [source.profile for source in sources] == ["owner", "owner", "owner", "other"]
        assert [source.chat_type for source in sources] == ["group", "channel", "dm", "group"]
        assert [runner._session_key_for_source(source).split(":", 2)[1] for source in sources] == [
            "owner", "owner", "owner", "other",
        ]


class TestTeamsAttachmentClassification:
    """Document attachments must set MessageType.DOCUMENT so run.py's
    document-context injection surfaces the cached file to the agent
    (same bug class as Signal/Email/SimpleX, PR #44695)."""

    def _make_adapter(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()
        adapter._graph_client = False
        return adapter

    def _make_activity(self, attachments, text="see attached"):
        activity = MagicMock()
        activity.text = text
        activity.id = "activity-att-001"
        activity.from_ = MagicMock()
        activity.from_.id = "user-123"
        activity.from_.aad_object_id = "aad-456"
        activity.from_.name = "Test User"
        activity.conversation = MagicMock()
        activity.conversation.id = "19:abc@thread.v2"
        activity.conversation.conversation_type = "personal"
        activity.conversation.name = "Test Chat"
        activity.conversation.tenant_id = "tenant-789"
        activity.attachments = attachments
        return activity

    def _make_ctx(self, activity):
        ctx = MagicMock()
        ctx.activity = activity
        return ctx

    def _file_download_attachment(self, name="report.pdf", file_type="pdf"):
        att = MagicMock()
        att.content_type = "application/vnd.microsoft.teams.file.download.info"
        att.content_url = None
        att.name = name
        att.content = {
            "downloadUrl": "https://contoso.sharepoint.com/download/x",
            "fileType": file_type,
        }
        return att

    def _image_attachment(self):
        att = MagicMock()
        att.content_type = "image/png"
        att.content_url = "https://smba.example.com/img.png"
        att.name = "img.png"
        return att

    def _html_body_attachment(self):
        # Teams mirrors the message body as a text/html attachment
        att = MagicMock()
        att.content_type = "text/html"
        att.content_url = None
        att.name = ""
        return att

    @pytest.mark.anyio
    async def test_file_download_info_sets_document_type(self):
        from gateway.platforms.event import MessageType

        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"%PDF-1.4 fake")

        activity = self._make_activity([self._file_download_attachment()])
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT, (
            f"Expected DOCUMENT, got {event.message_type}. "
            "Documents must be classified as DOCUMENT so run.py injects file context."
        )
        assert len(event.media_urls) == 1
        assert event.media_types == ["application/pdf"]

    @pytest.mark.anyio
    async def test_mixed_image_and_document_prefers_document(self):
        from gateway.platforms.event import MessageType

        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"%PDF-1.4 fake")

        async def fake_cache_image(url, *a, **kw):
            return "/tmp/img.png"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_teams_mod, "cache_image_from_url", fake_cache_image)
            activity = self._make_activity([
                self._image_attachment(),
                self._file_download_attachment(),
            ])
            await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT
        assert len(event.media_urls) == 2

    @pytest.mark.anyio
    async def test_direct_url_pdf_sets_document_type(self):
        from gateway.platforms.event import MessageType

        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"%PDF-1.4 fake")
        att = MagicMock()
        att.content_type = "application/pdf"
        att.content_url = "https://contoso.sharepoint.com/file.pdf"
        att.name = "file.pdf"
        activity = self._make_activity([att])
        await adapter._on_message(self._make_ctx(activity))
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT
        assert event.media_types == ["application/pdf"]
        adapter._fetch_attachment_bytes.assert_awaited_once_with(
            "https://contoso.sharepoint.com/file.pdf")

    @pytest.mark.anyio
    async def test_camelcase_dict_file_download_info_sets_document_type(self):
        """Bot Framework JSON uses contentType/contentUrl; dict attachments must work."""
        from gateway.platforms.event import MessageType

        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"%PDF-1.4 fake")
        att = {
            "contentType": "application/vnd.microsoft.teams.file.download.info",
            "contentUrl": None,
            "name": "report.pdf",
            "content": {
                "downloadUrl": "https://contoso.sharepoint.com/download/x",
                "fileType": "pdf",
            },
        }
        await adapter._on_message(self._make_ctx(self._make_activity([att])))
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT
        assert len(event.media_urls) == 1
        adapter._fetch_attachment_bytes.assert_awaited_once_with(
            "https://contoso.sharepoint.com/download/x")

    @pytest.mark.anyio
    async def test_file_download_info_missing_download_url_warns(self, caplog):
        import logging
        from gateway.platforms.event import MessageType

        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(
            side_effect=AssertionError("must not fetch without a URL"))
        att = {
            "contentType": "application/vnd.microsoft.teams.file.download.info",
            "name": "notes.txt",
            "content": {
                "uniqueId": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "fileType": "txt",
            },
        }
        with caplog.at_level(logging.WARNING):
            await adapter._on_message(self._make_ctx(
                self._make_activity([att], text="are you able to read this")))
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.TEXT
        assert event.media_urls == []
        assert event.text == "are you able to read this"
        assert any("downloadUrl" in rec.getMessage() for rec in caplog.records)
        adapter._fetch_attachment_bytes.assert_not_awaited()

    @pytest.mark.anyio
    async def test_named_text_plain_is_not_treated_as_body_mirror(self):
        from gateway.platforms.event import MessageType

        adapter = self._make_adapter()
        att = MagicMock()
        att.content_type = "text/plain"
        att.content_url = None
        att.name = "notes.txt"
        att.content = "hello from the file"
        await adapter._on_message(self._make_ctx(
            self._make_activity([att], text="are you able to read this")))
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT
        assert len(event.media_urls) == 1

    @pytest.mark.anyio
    async def test_anonymous_html_body_mirror_is_skipped(self):
        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(
            side_effect=AssertionError("must not fetch body mirror"))
        await adapter._on_message(self._make_ctx(
            self._make_activity([self._html_body_attachment()])))
        event = adapter.handle_message.call_args[0][0]
        assert event.media_urls == []
        adapter._fetch_attachment_bytes.assert_not_awaited()

    @pytest.mark.anyio
    async def test_missing_download_url_uses_graph_unique_id(self):
        from gateway.platforms.event import MessageType
        from plugins.platforms.teams.graph_files import GraphFileTarget

        download = "https://contoso.sharepoint.com/download/notes"
        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"hello file")

        async def get_json(path, **kwargs):
            if "filesFolder" in path:
                return {"id": "folder-1", "parentReference": {"driveId": "drive-1"}}
            return {
                "id": "item-1",
                "name": "notes.txt",
                "@microsoft.graph.downloadUrl": download,
            }

        graph = MagicMock()
        graph.get_json = get_json
        adapter._graph_client = graph
        conv_id = "19:abc@thread.v2"
        adapter._graph_file_targets[conv_id] = GraphFileTarget(
            conversation_type="channel", team_id="team-guid", channel_id=conv_id,
        )
        att = {
            "contentType": "application/vnd.microsoft.teams.file.download.info",
            "name": "notes.txt",
            "content": {
                "uniqueId": "{aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee}",
                "fileType": "txt",
            },
        }
        activity = self._make_activity([att], text="are you able to read this")
        activity.conversation.conversation_type = "channel"
        await adapter._on_message(self._make_ctx(activity))
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT
        assert len(event.media_urls) == 1
        adapter._fetch_attachment_bytes.assert_awaited_once_with(download)

    @pytest.mark.anyio
    async def test_channel_html_only_fetches_file_via_graph_message(self):
        """Channel file drops often arrive as unnamed text/html only — Graph GET message."""
        from gateway.platforms.event import MessageType
        from plugins.platforms.teams.graph_files import GraphFileTarget

        download = "https://contoso.sharepoint.com/download/notes"
        share_url = "https://contoso.sharepoint.com/sites/team/Shared%20Documents/notes.txt"
        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"hello from sharepoint")
        conv_id = "19:abc@thread.v2"
        adapter._graph_file_targets[conv_id] = GraphFileTarget(
            conversation_type="channel", team_id="team-guid", channel_id=conv_id,
        )

        async def get_json(path, **kwargs):
            if "/messages/" in path:
                return {
                    "id": "activity-att-001",
                    "attachments": [
                        {
                            "id": "att-1",
                            "contentType": "reference",
                            "contentUrl": share_url,
                            "name": "notes.txt",
                        },
                        {"contentType": "text/html", "content": "<p>caption</p>"},
                    ],
                }
            if "/shares/" in path:
                return {"@microsoft.graph.downloadUrl": download}
            return {}

        graph = MagicMock()
        graph.get_json = get_json
        adapter._graph_client = graph
        html = self._html_body_attachment()
        html.content = "<p>are you able to read this</p>"
        activity = self._make_activity([html], text="are you able to read this")
        activity.conversation.conversation_type = "channel"
        await adapter._on_message(self._make_ctx(activity))
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT
        assert len(event.media_urls) == 1
        assert event.text == "are you able to read this"
        adapter._fetch_attachment_bytes.assert_awaited_once_with(download)

    @pytest.mark.anyio
    async def test_channel_html_only_graph_403_keeps_text_and_warns(self, caplog):
        import logging
        from plugins.platforms.teams.graph_files import GraphFileTarget
        from tools.microsoft_graph_client import MicrosoftGraphAPIError

        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(
            side_effect=AssertionError("must not fetch after Graph 403"))
        conv_id = "19:abc@thread.v2"
        adapter._graph_file_targets[conv_id] = GraphFileTarget(
            conversation_type="channel", team_id="team-guid", channel_id=conv_id,
        )

        async def get_json(path, **kwargs):
            raise MicrosoftGraphAPIError(403, "GET", path, "Access denied")

        graph = MagicMock()
        graph.get_json = get_json
        adapter._graph_client = graph
        activity = self._make_activity(
            [self._html_body_attachment()], text="are you able to read this")
        activity.conversation.conversation_type = "channel"
        with caplog.at_level(logging.WARNING):
            await adapter._on_message(self._make_ctx(activity))
        event = adapter.handle_message.call_args[0][0]
        assert event.media_urls == []
        assert event.text == "are you able to read this"
        assert any("ChannelMessage.Read" in rec.getMessage() for rec in caplog.records)
        adapter._fetch_attachment_bytes.assert_not_awaited()


# ── Bot Framework connector attachments (pasted images) ──────────────────


class TestTeamsBotFrameworkAttachments:
    """Pasted/inline images arrive on smba.trafficmanager.net hosts and need
    the bot's own bearer token (unlike SharePoint downloadUrls). These tests
    pin the auth routing, the token cache, the attacker-host block, and the
    failure fallbacks of that path."""

    def _make_adapter(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()
        return adapter

    def _make_activity(self, attachments):
        activity = MagicMock()
        activity.text = "see attached"
        activity.id = "activity-att-001"
        activity.from_ = MagicMock()
        activity.from_.id = "user-123"
        activity.from_.aad_object_id = "aad-456"
        activity.from_.name = "Test User"
        activity.conversation = MagicMock()
        activity.conversation.id = "19:abc@thread.v2"
        activity.conversation.conversation_type = "personal"
        activity.conversation.name = "Test Chat"
        activity.conversation.tenant_id = "tenant-789"
        activity.attachments = attachments
        return activity

    def _make_ctx(self, activity):
        ctx = MagicMock()
        ctx.activity = activity
        return ctx

    def _bf_image_attachment(self, url=None):
        att = MagicMock()
        att.content_type = "image/png"
        att.content_url = url or "https://smba.trafficmanager.net/emea/b1/v3/attachments/0-abc/views/original"
        att.name = "pasted.png"
        return att

    @pytest.mark.anyio
    async def test_bf_url_predicate_exact_match_allowlist(self):
        """Only exact allowlisted hosts on https default port may receive the
        bot's bearer token — lookalikes, other schemes, and non-443 ports must
        NOT (any Azure customer can register <name>.trafficmanager.net)."""
        f = _teams_mod._is_botframework_attachment_url
        assert f("https://smba.trafficmanager.net/emea/v3/attachments/x")
        assert f("https://smba.infra.gov.teams.microsoft.us/amer/v3/attachments/x")
        assert f("https://smba.trafficmanager.net:443/emea/v3/attachments/x")
        # Attacker lookalikes / non-allowlisted / wrong scheme / wrong port
        assert not f("https://evil-trafficmanager.net/steal")
        assert not f("https://emea.smba.trafficmanager.net/v3/attachments/x")
        assert not f("https://notbotframework.com/steal")
        assert not f("https://trafficmanager.net.evil.com/steal")
        assert not f("http://smba.trafficmanager.net/v3/attachments/x")
        assert not f("https://smba.trafficmanager.net:444/v3/attachments/x")
        assert not f("")
        assert not f("https://sharepoint.com/x")

    @pytest.mark.anyio
    async def test_bf_image_routes_through_authenticated_fetch(self):
        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"\x89PNG fake")
        adapter._get_botframework_token = AsyncMock(return_value="tok")

        async def fake_cache_media_bytes(data, **kwargs):
            return SimpleNamespace(
                path="/tmp/img.png", media_type="image/png", kind="image"
            )

        with patch.object(_teams_mod, "cache_media_bytes_async", fake_cache_media_bytes):
            activity = self._make_activity([self._bf_image_attachment()])
            await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert len(event.media_urls) == 1
        assert event.media_types == ["image/png"]
        # URL was fetched with auth (via _fetch_attachment_bytes, which the
        # token routing test below exercises end-to-end)
        adapter._fetch_attachment_bytes.assert_awaited_once_with(
            "https://smba.trafficmanager.net/emea/b1/v3/attachments/0-abc/views/original"
        )

    @pytest.mark.anyio
    async def test_non_bf_image_uses_generic_cache_helper(self):
        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(side_effect=AssertionError("must not be called"))

        async def fake_cache_image(url, *a, **kw):
            return "/tmp/img.jpg"

        with patch.object(_teams_mod, "cache_image_from_url", fake_cache_image):
            activity = self._make_activity(
                [self._bf_image_attachment(url="https://contoso.sharepoint.com/img.png")]
            )
            await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert len(event.media_urls) == 1
        assert event.media_urls[0] == "/tmp/img.jpg"

    @pytest.mark.anyio
    async def test_fetch_attachment_bytes_sends_bearer_for_bf_host(self):
        """End-to-end over _fetch_attachment_bytes: BF host → token acquired
        and Authorization attached; non-BF host → no token call."""
        adapter = self._make_adapter()
        adapter._get_botframework_token = AsyncMock(return_value="the-token")

        captured = {}

        class _FakeStreamResponse:
            def __init__(self):
                self.headers = {}

            def raise_for_status(self):
                pass

            async def aiter_bytes(self):
                yield b"\x89PNG fake"

        class _FakeStreamCtx:
            def __init__(self, response):
                self._response = response

            async def __aenter__(self):
                return self._response

            async def __aexit__(self, *a):
                return None

        class _FakeClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            def stream(self, method, url, headers=None):
                captured["headers"] = headers or {}
                return _FakeStreamCtx(_FakeStreamResponse())

        with patch("tools.url_safety.create_ssrf_safe_async_client", lambda **kw: _FakeClient()), \
             patch("tools.url_safety.is_safe_url", lambda url: True):
            # BF host: bearer attached
            data = await adapter._fetch_attachment_bytes("https://smba.trafficmanager.net/emea/v3/attachments/x")
        assert captured["headers"].get("Authorization") == "Bearer the-token"
        assert data == b"\x89PNG fake"

        adapter._get_botframework_token = AsyncMock(return_value="the-token")
        with patch("tools.url_safety.create_ssrf_safe_async_client", lambda **kw: _FakeClient()), \
             patch("tools.url_safety.is_safe_url", lambda url: True):
            # Attacker lookalike host: NO bearer (exact-match allowlist)
            await adapter._fetch_attachment_bytes("https://evil-trafficmanager.net/steal")
        assert "Authorization" not in captured["headers"], (
            "bearer token must not be sent to attacker lookalike hosts"
        )
        adapter._get_botframework_token.assert_not_awaited()

    @pytest.mark.anyio
    async def test_token_refresh_is_serialized_under_lock(self):
        """Two concurrent token fetches on a cold cache share ONE POST —
        the lock prevents a token-endpoint stampede."""
        import asyncio as _asyncio

        adapter = self._make_adapter()
        posts = []
        release = _asyncio.Event()

        class _TokenResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"access_token": "tok-1", "expires_in": 3600}

        class _SlowTokenClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def post(self, url, data=None):
                posts.append((url, dict(data or {})))
                await release.wait()  # hold both callers at the STS door
                return _TokenResp()

        async def release_later():
            await _asyncio.sleep(0.05)
            release.set()

        with patch("httpx.AsyncClient", _SlowTokenClient):
            t1 = _asyncio.create_task(adapter._get_botframework_token())
            t2 = _asyncio.create_task(adapter._get_botframework_token())
            await release_later()
            tok1, tok2 = await t1, await t2
        assert tok1 == "tok-1" and tok2 == "tok-1"
        assert len(posts) == 1, f"concurrent cold-cache fetches must share one POST, got {len(posts)}"

    @pytest.mark.anyio
    async def test_token_acquisition_and_cache_reuse(self):
        adapter = self._make_adapter()

        posts = []

        class _TokenResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"access_token": "tok-1", "expires_in": 3600}

        class _TokenClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def post(self, url, data=None):
                posts.append((url, dict(data or {})))
                return _TokenResp()

        with patch("httpx.AsyncClient", _TokenClient):
            tok1 = await adapter._get_botframework_token()
            tok2 = await adapter._get_botframework_token()
        assert tok1 == "tok-1" and tok2 == "tok-1"
        assert len(posts) == 1, "second call must hit the cache"
        assert posts[0][0] == "https://login.microsoftonline.com/tenant/oauth2/v2.0/token"
        assert posts[0][1]["scope"] == "https://api.botframework.com/.default"
        assert posts[0][1]["client_id"] == "bot-id"
        assert posts[0][1]["client_secret"] == "secret"

    @pytest.mark.anyio
    async def test_token_acquisition_failure_degrades_to_unauthenticated_fetch(self):
        """Token failure must not break the fetch: warning + fetch without
        Authorization (same net behavior as the pre-fix path)."""
        import httpx as _httpx

        adapter = self._make_adapter()
        adapter._get_botframework_token = AsyncMock(side_effect=ValueError("no creds"))

        captured = {}

        class _FakeStreamResponse:
            def __init__(self):
                self.headers = {}

            def raise_for_status(self):
                raise _httpx.HTTPStatusError(
                    "401", request=MagicMock(), response=MagicMock(status_code=401)
                )

            async def aiter_bytes(self):
                yield b""

        class _FakeStreamCtx:
            def __init__(self, response):
                self._response = response

            async def __aenter__(self):
                return self._response

            async def __aexit__(self, *a):
                return None

        class _FakeClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            def stream(self, method, url, headers=None):
                captured["headers"] = headers or {}
                return _FakeStreamCtx(_FakeStreamResponse())

        with patch("tools.url_safety.create_ssrf_safe_async_client", lambda **kw: _FakeClient()), \
             patch("tools.url_safety.is_safe_url", lambda url: True):
            with pytest.raises(_httpx.HTTPStatusError):
                await adapter._fetch_attachment_bytes("https://smba.trafficmanager.net/v3/attachments/x")
        assert "Authorization" not in captured["headers"]

    @pytest.mark.anyio
    async def test_bf_image_invalid_bytes_logs_warning(self):
        """Non-image bytes from the BF endpoint must not be silently dropped
        — the else branch warns (regression guard for the silent-drop)."""
        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"<html>error page</html>")

        async def _no_media(*a, **kw):
            return None

        with patch.object(_teams_mod, "cache_media_bytes_async", _no_media):
            with patch.object(_teams_mod.logger, "warning") as warn:
                activity = self._make_activity([self._bf_image_attachment()])
                await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.media_urls == []
        assert warn.called, "silent drop of invalid BF image bytes must log a warning"


# ── _standalone_send (out-of-process cron delivery) ──────────────────────


class _FakeAiohttpResponse:
    def __init__(self, status: int, payload, text_body: str = ""):
        self.status = status
        self._payload = payload
        self._text = text_body or (str(payload) if payload is not None else "")

    async def json(self):
        return self._payload

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeAiohttpSession:
    """Scripted aiohttp.ClientSession with a queue of responses so tests
    can assert calls in order."""

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self._scripts:
            raise AssertionError(f"No scripted response for POST {url}")
        return self._scripts.pop(0)


def _install_fake_aiohttp(monkeypatch, session):
    """Replace ``aiohttp`` in ``sys.modules`` so ``import aiohttp as _aiohttp``
    inside ``_standalone_send`` picks up our fake."""
    fake_aiohttp = types.SimpleNamespace(
        ClientSession=lambda timeout=None, **kwargs: session,
        ClientTimeout=lambda total=None: None,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)


class TestTeamsStandaloneSend:

    @pytest.mark.asyncio
    async def test_standalone_send_acquires_token_and_posts_activity(self, monkeypatch):
        monkeypatch.setenv("TEAMS_CLIENT_ID", "client-id")
        monkeypatch.setenv("TEAMS_CLIENT_SECRET", "secret")
        monkeypatch.setenv("TEAMS_TENANT_ID", "tenant")
        monkeypatch.delenv("TEAMS_SERVICE_URL", raising=False)

        token_resp = _FakeAiohttpResponse(200, {"access_token": "the-token"})
        activity_resp = _FakeAiohttpResponse(200, {"id": "msg-99"})
        session = _FakeAiohttpSession([token_resp, activity_resp])
        _install_fake_aiohttp(monkeypatch, session)

        result = await _teams_mod._standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "19:abc@thread.skype",
            "hello cron",
        )

        assert result == {"success": True, "message_id": "msg-99"}
        assert len(session.calls) == 2

        token_url, token_kwargs = session.calls[0]
        assert "login.microsoftonline.com/tenant/oauth2/v2.0/token" in token_url
        assert token_kwargs["data"]["client_id"] == "client-id"
        assert token_kwargs["data"]["client_secret"] == "secret"
        assert token_kwargs["data"]["scope"] == "https://api.botframework.com/.default"

        activity_url, activity_kwargs = session.calls[1]
        # Default service URL when TEAMS_SERVICE_URL is unset
        assert "smba.trafficmanager.net" in activity_url
        assert "/v3/conversations/19:abc@thread.skype/activities" in activity_url
        assert activity_kwargs["headers"]["Authorization"] == "Bearer the-token"
        assert activity_kwargs["json"]["text"] == "hello cron"
        assert activity_kwargs["json"]["type"] == "message"


    @pytest.mark.asyncio
    async def test_standalone_send_propagates_token_failure(self, monkeypatch):
        monkeypatch.setenv("TEAMS_CLIENT_ID", "client-id")
        monkeypatch.setenv("TEAMS_CLIENT_SECRET", "secret")
        monkeypatch.setenv("TEAMS_TENANT_ID", "tenant")

        token_resp = _FakeAiohttpResponse(
            401,
            {"error": "unauthorized_client"},
            text_body='{"error":"unauthorized_client"}',
        )
        session = _FakeAiohttpSession([token_resp])
        _install_fake_aiohttp(monkeypatch, session)

        result = await _teams_mod._standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "19:abc@thread.skype",
            "hi",
        )

        assert "error" in result
        assert "401" in result["error"]
        assert "token" in result["error"].lower()


class TestTeamsMediaAttachments:
    """send_document routes personal chats through FileConsent and channel/group
    chats through inline text or a Graph SharePoint upload."""

    def _make_adapter(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter._app.send = AsyncMock(return_value=MagicMock(id="msg-001"))
        adapter._app.activity_sender.send = AsyncMock(return_value=MagicMock(id="msg-001"))
        # Do not construct a real Graph token client in tests (would hit Azure).
        adapter._graph_client = False
        return adapter

    @pytest.mark.asyncio
    async def test_send_document_local_file_base64(self, tmp_path):
        adapter = self._make_adapter()
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")
        result = await adapter.send_document("19:abc@thread.v2", str(doc))
        assert result.success
        adapter._app.send.assert_awaited_once()
        # Personal/unknown conversation type uses file-consent, so bytes are staged.
        assert len(adapter._pending_uploads) == 1
        pending = next(iter(adapter._pending_uploads.values()))
        assert pending["name"] == "report.pdf"
        assert pending["bytes"].startswith(b"%PDF")

    @pytest.mark.asyncio
    async def test_send_document_channel_inlines_small_text(self, tmp_path):
        adapter = self._make_adapter()
        adapter._conv_refs["19:abc@thread.v2"] = SimpleNamespace(
            conversation=SimpleNamespace(conversation_type="channel"))
        doc = tmp_path / "chess_rules.txt"
        doc.write_text("1. e4 e5")
        result = await adapter.send_document(
            "19:abc@thread.v2", str(doc), file_name="chess_rules.txt")
        assert result.success
        assert adapter._pending_uploads == {}
        adapter._app.send.assert_awaited()
        sent = adapter._app.send.await_args.args[1]
        assert "chess_rules.txt" in sent
        assert "1. e4 e5" in sent
        adapter._app.activity_sender.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_document_group_inlines_small_text(self, tmp_path):
        adapter = self._make_adapter()
        adapter._conv_refs["19:abc@thread.v2"] = SimpleNamespace(
            conversation=SimpleNamespace(conversation_type="groupChat"))
        doc = tmp_path / "notes.md"
        doc.write_text("# hello")
        result = await adapter.send_document("19:abc@thread.v2", str(doc), file_name="notes.md")
        assert result.success
        sent = adapter._app.send.await_args.args[1]
        assert "notes.md" in sent
        assert "# hello" in sent

    @pytest.mark.asyncio
    async def test_send_document_channel_binary_returns_clear_error(self, tmp_path):
        adapter = self._make_adapter()
        adapter._conv_refs["19:abc@thread.v2"] = SimpleNamespace(
            conversation=SimpleNamespace(conversation_type="channel"))
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 binary")
        result = await adapter.send_document("19:abc@thread.v2", str(doc), file_name="report.pdf")
        assert not result.success
        assert "400" not in (result.error or "")
        assert "FileConsent" in result.error
        assert "DM" in result.error or "1:1" in result.error
        assert "Graph" in result.error or "MSGRAPH_" in result.error
        adapter._app.activity_sender.send.assert_not_awaited()
        adapter._app.send.assert_awaited()
        assert "report.pdf" in adapter._app.send.await_args.args[1]
        # Never revive Bot Framework document attachments in channels.
        sent = adapter._app.send.await_args.args[1]
        assert not hasattr(sent, "add_attachments")

    @pytest.mark.asyncio
    async def test_send_document_channel_oversize_text_is_not_inlined(self, tmp_path):
        adapter = self._make_adapter()
        adapter._conv_refs["19:abc@thread.v2"] = SimpleNamespace(
            conversation=SimpleNamespace(conversation_type="channel"))
        doc = tmp_path / "huge.txt"
        doc.write_text("x" * (_teams_mod._INLINE_CHANNEL_TEXT_MAX_BYTES + 1))
        result = await adapter.send_document("19:abc@thread.v2", str(doc), file_name="huge.txt")
        assert not result.success
        assert "FileConsent" in result.error

    @pytest.mark.asyncio
    async def test_send_document_channel_binary_uploads_via_graph(self, tmp_path):
        from plugins.platforms.teams.graph_files import GraphFileTarget, GraphUploadedFile

        adapter = self._make_adapter()
        adapter._conv_refs["19:chan@thread.tacv2"] = SimpleNamespace(
            conversation=SimpleNamespace(conversation_type="channel"))
        adapter._graph_file_targets["19:chan@thread.tacv2"] = GraphFileTarget(
            conversation_type="channel",
            team_id="team-guid",
            channel_id="19:chan@thread.tacv2",
        )
        adapter._graph_client = object()
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 binary")

        async def _upload(graph, target, *, file_name, data, content_type="application/octet-stream", **_kw):
            assert graph is adapter._graph_client
            assert target.team_id == "team-guid"
            assert file_name == "report.pdf"
            assert data.startswith(b"%PDF")
            return GraphUploadedFile(
                name="report.pdf",
                web_url="https://contoso.sharepoint.com/sites/team/report.pdf",
                share_url="https://contoso.sharepoint.com/:b:/s/team/abc",
            )

        with patch("plugins.platforms.teams.graph_files.upload_conversation_file", _upload):
            result = await adapter.send_document(
                "19:chan@thread.tacv2", str(doc), file_name="report.pdf", caption="Q3 report")

        assert result.success
        adapter._app.send.assert_awaited()
        sent = adapter._app.send.await_args.args[1]
        assert "Q3 report" in sent
        assert "https://contoso.sharepoint.com/:b:/s/team/abc" in sent
        assert "report.pdf" in sent
        adapter._app.activity_sender.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_document_group_binary_uploads_via_graph(self, tmp_path):
        from plugins.platforms.teams.graph_files import GraphUploadedFile

        adapter = self._make_adapter()
        adapter._conv_refs["19:chat@thread.v2"] = SimpleNamespace(
            conversation=SimpleNamespace(conversation_type="groupChat"))
        adapter._graph_client = object()
        doc = tmp_path / "deck.pptx"
        doc.write_bytes(b"PK\x03\x04fake-pptx")

        async def _upload(graph, target, *, file_name, data, **_kw):
            assert target.conversation_type == "groupChat"
            assert target.chat_id == "19:chat@thread.v2"
            return GraphUploadedFile(
                name="deck.pptx",
                web_url="https://contoso-my.sharepoint.com/personal/bot/deck.pptx",
            )

        with patch("plugins.platforms.teams.graph_files.upload_conversation_file", _upload):
            result = await adapter.send_document(
                "19:chat@thread.v2", str(doc), file_name="deck.pptx")

        assert result.success
        sent = adapter._app.send.await_args.args[1]
        assert "deck.pptx" in sent
        assert "https://contoso-my.sharepoint.com/personal/bot/deck.pptx" in sent

    @pytest.mark.asyncio
    async def test_send_document_channel_graph_403_mentions_permissions(self, tmp_path):
        from plugins.platforms.teams.graph_files import GraphFileTarget
        from tools.microsoft_graph_client import MicrosoftGraphAPIError

        adapter = self._make_adapter()
        adapter._conv_refs["19:chan@thread.tacv2"] = SimpleNamespace(
            conversation=SimpleNamespace(conversation_type="channel"))
        adapter._graph_file_targets["19:chan@thread.tacv2"] = GraphFileTarget(
            conversation_type="channel", team_id="team-guid",
            channel_id="19:chan@thread.tacv2")
        adapter._graph_client = object()
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 binary")

        async def _upload(*_a, **_k):
            raise MicrosoftGraphAPIError(
                403, "GET", "/teams/team-guid/channels/x/filesFolder", "Access denied")

        with patch("plugins.platforms.teams.graph_files.upload_conversation_file", _upload):
            result = await adapter.send_document(
                "19:chan@thread.tacv2", str(doc), file_name="report.pdf")

        assert not result.success
        assert "403" in (result.error or "")
        assert "Files.ReadWrite.All" in result.error
        assert "FileConsent" in result.error
        sent = adapter._app.send.await_args.args[1]
        assert "report.pdf" in sent
        adapter._app.activity_sender.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_document_channel_missing_team_id_is_clear(self, tmp_path):
        adapter = self._make_adapter()
        adapter._graph_client = object()
        adapter._conv_refs["19:chan@thread.tacv2"] = SimpleNamespace(
            conversation=SimpleNamespace(conversation_type="channel"))
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 binary")
        result = await adapter.send_document(
            "19:chan@thread.tacv2", str(doc), file_name="report.pdf")
        assert not result.success
        assert "TEAMS_TEAM_ID" in result.error
        assert "FileConsent" in result.error

    def test_inlineable_channel_document_helpers(self, tmp_path):
        f = _teams_mod._is_inlineable_channel_document
        assert f("/tmp/a.txt")
        assert f("/tmp/a.bin", "notes.md")
        assert not f("/tmp/a.pdf")
        text_path = tmp_path / "ok.txt"
        text_path.write_text("pawn to e4")
        assert _teams_mod._read_inline_channel_text(str(text_path)) == "pawn to e4"
        bin_path = tmp_path / "x.bin"
        bin_path.write_bytes(b"\x00\x01")
        assert _teams_mod._read_inline_channel_text(str(bin_path)) is None




# ---------------------------------------------------------------------------
# Tests: require_mention gating (RSC-delivered history)
# ---------------------------------------------------------------------------

class TestTeamsRequireMention:
    """With resource-specific consent Teams delivers every channel/groupChat message, not just
    mentions. ``require_mention`` must drop unaddressed non-personal posts BEFORE the attachment
    loop, keep @mentions (wire id ``28:<app id>``) / replies to the bot / personal chats, and be
    read env-over-YAML like every other adapter."""

    APP_ID = "bot-id"

    def _make_adapter(self, monkeypatch=None, **extra):
        adapter = TeamsAdapter(_make_config(
            client_id=self.APP_ID, client_secret="secret", tenant_id="tenant", **extra))
        adapter._app = MagicMock()
        adapter._app.id = self.APP_ID
        adapter.handle_message = AsyncMock()
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"\x89PNG" + b"\0" * 32)
        return adapter

    def _activity(self, conversation_type, *, text="hello", mentioned_id=None, reply_to_id=None):
        activity = MagicMock()
        activity.text = text
        activity.id = f"act-{conversation_type}-{mentioned_id}-{reply_to_id}"
        activity.from_ = MagicMock(aad_object_id="aad-456", name="Test User")
        activity.from_.id = "29:user-123"
        activity.recipient = MagicMock()
        activity.recipient.id = f"28:{self.APP_ID}"
        activity.conversation = MagicMock(conversation_type=conversation_type, tenant_id="t")
        activity.conversation.id = "19:conv@thread.v2"
        activity.conversation.name = "Conv"
        att = MagicMock(content_type="image/png")
        att.name = "a.png"
        att.content_url = "https://smba.trafficmanager.net/emea/v3/attachments/1/views/original"
        activity.attachments = [att]
        activity.reply_to_id = reply_to_id
        activity.entities = []
        if mentioned_id:
            entity = MagicMock(type="mention")
            entity.mentioned = MagicMock()
            entity.mentioned.id = mentioned_id
            activity.entities = [entity]
        return activity

    @pytest.mark.anyio
    @pytest.mark.parametrize("conversation_type, kwargs, dispatched", [
        ("channel", {}, False),
        ("groupChat", {}, False),
        ("channel", {"text": "<at>Alice</at> hi", "mentioned_id": "29:alice"}, False),  # someone else
        ("channel", {"text": "<at>Hermes</at> hi", "mentioned_id": "28:bot-id"}, True),  # wire form of the bot id
        ("groupChat", {"text": "<at>Hermes</at> hi", "mentioned_id": "bot-id"}, True),
        ("channel", {"reply_to_id": "bot-msg-1"}, True),
        ("personal", {}, True),
    ])
    async def test_gate_drops_unaddressed_non_personal_before_attachment_download(
        self, conversation_type, kwargs, dispatched,
    ):
        adapter = self._make_adapter(require_mention=True)
        adapter._sent_ids.append("bot-msg-1")
        ctx = MagicMock()
        ctx.activity = self._activity(conversation_type, **kwargs)
        await adapter._on_message(ctx)
        assert adapter.handle_message.await_count == (1 if dispatched else 0)
        assert adapter._fetch_attachment_bytes.await_count == (1 if dispatched else 0)

    @pytest.mark.anyio
    async def test_gate_drops_file_download_info_before_fetch(self):
        """File attachments must not be downloaded when require_mention drops the message."""
        adapter = self._make_adapter(require_mention=True)
        activity = self._activity("channel")
        att = MagicMock()
        att.content_type = "application/vnd.microsoft.teams.file.download.info"
        att.content_url = None
        att.name = "secret.pdf"
        att.content = {
            "downloadUrl": "https://contoso.sharepoint.com/download/secret",
            "fileType": "pdf",
        }
        activity.attachments = [att]
        ctx = MagicMock()
        ctx.activity = activity
        await adapter._on_message(ctx)
        adapter.handle_message.assert_not_awaited()
        adapter._fetch_attachment_bytes.assert_not_awaited()

    @pytest.mark.parametrize("yaml_value, env_value, expected", [
        (None, None, False),      # opt-in: absent key leaves every conversation ungated
        (True, None, True),
        ("false", None, False),
        (True, "false", False),   # explicit env beats YAML, like MATRIX_/MATTERMOST_REQUIRE_MENTION
        (False, "true", True),
    ])
    def test_require_mention_read_env_over_yaml(self, monkeypatch, yaml_value, env_value, expected):
        monkeypatch.delenv("TEAMS_REQUIRE_MENTION", raising=False)
        if env_value is not None:
            monkeypatch.setenv("TEAMS_REQUIRE_MENTION", env_value)
        extra = {} if yaml_value is None else {"require_mention": yaml_value}
        adapter = self._make_adapter(**extra)
        assert adapter._require_mention is expected
        assert adapter._extra.get("require_mention") == yaml_value  # extras stay readable on the instance


class _FakeTeamsSessionEntry:
    session_id = "teams-channel-session"


class _FakeTeamsSessionStore:
    def __init__(self):
        self.sources = []
        self.messages = []

    def get_or_create_session(self, source):
        self.sources.append(source)
        return _FakeTeamsSessionEntry()

    def append_to_transcript(self, session_id, message, skip_db=False):
        self.messages.append((session_id, message, skip_db))


class TestTeamsObserveUnmentioned:
    """RSC + require_mention: unaddressed posts are observed, not dispatched."""

    APP_ID = "bot-id"

    def _make_adapter(self, **extra):
        adapter = TeamsAdapter(_make_config(
            client_id=self.APP_ID, client_secret="secret", tenant_id="tenant",
            require_mention=True, **extra))
        adapter._app = MagicMock()
        adapter._app.id = self.APP_ID
        adapter.handle_message = AsyncMock()
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"\x89PNG" + b"\0" * 32)
        adapter._session_store = _FakeTeamsSessionStore()
        return adapter

    def _activity(self, *, text="side chatter", mentioned_id=None, reply_to_id=None):
        activity = MagicMock()
        activity.text = text
        activity.id = "act-obs-1"
        from_account = MagicMock()
        from_account.aad_object_id = "aad-456"
        from_account.name = "Alice"
        from_account.id = "29:user-123"
        activity.from_ = from_account
        activity.recipient = MagicMock()
        activity.recipient.id = f"28:{self.APP_ID}"
        activity.conversation = MagicMock(conversation_type="channel", tenant_id="t")
        activity.conversation.id = "19:conv@thread.v2"
        activity.conversation.name = "Conv"
        att = MagicMock(content_type="image/png")
        att.name = "a.png"
        att.content_url = "https://smba.trafficmanager.net/emea/v3/attachments/1/views/original"
        activity.attachments = [att]
        activity.reply_to_id = reply_to_id
        activity.entities = []
        if mentioned_id:
            entity = MagicMock(type="mention")
            entity.mentioned = MagicMock()
            entity.mentioned.id = mentioned_id
            activity.entities = [entity]
        return activity

    @pytest.mark.anyio
    async def test_unmentioned_is_observed_not_dispatched(self):
        adapter = self._make_adapter()
        ctx = MagicMock()
        ctx.activity = self._activity()
        await adapter._on_message(ctx)
        adapter.handle_message.assert_not_awaited()
        adapter._fetch_attachment_bytes.assert_not_awaited()
        store = adapter._session_store
        assert len(store.messages) == 1
        session_id, message, _skip = store.messages[0]
        assert session_id == "teams-channel-session"
        assert message["role"] == "user"
        assert message["content"] == "[Alice] side chatter"
        assert message["observed"] is True
        assert message["message_id"] == "act-obs-1"
        assert store.sources[0].user_id is None
        assert store.sources[0].chat_id == "19:conv@thread.v2"

    @pytest.mark.anyio
    async def test_observe_off_drops_without_transcript(self):
        adapter = self._make_adapter(observe_unmentioned=False)
        ctx = MagicMock()
        ctx.activity = self._activity()
        await adapter._on_message(ctx)
        adapter.handle_message.assert_not_awaited()
        adapter._fetch_attachment_bytes.assert_not_awaited()
        assert adapter._session_store.messages == []

    @pytest.mark.anyio
    async def test_mention_dispatches_with_observed_context_marker(self):
        adapter = self._make_adapter()
        ctx = MagicMock()
        ctx.activity = self._activity(
            text="<at>Hermes</at> what did Alice say?", mentioned_id="28:bot-id")
        await adapter._on_message(ctx)
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args[0][0]
        assert "observed Teams channel context" in (event.channel_prompt or "")
        assert event.source.user_id is None
        assert "[Alice]" in event.text
        assert "what did Alice say?" in event.text
        assert adapter._session_store.messages == []

    def test_run_wraps_teams_observed_context_and_keeps_telegram_marker(self):
        from gateway.run import (
            _build_gateway_agent_history,
            _uses_telegram_observed_group_context,
            _wrap_current_message_with_observed_context,
        )
        history = [
            {"role": "user", "content": "[Alice] side chatter", "observed": True},
            {"role": "user", "content": "[Bob] what did Alice say?"},
        ]
        teams_prompt = "observed Teams channel context may be provided"
        telegram_prompt = "observed Telegram group context may be provided"
        assert _uses_telegram_observed_group_context(teams_prompt)
        assert _uses_telegram_observed_group_context(telegram_prompt)
        replay, observed = _build_gateway_agent_history(history, channel_prompt=teams_prompt)
        assert observed == "[Alice] side chatter"
        assert [row["content"] for row in replay] == ["[Bob] what did Alice say?"]
        wrapped = _wrap_current_message_with_observed_context("answer me", observed)
        assert "[Alice] side chatter" in wrapped
        assert "Current addressed message" in wrapped
        replay_tg, observed_tg = _build_gateway_agent_history(
            history, channel_prompt=telegram_prompt)
        assert observed_tg == observed


# ---------------------------------------------------------------------------
# Tests: reactions + file consent
# ---------------------------------------------------------------------------


class TestTeamsReactionMapping:
    def test_unicode_and_alias_map_to_connector_types(self):
        f = _teams_mod._to_teams_reaction_type
        assert f("👍") == "like"
        assert f("heart") == "heart"
        assert f("👀") == "1f440_eyes"
        assert f("✅") == "2705_whiteheavycheckmark"
        assert f("❌") == "angry"
        assert f("like") == "like"
        assert f("2705_whiteheavycheckmark") == "2705_whiteheavycheckmark"
        assert f("") is None
        assert f("not an emoji !!!") is None

    def test_reaction_type_to_emoji(self):
        assert _teams_mod._reaction_to_emoji("like") == "👍"
        assert _teams_mod._reaction_to_emoji("1f440_eyes") == "👀"
        assert _teams_mod._reaction_to_emoji("custom_id") == "custom_id"

    def test_onedrive_upload_url_allowlist(self):
        f = _teams_mod._is_allowed_onedrive_upload_url
        assert f("https://contoso.sharepoint.com/personal/u/upload")
        assert f("https://my.sharepoint.com:443/upload")
        assert not f("http://contoso.sharepoint.com/upload")
        assert not f("https://evilsharepoint.com/upload")
        assert not f("https://sharepoint.com.evil.example/upload")
        assert not f("https://example.com/upload")

    def test_normalize_consent_action_handles_sdk_enum_and_strings(self):
        # Mirrors microsoft_teams.api.models.action.Action (str, Enum).
        # On Python 3.11, str(Action.ACCEPT) is 'Action.ACCEPT', not 'accept'.
        class Action(str, Enum):
            ACCEPT = "accept"
            DECLINE = "decline"

        f = _teams_mod._normalize_consent_action
        # 3.11: str(Action.ACCEPT) == 'Action.ACCEPT'; other versions may stringify to 'accept'.
        assert str(Action.ACCEPT).lower() in {"action.accept", "accept"}
        assert Action.ACCEPT.value == "accept"
        assert f(Action.ACCEPT) == "accept"
        assert f(Action.DECLINE) == "decline"
        assert f("Action.ACCEPT") == "accept"
        assert f("accept") == "accept"
        assert f("DECLINE") == "decline"
        assert f(None) == ""
        assert f("action.accept") == "accept"


class TestTeamsReactions:
    def _make_adapter(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter._app.api.reactions.add = AsyncMock()
        adapter._app.api.reactions.delete = AsyncMock()
        return adapter

    @pytest.mark.anyio
    async def test_add_reaction_maps_thumbs_up_to_like(self):
        adapter = self._make_adapter()
        result = await adapter.add_reaction("19:abc@thread.v2", "👍", message_id="act-1")
        assert result["success"] is True
        assert result["reaction"] == "like"
        adapter._app.api.reactions.add.assert_awaited_once_with(
            "19:abc@thread.v2", "act-1", "like")

    @pytest.mark.anyio
    async def test_add_reaction_defaults_to_last_inbound(self):
        adapter = self._make_adapter()
        adapter._last_inbound_by_chat["19:abc@thread.v2"] = "last-in"
        result = await adapter.add_reaction("19:abc@thread.v2", "❤️")
        assert result["success"] is True
        assert result["message_id"] == "last-in"
        adapter._app.api.reactions.add.assert_awaited_once_with(
            "19:abc@thread.v2", "last-in", "heart")

    @pytest.mark.anyio
    async def test_add_reaction_without_target_errors(self):
        adapter = self._make_adapter()
        result = await adapter.add_reaction("19:abc@thread.v2", "like")
        assert result["success"] is False
        assert "message_id" in result["error"]
        adapter._app.api.reactions.add.assert_not_awaited()

    @pytest.mark.anyio
    async def test_remove_reaction_uses_last_bot_set_type(self):
        adapter = self._make_adapter()
        await adapter.add_reaction("19:abc@thread.v2", "like", message_id="act-1")
        result = await adapter.remove_reaction("19:abc@thread.v2", message_id="act-1")
        assert result["success"] is True
        adapter._app.api.reactions.delete.assert_awaited_once_with(
            "19:abc@thread.v2", "act-1", "like")

    @pytest.mark.anyio
    async def test_processing_start_adds_eyes_when_enabled(self):
        adapter = self._make_adapter()
        event = MagicMock()
        event.source.chat_id = "19:abc@thread.v2"
        event.message_id = "act-1"
        await adapter.on_processing_start(event)
        adapter._app.api.reactions.add.assert_awaited_once_with(
            "19:abc@thread.v2", "act-1", "1f440_eyes")

    @pytest.mark.anyio
    async def test_processing_start_skipped_when_reactions_disabled(self, monkeypatch):
        monkeypatch.setenv("TEAMS_REACTIONS", "false")
        adapter = self._make_adapter()
        event = MagicMock()
        event.source.chat_id = "19:abc@thread.v2"
        event.message_id = "act-1"
        await adapter.on_processing_start(event)
        adapter._app.api.reactions.add.assert_not_awaited()

    @pytest.mark.anyio
    async def test_inbound_reaction_forwards_to_handler(self):
        adapter = self._make_adapter()
        adapter._reaction_handler = AsyncMock()
        activity = MagicMock()
        activity.from_ = MagicMock(id="29:user", aad_object_id="aad-1", name="Ada")
        activity.recipient = MagicMock(id="28:bot-id")
        activity.conversation = MagicMock(
            id="19:abc@thread.v2", conversation_type="personal",
            name="Chat", tenant_id="tenant")
        activity.reply_to_id = "orig-msg"
        activity.id = "reaction-act"
        activity.reactions_added = [SimpleNamespace(type="like")]
        activity.reactions_removed = []
        ctx = MagicMock()
        ctx.activity = activity
        await adapter._on_message_reaction(ctx)
        adapter._reaction_handler.assert_awaited_once()
        payload = adapter._reaction_handler.await_args[0][0]
        assert payload["platform"] == "teams"
        assert payload["event_name"] == "reaction:added"
        assert payload["reaction"] == "👍"
        assert payload["channel_id"] == "19:abc@thread.v2"
        assert payload["message_ts"] == "orig-msg"

    @pytest.mark.anyio
    async def test_inbound_self_reaction_is_ignored(self):
        adapter = self._make_adapter()
        adapter._reaction_handler = AsyncMock()
        activity = MagicMock()
        activity.from_ = MagicMock(id="28:bot-id", aad_object_id=None, name="Hermes")
        activity.recipient = MagicMock(id="28:bot-id")
        activity.conversation = MagicMock(id="19:abc@thread.v2", conversation_type="personal")
        activity.reply_to_id = "orig-msg"
        activity.reactions_added = [SimpleNamespace(type="like")]
        activity.reactions_removed = []
        ctx = MagicMock()
        ctx.activity = activity
        await adapter._on_message_reaction(ctx)
        adapter._reaction_handler.assert_not_awaited()

    @pytest.mark.anyio
    async def test_react_falls_back_to_rest_when_sdk_client_missing(self):
        adapter = self._make_adapter()
        adapter._app.api = None
        adapter._react_via_rest = AsyncMock()
        ok = await adapter._add_reaction("19:abc@thread.v2", "act-1", "👍")
        assert ok is True
        adapter._react_via_rest.assert_awaited_once_with(
            "19:abc@thread.v2", "act-1", "like", remove=False)


class TestTeamsFileConsent:
    def _make_adapter(self, monkeypatch=None):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter._app.send = AsyncMock(return_value=MagicMock(id="consent-1"))
        adapter._upload_consented_file = AsyncMock()
        adapter._send_file_info_card = AsyncMock()
        return adapter

    def _wire_dismiss(self, adapter):
        delete = AsyncMock()
        ops = MagicMock()
        ops.delete = delete
        adapter._app.api.conversations.activities = MagicMock(return_value=ops)
        return delete

    def _ctx(self, *, action, file_id="fid-1", upload_url="https://contoso.sharepoint.com/upload",
             reply_to_id=None):
        activity = MagicMock()
        activity.from_ = MagicMock(id="29:user", aad_object_id="aad-1", name="Ada")
        activity.conversation = MagicMock(id="19:abc@thread.v2")
        activity.reply_to_id = reply_to_id
        activity.replyToId = reply_to_id
        activity.value = {
            "action": action,
            "context": {"file_id": file_id},
            "uploadInfo": {
                "uploadUrl": upload_url,
                "name": "report.pdf",
                "uniqueId": "uid",
                "fileType": "pdf",
                "contentUrl": "https://contoso.sharepoint.com/file",
            },
        }
        ctx = MagicMock()
        ctx.activity = activity
        return ctx

    @pytest.mark.anyio
    async def test_accept_uploads_and_clears_pending(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
        adapter = self._make_adapter()
        adapter._pending_uploads["fid-1"] = {"name": "report.pdf", "bytes": b"%PDF"}
        with patch("tools.url_safety.is_safe_url", lambda url: True):
            await adapter._on_file_consent(self._ctx(action="accept"))
        adapter._upload_consented_file.assert_awaited_once()
        adapter._send_file_info_card.assert_awaited_once()
        assert "fid-1" not in adapter._pending_uploads

    @pytest.mark.anyio
    async def test_accept_sdk_enum_uploads_and_clears_pending(self, monkeypatch):
        """FileConsent invoke types action as Action (str, Enum); str() is not 'accept'."""
        class Action(str, Enum):
            ACCEPT = "accept"
            DECLINE = "decline"

        monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
        adapter = self._make_adapter()
        adapter._pending_uploads["fid-1"] = {"name": "report.pdf", "bytes": b"%PDF"}
        ctx = self._ctx(action="accept")
        # SDK models expose attributes, not only dict keys.
        ctx.activity.value = SimpleNamespace(
            action=Action.ACCEPT,
            context=SimpleNamespace(file_id="fid-1"),
            uploadInfo={
                "uploadUrl": "https://contoso.sharepoint.com/upload",
                "name": "report.pdf",
                "uniqueId": "uid",
                "fileType": "pdf",
                "contentUrl": "https://contoso.sharepoint.com/file",
            },
        )
        with patch("tools.url_safety.is_safe_url", lambda url: True):
            await adapter._on_file_consent(ctx)
        adapter._upload_consented_file.assert_awaited_once()
        adapter._send_file_info_card.assert_awaited_once()
        assert "fid-1" not in adapter._pending_uploads

    @pytest.mark.anyio
    async def test_accept_rejects_unsafe_upload_url(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
        adapter = self._make_adapter()
        adapter._pending_uploads["fid-1"] = {"name": "report.pdf", "bytes": b"%PDF"}
        await adapter._on_file_consent(self._ctx(
            action="accept", upload_url="https://evil.example/steal"))
        adapter._upload_consented_file.assert_not_awaited()
        assert "fid-1" in adapter._pending_uploads

    @pytest.mark.anyio
    async def test_decline_drops_pending(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
        adapter = self._make_adapter()
        adapter.send = AsyncMock(return_value=MagicMock(success=True))
        adapter._pending_uploads["fid-1"] = {"name": "report.pdf", "bytes": b"%PDF"}
        await adapter._on_file_consent(self._ctx(action="decline"))
        adapter._upload_consented_file.assert_not_awaited()
        assert "fid-1" not in adapter._pending_uploads

    @pytest.mark.anyio
    async def test_unauthorized_click_does_not_upload(self, monkeypatch):
        monkeypatch.delenv("TEAMS_ALLOW_ALL_USERS", raising=False)
        monkeypatch.setenv("TEAMS_ALLOWED_USERS", "someone-else")
        adapter = self._make_adapter()
        adapter._pending_uploads["fid-1"] = {"name": "report.pdf", "bytes": b"%PDF"}
        await adapter._on_file_consent(self._ctx(action="accept"))
        adapter._upload_consented_file.assert_not_awaited()
        assert "fid-1" in adapter._pending_uploads

    @pytest.mark.anyio
    async def test_accept_dismisses_consent_card(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
        adapter = self._make_adapter()
        delete = self._wire_dismiss(adapter)
        adapter._pending_uploads["fid-1"] = {"name": "report.pdf", "bytes": b"%PDF"}
        with patch("tools.url_safety.is_safe_url", lambda url: True):
            await adapter._on_file_consent(self._ctx(action="accept", reply_to_id="consent-card-1"))
        adapter._app.api.conversations.activities.assert_called_once_with("19:abc@thread.v2")
        delete.assert_awaited_once_with("consent-card-1")
        adapter._upload_consented_file.assert_awaited_once()

    @pytest.mark.anyio
    async def test_decline_dismisses_consent_card(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
        adapter = self._make_adapter()
        adapter.send = AsyncMock(return_value=MagicMock(success=True))
        delete = self._wire_dismiss(adapter)
        adapter._pending_uploads["fid-1"] = {"name": "report.pdf", "bytes": b"%PDF"}
        await adapter._on_file_consent(self._ctx(action="decline", reply_to_id="consent-card-1"))
        delete.assert_awaited_once_with("consent-card-1")

    @pytest.mark.anyio
    async def test_stale_pending_dismisses_consent_card(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
        adapter = self._make_adapter()
        adapter.send = AsyncMock(return_value=MagicMock(success=True))
        delete = self._wire_dismiss(adapter)
        with patch("tools.url_safety.is_safe_url", lambda url: True):
            await adapter._on_file_consent(self._ctx(action="accept", reply_to_id="consent-card-1"))
        adapter._upload_consented_file.assert_not_awaited()
        delete.assert_awaited_once_with("consent-card-1")

    @pytest.mark.anyio
    async def test_unauthorized_dismisses_consent_card(self, monkeypatch):
        monkeypatch.delenv("TEAMS_ALLOW_ALL_USERS", raising=False)
        monkeypatch.setenv("TEAMS_ALLOWED_USERS", "someone-else")
        adapter = self._make_adapter()
        delete = self._wire_dismiss(adapter)
        adapter._pending_uploads["fid-1"] = {"name": "report.pdf", "bytes": b"%PDF"}
        await adapter._on_file_consent(self._ctx(action="accept", reply_to_id="consent-card-1"))
        adapter._upload_consented_file.assert_not_awaited()
        delete.assert_awaited_once_with("consent-card-1")

    @pytest.mark.anyio
    async def test_dismiss_failure_does_not_fail_upload(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
        adapter = self._make_adapter()
        delete = self._wire_dismiss(adapter)
        delete.side_effect = RuntimeError("connector 404")
        adapter._pending_uploads["fid-1"] = {"name": "report.pdf", "bytes": b"%PDF"}
        with patch("tools.url_safety.is_safe_url", lambda url: True):
            await adapter._on_file_consent(self._ctx(action="accept", reply_to_id="consent-card-1"))
        adapter._upload_consented_file.assert_awaited_once()
        adapter._send_file_info_card.assert_awaited_once()
        assert "fid-1" not in adapter._pending_uploads

    def test_consent_card_activity_id_reads_camel_and_snake(self):
        f = _teams_mod._consent_card_activity_id
        assert f(SimpleNamespace(reply_to_id="act-1", replyToId=None)) == "act-1"
        assert f(SimpleNamespace(replyToId="act-2")) == "act-2"
        assert f({"replyToId": "act-3"}) == "act-3"
        assert f(SimpleNamespace(reply_to_id=None, replyToId=None)) is None
        assert f(SimpleNamespace()) is None


class TestTeamsStreaming:
    def _make_adapter(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.send = AsyncMock(return_value=MagicMock(id="act-1"))
        return adapter

    def _wire_update(self, adapter):
        update = AsyncMock()
        ops = MagicMock()
        ops.update = update
        ops.delete = AsyncMock()
        adapter._app.api.conversations.activities = MagicMock(return_value=ops)
        return update, ops

    def test_flat_conversation_id_strips_messageid_suffix(self):
        f = _teams_mod._flat_conversation_id
        assert f("19:abc@thread.tacv2") == "19:abc@thread.tacv2"
        assert f("19:abc@thread.tacv2;messageid=12345") == "19:abc@thread.tacv2"
        assert f("19:abc@thread.tacv2;messageId=12345") == "19:abc@thread.tacv2"
        assert f("") == ""

    def test_draft_stream_is_message_stays_false(self):
        adapter = self._make_adapter()
        assert adapter.draft_stream_is_message is False
        assert TeamsAdapter.draft_stream_is_message is False

    @pytest.mark.anyio
    async def test_send_then_progressive_edit_then_finalize(self):
        adapter = self._make_adapter()
        update, _ops = self._wire_update(adapter)

        created = await adapter.send("19:abc@thread.v2", "Hel")
        assert created.success is True
        assert created.message_id == "act-1"
        adapter._app.send.assert_awaited_once()

        mid = await adapter.edit_message(
            "19:abc@thread.v2", "act-1", "Hello wor", finalize=False)
        assert mid.success is True
        assert mid.message_id == "act-1"
        fin = await adapter.edit_message(
            "19:abc@thread.v2", "act-1", "Hello world", finalize=True)
        assert fin.success is True
        assert fin.message_id == "act-1"

        adapter._app.api.conversations.activities.assert_called_with("19:abc@thread.v2")
        assert update.await_count == 2
        assert update.await_args_list[0].args[0] == "act-1"
        assert update.await_args_list[1].args[0] == "act-1"

    @pytest.mark.anyio
    async def test_edit_strips_messageid_before_activity_update(self):
        adapter = self._make_adapter()
        update, _ops = self._wire_update(adapter)
        result = await adapter.edit_message(
            "19:abc@thread.tacv2;messageid=999", "act-1", "Hello")
        assert result.success is True
        adapter._app.api.conversations.activities.assert_called_once_with(
            "19:abc@thread.tacv2")
        update.assert_awaited_once()
        assert update.await_args.args[0] == "act-1"

    @pytest.mark.anyio
    async def test_identical_midstream_edits_are_coalesced(self):
        adapter = self._make_adapter()
        update, _ops = self._wire_update(adapter)
        first = await adapter.edit_message(
            "19:abc@thread.v2", "act-1", "same text", finalize=False)
        second = await adapter.edit_message(
            "19:abc@thread.v2", "act-1", "same text", finalize=False)
        assert first.success is True and second.success is True
        assert update.await_count == 1

    @pytest.mark.anyio
    async def test_update_unsupported_falls_back_to_nonstreaming_send(self):
        adapter = self._make_adapter()
        update, _ops = self._wire_update(adapter)

        class _Unsupported(Exception):
            status_code = 405

        update.side_effect = _Unsupported("Method Not Allowed")
        edited = await adapter.edit_message(
            "19:abc@thread.v2", "act-1", "partial", finalize=False)
        assert edited.success is False
        sent = await adapter.send("19:abc@thread.v2", "full reply")
        assert sent.success is True
        assert sent.message_id == "act-1"

    @pytest.mark.anyio
    async def test_edit_falls_back_to_rest_when_sdk_client_missing(self):
        adapter = self._make_adapter()
        adapter._app.api = None
        adapter._update_activity_via_rest = AsyncMock()
        result = await adapter.edit_message("19:abc@thread.v2", "act-1", "Hello")
        assert result.success is True
        adapter._update_activity_via_rest.assert_awaited_once_with(
            "19:abc@thread.v2", "act-1", "Hello")

    @pytest.mark.anyio
    async def test_rest_update_uses_flat_conversation_id(self):
        adapter = self._make_adapter()
        adapter._app.api = None
        adapter._get_botframework_token = AsyncMock(return_value="tok")
        captured = {}

        class _Resp:
            status_code = 200
            request = MagicMock()

            def raise_for_status(self):
                pass

        class _Client:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def put(self, url, json=None, headers=None):
                captured["url"] = url
                captured["json"] = json
                captured["headers"] = headers
                return _Resp()

        with patch("httpx.AsyncClient", _Client):
            result = await adapter.edit_message(
                "19:abc@thread.tacv2;messageid=42", "act-9", "**hi**")
        assert result.success is True
        assert "/v3/conversations/19:abc@thread.tacv2/activities/act-9" in captured["url"]
        assert ";messageid=" not in captured["url"]
        assert captured["json"]["type"] == "message"
        assert captured["json"]["id"] == "act-9"
        assert captured["headers"]["Authorization"] == "Bearer tok"

    @pytest.mark.anyio
    async def test_short_429_retries_inline(self):
        adapter = self._make_adapter()

        class _RateLimit(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"Retry-After": "0"})

        adapter._update_activity = AsyncMock(side_effect=[_RateLimit("slow down"), None])
        result = await adapter.edit_message("19:abc@thread.v2", "act-1", "Hello")
        assert result.success is True
        assert adapter._update_activity.await_count == 2

    @pytest.mark.anyio
    async def test_dismiss_still_uses_activities_delete(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter._app.send = AsyncMock(return_value=MagicMock(id="consent-1"))
        adapter._upload_consented_file = AsyncMock()
        adapter._send_file_info_card = AsyncMock()
        delete = AsyncMock()
        ops = MagicMock()
        ops.delete = delete
        adapter._app.api.conversations.activities = MagicMock(return_value=ops)
        adapter._pending_uploads["fid-1"] = {"name": "report.pdf", "bytes": b"%PDF"}
        activity = MagicMock()
        activity.from_ = MagicMock(id="29:user", aad_object_id="aad-1", name="Ada")
        activity.conversation = MagicMock(id="19:abc@thread.tacv2;messageid=77")
        activity.reply_to_id = "consent-card-1"
        activity.replyToId = "consent-card-1"
        activity.value = {
            "action": "accept",
            "context": {"file_id": "fid-1"},
            "uploadInfo": {
                "uploadUrl": "https://contoso.sharepoint.com/upload",
                "name": "report.pdf",
                "uniqueId": "uid",
                "fileType": "pdf",
                "contentUrl": "https://contoso.sharepoint.com/file",
            },
        }
        ctx = MagicMock()
        ctx.activity = activity
        with patch("tools.url_safety.is_safe_url", lambda url: True):
            await adapter._on_file_consent(ctx)
        adapter._app.api.conversations.activities.assert_called_with("19:abc@thread.tacv2")
        delete.assert_awaited_once_with("consent-card-1")
        adapter._upload_consented_file.assert_awaited_once()

