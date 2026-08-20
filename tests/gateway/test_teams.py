"""Tests for the Microsoft Teams platform adapter plugin."""

import json
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from gateway.config import Platform, PlatformConfig, HomeChannel
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
# SDK import is deferred (#62935); bind mocked symbols the same way connect() does.
assert _teams_mod.check_teams_requirements() is True
_teams_mod.TEAMS_SDK_AVAILABLE = True

# Ensure SDK symbols that were None (import failed on Python <3.12) are
# replaced with the mocked versions so runtime calls don't silently no-op.
import sys as _sys
_mt = _sys.modules.get("microsoft_teams.api.activities.typing")
if _mt and _teams_mod.TypingActivityInput is None:
    _teams_mod.TypingActivityInput = _mt.TypingActivityInput

TeamsAdapter = _teams_mod.TeamsAdapter
TeamsSummaryWriter = _teams_mod.TeamsSummaryWriter
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


    def test_returns_true_when_deps_available(self, monkeypatch):
        monkeypatch.setattr(_teams_mod, "TEAMS_SDK_AVAILABLE", True)
        monkeypatch.setattr(_teams_mod, "AIOHTTP_AVAILABLE", True)
        assert check_requirements() is True

    def test_check_teams_requirements_shortcircuits_when_present(self, monkeypatch):
        # When SDK symbols are already bound and aiohttp is available, the
        # active lazy-installer returns True immediately without re-importing.
        monkeypatch.setattr(_teams_mod, "App", object())
        monkeypatch.setattr(_teams_mod, "AIOHTTP_AVAILABLE", True)
        called = {"ensure_and_bind": 0}

        def _fake_ensure_and_bind(*_args, **_kwargs):
            called["ensure_and_bind"] += 1
            return True

        monkeypatch.setattr(
            "tools.lazy_deps.ensure_and_bind", _fake_ensure_and_bind
        )
        assert check_teams_requirements() is True
        assert called["ensure_and_bind"] == 0

    def test_check_teams_requirements_lazy_installs_when_missing(self, monkeypatch):
        # When deps are missing, the active installer delegates to
        # ensure_and_bind("platform.teams", ...) — parity with Slack/Discord.
        monkeypatch.setattr(_teams_mod, "App", None)
        monkeypatch.setattr(_teams_mod, "TEAMS_SDK_AVAILABLE", False)
        monkeypatch.setattr(_teams_mod, "AIOHTTP_AVAILABLE", False)
        seen = {}

        def _fake_ensure_and_bind(feature, importer, target_globals, **kwargs):
            seen["feature"] = feature
            return True

        monkeypatch.setattr(
            "tools.lazy_deps.ensure_and_bind", _fake_ensure_and_bind
        )
        assert check_teams_requirements() is True
        assert seen["feature"] == "platform.teams"

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


    def test_register_name(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["name"] == "teams"

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
        # Simulate the SDK being unavailable AND not installable (offline /
        # locked-down env): the lazy-installer can't rebind the globals, so
        # TEAMS_SDK_AVAILABLE stays False and connect() must fail.
        monkeypatch.setattr(
            "tools.lazy_deps.ensure_and_bind",
            lambda *_a, **_k: False,
        )
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        result = await adapter.connect()
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Send
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tests: chat-type classification (channel + channelData hardening)
# ---------------------------------------------------------------------------

class TestTeamsChannelClassification:
    def _make_activity(
        self,
        *,
        conversation_type="",
        conversation_id="19:abc@thread.tacv2",
        channel_data=None,
    ):
        activity = MagicMock()
        activity.text = "Hello"
        activity.id = "activity-001"
        activity.from_ = MagicMock()
        activity.from_.id = "user-123"
        activity.from_.aad_object_id = "aad-456"
        activity.from_.name = "Test User"
        activity.conversation = MagicMock()
        activity.conversation.id = conversation_id
        activity.conversation.conversation_type = conversation_type
        activity.conversation.name = "Test Channel"
        activity.conversation.tenant_id = "tenant-789"
        activity.attachments = []
        activity.entities = []
        if channel_data is not None:
            activity.channel_data = channel_data
        else:
            del activity.channel_data  # MagicMock: make getattr(..., None) really return None
        return activity

    def _make_ctx(self, activity):
        ctx = MagicMock()
        ctx.activity = activity
        return ctx

    def _make_adapter(self):
        # require_mention=False: this class tests conversationType ->
        # chat_type classification only. Mention-gating is an independent
        # axis (see TestTeamsRequireMention) and would otherwise drop these
        # activities before chat_type is ever observable, since none of them
        # carry a mention entity.
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
            require_mention=False,
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()
        return adapter

    @pytest.mark.anyio
    async def test_conversation_type_channel_maps_to_channel(self):
        """conversationType == "channel" was previously untested end to end."""
        adapter = self._make_adapter()
        activity = self._make_activity(conversation_type="channel")
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_type == "channel"

    @pytest.mark.anyio
    async def test_missing_conversation_type_with_channel_data_is_channel(self):
        """Hardening: a missing/unrecognized conversationType with channelData
        naming a channel/team must not fall through to the DM default, or the
        channel allowlist below would never see it."""
        adapter = self._make_adapter()
        activity = self._make_activity(
            conversation_type="",
            channel_data={"channel": {"id": "19:abc@thread.tacv2"}},
        )
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_type == "channel"

    @pytest.mark.anyio
    async def test_missing_conversation_type_without_channel_data_is_dm(self):
        """Preserves the pre-existing fail-toward-DM default when there is no
        channelData signal to classify by either."""
        adapter = self._make_adapter()
        activity = self._make_activity(conversation_type="", channel_data=None)
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_type == "dm"


# ---------------------------------------------------------------------------
# Tests: channel allowlist (TEAMS_ALLOWED_CHANNELS / allowed_channels)
# ---------------------------------------------------------------------------

class TestTeamsChannelAllowlist:
    def _make_activity(
        self,
        *,
        conversation_id="19:abc@thread.tacv2",
        channel_data=None,
        mentions_bot=True,
        bot_id="bot-id",
    ):
        activity = MagicMock()
        activity.text = "<at>Hermes</at> hello"
        activity.id = "activity-001"
        activity.from_ = MagicMock()
        activity.from_.id = "user-123"
        activity.from_.aad_object_id = "aad-456"
        activity.from_.name = "Test User"
        activity.conversation = MagicMock()
        activity.conversation.id = conversation_id
        activity.conversation.conversation_type = "channel"
        activity.conversation.name = "Test Channel"
        activity.conversation.tenant_id = "tenant-789"
        activity.attachments = []
        if channel_data is not None:
            activity.channel_data = channel_data
        else:
            del activity.channel_data
        activity.recipient = MagicMock()
        activity.recipient.id = bot_id
        mention_entity = MagicMock()
        mention_entity.type = "mention"
        mention_entity.mentioned = MagicMock()
        mention_entity.mentioned.id = bot_id if mentions_bot else "someone-else"
        activity.entities = [mention_entity]
        return activity

    def _make_ctx(self, activity):
        ctx = MagicMock()
        ctx.activity = activity
        return ctx

    def _make_adapter(self, **extra):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant", **extra,
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()
        return adapter

    @pytest.mark.anyio
    async def test_empty_allowlist_is_unchanged_and_not_role_authorized(self):
        """Empty allowed_channels leaves the CHANNEL-SCOPING behavior
        byte-identical to before this feature existed -- no channel is
        excluded, and role_authorized stays False (TEAMS_ALLOWED_USERS is
        still what gates it). The mention requirement is a separate,
        independent default (see TestTeamsRequireMention) -- this activity
        is mentioned so it isolates the channel-scoping behavior alone."""
        adapter = self._make_adapter()
        activity = self._make_activity(mentions_bot=True)
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.source.role_authorized is False

    @pytest.mark.anyio
    async def test_channel_not_in_allowlist_is_dropped_silently(self):
        adapter = self._make_adapter(allowed_channels=["19:other@thread.tacv2"])
        activity = self._make_activity(conversation_id="19:abc@thread.tacv2")
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.anyio
    async def test_channel_in_allowlist_dispatches_role_authorized(self):
        adapter = self._make_adapter(allowed_channels=["19:abc@thread.tacv2"])
        activity = self._make_activity(conversation_id="19:abc@thread.tacv2")
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.source.role_authorized is True

    @pytest.mark.anyio
    async def test_threaded_reply_still_matches_base_channel_id(self):
        """A channel reply's conversation.id carries a ;messageid=<root>
        suffix — the same channel, different string. Must still match."""
        adapter = self._make_adapter(allowed_channels=["19:abc@thread.tacv2"])
        activity = self._make_activity(
            conversation_id="19:abc@thread.tacv2;messageid=1699999999999"
        )
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.anyio
    async def test_matches_by_channel_data_channel_id(self):
        adapter = self._make_adapter(allowed_channels=["channel-xyz"])
        activity = self._make_activity(
            conversation_id="19:different@thread.tacv2",
            channel_data={"channel": {"id": "channel-xyz"}},
        )
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.anyio
    async def test_matches_by_channel_data_team_id(self):
        adapter = self._make_adapter(allowed_channels=["team-xyz"])
        activity = self._make_activity(
            conversation_id="19:different@thread.tacv2",
            channel_data={"team": {"id": "team-xyz"}},
        )
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.anyio
    async def test_wildcard_allows_any_channel(self):
        adapter = self._make_adapter(allowed_channels=["*"])
        activity = self._make_activity(conversation_id="19:whatever@thread.tacv2")
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.source.role_authorized is True

    @pytest.mark.anyio
    async def test_allowlist_match_is_case_sensitive(self):
        """Deliberately case-sensitive: Teams ids are opaque platform strings
        with no documented case-insensitive contract, and folding case could
        collapse two genuinely distinct ids -- widening the allowlist rather
        than narrowing it. A differently-cased id must NOT match."""
        adapter = self._make_adapter(allowed_channels=["19:ABC@Thread.Tacv2"])
        activity = self._make_activity(conversation_id="19:abc@thread.tacv2")
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.anyio
    async def test_allowlist_match_is_exact(self):
        adapter = self._make_adapter(allowed_channels=["19:ABC@Thread.Tacv2"])
        activity = self._make_activity(conversation_id="19:ABC@Thread.Tacv2")
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.anyio
    async def test_dm_is_unaffected_by_channel_allowlist(self):
        adapter = self._make_adapter(allowed_channels=["19:only-this-channel@thread.tacv2"])
        activity = self._make_activity(conversation_id="19:some-dm-conversation")
        activity.conversation.conversation_type = "personal"
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_type == "dm"
        assert event.source.role_authorized is False

    @pytest.mark.anyio
    async def test_group_chat_is_unaffected_by_channel_allowlist(self):
        adapter = self._make_adapter(allowed_channels=["19:only-this-channel@thread.tacv2"])
        activity = self._make_activity(conversation_id="19:some-group-chat")
        activity.conversation.conversation_type = "groupChat"
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_type == "group"
        assert event.source.role_authorized is False


# ---------------------------------------------------------------------------
# Tests: mention gate (require_mention / TEAMS_REQUIRE_MENTION)
# ---------------------------------------------------------------------------

class TestTeamsRequireMention:
    def _make_activity(self, *, mentioned_id=None, recipient_id="recipient-id"):
        activity = MagicMock()
        activity.text = "hello"
        activity.id = "activity-001"
        activity.from_ = MagicMock()
        activity.from_.id = "user-123"
        activity.from_.aad_object_id = "aad-456"
        activity.from_.name = "Test User"
        activity.conversation = MagicMock()
        activity.conversation.id = "19:abc@thread.tacv2"
        activity.conversation.conversation_type = "channel"
        activity.conversation.name = "Test Channel"
        activity.conversation.tenant_id = "tenant-789"
        activity.attachments = []
        del activity.channel_data
        activity.recipient = MagicMock()
        activity.recipient.id = recipient_id
        if mentioned_id is not None:
            mention_entity = MagicMock()
            mention_entity.type = "mention"
            mention_entity.mentioned = MagicMock()
            mention_entity.mentioned.id = mentioned_id
            activity.entities = [mention_entity]
        else:
            activity.entities = []
        return activity

    def _make_ctx(self, activity):
        ctx = MagicMock()
        ctx.activity = activity
        return ctx

    def _make_adapter(self, **extra):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
            allowed_channels=["19:abc@thread.tacv2"], **extra,
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()
        return adapter

    @pytest.mark.anyio
    async def test_mentioned_channel_message_dispatches(self):
        adapter = self._make_adapter()
        activity = self._make_activity(mentioned_id="recipient-id")
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.anyio
    async def test_mention_matches_activity_recipient_not_oauth_client_id(self):
        """Regression: an earlier version compared the mention entity against
        self._app.id (the OAuth client id used for the self-message filter),
        not activity.recipient.id (the addressee Teams actually stamped on
        this activity, per the SDK's own is_recipient_mentioned()). The two
        can differ -- a genuinely mentioned message must not be dropped just
        because those ids don't match."""
        adapter = self._make_adapter()
        adapter._app.id = "oauth-client-id"  # deliberately NOT the recipient id
        activity = self._make_activity(
            recipient_id="channel-scoped-recipient-id",
            mentioned_id="channel-scoped-recipient-id",
        )
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.anyio
    async def test_unmentioned_channel_message_is_dropped_by_default(self):
        adapter = self._make_adapter()
        activity = self._make_activity(mentioned_id=None)
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.anyio
    async def test_require_mention_false_lets_unmentioned_message_through(self):
        adapter = self._make_adapter(require_mention=False)
        activity = self._make_activity(mentioned_id=None)
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.anyio
    async def test_mention_of_a_different_user_does_not_count(self):
        adapter = self._make_adapter()
        activity = self._make_activity(mentioned_id="someone-else")
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.anyio
    async def test_dm_never_requires_a_mention(self):
        adapter = self._make_adapter()
        activity = self._make_activity(mentioned_id=None)
        activity.conversation.conversation_type = "personal"
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.anyio
    async def test_require_mention_applies_even_without_an_allowlist(self):
        """require_mention is independent of allowed_channels -- it is not a
        bonus that only activates once a channel allowlist is configured.
        An unmentioned channel message must be dropped by default even when
        TEAMS_ALLOWED_CHANNELS/allowed_channels is unset entirely, closing
        the RSC-delivered-unmentioned-message gap for every Teams channel,
        not just allowlisted ones."""
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()

        activity = self._make_activity(mentioned_id=None)
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: config.yaml -> env bridging (apply_yaml_config_fn)
# ---------------------------------------------------------------------------

class TestTeamsYamlConfigBridge:
    def test_seeds_allowed_channels_into_extra_from_list(self):
        result = _teams_mod._apply_yaml_config(
            {}, {"allowed_channels": ["19:abc@thread.tacv2", "19:def@thread.tacv2"]}
        )
        assert result == {"allowed_channels": ["19:abc@thread.tacv2", "19:def@thread.tacv2"]}

    def test_seeds_allowed_channels_into_extra_from_csv_string(self):
        result = _teams_mod._apply_yaml_config({}, {"allowed_channels": "19:abc@thread.tacv2"})
        assert result == {"allowed_channels": "19:abc@thread.tacv2"}

    def test_absent_key_is_a_no_op(self):
        assert _teams_mod._apply_yaml_config({}, {}) is None

    def test_does_not_touch_os_environ(self, monkeypatch):
        """Regression: an earlier version bridged through a process-global env
        var, which silently dropped a multiplexed secondary profile's
        allowlist (_platform_gate_env treats its own scope as authoritative
        and does not fall through to another profile's env write, #72348
        mirror) -- letting every channel through unrestricted. Seeding
        PlatformConfig.extra directly instead means this key must never touch
        os.environ at all."""
        monkeypatch.delenv("TEAMS_ALLOWED_CHANNELS", raising=False)
        _teams_mod._apply_yaml_config({}, {"allowed_channels": ["19:abc@thread.tacv2"]})
        assert "TEAMS_ALLOWED_CHANNELS" not in os.environ

    def test_allowed_channels_survive_a_multiplexed_profile_scope(self, monkeypatch):
        """Reproduces the exact failure class _platform_gate_env's docstring
        warns about (#72348): under multiplexing, a secondary profile's own
        secret scope is authoritative and does NOT fall through to a
        process-global env write. Bridging allowed_channels through
        os.environ (like the Telegram/DingTalk/Matrix siblings do) would
        silently return an empty set there -- and an empty set means
        UNRESTRICTED, not "deny all" (_teams_allowed_channels /
        _should_process_message) -- admitting every channel. Seeding
        PlatformConfig.extra directly must survive this scenario."""
        from agent import secret_scope

        seeded = _teams_mod._apply_yaml_config(
            {}, {"allowed_channels": ["19:abc@thread.tacv2"]}
        )
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant", **seeded,
        ))

        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
        token = secret_scope.set_secret_scope({})  # this profile's scope: no such key
        try:
            assert adapter._teams_allowed_channels() == {"19:abc@thread.tacv2"}
        finally:
            secret_scope.reset_secret_scope(token)

    def test_real_config_loader_wires_allowed_channels_through_to_the_adapter(
        self, tmp_path, monkeypatch
    ):
        """Exercises the actual wiring end to end, not just _apply_yaml_config
        in isolation: the module's real register() -> PlatformEntry(
        apply_yaml_config_fn=_apply_yaml_config) -> gateway/config.py's real
        load_gateway_config() dispatch loop -> PlatformConfig.extra ->
        _teams_allowed_channels(), surviving a multiplexed secret scope at
        the end. Registering through register() itself (not a hand-built
        PlatformEntry) means this test would fail if register() ever forgot
        the apply_yaml_config_fn kwarg -- a prior version of this test built
        the PlatformEntry directly and would have missed exactly that.
        Mirrors the harness tests/gateway/test_platform_registry.py already
        uses for the generic apply_yaml_config_fn contract.
        """
        from agent import secret_scope
        from gateway.config import Platform, load_gateway_config
        from gateway.platform_registry import PlatformEntry, platform_registry

        class _RealCtx:
            """Forwards register_platform(...) into the real platform_registry,
            building the same PlatformEntry the production PluginContext
            would -- but without needing a full PluginContext/manifest."""

            def register_platform(
                self, name, label, adapter_factory, check_fn,
                validate_config=None, required_env=None, install_hint="",
                **entry_kwargs,
            ):
                entry_kwargs.setdefault("plugin_name", "teams-platform")
                platform_registry.register(PlatformEntry(
                    name=name, label=label, adapter_factory=adapter_factory,
                    check_fn=check_fn, validate_config=validate_config,
                    required_env=required_env or [], install_hint=install_hint,
                    source="plugin", **entry_kwargs,
                ))

        _teams_mod.register(_RealCtx())
        try:
            hermes_home = tmp_path / ".hermes"
            hermes_home.mkdir()
            (hermes_home / "config.yaml").write_text(
                'teams:\n  allowed_channels:\n    - "19:abc@thread.tacv2"\n',
                encoding="utf-8",
            )
            monkeypatch.setenv("HERMES_HOME", str(hermes_home))

            config = load_gateway_config()
            platform_cfg = config.platforms[Platform("teams")]
            assert platform_cfg.extra["allowed_channels"] == ["19:abc@thread.tacv2"]

            adapter = TeamsAdapter(platform_cfg)
            monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
            token = secret_scope.set_secret_scope({})  # this profile's scope: no such key
            try:
                assert adapter._teams_allowed_channels() == {"19:abc@thread.tacv2"}
            finally:
                secret_scope.reset_secret_scope(token)
        finally:
            platform_registry.unregister("teams")


# ---------------------------------------------------------------------------
# Tests: config.yaml vs env precedence (explicit contract, not an accident)
# ---------------------------------------------------------------------------

class TestTeamsConfigPrecedence:
    """config.yaml (PlatformConfig.extra) wins over the env var when both
    are set for the same key. This is a DELIBERATE, precedented choice, not
    an oversight: it matches Telegram's own channel/chat allowlist
    (``_telegram_allowed_chats`` — ``plugins/platforms/telegram/adapter.py``
    checks ``extra.get("allowed_chats")`` before ``TELEGRAM_ALLOWED_CHATS``),
    DingTalk's own ``require_mention`` (``_dingtalk_require_mention`` —
    ``plugins/platforms/dingtalk/adapter.py`` checks extra before
    ``DINGTALK_REQUIRE_MENTION``), and this same Teams adapter's own
    pre-existing ``client_id``/``client_secret``/``tenant_id`` handling
    (``extra.get(...) or os.getenv(...)``). config.yaml is the richer,
    version-controlled surface; env vars are the fallback for operators who
    haven't migrated a given key to config.yaml yet -- not a
    higher-priority override once a key IS in config.yaml.
    """

    def test_allowed_channels_extra_wins_over_conflicting_env(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOWED_CHANNELS", "19:from-env@thread.tacv2")
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
            allowed_channels=["19:from-extra@thread.tacv2"],
        ))
        assert adapter._teams_allowed_channels() == {"19:from-extra@thread.tacv2"}

    def test_require_mention_extra_wins_over_conflicting_env(self, monkeypatch):
        monkeypatch.setenv("TEAMS_REQUIRE_MENTION", "true")
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
            require_mention=False,
        ))
        assert adapter._teams_require_mention() is False


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
        from gateway.platforms.base import MessageType

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
        from gateway.platforms.base import MessageType

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
    """send_video / send_voice / send_document route through the same
    Attachment mechanism as send_image so the gateway's media dispatch
    (run.py) delivers native attachments instead of the base-class text
    fallback (file path sent as plain text)."""

    def _make_adapter(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter._app.send = AsyncMock(return_value=MagicMock(id="msg-001"))
        return adapter


    @pytest.mark.asyncio
    async def test_send_voice_local_file_base64(self, tmp_path):
        adapter = self._make_adapter()
        audio = tmp_path / "reply.mp3"
        audio.write_bytes(b"ID3fakeaudio")
        result = await adapter.send_voice("19:abc@thread.v2", str(audio), caption="here you go")
        assert result.success
        adapter._app.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_document_local_file_base64(self, tmp_path):
        adapter = self._make_adapter()
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")
        result = await adapter.send_document("19:abc@thread.v2", str(doc))
        assert result.success
        adapter._app.send.assert_awaited_once()


