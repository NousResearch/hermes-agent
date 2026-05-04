"""Tests for the Microsoft Teams platform adapter plugin."""

import asyncio
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig, HomeChannel
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

    # MessageActivity mock
    microsoft_teams_api.MessageActivity = MagicMock
    microsoft_teams_api.ConversationReference = MagicMock
    microsoft_teams_api.MessageActivityInput = MagicMock

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

_teams_mod.TEAMS_SDK_AVAILABLE = True
_teams_mod.AIOHTTP_AVAILABLE = True

TeamsAdapter = _teams_mod.TeamsAdapter
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
    def test_returns_false_when_sdk_missing(self, monkeypatch):
        monkeypatch.setattr(_teams_mod, "TEAMS_SDK_AVAILABLE", False)
        assert check_requirements() is False

    def test_returns_false_when_aiohttp_missing(self, monkeypatch):
        monkeypatch.setattr(_teams_mod, "AIOHTTP_AVAILABLE", False)
        assert check_requirements() is False

    def test_returns_true_when_deps_available(self, monkeypatch):
        monkeypatch.setattr(_teams_mod, "TEAMS_SDK_AVAILABLE", True)
        monkeypatch.setattr(_teams_mod, "AIOHTTP_AVAILABLE", True)
        assert check_requirements() is True

    def test_alias_matches(self, monkeypatch):
        monkeypatch.setattr(_teams_mod, "TEAMS_SDK_AVAILABLE", True)
        monkeypatch.setattr(_teams_mod, "AIOHTTP_AVAILABLE", True)
        assert check_teams_requirements() is True

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

    def test_validate_config_missing(self, monkeypatch):
        monkeypatch.delenv("TEAMS_CLIENT_ID", raising=False)
        monkeypatch.delenv("TEAMS_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("TEAMS_TENANT_ID", raising=False)
        assert validate_config(_make_config()) is False

    def test_validate_config_missing_tenant(self, monkeypatch):
        monkeypatch.setenv("TEAMS_CLIENT_ID", "test-id")
        monkeypatch.setenv("TEAMS_CLIENT_SECRET", "test-secret")
        monkeypatch.delenv("TEAMS_TENANT_ID", raising=False)
        assert validate_config(_make_config()) is False


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

    def test_falls_back_to_env_vars(self, monkeypatch):
        monkeypatch.setenv("TEAMS_CLIENT_ID", "env-id")
        monkeypatch.setenv("TEAMS_CLIENT_SECRET", "env-secret")
        monkeypatch.setenv("TEAMS_TENANT_ID", "env-tenant")
        adapter = TeamsAdapter(_make_config())
        assert adapter._client_id == "env-id"
        assert adapter._client_secret == "env-secret"
        assert adapter._tenant_id == "env-tenant"

    def test_default_port(self):
        adapter = TeamsAdapter(_make_config(client_id="id", client_secret="secret", tenant_id="tenant"))
        assert adapter._port == 3978

    def test_custom_port_from_extra(self):
        adapter = TeamsAdapter(_make_config(client_id="id", client_secret="secret", tenant_id="tenant", port=4000))
        assert adapter._port == 4000

    def test_custom_port_from_env(self, monkeypatch):
        monkeypatch.setenv("TEAMS_PORT", "5000")
        adapter = TeamsAdapter(_make_config(client_id="id", client_secret="secret", tenant_id="tenant"))
        assert adapter._port == 5000

    def test_platform_value(self):
        adapter = TeamsAdapter(_make_config(client_id="id", client_secret="secret", tenant_id="tenant"))
        assert adapter.platform.value == "teams"


# ---------------------------------------------------------------------------
# Tests: Plugin registration
# ---------------------------------------------------------------------------

class TestTeamsPluginRegistration:

    def test_register_calls_ctx(self):
        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()

    def test_register_name(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["name"] == "teams"

    def test_register_auth_env_vars(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["allowed_users_env"] == "TEAMS_ALLOWED_USERS"
        assert kwargs["allow_all_env"] == "TEAMS_ALLOW_ALL_USERS"

    def test_register_max_message_length(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["max_message_length"] == 28000

    def test_register_has_setup_fn(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert callable(kwargs.get("setup_fn"))

    def test_register_has_platform_hint(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs.get("platform_hint")


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
    @pytest.mark.asyncio
    async def test_connect_fails_without_sdk(self, monkeypatch):
        monkeypatch.setattr(_teams_mod, "TEAMS_SDK_AVAILABLE", False)
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_fails_without_credentials(self):
        adapter = TeamsAdapter(_make_config())
        adapter._client_id = ""
        adapter._client_secret = ""
        adapter._tenant_id = ""
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._running = True
        mock_runner = AsyncMock()
        adapter._runner = mock_runner
        adapter._app = MagicMock()

        await adapter.disconnect()
        assert adapter._running is False
        assert adapter._app is None
        assert adapter._runner is None
        mock_runner.cleanup.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: Send
# ---------------------------------------------------------------------------

class TestTeamsSend:
    @pytest.mark.asyncio
    async def test_send_returns_error_without_app(self):
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = None
        result = await adapter.send("conv-id", "Hello")
        assert result.success is False
        assert "not initialized" in result.error

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_send_handles_error(self):
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_app = MagicMock()
        mock_app.send = AsyncMock(side_effect=Exception("Network error"))
        adapter._app = mock_app

        result = await adapter.send("conv-id", "Hello")
        assert result.success is False
        assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_send_typing(self):
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_app = MagicMock()
        mock_app.send = AsyncMock()
        adapter._app = mock_app

        await adapter.send_typing("conv-id")
        mock_app.send.assert_awaited_once()
        call_args = mock_app.send.call_args
        assert call_args[0][0] == "conv-id"


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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_channel_message_creates_channel_event(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()

        activity = self._make_activity(conversation_type="channel")
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_type == "channel"

    @pytest.mark.asyncio
    async def test_user_id_uses_aad_object_id(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()

        activity = self._make_activity(from_aad_id="aad-stable-id", from_id="teams-id")
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.source.user_id == "aad-stable-id"

    @pytest.mark.asyncio
    async def test_self_message_filtered(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()

        activity = self._make_activity(from_id="bot-id")
        await adapter._on_message(self._make_ctx(activity))

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bot_mention_stripped_from_text(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()

        activity = self._make_activity(
            text="<at>Hermes</at> what is the weather?",
            from_id="user-id",
        )
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.text == "what is the weather?"

    @pytest.mark.asyncio
    async def test_deduplication(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()

        activity = self._make_activity(activity_id="msg-dup-001", from_id="user-id")
        ctx = self._make_ctx(activity)

        await adapter._on_message(ctx)
        await adapter._on_message(ctx)

        assert adapter.handle_message.await_count == 1
