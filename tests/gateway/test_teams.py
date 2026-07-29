"""Tests for the Microsoft Teams platform adapter plugin."""

import json
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

_teams_mod.TEAMS_SDK_AVAILABLE = True
_teams_mod.AIOHTTP_AVAILABLE = True

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

    def test_check_teams_requirements_shortcircuits_when_present(self, monkeypatch):
        # When the SDK + aiohttp are already importable, the active lazy-
        # installer returns True immediately without attempting an install.
        monkeypatch.setattr(_teams_mod, "TEAMS_SDK_AVAILABLE", True)
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

    def test_invalid_port_from_extra_falls_back_to_default(self):
        adapter = TeamsAdapter(
            _make_config(client_id="id", client_secret="secret", tenant_id="tenant", port="abc")
        )
        assert adapter._port == 3978

    def test_invalid_port_from_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("TEAMS_PORT", "abc")
        adapter = TeamsAdapter(_make_config(client_id="id", client_secret="secret", tenant_id="tenant"))
        assert adapter._port == 3978

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

    @pytest.mark.anyio
    async def test_connect_fails_without_credentials(self):
        adapter = TeamsAdapter(_make_config())
        adapter._client_id = ""
        adapter._client_secret = ""
        adapter._tenant_id = ""
        result = await adapter.connect()
        assert result is False

    @pytest.mark.anyio
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
    @pytest.mark.anyio
    async def test_send_returns_error_without_app(self):
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = None
        result = await adapter.send("conv-id", "Hello")
        assert result.success is False
        assert "not initialized" in result.error

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

    @pytest.mark.anyio
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
    async def test_incoming_webhook_posts_summary_text(self):
        seen = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"ok": True})

        writer = TeamsSummaryWriter(transport=httpx.MockTransport(_handler))
        payload = _make_summary_payload()

        result = await writer.write_summary(
            payload,
            {
                "delivery_mode": "incoming_webhook",
                "incoming_webhook_url": "https://example.test/teams-webhook",
            },
        )

        assert result["delivery_mode"] == "incoming_webhook"
        assert seen["url"] == "https://example.test/teams-webhook"
        assert "Weekly Sync" in seen["body"]["text"]
        assert "Proceed with staged rollout." in seen["body"]["text"]

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

    @pytest.mark.anyio
    async def test_graph_delivery_falls_back_to_platform_home_channel(self):
        graph_client = SimpleNamespace(post_json=AsyncMock(return_value={"id": "msg-home"}))
        platform_config = PlatformConfig(
            enabled=True,
            extra={"team_id": "team-home", "delivery_mode": "graph"},
            home_channel=HomeChannel(
                platform=Platform("teams"),
                chat_id="channel-home",
                name="Teams Home",
            ),
        )
        writer = TeamsSummaryWriter(platform_config=platform_config, graph_client=graph_client)

        await writer.write_summary(_make_summary_payload(), {})

        graph_client.post_json.assert_awaited_once()
        assert graph_client.post_json.await_args.args[0] == "/teams/team-home/channels/channel-home/messages"

    @pytest.mark.anyio
    async def test_existing_record_is_reused_without_force_resend(self):
        graph_client = SimpleNamespace(post_json=AsyncMock())
        writer = TeamsSummaryWriter(graph_client=graph_client)
        existing = {"delivery_mode": "graph", "message_id": "msg-existing"}

        result = await writer.write_summary(
            _make_summary_payload(),
            {
                "delivery_mode": "graph",
                "team_id": "team-1",
                "channel_id": "channel-1",
            },
            existing_record=existing,
        )

        assert result == existing
        graph_client.post_json.assert_not_awaited()


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

    @pytest.mark.anyio
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

    @pytest.mark.anyio
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

    @pytest.mark.anyio
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

    @pytest.mark.anyio
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
        adapter._fetch_teams_image_bytes = AsyncMock(return_value=b"\x89PNGfake")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_teams_mod, "cache_image_from_bytes", lambda data, **kw: "/tmp/img.png")
            activity = self._make_activity([
                self._image_attachment(),
                self._file_download_attachment(),
            ])
            await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT
        assert len(event.media_urls) == 2

    @pytest.mark.anyio
    async def test_html_body_attachment_stays_text(self):
        from gateway.platforms.base import MessageType

        adapter = self._make_adapter()
        activity = self._make_activity([self._html_body_attachment()])
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.TEXT
        assert event.media_urls == []

    @pytest.mark.anyio
    async def test_image_only_still_photo(self):
        from gateway.platforms.base import MessageType

        adapter = self._make_adapter()
        adapter._fetch_teams_image_bytes = AsyncMock(return_value=b"\x89PNGfake")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_teams_mod, "cache_image_from_bytes", lambda data, **kw: "/tmp/img.png")
            activity = self._make_activity([self._image_attachment()])
            await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.PHOTO
        assert event.media_urls == ["/tmp/img.png"]

    @pytest.mark.anyio
    async def test_download_failure_degrades_to_text(self):
        from gateway.platforms.base import MessageType

        adapter = self._make_adapter()
        adapter._fetch_attachment_bytes = AsyncMock(side_effect=Exception("boom"))

        activity = self._make_activity([self._file_download_attachment()])
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.TEXT
        assert event.media_urls == []


# ── inbound image auth (pasted-image fetch) ──────────────────────────────


def _httpx_response(status, *, content=b"", json_data=None):
    """Build a real httpx.Response so .json()/.text/.raise_for_status() work."""
    req = httpx.Request("GET", "https://smba.trafficmanager.net/x")
    if json_data is not None:
        body = json.dumps(json_data).encode()
        return httpx.Response(
            status, content=body, request=req,
            headers={"content-type": "application/json"},
        )
    return httpx.Response(status, content=content, request=req)


def _patch_httpx(monkeypatch, *, get_response=None, post_response=None, get_responses=None):
    """Replace httpx.AsyncClient with a fake capturing get/post calls.

    ``get_responses`` (a list) returns responses in sequence across successive
    GETs — used to exercise the retry loop; ``get_response`` returns the same
    response every call.
    """
    calls = {"get": [], "post": [], "init_kwargs": None}
    seq = list(get_responses) if get_responses is not None else None

    class _FakeClient:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            calls["get"].append((url, kw))
            if seq is not None:
                return seq.pop(0)
            return get_response

        async def post(self, url, **kw):
            calls["post"].append((url, kw))
            return post_response

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    return calls


def _patch_httpx_transport(monkeypatch, handler):
    """Drive a REAL httpx.AsyncClient through a MockTransport.

    Unlike ``_patch_httpx`` (which stubs ``.get``), this lets the adapter's
    ``client.stream(...)`` + the shared bounded reader + httpx's redirect/auth
    semantics all run for real. ``handler(request) -> httpx.Response``.
    """
    real = httpx.AsyncClient

    def _mk(**kw):
        return real(**kw, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", _mk)


_PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-payload"


class TestTeamsInboundImageAuth:
    """Pasted inline images are token-gated Bot Connector URLs; the adapter
    must fetch them with the bot's api.botframework.com bearer (an anonymous
    GET 401s and the image was silently dropped) and cache to a local path
    so run.py's native image pipeline attaches them."""

    def _make_adapter(self, **creds):
        cfg = _make_config(**(creds or dict(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        )))
        adapter = TeamsAdapter(cfg)
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        adapter.handle_message = AsyncMock()
        return adapter

    def _image_attachment(self, content_url="https://smba.trafficmanager.net/fr/v3/attachments/abc/views/original"):
        att = MagicMock()
        att.content_type = "image/png"
        att.content_url = content_url
        att.name = "pasted.png"
        return att

    def _make_activity(self, attachments, text="look at this"):
        activity = MagicMock()
        activity.text = text
        activity.id = "activity-img-001"
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

    # -- _mint_bot_connector_token --------------------------------------

    @pytest.mark.anyio
    async def test_mint_token_success(self, monkeypatch):
        calls = _patch_httpx(
            monkeypatch,
            post_response=_httpx_response(200, json_data={"access_token": "tok", "expires_in": 1234}),
        )
        token, expires = await _teams_mod._mint_bot_connector_token(
            tenant_id="t", client_id="c", client_secret="s",
        )
        assert token == "tok"
        assert expires == 1234
        url, kw = calls["post"][0]
        assert "login.microsoftonline.com/t/oauth2/v2.0/token" in url
        assert kw["data"]["grant_type"] == "client_credentials"
        assert kw["data"]["scope"] == "https://api.botframework.com/.default"

    @pytest.mark.anyio
    async def test_mint_token_http_error_raises(self, monkeypatch):
        _patch_httpx(
            monkeypatch,
            post_response=_httpx_response(401, json_data={"error": "unauthorized_client"}),
        )
        with pytest.raises(RuntimeError):
            await _teams_mod._mint_bot_connector_token(
                tenant_id="t", client_id="c", client_secret="s",
            )

    @pytest.mark.anyio
    async def test_mint_token_missing_creds_raises(self):
        with pytest.raises(RuntimeError):
            await _teams_mod._mint_bot_connector_token(
                tenant_id="", client_id="c", client_secret="s",
            )

    # -- _acquire_bot_connector_token (caching) -------------------------

    @pytest.mark.anyio
    async def test_acquire_token_caches(self, monkeypatch):
        adapter = self._make_adapter()
        mint = AsyncMock(return_value=("tok", 3600))
        monkeypatch.setattr(_teams_mod, "_mint_bot_connector_token", mint)

        t1 = await adapter._acquire_bot_connector_token()
        t2 = await adapter._acquire_bot_connector_token()

        assert t1 == t2 == "tok"
        mint.assert_awaited_once()  # second call served from cache

    @pytest.mark.anyio
    async def test_acquire_token_none_when_creds_missing(self, monkeypatch):
        for var in ("TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET", "TEAMS_TENANT_ID"):
            monkeypatch.delenv(var, raising=False)
        adapter = TeamsAdapter(_make_config())
        adapter._app = MagicMock()
        assert await adapter._acquire_bot_connector_token() is None

    @pytest.mark.anyio
    async def test_acquire_token_none_on_mint_failure(self, monkeypatch):
        adapter = self._make_adapter()

        async def _boom(**kwargs):
            raise RuntimeError("sts down")

        monkeypatch.setattr(_teams_mod, "_mint_bot_connector_token", _boom)
        assert await adapter._acquire_bot_connector_token() is None

    # -- _fetch_teams_image_bytes ---------------------------------------

    @pytest.mark.anyio
    async def test_fetch_image_attaches_bearer_for_bot_framework_host(self, monkeypatch):
        adapter = self._make_adapter()
        monkeypatch.setattr(_teams_mod, "_mint_bot_connector_token", AsyncMock(return_value=("tok", 3600)))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: True)
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, content=_PNG_BYTES)

        _patch_httpx_transport(monkeypatch, handler)

        data = await adapter._fetch_teams_image_bytes(
            "https://smba.trafficmanager.net/fr/v3/attachments/abc/views/original"
        )

        assert data == _PNG_BYTES
        assert seen[0].headers.get("Authorization") == "Bearer tok"

    @pytest.mark.anyio
    async def test_fetch_image_no_bearer_for_foreign_host(self, monkeypatch):
        adapter = self._make_adapter()
        mint = AsyncMock(return_value=("tok", 3600))
        monkeypatch.setattr(_teams_mod, "_mint_bot_connector_token", mint)
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: True)
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, content=_PNG_BYTES)

        _patch_httpx_transport(monkeypatch, handler)

        data = await adapter._fetch_teams_image_bytes("https://evil.example.com/img.png")

        assert data == _PNG_BYTES
        assert seen[0].headers.get("Authorization") is None
        mint.assert_not_awaited()  # token never minted for a non-Bot-Framework host

    @pytest.mark.anyio
    async def test_fetch_image_data_uri_decodes_inline(self, monkeypatch):
        import base64

        adapter = self._make_adapter()

        class _NoNetwork:
            def __init__(self, **kw):
                raise AssertionError("data: URI must not hit the network")

        monkeypatch.setattr(httpx, "AsyncClient", _NoNetwork)

        uri = "data:image/png;base64," + base64.b64encode(_PNG_BYTES).decode()
        data = await adapter._fetch_teams_image_bytes(uri)
        assert data == _PNG_BYTES

    @pytest.mark.anyio
    async def test_fetch_image_unsafe_url_raises(self, monkeypatch):
        adapter = self._make_adapter()
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: False)
        with pytest.raises(ValueError):
            await adapter._fetch_teams_image_bytes("https://smba.trafficmanager.net/x")

    @pytest.mark.anyio
    async def test_fetch_image_strips_bearer_on_cross_host_redirect(self, monkeypatch):
        # Security-critical: a token-gated smba URL commonly 302-redirects to a
        # pre-signed blob host. The bot bearer must NOT follow cross-host, or it
        # leaks. This drives a REAL httpx AsyncClient (via MockTransport) so both
        # the SSRF redirect guard and httpx's own cross-origin Authorization
        # stripping actually run — the property the whole design relies on.
        adapter = self._make_adapter()
        monkeypatch.setattr(_teams_mod, "_mint_bot_connector_token", AsyncMock(return_value=("tok", 3600)))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: True)

        seen = []

        def handler(request):
            seen.append((str(request.url), request.headers.get("Authorization")))
            if len(seen) == 1:
                return httpx.Response(302, headers={"location": "https://blob.example.com/signed"})
            return httpx.Response(200, content=_PNG_BYTES)

        real_client = httpx.AsyncClient

        def _client_with_mock(**kw):
            return real_client(**kw, transport=httpx.MockTransport(handler))

        monkeypatch.setattr(httpx, "AsyncClient", _client_with_mock)

        data = await adapter._fetch_teams_image_bytes(
            "https://smba.trafficmanager.net/fr/v3/attachments/abc/views/original"
        )

        assert data == _PNG_BYTES
        assert len(seen) == 2, "expected one redirect hop"
        # Hop 1 (smba, allowlisted) carries the bearer; hop 2 (foreign host) must not.
        assert seen[0][1] == "Bearer tok"
        assert seen[1][0].startswith("https://blob.example.com/")
        assert seen[1][1] is None

    # -- retry on the transient post-paste auth window ------------------

    @pytest.mark.anyio
    async def test_fetch_image_retries_transient_403_then_succeeds(self, monkeypatch):
        adapter = self._make_adapter()
        monkeypatch.setattr(_teams_mod, "_IMAGE_FETCH_RETRY_DELAYS", (0.0, 0.0, 0.0))
        monkeypatch.setattr(_teams_mod, "_mint_bot_connector_token", AsyncMock(return_value=("tok", 3600)))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: True)
        seq = [
            httpx.Response(403, json={"message": "Authorization has been denied for this request."}),
            httpx.Response(200, content=_PNG_BYTES),
        ]
        n = {"i": 0}

        def handler(request):
            r = seq[min(n["i"], len(seq) - 1)]
            n["i"] += 1
            return r

        _patch_httpx_transport(monkeypatch, handler)

        data = await adapter._fetch_teams_image_bytes(
            "https://smba.trafficmanager.net/fr/v3/attachments/abc/views/original"
        )
        assert data == _PNG_BYTES
        assert n["i"] == 2  # one transient 403, then success

    @pytest.mark.anyio
    async def test_fetch_image_exhausts_retries_and_raises(self, monkeypatch):
        adapter = self._make_adapter()
        monkeypatch.setattr(_teams_mod, "_IMAGE_FETCH_RETRY_DELAYS", (0.0, 0.0, 0.0))
        monkeypatch.setattr(_teams_mod, "_mint_bot_connector_token", AsyncMock(return_value=("tok", 3600)))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: True)
        n = {"i": 0}

        def handler(request):
            n["i"] += 1
            return httpx.Response(403, json={"message": "denied"})

        _patch_httpx_transport(monkeypatch, handler)

        with pytest.raises(httpx.HTTPStatusError):
            await adapter._fetch_teams_image_bytes(
                "https://smba.trafficmanager.net/fr/v3/attachments/abc/views/original"
            )
        assert n["i"] == 3  # all attempts consumed, then raised

    @pytest.mark.anyio
    async def test_fetch_image_permanent_404_not_retried(self, monkeypatch):
        adapter = self._make_adapter()
        monkeypatch.setattr(_teams_mod, "_IMAGE_FETCH_RETRY_DELAYS", (0.0, 0.0, 0.0))
        monkeypatch.setattr(_teams_mod, "_mint_bot_connector_token", AsyncMock(return_value=("tok", 3600)))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: True)
        n = {"i": 0}

        def handler(request):
            n["i"] += 1
            return httpx.Response(404, json={"message": "not found"})

        _patch_httpx_transport(monkeypatch, handler)

        with pytest.raises(httpx.HTTPStatusError):
            await adapter._fetch_teams_image_bytes(
                "https://smba.trafficmanager.net/fr/v3/attachments/abc/views/original"
            )
        assert n["i"] == 1  # 404 is permanent — no retry

    @pytest.mark.anyio
    async def test_fetch_image_403_not_retried_when_unauthenticated(self, monkeypatch):
        # Foreign host → no bearer attached → a 403 is genuinely permanent, so
        # we must NOT burn the retry budget on it.
        adapter = self._make_adapter()
        monkeypatch.setattr(_teams_mod, "_IMAGE_FETCH_RETRY_DELAYS", (0.0, 0.0, 0.0))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: True)
        n = {"i": 0}

        def handler(request):
            n["i"] += 1
            return httpx.Response(403, json={"message": "forbidden"})

        _patch_httpx_transport(monkeypatch, handler)

        with pytest.raises(httpx.HTTPStatusError):
            await adapter._fetch_teams_image_bytes("https://evil.example.com/img.png")
        assert n["i"] == 1  # unauthenticated 403 → no retry

    @pytest.mark.anyio
    async def test_fetch_image_oversized_rejected_via_streaming_reader(self, monkeypatch):
        # The bounded streaming reader must reject an oversized body mid-stream:
        # no Content-Length → the pre-check is skipped, so only the per-chunk
        # running-total guard can catch it (proving we stream, not buffer
        # .content). Spy confirms the bounded reader was actually invoked.
        import gateway.platforms.base as _base

        adapter = self._make_adapter()
        monkeypatch.setattr(_teams_mod, "_mint_bot_connector_token", AsyncMock(return_value=("tok", 3600)))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: True)
        monkeypatch.setattr(_base, "get_inbound_media_max_bytes", lambda: 16)

        seen = {"n": 0}
        _orig = _base._read_httpx_body_with_limit

        async def _spy(response, **kw):
            seen["n"] += 1
            return await _orig(response, **kw)

        monkeypatch.setattr(_base, "_read_httpx_body_with_limit", _spy)

        async def _chunks():
            yield b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
            yield b"\x00" * 40

        def handler(request):
            return httpx.Response(200, content=_chunks())

        _patch_httpx_transport(monkeypatch, handler)

        with pytest.raises(ValueError):
            await adapter._fetch_teams_image_bytes(
                "https://smba.trafficmanager.net/fr/v3/attachments/abc/views/original"
            )
        assert seen["n"] >= 1  # the bounded streaming reader was on the path

    # -- end-to-end image branch in _on_message -------------------------

    @pytest.mark.anyio
    async def test_pasted_image_sets_photo_with_local_path(self, monkeypatch):
        from gateway.platforms.base import MessageType

        adapter = self._make_adapter()
        adapter._fetch_teams_image_bytes = AsyncMock(return_value=_PNG_BYTES)
        monkeypatch.setattr(_teams_mod, "cache_image_from_bytes", lambda data, **kw: "/tmp/pasted.png")

        activity = self._make_activity([self._image_attachment()])
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.PHOTO
        assert event.media_urls == ["/tmp/pasted.png"]
        assert event.media_types == ["image/png"]

    @pytest.mark.anyio
    async def test_image_fetch_failure_falls_back_to_url_cache(self, monkeypatch):
        from gateway.platforms.base import MessageType

        adapter = self._make_adapter()
        adapter._fetch_teams_image_bytes = AsyncMock(side_effect=Exception("401 Unauthorized"))

        async def _fake_url_cache(url, *a, **kw):
            return "/tmp/fallback.png"

        monkeypatch.setattr(_teams_mod, "cache_image_from_url", _fake_url_cache)

        activity = self._make_activity([self._image_attachment()])
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.PHOTO
        assert event.media_urls == ["/tmp/fallback.png"]

    @pytest.mark.anyio
    async def test_image_fetch_and_fallback_both_fail_degrade_to_text(self, monkeypatch):
        from gateway.platforms.base import MessageType

        adapter = self._make_adapter()
        adapter._fetch_teams_image_bytes = AsyncMock(side_effect=Exception("401"))

        async def _boom_url_cache(url, *a, **kw):
            raise Exception("also 401")

        monkeypatch.setattr(_teams_mod, "cache_image_from_url", _boom_url_cache)

        activity = self._make_activity([self._image_attachment()])
        await adapter._on_message(self._make_ctx(activity))

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.TEXT
        assert event.media_urls == []


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

        # The token is minted via the shared helper (httpx); only the activity
        # POST goes through aiohttp now.
        mint = AsyncMock(return_value=("the-token", 3600))
        monkeypatch.setattr(_teams_mod, "_mint_bot_connector_token", mint)

        activity_resp = _FakeAiohttpResponse(200, {"id": "msg-99"})
        session = _FakeAiohttpSession([activity_resp])
        _install_fake_aiohttp(monkeypatch, session)

        result = await _teams_mod._standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "19:abc@thread.skype",
            "hello cron",
        )

        assert result == {"success": True, "message_id": "msg-99"}

        # Token minted once with the resolved tenant credentials.
        mint.assert_awaited_once()
        assert mint.await_args.kwargs["tenant_id"] == "tenant"
        assert mint.await_args.kwargs["client_id"] == "client-id"
        assert mint.await_args.kwargs["client_secret"] == "secret"

        assert len(session.calls) == 1
        activity_url, activity_kwargs = session.calls[0]
        # Default service URL when TEAMS_SERVICE_URL is unset
        assert "smba.trafficmanager.net" in activity_url
        assert "/v3/conversations/19:abc@thread.skype/activities" in activity_url
        assert activity_kwargs["headers"]["Authorization"] == "Bearer the-token"
        assert activity_kwargs["json"]["text"] == "hello cron"
        assert activity_kwargs["json"]["type"] == "message"

    @pytest.mark.asyncio
    async def test_standalone_send_returns_error_when_unconfigured(self, monkeypatch):
        for var in ("TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET", "TEAMS_TENANT_ID"):
            monkeypatch.delenv(var, raising=False)

        result = await _teams_mod._standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "19:abc@thread.skype",
            "hi",
        )

        assert "error" in result
        assert "TEAMS_CLIENT_ID" in result["error"]

    @pytest.mark.asyncio
    async def test_standalone_send_propagates_token_failure(self, monkeypatch):
        monkeypatch.setenv("TEAMS_CLIENT_ID", "client-id")
        monkeypatch.setenv("TEAMS_CLIENT_SECRET", "secret")
        monkeypatch.setenv("TEAMS_TENANT_ID", "tenant")
        monkeypatch.delenv("TEAMS_SERVICE_URL", raising=False)

        async def _boom(**kwargs):
            raise RuntimeError("token request failed (401): unauthorized_client")

        monkeypatch.setattr(_teams_mod, "_mint_bot_connector_token", _boom)

        # The activity POST must never fire when the token can't be minted.
        session = _FakeAiohttpSession([])
        _install_fake_aiohttp(monkeypatch, session)

        result = await _teams_mod._standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "19:abc@thread.skype",
            "hi",
        )

        assert "error" in result
        assert "401" in result["error"]
        assert "token" in result["error"].lower()
        assert len(session.calls) == 0

    @pytest.mark.asyncio
    async def test_standalone_send_rejects_off_allowlist_service_url(self, monkeypatch):
        monkeypatch.setenv("TEAMS_CLIENT_ID", "client-id")
        monkeypatch.setenv("TEAMS_CLIENT_SECRET", "secret")
        monkeypatch.setenv("TEAMS_TENANT_ID", "tenant")
        # SSRF attempt: point us at an attacker-controlled host
        monkeypatch.setenv("TEAMS_SERVICE_URL", "https://attacker.example.com/teams/")

        # If the allowlist check fails to fire, the fake session will assert
        # because no scripts are queued; a passing test means we returned
        # before any HTTP call.
        session = _FakeAiohttpSession([])
        _install_fake_aiohttp(monkeypatch, session)

        result = await _teams_mod._standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "19:abc@thread.skype",
            "hi",
        )

        assert "error" in result
        assert "allowlist" in result["error"].lower()
        assert len(session.calls) == 0, "must not call any HTTP endpoint with a tampered service URL"

    @pytest.mark.asyncio
    async def test_standalone_send_rejects_chat_id_with_path_traversal(self, monkeypatch):
        monkeypatch.setenv("TEAMS_CLIENT_ID", "client-id")
        monkeypatch.setenv("TEAMS_CLIENT_SECRET", "secret")
        monkeypatch.setenv("TEAMS_TENANT_ID", "tenant")
        monkeypatch.delenv("TEAMS_SERVICE_URL", raising=False)

        session = _FakeAiohttpSession([])
        _install_fake_aiohttp(monkeypatch, session)

        # Attempt to break out of /v3/conversations/<id>/activities via a `/`
        result = await _teams_mod._standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "19:abc/activities/19:other@thread.skype",
            "hi",
        )

        assert "error" in result
        assert "Bot Framework conversation ID" in result["error"]
        assert len(session.calls) == 0


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
    async def test_send_video_remote_url_succeeds(self):
        adapter = self._make_adapter()
        result = await adapter.send_video("19:abc@thread.v2", "https://cdn.example.com/clip.mp4")
        assert result.success
        assert result.message_id == "msg-001"
        adapter._app.send.assert_awaited_once()

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

    @pytest.mark.asyncio
    async def test_send_video_without_app_fails(self):
        adapter = self._make_adapter()
        adapter._app = None
        result = await adapter.send_video("19:abc@thread.v2", "https://cdn.example.com/clip.mp4")
        assert not result.success
        assert "not initialized" in result.error

    @pytest.mark.asyncio
    async def test_send_document_missing_file_fails_gracefully(self):
        adapter = self._make_adapter()
        result = await adapter.send_document("19:abc@thread.v2", "/no/such/file.pdf")
        assert not result.success
        adapter._app.send.assert_not_awaited()
