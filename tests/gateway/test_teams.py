"""Tests for the Microsoft Teams platform adapter plugin."""

import base64
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

    @pytest.mark.anyio
    async def test_send_oversized_body_splits_into_multiple_chunks(self):
        # A body well over MAX_MESSAGE_LENGTH (28 KB) must be delivered as
        # more than one Teams activity.
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_result = MagicMock()
        mock_result.id = "chunk-id"
        mock_app = MagicMock()
        mock_app.send = AsyncMock(return_value=mock_result)
        adapter._app = mock_app

        big_body = "word " * 8000  # 40000 chars, > 28000
        result = await adapter.send("conv-id", big_body)

        assert result.success is True
        assert mock_app.send.await_count > 1, (
            "A body larger than MAX_MESSAGE_LENGTH must be split across "
            "multiple send() calls."
        )
        # Every chunk goes to the same conversation.
        for call in mock_app.send.await_args_list:
            assert call.args[0] == "conv-id"

    @pytest.mark.anyio
    async def test_send_ten_kb_body_is_single_chunk(self):
        # Regression for CHUNK: send() must chunk at Teams' MAX_MESSAGE_LENGTH
        # (28 KB), not the base-class default (4096). A ~10 KB body fits in one
        # Teams message and must NOT be fragmented.
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_result = MagicMock()
        mock_result.id = "only-chunk"
        mock_app = MagicMock()
        mock_app.send = AsyncMock(return_value=mock_result)
        adapter._app = mock_app

        body = "x" * 10000  # ~10 KB, < 28000
        result = await adapter.send("conv-id", body)

        assert result.success is True
        assert result.message_id == "only-chunk"
        assert mock_app.send.await_count == 1, (
            "A ~10 KB body fits in one Teams message; send() must not split it. "
            "truncate_message must be called with MAX_MESSAGE_LENGTH (28000), "
            "not the 4096 base default."
        )

    @pytest.mark.anyio
    async def test_send_with_numeric_reply_to_uses_reply(self):
        # A digit reply_to routes through the SDK's threaded reply() path.
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_result = MagicMock()
        mock_result.id = "reply-id"
        mock_app = MagicMock()
        mock_app.reply = AsyncMock(return_value=mock_result)
        mock_app.send = AsyncMock()
        adapter._app = mock_app

        result = await adapter.send("conv-id", "Hello", reply_to="42")

        assert result.success is True
        assert result.message_id == "reply-id"
        mock_app.reply.assert_awaited_once_with("conv-id", "42", "Hello")
        mock_app.send.assert_not_awaited()

    @pytest.mark.anyio
    async def test_send_reply_falls_back_to_flat_send_on_error(self):
        # Group chats 400 on threaded sends; reply() failure must fall back to
        # a flat send rather than failing the whole send.
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_result = MagicMock()
        mock_result.id = "flat-id"
        mock_app = MagicMock()
        mock_app.reply = AsyncMock(side_effect=Exception("threaded send not supported"))
        mock_app.send = AsyncMock(return_value=mock_result)
        adapter._app = mock_app

        result = await adapter.send("conv-id", "Hello", reply_to="42")

        assert result.success is True
        assert result.message_id == "flat-id"
        mock_app.reply.assert_awaited_once()
        mock_app.send.assert_awaited_once_with("conv-id", "Hello")

    @pytest.mark.anyio
    async def test_send_failure_is_retryable(self):
        # A transient send failure (httpx 5xx) must surface as retryable so the
        # base layer retries instead of falling back to plain text.
        import httpx

        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        transient = httpx.HTTPStatusError(
            "server error",
            request=httpx.Request("POST", "https://example.test"),
            response=httpx.Response(503),
        )
        mock_app = MagicMock()
        mock_app.send = AsyncMock(side_effect=transient)
        adapter._app = mock_app

        result = await adapter.send("conv-id", "Hello")
        assert result.success is False
        assert result.retryable is True

    @pytest.mark.anyio
    async def test_send_permanent_failure_is_not_retryable(self):
        # A permanent failure (httpx 4xx / local error) must NOT be retryable so
        # base falls back to its plain-text path instead of retrying the same
        # payload.
        import httpx

        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        permanent = httpx.HTTPStatusError(
            "bad request",
            request=httpx.Request("POST", "https://example.test"),
            response=httpx.Response(400),
        )
        mock_app = MagicMock()
        mock_app.send = AsyncMock(side_effect=permanent)
        adapter._app = mock_app

        result = await adapter.send("conv-id", "Hello")
        assert result.success is False
        assert result.retryable is False

    @pytest.mark.anyio
    async def test_send_success_is_not_retryable(self):
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_result = MagicMock()
        mock_result.id = "ok-id"
        mock_app = MagicMock()
        mock_app.send = AsyncMock(return_value=mock_result)
        adapter._app = mock_app

        result = await adapter.send("conv-id", "Hello")
        assert result.success is True
        assert result.retryable is False


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
        # Both the image branch and the document branch download via
        # _fetch_attachment_bytes; the document branch is the only one that
        # keeps cache_media_bytes' real classification (no per-call mock), so
        # mock both downloads to harmless bytes.
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"%PDF-1.4 fake")
        # Inline Teams images need the bot bearer token; with no live host the
        # adapter would otherwise try to mint one — stub the auth header.
        adapter._bot_auth_header = AsyncMock(return_value={})

        # The image branch now caches raw bytes via cache_media_bytes(...,
        # default_kind="image"); the document branch caches by filename/mime.
        # Return the right kind for each so classification (document wins) is
        # exercised the same way production sees it.
        def fake_cache_media_bytes(data, *, filename="", mime_type="", default_kind=None):
            if default_kind == "image" or mime_type.startswith("image/"):
                return SimpleNamespace(
                    path="/tmp/img.png", media_type="image/png", kind="image"
                )
            return SimpleNamespace(
                path="/tmp/report.pdf", media_type="application/pdf", kind="document"
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_teams_mod, "cache_media_bytes", fake_cache_media_bytes)
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
        # Inline images are fetched with the scoped bot bearer header and then
        # cached as bytes (parity with the file/document branches). Mock at
        # those boundaries: auth header, byte download, and the cache.
        adapter._bot_auth_header = AsyncMock(return_value={})
        adapter._fetch_attachment_bytes = AsyncMock(return_value=b"\x89PNG fake")

        def fake_cache_media_bytes(data, *, filename="", mime_type="", default_kind=None):
            return SimpleNamespace(
                path="/tmp/img.png", media_type="image/png", kind="image"
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_teams_mod, "cache_media_bytes", fake_cache_media_bytes)
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


# ---------------------------------------------------------------------------
# Tests: connect() success path
# ---------------------------------------------------------------------------

class TestTeamsConnectSuccess:
    @pytest.mark.anyio
    async def test_connect_success_marks_connected_and_registers_webhook(self, monkeypatch):
        """connect() should build the aiohttp app, start the SDK + runner/site,
        register the webhook route, and mark the adapter connected."""
        monkeypatch.setattr(_teams_mod, "TEAMS_SDK_AVAILABLE", True)
        monkeypatch.setattr(_teams_mod, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr(_teams_mod, "check_teams_requirements", lambda: True)

        # Mock the aiohttp ``web`` module so no real socket is bound. The same
        # Application instance is returned on each web.Application() call
        # (MagicMock return_value identity), so route wiring is assertable.
        fake_web = MagicMock()
        runner = AsyncMock()
        site = AsyncMock()
        fake_web.AppRunner.return_value = runner
        fake_web.TCPSite.return_value = site
        monkeypatch.setattr(_teams_mod, "web", fake_web)

        # Fake App whose initialize() drives the bridge's register_route the
        # way the real SDK http server does (POST /api/messages).
        captured = {}

        class _FakeApp:
            def __init__(self, **kwargs):
                self._adapter = kwargs.get("http_server_adapter")
                captured["app"] = self
                captured["kwargs"] = kwargs

            def on_message(self, func):
                return func

            def on_card_action(self, func):
                return func

            async def initialize(self):
                self._adapter.register_route(
                    "POST", "/api/messages", AsyncMock()
                )

        monkeypatch.setattr(_teams_mod, "App", _FakeApp)
        # ClientOptions is consumed by App(**kwargs); keep it a harmless factory.
        monkeypatch.setattr(_teams_mod, "ClientOptions", MagicMock())

        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant", port=3978,
        ))

        result = await adapter.connect()

        assert result is True
        assert adapter.is_connected is True
        assert adapter._running is True
        assert adapter._app is captured["app"]

        # Runner + site lifecycle was driven.
        fake_web.AppRunner.assert_called_once()
        runner.setup.assert_awaited_once()
        fake_web.TCPSite.assert_called_once()
        site.start.assert_awaited_once()

        # The SDK webhook route was registered onto the aiohttp app via the bridge.
        app_obj = fake_web.Application.return_value
        registered = [c.args for c in app_obj.router.add_route.call_args_list]
        assert ("POST", "/api/messages") == registered[0][:2]

    @pytest.mark.anyio
    async def test_connect_failure_sets_retryable_fatal_error(self, monkeypatch):
        """An exception during setup should be caught and reported as a
        retryable CONNECT_FAILED fatal error (not propagate)."""
        monkeypatch.setattr(_teams_mod, "TEAMS_SDK_AVAILABLE", True)
        monkeypatch.setattr(_teams_mod, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr(_teams_mod, "check_teams_requirements", lambda: True)

        fake_web = MagicMock()
        fake_web.Application.side_effect = RuntimeError("boom")
        monkeypatch.setattr(_teams_mod, "web", fake_web)

        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        result = await adapter.connect()
        assert result is False
        assert adapter.is_connected is False


# ---------------------------------------------------------------------------
# Tests: send_image
# ---------------------------------------------------------------------------

class TestTeamsSendImage:
    @pytest.mark.anyio
    async def test_send_image_returns_error_without_app(self):
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = None
        result = await adapter.send_image("conv-id", "https://example.test/x.png")
        assert result.success is False
        assert "not initialized" in result.error

    @pytest.mark.anyio
    async def test_send_image_url_uses_app_send(self):
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_result = MagicMock()
        mock_result.id = "img-msg-1"
        mock_app = MagicMock()
        mock_app.send = AsyncMock(return_value=mock_result)
        adapter._app = mock_app

        result = await adapter.send_image(
            "conv-id", "https://example.test/pic.png", caption="hi"
        )

        assert result.success is True
        assert result.message_id == "img-msg-1"
        # No cached conversation reference → flat app.send path.
        mock_app.send.assert_awaited_once()
        assert mock_app.send.await_args.args[0] == "conv-id"

    @pytest.mark.anyio
    async def test_send_image_local_path_encodes_data_uri(self, tmp_path, monkeypatch):
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_result = MagicMock()
        mock_result.id = "img-msg-2"
        mock_app = MagicMock()
        mock_app.send = AsyncMock(return_value=mock_result)
        adapter._app = mock_app

        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")

        # Capture the Attachment constructed for the local-path branch.
        captured = {}

        def _fake_attachment(content_type=None, content_url=None, **kw):
            captured["content_type"] = content_type
            captured["content_url"] = content_url
            return MagicMock()

        import microsoft_teams.api as _api_mod
        monkeypatch.setattr(_api_mod, "Attachment", _fake_attachment, raising=False)

        result = await adapter.send_image("conv-id", str(img))

        assert result.success is True
        assert result.message_id == "img-msg-2"
        assert captured["content_type"] == "image/png"
        assert captured["content_url"].startswith("data:image/png;base64,")
        # Round-trip the encoded bytes to be sure the file was embedded.
        b64 = captured["content_url"].split(",", 1)[1]
        assert base64.b64decode(b64) == b"\x89PNG\r\n\x1a\nFAKE"

    @pytest.mark.anyio
    async def test_send_image_uses_conversation_reference_when_cached(self):
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_result = MagicMock()
        mock_result.id = "img-msg-3"
        mock_app = MagicMock()
        mock_app.send = AsyncMock(return_value=mock_result)
        mock_app.activity_sender = MagicMock()
        mock_app.activity_sender.send = AsyncMock(return_value=mock_result)
        adapter._app = mock_app
        adapter._conv_refs["conv-id"] = MagicMock()  # cached reference

        result = await adapter.send_image("conv-id", "https://example.test/pic.png")

        assert result.success is True
        mock_app.activity_sender.send.assert_awaited_once()
        mock_app.send.assert_not_awaited()

    @pytest.mark.anyio
    async def test_send_image_handles_error(self):
        # send_image swallows the exception and reports a transient transport
        # failure as retryable.
        import httpx

        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        mock_app = MagicMock()
        mock_app.send = AsyncMock(
            side_effect=httpx.ConnectError("upload failed")
        )
        adapter._app = mock_app

        result = await adapter.send_image("conv-id", "https://example.test/pic.png")
        assert result.success is False
        assert "upload failed" in result.error
        assert result.retryable is True


# ---------------------------------------------------------------------------
# Tests: get_chat_info
# ---------------------------------------------------------------------------

class TestTeamsGetChatInfo:
    @pytest.mark.anyio
    async def test_get_chat_info_unseen_chat_defaults_to_dm(self):
        # Teams has no on-demand lookup; an unseen chat must fall back to a
        # valid enum type ("dm") and name=chat_id — never "unknown".
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        info = await adapter.get_chat_info("19:abc@thread.v2")
        assert info == {
            "name": "19:abc@thread.v2",
            "type": "dm",
            "chat_id": "19:abc@thread.v2",
        }
        assert info["type"] != "unknown"

    @pytest.mark.anyio
    async def test_get_chat_info_reflects_cached_meta(self):
        # get_chat_info replays the type/name observed on inbound messages.
        adapter = TeamsAdapter(_make_config(
            client_id="id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._chat_meta["19:room@thread.v2"] = {
            "name": "Project Room",
            "type": "channel",
        }
        info = await adapter.get_chat_info("19:room@thread.v2")
        assert info == {
            "name": "Project Room",
            "type": "channel",
            "chat_id": "19:room@thread.v2",
        }


# ---------------------------------------------------------------------------
# Tests: Adaptive Card action authorization gate (_on_card_action)
#
# This is the most security-sensitive path in the adapter: clicking an
# approval button resolves a *gateway command approval*. The handler must
# default-deny (require TEAMS_ALLOWED_USERS or explicit TEAMS_ALLOW_ALL_USERS)
# and must never resolve an approval for an unauthorized clicker.
# ---------------------------------------------------------------------------

class TestTeamsCardActionAuth:
    def _make_adapter(self):
        adapter = TeamsAdapter(_make_config(
            client_id="bot-id", client_secret="secret", tenant_id="tenant",
        ))
        adapter._app = MagicMock()
        adapter._app.id = "bot-id"
        return adapter

    def _make_ctx(self, *, clicker_aad="clicker-aad", clicker_id="clicker-id",
                  hermes_action="approve_once", session_key="sess-1"):
        ctx = MagicMock()
        ctx.activity.value.action.data = {
            "hermes_action": hermes_action,
            "session_key": session_key,
        }
        ctx.activity.from_ = MagicMock()
        ctx.activity.from_.aad_object_id = clicker_aad
        ctx.activity.from_.id = clicker_id
        return ctx

    def _patch_approval(self, monkeypatch, *, blocking=True):
        """Patch the approval helpers _on_card_action imports lazily.

        Returns the resolve spy; has_blocking_approval is stubbed so the
        accept paths reach the resolve call without a real pending request.
        """
        resolve_spy = MagicMock()
        monkeypatch.setattr("tools.approval.resolve_gateway_approval", resolve_spy)
        monkeypatch.setattr("tools.approval.has_blocking_approval", lambda _k: blocking)
        return resolve_spy

    @pytest.mark.anyio
    async def test_default_deny_when_no_env_set(self, monkeypatch):
        monkeypatch.delenv("TEAMS_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("TEAMS_ALLOW_ALL_USERS", raising=False)
        resolve_spy = self._patch_approval(monkeypatch)

        adapter = self._make_adapter()
        resp = await adapter._on_card_action(self._make_ctx())

        assert resp.status == 200
        resolve_spy.assert_not_called()

    @pytest.mark.anyio
    async def test_reject_clicker_not_in_allowlist(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOWED_USERS", "alice-aad,bob-aad")
        monkeypatch.delenv("TEAMS_ALLOW_ALL_USERS", raising=False)
        resolve_spy = self._patch_approval(monkeypatch)

        adapter = self._make_adapter()
        ctx = self._make_ctx(clicker_aad="mallory-aad", clicker_id="mallory-id")
        resp = await adapter._on_card_action(ctx)

        assert resp.status == 200
        resolve_spy.assert_not_called()

    @pytest.mark.anyio
    async def test_accept_clicker_in_allowlist(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOWED_USERS", "alice-aad,bob-aad")
        monkeypatch.delenv("TEAMS_ALLOW_ALL_USERS", raising=False)
        resolve_spy = self._patch_approval(monkeypatch)

        adapter = self._make_adapter()
        ctx = self._make_ctx(clicker_aad="bob-aad", hermes_action="approve_once")
        resp = await adapter._on_card_action(ctx)

        assert resp.status == 200
        resolve_spy.assert_called_once_with("sess-1", "once")

    @pytest.mark.anyio
    async def test_accept_wildcard_allowlist(self, monkeypatch):
        monkeypatch.setenv("TEAMS_ALLOWED_USERS", "*")
        monkeypatch.delenv("TEAMS_ALLOW_ALL_USERS", raising=False)
        resolve_spy = self._patch_approval(monkeypatch)

        adapter = self._make_adapter()
        ctx = self._make_ctx(clicker_aad="anyone-aad", hermes_action="deny")
        resp = await adapter._on_card_action(ctx)

        assert resp.status == 200
        resolve_spy.assert_called_once_with("sess-1", "deny")

    @pytest.mark.anyio
    async def test_accept_allow_all_users_env(self, monkeypatch):
        monkeypatch.delenv("TEAMS_ALLOWED_USERS", raising=False)
        monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
        resolve_spy = self._patch_approval(monkeypatch)

        adapter = self._make_adapter()
        ctx = self._make_ctx(clicker_aad="random-aad", hermes_action="approve_session")
        resp = await adapter._on_card_action(ctx)

        assert resp.status == 200
        resolve_spy.assert_called_once_with("sess-1", "session")
