"""Tests for the Slack Block Kit interactive model picker.

Mirrors test_slack_approval_buttons.py (harness) and
test_discord_model_picker.py (semantics) for the ``send_model_picker``
override and the ``hermes_model_provider`` / ``hermes_model_model`` /
``hermes_model_back`` / ``hermes_model_cancel`` action dispatch.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Slack SDK mock so SlackAdapter can be imported (mirrors
# test_slack_approval_buttons.py)
# ---------------------------------------------------------------------------
def _ensure_slack_mock():
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

from plugins.platforms.slack.adapter import SlackAdapter
from gateway.config import PlatformConfig, Platform


def _make_adapter():
    """Create a SlackAdapter instance with mocked internals."""
    config = PlatformConfig(enabled=True, token="«redacted:xox…»")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    return adapter


class _AuthRunner:
    def __init__(self, auth_fn=None):
        self._auth_fn = auth_fn or (lambda _source: True)
        self.seen_sources = []

    async def handle(self, event):
        return None

    def _is_user_authorized(self, source):
        self.seen_sources.append(source)
        return self._auth_fn(source)


def _attach_auth_runner(adapter, auth_fn=None):
    runner = _AuthRunner(auth_fn=auth_fn)
    adapter.set_message_handler(runner.handle)
    return runner


_PROVIDERS = [
    {
        "slug": "openrouter",
        "name": "OpenRouter",
        "models": ["anthropic/claude-sonnet-4", "openai/gpt-5"],
        "total_models": 2,
        "is_current": True,
    },
    {
        "slug": "anthropic",
        "name": "Anthropic",
        "models": ["claude-sonnet-4-20250514"],
        "total_models": 1,
        "is_current": False,
    },
]


def _interaction_body(msg_ts="1234.5678", channel_id="C1", user="alice", uid="U_ALICE"):
    return {
        "message": {"ts": msg_ts, "blocks": []},
        "channel": {"id": channel_id},
        "user": {"name": user, "id": uid},
        "team_id": "T1",
    }


# ===========================================================================
# send_model_picker — Block Kit structure
# ===========================================================================

class TestSlackModelPickerSend:
    """Test send_model_picker sends the provider-stage Block Kit."""

    @pytest.mark.asyncio
    async def test_sends_provider_dropdown(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1234.5678"})

        async def on_selected(chat_id, model_id, provider_slug):
            return "Switched!"

        result = await adapter.send_model_picker(
            chat_id="C1",
            providers=_PROVIDERS,
            current_model="anthropic/claude-sonnet-4",
            current_provider="openrouter",
            session_key="test-session",
            on_model_selected=on_selected,
        )

        assert result.success is True
        assert result.message_id == "1234.5678"

        kwargs = mock_client.chat_postMessage.call_args[1]
        assert "blocks" in kwargs
        blocks = kwargs["blocks"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "section"
        assert "anthropic/claude-sonnet-4" in blocks[0]["text"]["text"]
        assert blocks[1]["type"] == "actions"

        elements = blocks[1]["elements"]
        assert len(elements) == 2
        assert elements[0]["type"] == "static_select"
        assert elements[0]["action_id"] == "hermes_model_provider"
        assert len(elements[0]["options"]) == 2
        assert elements[0]["options"][0]["value"] == "openrouter"
        assert elements[1]["type"] == "button"
        assert elements[1]["action_id"] == "hermes_model_cancel"

        # State should be stashed for the callback (no metadata → bare ts key)
        assert "1234.5678" in adapter._model_picker_state
        state = adapter._model_picker_state["1234.5678"]
        assert state["stage"] == "provider"
        assert state["session_key"] == "test-session"

    @pytest.mark.asyncio
    async def test_sends_in_thread(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1.2"})

        async def on_selected(chat_id, model_id, provider_slug):
            return "ok"

        await adapter.send_model_picker(
            chat_id="C1",
            providers=_PROVIDERS,
            current_model="m",
            current_provider="openrouter",
            session_key="s",
            on_model_selected=on_selected,
            metadata={"thread_id": "9999.0000"},
        )

        kwargs = mock_client.chat_postMessage.call_args[1]
        assert kwargs.get("thread_ts") == "9999.0000"

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._app = None
        result = await adapter.send_model_picker(
            chat_id="C1", providers=[], current_model="", current_provider="",
            session_key="s", on_model_selected=AsyncMock(),
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_providers(self):
        adapter = _make_adapter()
        result = await adapter.send_model_picker(
            chat_id="C1", providers=[], current_model="", current_provider="",
            session_key="s", on_model_selected=AsyncMock(),
        )
        assert result.success is False


# ===========================================================================
# _handle_model_picker_action — drill-down dispatch
# ===========================================================================

class TestSlackModelPickerAction:
    """Test the model picker action handler."""

    def _seed_state(self, adapter, msg_ts="1234.5678", on_selected=None, stage="provider",
                    selected_provider="", bare_ts=False, providers=None):
        # bare_ts=True mirrors the metadata-poor send path: no team id, so the
        # state is keyed by the raw ts instead of the (team_id, ts) marker.
        key = msg_ts if bare_ts else ("T1", msg_ts)
        adapter._model_picker_state[key] = {
            "providers": _PROVIDERS if providers is None else providers,
            "session_key": "s1",
            "chat_id": "C1",
            "team_id": "" if bare_ts else "T1",
            "current_model": "old",
            "current_provider": "openrouter",
            "on_model_selected": on_selected or AsyncMock(return_value="ok"),
            "stage": stage,
            "selected_provider_slug": selected_provider,
        }

    @pytest.mark.asyncio
    async def test_provider_select_shows_models(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()
        self._seed_state(adapter)

        ack = AsyncMock()
        action = {
            "action_id": "hermes_model_provider",
            "selected_option": {"value": "openrouter"},
        }

        await adapter._handle_model_picker_action(ack, _interaction_body(), action)

        ack.assert_called_once()
        mock_client.chat_update.assert_called_once()
        update_kwargs = mock_client.chat_update.call_args[1]
        elements = update_kwargs["blocks"][1]["elements"]
        assert elements[0]["action_id"] == "hermes_model_model"
        # Two models for openrouter, values are indices
        assert [o["value"] for o in elements[0]["options"]] == ["0", "1"]
        # Back + Cancel buttons present
        assert elements[1]["action_id"] == "hermes_model_back"
        assert elements[2]["action_id"] == "hermes_model_cancel"

        state = adapter._model_picker_state[("T1", "1234.5678")]
        assert state["stage"] == "model"
        assert state["selected_provider_slug"] == "openrouter"

    @pytest.mark.asyncio
    async def test_bare_ts_state_resolves_on_team_scoped_click(self):
        """A team-scoped click must find state stored under the bare ts.

        The metadata-poor send path keys state by raw ts (no team id); the
        click event still carries one. The handler's dual-key lookup must not
        swallow that legitimate interaction.
        """
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()
        self._seed_state(adapter, bare_ts=True)

        ack = AsyncMock()
        action = {
            "action_id": "hermes_model_provider",
            "selected_option": {"value": "openrouter"},
        }

        await adapter._handle_model_picker_action(ack, _interaction_body(), action)

        ack.assert_called_once()
        mock_client.chat_update.assert_called_once()
        state = adapter._model_picker_state["1234.5678"]
        assert state["stage"] == "model"
        assert state["selected_provider_slug"] == "openrouter"

    @pytest.mark.asyncio
    async def test_provider_select_with_no_models_dismisses_picker(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()
        self._seed_state(
            adapter,
            providers=[{
                "slug": "empty", "name": "EmptyCo", "models": [],
                "total_models": 0, "is_current": False,
            }],
        )

        ack = AsyncMock()
        action = {
            "action_id": "hermes_model_provider",
            "selected_option": {"value": "empty"},
        }

        await adapter._handle_model_picker_action(ack, _interaction_body(), action)

        ack.assert_called_once()
        mock_client.chat_update.assert_called_once()
        assert "No models available" in mock_client.chat_update.call_args[1]["text"]
        assert ("T1", "1234.5678") not in adapter._model_picker_state

    @pytest.mark.asyncio
    async def test_model_select_calls_callback(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        events = []

        async def on_selected(chat_id, model_id, provider_slug):
            events.append((chat_id, model_id, provider_slug))
            return "✅ Switched to claude-sonnet-4"

        self._seed_state(
            adapter, stage="model", selected_provider="openrouter", on_selected=on_selected
        )

        ack = AsyncMock()
        action = {
            "action_id": "hermes_model_model",
            "selected_option": {"value": "0"},  # index 0 = anthropic/claude-sonnet-4
        }

        await adapter._handle_model_picker_action(ack, _interaction_body(), action)

        ack.assert_called_once()
        assert events == [("C1", "anthropic/claude-sonnet-4", "openrouter")]

        # State cleaned up
        assert ("T1", "1234.5678") not in adapter._model_picker_state

        # chat_update called twice: "Switching..." then the confirmation
        assert mock_client.chat_update.call_count == 2
        last_update = mock_client.chat_update.call_args_list[-1][1]
        assert "Switched to claude-sonnet-4" in last_update["text"]

    @pytest.mark.asyncio
    async def test_model_select_callback_error_is_reported(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        async def on_selected(chat_id, model_id, provider_slug):
            raise RuntimeError("boom")

        self._seed_state(
            adapter, stage="model", selected_provider="openrouter", on_selected=on_selected
        )

        ack = AsyncMock()
        action = {"action_id": "hermes_model_model", "selected_option": {"value": "1"}}

        await adapter._handle_model_picker_action(ack, _interaction_body(), action)

        last_update = mock_client.chat_update.call_args_list[-1][1]
        assert "Model switch failed" in last_update["text"]

    @pytest.mark.asyncio
    async def test_model_select_invalid_index_is_ignored(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()
        on_selected = AsyncMock()
        self._seed_state(
            adapter, stage="model", selected_provider="openrouter", on_selected=on_selected
        )

        ack = AsyncMock()
        action = {
            "action_id": "hermes_model_model",
            "selected_option": {"value": "bogus"},
        }

        await adapter._handle_model_picker_action(ack, _interaction_body(), action)

        ack.assert_called_once()
        on_selected.assert_not_called()
        mock_client.chat_update.assert_not_called()
        # State is left intact so a corrected re-pick still resolves.
        state = adapter._model_picker_state[("T1", "1234.5678")]
        assert state["stage"] == "model"

    @pytest.mark.asyncio
    async def test_cancel_clears_state(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()
        self._seed_state(adapter)

        ack = AsyncMock()
        action = {"action_id": "hermes_model_cancel", "value": "cancel"}

        await adapter._handle_model_picker_action(ack, _interaction_body(), action)

        ack.assert_called_once()
        assert ("T1", "1234.5678") not in adapter._model_picker_state
        mock_client.chat_update.assert_called_once()
        assert "cancelled" in mock_client.chat_update.call_args[1]["text"].lower()

    @pytest.mark.asyncio
    async def test_back_returns_to_provider(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()
        self._seed_state(adapter, stage="model", selected_provider="openrouter")

        ack = AsyncMock()
        action = {"action_id": "hermes_model_back", "value": "openrouter"}

        await adapter._handle_model_picker_action(ack, _interaction_body(), action)

        mock_client.chat_update.assert_called_once()
        update_kwargs = mock_client.chat_update.call_args[1]
        elements = update_kwargs["blocks"][1]["elements"]
        assert elements[0]["action_id"] == "hermes_model_provider"
        state = adapter._model_picker_state[("T1", "1234.5678")]
        assert state["stage"] == "provider"
        assert state["selected_provider_slug"] == ""

    @pytest.mark.asyncio
    async def test_state_not_found_silently_returns(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        ack = AsyncMock()
        action = {"action_id": "hermes_model_provider", "selected_option": {"value": "openrouter"}}

        await adapter._handle_model_picker_action(
            ack, _interaction_body(msg_ts="nonexistent"), action
        )
        ack.assert_called_once()
        # No crash

    @pytest.mark.asyncio
    async def test_unauthorized_user_blocked(self, monkeypatch):
        adapter = _make_adapter()
        _attach_auth_runner(adapter, auth_fn=lambda s: s.user_id == "U_OWNER")
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()
        self._seed_state(adapter)

        ack = AsyncMock()
        action = {"action_id": "hermes_model_provider", "selected_option": {"value": "openrouter"}}

        await adapter._handle_model_picker_action(
            ack, _interaction_body(user="intruder", uid="U_BAD"), action
        )

        ack.assert_called_once()
        mock_client.chat_update.assert_not_called()
        # State unchanged
        assert adapter._model_picker_state[("T1", "1234.5678")]["stage"] == "provider"


# ===========================================================================
# Gateway integration — routes /model to the picker
# ===========================================================================

class TestSlackModelPickerGatewayIntegration:
    """Verify _handle_model_command detects the Slack picker capability."""

    @pytest.mark.asyncio
    async def test_bare_model_triggers_picker(self, tmp_path, monkeypatch):
        import types

        import yaml

        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.session import SessionSource

        captured = {}

        class SlackPickerAdapter:
            async def send_model_picker(self, *, on_model_selected, **kwargs):
                captured["called"] = True
                captured["callback"] = on_model_selected
                return types.SimpleNamespace(success=True)

        adapter = SlackPickerAdapter()

        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter}
        runner._voice_mode = {}
        runner._session_model_overrides = {}
        runner._running_agents = {}

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        cfg_path = hermes_home / "config.yaml"
        cfg_path.write_text(
            yaml.safe_dump({
                "model": {"default": "old-model", "provider": "openrouter"},
                "providers": {},
            }),
            encoding="utf-8",
        )

        import gateway.run as gateway_run
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr(
            "hermes_cli.model_switch.list_picker_providers",
            lambda **kw: [{"slug": "openrouter", "name": "OR", "models": ["m1"], "total_models": 1}],
        )

        event = MessageEvent(
            text="/model",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.SLACK, chat_id="C1", chat_type="group", user_id="U1"
            ),
        )

        result = await runner._handle_model_command(event)

        assert result is None
        assert captured.get("called") is True

    @pytest.mark.asyncio
    async def test_text_fallback_when_no_picker(self, tmp_path, monkeypatch):
        import yaml

        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.session import SessionSource

        class SlackNoPickerAdapter:
            pass  # no send_model_picker

        adapter = SlackNoPickerAdapter()

        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter}
        runner._voice_mode = {}
        runner._session_model_overrides = {}
        runner._running_agents = {}

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        cfg_path = hermes_home / "config.yaml"
        cfg_path.write_text(
            yaml.safe_dump({
                "model": {"default": "old-model", "provider": "openrouter"},
                "providers": {},
            }),
            encoding="utf-8",
        )

        import gateway.run as gateway_run
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr(
            "hermes_cli.model_switch.list_authenticated_providers",
            lambda **kw: [],
        )

        event = MessageEvent(
            text="/model",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.SLACK, chat_id="C1", chat_type="group", user_id="U1"
            ),
        )

        result = await runner._handle_model_command(event)

        assert isinstance(result, str)
        assert "old-model" in result
