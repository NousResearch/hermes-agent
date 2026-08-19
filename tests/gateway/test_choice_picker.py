"""Tests for the gateway interactive choice picker (/reasoning, /fast).

The picker mirrors the /model picker architecture: the gateway gates on the
adapter *type* exposing ``send_choice_picker``, sends a flat choice list, and
falls back to the text status card when the platform has no picker or the
send fails. Selection flows through the same application path as the typed
command, so picker and typed arguments can never diverge.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource


def _make_event(text="/reasoning", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


class _PickerAdapter:
    """Adapter whose *type* exposes ``send_choice_picker`` (the gate the
    handler checks via ``getattr(type(adapter), 'send_choice_picker', None)``)."""

    def __init__(self, success=True):
        self.calls = []
        self._success = success

    async def send_choice_picker(self, **kwargs):
        self.calls.append(kwargs)
        return SendResult(success=self._success, message_id="m1")


class _NoPickerAdapter:
    """Adapter with no choice-picker capability."""


def _make_runner(adapter=None):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._adapter_for_source = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._reply_anchor_for_event = lambda event: None
    return runner


class TestReasoningChoicePicker:
    @pytest.mark.asyncio
    async def test_bare_reasoning_sends_picker_when_adapter_supports_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)

        result = await runner._handle_reasoning_command(_make_event("/reasoning"))

        assert result is None  # picker sent — adapter owns the response
        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        values = [c["value"] for c in call["choices"]]
        # Full canonical ladder + none + subcommands, in order
        from hermes_constants import VALID_REASONING_EFFORTS
        assert values[0] == "none"
        assert values[1:1 + len(VALID_REASONING_EFFORTS)] == list(VALID_REASONING_EFFORTS)
        assert values[-3:] == ["reset", "show", "hide"]


    @pytest.mark.asyncio
    async def test_picker_selection_applies_same_as_typed(self, tmp_path, monkeypatch):
        """The picker's on_choice_selected must produce the identical state
        change as typing the argument (single application path)."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        event = _make_event("/reasoning")
        session_key = runner._session_key_for_source(event.source)

        await runner._handle_reasoning_command(event)
        on_choice = adapter.calls[0]["on_choice_selected"]

        reply = await on_choice(event.source.chat_id, "ultra")

        assert "ultra" in reply
        override = runner._session_reasoning_overrides.get(session_key)
        assert override == {"enabled": True, "effort": "ultra"}


class TestFastChoicePicker:
    def _patch_fast_support(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda cfg: "gpt-5.6")
        import hermes_cli.models as models_mod
        monkeypatch.setattr(models_mod, "model_supports_fast_mode", lambda m: True)

    @pytest.mark.asyncio
    async def test_bare_fast_sends_picker_when_adapter_supports_it(self, tmp_path, monkeypatch):
        self._patch_fast_support(monkeypatch, tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)

        result = await runner._handle_fast_command(_make_event("/fast"))

        assert result is None
        values = [c["value"] for c in adapter.calls[0]["choices"]]
        assert values == ["fast", "normal"]

    @pytest.mark.asyncio
    async def test_fast_picker_selection_is_session_scoped(self, tmp_path, monkeypatch):
        """A bare /fast picker tap applies a session override, not a config write."""
        self._patch_fast_support(monkeypatch, tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        event = _make_event("/fast")

        await runner._handle_fast_command(event)
        on_choice = adapter.calls[0]["on_choice_selected"]
        await on_choice(event.source.chat_id, "fast")

        assert runner._service_tier == "priority"
        assert runner._session_service_tier_overrides
        assert not (tmp_path / "config.yaml").exists()


class TestTelegramChoicePickerLayout:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("metadata", "expected_row_lengths"),
        [
            ({}, [2]),
            ({"choice_layout": "vertical"}, [1, 1]),
        ],
    )
    async def test_vertical_hint_renders_one_button_per_row(
        self, metadata, expected_row_lengths, monkeypatch
    ):
        from plugins.platforms.telegram import adapter as telegram_adapter
        from plugins.platforms.telegram.adapter import TelegramAdapter

        class _Markup:
            def __init__(self, rows):
                self.inline_keyboard = rows

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _Markup)

        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter._bot = object()
        adapter._reply_to_mode = "off"
        adapter._choice_picker_state = {}
        adapter.format_message = lambda text: text
        adapter._reply_to_message_id_for_send = lambda *_args, **_kwargs: None
        adapter._thread_kwargs_for_send = lambda *_args, **_kwargs: {}
        adapter._link_preview_kwargs = lambda: {}
        adapter._send_message_with_thread_fallback = AsyncMock(
            return_value=SimpleNamespace(message_id=42)
        )

        result = await adapter.send_choice_picker(
            chat_id="123",
            title="Choose",
            choices=[
                {"value": "continue", "label": "✅ Continue"},
                {"value": "defaults", "label": "↩️ Defaults"},
            ],
            session_key="session-key",
            on_choice_selected=AsyncMock(),
            metadata=metadata,
        )

        assert result.success is True
        markup = adapter._send_message_with_thread_fallback.await_args.kwargs[
            "reply_markup"
        ]
        assert [len(row) for row in markup.inline_keyboard] == expected_row_lengths

    @pytest.mark.asyncio
    async def test_shared_session_pickers_allow_distinct_authorized_actors(
        self, monkeypatch
    ):
        from plugins.platforms.telegram import adapter as telegram_adapter
        from plugins.platforms.telegram.adapter import TelegramAdapter

        class _Markup:
            def __init__(self, rows):
                self.inline_keyboard = rows

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _Markup)

        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter._bot = object()
        adapter._reply_to_mode = "off"
        adapter._choice_picker_state = {}
        adapter._choice_picker_cleanup_tasks = set()
        adapter._is_callback_user_authorized = lambda *_args, **_kwargs: True
        adapter.format_message = lambda text: text
        adapter._reply_to_message_id_for_send = lambda *_args, **_kwargs: None
        adapter._thread_kwargs_for_send = lambda *_args, **_kwargs: {
            "message_thread_id": 10
        }
        adapter._link_preview_kwargs = lambda: {}
        adapter._send_message_with_thread_fallback = AsyncMock(
            side_effect=[
                SimpleNamespace(message_id=41, message_thread_id=10),
                SimpleNamespace(message_id=42, message_thread_id=10),
            ]
        )

        picker_one = AsyncMock(return_value="user one selected")
        picker_two = AsyncMock(return_value="user two selected")
        for session_key, callback in (
            ("session-user-1", picker_one),
            ("session-user-2", picker_two),
        ):
            result = await adapter.send_choice_picker(
                chat_id="123",
                title="Choose",
                choices=[{"value": "continue", "label": "Continue"}],
                session_key=session_key,
                on_choice_selected=callback,
                metadata={"thread_id": "10"},
            )
            assert result.success is True

        assert set(adapter._choice_picker_state) == {
            ("123", "41"),
            ("123", "42"),
        }

        def _query(message_id, user_id):
            return SimpleNamespace(
                message=SimpleNamespace(
                    message_id=message_id,
                    message_thread_id=10,
                    chat_id=123,
                    chat=SimpleNamespace(type="supergroup"),
                ),
                from_user=SimpleNamespace(id=user_id, first_name="User"),
                edit_message_text=AsyncMock(),
                answer=AsyncMock(),
            )

        await adapter._handle_choice_picker_callback(_query(41, 7), "cp:0", "123")

        picker_one.assert_awaited_once_with("123", "continue")
        picker_two.assert_not_awaited()
        assert ("123", "41") not in adapter._choice_picker_state
        assert ("123", "42") in adapter._choice_picker_state

        await adapter._handle_choice_picker_callback(_query(42, 8), "cp:0", "123")

        picker_two.assert_awaited_once_with("123", "continue")
        assert adapter._choice_picker_state == {}

    @pytest.mark.asyncio
    async def test_owned_picker_denies_other_authorized_actor_without_side_effects(
        self, monkeypatch
    ):
        from plugins.platforms.telegram import adapter as telegram_adapter
        from plugins.platforms.telegram.adapter import TelegramAdapter

        class _Markup:
            def __init__(self, rows):
                self.inline_keyboard = rows

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _Markup)
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter._bot = object()
        adapter._reply_to_mode = "off"
        adapter._choice_picker_state = {}
        adapter._choice_picker_cleanup_tasks = set()
        # Both actors pass the chat-level authorization gate. Picker ownership
        # is the additional per-user-session boundary under test.
        adapter._is_callback_user_authorized = lambda *_args, **_kwargs: True
        adapter.format_message = lambda text: text
        adapter._reply_to_message_id_for_send = lambda *_args, **_kwargs: None
        adapter._thread_kwargs_for_send = lambda *_args, **_kwargs: {}
        adapter._link_preview_kwargs = lambda: {}
        adapter._send_message_with_thread_fallback = AsyncMock(
            return_value=SimpleNamespace(message_id=42)
        )
        mutate_override_and_replay_held_message = AsyncMock(
            return_value="owner selected"
        )

        result = await adapter.send_choice_picker(
            chat_id="123",
            title="Choose",
            choices=[{"value": "defaults", "label": "Defaults"}],
            session_key="session-user-7",
            on_choice_selected=mutate_override_and_replay_held_message,
            metadata={"requester_user_id": "7"},
        )
        assert result.success is True

        def _query(user_id):
            return SimpleNamespace(
                message=SimpleNamespace(
                    message_id=42,
                    message_thread_id=None,
                    chat_id=123,
                    chat=SimpleNamespace(type="supergroup"),
                ),
                from_user=SimpleNamespace(id=user_id, first_name="User"),
                edit_message_text=AsyncMock(),
                answer=AsyncMock(),
            )

        other_actor = _query(8)
        await adapter._handle_choice_picker_callback(
            other_actor, "cp:0", "123"
        )

        other_actor.answer.assert_awaited_once_with(
            text="⛔ Only the user who opened this picker can use it."
        )
        other_actor.edit_message_text.assert_not_awaited()
        mutate_override_and_replay_held_message.assert_not_awaited()
        assert ("123", "42") in adapter._choice_picker_state

        owner = _query(7)
        await adapter._handle_choice_picker_callback(owner, "cp:0", "123")

        mutate_override_and_replay_held_message.assert_awaited_once_with(
            "123", "defaults"
        )
        owner.edit_message_text.assert_awaited_once()
        assert ("123", "42") not in adapter._choice_picker_state

    @pytest.mark.asyncio
    async def test_owned_picker_missing_actor_identity_fails_closed(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = TelegramAdapter.__new__(TelegramAdapter)
        callback = AsyncMock(return_value="must not run")
        adapter._choice_picker_state = {
            ("123", "42"): {
                "choices": [{"value": "continue"}],
                "owner_user_id": "7",
                "on_choice_selected": callback,
            }
        }
        adapter._is_callback_user_authorized = lambda *_args, **_kwargs: True
        query = SimpleNamespace(
            message=SimpleNamespace(
                message_id=42,
                message_thread_id=None,
                chat_id=123,
                chat=SimpleNamespace(type="supergroup"),
            ),
            from_user=SimpleNamespace(id=None, first_name=None),
            answer=AsyncMock(),
        )

        await adapter._handle_choice_picker_callback(query, "cp:0", "123")

        query.answer.assert_awaited_once_with(
            text="⛔ Only the user who opened this picker can use it."
        )
        callback.assert_not_awaited()
        assert ("123", "42") in adapter._choice_picker_state

    @pytest.mark.asyncio
    async def test_timed_picker_cleans_up_only_its_own_message_state(self, monkeypatch):
        from plugins.platforms.telegram import adapter as telegram_adapter
        from plugins.platforms.telegram.adapter import TelegramAdapter

        class _Markup:
            def __init__(self, rows):
                self.inline_keyboard = rows

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _Markup)
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter._bot = object()
        adapter._reply_to_mode = "off"
        adapter._choice_picker_state = {
            ("123", "41"): {"existing": True},
        }
        adapter._choice_picker_cleanup_tasks = set()
        adapter.format_message = lambda text: text
        adapter._reply_to_message_id_for_send = lambda *_args, **_kwargs: None
        adapter._thread_kwargs_for_send = lambda *_args, **_kwargs: {}
        adapter._link_preview_kwargs = lambda: {}
        adapter._send_message_with_thread_fallback = AsyncMock(
            return_value=SimpleNamespace(message_id=42)
        )

        result = await adapter.send_choice_picker(
            chat_id="123",
            title="Choose",
            choices=[{"value": "continue", "label": "Continue"}],
            session_key="session-key",
            on_choice_selected=AsyncMock(),
            metadata={"choice_timeout_seconds": 0},
        )
        assert result.success is True
        await asyncio.gather(*adapter._choice_picker_cleanup_tasks)

        assert ("123", "41") in adapter._choice_picker_state
        assert ("123", "42") not in adapter._choice_picker_state


class TestDiscordChoicePickerOwnership:
    @staticmethod
    def _interaction(user_id):
        return SimpleNamespace(
            user=(
                None
                if user_id is None
                else SimpleNamespace(id=user_id)
            ),
            channel_id=123,
            data={"values": ["defaults"]},
            response=SimpleNamespace(
                send_message=AsyncMock(),
                edit_message=AsyncMock(),
                defer=AsyncMock(),
            ),
        )

    @pytest.mark.asyncio
    async def test_owned_picker_denials_are_inert_then_owner_can_resolve(
        self, monkeypatch
    ):
        discord = pytest.importorskip("discord")
        from plugins.platforms.discord import adapter as discord_adapter
        from plugins.platforms.discord.adapter import DiscordAdapter

        if not isinstance(discord_adapter.ChoicePickerView, type):
            pytest.skip("e2e collection installed a synthetic discord module")
        monkeypatch.setattr(discord_adapter, "DISCORD_AVAILABLE", True)
        sent_views = []

        async def _send(**kwargs):
            sent_views.append(kwargs["view"])
            return SimpleNamespace(id=len(sent_views))

        channel = SimpleNamespace(send=_send)
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter._client = SimpleNamespace(
            get_channel=lambda _channel_id: channel,
            fetch_channel=AsyncMock(return_value=channel),
        )
        adapter._allowed_user_ids = {"7", "8"}
        adapter._allowed_role_ids = set()
        callback = AsyncMock(return_value="owner selected")

        result = await adapter.send_choice_picker(
            chat_id="123",
            title="Choose",
            choices=[{"value": "defaults", "label": "Defaults"}],
            session_key="session-user-7",
            on_choice_selected=callback,
            metadata={"requester_user_id": " 7 "},
        )
        assert result.success is True
        view = sent_views[-1]
        view._check_auth = MagicMock(return_value=True)
        view.stop = MagicMock()

        other = self._interaction(8)
        await view._on_select(other)
        other.response.send_message.assert_awaited_once_with(
            "⛔ Only the user who opened this picker can use it.",
            ephemeral=True,
        )
        other.response.edit_message.assert_not_awaited()
        callback.assert_not_awaited()
        assert view.resolved is False

        missing = self._interaction(None)
        await view._on_select(missing)
        missing.response.send_message.assert_awaited_once_with(
            "⛔ Only the user who opened this picker can use it.",
            ephemeral=True,
        )
        missing.response.edit_message.assert_not_awaited()
        callback.assert_not_awaited()
        assert view.resolved is False

        owner = self._interaction(7)
        await view._on_select(owner)
        callback.assert_awaited_once_with("123", "defaults")
        owner.response.edit_message.assert_awaited_once()
        assert view.resolved is True
        assert isinstance(owner.response.edit_message.await_args.kwargs["embed"], discord.Embed)

    @pytest.mark.asyncio
    async def test_metadata_absent_preserves_shared_authorized_actor_behavior(
        self, monkeypatch
    ):
        pytest.importorskip("discord")
        from plugins.platforms.discord import adapter as discord_adapter
        from plugins.platforms.discord.adapter import DiscordAdapter

        if not isinstance(discord_adapter.ChoicePickerView, type):
            pytest.skip("e2e collection installed a synthetic discord module")
        monkeypatch.setattr(discord_adapter, "DISCORD_AVAILABLE", True)
        sent = {}

        async def _send(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(id=1)

        channel = SimpleNamespace(send=_send)
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter._client = SimpleNamespace(
            get_channel=lambda _channel_id: channel,
            fetch_channel=AsyncMock(return_value=channel),
        )
        adapter._allowed_user_ids = {"8"}
        adapter._allowed_role_ids = set()
        callback = AsyncMock(return_value="shared selection")

        await adapter.send_choice_picker(
            chat_id="123",
            title="Choose",
            choices=[{"value": "defaults", "label": "Defaults"}],
            session_key="shared-session",
            on_choice_selected=callback,
            metadata={},
        )
        view = sent["view"]
        assert view.requester_user_id is None
        view._check_auth = MagicMock(return_value=True)
        view.stop = MagicMock()
        actor = self._interaction(8)
        await view._on_select(actor)

        callback.assert_awaited_once_with("123", "defaults")
        actor.response.edit_message.assert_awaited_once()


