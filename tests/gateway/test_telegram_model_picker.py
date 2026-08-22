"""Tests for Telegram model picker thread fallback."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(*, platform_extra=None):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="test-token",
                extra=platform_extra or {},
            )
        }
    )
    runner._is_user_authorized = lambda _source: True
    adapter.set_message_handler(runner._handle_message)
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.TELEGRAM))
    adapter.set_slash_access_check(runner._check_slash_access)
    return adapter


def _query(
    *,
    user_id=111,
    data="mb",
    chat_type="group",
    chat_id=12345,
    message_id=42,
    thread_id=None,
):
    query = SimpleNamespace(
        data=data,
        message=SimpleNamespace(
            chat_id=chat_id,
            chat=SimpleNamespace(type=chat_type),
            message_id=message_id,
            message_thread_id=thread_id,
        ),
        from_user=(
            SimpleNamespace(
                id=user_id,
                username="alice",
                full_name="Alice Example",
                first_name="Alice",
            )
            if user_id is not None
            else None
        ),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )
    return query


async def _send_picker(
    adapter,
    *,
    message_id,
    thread_id,
    owner_user_id,
    session_key,
    selected,
    model_id,
):
    """Send a real adapter picker, then prime its model-selection page."""
    adapter._bot.send_message = AsyncMock(
        return_value=SimpleNamespace(
            message_id=message_id,
            message_thread_id=thread_id,
        )
    )
    result = await adapter.send_model_picker(
        chat_id="12345",
        providers=[
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "models": [model_id],
            }
        ],
        current_model="old",
        current_provider="openrouter",
        session_key=session_key,
        on_model_selected=selected,
        metadata={
            "thread_id": str(thread_id) if thread_id is not None else None,
            "picker_user_id": str(owner_user_id),
        },
    )
    assert result.success is True
    state = next(
        state
        for state in adapter._model_picker_state.values()
        if state["msg_id"] == message_id
    )
    state.update(
        selected_provider="openrouter",
        selected_provider_name="OpenRouter",
        model_list=[model_id],
        model_page=0,
    )
    return state


class TestTelegramModelPicker:
    @pytest.mark.asyncio
    async def test_send_model_picker_escapes_dynamic_provider_label(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_send_message(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=101)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_model_picker(
            chat_id="12345",
            providers=[
                {"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}
            ],
            current_model="model_1",
            current_provider="provider_one",
            session_key="s",
            on_model_selected=AsyncMock(),
            metadata={"thread_id": "99999"},
        )

        assert result.success is True
        assert "MARKDOWN_V2" in repr(sent["parse_mode"])
        assert "provider\\_one" in sent["text"]
        assert "`model_1`" in sent["text"]

    @pytest.mark.asyncio
    async def test_back_button_escapes_dynamic_provider_label(self):
        adapter = _make_adapter()
        adapter._model_picker_state[("12345", "42")] = {
            "providers": [{"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}],
            "current_model": "model_1",
            "current_provider": "provider_one",
            "session_key": "s",
            "on_model_selected": AsyncMock(),
            "msg_id": 42,
            "thread_id": None,
            "owner_user_id": "111",
        }

        query = _query()

        await adapter._handle_model_picker_callback(query, "mb", "12345")

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "provider\\_one" in edit_kwargs["text"]
        assert "`model_1`" in edit_kwargs["text"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("user_id", [222, None], ids=["non-admin", "missing-actor"])
    async def test_model_picker_callback_denies_before_navigation_or_selection(
        self, user_id
    ):
        adapter = _make_adapter(
            platform_extra={
                "group_allow_admin_from": ["111"],
                "group_user_allowed_commands": [],
            }
        )
        selected = AsyncMock(return_value="switched")
        adapter._model_picker_state[("12345", "42")] = {
            "providers": [
                {"slug": "openrouter", "name": "OpenRouter", "models": ["gpt-x"]}
            ],
            "current_model": "old",
            "current_provider": "openrouter",
            "session_key": "agent:main:telegram:group:12345",
            "on_model_selected": selected,
            "msg_id": 42,
            "thread_id": None,
            "owner_user_id": "111",
        }
        query = _query(user_id=user_id, data="mp:openrouter")

        await adapter._handle_callback_query(
            SimpleNamespace(callback_query=query), SimpleNamespace()
        )

        selected.assert_not_awaited()
        query.edit_message_text.assert_not_awaited()
        assert "not authorized" in query.answer.await_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_callback_message_id_mismatch_never_executes_picker_callback(
        self, monkeypatch
    ):
        adapter = _make_adapter()
        selected = AsyncMock(return_value="switched")
        await _send_picker(
            adapter,
            message_id=42,
            thread_id=77,
            owner_user_id=111,
            session_key="agent:main:telegram:thread:12345:77",
            selected=selected,
            model_id="model-a",
        )
        monkeypatch.setattr(
            "hermes_cli.model_selection_guards.combined_selection_warning",
            lambda *_args, **_kwargs: None,
        )
        query = _query(data="mm:0", message_id=99, thread_id=77)

        await adapter._handle_callback_query(
            SimpleNamespace(callback_query=query), SimpleNamespace()
        )

        selected.assert_not_awaited()
        query.edit_message_text.assert_not_awaited()
        assert "expired" in query.answer.await_args.kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_callback_topic_mismatch_never_executes_picker_callback(
        self, monkeypatch
    ):
        adapter = _make_adapter()
        selected = AsyncMock(return_value="switched")
        await _send_picker(
            adapter,
            message_id=42,
            thread_id=77,
            owner_user_id=111,
            session_key="agent:main:telegram:thread:12345:77",
            selected=selected,
            model_id="model-a",
        )
        monkeypatch.setattr(
            "hermes_cli.model_selection_guards.combined_selection_warning",
            lambda *_args, **_kwargs: None,
        )
        query = _query(data="mm:0", message_id=42, thread_id=88)

        await adapter._handle_callback_query(
            SimpleNamespace(callback_query=query), SimpleNamespace()
        )

        selected.assert_not_awaited()
        query.edit_message_text.assert_not_awaited()
        assert "expired" in query.answer.await_args.kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_different_authorized_actor_cannot_use_another_users_picker(
        self, monkeypatch
    ):
        adapter = _make_adapter(
            platform_extra={"group_allow_admin_from": ["111", "222"]}
        )
        selected = AsyncMock(return_value="switched")
        await _send_picker(
            adapter,
            message_id=42,
            thread_id=77,
            owner_user_id=111,
            session_key="agent:main:telegram:thread:12345:77",
            selected=selected,
            model_id="model-a",
        )
        monkeypatch.setattr(
            "hermes_cli.model_selection_guards.combined_selection_warning",
            lambda *_args, **_kwargs: None,
        )
        query = _query(user_id=222, data="mm:0", message_id=42, thread_id=77)

        await adapter._handle_callback_query(
            SimpleNamespace(callback_query=query), SimpleNamespace()
        )

        selected.assert_not_awaited()
        query.edit_message_text.assert_not_awaited()
        assert "not authorized" in query.answer.await_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_two_topics_in_one_chat_keep_callbacks_bound_to_their_picker(
        self, monkeypatch
    ):
        adapter = _make_adapter()
        selected_a = AsyncMock(return_value="switched-a")
        selected_b = AsyncMock(return_value="switched-b")
        await _send_picker(
            adapter,
            message_id=41,
            thread_id=10,
            owner_user_id=111,
            session_key="agent:main:telegram:thread:12345:10",
            selected=selected_a,
            model_id="model-a",
        )
        await _send_picker(
            adapter,
            message_id=42,
            thread_id=20,
            owner_user_id=111,
            session_key="agent:main:telegram:thread:12345:20",
            selected=selected_b,
            model_id="model-b",
        )
        monkeypatch.setattr(
            "hermes_cli.model_selection_guards.combined_selection_warning",
            lambda *_args, **_kwargs: None,
        )

        query_a = _query(data="mm:0", message_id=41, thread_id=10)
        await adapter._handle_callback_query(
            SimpleNamespace(callback_query=query_a), SimpleNamespace()
        )

        selected_a.assert_awaited_once_with("12345", "model-a", "openrouter")
        selected_b.assert_not_awaited()
        assert any(
            state["msg_id"] == 42
            for state in adapter._model_picker_state.values()
        )

    @pytest.mark.asyncio
    async def test_new_picker_expires_older_picker_for_the_same_session(
        self, monkeypatch
    ):
        adapter = _make_adapter()
        selected_old = AsyncMock(return_value="switched-old")
        selected_new = AsyncMock(return_value="switched-new")
        session_key = "agent:main:telegram:thread:12345:10"
        await _send_picker(
            adapter,
            message_id=41,
            thread_id=10,
            owner_user_id=111,
            session_key=session_key,
            selected=selected_old,
            model_id="model-old",
        )
        await _send_picker(
            adapter,
            message_id=42,
            thread_id=10,
            owner_user_id=111,
            session_key=session_key,
            selected=selected_new,
            model_id="model-new",
        )
        monkeypatch.setattr(
            "hermes_cli.model_selection_guards.combined_selection_warning",
            lambda *_args, **_kwargs: None,
        )

        stale_query = _query(data="mm:0", message_id=41, thread_id=10)
        await adapter._handle_callback_query(
            SimpleNamespace(callback_query=stale_query), SimpleNamespace()
        )

        selected_old.assert_not_awaited()
        selected_new.assert_not_awaited()
        stale_query.edit_message_text.assert_not_awaited()
        assert "expired" in stale_query.answer.await_args.kwargs["text"].lower()

    def test_model_picker_callback_checks_actor_auth_before_slash_policy(self):
        adapter = _make_adapter(
            platform_extra={
                "group_allow_admin_from": ["111"],
                "group_user_allowed_commands": [],
            }
        )
        slash_check = MagicMock(return_value=None)
        adapter.set_authorization_check(lambda *_args: False)
        adapter.set_slash_access_check(slash_check)

        assert adapter._is_model_picker_callback_authorized(_query(user_id=222)) is False
        slash_check.assert_not_called()

    def test_model_picker_callback_fails_closed_without_slash_policy_checker(self):
        adapter = _make_adapter()
        adapter.set_slash_access_check(None)

        assert adapter._is_model_picker_callback_authorized(_query(user_id=111)) is False

    @pytest.mark.parametrize(
        ("chat_type", "thread_id"),
        [("private", None), ("group", None), ("supergroup", "77")],
        ids=["dm", "group", "topic"],
    )
    def test_primary_multiplex_production_shape_authorizes_all_telegram_surfaces(
        self, monkeypatch, chat_type, thread_id
    ):
        import gateway.run as gateway_run

        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="test-token",
                    extra={
                        "allow_admin_from": ["111"],
                        "group_allow_admin_from": ["111"],
                    },
                )
            },
        )
        runner._is_user_authorized = lambda _source: True
        entered = []

        @contextmanager
        def fake_scope(home):
            entered.append(Path(home))
            yield

        monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
        monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: "/profiles/default")
        handler = runner._make_default_profile_message_handler()
        adapter.set_message_handler(handler)
        adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.TELEGRAM))
        adapter.set_slash_access_check(runner._primary_slash_access_check())
        query = _query(user_id=111, chat_type=chat_type)
        query.message.message_thread_id = thread_id

        assert getattr(handler, "__self__", None) is None
        assert adapter._is_model_picker_callback_authorized(query) is True
        assert entered == [Path("/profiles/default")]

    def test_secondary_multiplex_configuration_uses_profile_policy_and_scope(
        self, monkeypatch
    ):
        import gateway.run as gateway_run

        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secondary-token"))
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="primary-token",
                    extra={"group_allow_admin_from": ["999"]},
                )
            },
        )
        runner.session_store = MagicMock()
        runner._busy_text_mode = "interrupt"
        runner._is_user_authorized = lambda _source: True
        secondary_config = GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="secondary-token",
                    extra={"group_allow_admin_from": ["111"]},
                )
            },
        )
        entered = []

        @contextmanager
        def fake_scope(home):
            entered.append(Path(home))
            yield

        monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda profile: Path("/profiles") / profile,
        )

        runner._configure_profile_adapter(
            adapter,
            "reviewer",
            Platform.TELEGRAM,
            gateway_config=secondary_config,
        )

        assert getattr(adapter._message_handler, "__self__", None) is None
        assert adapter._is_model_picker_callback_authorized(_query(user_id=111)) is True
        assert entered == [Path("/profiles/reviewer")]

    @pytest.mark.asyncio
    async def test_authorized_model_picker_callback_applies_model_slash_policy(
        self, monkeypatch
    ):
        adapter = _make_adapter(
            platform_extra={
                "group_allow_admin_from": ["111"],
                "group_user_allowed_commands": [],
            }
        )
        selected = AsyncMock(return_value="Model switched")
        shared_key = "agent:main:telegram:group:12345"
        adapter._model_picker_state[("12345", "42")] = {
            "providers": [
                {"slug": "openrouter", "name": "OpenRouter", "models": ["gpt-x"]}
            ],
            "selected_provider": "openrouter",
            "selected_provider_name": "OpenRouter",
            "model_list": ["gpt-x"],
            "current_model": "old",
            "current_provider": "openrouter",
            "session_key": shared_key,
            "on_model_selected": selected,
            "msg_id": 42,
            "thread_id": None,
            "owner_user_id": "111",
        }
        monkeypatch.setattr(
            "hermes_cli.model_selection_guards.combined_selection_warning",
            lambda *_args, **_kwargs: None,
        )
        query = _query(user_id=111, data="mm:0")

        await adapter._handle_callback_query(
            SimpleNamespace(callback_query=query), SimpleNamespace()
        )

        selected.assert_awaited_once_with("12345", "gpt-x", "openrouter")
        assert ("12345", "42") not in adapter._model_picker_state


