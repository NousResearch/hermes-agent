"""A busy Telegram `/new -project` command is handled without stopping the active turn."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_busy_project_picker_open_does_not_stop_owner_turn(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.telegram.adapter import TelegramAdapter

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    from hermes_cli import projects_db

    with projects_db.connect_closing(home / "projects.db") as conn:
        projects_db.create_project(conn, name="Busy project", folders=[str(workspace)])

    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")
            }
        )
    )
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._send_control_message = AsyncMock(
        return_value=SimpleNamespace(message_id=711)
    )
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._wire_adapter_handlers(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm"
    )
    key = runner.session_store._generate_session_key(source)
    runner.session_store.get_or_create_session(source)
    stop = AsyncMock()
    runner._hm_stop_owner_turn = stop

    # The adapter's active-task bypass sends the picker inline; it must not take
    # the cancel-and-drain path reserved for ordinary /new.
    await adapter._handle_message_while_active(
        MessageEvent(text="/new -project", source=source), key
    )
    assert runner.session_store.peek_session_id(key) is not None
    assert len(adapter._project_picker_state) == 1
    first_token = next(iter(adapter._project_picker_state))
    stop.assert_not_awaited()

    handled, reply = await runner._hm_busy_slash_or_photo(
        MessageEvent(text="/new -project", source=source), source, key
    )
    assert handled and reply is None
    stop.assert_not_awaited()
    assert runner.session_store.peek_session_id(key) is not None
    # The second picker supersedes the first for this chat/user: exactly one live token.
    assert len(adapter._project_picker_state) == 1
    superseded = first_token
    assert superseded not in adapter._project_picker_state

    # A malformed flag on the same active route is also non-destructive and explicit.
    handled, reply = await runner._hm_busy_slash_or_photo(
        MessageEvent(text="/new --not-a-project-option", source=source), source, key
    )
    assert handled and "unknown flags" in reply.lower()
    stop.assert_not_awaited()
    assert len(adapter._project_picker_state) == 1

    token = next(iter(adapter._project_picker_state))
    assert token != first_token
    old_id = runner.session_store.peek_session_id(key)

    def tap(token_value):
        return SimpleNamespace(
            data=f"pj:{token_value}:select_0",
            from_user=SimpleNamespace(id=42, first_name="tester"),
            message=SimpleNamespace(
                message_id=711, chat_id=42, message_thread_id=None,
                chat=SimpleNamespace(type="private"),
            ),
            answer=AsyncMock(), edit_message_text=AsyncMock(),
        )

    # A tap on the superseded picker is rejected: no session switch, no stop.
    stale_query = tap(first_token)
    await adapter._handle_callback_query(SimpleNamespace(callback_query=stale_query), None)
    assert runner.session_store.peek_session_id(key) == old_id
    stop.assert_not_awaited()
    assert token in adapter._project_picker_state

    action = f"pj:{token}:select_0"
    query = SimpleNamespace(
        data=action,
        from_user=SimpleNamespace(id=42, first_name="tester"),
        message=SimpleNamespace(
            message_id=711,
            chat_id=42,
            message_thread_id=None,
            chat=SimpleNamespace(type="private"),
        ),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )
    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)
    assert runner.session_store.peek_session_id(key) != old_id
    stop.assert_awaited_once_with(key)
    assert token not in adapter._project_picker_state
    assert len(adapter._project_picker_state) == 0
    runner._shutdown_executor()
