from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock



def test_root_command_is_gateway_visible_in_registry_and_telegram_menu():
    from hermes_cli.commands import (
        GATEWAY_KNOWN_COMMANDS,
        resolve_command,
        telegram_menu_commands,
    )

    cmd = resolve_command("root")

    assert cmd is not None
    assert cmd.description == "Browse the Obsidian vault root"
    assert cmd.gateway_only is True
    assert "root" in GATEWAY_KNOWN_COMMANDS
    menu_commands, _hidden = telegram_menu_commands(max_commands=100)
    assert any(name == "root" for name, _desc in menu_commands)



def test_obsidian_browser_payload_lists_root_folders_and_note_urls(tmp_path, monkeypatch):
    from gateway.obsidian_browser import build_obsidian_browser_payload

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Inbox").mkdir()
    (vault / "Projects").mkdir()
    (vault / ".obsidian").mkdir()
    (vault / "Root Note.md").write_text("# Root Note\n", encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    payload = build_obsidian_browser_payload("")

    assert payload.relative_path == ""
    assert [entry.name for entry in payload.dirs] == ["Inbox", "Projects"]
    assert [entry.name for entry in payload.files] == ["Root Note.md"]
    assert payload.files[0].url.startswith("obsidian://open?vault=")
    assert "Root%20Note.md" in payload.files[0].url



def test_telegram_adapter_sends_root_browser_keyboard(tmp_path, monkeypatch):
    async def scenario():
        from gateway.platforms.base import SendResult
        import gateway.platforms.telegram as telegram_mod
        from gateway.platforms.telegram import TelegramAdapter

        class FakeButton:
            def __init__(self, text, callback_data=None, url=None):
                self.text = text
                self.callback_data = callback_data
                self.url = url

        class FakeMarkup:
            def __init__(self, rows):
                self.inline_keyboard = rows

        monkeypatch.setattr(telegram_mod, "InlineKeyboardButton", FakeButton)
        monkeypatch.setattr(telegram_mod, "InlineKeyboardMarkup", FakeMarkup)

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "Inbox").mkdir()
        (vault / "Projects").mkdir()
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

        from gateway.config import Platform

        adapter = object.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter._bot = object()
        adapter._reply_to_mode = "never"
        adapter._obsidian_browser_tokens = {}
        adapter._send_message_with_thread_fallback = AsyncMock(
            return_value=SimpleNamespace(message_id=123)
        )
        adapter._metadata_thread_id = lambda metadata: (metadata or {}).get("thread_id")
        adapter._reply_to_message_id_for_send = lambda *_args, **_kwargs: None
        adapter._thread_kwargs_for_send = lambda *_args, **_kwargs: {}
        adapter._link_preview_kwargs = lambda: {}
        adapter.format_message = lambda text: text

        result = await adapter.send_obsidian_browser(
            "100",
            start_path="",
            metadata={"thread_id": "9"},
        )

        assert result == SendResult(success=True, message_id="123")
        send_kwargs = adapter._send_message_with_thread_fallback.await_args.kwargs
        assert send_kwargs["chat_id"] == 100
        assert "Obsidian Root" in send_kwargs["text"]
        keyboard = send_kwargs["reply_markup"]
        labels = [button.text for row in keyboard.inline_keyboard for button in row]
        assert "📁 Inbox" in labels
        assert "📁 Projects" in labels

    asyncio.run(scenario())



def test_root_command_dispatch_uses_platform_browser_with_thread_metadata():
    async def scenario():
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent, MessageType, SendResult
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        runner = object.__new__(GatewayRunner)
        adapter = SimpleNamespace(
            send_obsidian_browser=AsyncMock(return_value=SendResult(success=True, message_id="1"))
        )
        runner.adapters = {Platform.TELEGRAM: adapter}
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="100",
            chat_type="dm",
            user_id="u1",
            thread_id="9",
        )
        event = MessageEvent(
            text="/root",
            message_type=MessageType.TEXT,
            source=source,
            message_id="m1",
        )

        result = await runner._handle_obsidian_root_command(event)

        assert result == ""
        adapter.send_obsidian_browser.assert_awaited_once_with(
            "100",
            start_path="",
            metadata={"thread_id": "9"},
        )

    asyncio.run(scenario())
