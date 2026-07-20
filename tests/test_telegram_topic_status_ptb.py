"""PTB handler-routing coverage for Telegram DM topic status updates."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_registered_status_observer_matches_dm_topic_updates_without_dispatching():
    """Exercise real PTB filters in a clean interpreter, outside gateway mocks."""
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        r'''
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import telegram

        from gateway.config import PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter


        class HandlerCollector:
            def __init__(self):
                self.handlers = []
                self.bot = None

            def add_handler(self, handler, *args, **kwargs):
                self.handlers.append(handler)


        def message_update(*, update_id, chat_type="private", created=None,
                           edited=None, text=None):
            chat = {"id": 111, "type": chat_type}
            if chat_type == "private":
                chat["first_name"] = "Test"
            else:
                chat["title"] = "Test Group"
            message = {
                "message_id": update_id,
                "date": 1,
                "chat": chat,
                "message_thread_id": 201,
                "is_topic_message": True,
            }
            if created is not None:
                message["forum_topic_created"] = created
            if edited is not None:
                message["forum_topic_edited"] = edited
            if text is not None:
                message["text"] = text
            return telegram.Update.de_json(
                {"update_id": update_id, "message": message},
                bot=None,
            )


        async def main():
            adapter = TelegramAdapter(
                PlatformConfig(enabled=True, token="123456:test-token")
            )
            collector = HandlerCollector()
            adapter._app = collector
            adapter.handle_message = AsyncMock()
            adapter._register_handlers()

            status_handlers = [
                handler
                for handler in collector.handlers
                if handler.callback == adapter._handle_dm_topic_status_update
            ]
            assert len(status_handlers) == 1
            handler = status_handlers[0]

            created_update = message_update(
                update_id=1,
                created={"name": "ProjectAtlas", "icon_color": 7322096},
            )
            edited_update = message_update(
                update_id=2,
                edited={"icon_custom_emoji_id": "manual-id"},
            )
            text_update = message_update(update_id=3, text="hello")
            group_update = message_update(
                update_id=4,
                chat_type="supergroup",
                created={"name": "Group Topic", "icon_color": 7322096},
            )

            assert handler.check_update(created_update)
            assert handler.check_update(edited_update)
            assert not handler.check_update(text_update)
            assert not handler.check_update(group_update)

            await handler.callback(created_update, SimpleNamespace())
            assert adapter.dm_topic_custom_icon_state("111", "201") is False
            assert adapter._dm_topics["111:ProjectAtlas"] == 201

            await handler.callback(edited_update, SimpleNamespace())
            assert adapter.dm_topic_custom_icon_state("111", "201") is True
            adapter.handle_message.assert_not_awaited()


        asyncio.run(main())
        '''
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
