"""Matrix-specific command-prefix compatibility tests."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.matrix import MatrixAdapter, _normalize_matrix_bang_command


def _make_adapter() -> MatrixAdapter:
    return MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
            },
        )
    )


class TestMatrixBangCommandNormalization:
    def test_known_bang_command_maps_to_slash(self):
        assert _normalize_matrix_bang_command("!yolo") == "/yolo"
        assert _normalize_matrix_bang_command("!reasoning high") == "/reasoning high"

    def test_alias_and_bot_suffix_map_to_slash(self):
        assert _normalize_matrix_bang_command("!reset") == "/reset"
        assert _normalize_matrix_bang_command("!yolo@HermesBot") == "/yolo@HermesBot"

    def test_unknown_bang_text_stays_plain_text(self):
        assert _normalize_matrix_bang_command("!not-a-hermes-command") == "!not-a-hermes-command"
        assert _normalize_matrix_bang_command("!path/to/file.py") == "!path/to/file.py"
        assert _normalize_matrix_bang_command("hello !yolo") == "hello !yolo"

    def test_cli_only_command_stays_plain_text(self):
        assert _normalize_matrix_bang_command("!history is useful") == "!history is useful"
        assert _normalize_matrix_bang_command("!config should be yaml") == "!config should be yaml"
        assert _normalize_matrix_bang_command("!tools are useful") == "!tools are useful"

    @pytest.mark.asyncio
    async def test_text_handler_emits_command_event_for_known_bang_command(self):
        adapter = _make_adapter()
        adapter._is_dm_room = AsyncMock(return_value=True)
        adapter._get_display_name = AsyncMock(return_value="Alice")
        adapter._text_batch_delay_seconds = 0

        captured = None

        async def capture(event):
            nonlocal captured
            captured = event

        adapter.handle_message = capture

        await adapter._handle_text_message(
            "!room:example.org",
            "@alice:example.org",
            "$event1",
            0.0,
            {"msgtype": "m.text", "body": "!yolo"},
            {},
        )

        assert captured is not None
        assert captured.text == "/yolo"
        assert captured.message_type == MessageType.COMMAND
        assert captured.get_command() == "yolo"

    @pytest.mark.asyncio
    async def test_text_handler_keeps_unknown_bang_text_plain(self):
        adapter = _make_adapter()
        adapter._is_dm_room = AsyncMock(return_value=True)
        adapter._get_display_name = AsyncMock(return_value="Alice")
        adapter._text_batch_delay_seconds = 0

        captured = None

        async def capture(event):
            nonlocal captured
            captured = event

        adapter.handle_message = capture

        await adapter._handle_text_message(
            "!room:example.org",
            "@alice:example.org",
            "$event2",
            0.0,
            {"msgtype": "m.text", "body": "!not-a-hermes-command"},
            {},
        )

        assert captured is not None
        assert captured.text == "!not-a-hermes-command"
        assert captured.message_type == MessageType.TEXT
        assert captured.get_command() is None

    @pytest.mark.asyncio
    async def test_text_handler_emits_command_event_after_reply_fallback(self):
        adapter = _make_adapter()
        adapter._is_dm_room = AsyncMock(return_value=True)
        adapter._get_display_name = AsyncMock(return_value="Alice")
        adapter._text_batch_delay_seconds = 0

        captured = None

        async def capture(event):
            nonlocal captured
            captured = event

        adapter.handle_message = capture

        await adapter._handle_text_message(
            "!room:example.org",
            "@alice:example.org",
            "$event3",
            0.0,
            {"msgtype": "m.text", "body": "> <@bob:example.org> quoted\n\n!yolo"},
            {"m.in_reply_to": {"event_id": "$parent"}},
        )

        assert captured is not None
        assert captured.text == "/yolo"
        assert captured.message_type == MessageType.COMMAND
        assert captured.get_command() == "yolo"
