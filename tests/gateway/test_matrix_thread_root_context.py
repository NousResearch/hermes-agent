"""Thread root delivery for a new Matrix thread session."""
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

ROOT_TAIL = "Option C: wait for the next release."
LONG_ROOT = "Three options. " + "background detail " * 120 + ROOT_TAIL
ROOM = "!dm:example.org"


def _make_adapter():
    """Create a MatrixAdapter with mocked config."""
    from plugins.platforms.matrix.adapter import MatrixAdapter
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    return adapter


def _dm_adapter():
    """A DM adapter whose thread root is a long message from the bot itself."""
    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(
        side_effect=lambda room, user: user.split(":")[0][1:]
    )
    adapter._client.get_event = AsyncMock(
        return_value=types.SimpleNamespace(
            sender="@bot:example.org",
            content={"msgtype": "m.text", "body": LONG_ROOT},
        )
    )
    return adapter


def _line(context, prefix):
    """Return the single line of ``context`` that starts with ``prefix``."""
    lines = [line for line in context.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1, lines
    return lines[0]


@pytest.mark.asyncio
async def test_new_thread_session_keeps_the_whole_bot_root_in_a_dm():
    adapter = _dm_adapter()
    adapter._client.api.request = AsyncMock(
        side_effect=[
            {"start": "trigger-boundary"},
            {
                "chunk": [
                    {
                        "event_id": "$current",
                        "sender": "@alice:example.org",
                        "content": {"msgtype": "m.text", "body": "go with option A"},
                    },
                ]
            },
        ]
    )

    context = await adapter.fetch_thread_context(
        ROOM, "$root", before_event_id="$current"
    )

    assert context is not None
    assert ROOT_TAIL in context
    assert not _line(context, "[bot] ").endswith("...")


@pytest.mark.asyncio
async def test_thread_replies_other_than_the_root_stay_capped():
    adapter = _dm_adapter()
    adapter._client.api.request = AsyncMock(
        side_effect=[
            {"start": "trigger-boundary"},
            {
                "chunk": [
                    {
                        "event_id": "$current",
                        "sender": "@alice:example.org",
                        "content": {"msgtype": "m.text", "body": "go with option A"},
                    },
                    {
                        "event_id": "$reply",
                        "sender": "@alice:example.org",
                        "content": {
                            "msgtype": "m.text",
                            "body": "x" * 2000,
                            "m.relates_to": {
                                "rel_type": "m.thread",
                                "event_id": "$root",
                            },
                        },
                    },
                ]
            },
        ]
    )

    context = await adapter.fetch_thread_context(
        ROOM, "$root", before_event_id="$current"
    )

    assert context is not None
    reply_line = _line(context, "[alice] ")
    assert reply_line.endswith("...")
    assert len(reply_line) <= len("[alice] ") + 1200
    assert ROOT_TAIL in context


@pytest.mark.asyncio
async def test_new_thread_session_gets_the_root_when_the_thread_page_fails():
    adapter = _dm_adapter()
    adapter._client.api.request = AsyncMock(
        side_effect=RuntimeError("homeserver unavailable")
    )

    context = await adapter.fetch_thread_context(
        ROOM, "$root", before_event_id="$current"
    )

    assert context is not None
    assert ROOT_TAIL in context


@pytest.mark.asyncio
async def test_catch_up_skips_the_root_when_the_thread_page_fails():
    from plugins.platforms.matrix.reply_context import MatrixEventContextCache
    from plugins.platforms.matrix.thread_context import fetch_thread_entries

    client = MagicMock()
    client.api.request = AsyncMock(side_effect=RuntimeError("homeserver unavailable"))
    client.get_event = AsyncMock(
        return_value=types.SimpleNamespace(
            sender="@alice:example.org",
            content={"msgtype": "m.text", "body": "Root"},
        )
    )

    entries = await fetch_thread_entries(
        client,
        MatrixEventContextCache(),
        ROOM,
        "$root",
        limit=20,
        before_event_id="$current",
        is_previous_turn=lambda sender, content: False,
    )

    assert entries == []


@pytest.mark.asyncio
async def test_catch_up_root_from_an_unverified_sender_is_whole_and_tagged():
    adapter = _dm_adapter()
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_sender_authorized = MagicMock(
        side_effect=lambda user, **kwargs: user != "@stranger:example.org"
    )
    adapter._client.get_event = AsyncMock(
        return_value=types.SimpleNamespace(
            sender="@stranger:example.org",
            content={"msgtype": "m.text", "body": LONG_ROOT.replace("Option C", "\nOption C")},
        )
    )
    adapter._client.api.request = AsyncMock(
        side_effect=[
            {"start": "trigger-boundary"},
            {
                "chunk": [
                    {
                        "event_id": "$current",
                        "sender": "@alice:example.org",
                        "content": {"msgtype": "m.text", "body": "@bot what now"},
                    }
                ]
            },
        ]
    )

    context = await adapter.fetch_thread_context(
        "!room:example.org",
        "$root",
        before_event_id="$current",
        is_previous_turn=lambda sender, content: False,
    )

    assert context is not None
    root_line = _line(context, "[unverified] [stranger] ")
    assert root_line.endswith(ROOT_TAIL)
