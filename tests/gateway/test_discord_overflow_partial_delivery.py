"""Regression tests for Discord overflow-edit partial delivery (issue #120073).

A final edit that overflows is split into continuations.  When a continuation
send fails after the first chunk landed, the adapter used to report
``success=True`` + ``partial_overflow`` — but the stream consumer only reads
that contract on the FAILURE path (``_on_edit_failure``), so the failed tail
was recorded as fully delivered, arming neither the fallback tail send nor the
gateway's normal final send.  The fix reports ``success=False`` with the same
``partial_overflow`` keys Telegram's edit-overflow path sets (including
``delivered_prefix``), so the consumer arms the fallback and the gateway does
not suppress the final send.
"""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

# The raw text stays under the consumer's edit budget (1900) while Discord's
# table-to-bullets formatting inflates it past the 2000-char adapter split cap
# — the only shape that reaches the adapter's overflow-finalize path.
TABLE_TEXT = (
    "| Item | Detail that is intentionally repeated in every rendered bullet label |\n"
    "| --- | --- |\n" + "| x | v |\n" * 50 + "\nTAIL_MARKER"
)


def _make_adapter():
    return DiscordAdapter(PlatformConfig(enabled=True))


async def _failing_send(content, reference=None):
    raise RuntimeError("Forbidden: missing send permission")


async def _make_channel(send_fn):
    async def edit(*, content):
        return None

    original = SimpleNamespace(id=1, edit=edit, to_reference=lambda **kw: object())
    channel = SimpleNamespace(
        id=2, send=send_fn, get_partial_message=lambda mid: original
    )
    return channel, original


class TestAdapterPartialOverflowContract:
    """Adapter-level: a failed continuation is a failure with partial-overflow metadata."""

    @pytest.mark.asyncio
    async def test_first_continuation_failure_reports_partial_not_success(self):
        adapter = _make_adapter()
        adapter._client = object()
        channel, original = await _make_channel(_failing_send)
        adapter._resolve_channel = AsyncMock(return_value=channel)

        result = await adapter._edit_overflow_split(channel, original, "1", TABLE_TEXT)

        assert result.success is False
        assert result.error == "overflow_continuation_failed"
        assert result.retryable is True
        assert result.message_id == "1"
        assert result.continuation_message_ids == ()
        raw = result.raw_response
        assert raw["partial_overflow"] is True
        assert raw["delivered_chunks"] == 1
        assert raw["total_chunks"] > 1
        assert raw["last_message_id"] == "1"
        # delivered_prefix is the landed first chunk (formatted space, indicators stripped).
        formatted = adapter.format_message(TABLE_TEXT)
        assert len(formatted) > adapter.MAX_MESSAGE_LENGTH
        assert formatted.startswith(raw["delivered_prefix"])

    @pytest.mark.asyncio
    async def test_mid_split_failure_keeps_landed_continuation(self):
        adapter = _make_adapter()
        adapter._client = object()
        calls = {"n": 0}

        async def send_then_fail(content, reference=None):
            calls["n"] += 1
            if calls["n"] >= 2:  # second continuation: both attempts fail
                raise RuntimeError("Forbidden: missing send permission")
            return SimpleNamespace(id=101)

        channel, original = await _make_channel(send_then_fail)
        adapter._resolve_channel = AsyncMock(return_value=channel)

        result = await adapter._edit_overflow_split(channel, original, "1", TABLE_TEXT)

        assert result.success is False
        raw = result.raw_response
        assert raw["partial_overflow"] is True
        assert raw["delivered_chunks"] == 2
        assert raw["last_message_id"] == "101"
        assert result.continuation_message_ids == ("101",)
        # delivered_prefix = the landed chunks joined, "(n/N)" indicators stripped.
        chunks = adapter._cap_split_chunks(
            adapter.truncate_message(
                adapter.format_message(TABLE_TEXT), adapter.MAX_MESSAGE_LENGTH
            )
        )
        expected = "".join(re.sub(r" \(\d+/\d+\)$", "", c) for c in chunks[:2])
        assert raw["delivered_prefix"] == expected

    @pytest.mark.asyncio
    async def test_full_success_control_unchanged(self):
        adapter = _make_adapter()
        adapter._client = object()
        next_id = {"n": 100}

        async def send_ok(content, reference=None):
            next_id["n"] += 1
            return SimpleNamespace(id=next_id["n"])

        channel, original = await _make_channel(send_ok)
        adapter._resolve_channel = AsyncMock(return_value=channel)

        result = await adapter._edit_overflow_split(channel, original, "1", TABLE_TEXT)

        assert result.success is True
        raw = result.raw_response or {}
        assert not raw.get("partial_overflow")
        assert result.continuation_message_ids == ("101", "102")


class _VisibleChannel:
    """Channel double that records every visible message (edits and sends)."""

    def __init__(self, send_fn):
        self.visible: dict[str, str] = {}
        self.sends: list[str] = []
        self._send_fn = send_fn
        self.id = 2
        self.original = SimpleNamespace(
            id=1, edit=self._edit, to_reference=lambda **kw: object()
        )
        self.next_id = 100

    async def _edit(self, *, content):
        self.visible["1"] = content

    def get_partial_message(self, mid):
        return self.original

    async def send(self, content, reference=None):
        return await self._send_fn(self, content, reference)


def _seed_consumer(adapter, channel):
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    adapter._resolve_channel = AsyncMock(return_value=channel)
    consumer = GatewayStreamConsumer(
        adapter, "2", config=StreamConsumerConfig(cursor="")
    )
    consumer._message_id = "1"
    consumer._last_sent_text = "preview"
    consumer._already_sent = True
    consumer._accumulated = consumer._stream_ledger = "preview"
    return consumer


class TestConsumerChainPartialOverflow:
    """Full chain: adapter → stream consumer → gateway final-suppression decision.

    Mirrors the issue's reproduction (only transport operations are faked).
    """

    def _setup(self):
        from gateway.run_turn import GatewayTurnMixin

        return _make_adapter(), GatewayTurnMixin()

    @pytest.mark.asyncio
    async def test_failed_continuation_leaves_final_undelivered(self):
        adapter, turn = self._setup()
        adapter._client = object()
        channel = _VisibleChannel(_failing_send)
        consumer = _seed_consumer(adapter, channel)

        consumer.finish(TABLE_TEXT)
        await consumer.run()

        # Fail-closed: the missing tail must not be recorded as delivered.
        assert consumer.final_content_delivered is False
        assert consumer.final_response_sent is False
        response: dict = {"final_response": TABLE_TEXT}
        holder = SimpleNamespace(
            stream_consumer_holder=[consumer], source=None, session_key="fixture"
        )
        await turn._run_agent_mark_streamed_delivery(response, holder)
        assert "already_sent" not in response  # the normal final send stays armed

    @pytest.mark.asyncio
    async def test_failed_continuation_then_fallback_delivers_tail(self):
        adapter, turn = self._setup()
        adapter._client = object()
        attempts = {"n": 0}
        deleted: list[str] = []

        async def refuse_then_recover(channel, content, reference):
            attempts["n"] += 1
            if (
                attempts["n"] <= 2
            ):  # the first continuation's send + reference-less retry
                raise RuntimeError("Forbidden: missing send permission")
            channel.next_id += 1
            mid = str(channel.next_id)
            channel.visible[mid] = content
            channel.sends.append(content)
            return SimpleNamespace(id=int(mid))

        channel = _VisibleChannel(refuse_then_recover)
        consumer = _seed_consumer(adapter, channel)

        async def delete_ok(chat_id, message_id):
            deleted.append(message_id)
            return True

        adapter.delete_message = delete_ok

        consumer.finish(TABLE_TEXT)
        await consumer.run()

        # The fallback delivered the answer (tail reachable on screen) and the
        # stale partial was cleaned up before the full-text resend — the same
        # contract Telegram's edit-overflow partial path follows.
        assert consumer.final_response_sent is True
        assert consumer.final_content_delivered is True
        assert any("TAIL_MARKER" in c for c in channel.visible.values())
        assert deleted == ["1"]
        response: dict = {"final_response": TABLE_TEXT}
        holder = SimpleNamespace(
            stream_consumer_holder=[consumer], source=None, session_key="fixture"
        )
        await turn._run_agent_mark_streamed_delivery(response, holder)
        assert response.get("already_sent") is True  # delivered via fallback
