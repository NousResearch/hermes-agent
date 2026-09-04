"""Regression tests for Discord link-preview suppression (issue #91657).

Discord unfurls URLs in bot messages into preview cards.  The dependable lever
is the per-message ``SUPPRESS_EMBEDS`` flag, set at send time via
``channel.send(suppress_embeds=...)``.  The adapter exposes this as
``extra.suppress_link_previews`` (default **on**) so streamed answers that cite
links (docs lookups, search results, PR links) don't clutter the channel with
embeds.

The streaming edit path edits through ``PartialMessage.edit(content=...)``,
which — in the pinned discord.py 2.7.1 — sends no ``flags`` field, so it
*preserves* a suppression applied at send time (Discord only regenerates the
embed when an edit actively clears the flag).  These tests pin both halves:
the send carries the flag per config, and the streaming edit never passes a
``suppress``/``flags`` keyword that could clear it.
"""

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

MAX = DiscordAdapter.MAX_MESSAGE_LENGTH  # 2000


def _make_adapter(suppress_link_previews=None):
    extra = {}
    if suppress_link_previews is not None:
        extra["suppress_link_previews"] = suppress_link_previews
    return DiscordAdapter(PlatformConfig(enabled=True, token="***", extra=extra))


def _wire_channel(adapter, *, original_msg=None):
    """Fake client whose channel records every ``channel.send`` call (kwargs
    included) and returns ``original_msg`` from ``get_partial_message``."""
    sends = []

    async def fake_send(*, content, reference=None, suppress_embeds=False):
        sends.append({"content": content, "suppress_embeds": suppress_embeds})
        return SimpleNamespace(id=9000 + len(sends))

    channel = SimpleNamespace(
        id=555,
        get_partial_message=MagicMock(return_value=original_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    return channel, sends


# --------------------------------------------------------------------------- #
# Config coercion — default on, override off, string spellings
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_defaults_on_when_key_absent(self):
        assert _make_adapter()._suppress_link_previews is True

    def test_explicit_true(self):
        assert _make_adapter(True)._suppress_link_previews is True

    def test_explicit_false(self):
        assert _make_adapter(False)._suppress_link_previews is False

    @pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off", ""])
    def test_string_falsey_spellings_disable(self, raw):
        assert _make_adapter(raw)._suppress_link_previews is False

    @pytest.mark.parametrize("raw", ["true", "yes", "on", "1"])
    def test_string_truthy_spellings_enable(self, raw):
        assert _make_adapter(raw)._suppress_link_previews is True


# --------------------------------------------------------------------------- #
# Send path — the flag rides on channel.send per config
# --------------------------------------------------------------------------- #


class TestSendSuppression:
    @pytest.mark.asyncio
    async def test_send_suppresses_by_default(self):
        adapter = _make_adapter()
        _, sends = _wire_channel(adapter)
        result = await adapter.send("555", "see https://example.com for docs")
        assert result.success is True
        assert len(sends) == 1
        assert sends[0]["suppress_embeds"] is True

    @pytest.mark.asyncio
    async def test_send_keeps_previews_when_disabled(self):
        adapter = _make_adapter(False)
        _, sends = _wire_channel(adapter)
        result = await adapter.send("555", "see https://example.com for docs")
        assert result.success is True
        assert sends[0]["suppress_embeds"] is False

    @pytest.mark.asyncio
    async def test_overflow_continuations_carry_the_flag(self):
        # A finalized oversized edit splits into a first-chunk edit plus
        # continuation channel.send() calls — those continuations are part of
        # the same streamed reply and must stay suppressed too.
        adapter = _make_adapter()
        partial = SimpleNamespace(
            id=42,
            edit=AsyncMock(),
            to_reference=MagicMock(return_value=SimpleNamespace(kind="ref")),
        )
        _, sends = _wire_channel(adapter, original_msg=partial)
        result = await adapter.edit_message("555", "42", "link " * 1200, finalize=True)
        assert result.success is True
        assert sends, "expected overflow continuations"
        assert all(s["suppress_embeds"] is True for s in sends)


# --------------------------------------------------------------------------- #
# Edit path — the content-only edit never clears the flag
# --------------------------------------------------------------------------- #


class TestEditPreservesSuppression:
    @pytest.mark.asyncio
    async def test_streaming_edit_sends_no_suppress_or_flags_kwarg(self):
        """The streaming edit must not pass ``suppress``/``flags`` to
        ``PartialMessage.edit``.  Passing ``suppress=False`` (or ``flags`` with
        the bit cleared) is exactly what makes Discord regenerate the embed
        mid-stream; a content-only edit leaves the send-time suppression intact.
        """
        adapter = _make_adapter()
        edit_calls = []

        async def record_edit(**kwargs):
            edit_calls.append(kwargs)

        msg = SimpleNamespace(id=42, edit=AsyncMock(side_effect=record_edit))
        _wire_channel(adapter, original_msg=msg)

        result = await adapter.edit_message("555", "42", "streamed https://example.com")
        assert result.success is True
        assert len(edit_calls) == 1
        assert edit_calls[0] == {"content": "streamed https://example.com"}
        assert "suppress" not in edit_calls[0]
        assert "flags" not in edit_calls[0]
