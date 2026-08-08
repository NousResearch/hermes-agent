"""Tests for Discord embed marker support ([EMBED]...[/EMBED]).

Verifies the behaviour contract:
  * a valid [EMBED] JSON block renders as a discord.Embed
  * text before/after the marker is preserved as plain message content
  * invalid JSON or a missing marker degrades to the original content
  * literal newlines inside JSON strings are accepted (strict=False)
  * colors accept both int and #hex forms; footer accepts dict or str
  * send() attaches the embed on the first chunk only
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord import adapter as discord_module
from plugins.platforms.discord.adapter import DiscordAdapter


def _make_adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    a = DiscordAdapter(config)
    a._client = MagicMock()
    channel = AsyncMock()
    sent = AsyncMock()
    sent.id = "123"
    channel.send = AsyncMock(return_value=sent)
    a._client.get_channel = MagicMock(return_value=channel)
    a._reply_to_mode = "off"
    a._record_discord_response = MagicMock()
    return a, channel


def _bare_adapter():
    return object.__new__(DiscordAdapter)


VALID_BODY = """{"title": "Test", "color": "#3fb950", "description": "```\\nDAY  IMPORT\\n30 Jul  12.8\\n```", "fields": [{"name": "A", "value": "1", "inline": true}], "footer": {"text": "foot"}}"""


class TestExtractEmbed:
    def test_valid_embed(self):
        a = _bare_adapter()
        rest, emb = a._extract_embed(f"[EMBED]\n{VALID_BODY}\n[/EMBED]")
        assert emb is not None
        assert emb.title == "Test"
        assert len(emb.fields) == 1
        assert emb.footer["text"] == "foot"

    def test_preamble_and_tail_preserved(self):
        a = _bare_adapter()
        content = f"hello\n\n[EMBED]\n{VALID_BODY}\n[/EMBED]\n\nbye"
        rest, emb = a._extract_embed(content)
        assert emb is not None
        assert "hello" in rest and "bye" in rest

    def test_no_marker_unchanged(self):
        a = _bare_adapter()
        content = "plain message"
        rest, emb = a._extract_embed(content)
        assert emb is None
        assert rest == content

    def test_invalid_json_falls_back(self):
        a = _bare_adapter()
        content = "[EMBED]\n{not valid json\n[/EMBED]"
        rest, emb = a._extract_embed(content)
        assert emb is None
        assert rest == content

    def test_llm_artifacts_are_tolerated(self):
        """(1/2)-style segment markers and trailing commas written by the
        generating model must not break the parse (regression: a cron report
        arrived as raw text because the JSON contained '}, (1/2)')."""
        a = _bare_adapter()
        content = (
            '[EMBED]\n{"title": "T", "fields": ['
            '{"name": "A", "value": "1", "inline": true}, (1/2)\n'
            '{"name": "B", "value": "2", "inline": true},'
            ']}\n[/EMBED]'
        )
        rest, emb = a._extract_embed(content)
        assert emb is not None
        assert len(emb.fields) == 2

    def test_clean_embed_json_preserves_strings(self):
        a = _bare_adapter()
        cleaned = a._clean_embed_json('{"desc": "keep ,} and (1/2) here", "x": [1, 2,],}')
        assert cleaned == '{"desc": "keep ,} and (1/2) here", "x": [1, 2]}'


class TestStandaloneEmbedPayload:
    def test_split_embed_from_message(self):
        from plugins.platforms.discord.adapter import _split_embed_from_message
        msg = (
            "Cronjob Response: X\n-------------\n\n"
            '[EMBED]\n{"title": "T", "color": "#3fb950", '
            '"fields": [{"name": "A", "value": "1", "inline": true}]}\n[/EMBED]\n\n'
            "To stop, send a message."
        )
        text, payload = _split_embed_from_message(msg)
        assert payload is not None
        assert payload["title"] == "T"
        assert payload["color"] == 0x3FB950
        assert len(payload["fields"]) == 1
        assert "[EMBED]" not in text
        assert "Cronjob Response" in text and "To stop" in text

    def test_split_no_marker_unchanged(self):
        from plugins.platforms.discord.adapter import _split_embed_from_message
        text, payload = _split_embed_from_message("plain message")
        assert payload is None
        assert text == "plain message"

    def test_split_bad_json_unchanged(self):
        from plugins.platforms.discord.adapter import _split_embed_from_message
        text, payload = _split_embed_from_message("[EMBED]\n{broken\n[/EMBED]")
        assert payload is None
        assert "[EMBED]" in text

    def test_literal_newlines_accepted(self):
        a = _bare_adapter()
        content = '[EMBED]\n{"title": "T", "description": "line1\nline2"}\n[/EMBED]'
        rest, emb = a._extract_embed(content)
        assert emb is not None
        assert emb.description == "line1\nline2"

    def test_color_int_and_hex(self):
        a = _bare_adapter()
        _, emb = a._extract_embed('[EMBED]\n{"title": "T", "color": 3447003}\n[/EMBED]')
        assert emb.color == 3447003
        _, emb = a._extract_embed('[EMBED]\n{"title": "T", "color": "#3fb950"}\n[/EMBED]')
        assert emb.color == 0x3FB950

    def test_footer_string_or_dict(self):
        a = _bare_adapter()
        _, emb = a._extract_embed('[EMBED]\n{"title": "T", "footer": "plain"}\n[/EMBED]')
        assert emb.footer["text"] == "plain"
        _, emb = a._extract_embed('[EMBED]\n{"title": "T", "footer": {"text": "dict"}}\n[/EMBED]')
        assert emb.footer["text"] == "dict"

    def test_fields_capped_at_25(self):
        a = _bare_adapter()
        fields = ",".join('{"name": "f%d", "value": "v", "inline": true}' % i for i in range(30))
        _, emb = a._extract_embed(f'[EMBED]\n{{"title": "T", "fields": [{fields}]}}\n[/EMBED]')
        assert len(emb.fields) == 25

    def test_embed_requires_discord(self, monkeypatch):
        a = _bare_adapter()
        monkeypatch.setattr(discord_module, "discord", None)
        content = f"[EMBED]\n{VALID_BODY}\n[/EMBED]"
        rest, emb = a._extract_embed(content)
        assert emb is None
        assert rest == content


class TestSendAttachesEmbed:
    @pytest.mark.asyncio
    async def test_send_with_embed(self):
        a, channel = _make_adapter()
        await a.send("123", f"hi\n[EMBED]\n{VALID_BODY}\n[/EMBED]")
        call_kwargs = channel.send.call_args.kwargs
        assert call_kwargs.get("embed") is not None
        assert "[EMBED]" not in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_send_without_embed_unaffected(self):
        a, channel = _make_adapter()
        await a.send("123", "normal message")
        call_kwargs = channel.send.call_args.kwargs
        assert call_kwargs.get("embed") is None
        assert call_kwargs["content"] == "normal message"

    @pytest.mark.asyncio
    async def test_forum_send_gets_embed(self):
        """Forum routing must receive the parsed embed + marker-free content,
        not the raw [EMBED] JSON (regression: extraction previously ran
        after the forum branch)."""
        a, _ = _make_adapter()
        a._is_forum_parent = MagicMock(return_value=True)
        a._send_to_forum = AsyncMock(
            return_value=MagicMock(success=True, message_id="999")
        )
        await a.send("123", f"forum post\n[EMBED]\n{VALID_BODY}\n[/EMBED]")
        call_args = a._send_to_forum.call_args
        assert call_args.kwargs.get("embed") is not None
        assert "[EMBED]" not in call_args.args[1]


class TestEditMessageEmbed:
    @pytest.mark.asyncio
    async def test_finalize_oversized_embed_parses_before_split(self):
        """A finalized stream over the length cap must parse the embed
        before overflow splitting, so chunks never expose the marker
        (regression: extraction previously ran after the overflow branch)."""
        a, channel = _make_adapter()
        msg = AsyncMock()
        msg.id = "123"
        channel.fetch_message = AsyncMock(return_value=msg)
        a._edit_overflow_split = AsyncMock(
            return_value=MagicMock(success=True, message_id="123")
        )
        huge_tail = "x" * 3000
        content = f"[EMBED]\n{VALID_BODY}\n[/EMBED]\n{huge_tail}"
        await a.edit_message("123", "123", content, finalize=True)
        call_args = a._edit_overflow_split.call_args
        assert call_args.kwargs.get("embed") is not None
        assert "[EMBED]" not in call_args.args[3]
        assert call_args.args[3].endswith(huge_tail)

    @pytest.mark.asyncio
    async def test_finalize_normal_embed_edits_in_place(self):
        a, channel = _make_adapter()
        msg = AsyncMock()
        msg.id = "123"
        channel.fetch_message = AsyncMock(return_value=msg)
        await a.edit_message("123", "123", f"[EMBED]\n{VALID_BODY}\n[/EMBED]", finalize=True)
        edit_kwargs = msg.edit.call_args.kwargs
        assert edit_kwargs.get("embed") is not None
        assert "[EMBED]" not in edit_kwargs["content"]

    @pytest.mark.asyncio
    async def test_midstream_shows_placeholder(self):
        a, channel = _make_adapter()
        msg = AsyncMock()
        msg.id = "123"
        channel.fetch_message = AsyncMock(return_value=msg)
        await a.edit_message("123", "123", f"[EMBED]\n{VALID_BODY}", finalize=False)
        edit_kwargs = msg.edit.call_args.kwargs
        assert "rendering embed" in edit_kwargs["content"]
