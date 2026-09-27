"""Outbound bot-to-bot mentions for the Feishu adapter.

Feishu delivers a group message to another bot only when the message carries a real mention:
plain text ``@Name`` is just characters, and a peer bot that admits bots on mention only
(``FEISHU_ALLOW_BOTS=mentions``) rejects it either way. So a bot that wants to address another
bot by name has no way to do it — the platform never pushes the message.

``feishu.bot_mention_map`` / ``FEISHU_BOT_MENTION_MAP`` maps a display name to that bot's
open_id; the adapter rewrites a configured ``@Name`` in outbound text into an ``at`` post
element carrying the mapped id, which is what Feishu counts as a mention. With no map
configured every outbound payload is byte-identical to the previous rendering.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.feishu_helpers import make_adapter_skeleton
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("feishu")

_MAP = {
    "GameDev": "ou_1657test00000000000000000000",
    "Default Hermes": "ou_59d46test0000000000000000000",
}


def _bare():
    adapter = make_adapter_skeleton()
    adapter._bot_mention_map = dict(_MAP)
    return adapter


def _extract(content: str):
    return _bare()._extract_mentions(content)


def _at_ids(payload_str: str) -> list[str]:
    payload = json.loads(payload_str)
    ids: list[str] = []
    for lang_val in payload.values():
        if not isinstance(lang_val, dict):
            continue
        for block in lang_val.get("content", []):
            for el in (block if isinstance(block, list) else [block]):
                if isinstance(el, dict) and el.get("tag") == "at":
                    ids.append(el.get("user_id", ""))
    return ids


def _ok_response() -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(message_id="om_sent"), success=lambda: True, code=0, msg="",
    )


# --- Configuration parsing -------------------------------------------------


class TestParseBotMentionMap:
    def test_yaml_mapping(self):
        assert _adapter._parse_bot_mention_map({"GameDev": "ou_a"}) == {"GameDev": "ou_a"}

    def test_env_csv(self):
        assert _adapter._parse_bot_mention_map("GameDev=ou_a, Other=ou_b") == {
            "GameDev": "ou_a", "Other": "ou_b",
        }

    def test_env_json(self):
        assert _adapter._parse_bot_mention_map('{"GameDev": "ou_a"}') == {"GameDev": "ou_a"}

    def test_malformed_entry_is_dropped_and_the_rest_survives(self):
        assert _adapter._parse_bot_mention_map("GameDev=ou_a,broken,=ou_c") == {"GameDev": "ou_a"}

    @pytest.mark.parametrize("raw", ["", "   ", None, [], "{}"])
    def test_empty_and_unusable_inputs_yield_no_map(self, raw):
        assert _adapter._parse_bot_mention_map(raw) == {}


# --- Outbound text rewriting ------------------------------------------------


class TestExtractMentions:
    def test_configured_name_becomes_a_placeholder(self):
        out, ids, names = _extract("@GameDev 请确认收到")
        assert out == "@_mention_0 请确认收到"
        assert ids == ["ou_1657test00000000000000000000"]
        assert names == ["GameDev"]

    def test_text_without_a_configured_name_is_untouched(self):
        out, ids, names = _extract("无mention纯文本")
        assert out == "无mention纯文本"
        assert ids == [] and names == []

    def test_trailing_letter_is_not_a_partial_match(self):
        # "@GameDevX" must not match the "GameDev" entry.
        out, ids, _ = _extract("@GameDevX 不匹配后缀字母")
        assert ids == []
        assert "@_mention_" not in out

    def test_longest_name_wins(self):
        out, ids, names = _extract("前缀 @Default Hermes 后缀")
        assert out == "前缀 @_mention_0 后缀"
        assert names == ["Default Hermes"]

    def test_multiple_names_get_distinct_placeholders(self):
        out, ids, names = _extract("同时 @GameDev 和 @Default Hermes")
        assert out == "同时 @_mention_0 和 @_mention_1"
        assert len(ids) == 2 and len(names) == 2

    def test_markdown_surrounding_the_name_is_kept(self):
        out, ids, _ = _extract("**加粗标题** @GameDev 内容")
        assert out == "**加粗标题** @_mention_0 内容"
        assert ids

    def test_no_map_configured_is_a_no_op(self):
        adapter = make_adapter_skeleton()
        adapter._bot_mention_map = {}
        assert adapter._extract_mentions("@GameDev 保持纯文本") == ("@GameDev 保持纯文本", [], [])


# --- Post payload rendering -------------------------------------------------


class TestOutboundPayload:
    def test_at_element_sits_between_the_md_elements(self):
        adapter = _bare()
        content, ids, _ = adapter._extract_mentions("**标题** @GameDev 请确认")
        msg_type, payload = adapter._build_outbound_payload(content, mention_ids=ids)
        assert msg_type == "post"
        rows = json.loads(payload)["zh_cn"]["content"]
        assert rows == [[
            {"tag": "md", "text": "**标题** "},
            {"tag": "at", "user_id": "ou_1657test00000000000000000000"},
            {"tag": "md", "text": " 请确认"},
        ]]

    def test_mention_forces_post_without_any_markdown_hint(self):
        adapter = _bare()
        content, ids, _ = adapter._extract_mentions("@GameDev 收到请回")
        msg_type, payload = adapter._build_outbound_payload(content, mention_ids=ids)
        assert msg_type == "post"
        assert _at_ids(payload) == ["ou_1657test00000000000000000000"]

    def test_payload_is_unchanged_when_no_mention_was_extracted(self):
        adapter = _bare()
        content, ids, _ = adapter._extract_mentions("普通 **文本** 无@")
        msg_type, payload = adapter._build_outbound_payload(content, mention_ids=ids)
        assert msg_type == "post"
        assert _at_ids(payload) == []
        assert payload == _adapter._build_markdown_post_payload("普通 **文本** 无@")

    def test_code_fence_keeps_the_mention_in_its_own_row(self):
        adapter = _bare()
        content, ids, _ = adapter._extract_mentions("说明\n```\npython\nprint(1)\n```\n@GameDev 收尾")
        _, payload = adapter._build_outbound_payload(content, mention_ids=ids)
        rows = json.loads(payload)["zh_cn"]["content"]
        assert rows[-1] == [
            {"tag": "at", "user_id": "ou_1657test00000000000000000000"},
            {"tag": "md", "text": " 收尾"},
        ]


# --- End-to-end through send() ---------------------------------------------


class TestSendShipsRealMentions:
    @pytest.mark.asyncio
    async def test_mapped_name_reaches_the_wire_as_an_at_element(self, monkeypatch):
        adapter = _bare()
        adapter._client = object()
        sent = AsyncMock(return_value=_ok_response())
        monkeypatch.setattr(adapter, "_feishu_send_with_retry", sent)

        result = await adapter.send("oc_1", "@GameDev 请确认")

        assert result.success is True
        kwargs = sent.call_args.kwargs
        assert kwargs["msg_type"] == "post"
        assert _at_ids(kwargs["payload"]) == ["ou_1657test00000000000000000000"]

    @pytest.mark.asyncio
    async def test_unmapped_name_still_sends_plain_text(self, monkeypatch):
        adapter = _bare()
        adapter._client = object()
        sent = AsyncMock(return_value=_ok_response())
        monkeypatch.setattr(adapter, "_feishu_send_with_retry", sent)

        await adapter.send("oc_1", "@Nobody 你好")

        kwargs = sent.call_args.kwargs
        assert _at_ids(kwargs["payload"]) == []
        assert "@Nobody" in kwargs["payload"]

    def test_text_fallback_restores_the_name_instead_of_the_placeholder(self):
        adapter = _bare()
        content, ids, names = adapter._extract_mentions("**标题** @GameDev 请确认")
        restored = adapter._restore_mention_text(content, ids, names)
        assert "@_mention_" not in restored
        assert "@GameDev" in restored
