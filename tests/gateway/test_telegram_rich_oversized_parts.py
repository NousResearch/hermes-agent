"""An oversized rich-eligible Telegram message keeps the rich path (#141272).

Above ``RICH_MESSAGE_MAX_CHARS`` the adapter used to send the whole message through the legacy
MarkdownV2 chunker, so ``<details>`` blocks, tables and task lists arrived as literal text at
4,096-unit boundaries while delivery still reported success. These tests pin that the message is
split into ``sendRichMessage``-sized parts instead, that no part ends inside a ``<details>``
element / fenced block / table, and that the parts reproduce the original content exactly.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``, so a real ``TelegramAdapter``
runs against a mock bot wired for ``do_api_request``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter, _rich_normalize_linebreaks
from plugins.platforms.telegram.rich_parts import split_rich_parts


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="fake-token", extra={"rich_messages": True, **(extra or {})})
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()
    bot.delete_message = AsyncMock(return_value=True)
    adapter._bot = bot
    return adapter


def _details_content(blocks=8, body_lines=520):
    """An oversized message whose blocks are ``<details>`` elements (rich-only constructs)."""
    sections = ["# Incident report"]
    for index in range(blocks):
        body = "\n".join(f"row {index}-{line}: the build stayed green on main" for line in range(body_lines))
        sections.append(f"<details><summary>Section {index}</summary>\n\n{body}\n\n</details>")
    return "\n\n".join(sections)


def _rich_calls(adapter):
    return [call for call in adapter._bot.do_api_request.await_args_list if call.args[0] == "sendRichMessage"]


def _rich_markdown(call):
    return call.kwargs["api_kwargs"]["rich_message"]["markdown"]


@pytest.mark.asyncio
async def test_oversized_details_message_is_delivered_in_rich_parts():
    adapter = _make_adapter()
    content = _details_content()
    assert len(content) > adapter.RICH_MESSAGE_MAX_CHARS

    result = await adapter.send("12345", content)

    assert result.success is True
    calls = _rich_calls(adapter)
    assert len(calls) >= 2
    adapter._bot.send_message.assert_not_called()
    for call in calls:
        assert len(_rich_markdown(call)) <= adapter.RICH_MESSAGE_MAX_CHARS


@pytest.mark.asyncio
async def test_oversized_rich_parts_keep_every_details_block_whole():
    adapter = _make_adapter()
    content = _details_content()

    await adapter.send("12345", content)

    parts = [_rich_markdown(call) for call in _rich_calls(adapter)]
    for part in parts:
        assert part.count("<details") == part.count("</details>")
    # Nothing dropped, reordered or re-wrapped: the parts re-join into the original message (after
    # the payload's own soft-break normalization).
    assert "\n\n".join(parts) == _rich_normalize_linebreaks(content)


@pytest.mark.asyncio
async def test_reply_anchor_covers_the_first_part_only():
    adapter = _make_adapter()
    content = _details_content()

    await adapter.send("12345", content, reply_to="777")

    calls = _rich_calls(adapter)
    assert calls[0].kwargs["api_kwargs"]["reply_parameters"] == {"message_id": 777}
    for call in calls[1:]:
        assert "reply_parameters" not in call.kwargs["api_kwargs"]


@pytest.mark.asyncio
async def test_oversized_cjk_message_still_uses_the_legacy_path():
    """The Desktop CJK-rich garble guard outranks size: parts stay on MarkdownV2."""
    adapter = _make_adapter()
    content = _details_content().replace("Incident report", "事故报告")

    result = await adapter.send("12345", content)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_called()


def test_split_rich_parts_keeps_fences_and_tables_whole():
    fence = "```python\n" + "\n".join(f"print({index})" for index in range(40)) + "\n```"
    table = "\n".join(["| a | b |", "|---|---|"] + [f"| {index} | value {index} |" for index in range(30)])
    content = "\n\n".join(["intro paragraph", fence, table, "closing paragraph"])

    parts = split_rich_parts(content, 700)

    assert all(len(part) <= 700 for part in parts)
    fence_parts = [part for part in parts if "```" in part]
    assert len(fence_parts) == 1 and fence_parts[0].count("```") == 2
    assert "print(39)" in fence_parts[0]
    table_parts = [part for part in parts if "| a | b |" in part]
    assert len(table_parts) == 1 and "| 29 | value 29 |" in table_parts[0]
    assert "\n\n".join(parts) == content


def test_a_fence_larger_than_the_cap_is_closed_and_reopened():
    fence = "```python\n" + "\n".join(f"value = {index}" for index in range(80)) + "\n```"
    assert len(fence) > 300

    parts = split_rich_parts(fence, 300)

    assert len(parts) >= 2
    for part in parts:
        assert len(part) <= 300
        assert part.count("```") == 2
        assert part.splitlines()[0].startswith("```python")


def test_a_table_larger_than_the_cap_repeats_its_header():
    table = "| n | v |\n|---|---|\n" + "\n".join(f"| {index} | {index * index} |" for index in range(80))
    assert len(table) > 300

    parts = split_rich_parts(table, 300)

    assert len(parts) >= 2
    for part in parts:
        assert len(part) <= 300
        assert part.splitlines()[0] == "| n | v |"
        assert part.splitlines()[1] == "|---|---|"


def test_split_rich_parts_cuts_a_single_oversized_block_on_line_boundaries():
    content = "x" * 90 + "\n" + "y" * 90 + "\n" + "z" * 90

    parts = split_rich_parts(content, 100)

    assert all(len(part) <= 100 for part in parts)
    assert [part.strip() for part in parts] == ["x" * 90, "y" * 90, "z" * 90]


def test_split_rich_parts_returns_the_original_when_it_fits():
    content = "<details><summary>small</summary>\nbody\n</details>"

    assert split_rich_parts(content, 4096) == [content]
