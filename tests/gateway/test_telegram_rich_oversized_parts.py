"""An oversized rich-eligible Telegram message keeps the rich path (#141272).

Above ``RICH_MESSAGE_MAX_CHARS`` the adapter used to send the whole message through the legacy
MarkdownV2 chunker, so ``<details>`` blocks, tables and task lists arrived as literal text at
4,096-unit boundaries while delivery still reported success. These tests pin that the message is
split into ``sendRichMessage``-sized parts instead, that no part ends inside a ``<details>``
element / fenced block / table, and that the parts reproduce the original content exactly.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``, so a real ``TelegramAdapter``
runs against a mock bot wired for ``do_api_request``.
"""

import re
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_agent.gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
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
async def test_oversized_rich_final_edit_updates_preview_and_sends_rich_continuations():
    """A streamed final (edit_message) must not downgrade details to MarkdownV2 chunks."""
    adapter = _make_adapter()
    adapter._bot.edit_message_text = AsyncMock(return_value=SimpleNamespace(message_id=100))
    next_id = iter(range(201, 250))

    async def rich_request(method, api_kwargs):
        if method == "editMessageText":
            return True
        assert method == "sendRichMessage"
        return SimpleNamespace(message_id=next(next_id))

    adapter._bot.do_api_request = AsyncMock(side_effect=rich_request)
    content = _details_content()
    assert len(content) > adapter.RICH_MESSAGE_MAX_CHARS

    result = await adapter.edit_message("12345", "100", content, finalize=True)

    calls = adapter._bot.do_api_request.await_args_list
    assert result.success is True and len(calls) >= 2
    assert calls[0].args[0] == "editMessageText"
    assert calls[0].kwargs["api_kwargs"]["message_id"] == 100
    assert all(c.args[0] == "sendRichMessage" for c in calls[1:])
    for call in calls:
        rendered = _rich_markdown(call)
        assert len(rendered) <= adapter.RICH_MESSAGE_MAX_CHARS
        assert rendered.count("<details") == rendered.count("</details>")
    continuation_ids = tuple(str(201 + i) for i in range(len(calls) - 1))
    assert result.continuation_message_ids == continuation_ids
    assert result.message_id == continuation_ids[-1]
    adapter._bot.edit_message_text.assert_not_called()
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_oversized_rich_parts_keep_every_details_block_whole():
    adapter = _make_adapter()
    content = _details_content()

    await adapter.send("12345", content)

    parts = [_rich_markdown(call) for call in _rich_calls(adapter)]
    for part in parts:
        assert part.count("<details") == part.count("</details>")
    # Nothing dropped, reordered or re-wrapped: the parts re-join into the original message (after
    # the payload's own normalization).
    assert "\n\n".join(parts) == adapter._rich_message_payload(content)["markdown"]


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


@pytest.mark.asyncio
async def test_a_part_refused_after_earlier_parts_landed_is_never_resent_whole(monkeypatch):
    """Part 2 fails on a transient error: _send_with_retry must not re-send the whole message
    (rich or plain), which would duplicate part 1 in the chat."""
    adapter = _make_adapter()
    monkeypatch.setattr("hermes_agent.gateway.platforms.base.asyncio.sleep", AsyncMock())
    outcomes = iter([SimpleNamespace(message_id=501)] + [RuntimeError("Bad Gateway")] * 10)

    async def do_api_request(method, api_kwargs=None):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    adapter._bot.do_api_request = AsyncMock(side_effect=do_api_request)

    result = await adapter._send_with_retry("12345", _details_content())

    assert result.success is False
    assert result.raw_response["partial_overflow"] is True and result.raw_response["delivered_chunks"] == 1
    first_part = _rich_markdown(adapter._bot.do_api_request.await_args_list[0])
    assert sum(_rich_markdown(c) == first_part for c in adapter._bot.do_api_request.await_args_list) == 1
    # Only the exhausted-retry notice goes out as MarkdownV2; no reply text is re-sent in another shape.
    assert not any("row " in c.kwargs.get("text", "") for c in adapter._bot.send_message.await_args_list)


def test_standalone_send_delivers_an_oversized_rich_message_in_parts(monkeypatch):
    """Cron delivery and the send_message tool use the same rich-parts split as the gateway."""
    import asyncio
    import sys

    from plugins.platforms.telegram.standalone import _send_telegram

    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("hermes_agent.gateway.platforms.base._detect_macos_system_proxy", lambda: None)
    bot = _make_adapter()._bot
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants = SimpleNamespace(ParseMode=parse_mode)
    monkeypatch.setitem(sys.modules, "telegram", SimpleNamespace(Bot=lambda *a, **k: bot, constants=constants))
    monkeypatch.setitem(sys.modules, "telegram.constants", constants)

    result = asyncio.run(_send_telegram("tok", "12345", _details_content(), extra={"rich_messages": True}))

    assert result["success"] is True
    calls = [c for c in bot.do_api_request.await_args_list if c.args[0] == "sendRichMessage"]
    assert len(calls) >= 2 and all(len(_rich_markdown(c)) <= TelegramAdapter.RICH_MESSAGE_MAX_CHARS for c in calls)
    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_raw_text_under_the_cap_whose_sent_markdown_is_over_it_still_goes_out_rich():
    """Normalization (hard breaks) pushes a just-under-cap table report over the cap: it must be
    split at a smaller budget, not handed whole to the MarkdownV2 chunker."""
    adapter = _make_adapter()
    prose = "\n".join(f"line {index:04d} of the long report text" for index in range(940))
    content = "| a | b |\n|---|---|\n| 1 | 2 |\n\n" + prose
    assert len(content) <= adapter.RICH_MESSAGE_MAX_CHARS
    assert len(adapter._rich_message_payload(content)["markdown"]) > adapter.RICH_MESSAGE_MAX_CHARS

    result = await adapter.send("12345", content)

    calls = _rich_calls(adapter)
    assert result.success is True and len(calls) >= 2
    assert all(len(_rich_markdown(call)) <= adapter.RICH_MESSAGE_MAX_CHARS for call in calls)
    adapter._bot.send_message.assert_not_called()


def test_a_four_backtick_fence_holding_a_triple_backtick_example_stays_one_fence():
    inner = "```python\nprint('hi')\n```"
    body = "\n".join(f"step {index}: see the example below" for index in range(30))
    fence = "````markdown\n" + body + "\n" + inner + "\n" + body + "\n````"
    content = "\n\n".join(["intro paragraph", fence, "closing paragraph"])

    whole = split_rich_parts(content, len(fence) + 10)
    assert [part for part in whole if "````markdown" in part] == [fence]

    parts = split_rich_parts(fence, 600)
    assert len(parts) >= 2
    for part in parts:
        lines = part.splitlines()
        assert len(part) <= 600 and lines[0] == "````markdown" and lines[-1] == "````"


class _BadRequest(Exception):
    pass


def _refuse_rich_part(adapter, refused_index):
    """Rich sends succeed except the ``refused_index``-th sendRichMessage, refused for good."""
    seen = {"rich": 0}

    async def do_api_request(method, api_kwargs=None):
        if method == "sendRichMessage":
            seen["rich"] += 1
            if seen["rich"] == refused_index:
                raise _BadRequest("Bad Request: can't parse rich message")
            return SimpleNamespace(message_id=500 + seen["rich"])
        return True

    adapter._bot.do_api_request = AsyncMock(side_effect=do_api_request)
    legacy_ids = iter(range(900, 999))
    adapter._bot.send_message = AsyncMock(side_effect=lambda **kw: SimpleNamespace(message_id=next(legacy_ids)))


def _legacy_texts(adapter):
    return "".join(call.kwargs["text"] for call in adapter._bot.send_message.await_args_list)


@pytest.mark.asyncio
async def test_a_later_part_refused_for_good_sends_the_rest_as_markdown_once(monkeypatch):
    """Part 2 permanently refused: the tail reaches the chat as MarkdownV2 after part 1, and part 1
    is never sent again (redelivery would otherwise re-show it and hit the same refusal)."""
    adapter = _make_adapter()
    monkeypatch.setattr("hermes_agent.gateway.platforms.base.asyncio.sleep", AsyncMock())
    _refuse_rich_part(adapter, 2)
    content = _details_content()

    result = await adapter._send_with_retry("12345", content)

    rich = _rich_calls(adapter)
    assert result.success is True
    assert len(rich) == 2 and _rich_markdown(rich[0]).startswith("# Incident report")
    assert result.raw_response["message_ids"][0] == "501"
    shown = _legacy_texts(adapter).replace("\\", "")
    assert "row 7-519: the build stayed green on main" in shown
    assert "Incident report" not in shown  # the delivered head is not repeated


@pytest.mark.asyncio
async def test_a_refused_continuation_of_a_rich_final_edit_sends_the_rest_as_markdown():
    adapter = _make_adapter()
    _refuse_rich_part(adapter, 1)

    result = await adapter.edit_message("12345", "100", _details_content(), finalize=True)

    calls = adapter._bot.do_api_request.await_args_list
    assert result.success is True and calls[0].args[0] == "editMessageText"
    assert "row 7-519: the build stayed green on main" in _legacy_texts(adapter).replace("\\", "")
    assert result.message_id == result.continuation_message_ids[-1]
    adapter._bot.edit_message_text.assert_not_called()


def test_a_details_block_larger_than_the_cap_is_closed_and_reopened():
    body = "\n".join(f"- row {index}: fine" for index in range(120))
    content = f"<details><summary>Build log</summary>\n\n{body}\n\n</details>"

    parts = split_rich_parts(content, 600)

    assert len(parts) >= 2
    for part in parts:
        assert len(part) <= 600
        assert part.startswith("<details><summary>Build log</summary>")
        assert part.count("<details") == part.count("</details>") == 1


class NetworkError(Exception):
    pass


@pytest.mark.asyncio
async def test_a_streamed_reply_whose_first_preview_failed_shows_rich_part_one_once():
    """The first preview send fails, the split-first send delivers rich part 1 and part 2 fails
    transiently: only the unseen tail goes out again, so every row reaches the chat exactly once."""
    import asyncio

    from hermes_agent.gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from tests.waiting import async_wait_until

    adapter = _make_adapter()
    seen, screen, ids = {"rich": 0, "legacy": 0}, {}, iter(range(100, 999))

    async def do_api_request(method, api_kwargs=None, **_kw):
        if method == "sendRichMessage":
            seen["rich"] += 1
            if seen["rich"] == 2:
                raise NetworkError("Connection reset by peer")
            screen[mid := next(ids)] = api_kwargs["rich_message"]["markdown"]
            return SimpleNamespace(message_id=mid)
        return True

    async def send_message(text="", **_kw):
        seen["legacy"] += 1
        if seen["legacy"] == 1:
            raise NetworkError("Connection reset by peer")
        screen[mid := next(ids)] = text
        return SimpleNamespace(message_id=mid)

    adapter._bot.do_api_request = AsyncMock(side_effect=do_api_request)
    adapter._bot.send_message = AsyncMock(side_effect=send_message)
    content = _details_content(blocks=12, body_lines=120)
    consumer = GatewayStreamConsumer(adapter, "4242", StreamConsumerConfig(
        edit_interval=0.01, buffer_threshold=50, cursor=" ▉"))
    task = asyncio.create_task(consumer.run())
    consumer.on_delta(content[:1000])
    await async_wait_until(lambda: seen["legacy"] >= 1, timeout=5, what="first preview send attempted")
    consumer.on_delta(content[1000:])
    consumer.finish(content)
    await asyncio.wait_for(task, 20)

    rows = Counter(re.findall(r"row \d+-\d+:", "\n".join(screen.values()).replace("\\", "")))
    wanted = re.findall(r"row \d+-\d+:", content)
    assert seen["rich"] >= 2, "rich part 2 was never attempted"
    assert [row for row in wanted if rows[row] != 1] == []


@pytest.mark.asyncio
async def test_a_first_part_that_never_left_keeps_the_normal_retry(monkeypatch):
    """Part 1 refused by a connect error (nothing reached Telegram): the retry sends the whole
    reply instead of stopping at a partial result with no part delivered."""
    import telegram.error as tg_error

    adapter = _make_adapter()
    monkeypatch.setattr("hermes_agent.gateway.platforms.base.asyncio.sleep", AsyncMock())
    attempts = {"rich": 0}

    async def do_api_request(method, api_kwargs=None):
        attempts["rich"] += method == "sendRichMessage"
        if attempts["rich"] == 1:
            raise tg_error.NetworkError("Cannot connect to host api.telegram.org")
        return SimpleNamespace(message_id=600 + attempts["rich"])

    adapter._bot.do_api_request = AsyncMock(side_effect=do_api_request)
    content = _details_content()

    result = await adapter._send_with_retry("12345", content)

    assert result.success is True
    shown = "".join(_rich_markdown(call) for call in _rich_calls(adapter)[1:])
    assert "row 0-0:" in shown and "row 7-519:" in shown


@pytest.mark.asyncio
async def test_a_timed_out_part_names_the_parts_that_never_left():
    """Part 2 times out (it may show): the parts after it never left, so they are reported as
    ``unattempted_chunks`` and the stream consumer sends them before counting the reply shown."""
    import telegram.error as tg_error

    from hermes_agent.gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    adapter = _make_adapter()
    _refuse_rich_part(adapter, 0)
    rich = adapter._bot.do_api_request.side_effect

    async def do_api_request(method, api_kwargs=None):
        if method == "sendRichMessage" and len(_rich_calls(adapter)) == 2:
            raise tg_error.TimedOut("Timed out")
        return await rich(method, api_kwargs)

    adapter._bot.do_api_request = AsyncMock(side_effect=do_api_request)
    result = await adapter.send("12345", _details_content())

    consumer = GatewayStreamConsumer(adapter, "12345", StreamConsumerConfig())
    assert await consumer._uncertain_send_fully_shown(result) is True
    assert "row 7-519: the build stayed green on main" in _legacy_texts(adapter).replace("\\", "")


def test_a_table_without_outer_pipes_repeats_its_header_in_every_part():
    rows = "\n".join(f"name{index} | value {index} with some padding text" for index in range(3000))
    parts = split_rich_parts("name | value\n---|---\n" + rows, 32000)
    assert len(parts) > 1 and all(part.startswith("name | value\n---|---\n") for part in parts)


@pytest.mark.asyncio
async def test_a_later_part_that_never_left_is_resent_rich_after_the_parts_shown(monkeypatch):
    """Part 2 refused by a connect error (it never reached Telegram) after part 1 landed: the
    retry resends part 2 onward as rich parts, never part 1, instead of stopping at a partial."""
    import telegram.error as tg_error

    adapter = _make_adapter()
    monkeypatch.setattr("hermes_agent.gateway.platforms.base.asyncio.sleep", AsyncMock())
    screen, attempts = [], {"rich": 0}

    async def do_api_request(method, api_kwargs=None):
        attempts["rich"] += 1
        if attempts["rich"] == 2:
            raise tg_error.NetworkError("Cannot connect to host api.telegram.org")
        screen.append(api_kwargs["rich_message"]["markdown"])
        return SimpleNamespace(message_id=700 + attempts["rich"])

    adapter._bot.do_api_request = AsyncMock(side_effect=do_api_request)
    content = _details_content(blocks=4, body_lines=400)

    result = await adapter._send_with_retry("12345", content)

    assert result.success is True
    adapter._bot.send_message.assert_not_called()
    rows = re.findall(r"row \d+-\d+:", "\n".join(screen))
    assert sorted(rows) == sorted(re.findall(r"row \d+-\d+:", content))


@pytest.mark.asyncio
async def test_a_final_edit_continuation_accepted_before_its_receipt_timed_out_is_not_resent():
    """The rich final edit lands part 1, Telegram accepts part 2 but the receipt times out: the
    stream fallback sends only the parts after it, so no row shows twice and none is missing."""
    import telegram.error as tg_error

    from hermes_agent.gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    adapter = _make_adapter()
    content = _details_content(blocks=4, body_lines=400)
    screen, sends = {"100": "preview"}, {"rich": 0}

    async def do_api_request(method, api_kwargs=None):
        if method == "editMessageText":
            screen[str(api_kwargs["message_id"])] = api_kwargs["rich_message"]["markdown"]
            return True
        sends["rich"] += 1
        screen[str(200 + sends["rich"])] = api_kwargs["rich_message"]["markdown"]
        if sends["rich"] == 1:
            raise tg_error.TimedOut("Timed out")
        return SimpleNamespace(message_id=200 + sends["rich"])

    async def send_message(text="", **_kw):
        screen[mid := str(1000 + len(screen))] = text
        return SimpleNamespace(message_id=mid)

    adapter._bot.do_api_request = AsyncMock(side_effect=do_api_request)
    adapter._bot.send_message = AsyncMock(side_effect=send_message)
    consumer = GatewayStreamConsumer(adapter, "12345", StreamConsumerConfig())
    consumer._message_id, consumer._last_sent_text = "100", "preview"
    consumer._accumulated = consumer._stream_ledger = content

    if not await consumer._edit_existing(content, finalize=True, is_turn_final=True):
        await consumer._send_fallback_final(content)

    rows = Counter(re.findall(r"row \d+-\d+:", "\n".join(screen.values()).replace("\\", "")))
    assert [row for row in re.findall(r"row \d+-\d+:", content) if rows[row] != 1] == []
    assert consumer.delivery_uncertain is True
