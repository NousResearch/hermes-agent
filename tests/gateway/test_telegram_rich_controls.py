"""Rich control conversion + send/edit contracts: escaping boundaries, callback passthrough,
MarkdownV2 preservation, permanent-vs-transient fallback, registry lifecycle, query facade.

No PTB required: keyboards are fed as ``to_dict()``-shaped dicts; the mixin host is a bare
object (the adapter's ``_is_rich_fallback_error`` / ``_coerce_bool_extra`` / latching are
simulated with the adapter's own semantics via small stubs).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from plugins.platforms.telegram import telegram_rich_controls as rc
from plugins.platforms.telegram.telegram_rich_controls import (
    TelegramRichControlsMixin, buttons_html, esc_attr, esc_text, markup_to_tg_button_rows,
    mdv2_to_html, rich_control_html, rich_control_payload)


def keyboard(*rows):
    """InlineKeyboardMarkup.to_dict() shape; each *rows* argument is a tuple of
    (label, callback_data) buttons rendered as ONE row."""
    return {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in rows]}


class Host(TelegramRichControlsMixin):
    """Bare mixin host: adapter semantics copied at minimum fidelity."""

    def __init__(self, rich_controls=True, rich_send_disabled=False):
        self.config = SimpleNamespace(extra={"rich_controls": rich_controls})
        self._rich_send_disabled = rich_send_disabled
        self._bot = SimpleNamespace(do_api_request=AsyncMock(return_value={"message_id": 555}))
        self.sent = []

    def _coerce_bool_extra(self, key, default=False):
        value = self.config.extra.get(key)
        return self.config.extra.get(key, default) if not isinstance(value, str) else value.lower() in {"1", "true", "yes", "on"}

    def _is_rich_fallback_error(self, exc):
        s = str(exc).lower()
        return "bad request" in s or "unsupported" in s or "not implemented" in s or "method not found" in s


def ok_message(message_id=555):
    return {"message_id": message_id}


# --- escaping / security boundaries --------------------------------------------------------
def test_esc_attr_neutralizes_quotes_and_markup_in_callback_data():
    assert esc_attr('mp:1"><tg-button data="evil">') == 'mp:1&quot;&gt;&lt;tg-button data=&quot;evil&quot;&gt;'


def test_esc_text_escapes_label_entities():
    assert esc_text("<b>&") == "&lt;b&gt;&amp;"


def test_callback_prefixes_pass_through_verbatim():
    rows = markup_to_tg_button_rows(keyboard((("Approve", "ea:once:7"),), (("Cancel", "sc:cancel:42"),)))
    assert [b["callback_data"] for row in rows for b in row] == ["ea:once:7", "sc:cancel:42"]


def test_buttons_html_embeds_callback_data_attr_safely():
    html = buttons_html([markup_to_tg_button_rows(keyboard((("Do", "cp:0"), ("No", "cp:1"))))[0]])[0]


def test_malicious_label_cannot_break_out_of_button_tag():
    html = buttons_html([markup_to_tg_button_rows(keyboard((('x</tg-button><tg-button data="hijack">', "cp:0"),)))[0]])
    # Exactly one real button tag; the label's markup is inert (escaped) text — no second tag.
    assert html.count("<tg-button ") == 1
    assert html.count("</tg-button>") == 1
    assert "&lt;tg-button" in html  # injected tag escaped, inert
    assert html.endswith("</tg-button-row>")


def test_non_callback_buttons_dropped_and_empty_rows_removed():
    markup = {"inline_keyboard": [[
        {"text": "Site", "url": "https://x"}, {"text": "Go", "callback_data": "cp:0"},
        {"text": "Web", "web_app": {"url": "https://y"}}], [{"text": "only-url", "url": "https://z"}]]}
    rows = markup_to_tg_button_rows(markup)
    assert rows == [[{"text": "Go", "callback_data": "cp:0"}]]
    assert buttons_html([]) == ""


def test_callback_data_byte_bounds_enforced():
    assert markup_to_tg_button_rows(keyboard((("empty", ""),),)) == []
    assert markup_to_tg_button_rows(keyboard((("long", "x" * 65),),)) == []
    assert markup_to_tg_button_rows(keyboard((("ok", "x" * 64),),)) == [[{"text": "ok", "callback_data": "x" * 64}]]


def test_row_button_cap_enforced():
    row = [("b", f"cp:{i}") for i in range(10)]
    rows = markup_to_tg_button_rows(keyboard(row,))
    assert len(rows[0]) == 8


def test_style_inference_whitelisted_only():
    assert "danger" in buttons_html([markup_to_tg_button_rows(keyboard((("Cancel", "cp:0"),)))[0]])
    assert "success" in buttons_html([markup_to_tg_button_rows(keyboard((("Approve once", "cp:0"),)))[0]])
    assert "style" not in buttons_html([markup_to_tg_button_rows(keyboard((("Model X", "mp:0"),)))[0]])


# --- MarkdownV2 → HTML conversion -----------------------------------------------------------
def test_mdv2_preserves_bold_italic_code_and_specials():
    html = mdv2_to_html(r"Header *bold* _italic_ `code\`x` a\.b \(c\) \-d")
    assert html == "Header <b>bold</b> <i>italic</i> <code>code`x</code> a.b (c) -d"


def test_mdv2_escaped_marker_is_literal_not_formatting():
    # An escaped marker renders literally — no <b> is emitted.
    assert mdv2_to_html(r"Header \*bold\*") == "Header *bold*"


def test_mdv2_preserves_pre_block_and_link():
    html = mdv2_to_html("```\nfn\\(x\\) \\`\n```\nsee [docs](https://t.me)")
    assert html == '<pre>\nfn(x) `\n</pre>\nsee <a href="https://t.me">docs</a>'


def test_mdv2_spoiler_and_underline():
    html = mdv2_to_html(r"||secret|| __under__")
    assert html == "<tg-spoiler>secret</tg-spoiler> <u>under</u>"


def test_mdv2_escapes_unpaired_markers():
    html = mdv2_to_html(r"literal \<script\> 2\*3")
    assert html == "literal &lt;script&gt; 2*3"

def test_mdv2_escapes_raw_entities_in_bare_text():
    assert mdv2_to_html("*bold* & <tag>") == "<b>bold</b> &amp; &lt;tag&gt;"


def test_plain_mode_escapes_body():
    html = rich_control_html("<i>raw</i>", None)
    assert html == "&lt;i&gt;raw&lt;/i&gt;"


def test_html_mode_passes_body_through():
    html = rich_control_html("<b>card</b>", "HTML")
    assert html == "<b>card</b>"


def test_payload_is_exactly_html_field():
    assert rich_control_payload("t", "MarkdownV2", keyboard((("A", "ea:once:1"),))) == {
        "html": 't\n<tg-button-row><tg-button type="callback_data" data="ea:once:1">A</tg-button></tg-button-row>'}

# --- _try_send_rich_control contract --------------------------------------------------------
@pytest.mark.asyncio
async def test_rich_control_send_success_registers_and_returns_message_like():
    host = Host()
    result = await host._try_send_rich_control(
        {"chat_id": 123, "text": "Proceed?", "parse_mode": "MarkdownV2", "reply_markup": keyboard((("Yes", "ea:once:1"),))})
    assert result.success and result.message_id == "555"
    assert result.raw_response["rich_control"] and result.raw_response["message"].message_id == 555
    assert host.is_rich_control_message(123, 555)
    payload = host._bot.do_api_request.call_args.kwargs["api_kwargs"]
    assert payload["chat_id"] == 123
    assert '<tg-button type="callback_data" data="ea:once:1"' in payload["rich_message"]["html"]
    assert "message_id" not in payload and "text" not in payload


@pytest.mark.asyncio
async def test_rich_control_send_threads_and_anchor_route_like_sendRichMessage():
    host = Host()
    await host._try_send_rich_control(
        {"chat_id": "123", "text": "t", "parse_mode": None, "message_thread_id": "9",
         "reply_to_message_id": "31", "disable_notification": True})
    payload = host._bot.do_api_request.call_args.kwargs["api_kwargs"]
    assert payload["message_thread_id"] == 9  # int, not raw string thread
    assert payload["reply_parameters"] == {"message_id": 31}
    assert payload["disable_notification"] is True


@pytest.mark.asyncio
async def test_permanent_rejection_returns_none_for_legacy_fallback():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Bad Request: BUTTON_DATA_INVALID"))
    result = await host._try_send_rich_control({"chat_id": 1, "text": "t", "parse_mode": None})
    assert result is None


@pytest.mark.asyncio
async def test_capability_rejection_returns_none():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Method Not Found"))
    assert await host._try_send_rich_control({"chat_id": 1, "text": "t", "parse_mode": None}) is None


@pytest.mark.asyncio
async def test_transient_failure_is_ambiguous_and_non_retryable():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=TimeoutError("timed out"))
    result = await host._try_send_rich_control({"chat_id": 1, "text": "t", "parse_mode": None})
    assert isinstance(result, SendResult)
    assert not result.success and not result.retryable
    assert result.raw_response["ambiguous"] is True and result.raw_response["what"] == "sendRichMessage"
    assert not host.is_rich_control_message(1, 555)  # nothing registered: no fake registry entry


@pytest.mark.asyncio
async def test_disabled_flag_and_latch_short_circuit_to_none():
    off = Host(rich_controls=False)
    assert await off._try_send_rich_control({"chat_id": 1, "text": "t"}) is None
    assert off._bot.do_api_request.call_count == 0
    latched = Host(rich_send_disabled=True)
    assert await latched._try_send_rich_control({"chat_id": 1, "text": "t"}) is None


@pytest.mark.asyncio
async def test_oversized_payload_falls_back_without_api_call():
    host = Host()
    assert await host._try_send_rich_control({"chat_id": 1, "text": "x" * 40000, "parse_mode": None}) is None
    assert host._bot.do_api_request.call_count == 0


# --- registry lifecycle -----------------------------------------------------------------------
def test_registry_bounded_and_fifo():
    host = Host()
    for i in range(rc._RICH_CONTROL_REGISTRY_CAP + 10):
        host.register_rich_control_message(1, i)
    reg = host._rich_control_messages
    assert len(reg) == rc._RICH_CONTROL_REGISTRY_CAP
    assert (str(1), str(rc._RICH_CONTROL_REGISTRY_CAP + 9)) in reg
    assert (str(1), "0") not in reg


def test_forget_drops_entry():
    host = Host()
    host.register_rich_control_message(7, 8)
    host.forget_rich_control_message(7, 8)
    assert not host.is_rich_control_message(7, 8)


# --- query facade ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_wrap_returns_raw_query_for_unregistered_message():
    host = Host()
    raw = SimpleNamespace(edit_message_text=AsyncMock())
    assert host.wrap_rich_control_query(raw) is raw


@pytest.mark.asyncio
async def test_facade_rich_edits_registered_control_message():
    host = Host()
    host.register_rich_control_message(123, 555)
    raw = SimpleNamespace(edit_message_text=AsyncMock())
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555))
    facade = host.wrap_rich_control_query(query)
    # Facade only enriches edit_message_text; raw attribute access delegates.
    assert facade.message.message_id == 555
    await facade.edit_message_text("Done", parse_mode="MarkdownV2", reply_markup=None)
    call = host._bot.do_api_request.call_args
    assert call.args[0] == "editMessageText"
    payload = call.kwargs["api_kwargs"]
    assert payload["message_id"] == 555 and "rich_message" in payload
    # Terminal edit (no keyboard) → registry entry dropped.
    assert not host.is_rich_control_message(123, 555)


@pytest.mark.asyncio
async def test_facade_pagination_edit_keeps_registry_entry():
    host = Host()
    host.register_rich_control_message(123, 555)
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555))
    facade = host.wrap_rich_control_query(query)
    await facade.edit_message_text("Page 2", parse_mode="HTML", reply_markup=keyboard((("m2", "mm:2"),)))
    assert host.is_rich_control_message(123, 555)


@pytest.mark.asyncio
async def test_facade_falls_back_to_legacy_edit_on_permanent_rejection():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Bad Request: BUTTON_PAYLOAD_INVALID"))
    host.register_rich_control_message(123, 555)
    raw_edit = AsyncMock()
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555), edit_message_text=raw_edit)
    facade = host.wrap_rich_control_query(query)
    await facade.edit_message_text(text="legacy", parse_mode="MarkdownV2", reply_markup=None)
    raw_edit.assert_awaited_once_with(text="legacy", parse_mode="MarkdownV2", reply_markup=None)


@pytest.mark.asyncio
async def test_facade_no_legacy_edit_on_rich_transient_failure():
    """Transient rich-edit failure must NOT trigger a legacy edit (the edit may have landed; both
    would race) — the facade lets the exception surface to the caller's error handling."""
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=TimeoutError("timed out"))
    host.register_rich_control_message(123, 555)
    raw_edit = AsyncMock()
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555), edit_message_text=raw_edit)
    facade = host.wrap_rich_control_query(query)
    with pytest.raises(TimeoutError):
        await facade.edit_message_text(text="x", parse_mode=None, reply_markup=None)
    raw_edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_facade_legacy_edit_not_attempted_on_not_modified_noop():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Bad Request: message is not modified"))
    host.register_rich_control_message(123, 555)
    raw_edit = AsyncMock()
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555), edit_message_text=raw_edit)
    facade = host.wrap_rich_control_query(query)
    await facade.edit_message_text(text="same", parse_mode=None, reply_markup=None)
    raw_edit.assert_not_awaited()
    assert not host.is_rich_control_message(123, 555)


# --- _edit_control_rich direct contract ------------------------------------------------------
@pytest.mark.asyncio
async def test_edit_control_rich_transient_is_ambiguous_result():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=TimeoutError("timed out"))
    result = await host._edit_control_rich(1, 2, "text", None, None)
    assert not result.success and not result.retryable
    assert result.raw_response["ambiguous"] is True


@pytest.mark.asyncio
async def test_edit_control_rich_permanent_returns_none():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Unsupported rich edit"))
    assert await host._edit_control_rich(1, 2, "text", None, None) is None
