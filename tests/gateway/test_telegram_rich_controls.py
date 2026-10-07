"""Rich control conversion + send/edit contracts: escaping boundaries, callback passthrough,
ALL Bot API 10.3 button actions (url/web_app/login_url/switch_inline_query*/copy_text/disabled),
explicit styles + row alignment, MarkdownV2 preservation, permanent-vs-transient fallback,
registry lifecycle, query facade (positional + keyword edit calls).

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
    NativeControlMarkup, TelegramRichControlsMixin, buttons_html, esc_attr, esc_text, legacy_control_markup,
    markup_to_tg_button_rows, mdv2_to_html, rich_control_html, rich_control_markup_supported, rich_control_payload)


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

    def _coerce_bool_extra(self, key, default=False):
        value = self.config.extra.get(key)
        return self.config.extra.get(key, default) if not isinstance(value, str) else value.lower() in {"1", "true", "yes", "on"}

    def _is_rich_fallback_error(self, exc):
        s = str(exc).lower()
        return "bad request" in s or "unsupported" in s or "not implemented" in s or "method not found" in s



# --- escaping / security boundaries --------------------------------------------------------
def test_esc_attr_neutralizes_quotes_and_markup_in_callback_data():
    assert esc_attr('mp:1"><tg-button data="evil">') == 'mp:1&quot;&gt;&lt;tg-button data=&quot;evil&quot;&gt;'


def test_esc_text_escapes_label_entities():
    assert esc_text("<b>&") == "&lt;b&gt;&amp;"


def test_callback_prefixes_pass_through_verbatim():
    rows = markup_to_tg_button_rows(keyboard((("Approve", "ea:once:7"),), (("Cancel", "sc:cancel:42"),)))
    assert [b["action"][1] for row in rows for b in row] == ["ea:once:7", "sc:cancel:42"]


def test_malicious_label_cannot_break_out_of_button_tag():
    html = buttons_html([markup_to_tg_button_rows(keyboard((('x</tg-button><tg-button data="hijack">', "cp:0"),)))[0]])
    # Exactly one real button tag; the label's markup is inert (escaped) text — no second tag.
    assert html.count("<tg-button ") == 1
    assert html.count("</tg-button>") == 1
    assert "&lt;tg-button" in html  # injected tag escaped, inert
    assert html.endswith("</tg-button-row>")


def test_malicious_url_attribute_cannot_break_out():
    markup = {"inline_keyboard": [[{"text": "L", "url": 'https://x"><tg-button data="evil">'}]]}
    html = rich_control_html("t", None, markup)
    assert html.count("<tg-button ") == 1
    assert 'url="https://x&quot;&gt;&lt;tg-button' in html


def test_malicious_copy_text_and_login_url_attributes_escaped():
    markup = {"inline_keyboard": [[
        {"text": "C", "copy_text": {"text": 'x"><tg-button type="url">'}}]]}
    html = rich_control_html("t", None, markup)
    assert html.count("<tg-button ") == 1
    markup2 = {"inline_keyboard": [[
        {"text": "L", "login_url": {"url": 'https://e"><tg-button', "forward_text": 'f">x'}}]]}
    html2 = rich_control_html("t", None, markup2)
    assert html2.count("<tg-button ") == 1


# --- ALL action kinds round-trip -------------------------------------------------------------
def test_every_action_kind_converts_to_official_grammar():
    markup = {"inline_keyboard": [
        [{"text": "Docs", "url": "https://t.me"},
         {"text": "Me", "url": "tg://user?id=777000"},
         {"text": "App", "web_app": {"url": "https://telegram.org"}}],
        [{"text": "Login", "login_url": {"url": "https://t.me", "forward_text": "fwd",
                                         "request_write_access": True}}],
        [{"text": "Inline", "switch_inline_query": "inline"},
         {"text": "Here", "switch_inline_query_current_chat": "inline 2"},
         {"text": "Chosen", "switch_inline_query_chosen_chat": {
             "query": "inline 3", "allow_user_chats": True, "allow_bot_chats": True,
             "allow_group_chats": True, "allow_channel_chats": True}}],
        [{"text": "Copy", "copy_text": {"text": "...copy"}},
         {"text": "Dead", "disabled": {}}]]}
    html = rich_control_html("t", None, markup)
    assert '<tg-button type="url" url="https://t.me">Docs</tg-button>' in html
    assert '<tg-button type="url" url="tg://user?id=777000">Me</tg-button>' in html
    assert '<tg-button type="web_app" url="https://telegram.org">App</tg-button>' in html
    assert ('<tg-button type="login_url" url="https://t.me" forward-text="fwd" request-write-access>'
            'Login</tg-button>') in html
    assert '<tg-button type="switch_inline_query" query="inline">Inline</tg-button>' in html
    assert '<tg-button type="switch_inline_query_current_chat" query="inline 2">Here</tg-button>' in html
    assert ('<tg-button type="switch_inline_query_chosen_chat" query="inline 3" '
            'allow-user-chats allow-bot-chats allow-group-chats allow-channel-chats>Chosen</tg-button>') in html
    assert '<tg-button type="copy_text" text="...copy">Copy</tg-button>' in html
    # Official grammar: a disabled button carries ONLY the type (no action attrs).
    assert '<tg-button type="disabled">Dead</tg-button>' in html
    assert rich_control_markup_supported(markup)


def test_empty_inline_query_preserved_as_empty_query_attr():
    markup = {"inline_keyboard": [[
        {"text": "E", "switch_inline_query": ""},
        {"text": "E2", "switch_inline_query_current_chat": ""}]]}
    html = rich_control_html("", None, markup)
    assert '<tg-button type="switch_inline_query" query="">E</tg-button>' in html
    assert '<tg-button type="switch_inline_query_current_chat" query="">E2</tg-button>' in html


def test_chosen_chat_with_no_query_still_emits_empty_query():
    markup = {"inline_keyboard": [[{"text": "C", "switch_inline_query_chosen_chat": {"allow_group_chats": True}}]]}
    html = rich_control_html("", None, markup)
    assert '<tg-button type="switch_inline_query_chosen_chat" query="" allow-group-chats>C</tg-button>' in html


def test_disabled_marker_renders_inactive_dropping_carrier_action():
    # PTB 22.8 has no `disabled` param: callers ride button-level api_kwargs next to a carrier.
    markup = {"inline_keyboard": [[
        {"text": "Picked", "callback_data": "ea:once:1", "disabled": {}},
        {"text": "Cancel", "callback_data": "sc:cancel:2"}]]}
    html = rich_control_html("t", None, markup)
    assert '<tg-button type="disabled">Picked</tg-button>' in html
    assert "ea:once:1" not in html  # carrier action dropped: the button does nothing
    assert 'type="callback_data" data="sc:cancel:2"' in html  # siblings keep their actions


def test_disabled_button_via_direct_field_empty_dict():
    markup = {"inline_keyboard": [[{"text": "Dead", "disabled": {}}]]}
    html = rich_control_html("t", None, markup)
    assert html == 't\n<tg-button-row><tg-button type="disabled">Dead</tg-button></tg-button-row>'


def test_disabled_login_url_carrier_also_inactive():
    markup = {"inline_keyboard": [[
        {"text": "L", "login_url": {"url": "https://t.me"}, "disabled": {}}]]}
    html = rich_control_html("", None, markup)
    assert '<tg-button type="disabled">L</tg-button>' in html
    assert "https://t.me" not in html


# --- non-representable buttons → whole markup rejected (never silently dropped) -------------
def test_callback_game_button_makes_markup_unsupported():
    markup = {"inline_keyboard": [[
        {"text": "Play", "callback_game": {}}, {"text": "Go", "callback_data": "cp:0"}]]}
    assert rich_control_markup_supported(markup) is False
    assert markup_to_tg_button_rows(markup) == []


def test_pay_button_makes_markup_unsupported():
    markup = {"inline_keyboard": [[{"text": "Pay", "pay": True}]]}
    assert rich_control_markup_supported(markup) is False


def test_actionless_button_makes_markup_unsupported():
    markup = {"inline_keyboard": [[{"text": "Nothing"}]]}
    assert rich_control_markup_supported(markup) is False


def test_malformed_callback_data_makes_markup_unsupported_not_dropped():
    markup = {"inline_keyboard": [[
        {"text": "empty", "callback_data": ""}, {"text": "Go", "callback_data": "cp:0"}]]}
    assert rich_control_markup_supported(markup) is False
    markup2 = {"inline_keyboard": [[
        {"text": "long", "callback_data": "x" * 65}, {"text": "Go", "callback_data": "cp:0"}]]}
    assert rich_control_markup_supported(markup2) is False


def test_malformed_subobject_buttons_unsupported():
    for bad in ({"text": "W", "web_app": "https://not-a-dict"},
                {"text": "L", "login_url": None},
                {"text": "C", "copy_text": 42},
                {"text": "U", "url": None},
                {"text": "S", "switch_inline_query": 3}):
        assert rich_control_markup_supported({"inline_keyboard": [[bad]]}) is False


@pytest.mark.asyncio
async def test_unsupported_markup_blocks_rich_send_gate():
    host = Host()
    markup = {"inline_keyboard": [[{"text": "Play", "callback_game": {}}]]}
    result = await host._try_send_rich_control({"chat_id": 1, "text": "t", "reply_markup": markup})
    assert result is None and host._bot.do_api_request.call_count == 0




# --- bounds: split, never silently truncate --------------------------------------------------
def test_row_button_cap_splits_extras_into_new_rows():
    row = [("b", f"cp:{i}") for i in range(10)]
    rows = markup_to_tg_button_rows(keyboard(row,))
    assert [len(r) for r in rows] == [8, 2]
    assert [b["action"][1] for b in rows[0]] == [f"cp:{i}" for i in range(8)]
    assert [b["action"][1] for b in rows[1]] == ["cp:8", "cp:9"]


def test_empty_rows_dropped():
    markup = {"inline_keyboard": [[], [{"text": "Go", "callback_data": "cp:0"}]]}
    rows = markup_to_tg_button_rows(markup)
    assert len(rows) == 1 and rows[0][0]["action"][1] == "cp:0"
    assert buttons_html([]) == ""


def test_callback_data_byte_bounds():
    assert rich_control_markup_supported(keyboard((("ok", "x" * 64),),)) is True
    assert rich_control_markup_supported(keyboard((("long", "x" * 65),),)) is False
    rows = markup_to_tg_button_rows(keyboard((("long", "x" * 65),),))
    assert rows == []


# --- styles: explicit beats inferred; link restricted to callback ----------------------------
def test_style_inference_whitelisted_only():
    assert 'style="danger"' in buttons_html([markup_to_tg_button_rows(keyboard((("Cancel", "cp:0"),)))[0]])
    assert 'style="success"' in buttons_html([markup_to_tg_button_rows(keyboard((("Approve once", "cp:0"),)))[0]])
    assert "style" not in buttons_html([markup_to_tg_button_rows(keyboard((("Model X", "mp:0"),)))[0]])


def test_explicit_style_beats_inference():
    markup = {"inline_keyboard": [[{"text": "Cancel", "callback_data": "cp:0", "style": "primary"}]]}
    html = rich_control_html("", None, markup)
    assert 'style="primary"' in html and 'style="danger"' not in html


def test_link_style_allowed_only_on_callback_buttons():
    cb = {"inline_keyboard": [[{"text": "Open", "callback_data": "op:1", "style": "link"}]]}
    assert 'style="link"' in rich_control_html("", None, cb)
    url = {"inline_keyboard": [[{"text": "View", "url": "https://t.me", "style": "link"}]]}
    assert "style" not in rich_control_html("", None, url)
    disabled = {"inline_keyboard": [[{"text": "D", "callback_data": "x", "disabled": {}, "style": "link"}]]}
    assert "style" not in rich_control_html("", None, disabled)


def test_unknown_explicit_style_falls_back_to_inference():
    markup = {"inline_keyboard": [[{"text": "Cancel", "callback_data": "cp:0", "style": "neon"}]]}
    html = rich_control_html("", None, markup)
    assert 'style="danger"' in html and "neon" not in html


# --- row alignment (markup api_kwargs: align / row_alignments) -------------------------------
def test_markup_level_align_applies_to_all_rows():
    markup = {"inline_keyboard": [[{"text": "A", "callback_data": "a"}],
                                  [{"text": "B", "url": "https://t.me"}]], "align": "right"}
    html = rich_control_html("t", None, markup)
    assert html.count('<tg-button-row align="right">') == 2


def test_row_alignments_apply_per_row():
    markup = {"inline_keyboard": [[{"text": "A", "callback_data": "a"}],
                                  [{"text": "B", "callback_data": "b"}]],
              "row_alignments": ["left", "center"]}
    html = rich_control_html("t", None, markup)
    assert '<tg-button-row align="left">' in html
    assert '<tg-button-row align="center">' in html


def test_split_rows_inherit_source_row_alignment():
    big_row = [{"text": f"b{i}", "callback_data": f"cp:{i}"} for i in range(10)]
    markup = {"inline_keyboard": [big_row], "row_alignments": ["center"]}
    html = rich_control_html("", None, markup)
    assert html.count('<tg-button-row align="center">') == 2


def test_invalid_or_missing_alignments_drop_the_attribute():
    markup = {"inline_keyboard": [[{"text": "A", "callback_data": "a"}],
                                  [{"text": "B", "callback_data": "b"}],
                                  [{"text": "C", "callback_data": "c"}]],
              "align": "center", "row_alignments": ["left", "bogus"]}
    html = rich_control_html("t", None, markup)
    assert '<tg-button-row align="left">' in html   # valid per-row entry wins
    assert '<tg-button-row>' in html                # invalid per-row entry → dropped
    assert '<tg-button-row align="center">' in html  # missing entry → global align


def test_alignment_from_ptb_markup_api_kwargs_to_dict_shape():
    # InlineKeyboardMarkup(api_kwargs={"align": ...}).to_dict() flattens api_kwargs at top level.
    markup = {"inline_keyboard": [[{"text": "A", "callback_data": "a"}]], "align": "right"}
    assert rich_control_markup_supported(markup) is True
    assert '<tg-button-row align="right">' in rich_control_html("", None, markup)


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


def test_mdv2_formatted_content_cannot_inject_html_or_link_attributes():
    assert mdv2_to_html('*<tg-button data="evil">&*') == '<b>&lt;tg-button data="evil"&gt;&amp;</b>'
    assert mdv2_to_html('[docs](https://example.com/?x="quoted"&y=1)') == (
        '<a href="https://example.com/?x=&quot;quoted&quot;&amp;y=1">docs</a>')
    assert mdv2_to_html(r'[docs](https://example.com/?x=\"quoted\")') == (
        '<a href="https://example.com/?x=&quot;quoted&quot;">docs</a>')


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


@pytest.mark.asyncio
async def test_ambiguous_rich_control_result_is_not_legacy_resent():
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={"rich_controls": True}))
    adapter._private_controls = False
    adapter._bot = SimpleNamespace(
        do_api_request=AsyncMock(side_effect=TimeoutError("timed out")),
        send_message=AsyncMock(),
    )
    result = await adapter._send_prompt(
        "control", "123", None,
        lambda: ("Proceed?", {"inline_keyboard": [[{"text": "Yes", "callback_data": "cp:1"}]]}, None),
    )
    assert result.success is False and result.raw_response["ambiguous"] is True
    adapter._bot.send_message.assert_not_called()


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

@pytest.mark.asyncio
async def test_send_with_url_and_disabled_buttons_embedded():
    host = Host()
    markup = {"inline_keyboard": [
        [{"text": "Docs", "url": "https://t.me"},
         {"text": "Picked", "callback_data": "ea:once:9", "disabled": {}}]]}
    result = await host._try_send_rich_control({"chat_id": 5, "text": "Choose", "reply_markup": markup})
    assert result.success
    html = host._bot.do_api_request.call_args.kwargs["api_kwargs"]["rich_message"]["html"]
    assert '<tg-button type="url" url="https://t.me">Docs</tg-button>' in html
    assert '<tg-button type="disabled">Picked</tg-button>' in html


@pytest.mark.asyncio
async def test_edit_control_rich_preserves_actions_styles_alignment():
    host = Host()
    markup = {"inline_keyboard": [
        [{"text": "Cancel", "callback_data": "sc:cancel:1", "style": "danger"},
         {"text": "Retry", "callback_data": "sc:retry:1", "style": "primary"}],
        [{"text": "Share", "switch_inline_query": ""}, {"text": "Copy", "copy_text": {"text": "id"}}]],
        "row_alignments": ["left", "right"]}
    await host._edit_control_rich(1, 2, "Page 2", None, markup)
    payload = host._bot.do_api_request.call_args.kwargs["api_kwargs"]
    html = payload["rich_message"]["html"]
    assert '<tg-button-row align="left">' in html and '<tg-button-row align="right">' in html
    assert '<tg-button type="callback_data" data="sc:cancel:1" style="danger">Cancel</tg-button>' in html
    assert '<tg-button type="callback_data" data="sc:retry:1" style="primary">Retry</tg-button>' in html
    assert '<tg-button type="switch_inline_query" query="">Share</tg-button>' in html
    assert '<tg-button type="copy_text" text="id">Copy</tg-button>' in html
    # Pagination edit (reply_markup present) is non-terminal.


@pytest.mark.asyncio
async def test_facade_accepts_positional_ptb_style_edit_call():
    host = Host()
    host.register_rich_control_message(123, 555)
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555))
    facade = host.wrap_rich_control_query(query)
    await facade.edit_message_text("*Done*", "MarkdownV2", None)
    payload = host._bot.do_api_request.call_args.kwargs["api_kwargs"]
    assert payload["message_id"] == 555
    # Positional (text, parse_mode, reply_markup) routed to the rich edit, body converted.
    assert "<b>Done</b>" in payload["rich_message"]["html"]
    assert not host.is_rich_control_message(123, 555)  # terminal edit dropped the entry


@pytest.mark.asyncio
async def test_facade_passes_link_preview_kwargs_into_rich_edit():
    host = Host()
    host.register_rich_control_message(123, 555)
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555))
    facade = host.wrap_rich_control_query(query)
    await facade.edit_message_text("t", None, None, disable_web_page_preview=True)
    payload = host._bot.do_api_request.call_args.kwargs["api_kwargs"]
    assert payload["link_preview_options"] == {"is_disabled": True}


@pytest.mark.asyncio
async def test_facade_positional_fallback_forwards_all_args_to_legacy_edit():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Bad Request: rejected"))
    host.register_rich_control_message(123, 555)
    raw_edit = AsyncMock()
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555), edit_message_text=raw_edit)
    facade = host.wrap_rich_control_query(query)
    await facade.edit_message_text("legacy", "MarkdownV2", None)
    raw_edit.assert_awaited_once_with(text="legacy", parse_mode="MarkdownV2", reply_markup=None)


@pytest.mark.asyncio
async def test_facade_extra_kwargs_ride_legacy_fallback_verbatim():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Bad Request: rejected"))
    host.register_rich_control_message(123, 555)
    raw_edit = AsyncMock()
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555), edit_message_text=raw_edit)
    facade = host.wrap_rich_control_query(query)
    await facade.edit_message_text("x", None, None, read_timeout=10)
    raw_edit.assert_awaited_once_with(text="x", parse_mode=None, reply_markup=None, read_timeout=10)


@pytest.mark.asyncio
async def test_facade_pagination_edit_with_noncallback_markup_keeps_registry():
    host = Host()
    host.register_rich_control_message(123, 555)
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555))
    facade = host.wrap_rich_control_query(query)
    markup = {"inline_keyboard": [[{"text": "Next", "url": "https://t.me/next"}]]}
    await facade.edit_message_text("Page 2", "HTML", markup)
    assert host.is_rich_control_message(123, 555)
    html = host._bot.do_api_request.call_args.kwargs["api_kwargs"]["rich_message"]["html"]
    assert '<tg-button type="url" url="https://t.me/next">Next</tg-button>' in html


# --- NativeControlMarkup: semantic styles, overrides, row contents, unwrap ------------------
def _native_markup(rows, **kw):
    return NativeControlMarkup(keyboard(*rows), **kw)


def test_semantic_styles_follow_callback_action_not_label():
    """Translated labels (Russian approve/cancel words) must NOT drive styles; the
    callback ACTION does: ea:once success, sc:cancel danger, mp selection primary."""
    markup = _native_markup([
        (("Одобрить один раз", "ea:once:1"), ("Сессия", "ea:session:1")),
        (("Всегда", "ea:always:1"), ("Отклонить", "ea:deny:1")),
        (("Отмена", "sc:cancel:9"), ("Разрешить всегда", "sc:always:9")),
        (("Модель X", "mp:slug"), ("1/3", "mx:noop")),
        (("Назад", "mb"), ("Отмена", "mx")),
        (("Вперед", "mpv:2"), ("Далее", "mg:4")),
        (("Да", "update_prompt:y"), ("Нет", "update_prompt:n")),
    ])
    html = rich_control_html("t", None, markup)
    assert 'data="ea:once:1" style="success"' in html
    assert 'data="ea:session:1" style="primary"' in html
    assert 'data="ea:always:1" style="primary"' in html
    assert 'data="ea:deny:1" style="danger"' in html
    assert 'data="sc:cancel:9" style="danger"' in html
    assert 'data="sc:always:9" style="primary"' in html
    assert 'data="mp:slug" style="primary"' in html
    assert '<tg-button type="disabled">1/3</tg-button>' in html  # mx:noop page counter
    assert 'data="mb" style="link"' in html
    assert 'data="mx" style="danger"' in html
    assert 'data="mpv:2" style="link"' in html
    assert 'data="mg:4" style="link"' in html
    assert 'data="update_prompt:y" style="success"' in html
    assert 'data="update_prompt:n" style="danger"' in html
    # Label-inference words ("Отмена" ~ cancel) never leak into semantic mode.
    assert html.count('style="danger"') == 4  # ea:deny, sc:cancel, mx, update_prompt:n


def test_plain_dicts_keep_legacy_label_inference_no_semantics():
    """Non-wrapper callers keep the generic converter behavior: no semantic styles,
    label inference still applies ("Cancel" → danger)."""
    html = rich_control_html("t", None, keyboard((("Cancel", "cp:0"),)))
    assert 'style="danger"' in html
    html2 = rich_control_html("t", None, keyboard((("Neutral", "mp:0"),)))
    assert "style" not in html2  # selection callback in generic mode: no style


def test_overrides_drive_selected_and_disabled_not_labels():
    markup = _native_markup(
        [(("✓ Provider A", "mp:a"), ("Provider B", "mp:b"))],
        overrides={"mp:a": {"style": "primary", "disabled": False},
                   "mp:b": {"disabled": True}})
    html = rich_control_html("t", None, markup)
    assert 'data="mp:a" style="primary"' in html
    assert '<tg-button type="disabled">Provider B</tg-button>' in html


def test_row_contents_interleave_per_source_row_verbatim():
    """Fragments (already escaped) render directly before their row's buttons; the
    body does not duplicate them; a split row's continuation gets no second copy."""
    big_row = tuple((f"opt{i}", f"cl:1:{i}") for i in range(10))  # splits at 8
    markup = _native_markup(
        [big_row, (("Other", "cl:1:other"),)],
        row_contents=["1. First &amp; escaped", "", "unrelated"], rich_text="❓ Q")
    html = rich_control_html("FULL LEGACY BODY WITH OPTIONS", "HTML", markup)
    assert html.startswith("❓ Q\n")  # rich_text replaced the body
    assert "FULL LEGACY BODY" not in html
    lines = html.split("\n")
    assert lines[0] == "❓ Q"
    assert lines[1] == "1. First &amp; escaped"  # verbatim, never re-escaped
    assert "<tg-button-row" in lines[2]  # fragment sits directly above its row
    # Second split row: no repeated fragment (source row 0 already rendered its copy).
    assert "1. First" not in html.split("1. First &amp; escaped", 1)[1]
    assert lines[-1].endswith("</tg-button-row>")


def test_wrapper_default_alignment_is_left_and_malformed_base_declines():
    """No explicit alignment → every row aligns left (native contract), never the
    client's guess; an invalid explicit global align keeps the client default. A
    non-dict malformed legacy base makes to_dict() return it verbatim so the
    converter/gate DECLINES (unsupported → legacy fallback), not an empty keyboard."""
    wrapped = _native_markup([(("A", "mp:a"),)])
    html = rich_control_html("t", None, wrapped)
    assert html.count('<tg-button-row align="left">') == 1
    aligned = _native_markup([(("A", "mp:a"),)], row_alignments=["center"])
    assert rich_control_html("t", None, aligned).count('<tg-button-row align="center">') == 1
    bad = NativeControlMarkup("not-a-markup")
    assert bad.to_dict() == "not-a-markup"
    assert rich_control_markup_supported(bad) is False


@pytest.mark.asyncio
async def test_rich_edit_refuses_unsupported_markup_without_dropping_actions():
    """A pagination edit carrying an unrepresentable button must NOT rich-edit
    (silent action drop) — it falls back to the legacy edit (None)."""
    host = Host()
    host.register_rich_control_message(123, 555)
    markup = {"inline_keyboard": [[{"text": "Pay", "pay": True}]]}
    assert await host._edit_control_rich(123, 555, "p2", None, markup) is None
    assert host._bot.do_api_request.call_count == 0
    assert host.is_rich_control_message(123, 555)  # gate refusal ≠ registry loss


@pytest.mark.asyncio
async def test_facade_permanent_fallback_unwraps_wrapper_and_forgets_registry():
    """Permanent rich rejection → legacy edit carries the RAW keyboard (no
    rich-only metadata) and the registry entry is dropped: later pagination
    edits must not treat the legacy message as rich."""
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Bad Request: rich unsupported"))
    host.register_rich_control_message(123, 555)
    raw_edit = AsyncMock()
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555), edit_message_text=raw_edit)
    facade = host.wrap_rich_control_query(query)
    legacy_kb = {"inline_keyboard": [[{"text": "A", "callback_data": "mp:a"}]]}
    wrapped = NativeControlMarkup(legacy_kb, overrides={"mp:a": {"disabled": True}},
                                  row_contents=["frag"], rich_text="rich")
    await facade.edit_message_text(text="p2", parse_mode="HTML", reply_markup=wrapped)
    raw_edit.assert_awaited_once_with(text="p2", parse_mode="HTML", reply_markup=legacy_kb)
    assert not host.is_rich_control_message(123, 555)


@pytest.mark.asyncio
async def test_facade_transient_failure_keeps_registry_and_raises():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=TimeoutError("t"))
    host.register_rich_control_message(123, 555)
    raw_edit = AsyncMock()
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=555), edit_message_text=raw_edit)
    facade = host.wrap_rich_control_query(query)
    with pytest.raises(TimeoutError):
        await facade.edit_message_text(text="x", reply_markup=None)
    raw_edit.assert_not_awaited()
    assert host.is_rich_control_message(123, 555)  # ambiguous: registry entry stays


@pytest.mark.asyncio
async def test_prefix_upgrade_wraps_unregistered_control_when_rich_enabled():
    """A built-in control callback on an UNREGISTERED (restart-era legacy) message
    gets the rich facade while rich_controls is on — the next edit upgrades it in
    place and registers it; the wrapper never reaches a raw PTB serialization."""
    host = Host()
    legacy_kb = {"inline_keyboard": [[{"text": "A", "callback_data": "mp:a"}]]}
    query = SimpleNamespace(data="mp:a", message=SimpleNamespace(chat_id=123, message_id=555),
                            edit_message_text=AsyncMock())
    facade = host.wrap_rich_control_query(query)
    assert facade is not query and facade.is_rich_control is True
    await facade.edit_message_text(text="p2", parse_mode="HTML", reply_markup=NativeControlMarkup(legacy_kb))
    html = host._bot.do_api_request.call_args.kwargs["api_kwargs"]["rich_message"]["html"]
    assert '<tg-button type="callback_data" data="mp:a" style="primary">A</tg-button>' in html
    assert host.is_rich_control_message(123, 555)  # pagination edit registered the upgraded message
    query.edit_message_text.assert_not_awaited()  # never the raw PTB path


@pytest.mark.asyncio
async def test_prefix_upgrade_off_foreign_or_private_keeps_raw_query():
    """rich_controls off, a foreign callback prefix, a missing message, or a private
    control (ephemeral facade) → the raw query, untouched."""
    off = Host(rich_controls=False)
    q = SimpleNamespace(data="mp:a", message=SimpleNamespace(chat_id=1, message_id=2))
    assert off.wrap_rich_control_query(q) is q
    host = Host()
    foreign = SimpleNamespace(data="zz:9", message=SimpleNamespace(chat_id=1, message_id=2))
    assert host.wrap_rich_control_query(foreign) is foreign
    no_message = SimpleNamespace(data="mp:a", message=SimpleNamespace(chat_id=None, message_id=None))
    assert host.wrap_rich_control_query(no_message) is no_message
    private = SimpleNamespace(data="mp:a", is_private_control=True,
                              message=SimpleNamespace(chat_id=1, message_id=2))
    assert host.wrap_rich_control_query(private) is private
