"""Weixin quote (``ref_msg``) extraction.

iLink delivers a quoted reply as ``ref_msg = {title?, message_item?}`` — both fields
optional. Besides the two shapes the adapter already handled (quoted body inline, quoted
media) a quote can arrive with a title only, or as an opaque ``message_item`` whose
``type`` is not a content type and whose only useful field is a server-side id (#133965).

The opaque shape used to fall through to ``return text``: the quote vanished from the
agent's view with no marker and no log line. These tests pin the degraded-but-visible
behaviour and guard the shapes that already worked.
"""

import logging

from plugins.platforms.weixin import adapter as weixin


def _text_item(text: str, ref: dict | None = None) -> dict:
    item = {"type": weixin.ITEM_TEXT, "text_item": {"text": text}}
    if ref is not None:
        item["ref_msg"] = ref
    return item


class _RecordingHandler(logging.Handler):
    """Collect formatted records without touching the session-wide logging config."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _capture_debug(logger: logging.Logger, action) -> list[str]:
    """Run ``action`` with a handler attached to one logger; return the messages."""
    handler = _RecordingHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        action()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return handler.messages


def _opaque_ref(msg_id: object = "7525407784626316552", svr_id: object = "") -> dict:
    """The shape seen live: ``message_item`` present, no title, no content key."""
    message_item = {
        "type": 0,
        "create_time_ms": 1794197032000,
        "is_completed": True,
        "button_item_list": [],
        "at_bot_username_list": [],
    }
    if msg_id:
        message_item["msg_id"] = msg_id
    ref: dict = {"message_item": message_item}
    if svr_id:
        ref["svr_id"] = svr_id
    return ref


class TestOpaqueRefMsg:
    def test_id_only_quote_keeps_a_marker(self):
        ref = _opaque_ref()
        out = weixin._extract_text([_text_item("123", ref)])

        assert out == f"[引用: msg {ref['message_item']['msg_id']}]\n123", out
        assert out.endswith("123")

    def test_marker_falls_back_to_svr_id(self):
        out = weixin._extract_text([_text_item("好", _opaque_ref(msg_id="", svr_id="svr-99"))])

        assert out == "[引用: msg svr-99]\n好", out

    def test_svr_id_only_ref_keeps_a_marker(self):
        out = weixin._extract_text([_text_item("hi", {"svr_id": "svr-1"})])

        assert out == "[引用: msg svr-1]\nhi", out

    def test_quote_without_any_id_degrades_to_placeholder(self):
        out = weixin._extract_text([_text_item("嗯", _opaque_ref(msg_id=""))])

        assert out == "[引用: 内容不可用]\n嗯", out

    def test_unknown_ref_type_does_not_raise(self):
        ref = _opaque_ref()
        ref["message_item"]["type"] = 99

        assert weixin._extract_text([_text_item("x", ref)]).endswith("x")

    def test_opaque_quote_is_logged_without_its_content(self):
        ref = _opaque_ref()
        messages = _capture_debug(weixin.logger, lambda: weixin._extract_text([_text_item("y", ref)]))

        assert any("without a delivered body" in message for message in messages), messages
        assert not any(ref["message_item"]["msg_id"] in message for message in messages), (
            "only shape metadata belongs in logs: not the quoted id, title or body"
        )

    def test_title_only_quote_is_not_logged_as_degraded(self):
        messages = _capture_debug(
            weixin.logger, lambda: weixin._extract_text([_text_item("z", {"title": "标题"})])
        )

        assert not any("without a delivered body" in message for message in messages), messages


class TestRefMsgShapesThatAlreadyWorked:
    def test_quoted_body_is_inlined(self):
        out = weixin._extract_text(
            [_text_item("继续", {"message_item": {"type": weixin.ITEM_TEXT, "text_item": {"text": "被引用的正文"}}})]
        )

        assert out == "[引用: 被引用的正文]\n继续", out

    def test_title_and_body_are_both_kept(self):
        out = weixin._extract_text(
            [
                _text_item(
                    "好",
                    {"title": "标题", "message_item": {"type": weixin.ITEM_TEXT, "text_item": {"text": "正文"}}},
                )
            ]
        )

        assert out == "[引用: 标题 | 正文]\n好", out

    def test_title_only_quote_is_kept(self):
        out = weixin._extract_text([_text_item("看这个", {"title": "被回复的消息"})])

        assert out == "[引用: 被回复的消息]\n看这个", out

    def test_quoted_media_keeps_its_title(self):
        out = weixin._extract_text(
            [_text_item("看图", {"title": "照片", "message_item": {"type": weixin.ITEM_IMAGE}})]
        )

        assert out == "[引用媒体: 照片]\n看图", out

    def test_quoted_media_without_title_is_labelled(self):
        out = weixin._extract_text([_text_item("看图", {"message_item": {"type": weixin.ITEM_VIDEO}})])

        assert out == "[引用媒体]\n看图", out


class TestNoQuote:
    def test_plain_text_is_returned_unchanged(self):
        assert weixin._extract_text([_text_item("纯文本")]) == "纯文本"

    def test_empty_ref_msg_is_returned_unchanged(self):
        assert weixin._extract_text([_text_item("hi", {})]) == "hi"

    def test_ref_msg_with_blank_svr_id_is_returned_unchanged(self):
        assert weixin._extract_text([_text_item("hi", {"svr_id": ""})]) == "hi"

    def test_ref_msg_with_unrelated_keys_is_returned_unchanged(self):
        assert weixin._extract_text([_text_item("hi", {"unexpected": "x"})]) == "hi"
