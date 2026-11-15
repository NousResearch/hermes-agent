"""A photo or file that arrives only inside ``rich_message`` blocks is found, at any nesting, and its caption is
message text (#155902): before, the adapter flattened block text alone and the agent got the caption, not the image."""

from types import SimpleNamespace

from plugins.platforms.telegram.telegram_entities import rich_message_text
from plugins.platforms.telegram.telegram_rich_media import rich_media_items


def _photo(fid):
    return {"type": "photo", "photo": [{"file_id": f"{fid}-small"}, {"file_id": fid, "file_size": 9}],
            "caption": {"text": f"caption {fid}"}}


def test_every_media_block_is_found_in_order_with_its_caption_as_text():
    msg = SimpleNamespace(text=None, api_kwargs={"rich_message": {"blocks": [
        _photo("p1"),
        {"type": "paragraph", "text": "look at these"},
        {"type": "collage", "blocks": [_photo("p2"), {"type": "video", "video": {"file_id": "v1"}}]},
        {"type": "list", "items": [{"label": "-", "blocks": [
            {"type": "document", "document": {"file_id": "d1", "file_name": "a.pdf", "mime_type": "application/pdf"}}]}]},
        {"type": "photo", "photo": []},  # no sizes: nothing to fetch
    ]}})
    assert [(fid, mime, kind) for fid, _media, mime, kind in rich_media_items(msg)] == [
        ("p1", "", "image"), ("p2", "", "image"), ("v1", "video/mp4", "video"), ("d1", "application/pdf", None)]
    assert rich_message_text(msg).splitlines()[:2] == ["caption p1", "look at these"]


def test_a_text_only_rich_message_has_no_media():
    msg = SimpleNamespace(text=None, rich_message={"blocks": [{"type": "paragraph", "text": "hi"}]})
    assert rich_media_items(msg) == []
