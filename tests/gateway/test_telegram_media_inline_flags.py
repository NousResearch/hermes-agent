"""Regression: attaching cached media must keep media_text_inlined aligned with media_urls.

A document with injected text carries media_text_inlined=[True]. When the same
message also replies to a media message, the replied-to attachment is appended
via _attach_cached. Before the fix the flags list stayed shorter than
media_urls, and native admission refused the whole inbound event with
invalid_params (media_text_inlined_length) - the user's message was dropped.
"""

from types import SimpleNamespace

from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter


def _event_with_inlined_document():
    return MessageEvent(
        text="[Content of notes.txt]:\nhello",
        message_type=MessageType.DOCUMENT,
        media_urls=["/tmp/notes.txt"],
        media_types=["text/plain"],
        media_text_inlined=[True],
    )


def _cached(kind="image", path="/tmp/replied.jpg", media_type="image/jpeg"):
    return SimpleNamespace(kind=kind, path=path, media_type=media_type,
                           display_name="replied.jpg")


def _attach(self, event, cached):
    TelegramAdapter._attach_cached(self, event, cached, "[note]", "%s")


def test_attach_cached_appends_inline_flag_when_flags_present():
    event = _event_with_inlined_document()
    self = SimpleNamespace(
        _CACHED_KIND_TO_MESSAGE_TYPE={"image": MessageType.PHOTO},
        _append_observed_note=TelegramAdapter._append_observed_note,
    )
    _attach(self, event, _cached())

    assert len(event.media_urls) == 2
    assert len(event.media_text_inlined) == len(event.media_urls)
    assert event.media_text_inlined == [True, False]


def test_attach_cached_leaves_flagless_event_flagless():
    event = MessageEvent(text="hi", media_urls=[], media_types=[], media_text_inlined=[])
    self = SimpleNamespace(
        _CACHED_KIND_TO_MESSAGE_TYPE={"image": MessageType.PHOTO},
        _append_observed_note=TelegramAdapter._append_observed_note,
    )
    _attach(self, event, _cached())

    assert len(event.media_urls) == 1
    assert event.media_text_inlined == []


def test_event_passes_media_clause_after_attach():
    from gateway.session_envelope import _media_clause

    event = _event_with_inlined_document()
    self = SimpleNamespace(
        _CACHED_KIND_TO_MESSAGE_TYPE={"image": MessageType.PHOTO},
        _append_observed_note=TelegramAdapter._append_observed_note,
    )
    _attach(self, event, _cached())

    assert _media_clause(event) is None


def test_album_merge_keeps_flags_aligned_for_documents():
    """Album of two text documents: each item carries [True]; merging must not lose a flag."""
    from gateway.session_envelope import _media_clause

    def doc(i):
        return MessageEvent(text=f"doc{i}", message_type=MessageType.DOCUMENT,
                            media_urls=[f"/tmp/doc{i}.txt"], media_types=["text/plain"],
                            media_text_inlined=[True])

    pending: dict = {}
    adapter = SimpleNamespace(_merge_caption=TelegramAdapter._merge_caption)
    TelegramAdapter._merge_into_pending(adapter, pending, "album:1", doc(1))
    TelegramAdapter._merge_into_pending(adapter, pending, "album:1", doc(2))
    merged = pending["album:1"]

    assert len(merged.media_urls) == 2
    assert merged.media_text_inlined == [True, True]
    assert _media_clause(merged) is None


def test_album_merge_pads_flagless_first_item():
    from gateway.session_envelope import _media_clause

    pending: dict = {}
    adapter = SimpleNamespace(_merge_caption=TelegramAdapter._merge_caption)
    first = MessageEvent(text="", message_type=MessageType.PHOTO,
                         media_urls=["/tmp/a.jpg"], media_types=["image/jpeg"], media_text_inlined=[])
    second = MessageEvent(text="[Content of b.txt]:\nhi", message_type=MessageType.DOCUMENT,
                          media_urls=["/tmp/b.txt"], media_types=["text/plain"],
                          media_text_inlined=[True])
    TelegramAdapter._merge_into_pending(adapter, pending, "album:2", first)
    TelegramAdapter._merge_into_pending(adapter, pending, "album:2", second)
    merged = pending["album:2"]

    assert merged.media_text_inlined == [None, True]
    assert _media_clause(merged) is None
