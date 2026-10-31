"""Tests for MessageEvent.extend_media() — the shared media-merge helper.

``media_urls`` / ``media_types`` / ``media_text_inlined`` are index-parallel: the
native admission contract requires ``len(media_text_inlined)`` to be either 0 or
exactly ``len(media_urls)`` — one flag per attachment. A merge that extends the
URLs without the flags leaves the lists mismatched, and native admission refuses
the whole message with ``invalid_params: media_text_inlined_length``.
"""

from types import SimpleNamespace

from gateway.platforms.event import MessageEvent, MessageType, inline_flags_padded


def _event(urls, types=None, flags=None):
    return MessageEvent(
        text="",
        message_type=MessageType.DOCUMENT,
        source=SimpleNamespace(channel_id="chat-1"),
        media_urls=list(urls),
        media_types=list(types) if types is not None else ["application/pdf"] * len(urls),
        media_text_inlined=list(flags) if flags is not None else [],
    )


class TestExtendMedia:
    def test_flags_stay_parallel_to_urls(self):
        """The merged event carries one inline flag per attachment."""
        existing = _event(["/tmp/a.pdf"], flags=[False])
        incoming = _event(["/tmp/b.pdf", "/tmp/c.pdf"], flags=[False, False])

        existing.extend_media(incoming)

        assert len(existing.media_urls) == 3
        assert len(existing.media_types) == 3
        assert len(existing.media_text_inlined) == 3
        assert existing.media_text_inlined == [False, False, False]

    def test_missing_flags_are_padded_with_none(self):
        """A source that tracks no per-file state contributes None, not a dropped flag."""
        existing = _event(["/tmp/a.pdf"])
        incoming = _event(["/tmp/b.pdf", "/tmp/c.pdf"])

        existing.extend_media(incoming)

        assert len(existing.media_text_inlined) == 3
        assert existing.media_text_inlined == [None, None, None]

    def test_existing_flags_are_padded_before_the_merge(self):
        """Existing flags are padded to the existing URL count first, so indices stay aligned."""
        existing = _event(["/tmp/a.pdf", "/tmp/b.pdf"], flags=[False])
        incoming = _event(["/tmp/c.pdf"], flags=[True])

        existing.extend_media(incoming)

        assert len(existing.media_text_inlined) == 3
        assert existing.media_text_inlined == [False, None, True]

    def test_single_attachment_is_unchanged(self):
        """Merging one attachment into an empty event keeps the one-flag-per-url shape."""
        existing = _event([])
        incoming = _event(["/tmp/a.pdf"], flags=[False])

        existing.extend_media(incoming)

        assert len(existing.media_urls) == 1
        assert existing.media_text_inlined == [False]


class TestInlineFlagsPadded:
    def test_pads_to_url_count(self):
        assert inline_flags_padded(_event(["/a", "/b"], flags=[True])) == [True, None]

    def test_leaves_a_longer_flag_list_alone(self):
        """A flag list longer than the URL count is returned unchanged (not truncated)."""
        assert inline_flags_padded(_event(["/a"], flags=[True, False])) == [True, False]

    def test_empty_event(self):
        assert inline_flags_padded(_event([])) == []
