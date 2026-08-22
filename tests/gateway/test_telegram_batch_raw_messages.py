"""Tests for TelegramAdapter._merge_raw_messages batch provenance.

Text-chunk batching, photo-burst batching, and media-group batching all fold a
follow-up ``MessageEvent`` into a pending one. Before ``_merge_raw_messages``
existed the merge only touched ``text``/``media_urls``/``media_types``, so the
pending event's ``raw_message`` stayed at the *first* message of the batch and
every later message's identity (``forward_origin``, sender, timestamp) was lost.
"""

from gateway.platforms.base import MessageEvent
from plugins.platforms.telegram.adapter import TelegramAdapter

merge_raw = TelegramAdapter._merge_raw_messages


def _event(text="", raw=None, media_urls=None):
    return MessageEvent(
        text=text,
        raw_message=raw,
        media_urls=list(media_urls or []),
    )


class TestRawMessageProvenance:
    def test_first_merge_keeps_both_raw_messages(self):
        existing = _event("first", raw="raw-1")
        incoming = _event("second", raw="raw-2")

        merge_raw(existing, incoming)

        assert existing._raw_messages == ["raw-1", "raw-2"]

    def test_raw_message_of_pending_event_is_untouched(self):
        existing = _event("first", raw="raw-1")

        merge_raw(existing, _event("second", raw="raw-2"))

        assert existing.raw_message == "raw-1"

    def test_further_merges_append_in_arrival_order(self):
        existing = _event("first", raw="raw-1")

        merge_raw(existing, _event("second", raw="raw-2"))
        merge_raw(existing, _event("third", raw="raw-3"))

        assert existing._raw_messages == ["raw-1", "raw-2", "raw-3"]

    def test_missing_raw_message_is_recorded_as_none(self):
        existing = _event("first")
        merge_raw(existing, _event("second", raw="raw-2"))

        assert existing._raw_messages == [None, "raw-2"]


class TestMediaOwnership:
    def test_each_url_maps_to_the_message_it_arrived_on(self):
        existing = _event(raw="raw-1", media_urls=["/cache/a.jpg"])
        incoming = _event(raw="raw-2", media_urls=["/cache/b.jpg"])

        merge_raw(existing, incoming)

        assert existing._media_owners == {
            "/cache/a.jpg": "raw-1",
            "/cache/b.jpg": "raw-2",
        }

    def test_album_of_three_keeps_every_owner(self):
        existing = _event(raw="raw-1", media_urls=["/cache/a.jpg"])

        merge_raw(existing, _event(raw="raw-2", media_urls=["/cache/b.jpg"]))
        merge_raw(existing, _event(raw="raw-3", media_urls=["/cache/c.jpg"]))

        assert existing._media_owners["/cache/c.jpg"] == "raw-3"
        assert len(existing._media_owners) == 3

    def test_text_only_merge_records_no_owners(self):
        existing = _event("first", raw="raw-1")

        merge_raw(existing, _event("second", raw="raw-2"))

        assert existing._media_owners == {}

    def test_duplicate_url_maps_to_the_later_message(self):
        existing = _event(raw="raw-1", media_urls=["/cache/same.jpg"])

        merge_raw(existing, _event(raw="raw-2", media_urls=["/cache/same.jpg"]))

        assert existing._media_owners == {"/cache/same.jpg": "raw-2"}

    def test_media_arriving_after_a_text_chunk_is_attributed(self):
        existing = _event("caption", raw="raw-1")

        merge_raw(existing, _event(raw="raw-2", media_urls=["/cache/b.jpg"]))

        assert existing._media_owners == {"/cache/b.jpg": "raw-2"}
