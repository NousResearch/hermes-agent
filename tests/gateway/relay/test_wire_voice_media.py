"""Unit tests for relay voice-note + per-attachment media-type mapping.

The relay contract gained ``message_type: "voice"`` (connector PR: voice
notes on Discord/Telegram/WhatsApp classify distinctly from audio-file
uploads). The gateway side has two obligations:

1. ``MessageType("voice")`` must parse — it already does, since the enum
   predates the wire value — and
2. the connector's rich ``media[]`` array (each entry carries the
   per-attachment ``mime``) must land on ``event.media_types`` so run.py's
   per-attachment classifiers (``_event_media_is_stt_input``, image vs
   document routing) work for relayed events, not just native adapters.

Live-verified failure (staging 2026-08-26): a voice note arrived as
``MessageType.AUDIO`` with ``media_types == []`` — STT never fired and the
agent got a "the user sent an audio file attachment" note instead.

Pure unit tests: no socket, no websockets dependency.
"""

from __future__ import annotations

from gateway.platforms.base import MessageType
from gateway.relay.ws_transport import _event_from_wire


def _wire_event(message_type: str, **extra):
    src = {
        "platform": "discord",
        "chat_id": "chan-1",
        "chat_type": "dm",
        "user_id": "u-1",
        "user_name": "ben",
    }
    return {
        "text": "",
        "message_type": message_type,
        "source": src,
        **extra,
    }


class TestVoiceMessageType:

    def test_wire_voice_parses_to_message_type_voice(self):
        ev = _event_from_wire(_wire_event("voice"))
        assert ev.message_type == MessageType.VOICE

    def test_wire_audio_still_parses_to_message_type_audio(self):
        """Music files keep the non-STT type — the connector must not have
        collapsed them (it doesn't), and the gateway must not either."""
        ev = _event_from_wire(_wire_event("audio"))
        assert ev.message_type == MessageType.AUDIO


class TestMediaTypesMapping:

    def test_media_mimes_land_on_event_media_types(self):
        media = [
            {
                "url": "https://cdn.discordapp.com/attachments/1/2/voice-message.ogg",
                "kind": "audio",
                "mime": "audio/ogg",
                "size": 19197,
                "filename": "voice-message.ogg",
            },
        ]
        ev = _event_from_wire(
            _wire_event("voice", media=media, media_urls=[m["url"] for m in media])
        )
        assert ev.media_types == ["audio/ogg"]
        # media_urls must remain the parallel legacy field (unchanged).
        assert ev.media_urls == [media[0]["url"]]

    def test_media_types_parallel_to_media_urls_for_multiple_attachments(self):
        media = [
            {"url": "https://x/photo.png", "kind": "image", "mime": "image/png"},
            {"url": "https://x/doc.pdf", "kind": "document", "mime": "application/pdf"},
        ]
        ev = _event_from_wire(
            _wire_event("image", media=media, media_urls=[m["url"] for m in media])
        )
        assert ev.media_types == ["image/png", "application/pdf"]
        assert len(ev.media_types) == len(ev.media_urls)

    def test_missing_mime_gives_empty_string_at_that_index(self):
        """A media entry without a mime must not shift the alignment between
        media_urls[i] and media_types[i] — run.py indexes both by position."""
        media = [
            {"url": "https://x/photo.png", "kind": "image", "mime": "image/png"},
            {"url": "https://x/blob", "kind": "document"},  # no mime
        ]
        ev = _event_from_wire(
            _wire_event("image", media=media, media_urls=[m["url"] for m in media])
        )
        assert ev.media_types == ["image/png", ""]

    def test_no_media_field_means_empty_media_types(self):
        """Older connectors (no media[]) — byte-identical to pre-fix."""
        ev = _event_from_wire(_wire_event("text"))
        assert ev.media_types == []

    def test_length_mismatch_resolves_by_url_without_misaligning(self):
        """media_urls and media[] are independent wire fields; consumers index
        BOTH by the same i. Resolution is BY URL, so a length disagreement can
        no longer misalign anything: each surviving URL keeps its own mime and
        an unmatched URL degrades to "" (message-level classification)."""
        media = [
            {"url": "https://x/a.png", "kind": "image", "mime": "image/png"},
            {"url": "https://x/b.pdf", "kind": "document", "mime": "application/pdf"},
        ]
        ev = _event_from_wire(
            _wire_event("image", media=media, media_urls=["https://x/a.png"])
        )
        assert ev.media_urls == ["https://x/a.png"]
        assert ev.media_types == ["image/png"]


class TestSttGate:
    """Direct assertions on run.py's STT gate against REAL wire-parsed
    events. The PR's acceptance criterion is STT routing, not object
    construction — pin the user-visible invariant here.

    Note the deployment-ordering consequence: MessageType.VOICE parses on
    ANY gateway (the enum predates this PR) and the gate accepts VOICE with
    empty media_types, so a NEW connector + OLD gateway already fires STT
    for voice notes. Desirable — but it means the gate's behaviour is part
    of the wire contract, and it must not silently change."""

    def test_new_connector_voice_event_is_stt_eligible(self):
        from gateway.run import _event_media_is_stt_input

        ev = _event_from_wire(
            _wire_event(
                "voice",
                media=[{"url": "https://gw/relay/media/x", "kind": "voice", "mime": "audio/ogg"}],
                media_urls=["https://gw/relay/media/x"],
            )
        )
        assert _event_media_is_stt_input(ev, 0) is True

    def test_voice_event_stt_eligible_even_without_media_types(self):
        """A new-connector/old-gateway-shaped event — voice type, no
        media[] — still fires STT (the gate's VOICE branch doesn't consult
        media_types). Pins the rollout behaviour, not just the new one."""
        from gateway.run import _event_media_is_stt_input

        ev = _event_from_wire(
            _wire_event("voice", media_urls=["https://gw/relay/media/x"])
        )
        # No media[] ⇒ no per-attachment mime, but the slot still exists so the
        # parallel-array invariant holds (see TestParallelArrayLengthInvariant).
        assert ev.media_types == [""]
        assert _event_media_is_stt_input(ev, 0) is True

    def test_legacy_audio_typed_voice_note_stays_out_of_stt(self):
        from gateway.run import _event_media_is_stt_input

        ev = _event_from_wire(
            _wire_event(
                "audio",
                media=[{"url": "https://gw/relay/media/x", "kind": "voice", "mime": "audio/ogg"}],
                media_urls=["https://gw/relay/media/x"],
            )
        )
        assert _event_media_is_stt_input(ev, 0) is False

    def test_music_upload_is_never_stt_eligible(self):
        from gateway.run import _event_media_is_stt_input

        ev = _event_from_wire(
            _wire_event(
                "audio",
                media=[{"url": "https://cdn/x/song.mp3", "kind": "audio", "mime": "audio/mpeg"}],
                media_urls=["https://cdn/x/song.mp3"],
            )
        )
        assert _event_media_is_stt_input(ev, 0) is False

class TestUrlMimePairingThroughLocalization:
    """The end-to-end invariant my unit tests originally missed.

    media_urls and media_types are indexed BY POSITION by every downstream
    classifier. _localize_inbound_media drops entries (a dead connector
    re-host) as a NORMAL best-effort path — so if it filters URLs without
    filtering MIMEs in lockstep, every surviving attachment inherits a
    neighbour's type. Drive the REAL path: wire parse -> localization ->
    run.py classifier."""

    def _adapter(self):
        from gateway.relay.adapter import RelayAdapter

        return RelayAdapter.__new__(RelayAdapter)

    def test_dropped_first_attachment_does_not_shift_the_second_mime(self):
        import asyncio

        from gateway.run import _event_media_is_image, _event_media_type_at

        rehost = "https://conn.example/relay/media/dead"
        kept = "https://cdn.discordapp.com/attachments/1/2/kept.png"
        ev = _event_from_wire(
            _wire_event(
                "document",
                media=[
                    {"url": rehost, "kind": "document", "mime": "application/pdf"},
                    {"url": kept, "kind": "image", "mime": "image/png"},
                ],
                media_urls=[rehost, kept],
            )
        )
        assert ev.media_types == ["application/pdf", "image/png"]

        adapter = self._adapter()
        adapter._media_client = None
        adapter._get_media_client = lambda: None  # type: ignore[method-assign]
        asyncio.run(adapter._localize_inbound_media(ev))

        # The dead re-host is dropped; the PNG must keep ITS OWN mime.
        assert ev.media_urls == [kept]
        assert ev.media_types == ["image/png"]
        assert _event_media_type_at(ev, 0) == "image/png"
        assert _event_media_is_image(ev, 0) is True

    def test_reordered_media_vs_media_urls_resolves_by_url_not_position(self):
        """Equal-length but differently-ordered wire fields must not pair up
        positionally — resolve each URL's mime by lookup."""
        png = "https://x/a.png"
        pdf = "https://x/b.pdf"
        ev = _event_from_wire(
            _wire_event(
                "image",
                media=[
                    {"url": pdf, "mime": "application/pdf"},
                    {"url": png, "mime": "image/png"},
                ],
                media_urls=[png, pdf],
            )
        )
        assert ev.media_types == ["image/png", "application/pdf"]

    def test_url_with_no_matching_media_entry_gets_empty_mime(self):
        known = "https://x/a.png"
        orphan = "https://x/unknown.bin"
        ev = _event_from_wire(
            _wire_event(
                "image",
                media=[{"url": known, "mime": "image/png"}],
                media_urls=[known, orphan],
            )
        )
        assert ev.media_types == ["image/png", ""]

    def test_media_without_media_urls_yields_no_types(self):
        """No URL list to align to ⇒ the indices are meaningless."""
        ev = _event_from_wire(
            _wire_event("image", media=[{"url": "https://x/a.png", "mime": "image/png"}])
        )
        assert ev.media_urls == []
        assert ev.media_types == []

class TestParallelArrayLengthInvariant:
    """media_types must ALWAYS have one slot per media_url.

    Not merely an indexing nicety: ``merge_pending_message_event``
    (gateway/platforms/base.py) EXTENDS both lists when a second media message
    is merged into a pending one. If a media_types-less event merges with a
    typed one, extend() concatenates lists of different lengths and shifts
    every later mime onto the wrong url. Found by self-review after two
    rounds of external review flagged this same bug class in adjacent seams."""

    def test_media_urls_without_media_still_get_one_empty_slot_each(self):
        """An older connector sends media_urls with no media[]. Padding keeps
        the invariant instead of emitting a length-0 media_types."""
        ev = _event_from_wire(
            _wire_event("image", media_urls=["https://x/a.png", "https://x/b.png"])
        )
        assert ev.media_types == ["", ""]
        assert len(ev.media_types) == len(ev.media_urls)

    def test_merging_an_untyped_event_with_a_typed_one_keeps_mimes_on_their_urls(self):
        from gateway.platforms.base import merge_pending_message_event
        from gateway.run import _event_media_type_at

        untyped = _event_from_wire(
            _wire_event("image", media_urls=["https://x/old1.png", "https://x/old2.png"])
        )
        typed = _event_from_wire(
            _wire_event(
                "document",
                media=[{"url": "https://x/new.pdf", "mime": "application/pdf"}],
                media_urls=["https://x/new.pdf"],
            )
        )
        pending = {"k": untyped}
        merge_pending_message_event(pending, "k", typed)
        merged = pending["k"]

        assert len(merged.media_types) == len(merged.media_urls)
        # The PDF's mime must stay on the PDF, not slide onto old1.png.
        assert _event_media_type_at(merged, 0) == ""
        assert _event_media_type_at(merged, 1) == ""
        assert _event_media_type_at(merged, 2) == "application/pdf"

    def test_localization_preserves_the_invariant_when_it_drops_entries(self):
        import asyncio

        from gateway.relay.adapter import RelayAdapter

        rehost = "https://conn.example/relay/media/dead"
        kept = "https://x/kept.png"
        ev = _event_from_wire(
            _wire_event("image", media_urls=[rehost, kept])  # no media[] at all
        )
        assert ev.media_types == ["", ""]

        adapter = RelayAdapter.__new__(RelayAdapter)
        adapter._media_client = None
        adapter._get_media_client = lambda: None  # type: ignore[method-assign]
        asyncio.run(adapter._localize_inbound_media(ev))

        assert ev.media_urls == [kept]
        assert len(ev.media_types) == len(ev.media_urls)

    def test_localization_normalizes_a_short_media_types_from_any_source(self):
        """Defence in depth: an event that reaches the localizer with a SHORT
        media_types (not produced by _event_from_wire — e.g. a synthetic or
        replayed event) must come out with one slot per surviving url, not
        with the short list passed through."""
        import asyncio

        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.relay.adapter import RelayAdapter

        ev = MessageEvent(text="", message_type=MessageType.PHOTO)
        ev.media_urls = ["https://x/a.png", "https://x/b.png"]
        ev.media_types = []  # empty, while urls exist — the shape that must
        # NOT be passed through: consumers would index/merge against a
        # zero-length mime list and shift every later entry.

        adapter = RelayAdapter.__new__(RelayAdapter)
        adapter._media_client = None
        adapter._get_media_client = lambda: None  # type: ignore[method-assign]
        asyncio.run(adapter._localize_inbound_media(ev))

        assert ev.media_urls == ["https://x/a.png", "https://x/b.png"]
        assert ev.media_types == ["", ""]
        assert len(ev.media_types) == len(ev.media_urls)
