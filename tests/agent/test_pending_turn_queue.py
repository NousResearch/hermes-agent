"""Tests for the structured pending-turn queue (orchestrator Phase 2).

These cover the leaf module ``agent.pending_turn_queue`` in isolation: the
:class:`PendingTurnItem` shape and its boundary/coalescing rules, the
:class:`PendingTurnQueue` container, lossless round-tripping of legacy CLI
payloads, the one-way gateway ``MessageEvent`` bridge, and JSON serializability.
"""

import json
import unittest
from datetime import datetime
from types import SimpleNamespace

from agent import pending_turn_queue as ptq


# --------------------------------------------------------------------------
# PendingTurnItem
# --------------------------------------------------------------------------
class TestPendingTurnItem(unittest.TestCase):
    def test_defaults_are_a_coalescible_text_item_once_text_is_set(self):
        item = ptq.PendingTurnItem(text="hello")
        self.assertEqual(item.kind, ptq.KIND_TEXT)
        self.assertEqual(item.boundary, ptq.BOUNDARY_COALESCE)
        self.assertEqual(item.source, ptq.SOURCE_UNKNOWN)
        self.assertTrue(item.is_text)
        self.assertFalse(item.is_command)
        self.assertFalse(item.has_media)
        self.assertTrue(item.is_coalescible_text())
        self.assertFalse(item.origin_busy)
        self.assertIsInstance(item.id, str)
        self.assertTrue(item.id)
        self.assertIsInstance(item.created_at, float)

    def test_empty_text_is_not_coalescible(self):
        self.assertFalse(ptq.PendingTurnItem(text="").is_coalescible_text())
        self.assertFalse(ptq.PendingTurnItem(text=None).is_coalescible_text())

    def test_command_and_media_and_hard_boundaries_are_not_coalescible(self):
        cmd = ptq.PendingTurnItem(kind=ptq.KIND_COMMAND, text="/busy", boundary=ptq.BOUNDARY_COMMAND)
        media = ptq.PendingTurnItem(kind=ptq.KIND_MEDIA, media_refs=["/tmp/a.png"], boundary=ptq.BOUNDARY_HARD)
        hard_text = ptq.PendingTurnItem(kind=ptq.KIND_TEXT, text="x", boundary=ptq.BOUNDARY_HARD)
        caption = ptq.PendingTurnItem(kind=ptq.KIND_TEXT, text="cap", boundary=ptq.BOUNDARY_CAPTION)
        for it in (cmd, media, hard_text, caption):
            self.assertFalse(it.is_coalescible_text(), it)
        self.assertTrue(cmd.is_command)
        self.assertTrue(media.has_media)

    def test_unknown_kind_or_boundary_is_tolerated(self):
        # The vocabulary constants are documentation, not a rejecting enum.
        item = ptq.PendingTurnItem(kind="weird-future-kind", boundary="weird-future-boundary", text="x")
        self.assertEqual(item.kind, "weird-future-kind")
        self.assertFalse(item.is_coalescible_text())

    def test_to_dict_is_json_safe_and_drops_raw(self):
        item = ptq.PendingTurnItem(
            source=ptq.SOURCE_CLI, text="hi", task_hint="task-7", origin_busy=True, raw=object()
        )
        d = item.to_dict()
        self.assertNotIn("raw", d)
        json.dumps(d)  # must not raise
        self.assertEqual(d["text"], "hi")
        self.assertEqual(d["task_hint"], "task-7")
        self.assertTrue(d["origin_busy"])

    def test_to_dict_does_not_copy_or_touch_uncopyable_raw(self):
        class UncopyableRaw:
            def __deepcopy__(self, memo):  # pragma: no cover - only called on regression
                raise RuntimeError("raw should not be copied")

        item = ptq.PendingTurnItem(text="hi", raw=UncopyableRaw())
        d = item.to_dict()

        self.assertNotIn("raw", d)
        self.assertEqual(d["text"], "hi")
        json.dumps(d)  # must not raise

    def test_to_dict_copies_mutable_lists_without_deepcopying_raw(self):
        item = ptq.PendingTurnItem(
            kind=ptq.KIND_MEDIA,
            media_refs=["/tmp/a.png"],
            media_types=["image/png"],
            raw=object(),
        )
        d = item.to_dict()
        d["media_refs"].append("/tmp/b.png")
        d["media_types"].append("image/jpeg")

        self.assertEqual(item.media_refs, ["/tmp/a.png"])
        self.assertEqual(item.media_types, ["image/png"])

    def test_from_dict_round_trips_and_ignores_unknown_keys(self):
        item = ptq.PendingTurnItem(source=ptq.SOURCE_TUI, text="hi", media_refs=["/tmp/x"], kind=ptq.KIND_MEDIA)
        d = item.to_dict()
        d["some_future_field"] = 123
        rebuilt = ptq.PendingTurnItem.from_dict(d)
        self.assertEqual(rebuilt.source, ptq.SOURCE_TUI)
        self.assertEqual(rebuilt.text, "hi")
        self.assertEqual(rebuilt.media_refs, ["/tmp/x"])
        self.assertEqual(rebuilt.kind, ptq.KIND_MEDIA)

    def test_copy_overrides(self):
        item = ptq.PendingTurnItem(text="a", boundary=ptq.BOUNDARY_COALESCE)
        clone = item.copy(boundary=ptq.BOUNDARY_HARD)
        self.assertEqual(item.boundary, ptq.BOUNDARY_COALESCE)
        self.assertEqual(clone.boundary, ptq.BOUNDARY_HARD)
        self.assertEqual(clone.text, "a")


# --------------------------------------------------------------------------
# PendingTurnQueue
# --------------------------------------------------------------------------
class TestPendingTurnQueue(unittest.TestCase):
    @staticmethod
    def _text(t, *, origin_busy=False):
        return ptq.PendingTurnItem(text=t, source=ptq.SOURCE_CLI, origin_busy=origin_busy)

    def test_container_protocol(self):
        q = ptq.PendingTurnQueue()
        self.assertEqual(len(q), 0)
        self.assertFalse(q)
        self.assertIsNone(q.peek())
        self.assertIsNone(q.pop())

        a, b = self._text("a"), self._text("b")
        q.append(a)
        q.append(b)
        self.assertEqual(len(q), 2)
        self.assertTrue(q)
        self.assertIs(q.peek(), a)
        self.assertEqual([it.text for it in q], ["a", "b"])  # iteration is non-consuming
        self.assertEqual(len(q), 2)
        self.assertIs(q.pop(), a)
        self.assertEqual(len(q), 1)
        q.appendleft(a)
        self.assertEqual([it.text for it in q.snapshot()], ["a", "b"])
        q.extend([self._text("c")])
        self.assertEqual([it.text for it in q], ["a", "b", "c"])
        q.clear()
        self.assertEqual(len(q), 0)

    def test_constructor_accepts_iterable(self):
        q = ptq.PendingTurnQueue(self._text(t) for t in ("x", "y"))
        self.assertEqual([it.text for it in q], ["x", "y"])

    def test_drain_coalescible_run_stops_at_first_non_text(self):
        q = ptq.PendingTurnQueue([
            self._text("a"),
            self._text("b"),
            ptq.PendingTurnItem(kind=ptq.KIND_MEDIA, media_refs=["/tmp/a.png"], boundary=ptq.BOUNDARY_HARD),
            self._text("c"),
        ])
        run = q.drain_coalescible_text_until_boundary()
        self.assertEqual([it.text for it in run], ["a", "b"])
        # The media item and everything after it stay, in order.
        self.assertEqual(len(q), 2)
        self.assertEqual(q.peek().kind, ptq.KIND_MEDIA)
        self.assertEqual(ptq.coalesced_text(run), "a\n\nb")

    def test_drain_stops_at_slash_command_item(self):
        q = ptq.PendingTurnQueue([
            self._text("a"),
            ptq.PendingTurnItem(kind=ptq.KIND_COMMAND, text="/busy", boundary=ptq.BOUNDARY_COMMAND),
            self._text("b"),
        ])
        run = q.drain_coalescible_text_until_boundary()
        self.assertEqual([it.text for it in run], ["a"])
        self.assertEqual([it.text for it in q], ["/busy", "b"])

    def test_drain_with_origin_busy_filter(self):
        q = ptq.PendingTurnQueue([
            self._text("a", origin_busy=True),
            self._text("b", origin_busy=True),
            self._text("c", origin_busy=False),  # different origin -> boundary
            self._text("d", origin_busy=True),
        ])
        run = q.drain_coalescible_text_until_boundary(origin_busy=True)
        self.assertEqual([it.text for it in run], ["a", "b"])
        self.assertEqual([it.text for it in q], ["c", "d"])

        # The opposite filter on a queue that starts with a non-busy item:
        q2 = ptq.PendingTurnQueue([self._text("p", origin_busy=False), self._text("q", origin_busy=True)])
        run2 = q2.drain_coalescible_text_until_boundary(origin_busy=False)
        self.assertEqual([it.text for it in run2], ["p"])
        self.assertEqual([it.text for it in q2], ["q"])

    def test_drain_empty_queue_returns_empty_run(self):
        q = ptq.PendingTurnQueue()
        self.assertEqual(q.drain_coalescible_text_until_boundary(), [])

    def test_drain_when_head_is_already_a_boundary(self):
        q = ptq.PendingTurnQueue([
            ptq.PendingTurnItem(kind=ptq.KIND_COMMAND, text="/x", boundary=ptq.BOUNDARY_COMMAND),
            self._text("a"),
        ])
        run = q.drain_coalescible_text_until_boundary()
        self.assertEqual(run, [])
        self.assertEqual(len(q), 2)


# --------------------------------------------------------------------------
# Integrated-busy sentinel helpers
# --------------------------------------------------------------------------
class TestIntegratedBusySentinel(unittest.TestCase):
    def test_make_is_unwrap_round_trip(self):
        payload = ptq.make_integrated_busy_payload("hello")
        self.assertTrue(ptq.is_integrated_busy_payload(payload))
        self.assertEqual(ptq.unwrap_integrated_busy_payload(payload), "hello")

    def test_non_tagged_values_pass_through(self):
        self.assertFalse(ptq.is_integrated_busy_payload("hello"))
        self.assertEqual(ptq.unwrap_integrated_busy_payload("hello"), "hello")
        # A plain (caption, images) tuple is NOT an integrated payload, and a
        # caption that happens to spell "integrated_busy" must stay an image.
        img = ("integrated_busy", ["/tmp/a.png"])
        self.assertFalse(ptq.is_integrated_busy_payload(img))
        self.assertEqual(ptq.unwrap_integrated_busy_payload(img), img)
        self.assertFalse(ptq.is_integrated_busy_payload(("a", "b", "c")))

    def test_sentinel_is_identity_only_not_a_string(self):
        self.assertNotEqual(ptq.INTEGRATED_BUSY_PAYLOAD, "integrated_busy")
        self.assertIsNot(ptq.INTEGRATED_BUSY_PAYLOAD, object())


# --------------------------------------------------------------------------
# looks_like_slash_command
# --------------------------------------------------------------------------
class TestLooksLikeSlashCommand(unittest.TestCase):
    def test_commands_vs_paths_vs_prose(self):
        self.assertTrue(ptq.looks_like_slash_command("/busy"))
        self.assertTrue(ptq.looks_like_slash_command("/model gpt-4"))
        self.assertTrue(ptq.looks_like_slash_command("/q"))
        # A pasted absolute path that merely starts with "/" is not a command.
        self.assertFalse(ptq.looks_like_slash_command("/Users/foo/bar.md fix this"))
        self.assertFalse(ptq.looks_like_slash_command("just text"))
        self.assertFalse(ptq.looks_like_slash_command(""))
        self.assertFalse(ptq.looks_like_slash_command(None))
        self.assertFalse(ptq.looks_like_slash_command(("/busy", [])))


# --------------------------------------------------------------------------
# Legacy CLI payload conversion
# --------------------------------------------------------------------------
class TestLegacyCliPayloadConversion(unittest.TestCase):
    def test_plain_text(self):
        it = ptq.from_legacy_cli_payload("hello world")
        self.assertEqual(it.kind, ptq.KIND_TEXT)
        self.assertEqual(it.boundary, ptq.BOUNDARY_COALESCE)
        self.assertEqual(it.text, "hello world")
        self.assertEqual(it.source, ptq.SOURCE_CLI)
        self.assertFalse(it.origin_busy)
        self.assertTrue(it.is_coalescible_text())
        self.assertEqual(it.raw, "hello world")
        self.assertEqual(ptq.maybe_to_legacy_cli_payload(it), "hello world")

    def test_slash_command_text_becomes_command_item(self):
        it = ptq.from_legacy_cli_payload("/busy status")
        self.assertEqual(it.kind, ptq.KIND_COMMAND)
        self.assertEqual(it.boundary, ptq.BOUNDARY_COMMAND)
        self.assertFalse(it.is_coalescible_text())
        self.assertEqual(ptq.maybe_to_legacy_cli_payload(it), "/busy status")

    def test_pasted_path_is_text_not_command(self):
        it = ptq.from_legacy_cli_payload("/Users/foo/bar.md please fix")
        self.assertEqual(it.kind, ptq.KIND_TEXT)
        self.assertTrue(it.is_coalescible_text())

    def test_empty_string_is_coalescible_payload_but_not_a_merge_starter(self):
        # Mirrors the legacy CLI guard/loop asymmetry: an empty string is a
        # "coalescible text" *payload* (the guard lets it through) yet does not
        # itself start a merge run (the loop check requires non-empty text).
        self.assertTrue(ptq.legacy_cli_payload_is_coalescible_text(""))
        self.assertFalse(ptq.from_legacy_cli_payload("").is_coalescible_text())

    def test_image_tuple_becomes_hard_media_item(self):
        payload = ("caption here", ["/tmp/a.png", "/tmp/b.jpg"])
        it = ptq.from_legacy_cli_payload(payload)
        self.assertEqual(it.kind, ptq.KIND_MEDIA)
        self.assertEqual(it.boundary, ptq.BOUNDARY_HARD)
        self.assertEqual(it.text, "caption here")
        self.assertEqual(it.media_refs, ["/tmp/a.png", "/tmp/b.jpg"])
        self.assertFalse(it.is_coalescible_text())
        self.assertEqual(ptq.maybe_to_legacy_cli_payload(it), payload)

    def test_image_tuple_without_caption(self):
        payload = ("", ["/tmp/a.png"])
        it = ptq.from_legacy_cli_payload(payload)
        self.assertEqual(it.kind, ptq.KIND_MEDIA)
        self.assertIsNone(it.text)
        self.assertEqual(it.media_refs, ["/tmp/a.png"])
        self.assertEqual(ptq.maybe_to_legacy_cli_payload(it), payload)

    def test_integrated_busy_text_payload(self):
        payload = ptq.make_integrated_busy_payload("busy fragment")
        it = ptq.from_legacy_cli_payload(payload)
        self.assertEqual(it.kind, ptq.KIND_TEXT)
        self.assertEqual(it.boundary, ptq.BOUNDARY_COALESCE)
        self.assertEqual(it.text, "busy fragment")
        self.assertTrue(it.origin_busy)
        self.assertTrue(it.is_coalescible_text())
        self.assertEqual(ptq.maybe_to_legacy_cli_payload(it), payload)

    def test_integrated_busy_slash_command_payload(self):
        payload = ptq.make_integrated_busy_payload("/busy status")
        it = ptq.from_legacy_cli_payload(payload)
        self.assertEqual(it.kind, ptq.KIND_COMMAND)
        self.assertEqual(it.boundary, ptq.BOUNDARY_COMMAND)
        self.assertTrue(it.origin_busy)
        self.assertFalse(it.is_coalescible_text())
        self.assertEqual(ptq.maybe_to_legacy_cli_payload(it), payload)

    def test_unknown_payload_becomes_opaque_control_item(self):
        sentinel = object()
        it = ptq.from_legacy_cli_payload(sentinel)
        self.assertEqual(it.kind, ptq.KIND_CONTROL)
        self.assertEqual(it.boundary, ptq.BOUNDARY_HARD)
        self.assertIsNone(it.text)
        self.assertFalse(it.is_coalescible_text())
        self.assertIs(ptq.maybe_to_legacy_cli_payload(it), sentinel)

    def test_maybe_to_legacy_reconstructs_when_raw_missing(self):
        # Items minted by hand (no `raw`) still degrade to a sensible payload.
        self.assertEqual(
            ptq.maybe_to_legacy_cli_payload(ptq.PendingTurnItem(text="hi")), "hi"
        )
        self.assertEqual(
            ptq.maybe_to_legacy_cli_payload(
                ptq.PendingTurnItem(kind=ptq.KIND_MEDIA, text="cap", media_refs=["/tmp/x"])
            ),
            ("cap", ["/tmp/x"]),
        )
        self.assertEqual(ptq.maybe_to_legacy_cli_payload(ptq.PendingTurnItem()), "")

    def test_session_key_is_threaded_through(self):
        it = ptq.from_legacy_cli_payload("hi", session_key="cli:main")
        self.assertEqual(it.session_key, "cli:main")

    def test_round_trip_through_a_queue_preserves_order_and_boundaries(self):
        legacy = [
            "first",
            ptq.make_integrated_busy_payload("busy-a"),
            "/busy status",
            ("cap", ["/tmp/a.png"]),
            "last",
        ]
        q = ptq.PendingTurnQueue(ptq.from_legacy_cli_payload(p) for p in legacy)
        # Only the leading non-busy text coalesces.
        run = q.drain_coalescible_text_until_boundary(origin_busy=False)
        self.assertEqual([it.text for it in run], ["first"])
        leftover = [ptq.maybe_to_legacy_cli_payload(it) for it in q]
        self.assertEqual(leftover, legacy[1:])


# --------------------------------------------------------------------------
# Gateway MessageEvent conversion (duck-typed, no gateway import)
# --------------------------------------------------------------------------
class _FakeType:
    def __init__(self, value):
        self.value = value


def _fake_event(*, text="", mtype="text", media_urls=None, media_types=None,
                platform="telegram", reply_to=None, ts=None):
    return SimpleNamespace(
        text=text,
        message_type=_FakeType(mtype),
        media_urls=list(media_urls or []),
        media_types=list(media_types or []),
        reply_to_message_id=reply_to,
        source=SimpleNamespace(platform=_FakeType(platform)),
        timestamp=ts or datetime(2026, 5, 12, 10, 0, 0),
    )


class TestGatewayEventConversion(unittest.TestCase):
    def test_text_event(self):
        it = ptq.from_gateway_event(_fake_event(text="hello"), session_key="telegram:42")
        self.assertEqual(it.kind, ptq.KIND_TEXT)
        self.assertEqual(it.boundary, ptq.BOUNDARY_COALESCE)
        self.assertEqual(it.text, "hello")
        self.assertEqual(it.source, ptq.SOURCE_TELEGRAM)
        self.assertEqual(it.session_key, "telegram:42")
        self.assertTrue(it.is_coalescible_text())
        self.assertIsInstance(it.created_at, float)

    def test_slash_command_text_event_is_classified_as_command(self):
        it = ptq.from_gateway_event(_fake_event(text="/restart"))
        self.assertEqual(it.kind, ptq.KIND_COMMAND)
        self.assertEqual(it.boundary, ptq.BOUNDARY_COMMAND)
        self.assertFalse(it.is_coalescible_text())

    def test_photo_album_event_is_hard_media_with_caption_in_text(self):
        it = ptq.from_gateway_event(
            _fake_event(text="album caption", mtype="photo",
                        media_urls=["/tmp/1.jpg", "/tmp/2.jpg"], media_types=["image/jpeg", "image/jpeg"])
        )
        self.assertEqual(it.kind, ptq.KIND_MEDIA)
        self.assertEqual(it.boundary, ptq.BOUNDARY_HARD)
        self.assertEqual(it.text, "album caption")
        self.assertEqual(it.media_refs, ["/tmp/1.jpg", "/tmp/2.jpg"])
        self.assertFalse(it.is_coalescible_text())

    def test_document_event_is_attachment(self):
        it = ptq.from_gateway_event(_fake_event(mtype="document", media_urls=["/tmp/x.pdf"]))
        self.assertEqual(it.kind, ptq.KIND_ATTACHMENT)
        self.assertEqual(it.boundary, ptq.BOUNDARY_HARD)
        self.assertFalse(it.is_coalescible_text())

    def test_location_event_is_control(self):
        it = ptq.from_gateway_event(_fake_event(mtype="location", text=""))
        self.assertEqual(it.kind, ptq.KIND_CONTROL)
        self.assertEqual(it.boundary, ptq.BOUNDARY_HARD)

    def test_media_type_with_no_refs_but_text_degrades_to_text(self):
        it = ptq.from_gateway_event(_fake_event(text="just a caption", mtype="photo", media_urls=[]))
        self.assertEqual(it.kind, ptq.KIND_TEXT)
        self.assertTrue(it.is_coalescible_text())

    def test_reply_to_is_stringified(self):
        it = ptq.from_gateway_event(_fake_event(text="re", reply_to=12345))
        self.assertEqual(it.reply_to, "12345")

    def test_unknown_message_type_defaults_to_text(self):
        it = ptq.from_gateway_event(_fake_event(text="weird", mtype="future-type"))
        self.assertEqual(it.kind, ptq.KIND_TEXT)

    def test_unknown_message_type_with_media_refs_is_hard_media_boundary(self):
        it = ptq.from_gateway_event(
            _fake_event(text="future caption", mtype="image", media_urls=["/tmp/future.webp"])
        )
        self.assertEqual(it.kind, ptq.KIND_MEDIA)
        self.assertEqual(it.boundary, ptq.BOUNDARY_HARD)
        self.assertEqual(it.media_refs, ["/tmp/future.webp"])
        self.assertFalse(it.is_coalescible_text())

    def test_raw_string_platform_and_message_type_are_supported(self):
        ev = SimpleNamespace(
            text="/busy status",
            message_type="text",
            media_urls=[],
            media_types=[],
            source=SimpleNamespace(platform="telegram"),
            timestamp=datetime(2026, 5, 12, 10, 0, 0),
        )
        it = ptq.from_gateway_event(ev, session_key="tg:1")
        self.assertEqual(it.source, ptq.SOURCE_TELEGRAM)
        self.assertEqual(it.kind, ptq.KIND_COMMAND)
        self.assertEqual(it.boundary, ptq.BOUNDARY_COMMAND)
        self.assertEqual(it.session_key, "tg:1")

    def test_command_message_type_with_path_like_text_degrades_to_text(self):
        it = ptq.from_gateway_event(_fake_event(text="/Users/foo/bar.md please inspect", mtype="command"))
        self.assertEqual(it.kind, ptq.KIND_TEXT)
        self.assertEqual(it.boundary, ptq.BOUNDARY_COALESCE)
        self.assertTrue(it.is_coalescible_text())

    def test_missing_attributes_are_tolerated(self):
        it = ptq.from_gateway_event(SimpleNamespace())
        self.assertEqual(it.source, ptq.SOURCE_UNKNOWN)
        self.assertEqual(it.kind, ptq.KIND_TEXT)
        self.assertIsNone(it.text)
        self.assertIsInstance(it.created_at, float)

    def test_raw_event_is_kept(self):
        ev = _fake_event(text="x")
        it = ptq.from_gateway_event(ev)
        self.assertIs(it.raw, ev)
        # ...but is excluded from the serializable form.
        self.assertNotIn("raw", it.to_dict())


# --------------------------------------------------------------------------
# coalesced_text helper
# --------------------------------------------------------------------------
class TestCoalescedText(unittest.TestCase):
    def test_joins_text_skipping_blanks(self):
        items = [
            ptq.PendingTurnItem(text="a"),
            ptq.PendingTurnItem(text=""),
            ptq.PendingTurnItem(text=None),
            ptq.PendingTurnItem(text="b"),
        ]
        self.assertEqual(ptq.coalesced_text(items), "a\n\nb")
        self.assertEqual(ptq.coalesced_text(items, sep=" | "), "a | b")
        self.assertEqual(ptq.coalesced_text([]), "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
