"""Tests for CLI interrupt payload merging with attached images."""

from pathlib import Path

from cli import _merge_input_payloads, _preview_input_payload, _split_input_payload


class TestSplitInputPayload:
    def test_text_payload(self):
        text, images = _split_input_payload("retry")
        assert text == "retry"
        assert images == []

    def test_tuple_payload_preserves_images(self, tmp_path):
        img = tmp_path / "clip.png"
        img.write_bytes(b"png")

        text, images = _split_input_payload(("what do you see?", [img]))

        assert text == "what do you see?"
        assert images == [img]

    def test_non_path_images_are_coerced(self, tmp_path):
        img = tmp_path / "clip.png"
        img.write_bytes(b"png")

        text, images = _split_input_payload(("", [str(img)]))

        assert text == ""
        assert images == [img]


class TestMergeInputPayloads:
    def test_text_only_payloads_join_with_newlines(self):
        merged = _merge_input_payloads(["stop", "show me the plan instead"])

        assert merged == "stop\nshow me the plan instead"

    def test_image_interrupts_preserve_paths_instead_of_crashing(self, tmp_path):
        img1 = tmp_path / "clip_1.png"
        img2 = tmp_path / "clip_2.png"
        img1.write_bytes(b"png")
        img2.write_bytes(b"png")

        merged = _merge_input_payloads([
            ("", [img1]),
            "retry",
            ("describe this one too", [img2]),
        ])

        assert isinstance(merged, tuple)
        text, images = merged
        assert text == "retry\ndescribe this one too"
        assert images == [img1, img2]

    def test_image_only_payloads_stay_multimodal(self, tmp_path):
        img = tmp_path / "clip.png"
        img.write_bytes(b"png")

        merged = _merge_input_payloads([("", [img])])

        assert merged == ("", [img])

    def test_all_empty_payloads_return_none(self):
        assert _merge_input_payloads(["", None, ("", [])]) is None


class TestPreviewInputPayload:
    def test_preview_mentions_images_when_text_is_empty(self, tmp_path):
        img1 = tmp_path / "clip_1.png"
        img2 = tmp_path / "clip_2.png"
        img1.write_bytes(b"png")
        img2.write_bytes(b"png")

        preview = _preview_input_payload(("", [img1, img2]))

        assert preview == "[2 images attached]"

    def test_preview_truncates_long_text(self):
        preview = _preview_input_payload("x" * 60, max_chars=10)
        assert preview == "xxxxxxxxxx..."
