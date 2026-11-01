"""Tests for mid-prompt CLI image paths (not only a leading drag/paste).

Closes #34542. Only an existing file whose typed suffix and resolved suffix
are both image extensions is attached.
"""

import pytest

from cli import _apply_inline_image_paths, _detect_inline_image_paths


@pytest.fixture()
def tmp_image(tmp_path):
    img = tmp_path / "screenshot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    return img


@pytest.fixture()
def tmp_image2(tmp_path):
    img = tmp_path / "diagram.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0")
    return img


@pytest.fixture()
def tmp_image_with_spaces(tmp_path):
    img = tmp_path / "My Shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    return img


@pytest.fixture()
def tmp_text(tmp_path):
    source = tmp_path / "main.py"
    source.write_text("print('hello')\n")
    return source


class TestNoMatch:
    def test_empty_string(self):
        assert _detect_inline_image_paths("") is None

    def test_whitespace_only(self):
        assert _detect_inline_image_paths("   ") is None

    def test_non_string(self):
        assert _detect_inline_image_paths(42) is None

    def test_plain_prose(self):
        assert _detect_inline_image_paths("how do I center a div") is None

    def test_slash_command_untouched(self):
        assert _detect_inline_image_paths("/help") is None

    def test_slash_command_with_image_word_untouched(self):
        assert _detect_inline_image_paths("/model gpt-image.png") is None

    def test_nonexistent_image_path(self):
        assert _detect_inline_image_paths("look at /nope/missing.png") is None

    def test_image_extension_word_but_not_a_file(self):
        assert _detect_inline_image_paths("our logo.png is blue") is None

    def test_non_image_file_not_matched(self, tmp_text):
        assert _detect_inline_image_paths(f"review {tmp_text}") is None

    def test_url_not_matched(self):
        assert (
            _detect_inline_image_paths("see https://example.com/a.png please") is None
        )


class TestInlineDetection:
    def test_image_in_middle(self, tmp_image):
        result = _detect_inline_image_paths(
            f"look at {tmp_image} and tell me what's wrong"
        )
        assert result is not None
        assert result["images"] == [tmp_image]
        assert result["remainder"] == "look at and tell me what's wrong"

    def test_image_at_end_with_question_mark(self, tmp_image):
        result = _detect_inline_image_paths(f"whats in {tmp_image}?")
        assert result is not None
        assert result["images"] == [tmp_image]
        assert result["remainder"] == "whats in ?"

    def test_image_at_start_still_works(self, tmp_image):
        result = _detect_inline_image_paths(f"{tmp_image} describe it")
        assert result is not None
        assert result["images"] == [tmp_image]
        assert result["remainder"] == "describe it"

    def test_two_images(self, tmp_image, tmp_image2):
        result = _detect_inline_image_paths(f"compare {tmp_image} and {tmp_image2}")
        assert result is not None
        assert result["images"] == [tmp_image, tmp_image2]
        assert result["remainder"] == "compare and"

    def test_duplicate_image_deduped(self, tmp_image):
        result = _detect_inline_image_paths(f"is {tmp_image} the same as {tmp_image}")
        assert result is not None
        assert result["images"] == [tmp_image]
        assert str(tmp_image) not in result["remainder"]
        assert result["remainder"] == "is the same as"

    def test_quoted_path_with_spaces(self, tmp_image_with_spaces):
        result = _detect_inline_image_paths(
            f'what is in "{tmp_image_with_spaces}" exactly'
        )
        assert result is not None
        assert result["images"] == [tmp_image_with_spaces]
        assert result["remainder"] == "what is in exactly"

    def test_tilde_path(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        img = home / "Pictures" / "cat.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        result = _detect_inline_image_paths("describe ~/Pictures/cat.png please")
        assert result is not None
        assert result["images"] == [img]
        assert result["remainder"] == "describe please"

    def test_only_image_path_no_other_text(self, tmp_image):
        result = _detect_inline_image_paths(f"{tmp_image}")
        assert result is not None
        assert result["images"] == [tmp_image]
        assert result["remainder"] == ""

    @pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".gif", ".webp"])
    def test_common_extensions(self, tmp_path, ext):
        img = tmp_path / f"shot{ext}"
        img.write_bytes(b"fake")
        result = _detect_inline_image_paths(f"see {img}")
        assert result is not None
        assert result["images"] == [img]

    def test_uppercase_extension(self, tmp_path):
        img = tmp_path / "PHOTO.JPG"
        img.write_bytes(b"fake")
        result = _detect_inline_image_paths(f"check {img} out")
        assert result is not None
        assert result["images"] == [img]
        assert result["remainder"] == "check out"

    @pytest.mark.platforms("posix")
    def test_symlink_to_non_image_not_attached(self, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("token=sekrit\n")
        link = tmp_path / "looks.png"
        link.symlink_to(secret)
        assert _detect_inline_image_paths(f"see {link} please") is None

    @pytest.mark.platforms("posix")
    def test_symlink_to_image_resolves_target(self, tmp_image, tmp_path):
        link = tmp_path / "alias.png"
        link.symlink_to(tmp_image)
        result = _detect_inline_image_paths(f"see {link}")
        assert result is not None
        assert result["images"] == [tmp_image.resolve()]
        assert result["remainder"] == "see"


class TestApplyGate:
    def test_skips_when_leading_drop_already_matched(self, tmp_image):
        images: list = []
        text = f"look at {tmp_image}"
        assert (
            _apply_inline_image_paths(text, images, {"path": tmp_image}, False) == text
        )
        assert images == []

    def test_skips_seeded_query(self, tmp_image):
        images: list = []
        text = f"look at {tmp_image}"
        assert _apply_inline_image_paths(text, images, None, True) == text
        assert images == []

    def test_skips_slash_command(self, tmp_image):
        images: list = []
        text = f"/model {tmp_image}"
        assert _apply_inline_image_paths(text, images, None, False) == text
        assert images == []
