from pathlib import Path
from unittest.mock import patch

from cli import (
    HermesCLI,
    _collect_query_images,
    _format_image_attachment_badges,
    _termux_example_image_path,
)


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._attached_images = []
    return cli_obj


def _make_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


class TestImageCommand:
    def test_handle_image_command_attaches_local_image(self, tmp_path):
        img = _make_image(tmp_path / "photo.png")
        cli_obj = _make_cli()

        with patch("cli._cprint"):
            cli_obj._handle_image_command(f"/image {img}")

        assert cli_obj._attached_images == [img]

    def test_handle_image_command_supports_quoted_path_with_spaces(self, tmp_path):
        img = _make_image(tmp_path / "my photo.png")
        cli_obj = _make_cli()

        with patch("cli._cprint"):
            cli_obj._handle_image_command(f'/image "{img}"')

        assert cli_obj._attached_images == [img]

    def test_handle_image_command_rejects_non_image_file(self, tmp_path):
        file_path = tmp_path / "notes.txt"
        file_path.write_text("hello\n", encoding="utf-8")
        cli_obj = _make_cli()

        with patch("cli._cprint") as mock_print:
            cli_obj._handle_image_command(f"/image {file_path}")

        assert cli_obj._attached_images == []
        rendered = " ".join(str(arg) for call in mock_print.call_args_list for arg in call.args)
        assert "Not a supported image file" in rendered


class TestCollectQueryImages:
    def test_collect_query_images_accepts_explicit_image_arg(self, tmp_path):
        img = _make_image(tmp_path / "diagram.png")

        message, images = _collect_query_images("describe this", str(img))

        assert message == "describe this"
        assert images == [img]

    def test_collect_query_images_extracts_leading_path(self, tmp_path):
        img = _make_image(tmp_path / "camera.png")

        message, images = _collect_query_images(f"{img} what do you see?")

        assert message == "what do you see?"
        assert images == [img]

    def test_collect_query_images_supports_tilde_paths(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        img = _make_image(home / "storage" / "shared" / "Pictures" / "cat.png")
        monkeypatch.setenv("HOME", str(home))

        message, images = _collect_query_images("describe this", "~/storage/shared/Pictures/cat.png")

        assert message == "describe this"
        assert images == [img]


class TestTermuxImageHints:
    def test_termux_example_image_path_prefers_real_shared_storage_root(self, monkeypatch):
        existing = {"/sdcard", "/storage/emulated/0"}
        monkeypatch.setattr("cli.os.path.isdir", lambda path: path in existing)

        hint = _termux_example_image_path()

        assert hint == "/sdcard/Pictures/cat.png"


class TestImageBadgeFormatting:
    def test_compact_badges_use_filename_on_narrow_terminals(self, tmp_path):
        img = _make_image(tmp_path / "Screenshot 2026-04-09 at 11.22.33 AM.png")

        badges = _format_image_attachment_badges([img], image_counter=1, width=40)

        assert badges.startswith("[📎 ")
        assert "Image #1" not in badges

    def test_compact_badges_summarize_multiple_images(self, tmp_path):
        img1 = _make_image(tmp_path / "one.png")
        img2 = _make_image(tmp_path / "two.png")

        badges = _format_image_attachment_badges([img1, img2], image_counter=2, width=45)

        assert badges == "[📎 2 images attached]"
