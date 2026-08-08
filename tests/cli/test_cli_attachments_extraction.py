"""Regression tests for the s1-w1b extraction of the file-drop / attachment
cluster (cluster c18) from cli.py into ``hermes_cli/cli_attachments.py``.

Covers the pure helpers (path splitting, resolution, badges, termux hint,
clipboard paste policy) and verifies cli.py still re-exports every moved
name so the ``from cli import ...`` API
(tests/cli/test_cli_file_drop.py, tui_gateway/server.py) is unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ATTACHMENT_NAMES = [
    "_IMAGE_EXTENSIONS",
    "_detect_file_drop",
    "_format_image_attachment_badges",
    "_resolve_attachment_path",
    "_should_auto_attach_clipboard_image_on_paste",
    "_split_path_input",
    "_termux_example_image_path",
]


def test_mixin_reexports_attachment_cluster():
    import hermes_cli.cli_attachments as att

    for name in ATTACHMENT_NAMES:
        assert hasattr(att, name), name


def test_cli_still_reexports_attachment_cluster():
    import cli as cli_mod
    import hermes_cli.cli_attachments as att

    for name in ATTACHMENT_NAMES:
        assert getattr(cli_mod, name) is getattr(att, name), name


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", ("", "")),
        ("   ", ("", "")),
        ("/tmp/pic.png describe this", ("/tmp/pic.png", "describe this")),
        ("/tmp/pic.png", ("/tmp/pic.png", "")),
        ('"/storage/emulated/0/DCIM/Camera/cat 1.png" summarize', ("/storage/emulated/0/DCIM/Camera/cat 1.png", "summarize")),
        ("~/storage/shared/My\\ Photos/cat.png what is this?", ("~/storage/shared/My Photos/cat.png", "what is this?")),
        ("'quoted path.png' rest", ("quoted path.png", "rest")),
        ('"unterminated', ("unterminated", "")),
    ],
)
def test_split_path_input(raw, expected):
    from hermes_cli.cli_attachments import _split_path_input

    assert _split_path_input(raw) == expected


def test_should_auto_attach_clipboard_image_on_paste():
    from hermes_cli.cli_attachments import _should_auto_attach_clipboard_image_on_paste

    assert _should_auto_attach_clipboard_image_on_paste("") is True
    assert _should_auto_attach_clipboard_image_on_paste("   ") is True
    assert _should_auto_attach_clipboard_image_on_paste("hello") is False


def test_format_image_attachment_badges_single_compact():
    from hermes_cli.cli_attachments import _format_image_attachment_badges

    imgs = [Path("/tmp/cat.png")]
    assert _format_image_attachment_badges(imgs, image_counter=1, width=40) == "[📎 cat.png]"
    assert _format_image_attachment_badges(imgs, image_counter=1, width=60) == "[📎 cat.png]"
    assert _format_image_attachment_badges(imgs, image_counter=1, width=100) == "[📎 Image #1]"


def test_format_image_attachment_badges_multi():
    from hermes_cli.cli_attachments import _format_image_attachment_badges

    imgs = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    assert _format_image_attachment_badges(imgs, image_counter=5, width=40) == "[📎 2 images attached]"
    assert _format_image_attachment_badges(imgs, image_counter=5, width=70) == "[📎 a.png] [+1]"
    assert _format_image_attachment_badges(imgs, image_counter=5, width=100) == "[📎 Image #4] [📎 Image #5]"


def test_format_image_attachment_badges_empty():
    from hermes_cli.cli_attachments import _format_image_attachment_badges

    assert _format_image_attachment_badges([], image_counter=1, width=40) == ""


def test_termux_example_image_path_default(monkeypatch):
    from hermes_cli.cli_attachments import _termux_example_image_path

    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    assert _termux_example_image_path() == "~/storage/shared/Pictures/cat.png"
    assert _termux_example_image_path("dog.jpg") == "~/storage/shared/Pictures/dog.jpg"


def test_resolve_attachment_path_absolute(tmp_path):
    from hermes_cli.cli_attachments import _resolve_attachment_path

    f = tmp_path / "img.png"
    f.write_bytes(b"x")
    resolved = _resolve_attachment_path(str(f))
    assert resolved is not None
    assert resolved == f.resolve()


def test_resolve_attachment_path_relative_uses_terminal_cwd(tmp_path, monkeypatch):
    from hermes_cli.cli_attachments import _resolve_attachment_path

    f = tmp_path / "img.png"
    f.write_bytes(b"x")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    resolved = _resolve_attachment_path("img.png")
    assert resolved is not None
    assert resolved == f.resolve()


def test_resolve_attachment_path_file_url(tmp_path):
    from hermes_cli.cli_attachments import _resolve_attachment_path

    f = tmp_path / "img.png"
    f.write_bytes(b"x")
    resolved = _resolve_attachment_path(f.as_uri())
    assert resolved is not None
    assert resolved == f.resolve()


def test_resolve_attachment_path_missing_returns_none(tmp_path):
    from hermes_cli.cli_attachments import _resolve_attachment_path

    assert _resolve_attachment_path(str(tmp_path / "nope.png")) is None
    assert _resolve_attachment_path("") is None


def test_detect_file_drop_image(tmp_path):
    from hermes_cli.cli_attachments import _detect_file_drop

    img = tmp_path / "cat.png"
    img.write_bytes(b"x")
    result = _detect_file_drop(str(img))
    assert result is not None
    assert result["is_image"] is True
    assert result["remainder"] == ""


def test_detect_file_drop_text_file(tmp_path):
    from hermes_cli.cli_attachments import _detect_file_drop

    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    result = _detect_file_drop(str(txt))
    assert result is not None
    assert result["is_image"] is False


def test_detect_file_drop_negative_cases(tmp_path):
    from hermes_cli.cli_attachments import _detect_file_drop

    assert _detect_file_drop("") is None
    assert _detect_file_drop("plain text") is None
    assert _detect_file_drop("/nonexistent/path/to/file.png") is None
    assert _detect_file_drop(str(tmp_path / "missing.png")) is None
    assert _detect_file_drop(12345) is None


def test_detect_file_drop_path_with_remainder(tmp_path):
    from hermes_cli.cli_attachments import _detect_file_drop

    img = tmp_path / "cat.png"
    img.write_bytes(b"x")
    result = _detect_file_drop(f"{img} describe this")
    assert result is not None
    assert result["is_image"] is True
    assert result["remainder"] == "describe this"
