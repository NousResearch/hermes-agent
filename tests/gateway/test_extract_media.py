"""Unit tests for ``BasePlatformAdapter.extract_media``.

The MEDIA:<path> directive is the canonical way for an agent's text
reply to ship a file as a native platform attachment. Hermes core
parses the directive out, strips it from the visible text, and routes
the path to the platform adapter's ``send_document`` /
``send_voice`` / ``send_image_file`` based on extension.

Until 2026-05-25 the regex's extension allowlist omitted several text
formats that agents commonly emit — most notably ``.md``. The result:
an agent that wrote ``MEDIA:/tmp/report.md`` saw the directive
silently dropped (cleaned out of visible text by a secondary regex,
but never extracted into ``media_files``), so the recipient got
neither the attachment nor an error explaining why. These tests pin
the broadened allowlist so the bug doesn't quietly come back.
"""

from __future__ import annotations

import pytest

from gateway.platforms.base import BasePlatformAdapter


# ── Newly supported text / data / source formats ────────────────────────


@pytest.mark.parametrize(
    "ext",
    [
        "md", "markdown",
        "json", "yaml", "yml", "toml",
        "csv", "tsv",
        "html", "htm", "xml",
        "log",
        "sh", "py", "js", "ts",
    ],
)
def test_extract_media_text_data_source_extensions(ext: str):
    """MEDIA: paths with text/data/source extensions must extract cleanly.

    These are extensions agents commonly emit (markdown reports, JSON
    payloads, log dumps, code review files). Before this fix, the
    extract path's allowlist silently dropped them.
    """
    path = f"/tmp/test.{ext}"
    content = f"Here is the file.\nMEDIA:{path}"
    media, cleaned = BasePlatformAdapter.extract_media(content)
    assert media == [(path, False)], (
        f"extract_media dropped .{ext}; the regex's extension "
        f"allowlist on gateway/platforms/base.py is missing it."
    )
    assert "MEDIA:" not in cleaned
    assert cleaned == "Here is the file."


# ── Regression: previously supported extensions still match ─────────────


@pytest.mark.parametrize(
    "ext",
    [
        "png", "jpg", "jpeg", "gif", "webp",   # images
        "mp4", "mov", "avi", "mkv", "webm",    # video
        "ogg", "opus", "mp3", "wav", "m4a", "flac",  # audio
        "epub", "pdf", "zip", "rar", "7z",     # bundles
        "docx", "xlsx", "pptx",                # office
        "txt",                                  # plain text
        "apk", "ipa",                           # mobile installers
    ],
)
def test_extract_media_keeps_previously_supported_extensions(ext: str):
    """Broadening the allowlist must not regress any prior extension."""
    path = f"/tmp/test.{ext}"
    media, cleaned = BasePlatformAdapter.extract_media(f"MEDIA:{path}")
    assert media == [(path, False)]
    assert "MEDIA:" not in cleaned


# ── Negative: still rejects unknown extensions ──────────────────────────


@pytest.mark.parametrize("ext", ["xyz", "exe", "dmg", "iso", "deb", "rpm"])
def test_extract_media_rejects_unknown_extensions(ext: str):
    """Unknown extensions must not be picked up — this guards against
    overshooting the allowlist into platform-incompatible files.
    """
    media, cleaned = BasePlatformAdapter.extract_media(f"MEDIA:/tmp/x.{ext}")
    assert media == []
    # Unknown extensions stay in the visible text (the secondary
    # cleanup regex in base.py:~3476 also won't match them, since
    # that regex is greedier but only runs when paired with the
    # extract path).
    assert "MEDIA:" in cleaned


# ── Real-world variants the broadened regex must keep handling ──────────


def test_extract_media_markdown_with_caption_preserves_text():
    """The agent's usual reply: caption + MEDIA: on its own line."""
    content = (
        "Aquí está el resumen del repositorio.\n"
        "MEDIA:/Users/me/.hermes/document_cache/summary.md"
    )
    media, cleaned = BasePlatformAdapter.extract_media(content)
    assert media == [
        ("/Users/me/.hermes/document_cache/summary.md", False),
    ]
    assert cleaned == "Aquí está el resumen del repositorio."


def test_extract_media_handles_multiple_mixed_extensions():
    """Multiple MEDIA: directives — mix of old and newly supported types."""
    content = (
        "Three artifacts attached.\n"
        "MEDIA:/tmp/report.md\n"
        "MEDIA:/tmp/data.json\n"
        "MEDIA:/tmp/screenshot.png"
    )
    media, cleaned = BasePlatformAdapter.extract_media(content)
    assert len(media) == 3
    paths = {p for p, _ in media}
    assert paths == {
        "/tmp/report.md",
        "/tmp/data.json",
        "/tmp/screenshot.png",
    }
    assert "MEDIA:" not in cleaned


def test_extract_media_audio_voice_tag_still_honored():
    """[[audio_as_voice]] still flips is_voice=True for any audio path."""
    content = "[[audio_as_voice]]\nMEDIA:/tmp/reply.opus"
    media, cleaned = BasePlatformAdapter.extract_media(content)
    assert media == [("/tmp/reply.opus", True)]
    assert "[[audio_as_voice]]" not in cleaned
