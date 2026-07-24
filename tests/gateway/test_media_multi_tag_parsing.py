"""Regression tests for adjacent MEDIA tag parsing."""

from gateway.platforms.base import BasePlatformAdapter


def test_extract_media_handles_multiple_tags_on_one_line_without_swallowing_second_path(tmp_path):
    """Comma/space-separated MEDIA tags must become separate attachments.

    The extractor supports paths with spaces, but that must not make one MEDIA:
    tag consume the next ``MEDIA:`` tag on the same line.
    """
    first = tmp_path / "first note.md"
    second = tmp_path / "second note.md"
    first.write_text("# first\n")
    second.write_text("# second\n")

    content = f"Attached: MEDIA:{first}, MEDIA:{second}"

    media, cleaned = BasePlatformAdapter.extract_media(content)

    assert [path for path, _is_voice in media] == [str(first), str(second)]
    assert "MEDIA:" not in cleaned
    assert str(first) not in cleaned
    assert str(second) not in cleaned


def test_extract_media_still_allows_single_unquoted_path_with_spaces(tmp_path):
    spaced = tmp_path / "monthly report.md"
    spaced.write_text("# report\n")

    media, cleaned = BasePlatformAdapter.extract_media(f"MEDIA:{spaced}")

    assert media == [(str(spaced), False)]
    assert cleaned == ""
