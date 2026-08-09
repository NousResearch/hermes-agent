"""``FILE:<path>`` is delivered exactly like ``MEDIA:<path>``.

Models routinely write ``FILE:/path/doc.pdf`` instead of the documented ``MEDIA:`` marker.
Neither the tag extractor nor the bare-path detector (whose lookbehind rejects a path glued
to ``:``) recognised it, so the user received the literal tag as chat text and no file.
"""

import base64

import pytest

from gateway.platforms.base import BasePlatformAdapter
from gateway.stream_consumer_directives import has_pending_media_directive

# 1x1 transparent PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


@pytest.mark.parametrize("wrap", ["{}", "**{}**", "`{}`"])
def test_file_tag_delivers_and_leaves_no_literal_tag(tmp_path, wrap):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    text = "Aquí va el PDF que pediste:\n" + wrap.format(f"FILE:{pdf}")

    media, cleaned = BasePlatformAdapter.extract_media(text)

    assert media == [(str(pdf), False)]
    assert cleaned == "Aquí va el PDF que pediste:"
    assert BasePlatformAdapter.strip_media_directives_for_display(text) == cleaned


def test_media_and_file_tags_coexist_and_dedup(tmp_path):
    png = tmp_path / "a.png"
    pdf = tmp_path / "b.pdf"
    text = f"MEDIA:{png}\nFILE:{pdf}\nagain FILE:{png}"

    media, cleaned = BasePlatformAdapter.extract_media(text)

    assert [p for p, _ in media] == [str(png), str(pdf)]
    assert cleaned == "again"


@pytest.mark.parametrize(
    "text",
    [
        "PROFILE:/tmp/x.pdf",
        "LOG_FILE: /tmp/x.log",
        "see file:///tmp/x.pdf",
        "File: /tmp/x.pdf",
        '{"result": "FILE:/tmp/stale.png"}',
    ],
)
def test_prose_and_stored_text_never_become_file_tags(text):
    media, cleaned = BasePlatformAdapter.extract_media(text)

    assert media == []
    assert cleaned == text


@pytest.mark.parametrize("tail", ["F", "FI", "FILE:", "FILE:/tmp/rep", "**FILE:~/x"])
def test_streaming_holds_back_a_file_tag_still_being_typed(tail):
    assert has_pending_media_directive("Here it is: " + tail)


@pytest.mark.parametrize("tail", ["OFF", "PROFILE:", "file: notes"])
def test_streaming_does_not_hold_back_prose(tail):
    assert not has_pending_media_directive("Here it is " + tail)


def test_api_server_inlines_file_tag_image(tmp_path):
    pytest.importorskip("aiohttp")
    from gateway.platforms.api_server import _resolve_media_to_data_urls

    png = tmp_path / "shot.png"
    png.write_bytes(_PNG)

    out = _resolve_media_to_data_urls(f"Here you go: FILE:{png}")

    assert out.startswith("Here you go: ![image](data:image/png;base64,")
    assert "FILE:" not in out
