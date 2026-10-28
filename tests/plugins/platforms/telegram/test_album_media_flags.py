"""Album batching preserves per-attachment text-inlining semantics at admission."""
from types import SimpleNamespace

import pytest

from gateway.platforms.event import MessageEvent
from gateway.session_envelope import _media_clause
from plugins.platforms.telegram.adapter import TelegramAdapter


def test_six_document_album_passes_native_media_validation(tmp_path):
    adapter = SimpleNamespace(_merge_caption=TelegramAdapter._merge_caption)
    pending = {}
    for index in range(6):
        event = MessageEvent(text="", media_urls=[str(tmp_path / f"invoice-{index}.pdf")],
                             media_types=["application/pdf"], media_text_inlined=[False])
        TelegramAdapter._merge_into_pending(adapter, pending, "album", event)
    album = pending["album"]
    assert _media_clause(album) is None
    assert album.media_text_inlined == [False] * 6
    assert album.media_urls == [str(tmp_path / f"invoice-{index}.pdf") for index in range(6)]


@pytest.mark.parametrize("flags", [([], [True]), ([False], []), ([True], [False]), ([False], [True]), ([], [])])
def test_album_preserves_legacy_and_explicit_inlining_flags(flags, tmp_path):
    adapter = SimpleNamespace(_merge_caption=TelegramAdapter._merge_caption)
    pending = {}
    for index, inlined in enumerate(flags):
        event = MessageEvent(text=f"caption-{index}", media_urls=[str(tmp_path / f"file-{index}")],
                             media_types=["text/plain"], media_text_inlined=inlined,
                             media_filenames=[f"file-{index}"])
        TelegramAdapter._merge_into_pending(adapter, pending, "album", event)
    album = pending["album"]
    assert _media_clause(album) is None
    assert album.media_text_inlined == [value for flag in flags for value in (flag or [None])]
    assert album.media_filenames == ["file-0", "file-1"]
    assert album.text == "caption-0\n\ncaption-1"
