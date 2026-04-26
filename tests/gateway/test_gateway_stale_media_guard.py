import hashlib
from pathlib import Path

from gateway.run import (
    _gateway_extract_media_paths_for_guard,
    _gateway_media_file_sha256,
    _gateway_strip_stale_media_tags,
)


def test_gateway_media_guard_scans_assistant_history_and_strips_same_path(tmp_path):
    old = tmp_path / "old.png"
    old.write_bytes(b"old-image")
    response = f"done\nMEDIA:{old}"
    cleaned, skipped = _gateway_strip_stale_media_tags(
        response,
        history_media_paths={str(old)},
        history_media_hashes=set(),
    )
    assert str(old) in skipped
    assert "MEDIA:" not in cleaned


def test_gateway_media_guard_strips_new_path_with_old_hash(tmp_path):
    old = tmp_path / "duck.png"
    new = tmp_path / "hotsell.png"
    old.write_bytes(b"same-image-bytes")
    new.write_bytes(b"same-image-bytes")
    old_digest = _gateway_media_file_sha256(str(old))
    assert old_digest == hashlib.sha256(b"same-image-bytes").hexdigest()

    cleaned, skipped = _gateway_strip_stale_media_tags(
        f"new poster\nMEDIA:{new}",
        history_media_paths={str(old)},
        history_media_hashes={old_digest},
    )
    assert str(new) in skipped
    assert "MEDIA:" not in cleaned


def test_gateway_media_guard_keeps_fresh_media(tmp_path):
    old = tmp_path / "old.png"
    fresh = tmp_path / "fresh.png"
    old.write_bytes(b"old")
    fresh.write_bytes(b"fresh")
    cleaned, skipped = _gateway_strip_stale_media_tags(
        f"new poster\nMEDIA:{fresh}",
        history_media_paths={str(old)},
        history_media_hashes={_gateway_media_file_sha256(str(old))},
    )
    assert skipped == []
    assert f"MEDIA:{fresh}" in cleaned


def test_gateway_media_guard_extracts_media_from_assistant_text(tmp_path):
    p = tmp_path / "poster.png"
    p.write_bytes(b"x")
    paths = _gateway_extract_media_paths_for_guard(f"assistant said MEDIA:{p}")
    assert paths == [str(p)]
