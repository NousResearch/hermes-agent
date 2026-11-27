"""Contract tests for per-conversation inbound-media layout.

``_bucket_inbound_media`` (gateway.platforms.base) moves every freshly cached inbound
attachment into ``<cache root>/<platform>/<chat id>/`` so files arriving on a multiplexed
gateway are attributable per channel and conversation instead of piling into one shared
flat directory. These tests lock in:

- all four inbound cache roots (images/documents/audio/videos) get bucketed;
- re-delivery of the same event (batch flush/merge re-dispatch) is idempotent;
- non-root parents — notably the content-addressed ``native-inputs`` replay copies —
  are never touched;
- missing chat id is a no-op; hostile chat ids are sanitized to one safe path segment.
"""

from pathlib import Path

import pytest

from gateway.platforms.base import _bucket_inbound_media

_ROOTS = [
    ("images", "img_abc123.jpg"),
    ("documents", "doc_abc123_report.pdf"),
    ("audio", "audio_abc123.silk"),
    ("videos", "video_abc123.mp4"),
]


@pytest.mark.parametrize("root,name", _ROOTS)
def test_every_inbound_root_is_bucketed(tmp_path, root, name):
    cache_root = tmp_path / root
    cache_root.mkdir()
    src = cache_root / name
    src.write_bytes(b"x")
    out = _bucket_inbound_media(str(src), "Telegram", "1190103421")
    assert out == str(cache_root / "telegram" / "1190103421" / name)
    assert Path(out).is_file() and not src.exists()


def test_redelivery_is_idempotent(tmp_path):
    cache_root = tmp_path / "images"
    cache_root.mkdir()
    src = cache_root / "img_x.jpg"
    src.write_bytes(b"x")
    once = _bucket_inbound_media(str(src), "WECOM", "do@example.cn")
    twice = _bucket_inbound_media(once, "WECOM", "do@example.cn")
    assert twice == once
    assert Path(once).is_file()


def test_non_root_parents_are_untouched(tmp_path):
    # native-inputs/<sha256>/<name> is replay-owned; its parent is the digest dir,
    # which must not match any cache root name.
    replay = tmp_path / "documents" / "native-inputs" / ("c" * 64)
    replay.mkdir(parents=True)
    src = replay / "doc_y.pdf"
    src.write_bytes(b"x")
    assert _bucket_inbound_media(str(src), "WECOM", "chat-1") == str(src)
    assert src.is_file()
    # already-bucketed files (parent is a chat dir) also stay put
    bucketed = tmp_path / "images" / "wecom" / "chat-1"
    bucketed.mkdir(parents=True)
    f = bucketed / "img_z.jpg"
    f.write_bytes(b"x")
    assert _bucket_inbound_media(str(f), "WECOM", "chat-1") == str(f)


def test_empty_chat_id_is_noop(tmp_path):
    cache_root = tmp_path / "images"
    cache_root.mkdir()
    src = cache_root / "img_q.jpg"
    src.write_bytes(b"x")
    assert _bucket_inbound_media(str(src), "TELEGRAM", "") == str(src)
    assert src.is_file()


def test_hostile_chat_id_is_sanitized(tmp_path):
    cache_root = tmp_path / "documents"
    cache_root.mkdir()
    src = cache_root / "doc_k.pdf"
    src.write_bytes(b"x")
    out = Path(_bucket_inbound_media(str(src), "WECOM", "../../escape /evil"))
    seg = out.parent.name
    assert seg not in ("", ".", "..")
    assert "/" not in seg
    assert out.is_file() and not src.exists()
