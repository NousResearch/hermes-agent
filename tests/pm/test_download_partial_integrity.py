"""A resume bitmap is usable only while its partial file still contains those bytes."""

import hashlib
import json
import os
import time

from pm.downloader import Download, Source, gc_protected_names
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


def test_missing_partial_cannot_turn_a_bitmap_into_completed_model(tmp_path, dl_server):
    payload = b"real model bytes"
    RangeHandler.payloads["/model"] = payload
    source_url = url(dl_server, "/model")
    partials = tmp_path / "partials"
    partials.mkdir()
    key = hashlib.sha256(source_url.encode()).hexdigest()
    (partials / f"{key}.ranges").write_text(json.dumps([[0, len(payload)]]), encoding="utf-8")
    dest = tmp_path / "model.gguf"
    Download([Source(source_url, dest)], partials_dir=partials).run()
    assert dest.read_bytes() == payload


def test_gc_protects_both_halves_of_live_partial(tmp_path):
    partials = tmp_path / "partials"
    partials.mkdir()
    part = partials / "key.part"
    side = partials / "key.ranges"
    part.write_bytes(b"partial")
    side.write_text("[]", encoding="utf-8")
    old = time.time() - 100
    os.utime(side, (old, old))
    assert {part.name, side.name} <= gc_protected_names(partials, grace_seconds=30)
