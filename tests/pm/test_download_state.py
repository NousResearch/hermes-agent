"""GC and downloads acquire the same partial owner before changing its files."""
from __future__ import annotations

from contextlib import nullcontext
import os
import time

import pytest

from pm.cli import _gc_store
from pm.download_state import partial_lock
from pm.lock import Facts
from pm.store import Store


@pytest.mark.parametrize("state", ["held", "recent-pair", "unused"])
def test_gc_respects_partial_ownership_and_recent_resume(tmp_path, monkeypatch, state):
    partials = tmp_path / "partials"
    partials.mkdir()
    part = partials / "example.part"
    side = partials / "example.ranges"
    part.write_bytes(b"partial")
    side.write_text("[]", encoding="utf-8")
    old = time.time() - 10 * 24 * 60 * 60
    for path in (part, side):
        os.utime(path, (old, old))
    if state == "recent-pair":
        side.touch()
    monkeypatch.setattr("pm.paths.partials_root", lambda: partials)
    store = Store(tmp_path / "store")
    with partial_lock(partials, "example") if state == "held" else nullcontext():
        _gc_store(store, Facts(store.root / "facts.json"))
        assert part.exists() is (state != "unused")
        assert side.exists() is (state != "unused")
    if state == "held":
        assert (partials / ".locks" / "example").is_file()
        _gc_store(store, Facts(store.root / "facts.json"))
        assert not part.exists() and not side.exists()
        assert (partials / ".locks" / "example").is_file()


def test_pause_interrupts_an_owner_wait(tmp_path):
    from threading import Event, Thread
    from pm.downloader import Download, DownloadPaused, Source

    destination = tmp_path / "already-present"
    destination.write_bytes(b"complete")
    download = Download([Source("https://example.invalid/model", destination)], partials_dir=tmp_path / "partials")
    waiting = Event()
    finished = Event()
    errors = []
    original_key = download._key

    def key(url):
        waiting.set()
        return original_key(url)

    download._key = key

    def run():
        try:
            download.run()
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    with partial_lock(download.partials_dir, original_key(download.sources[0].url)):
        worker = Thread(target=run)
        worker.start()
        assert waiting.wait(timeout=5)
        download.pause()
        stopped_while_locked = finished.wait(timeout=5)
    worker.join(timeout=5)
    assert stopped_while_locked
    assert len(errors) == 1 and isinstance(errors[0], DownloadPaused)
    assert destination.read_bytes() == b"complete"
