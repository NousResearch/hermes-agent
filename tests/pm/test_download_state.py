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


@pytest.mark.parametrize("state", ["held", "recent-part", "recent-ranges", "unused"])
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
    if state.startswith("recent-"):
        (part if state == "recent-part" else side).touch()
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


def test_gc_dry_run_probes_partials_without_creating_locks(tmp_path, monkeypatch):
    """A dry run deletes nothing, so it must create nothing either: probing an
    expired partial takes its lock only when a lock file already exists, which
    keeps the .locks dir uncreated on a fresh partials dir (and a read-only or
    restricted one reportable) instead of minting a lock inode per key."""
    partials = tmp_path / "partials"
    partials.mkdir()
    part = partials / "example.part"
    part.write_bytes(b"stale")
    old = time.time() - 10 * 24 * 60 * 60
    os.utime(part, (old, old))
    monkeypatch.setattr("pm.paths.partials_root", lambda: partials)
    store = Store(tmp_path / "store")
    facts = Facts(store.root / "facts.json")

    removed, _ = _gc_store(store, facts, dry_run=True)
    assert removed == 1, "an expired partial with no lock file is probeable unlocked"
    assert part.exists(), "dry run must not unlink the expired partial"
    assert not (partials / ".locks").exists(), "dry run must not create the .locks dir"

    with partial_lock(partials, "example"):
        pass  # leaves the lock inode behind with no owner
    removed, _ = _gc_store(store, facts, dry_run=True)
    assert removed == 1, "an idle lock file is still openable read-write by a probe"
    assert sorted(path.name for path in (partials / ".locks").iterdir()) == ["example"]

    with partial_lock(partials, "example"):
        removed, _ = _gc_store(store, facts, dry_run=True)
        assert removed == 0, "a held lock still hides the partial from the report"
        assert part.exists()
    assert sorted(path.name for path in (partials / ".locks").iterdir()) == ["example"]


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
