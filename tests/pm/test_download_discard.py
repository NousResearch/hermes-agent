"""Download.discard(): an abandoned download's partials go, a live owner's partials stay."""

from __future__ import annotations

import hashlib

from pm.download_state import partial_lock
from pm.downloader import Download, Source


def _key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def test_discard_removes_free_partials_and_leaves_an_owned_one(tmp_path):
    """Partials are shared machine-wide by URL: one another transfer holds is still in use."""
    partials = tmp_path / "partials"
    partials.mkdir()
    free, owned = "https://example.invalid/free.gguf", "https://example.invalid/owned.gguf"
    for url in (free, owned):
        for suffix in (".part", ".ranges"):
            (partials / f"{_key(url)}{suffix}").write_bytes(b"x")
    download = Download([Source(free, tmp_path / "free.gguf"), Source(owned, tmp_path / "owned.gguf")],
                        partials_dir=partials)

    with partial_lock(partials, _key(owned)):
        kept = download.discard()

    assert kept == [owned]
    assert not (partials / f"{_key(free)}.part").exists()
    assert not (partials / f"{_key(free)}.ranges").exists()
    assert (partials / f"{_key(owned)}.part").exists()
    assert (partials / f"{_key(owned)}.ranges").exists()


def test_discard_with_nothing_on_disk_is_quiet(tmp_path):
    download = Download([Source("https://example.invalid/never.gguf", tmp_path / "never.gguf")],
                        partials_dir=tmp_path / "partials")
    assert download.discard() == []
