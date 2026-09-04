"""Store.fetch resume end-to-end: a dropped archive download resumes on the
second Store.fetch instead of re-fetching the whole archive.

A real loopback Range-honoring server, a real tar.gz, no urllib mocking.
The first fetch is cut short by the server closing the connection mid-body
(so Store.fetch raises); the second fetch must resume from the durable
byte-range prefix — requesting only the missing tail, not the whole archive —
and return bytes that exactly match the original tar.gz. The managed partials
live OUTSIDE the scratch tempdir (in the store's machine-scoped partials
area), so the scratch context's finally-rmtree cannot destroy resume state.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from pm.store import Store

import pm.paths as paths

from tests.pm._range_server import RangeHandler as _Handler, dl_server, url as _url


def _make_archive() -> bytes:
    """A real tar.gz, big enough to stream across several 64 KiB serve
    pieces yet small enough (< 1 MiB) that the downloader fans out to ONE
    byte range — keeping the resume shape a single 'missing tail' request
    that the assertions can check literally. The payload is pseudo-random
    (hash chain) so gzip cannot compress it down below the serve-piece
    size, which would collapse the whole body into one write and spare the
    mid-body abort."""
    data = b""
    seed = b"hermes-pm-resume"
    while len(data) < 512 * 1024:
        seed = hashlib.sha256(seed).digest()
        data += seed
    data = data[:512 * 1024]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("payload.bin")
        info.size = len(data)
        info.mode = 0o644
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_store_fetch_resumes_interrupted_download(tmp_path, dl_server, monkeypatch):
    # Isolate the store root (and thus the managed partials area) to tmp_path.
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    paths._stamp.cache_clear()

    # Serve in small pieces so the mid-body abort fires inside the single
    # byte range (the shared fixture resets this to 1 MiB).
    _Handler.chunk = 1 << 16

    archive_bytes = _make_archive()
    assert len(archive_bytes) < (1 << 20), "single-range resume assumes < 1 MiB"
    name = "faketool-1.0.tar.gz"
    _Handler.payloads[f"/{name}"] = archive_bytes
    sha = hashlib.sha256(archive_bytes).hexdigest()
    url = _url(dl_server, f"/{name}")

    store = Store(runtime / "store")
    partials = paths.partials_root()
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()

    # First fetch: the server drops the connection ~halfway through the body,
    # so Store.fetch raises before publishing anything to the store.
    _Handler.abort_after = len(archive_bytes) // 2
    with store.scratch() as scratch:
        with pytest.raises(Exception):
            store.fetch(url, sha, scratch)

    # The partial + range bitmap survive OUTSIDE scratch (scratch was just
    # rmtree'd by the context manager) — that is the resume state.
    assert (partials / f"{key}.part").is_file()
    assert (partials / f"{key}.ranges").is_file()

    first = list(_Handler.ranges_seen)
    assert len(first) == 1, f"expected a single-range first fetch, saw {first}"

    # Second fetch, server intact: must resume from the durable prefix.
    _Handler.abort_after = None
    with store.scratch() as scratch:
        got = store.fetch(url, sha, scratch)

    # The returned archive's bytes exactly match the original tar.gz.
    assert got.read_bytes() == archive_bytes
    # The successful fetch consumed (moved) the partial out of the managed area.
    assert not (partials / f"{key}.part").exists()

    resumed = _Handler.ranges_seen[len(first):]
    # One resumed request, for ONLY the missing tail — the already-durable
    # prefix was NOT re-requested (it must not start at byte 0).
    assert len(resumed) == 1, f"expected a single resumed request, saw {resumed}"
    r_path, r_start, r_end = resumed[0]
    assert r_path == f"/{name}"
    assert r_start > 0, f"resume re-fetched the whole archive: {resumed}"
    assert r_end == len(archive_bytes) - 1
    # The resume request covered everything from where the first attempt
    # stopped to the end of the file.
    assert r_start < len(archive_bytes)
