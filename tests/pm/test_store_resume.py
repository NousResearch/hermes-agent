"""Store scratch cleanup retains resumable bytes outside the scratch directory."""

from __future__ import annotations

import hashlib

import pytest

from pm.store import Store

import pm.paths as paths

from tests.pm._range_server import RangeHandler as _Handler, url as _url
from tests.pm._range_server import dl_server as dl_server



def test_store_fetch_resumes_interrupted_download(tmp_path, dl_server, monkeypatch):
    # Isolate the store root (and thus the managed partials area) to tmp_path.
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))

    # Serve in small pieces so the mid-body abort fires inside the single
    # byte range (the shared fixture resets this to 1 MiB).
    _Handler.chunk = 1 << 16

    archive_bytes = bytes(range(256)) * 2048
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
    assert first[0][1] == 0
    assert len(first) > 1, "the persistent outage must exhaust automatic retries"
    assert all(start == _Handler.abort_after for _, start, _ in first[1:])

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
