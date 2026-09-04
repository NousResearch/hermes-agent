"""pm/downloader: resumable, hash-verified, 8-way parallel downloads.

Real downloads against a loopback Range-honoring server — no mocked
stores. Covers the seams agreed in the plan: run() semantics, the
progress(overall_done, overall_total, ranges) contract, parallelism,
resume-refetch-only-missing, pause, and the optional-hash policy.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from pm.downloader import (Download, DownloadError, DownloadPaused,
                           HashError, Source, gc_protected_names)

from tests.pm._range_server import RangeHandler as _Handler, dl_server, url as _url


def _payload(n: int, seed: bytes = b"x") -> bytes:
    return seed * n


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── parallelism ───────────────────────────────────────────────


def test_parallel_uses_8_connections(dl_server, tmp_path):
    total = 8 * (4 << 20)  # 32 MiB -> 8 x 4 MiB ranges
    payload = _payload(total)
    _Handler.payloads["/big"] = payload
    dest = tmp_path / "big.bin"
    dl = Download([Source(_url(dl_server, "/big"), dest, _sha(payload))],
                  partials_dir=tmp_path / "partials")
    dl.run()
    assert dest.read_bytes() == payload
    expected = [(i * total // 8, (i + 1) * total // 8 - 1) for i in range(8)]
    assert sorted((s, e) for _, s, e in _Handler.ranges_seen) == expected


def test_progress_carries_overall_and_ranges(dl_server, tmp_path):
    p1, p2 = _payload(1000, b"x"), _payload(2000, b"y")
    _Handler.payloads["/a"] = p1
    _Handler.payloads["/b"] = p2
    d1, d2 = tmp_path / "a.bin", tmp_path / "b.bin"
    dl = Download([
        Source(_url(dl_server, "/a"), d1, _sha(p1)),
        Source(_url(dl_server, "/b"), d2, _sha(p2)),
    ], partials_dir=tmp_path / "partials")
    seen = []
    dl.run(progress=lambda d, t, r: seen.append((d, t, dict(r))))
    final_d, final_t, final_r = seen[-1]
    assert final_d == 3000
    assert final_t == 3000
    assert final_r["a.bin"] == [(0, 1000)]
    assert final_r["b.bin"] == [(0, 2000)]


# ── optional hash ─────────────────────────────────────────────


def test_hash_mismatch_raises_and_deletes_part(dl_server, tmp_path):
    _Handler.payloads["/f"] = _payload(1 << 20)
    dest = tmp_path / "f.bin"
    partials = tmp_path / "partials"
    dl = Download([Source(_url(dl_server, "/f"), dest, "0" * 64)],
                  partials_dir=partials)
    with pytest.raises(HashError):
        dl.run()
    assert not dest.exists()
    assert list(partials.iterdir()) == []  # .part + .ranges both deleted


def test_no_hash_accepts_any_bytes_of_right_size(dl_server, tmp_path):
    payload = _payload(1 << 20)
    _Handler.payloads["/f"] = payload
    dest = tmp_path / "f.bin"
    dl = Download([Source(_url(dl_server, "/f"), dest)],
                  partials_dir=tmp_path / "partials")
    moved = dl.run()
    assert moved == [dest]
    assert dest.read_bytes() == payload


# ── resume ────────────────────────────────────────────────────


def test_resume_refetches_only_missing_ranges(dl_server, tmp_path):
    payload = _payload(8 << 20)
    _Handler.payloads["/r"] = payload
    dest = tmp_path / "r.bin"
    partials = tmp_path / "partials"
    # first run: one connection, server drops after 2 MiB
    _Handler.abort_after = 2 << 20
    dl = Download([Source(_url(dl_server, "/r"), dest, _sha(payload))],
                  partials_dir=partials, connections=1)
    with pytest.raises(Exception):
        dl.run()
    first = list(_Handler.ranges_seen)
    # second run resumes: only the missing tail is re-fetched
    _Handler.abort_after = None
    dl = Download([Source(_url(dl_server, "/r"), dest, _sha(payload))],
                  partials_dir=partials, connections=1)
    dl.run()
    assert dest.read_bytes() == payload
    assert _Handler.ranges_seen == first + [("/r", 2 << 20, (8 << 20) - 1)]


def test_resume_after_crash_reads_sidecar(dl_server, tmp_path):
    payload = _payload(4 << 20)
    _Handler.payloads["/c"] = payload
    dest = tmp_path / "c.bin"
    partials = tmp_path / "partials"
    partials.mkdir()
    key = hashlib.sha256(_url(dl_server, "/c").encode("utf-8")).hexdigest()
    (partials / f"{key}.part").write_bytes(payload[: 1 << 20])
    (partials / f"{key}.ranges").write_text(json.dumps([[0, 1 << 20]]))
    dl = Download([Source(_url(dl_server, "/c"), dest, _sha(payload))],
                  partials_dir=partials)
    dl.run()
    assert dest.read_bytes() == payload
    assert _Handler.ranges_seen == [("/c", 1 << 20, (4 << 20) - 1)]


def test_partials_never_in_scratch_or_dest(dl_server, tmp_path):
    payload = _payload(1 << 20)
    _Handler.payloads["/p"] = payload
    dest = tmp_path / "p.bin"
    partials = tmp_path / "partials"
    dl = Download([Source(_url(dl_server, "/p"), dest)],
                  partials_dir=partials)
    dl.run()
    # dest is the only file in its dir; partials dir is empty after mv.
    # (The test harness may drop its own marker dir into tmp_path.)
    assert dest.exists()
    assert list(partials.iterdir()) == []
    assert not [p for p in tmp_path.iterdir()
                if p.suffix in (".part", ".ranges")]


def test_completed_dest_is_skipped(dl_server, tmp_path):
    payload = _payload(1 << 20)
    _Handler.payloads["/s"] = payload
    dest = tmp_path / "s.bin"
    dest.write_bytes(payload)
    dl = Download([Source(_url(dl_server, "/s"), dest, _sha(payload))],
                  partials_dir=tmp_path / "partials")
    dl.run()
    assert _Handler.ranges_seen == []  # nothing fetched


def test_stale_dest_with_wrong_hash_is_refetched(dl_server, tmp_path):
    payload = _payload(1 << 20)
    _Handler.payloads["/s"] = payload
    dest = tmp_path / "s.bin"
    dest.write_bytes(_payload(1 << 20, seed=b"wrong"))  # wrong bytes on disk
    dl = Download([Source(_url(dl_server, "/s"), dest, _sha(payload))],
                  partials_dir=tmp_path / "partials")
    dl.run()
    assert dest.read_bytes() == payload  # stale dest replaced
    assert _Handler.ranges_seen  # a fetch actually happened


def test_completed_dest_without_hash_is_skipped(dl_server, tmp_path):
    # Model-catalog policy: no pinned hash -> a present file is accepted
    # (size is the only tripwire), matching fresh-download semantics.
    payload = _payload(1 << 20)
    _Handler.payloads["/s"] = payload
    dest = tmp_path / "s.bin"
    dest.write_bytes(payload)
    dl = Download([Source(_url(dl_server, "/s"), dest)],
                  partials_dir=tmp_path / "partials")
    dl.run()
    assert _Handler.ranges_seen == []  # nothing fetched


def test_redirect_to_non_https_refused():
    import urllib.request

    from pm.downloader import _HttpsRedirectHandler

    handler = _HttpsRedirectHandler()
    req = urllib.request.Request("https://example.com/a")
    with pytest.raises(DownloadError):
        handler.redirect_request(req, None, 302, "Found", {},
                                 "http://example.com/b")
    # An https redirect resolves through the default handler.
    assert handler.redirect_request(req, None, 302, "Found", {},
                                    "https://example.com/b") is not None


# ── pause ─────────────────────────────────────────────────────


def test_pause_leaves_partials_intact(dl_server, tmp_path):
    _Handler.payloads["/p"] = _payload(32 << 20)
    _Handler.slow_per_chunk = 0.01
    dest = tmp_path / "p.bin"
    partials = tmp_path / "partials"
    dl = Download([Source(_url(dl_server, "/p"), dest)],
                  partials_dir=partials)
    result: dict = {}

    def run_it():
        try:
            dl.run()
            result["ok"] = True
        except DownloadPaused:
            result["paused"] = True

    thread = threading.Thread(target=run_it)
    thread.start()
    for _ in range(200):
        if partials.exists() and any(partials.iterdir()):
            break
        time.sleep(0.05)
    dl.pause()
    thread.join(timeout=15)
    assert result.get("paused")
    assert not dest.exists()
    names = {p.name for p in partials.iterdir()}
    assert any(n.endswith(".part") for n in names)
    assert any(n.endswith(".ranges") for n in names)


# ── safety ────────────────────────────────────────────────────


def test_refuses_non_https_non_loopback(tmp_path):
    dl = Download([Source("http://example.com/x", tmp_path / "x.bin")],
                  partials_dir=tmp_path / "partials")
    with pytest.raises(ValueError):
        dl.run()


# ── edge cases: no-Range fallback, multi-source resume, pause mid-plan ──


def test_no_range_fallback_downloads_full_body(dl_server, tmp_path):
    """A server that ignores Range is served by the single-stream fallback:
    the whole body arrives and progress reports the full covered range."""
    _Handler.no_range = True
    payload = _payload(1 << 20)
    _Handler.payloads["/nr"] = payload
    dest = tmp_path / "nr.bin"
    dl = Download([Source(_url(dl_server, "/nr"), dest, _sha(payload))],
                  partials_dir=tmp_path / "partials")
    seen = []
    moved = dl.run(progress=lambda d, t, r: seen.append((d, t, dict(r))))
    assert moved == [dest]
    assert dest.read_bytes() == payload
    assert _Handler.ranges_seen == []  # never used Range
    d, t, r = seen[-1]
    assert d == t == len(payload)
    assert r["nr.bin"] == [(0, len(payload))]


def test_no_range_short_body_raises_and_leaves_no_dest(dl_server, tmp_path):
    """The fallback errors when the server sends FEWER bytes than its
    declared Content-Length (connection dropped mid-body), leaving the
    partial in the managed area and NO dest."""
    _Handler.no_range = True
    payload = _payload(2 << 20)
    _Handler.payloads["/ns"] = payload
    _Handler.abort_after = 1 << 20  # server declares 2 MiB, sends 1 MiB
    dest = tmp_path / "ns.bin"
    partials = tmp_path / "partials"
    dl = Download([Source(_url(dl_server, "/ns"), dest)],
                  partials_dir=partials)
    with pytest.raises(DownloadError):
        dl.run()
    assert not dest.exists()
    names = {p.name for p in partials.iterdir()}
    assert any(n.endswith(".part") for n in names)
    assert any(n.endswith(".ranges") for n in names)


def test_resume_across_plan_skips_completed_source(dl_server, tmp_path):
    """A 2-source plan: source 1 completes, source 2 aborts. A second run
    of the SAME plan completes BOTH files without re-fetching source 1 —
    source 1 gets exactly one request across both runs and source 2's
    second request starts at its abort point."""
    total_b = 8 << 20
    p_a, p_b = _payload(1 << 20, b"a"), _payload(total_b, b"b")
    _Handler.payloads["/a"] = p_a
    _Handler.payloads["/b"] = p_b
    da, db = tmp_path / "a.bin", tmp_path / "b.bin"
    partials = tmp_path / "partials"

    _Handler.abort_after = 2 << 20
    dl = Download([Source(_url(dl_server, "/a"), da, _sha(p_a)),
                   Source(_url(dl_server, "/b"), db, _sha(p_b))],
                  partials_dir=partials, connections=1)
    with pytest.raises(Exception):
        dl.run()

    _Handler.abort_after = None
    dl = Download([Source(_url(dl_server, "/a"), da, _sha(p_a)),
                   Source(_url(dl_server, "/b"), db, _sha(p_b))],
                  partials_dir=partials, connections=1)
    dl.run()
    assert da.read_bytes() == p_a
    assert db.read_bytes() == p_b
    a_reqs = [r for r in _Handler.ranges_seen if r[0] == "/a"]
    assert len(a_reqs) == 1  # exactly one request across both runs
    b_reqs = [r for r in _Handler.ranges_seen if r[0] == "/b"]
    # first request starts at 0; second (resume) starts at the abort point
    assert b_reqs[0][1] == 0
    assert b_reqs[1][1] == 2 << 20
    assert b_reqs[1][1] != b_reqs[0][1]


def test_resume_first_progress_shows_durable_prefix(dl_server, tmp_path):
    """On a resumed download, the SECOND run's first progress tick already
    reports the durable prefix as a covered range and overall_done starting
    above zero."""
    total = 8 << 20
    payload = _payload(total)
    _Handler.payloads["/rp"] = payload
    dest = tmp_path / "rp.bin"
    partials = tmp_path / "partials"

    _Handler.abort_after = 2 << 20
    dl = Download([Source(_url(dl_server, "/rp"), dest, _sha(payload))],
                  partials_dir=partials, connections=1)
    with pytest.raises(Exception):
        dl.run()

    _Handler.abort_after = None
    dl = Download([Source(_url(dl_server, "/rp"), dest, _sha(payload))],
                  partials_dir=partials, connections=1)
    seen = []
    dl.run(progress=lambda d, t, r: seen.append((d, t, dict(r))))
    assert dest.read_bytes() == payload
    d0, t0, r0 = seen[0]
    assert d0 > 0  # overall_done starts above zero (durable prefix + new)
    assert t0 == total
    ranges = r0["rp.bin"]
    assert len(ranges) == 1  # coalesced
    assert ranges[0][0] == 0  # the durable prefix is a covered range
    assert ranges[0][1] > 0


def test_pause_mid_plan_resumes_to_completion(dl_server, tmp_path):
    """Pause after source 1 completes: DownloadPaused is raised, source 1's
    dest is moved/complete, and source 2's partial survives in the managed
    area; a FRESH Download over the same plan resumes to full completion."""
    p_a, p_b = _payload(1 << 20, b"a"), _payload(32 << 20, b"b")
    _Handler.payloads["/a"] = p_a
    _Handler.payloads["/b"] = p_b
    da, db = tmp_path / "a.bin", tmp_path / "b.bin"
    partials = tmp_path / "partials"
    _Handler.slow_per_chunk = 0.01

    dl = Download([Source(_url(dl_server, "/a"), da, _sha(p_a)),
                   Source(_url(dl_server, "/b"), db, _sha(p_b))],
                  partials_dir=partials)
    result: dict = {}

    def run_it():
        try:
            dl.run()
            result["ok"] = True
        except DownloadPaused:
            result["paused"] = True
        except Exception as exc:  # noqa: BLE001
            result["err"] = exc

    thread = threading.Thread(target=run_it)
    thread.start()
    for _ in range(400):
        if da.exists():
            break
        time.sleep(0.05)
    assert da.exists(), "source 1 never completed"
    dl.pause()
    thread.join(timeout=20)
    assert result.get("paused"), result.get("err")
    assert da.read_bytes() == p_a  # source 1 moved despite the pause
    assert not db.exists()
    names = {p.name for p in partials.iterdir()}
    assert any(n.endswith(".part") for n in names)
    assert any(n.endswith(".ranges") for n in names)

    # a FRESH Download over the same plan resumes to full completion
    _Handler.slow_per_chunk = 0.0
    dl = Download([Source(_url(dl_server, "/a"), da, _sha(p_a)),
                   Source(_url(dl_server, "/b"), db, _sha(p_b))],
                  partials_dir=partials)
    dl.run()
    assert da.read_bytes() == p_a
    assert db.read_bytes() == p_b


def test_gc_protected_names_live_partial(tmp_path):
    """gc protection: entries touched inside the grace window are live."""
    import os

    partials = tmp_path / "partials"
    partials.mkdir()
    (partials / "abc.part").write_bytes(b"p")
    (partials / "abc.ranges").write_text("[]")
    (partials / "old.part").write_bytes(b"p")
    # Backdate the stale entry beyond any plausible grace window.
    old = time.time() - 10 * 24 * 3600
    os.utime(partials / "old.part", (old, old))

    protected = gc_protected_names(partials)
    assert "abc.part" in protected
    assert "abc.ranges" in protected  # either half protects the pair
    assert "old.part" not in protected

    # A grace window of zero protects nothing.
    assert gc_protected_names(partials, grace_seconds=0) == set()


def test_gc_protected_names_missing_dir(tmp_path):
    assert gc_protected_names(tmp_path / "nope") == set()
