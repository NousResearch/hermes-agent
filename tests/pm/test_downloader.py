"""pm/downloader: resumable, hash-verified, 8-way parallel downloads.

Real downloads against a loopback Range-honoring server — no mocked
stores. Covers the seams agreed in the plan: run() semantics, the
progress(overall_done, overall_total, ranges) contract, parallelism,
resume-refetch-only-missing, pause, and the optional-hash policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from pm.downloader import (Download, DownloadError, DownloadPaused,
                           HashError, Source, replace_when_released)

from tests.pm._range_server import RangeHandler as _Handler, url as _url
from tests.pm._range_server import dl_server as dl_server


def _payload(n: int, seed: bytes = b"x") -> bytes:
    return seed * n


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── parallelism ───────────────────────────────────────────────


@pytest.mark.parametrize("connections", [2, 8])
def test_parallel_requests_overlap_without_exceeding_limit(tmp_path, connections):
    from http.server import ThreadingHTTPServer

    total = 8 * (4 << 20)  # 32 MiB -> 8 x 4 MiB ranges
    payload = _payload(total)
    barrier = threading.Barrier(connections)
    lock = threading.Lock()
    active = peak = 0
    broken = []

    class ConcurrentHandler(_Handler):
        payloads = {"/big": payload}
        ranges_seen = []

        def do_GET(self):
            nonlocal active, peak
            if self.headers.get("Range") == "bytes=0-0":
                return super().do_GET()
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                try:
                    barrier.wait(timeout=5)
                except threading.BrokenBarrierError:
                    broken.append(self.headers.get("Range"))
                super().do_GET()
            finally:
                with lock:
                    active -= 1

    dest = tmp_path / "big.bin"
    with ThreadingHTTPServer(("127.0.0.1", 0), ConcurrentHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            Download([Source(_url(server, "/big"), dest, _sha(payload))],
                     connections=connections, partials_dir=tmp_path / "partials").run()
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert dest.read_bytes() == payload
    assert not broken, "range requests were serialized instead of overlapping"
    assert peak == connections and active == 0
    expected = [(i * total // connections, (i + 1) * total // connections - 1) for i in range(connections)]
    assert sorted((s, e) for _, s, e in ConcurrentHandler.ranges_seen) == expected


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
    assert final_r[str(d1)] == [(0, 1000)]
    assert final_r[str(d2)] == [(0, 2000)]


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
    assert not [path for path in partials.iterdir() if path.name != ".locks"]  # .part + .ranges both deleted


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
    assert not [path for path in partials.iterdir() if path.name != ".locks"]
    assert not [p for p in tmp_path.iterdir()
                if p.suffix in (".part", ".ranges")]


@pytest.mark.parametrize("hashed", [False, True])
def test_completed_dest_is_skipped(dl_server, tmp_path, hashed):
    payload = _payload(1 << 20)
    _Handler.payloads["/s"] = payload
    dest = tmp_path / "s.bin"
    dest.write_bytes(payload)
    dl = Download([Source(_url(dl_server, "/s"), dest, _sha(payload) if hashed else None)],
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
    # Loopback is a test-server affordance: an https origin may not land there.
    with pytest.raises(DownloadError):
        handler.redirect_request(req, None, 302, "Found", {}, "http://127.0.0.1:8000/b")
    local = urllib.request.Request("http://127.0.0.1:8000/a")
    assert handler.redirect_request(local, None, 302, "Found", {},
                                    "http://localhost:8000/b") is not None


# ── pause ─────────────────────────────────────────────────────



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
    assert r[str(dest)] == [(0, len(payload))]


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

    sidecars = list(partials.glob("*.ranges"))
    assert len(sidecars) == 1
    assert json.loads(sidecars[0].read_text())["ranges"] == [[0, 2 << 20]]
    assert da.read_bytes() == p_a and not db.exists()
    _Handler.abort_after = None
    dl = Download([Source(_url(dl_server, "/a"), da, _sha(p_a)),
                   Source(_url(dl_server, "/b"), db, _sha(p_b))],
                  partials_dir=partials, connections=1)
    ticks = []
    dl.run(progress=lambda d, t, r: ticks.append((d, t, dict(r))))
    assert ticks[0] == (len(p_a) + (2 << 20), len(p_a) + len(p_b),
                        {str(da): [(0, len(p_a))], str(db): [(0, 2 << 20)]})
    assert not list(partials.glob("*.ranges")) and not list(partials.glob("*.part"))
    assert da.read_bytes() == p_a
    assert db.read_bytes() == p_b
    a_reqs = [r for r in _Handler.ranges_seen if r[0] == "/a"]
    assert len(a_reqs) == 1  # exactly one request across both runs
    b_reqs = [r for r in _Handler.ranges_seen if r[0] == "/b"]
    # first request starts at 0; second (resume) starts at the abort point
    assert b_reqs[0][1] == 0
    assert b_reqs[1][1] == 2 << 20
    assert b_reqs[1][1] != b_reqs[0][1]



def test_pause_mid_plan_resumes_to_completion(dl_server, tmp_path):
    """Pause after source 1 completes: DownloadPaused is raised, source 1's
    dest is moved/complete, and source 2's partial survives in the managed
    area; a FRESH Download over the same plan resumes to full completion."""
    p_a, p_b = _payload(1 << 20, b"a"), _payload(32 << 20, b"b")
    _Handler.payloads["/a"] = p_a
    _Handler.payloads["/b"] = p_b
    da, db = tmp_path / "a.bin", tmp_path / "b.bin"
    partials = tmp_path / "partials"
    dl = Download([Source(_url(dl_server, "/a"), da, _sha(p_a)),
                   Source(_url(dl_server, "/b"), db, _sha(p_b))],
                  partials_dir=partials)

    def pause_second_source(done, total, ranges):
        if ranges.get(str(db)):
            dl.pause()

    with pytest.raises(DownloadPaused):
        dl.run(progress=pause_second_source)
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


# ── publication: a finished file the OS still holds ──────────


def _hold_first_two(replacements: list):
    """An ``os.replace`` that refuses the first two publications of a download's staged file with
    the permission error a Windows antivirus or indexing scan raises while it still has the
    finished file open. Sidecar writes publish through the same call, so the refusal is keyed to
    the downloader's staging suffix rather than to call order."""
    real_replace = os.replace

    def held(src, dst):
        if str(src).endswith(".download") and len(replacements) < 2:
            replacements.append(src)
            raise PermissionError(13, "Access is denied")
        real_replace(src, dst)

    return held


def test_replace_waits_out_a_transient_hold(tmp_path, monkeypatch):
    tmp = tmp_path / "model.download"
    dest = tmp_path / "model.gguf"
    tmp.write_bytes(b"weights")
    refusals = []
    monkeypatch.setattr(os, "replace", _hold_first_two(refusals))

    replace_when_released(tmp, dest, timeout=5)
    assert len(refusals) == 2
    assert dest.read_bytes() == b"weights"
    assert not tmp.exists()


def test_replace_gives_up_with_a_plain_language_error(tmp_path, monkeypatch):
    """A hold that outlasts the window is reported as a hold — chained to the OS error, and
    never degraded to a copy of the file."""
    tmp = tmp_path / "model.download"
    dest = tmp_path / "model.gguf"
    tmp.write_bytes(b"weights")

    def always_held(src, dst):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(os, "replace", always_held)
    with pytest.raises(RuntimeError) as failed:
        replace_when_released(tmp, dest, timeout=0.3)
    assert "model.download" in str(failed.value)
    assert "try again" in str(failed.value).lower()
    assert isinstance(failed.value.__cause__, PermissionError)
    assert not dest.exists()
    assert tmp.read_bytes() == b"weights"  # the finished bytes survive for the retry


def test_download_publishes_through_a_held_finished_file(dl_server, tmp_path, monkeypatch):
    """The publish the OS refuses twice must still land the complete file, and the download
    must report success — this is the 22 GB model case, not a partial transfer."""
    body = _payload(4 << 20, b"m")
    _Handler.payloads["/held"] = body
    dest = tmp_path / "held.gguf"
    refusals = []
    monkeypatch.setattr(os, "replace", _hold_first_two(refusals))

    dl = Download([Source(_url(dl_server, "/held"), dest, _sha(body))],
                  partials_dir=tmp_path / "partials")
    dl.run()

    assert len(refusals) == 2
    assert dest.read_bytes() == body
    assert not list(tmp_path.glob("*.download"))
    assert not [p for p in (tmp_path / "partials").iterdir() if p.suffix in (".part", ".ranges")]


def test_download_reports_a_hold_that_never_releases(dl_server, tmp_path, monkeypatch):
    """A stuck hold surfaces the rename error, not the failed cleanup of the partial it could
    not remove either."""
    body = _payload(1 << 20, b"h")
    _Handler.payloads["/stuck"] = body
    dest = tmp_path / "stuck.gguf"
    partials = tmp_path / "partials"

    def never_released(staged, target, **kw):
        raise RuntimeError(f"{staged.name} could not be renamed into place")

    real_unlink = Path.unlink

    def stuck_partial(self, missing_ok=False):
        if self.suffix in (".part", ".ranges"):
            raise PermissionError(13, "Access is denied")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr("pm.downloader.replace_when_released", never_released)
    monkeypatch.setattr(Path, "unlink", stuck_partial)

    dl = Download([Source(_url(dl_server, "/stuck"), dest, _sha(body))],
                  partials_dir=partials)
    with pytest.raises(RuntimeError, match="could not be renamed into place"):
        dl.run()
    assert not dest.exists()
