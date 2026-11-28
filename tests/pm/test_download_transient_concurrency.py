"""Transient range failures keep healthy siblings running and step down concurrency."""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler
import json
import re
import threading
import time

import pytest

from pm import downloader
from pm.downloader import Download, DownloadError, Source
from tests.pm._fixtures import threaded_server


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_transient_worker_failure_lets_healthy_siblings_finish(tmp_path, monkeypatch):
    chunk = downloader._CHUNK
    slice_size = 3 * chunk
    payload = bytes(range(256)) * ((4 * slice_size) // 256)
    barrier = threading.Barrier(4)
    siblings_midstream = threading.Event()
    first_failed = threading.Event()
    requests: list[tuple[int, int]] = []
    req_lock = threading.Lock()
    failed_once = False

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal failed_once
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups())
            if (start, end) == (0, 0):
                self.send_response(206)
                self.send_header("Content-Range", f"bytes 0-0/{len(payload)}")
                self.send_header("Content-Length", "1")
                self.send_header("ETag", '"stable"')
                self.end_headers()
                self.wfile.write(payload[:1])
                return
            with req_lock:
                requests.append((start, end))
                initial_wave = len(requests) <= 4
            if initial_wave:
                barrier.wait(timeout=5)
                if start == 0 and not failed_once:
                    failed_once = True
                    # Wait until all 3 siblings are mid-transfer before failing.
                    assert siblings_midstream.wait(timeout=5)
                    self.send_error(503)
                    return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", '"stable"')
            self.end_headers()
            try:
                if initial_wave:
                    # Chunk 1 puts each sibling into read() for chunk 2; after worker 0
                    # fails, chunk 2 finishes and the loop re-checks stop.is_set() before chunk 3.
                    self.wfile.write(payload[start : start + chunk])
                    self.wfile.flush()
                    assert first_failed.wait(timeout=5)
                    time.sleep(0.05)
                    self.wfile.write(payload[start + chunk : start + 2 * chunk])
                    self.wfile.flush()
                    self.wfile.write(payload[start + 2 * chunk : end + 1])
                else:
                    self.wfile.write(payload[start : end + 1])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    dest = tmp_path / "archive.zip"
    waits: list[float] = []
    recorded_siblings: set[int] = set()
    rec_lock = threading.Lock()
    with threaded_server(Handler) as server:
        source = Source(f"http://127.0.0.1:{server.server_port}/archive.zip", dest, _sha(payload))
        dl = Download([source], partials_dir=tmp_path / "partials", connections=4)
        original_read_range = dl._read_range

        def observed_read_range(src, remote, part, start, end, stop, record):
            def wrapped_record(s, pos):
                record(s, pos)
                if s != 0 and pos - s >= chunk:
                    with rec_lock:
                        recorded_siblings.add(s)
                        if len(recorded_siblings) == 3:
                            siblings_midstream.set()

            try:
                return original_read_range(src, remote, part, start, end, stop, wrapped_record)
            except Exception:
                if start == 0:
                    first_failed.set()
                raise

        monkeypatch.setattr(dl, "_read_range", observed_read_range)
        monkeypatch.setattr(dl, "_wait_retry", waits.append)
        dl.run()

    assert dest.read_bytes() == payload
    assert len(waits) == 1
    # Wave 1 ran 4 ranges; the 3 healthy siblings must have finished their full
    # 2 MiB slices so wave 2 only refetches the single failed range.
    assert len(requests) == 5
    assert requests[4] == (0, slice_size - 1)


def test_transient_parallel_failures_step_down_connections_per_source(tmp_path, monkeypatch):
    chunk = downloader._CHUNK
    payload = bytes(range(256)) * ((8 * chunk) // 256)
    seen_connections: dict[str, list[int]] = {"/first": [], "/second": []}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups())
            if (start, end) == (0, 0):
                self.send_response(206)
                self.send_header("Content-Range", f"bytes 0-0/{len(payload)}")
                self.send_header("Content-Length", "1")
                self.send_header("ETag", '"stable"')
                self.end_headers()
                self.wfile.write(payload[:1])
                return
            with lock:
                current = dl.connections
                recorded = seen_connections[self.path]
                if not recorded or recorded[-1] != current:
                    recorded.append(current)
            if self.path == "/first" and current > 1:
                self.send_error(503)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", '"stable"')
            self.end_headers()
            try:
                self.wfile.write(payload[start : end + 1])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    d1, d2 = tmp_path / "first.bin", tmp_path / "second.bin"
    waits: list[float] = []
    with threaded_server(Handler) as server:
        digest = _sha(payload)
        dl = Download([
            Source(f"http://127.0.0.1:{server.server_port}/first", d1, digest),
            Source(f"http://127.0.0.1:{server.server_port}/second", d2, digest),
        ], partials_dir=tmp_path / "partials", connections=8)
        monkeypatch.setattr(dl, "_wait_retry", waits.append)
        dl.run()

    assert d1.read_bytes() == payload
    assert d2.read_bytes() == payload
    assert seen_connections["/first"] == [8, 4, 2, 1]
    assert seen_connections["/second"] == [8]
    assert len(waits) == 3
    assert dl.connections == 8


def test_queued_ranges_do_not_open_new_requests_after_transient_error(tmp_path, monkeypatch):
    payload = bytes(range(256)) * 32  # 8192 bytes
    barrier = threading.Barrier(2)
    first_failed = threading.Event()
    wave1_requests: list[tuple[int, int]] = []
    wave2_requests: list[tuple[int, int]] = []
    lock = threading.Lock()
    in_wave2 = False

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups())
            if (start, end) == (0, 0):
                self.send_response(206)
                self.send_header("Content-Range", f"bytes 0-0/{len(payload)}")
                self.send_header("Content-Length", "1")
                self.send_header("ETag", '"fragmented"')
                self.end_headers()
                self.wfile.write(payload[:1])
                return
            with lock:
                second = in_wave2
                (wave2_requests if second else wave1_requests).append((start, end))
            if not second:
                if start in (512, 1536):
                    barrier.wait(timeout=5)
                if start == 512:
                    self.send_error(503)
                    return
                assert first_failed.wait(timeout=5)
                time.sleep(0.05)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", '"fragmented"')
            self.end_headers()
            self.wfile.write(payload[start : end + 1])
            self.wfile.flush()

        def log_message(self, *args):
            pass

    partials = tmp_path / "partials"
    partials.mkdir()
    dest = tmp_path / "resumed.bin"
    digest = _sha(payload)
    with threaded_server(Handler) as server:
        target = f"http://127.0.0.1:{server.server_port}/resumed"
        key = hashlib.sha256(target.encode()).hexdigest()
        (partials / f"{key}.part").write_bytes(payload)
        (partials / f"{key}.ranges").write_text(json.dumps({
            "total": len(payload),
            "etag": '"fragmented"',
            "sha256": digest,
            "ranges": [[0, 512], [1024, 1536], [2048, 2560], [3072, 3584], [4096, len(payload)]],
        }), encoding="utf-8")
        dl = Download([Source(target, dest, digest)], partials_dir=partials, connections=2)
        original_read_range = dl._read_range

        def observed_read_range(src, remote, part, start, end, stop, record):
            try:
                return original_read_range(src, remote, part, start, end, stop, record)
            except Exception:
                if start == 512:
                    first_failed.set()
                raise

        def on_wait(_delay):
            nonlocal in_wave2
            with lock:
                in_wave2 = True

        monkeypatch.setattr(dl, "_read_range", observed_read_range)
        monkeypatch.setattr(dl, "_wait_retry", on_wait)
        dl.run()

    assert dest.read_bytes() == payload
    # Only the 2 already-running gaps opened requests in wave 1; the 2 queued
    # gaps stood down after the transient failure and finished in wave 2.
    assert sorted(wave1_requests) == [(512, 1023), (1536, 2047)]
    assert sorted(wave2_requests) == [(512, 1023), (2560, 3071), (3584, 4095)]


def test_fatal_worker_error_still_aborts_siblings_immediately(tmp_path, monkeypatch):
    chunk = downloader._CHUNK
    slice_size = 3 * chunk
    payload = bytes(range(256)) * ((2 * slice_size) // 256)
    barrier = threading.Barrier(2)
    sibling_midstream = threading.Event()
    fatal_failed = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups())
            if (start, end) == (0, 0):
                self.send_response(206)
                self.send_header("Content-Range", f"bytes 0-0/{len(payload)}")
                self.send_header("Content-Length", "1")
                self.send_header("ETag", '"stable"')
                self.end_headers()
                self.wfile.write(payload[:1])
                return
            barrier.wait(timeout=5)
            if start == 0:
                assert sibling_midstream.wait(timeout=5)
                self.send_error(401)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", '"stable"')
            self.end_headers()
            try:
                self.wfile.write(payload[start : start + chunk])
                self.wfile.flush()
                assert fatal_failed.wait(timeout=5)
                time.sleep(0.05)
                self.wfile.write(payload[start + chunk : start + 2 * chunk])
                self.wfile.flush()
                self.wfile.write(payload[start + 2 * chunk : end + 1])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    partials = tmp_path / "partials"
    dest = tmp_path / "denied.bin"
    with threaded_server(Handler) as server:
        target = f"http://127.0.0.1:{server.server_port}/denied"
        dl = Download([Source(target, dest, _sha(payload))], partials_dir=partials, connections=2)
        original_read_range = dl._read_range

        def observed_read_range(src, remote, part, start, end, stop, record):
            def wrapped_record(s, pos):
                record(s, pos)
                if s != 0 and pos - s >= chunk:
                    sibling_midstream.set()

            try:
                return original_read_range(src, remote, part, start, end, stop, wrapped_record)
            except Exception:
                if start == 0:
                    fatal_failed.set()
                raise

        monkeypatch.setattr(dl, "_read_range", observed_read_range)
        with pytest.raises(DownloadError):
            dl.run()

    assert not dest.exists()
    key = hashlib.sha256(target.encode()).hexdigest()
    saved = json.loads((partials / f"{key}.ranges").read_text(encoding="utf-8-sig"))
    # Fatal 401 must have triggered stop.set(), aborting worker 1 before chunk 3.
    assert saved["ranges"] == [[slice_size, slice_size + 2 * chunk]]
