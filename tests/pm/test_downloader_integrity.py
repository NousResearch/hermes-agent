"""Every ranged worker must prove which representation and bytes it received."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import re
from threading import Thread

import pytest

from pm.downloader import Download, DownloadError, Source
from tests.pm._range_server import dl_server  # noqa: F401 — fixture


@pytest.mark.parametrize("reply", ["whole-body", "wrong-bounds", "changed-size", "changed-etag", "short-body"])
def test_invalid_worker_response_cannot_publish(reply, tmp_path):
    payload = bytes(range(256)) * (12 * 1024)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            bounds = self.headers.get("Range")
            if bounds == "bytes=0-0":
                status, content_range, body = 206, f"bytes 0-0/{len(payload)}", payload[:1]
            else:
                start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", bounds).groups())
                status, total, reported_start = 206, len(payload), start
                body = payload[start:end + 1]
                if reply == "whole-body":
                    status, body = 200, payload
                elif reply == "wrong-bounds":
                    reported_start += 1
                elif reply == "changed-size":
                    total += 1
                elif reply == "short-body":
                    body = body[:-1]
                content_range = f"bytes {reported_start}-{end}/{total}"
            self.send_response(status)
            self.send_header("Content-Range", content_range)
            self.send_header("Content-Length", str(len(body)))
            etag = '"new"' if reply == "changed-etag" and bounds != "bytes=0-0" else '"original"'
            self.send_header("ETag", etag)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The client correctly refuses headers before reading this body.

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "model.gguf"
    try:
        source = Source(f"http://127.0.0.1:{server.server_port}/model", destination)
        with pytest.raises(DownloadError):
            Download([source], partials_dir=tmp_path / "partials", connections=1).run()
        assert not destination.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("previous", [None, b"previous complete model"])
def test_failed_cross_volume_publication_keeps_destination_atomic(tmp_path, monkeypatch, previous):
    import errno
    import os
    from pathlib import Path
    import shutil

    part = tmp_path / "partials" / "model.part"
    part.parent.mkdir()
    payload = b"new complete model bytes"
    part.write_bytes(payload)
    side = part.with_suffix(".ranges")
    side.write_text("[]", encoding="utf-8")
    destination = tmp_path / "other-volume" / "model.gguf"
    destination.parent.mkdir()
    if previous is not None:
        destination.write_bytes(previous)
    source = Source("https://example.invalid/model", destination)
    download = Download([source], partials_dir=part.parent)

    original_rename = os.rename

    def cross_device(src, dst, *args, **kwargs):
        if Path(src) == part:
            raise OSError(errno.EXDEV, "cross-volume fixture")
        return original_rename(src, dst, *args, **kwargs)

    def interrupted_copy(src, dst, *args, **kwargs):
        Path(dst).write_bytes(Path(src).read_bytes()[:4])
        raise OSError(errno.ENOSPC, "destination full")

    def interrupted_stream(src, dst, *args, **kwargs):
        dst.write(src.read(4))
        raise OSError(errno.ENOSPC, "destination full")

    monkeypatch.setattr(os, "rename", cross_device)
    monkeypatch.setattr(shutil.move, "__defaults__", (interrupted_copy,))
    monkeypatch.setattr(shutil, "copyfileobj", interrupted_stream)
    with pytest.raises(OSError, match="destination full"):
        download._finalize(source, part, side)

    assert (destination.read_bytes() if destination.exists() else None) == previous
    assert part.read_bytes() == payload
    assert side.exists()
    assert set(destination.parent.iterdir()) == ({destination} if previous is not None else set())


def test_fragmented_resume_obeys_connection_limit(tmp_path):
    import hashlib
    import json
    from threading import Event, Lock
    import time

    payload = bytes(range(256)) * 16
    entered = Event()
    release = Event()
    lock = Lock()
    active = 0
    maximum = 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal active, maximum
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups())
            is_probe = (start, end) == (0, 0)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", '"same-object"')
            self.end_headers()
            if not is_probe:
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                    entered.set()
                release.wait(timeout=8)
            try:
                self.wfile.write(payload[start:end + 1])
            finally:
                if not is_probe:
                    with lock:
                        active -= 1

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = Thread(target=server.serve_forever, daemon=True)
    serving.start()
    url = f"http://127.0.0.1:{server.server_port}/fragmented"
    partials = tmp_path / "partials"
    partials.mkdir()
    key = hashlib.sha256(url.encode()).hexdigest()
    (partials / f"{key}.part").write_bytes(payload)
    (partials / f"{key}.ranges").write_text(json.dumps({
        "total": len(payload), "etag": '"same-object"',
        "sha256": hashlib.sha256(payload).hexdigest(),
        "ranges": [[0, 512], [1024, 1536], [2048, 2560]],
    }), encoding="utf-8")
    source = Source(url, tmp_path / "result", hashlib.sha256(payload).hexdigest())
    errors = []

    def run():
        try:
            Download([source], partials_dir=partials, connections=1).run()
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=run)
    worker.start()
    try:
        assert entered.wait(timeout=8)
        # Existing scheduler launches every gap. Leave the response pending
        # while its sibling sockets enter the real server.
        time.sleep(2)
    finally:
        release.set()
        worker.join(timeout=15)
        server.shutdown()
        server.server_close()
        serving.join(timeout=5)
    assert not worker.is_alive() and not errors, errors
    assert source.dest.read_bytes() == payload
    assert maximum == 1


def test_processes_share_partial_ownership_without_losing_destinations(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from threading import Event, Lock

    payload = bytes(range(256)) * (4096 * 2)
    entered = Event()
    both = Event()
    release = Event()
    lock = Lock()
    active = 0
    maximum = 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal active, maximum
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups())
            probe = (start, end) == (0, 0)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", '"unchanged"')
            self.end_headers()
            if not probe:
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                    entered.set()
                    if active > 1:
                        both.set()
                release.wait(timeout=15)
            try:
                self.wfile.write(payload[start:end + 1])
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if not probe:
                    with lock:
                        active -= 1

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = Thread(target=server.serve_forever, daemon=True)
    serving.start()
    script = (
        "from pathlib import Path\nimport sys\nfrom pm.downloader import Download, Source\n"
        "url,destination,partials = sys.argv[1:]\n"
        "Download([Source(url,Path(destination))],partials_dir=Path(partials),connections=1).run()\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    children = []
    destinations = [tmp_path / "first" / "model", tmp_path / "second" / "model"]
    try:
        for destination in destinations:
            children.append(subprocess.Popen(
                [sys.executable, "-c", script, f"http://127.0.0.1:{server.server_port}/model",
                 str(destination), str(tmp_path / "partials")],
                cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8",
            ))
            assert entered.wait(timeout=10)
        both.wait(timeout=2)
    finally:
        release.set()
        results = []
        for child in children:
            try:
                output, error = child.communicate(timeout=25)
            except subprocess.TimeoutExpired:
                child.kill()
                output, error = child.communicate(timeout=5)
            results.append((child.returncode, output, error))
        server.shutdown()
        server.server_close()
        serving.join(timeout=5)
    assert all(code == 0 for code, _, _ in results), results
    assert [destination.read_bytes() for destination in destinations] == [payload, payload]
    assert maximum == 1, "two processes wrote one partial concurrently"


@pytest.mark.parametrize("damage", ["changed-etag", "missing-part", "invalid-bounds", "no-validator"])
def test_resume_never_trusts_unbound_coverage(tmp_path, dl_server, damage):
    import hashlib
    import json
    from tests.pm._range_server import RangeHandler, url

    payload = b"new remote data" * 1000
    RangeHandler.payloads["/fresh"] = payload
    source_url = url(dl_server, "/fresh")
    key = hashlib.sha256(source_url.encode()).hexdigest()
    partials = tmp_path / "partials"
    partials.mkdir()
    part = partials / f"{key}.part"
    if damage != "missing-part":
        part.write_bytes(b"x" * len(payload))
    etag = '"' + hashlib.sha256(payload).hexdigest() + '"'
    if damage == "changed-etag":
        etag = '"previous-object"'
    if damage == "no-validator":
        etag = ""
        RangeHandler.etags = False
    ranges = [[0, len(payload)]] if damage != "invalid-bounds" else [[-1, len(payload)]]
    (partials / f"{key}.ranges").write_text(json.dumps({
        "total": len(payload), "etag": etag, "sha256": "", "ranges": ranges,
    }), encoding="utf-8")

    destination = tmp_path / "model"
    Download([Source(source_url, destination)], partials_dir=partials).run()

    assert destination.read_bytes() == payload
