"""Interrupted bodies retain durable progress across attempts and processes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler
from threading import Event

from pm.downloader import Download, Source
from tests.pm._fixtures import threaded_server
from tests.pm._range_server import RangeHandler, dl_server as dl_server, url  # noqa: PLC0414 -- pytest fixture re-export


def test_useful_progress_survives_more_than_four_interrupted_bodies(tmp_path, monkeypatch):
    block = 1 << 20
    payload = bytes(range(256)) * (10 * block // 256)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups())
            probe = (start, end) == (0, 0)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", '"opaque-stable-representation"')
            self.end_headers()
            if not probe:
                requests.append((start, end))
            stop = min(end + 1, start + block) if not probe and len(requests) <= 6 else end + 1
            self.wfile.write(payload[start:stop])
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *args):
            pass

    destination = tmp_path / "model.gguf"
    with threaded_server(Handler) as server:
        source = Source(f"http://127.0.0.1:{server.server_port}/model", destination,
                        hashlib.sha256(payload).hexdigest())
        download = Download([source], connections=1, partials_dir=tmp_path / "partials")
        monkeypatch.setattr(download, "_wait_retry", lambda _: None)
        download.run()
    assert destination.read_bytes() == payload
    assert len(requests) == 7
    assert [start for start, _ in requests] == [index * block for index in range(7)]


def test_forced_termination_recovers_checkpoint_before_request_finishes(tmp_path):
    payload = b"durable model bytes" * (1 << 20)
    waiting = Event()
    release = Event()
    requests = []
    checkpoint = 8 << 20

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups())
            probe = (start, end) == (0, 0)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", '"same-file-after-restart"')
            self.end_headers()
            if not probe:
                requests.append((start, end))
            try:
                if not probe and len(requests) == 1:
                    self.wfile.write(payload[start:start + checkpoint])
                    self.wfile.flush()
                    waiting.set()
                    release.wait(15)
                    return
                self.wfile.write(payload[start:end + 1])
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    partials = tmp_path / "partials"
    destination = tmp_path / "model.gguf"
    with threaded_server(Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/model"
        side = partials / (hashlib.sha256(url.encode()).hexdigest() + ".ranges")
        script = (
            "from pathlib import Path\nimport sys\nfrom pm.downloader import Download, Source\n"
            "Download([Source(sys.argv[1],Path(sys.argv[2]),sys.argv[4])],"
            "partials_dir=Path(sys.argv[3]),connections=1).run()\n"
        )
        child = subprocess.Popen([sys.executable, "-c", script, url, str(destination), str(partials),
                                  hashlib.sha256(payload).hexdigest()],
                                 cwd=Path(__file__).resolve().parents[2],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert waiting.wait(10), "child never received model data"
            from tests.waiting import wait_until

            assert wait_until(side.is_file, timeout=10), "no checkpoint during an unfinished request"
            saved = json.loads(side.read_text())
            retained = sum(end - start for start, end in saved["ranges"])
            assert retained >= checkpoint
        finally:
            child.kill()
            child.communicate(timeout=5)
            release.set()
        assert not destination.exists()
        Download([Source(url, destination, hashlib.sha256(payload).hexdigest())],
                 partials_dir=partials, connections=1).run()
    assert destination.read_bytes() == payload
    assert requests[1][0] >= checkpoint


def test_noncontiguous_durable_ranges_are_not_requested_again(tmp_path, dl_server):
    block = 1 << 20
    payload = bytes(range(256)) * (6 * block // 256)
    RangeHandler.payloads["/model"] = payload
    endpoint = url(dl_server, "/model")
    partials = tmp_path / "partials"
    partials.mkdir()
    key = hashlib.sha256(endpoint.encode()).hexdigest()
    coverage = [(0, block), (2 * block, 3 * block), (5 * block, 6 * block)]
    part = partials / f"{key}.part"
    with part.open("wb") as stream:
        stream.truncate(len(payload))
        for start, end in coverage:
            stream.seek(start)
            stream.write(payload[start:end])
    digest = hashlib.sha256(payload).hexdigest()
    (partials / f"{key}.ranges").write_text(json.dumps({
        "total": len(payload), "etag": f'"{digest}"', "sha256": digest, "ranges": coverage,
    }))
    destination = tmp_path / "model.gguf"
    Download([Source(endpoint, destination, digest)], partials_dir=partials, connections=2).run()
    assert destination.read_bytes() == payload
    assert sorted((start, end) for _, start, end in RangeHandler.ranges_seen) == [
        (block, 2 * block - 1), (3 * block, 5 * block - 1),
    ]
