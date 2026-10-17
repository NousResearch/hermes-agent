"""Checkpoint rate and retry delays do not grow with connection speed."""
from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace

import pytest

from pm import downloader
from tests.pm._fixtures import threaded_server
from tests.pm._range_server import RangeHandler, dl_server as dl_server, url  # noqa: PLC0414 -- pytest fixture re-export


def test_fast_transfer_does_not_sync_each_byte_threshold(tmp_path, monkeypatch, dl_server):
    payload = b"w" * (32 << 20)
    RangeHandler.payloads = {"/fast": payload}
    download = downloader.Download(
        [downloader.Source(url(dl_server, "/fast"), tmp_path / "model", hashlib.sha256(payload).hexdigest())],
        partials_dir=tmp_path / "partials", connections=1,
    )
    checkpoints = []
    original = download._write_sidecar

    def save(*args):
        checkpoints.append(None)
        return original(*args)

    monkeypatch.setattr(download, "_write_sidecar", save)
    monkeypatch.setattr(downloader, "time", SimpleNamespace(monotonic=lambda: 0.0))
    download.run()
    assert (tmp_path / "model").read_bytes() == payload
    assert len(checkpoints) == 1  # The attempt still saves its final checkpoint.


def test_transfer_saves_progress_before_the_attempt_ends(tmp_path, monkeypatch, dl_server):
    payload = b"w" * (16 << 20)
    RangeHandler.payloads = {"/slow": payload}
    RangeHandler.chunk = 1 << 19
    RangeHandler.slow_per_chunk = 0.01
    monkeypatch.setattr(downloader, "_CHECKPOINT_SECONDS", 0.03)
    source = downloader.Source(url(dl_server, "/slow"), tmp_path / "model")
    root = tmp_path / "partials"
    side = root / (hashlib.sha256(source.url.encode()).hexdigest() + ".ranges")
    saved = []

    def observe(*_):
        if side.exists():
            saved.append(sum(end - start for start, end in json.loads(side.read_bytes())["ranges"]))

    downloader.Download([source], partials_dir=root, connections=1).run(progress=observe)
    assert any(0 < count < len(payload) for count in saved)
    assert source.dest.read_bytes() == payload


@pytest.mark.parametrize("failed_stage", ["probe", "single-stream"])
def test_non_resumable_requests_keep_the_existing_retry_after_cap(tmp_path, monkeypatch, failed_stage):
    payload = b"model bytes"
    failed = False

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal failed
            probe = self.headers.get("Range") == "bytes=0-0"
            if not failed and probe == (failed_stage == "probe"):
                failed = True
                self.send_response(503)
                self.send_header("Retry-After", "60")
                self.end_headers()
                return
            self.send_response(200)  # This source does not support ranges.
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    waits = []
    with threaded_server(Handler) as server:
        source = downloader.Source(f"http://127.0.0.1:{server.server_port}/model", tmp_path / "model")
        download = downloader.Download([source], partials_dir=tmp_path / "partials")
        monkeypatch.setattr(download, "_wait_retry", waits.append)
        download.run()
    assert waits == [30]
    assert source.dest.read_bytes() == payload
