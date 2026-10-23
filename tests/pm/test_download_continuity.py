"""Durable checkpoints preserve downloads across interruptions and process termination."""
from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler
from threading import Event
from types import SimpleNamespace

import pytest

from pm import downloader, filesystem
from pm.downloader import Download, Source
from tests.pm._fixtures import threaded_server
from tests.pm._range_server import RangeHandler, dl_server as dl_server, url  # noqa: PLC0414 -- pytest fixture re-export
from tests.waiting import wait_until


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
            "from pathlib import Path\nimport sys\nfrom pm import downloader\nfrom pm.downloader import Download, Source\n"
            "downloader._CHECKPOINT_SECONDS = 0.2\n"
            "Download([Source(sys.argv[1],Path(sys.argv[2]),sys.argv[4])],"
            "partials_dir=Path(sys.argv[3]),connections=1).run()\n"
        )
        child = subprocess.Popen([sys.executable, "-c", script, url, str(destination), str(partials),
                                  hashlib.sha256(payload).hexdigest()],
                                 cwd=Path(__file__).resolve().parents[2],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert waiting.wait(10), "child never received model data"
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


def test_checkpoints_are_time_based_not_byte_based(tmp_path, monkeypatch, dl_server):
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


def test_failed_periodic_checkpoint_retries_without_claiming_saved_bytes(tmp_path, monkeypatch, caplog):
    payload = b"checkpointed model bytes"
    part = tmp_path / "model.part"
    side = part.with_suffix(".ranges")
    part.write_bytes(payload)
    source = Source("https://fixture.invalid/model", tmp_path / "model")
    download = Download([source], partials_dir=tmp_path, connections=1)
    remote = downloader._Remote(total=len(payload), ranged=True, etag='"stable"')
    attempts = []
    replace = filesystem.os.replace

    def publish(temporary, destination):
        if Path(destination) == side:
            attempts.append(download._durable_done)
            if len(attempts) == 1:
                raise PermissionError(errno.EACCES, "checkpoint temporarily held")
        return replace(temporary, destination)

    def read_range(source, remote, part, start, end, stop, record):
        record(start, end)
        assert wait_until(lambda: side.is_file() or stop.is_set(), timeout=5)

    monkeypatch.setattr(filesystem.os, "replace", publish)
    monkeypatch.setattr(filesystem, "_HELD_RETRY_DELAYS", ())
    monkeypatch.setattr(downloader, "_CHECKPOINT_SECONDS", 0.01)
    monkeypatch.setattr(download, "_read_range", read_range)
    covered, errors = download._ranged_attempt(source, remote, part, [], 1, lambda _: None)
    assert not errors and covered == [(0, len(payload))]
    assert attempts == [0, 0]
    assert json.loads(side.read_text())["ranges"] == [[0, len(payload)]]
    assert download._durable_done == len(payload)
    assert any(record.name == "pm.downloader" and record.levelname == "WARNING" for record in caplog.records)


def test_final_checkpoint_write_failure_does_not_publish_the_model(tmp_path, monkeypatch, dl_server):
    payload = b"model bytes"
    RangeHandler.payloads = {"/strict": payload}
    source = Source(url(dl_server, "/strict"), tmp_path / "model", hashlib.sha256(payload).hexdigest())
    download = Download([source], partials_dir=tmp_path / "partials", connections=1)
    side = download.partials_dir / f"{download._key(source.url)}.ranges"
    replace = filesystem.os.replace

    def publish(temporary, destination):
        if Path(destination) == side:
            raise PermissionError(errno.EACCES, "checkpoint temporarily held")
        return replace(temporary, destination)

    monkeypatch.setattr(filesystem.os, "replace", publish)
    monkeypatch.setattr(filesystem, "_HELD_RETRY_DELAYS", ())
    monkeypatch.setattr(downloader, "time", SimpleNamespace(monotonic=lambda: 0.0))
    with pytest.raises(PermissionError):
        download.run()
    assert not source.dest.exists() and download._durable_done == 0
