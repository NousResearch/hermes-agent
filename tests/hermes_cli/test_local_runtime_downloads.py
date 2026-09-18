"""Runtime downloads must not turn a transient transfer failure into a poisoned cache."""

import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError

import pytest

from hermes_cli.local_runtime import binaries


@pytest.fixture
def asset_server():
    responses = []
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            body, length, status = responses.pop(0)
            self.send_response(status)
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/{{asset}}", responses, requests
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("failure", ["short", "callback"])
@pytest.mark.parametrize("known_length", [True, False])
def test_download_only_publishes_complete_transfers(tmp_path, asset_server, failure, known_length):
    url, responses, requests = asset_server
    dest = tmp_path / "runtime.zip"
    payload = b"runtime archive bytes"
    responses.append((payload[:-1], len(payload), 200))

    def progress(done, total):
        if failure == "callback":
            raise RuntimeError("cancelled")

    error = RuntimeError if failure == "callback" else binaries.BinaryResolutionError
    with pytest.raises(error):
        binaries._download(url.format(asset=dest.name), dest, progress=progress)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))

    # A retry works, including servers that omit Content-Length.
    responses.append((payload, len(payload) if known_length else None, 200))
    ticks = []
    binaries._download(url.format(asset=dest.name), dest,
                       progress=lambda done, total: ticks.append((done, total)))
    assert dest.read_bytes() == payload
    assert ticks[-1] == (len(payload), len(payload) if known_length else 0)
    assert len(requests) == 2


def test_failed_request_preserves_another_active_download(tmp_path, asset_server):
    url, responses, requests = asset_server
    dest = tmp_path / "runtime.zip"
    payload = b"x" * (2 << 20)
    responses.extend([(payload, len(payload), 200), (b"", 0, 503)])
    started = threading.Event()
    resume = threading.Event()

    def pause_download(done, total):
        started.set()
        assert resume.wait(10), "active download was not resumed"

    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(binaries._download, url.format(asset=dest.name), dest,
                             progress=pause_download)
        try:
            assert started.wait(10), "active download did not write its first chunk"
            with pytest.raises(HTTPError) as failed:
                binaries._download(url.format(asset=dest.name), dest)
            assert failed.value.code == 503
        finally:
            resume.set()
        active.result(timeout=10)

    assert dest.read_bytes() == payload
    assert not list(tmp_path.glob("*.part"))
    assert len(requests) == 2
