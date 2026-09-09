"""The real downloader bounds retries and pauses without publishing partial bytes."""

import hashlib
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest

from pm import network
from pm.downloader import Download, DownloadError, DownloadPaused, HashError, Source
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


@pytest.mark.parametrize("failure", ["503", "404", "403", "hash", "disk"])
@pytest.mark.parametrize("phase", ["probe", "ranged", "single"])
def test_download_failure_never_publishes_partial_bytes(tmp_path, dl_server, monkeypatch, failure, phase):
    payload = b"verified bytes"
    RangeHandler.payloads["/tool"] = payload
    RangeHandler.no_range = phase == "single"
    original_get = RangeHandler.do_GET
    requests = []
    waits = []

    def respond(handler):
        request_range = handler.headers.get("Range")
        requests.append(request_range)
        is_probe = request_range == "bytes=0-0"
        if failure.isdigit() and is_probe == (phase == "probe"):
            handler.send_error(int(failure))
            return
        original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    dest = tmp_path / "tool"
    digest = "0" * 64 if failure == "hash" else hashlib.sha256(payload).hexdigest()
    dl = Download([Source(url(dl_server, "/tool"), dest, digest)],
                  partials_dir=tmp_path / "partials", connections=1)
    monkeypatch.setattr(dl, "_wait_retry", waits.append)
    if failure == "disk":
        dest.mkdir()
        (dest / "occupied").write_bytes(b"keep")
    expected_error = {"hash": HashError, "disk": OSError}.get(failure)
    if expected_error is None:
        expected_error = DownloadError if failure == "403" and phase == "probe" else urllib.error.HTTPError

    with pytest.raises(expected_error):
        dl.run()
    if failure == "disk":
        assert (dest / "occupied").read_bytes() == b"keep"
        assert not requests and not waits
    else:
        assert not dest.exists()
        error_requests = requests if phase == "probe" else [value for value in requests if value != "bytes=0-0"]
        if failure == "503":
            assert len(error_requests) == network._ATTEMPTS
            assert len(waits) == network._ATTEMPTS - 1
        else:
            assert not waits
            if failure != "hash":
                assert len(error_requests) == 1


@pytest.mark.parametrize("phase", ["probe", "ranged", "single"])
def test_pause_interrupts_real_http_retry_backoff(tmp_path, dl_server, monkeypatch, phase):
    payload = b"keep these bytes"
    RangeHandler.payloads["/tool"] = payload
    RangeHandler.no_range = phase == "single"
    original_get = RangeHandler.do_GET
    requests = []
    waiting = threading.Event()
    waits = []

    def respond(handler):
        request_range = handler.headers.get("Range")
        requests.append(request_range)
        if (request_range == "bytes=0-0") == (phase == "probe"):
            handler.send_response(503)
            handler.send_header("Retry-After", "30")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    dest = tmp_path / "tool"
    dl = Download([Source(url(dl_server, "/tool"), dest)],
                  partials_dir=tmp_path / "partials")
    original_wait = dl._wait_retry

    def wait(delay):
        waits.append(delay)
        waiting.set()
        original_wait(delay)

    monkeypatch.setattr(dl, "_wait_retry", wait)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(dl.run)
        try:
            assert waiting.wait(10), "the failed request never reached backoff"
            dl.pause()
            with pytest.raises(DownloadPaused):
                task.result(timeout=5)
        finally:
            dl.pause()
    assert len([value for value in requests if (value == "bytes=0-0") == (phase == "probe")]) == 1
    assert waits == [network._MAX_DELAY]
    assert not dest.exists()
