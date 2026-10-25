"""Active downloads respect source cooldowns and revisit a recovered mirror."""
import hashlib
import time
import urllib.error
from types import SimpleNamespace

from pm import downloader
from pm.downloader import Download, Source
from tests.pm._range_server import RangeHandler, dl_server as dl_server, url  # noqa: PLC0414 -- pytest fixture re-export


def test_recovery_revisits_sources_without_shortening_retry_after(tmp_path, monkeypatch, dl_server):
    payload = b"pinned model bytes"
    RangeHandler.payloads = {"/primary": payload, "/mirror": payload}
    original_get = RangeHandler.do_GET
    clock = [0.0]
    probes = {"/primary": [], "/mirror": []}
    failed_responses = []
    original_open = downloader._OPENER.open

    def capture_failure(*args, **kwargs):
        try:
            return original_open(*args, **kwargs)
        except urllib.error.HTTPError as exc:
            failed_responses.append(exc)
            raise

    monkeypatch.setattr(downloader._OPENER, "open", capture_failure)

    def respond(handler):
        if handler.headers.get("Range") == "bytes=0-0":
            attempts = probes[handler.path]
            attempts.append(clock[0])
            retry_after = None
            if len(attempts) == 1:
                retry_after = 60 if handler.path == "/primary" else 1
            elif handler.path == "/mirror" and len(attempts) == 3:
                retry_after = 290
            if retry_after is not None:
                handler.send_response(503)
                handler.send_header("Retry-After", str(retry_after))
                handler.send_header("Content-Length", "0")
                handler.end_headers()
                return
        original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    monkeypatch.setattr(downloader, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=time.sleep))
    destination = tmp_path / "model"
    source = Source(url(dl_server, "/primary"), destination, hashlib.sha256(payload).hexdigest(),
                    fallbacks=(url(dl_server, "/mirror"),))
    download = Download([source], connections=1, partials_dir=tmp_path / "partials", keep_retrying=True)
    monkeypatch.setattr(download, "_wait_retry", lambda delay: clock.__setitem__(0, clock[0] + delay))
    download.run()
    assert destination.read_bytes() == payload
    assert probes == {"/primary": [0.0, 60.0], "/mirror": [0.0, 1.0, 1.0]}
    assert failed_responses and all(response.closed for response in failed_responses)
