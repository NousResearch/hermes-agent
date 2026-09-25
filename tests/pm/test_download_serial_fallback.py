"""A CDN refusal downgrades this transfer once, including later network retries."""

import hashlib

import pytest

from pm.downloader import Download, Source
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


@pytest.mark.parametrize("refused_status", [403, 404])
def test_serial_fallback_survives_a_transient_retry(tmp_path, dl_server, monkeypatch, refused_status):
    payload = b"verified bytes"
    RangeHandler.payloads["/tool"] = payload
    original_get = RangeHandler.do_GET
    dest = tmp_path / "tool"
    dl = Download([Source(url(dl_server, "/tool"), dest, hashlib.sha256(payload).hexdigest())],
                  partials_dir=tmp_path / "partials")
    modes = []
    waits = []

    def respond(handler):
        if handler.headers.get("Range") != "bytes=0-0":
            modes.append(dl.connections)
            if len(modes) < 3:
                handler.send_error(refused_status if len(modes) == 1 else 503)
                return
        original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    monkeypatch.setattr(dl, "_wait_retry", waits.append)
    dl.run()
    assert dest.read_bytes() == payload
    assert modes[0] > 1
    assert modes[1:] == [1, 1], "a network retry must not undo the CDN downgrade"
    assert len(waits) == 1
    assert dl.connections == modes[0], "the downgrade belongs to this source, not later downloads"
