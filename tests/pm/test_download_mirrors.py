"""Pinned downloads keep their identity when the upstream disappears."""
import hashlib

import pytest

from pm.downloader import Download, DownloadError, HashError, Source
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


@pytest.mark.parametrize("phase", ["probe", "body"])
def test_fallback_retains_hash_destination_and_progress(tmp_path, dl_server, monkeypatch, phase):
    payload = b"the same pinned bytes" * 100
    RangeHandler.payloads = {"/primary": payload, "/mirror": payload, "/other": b"other"}
    requests = []
    original = RangeHandler.do_GET

    def respond(handler):
        requests.append(handler.path)
        if handler.path == "/primary" and (phase == "probe" or handler.headers.get("Range") != "bytes=0-0"):
            handler.send_error(404)
        else:
            original(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    digest = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "original-name.zip"
    sources = [Source(url(dl_server, "/primary"), destination, digest,
                      fallbacks=(url(dl_server, "/mirror"),)),
               Source(url(dl_server, "/other"), tmp_path / "other")]
    ticks = []
    result = Download(sources, partials_dir=tmp_path / "partials", connections=1).run(
        progress=lambda done, total, ranges: ticks.append((done, total, ranges)))
    assert result == [source.dest for source in sources]
    assert destination.read_bytes() == payload
    expected = len(payload) + len(b"other")
    assert ticks[-1][:2] == (expected, expected)
    assert set(ticks[-1][2]) == {str(source.dest) for source in sources}
    assert requests.index("/primary") < requests.index("/mirror")


def test_bad_hash_is_not_a_mirror_miss(tmp_path, dl_server):
    RangeHandler.payloads = {"/primary": b"changed upstream", "/mirror": b"pinned"}
    destination = tmp_path / "keep.zip"
    destination.write_bytes(b"previous destination")
    source = Source(url(dl_server, "/primary"), destination, hashlib.sha256(b"pinned").hexdigest(),
                    fallbacks=(url(dl_server, "/mirror"),))
    with pytest.raises(HashError):
        Download([source], partials_dir=tmp_path / "partials").run()
    assert destination.read_bytes() == b"previous destination"
    assert all(path != "/mirror" for path, *_ in RangeHandler.ranges_seen)


@pytest.mark.parametrize("mirror_status", [401, 403, 404])
def test_exhausted_mirrors_name_each_attempted_url(tmp_path, dl_server, monkeypatch, mirror_status):
    original = RangeHandler.do_GET

    def respond(handler):
        if handler.path == "/also-missing":
            handler.send_error(mirror_status)
        else:
            original(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    primary, mirror = (url(dl_server, path) for path in ("/missing", "/also-missing"))
    with pytest.raises(DownloadError) as failure:
        Download([Source(primary, tmp_path / "absent", "0" * 64, fallbacks=(mirror,))],
                 partials_dir=tmp_path / "partials").run()
    assert primary in str(failure.value) and mirror in str(failure.value)
    assert "404" in str(failure.value) and str(mirror_status) in str(failure.value)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("failure", ["503", "tls", "disk", "pause"])
def test_only_availability_failures_can_use_a_mirror(tmp_path, dl_server, monkeypatch, failure):
    import ssl
    from pm import downloader, network

    payload = b"still pinned"
    RangeHandler.payloads = {"/primary": payload, "/mirror": payload}
    original = downloader._OPENER.open
    requests = []
    waits = []
    dl = Download([Source(url(dl_server, "/primary"), tmp_path / "tool",
                          hashlib.sha256(payload).hexdigest(), fallbacks=(url(dl_server, "/mirror"),))],
                  partials_dir=tmp_path / "partials")
    monkeypatch.setattr(dl, "_wait_retry", waits.append)

    def open_request(request, **kwargs):
        requests.append(request.full_url)
        if request.full_url.endswith("/primary"):
            if failure == "tls":
                raise ssl.SSLCertVerificationError("untrusted certificate")
            if failure == "disk":
                raise PermissionError("local path denied")
            if failure == "pause":
                dl.pause()
            if failure == "503":
                import urllib.error
                raise urllib.error.HTTPError(request.full_url, 503, "Unavailable", {}, None)
        return original(request, **kwargs)

    monkeypatch.setattr(downloader._OPENER, "open", open_request)
    if failure == "503":
        dl.run()
        assert (tmp_path / "tool").read_bytes() == payload
        assert requests.count(url(dl_server, "/primary")) == network._ATTEMPTS
        assert len(waits) == network._ATTEMPTS - 1
    else:
        with pytest.raises(downloader.DownloadError):
            dl.run()
        assert not (tmp_path / "tool").exists()
        assert not waits
        assert url(dl_server, "/mirror") not in requests
