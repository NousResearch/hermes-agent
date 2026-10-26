"""The curl fallback publishes bytes when urllib cannot even connect.

The failure this guards against is a host that resets Python's OpenSSL handshake
before any byte moves: urllib dies identically in the probe and in every range
worker, so retrying inside urllib cannot help. These tests force that condition by
making the downloader's opener refuse outright, then assert the download still
completes, still verifies its hash, and still publishes atomically.
"""

import hashlib
import urllib.error
from pathlib import Path

import pytest

from pm import curl_transport, downloader
from pm.downloader import Download, DownloadTransportError, Source
from tests.pm._range_server import RangeHandler, dl_server, url


class _RefusingOpener:
    """Stands in for the module opener and dies the way a reset handshake does."""

    def __init__(self):
        self.calls = 0

    def open(self, request, timeout=None, **kwargs):
        self.calls += 1
        raise urllib.error.URLError(
            ConnectionResetError("An existing connection was forcibly closed by the remote host"))


@pytest.fixture
def dead_urllib(monkeypatch):
    """Make every urllib request fail at the transport, as a reset handshake does."""
    opener = _RefusingOpener()
    monkeypatch.setattr(downloader, "_OPENER", opener)
    return opener


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Keep the urllib retry budget from adding seconds to every test."""
    monkeypatch.setattr(Download, "_wait_retry", lambda self, delay: None)


def test_probe_falls_back_to_curl(tmp_path, dl_server, dead_urllib):
    payload = b"probe bytes" * 64
    RangeHandler.payloads["/tool"] = payload

    remote = Download([], partials_dir=tmp_path)._probe(url(dl_server, "/tool"))

    assert dead_urllib.calls, "urllib was never tried; the fallback is not a fallback"
    assert remote.total == len(payload)
    assert remote.ranged is True
    assert remote.etag, "the fallback lost the validator the ranged path needs"


def test_probe_reports_a_server_that_ignores_range(tmp_path, dl_server, dead_urllib):
    payload = b"x" * (2 << 20)  # over the probe cap, so curl aborts the full body
    RangeHandler.payloads["/tool"] = payload
    RangeHandler.no_range = True

    remote = Download([], partials_dir=tmp_path)._probe(url(dl_server, "/tool"))

    assert remote.total == len(payload)
    assert remote.ranged is False


def test_download_completes_via_curl_and_verifies_hash(tmp_path, dl_server, dead_urllib):
    payload = bytes(range(256)) * 4096
    RangeHandler.payloads["/tool"] = payload
    dest = tmp_path / "tool"
    digest = hashlib.sha256(payload).hexdigest()

    Download([Source(url(dl_server, "/tool"), dest, digest)],
             partials_dir=tmp_path / "partials", connections=1).run()

    assert dest.read_bytes() == payload
    # The fallback must not leak its working files into the partials area.
    assert list((tmp_path / "partials").iterdir()) == [
        tmp_path / "partials" / ".locks"] or not any(
        (tmp_path / "partials").glob("*.part"))


def test_download_via_curl_reports_progress(tmp_path, dl_server, dead_urllib):
    payload = b"progress" * 1024
    RangeHandler.payloads["/tool"] = payload
    dest = tmp_path / "tool"
    seen = []

    Download([Source(url(dl_server, "/tool"), dest)],
             partials_dir=tmp_path / "partials", connections=1).run(
        progress=lambda done, total, ranges: seen.append((done, total)))

    assert dest.read_bytes() == payload
    assert any(done == len(payload) for done, _ in seen), seen


@pytest.mark.parametrize("status", [404, 403])
def test_curl_fallback_is_skipped_for_a_definitive_http_status(
        tmp_path, dl_server, monkeypatch, status):
    """A 404 is a missing file, not a broken transport: do not re-fetch it.

    Uses the real opener — a dead opener refuses before any request, so it cannot
    exercise the status code path. The fallback must not fire at all; asserted by
    spying on the curl probe, so a test that merely tolerated an unused fallback
    could not pass.
    """
    payload = b"present"
    RangeHandler.payloads["/tool"] = payload
    original_get = RangeHandler.do_GET
    curl_calls = []

    def respond(handler):
        if handler.headers.get("Range") == "bytes=0-0":
            handler.send_error(status)
            return
        original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    monkeypatch.setattr(curl_transport, "probe", lambda url, **kwargs: curl_calls.append(url))

    with pytest.raises(DownloadTransportError) as error:
        Download([], partials_dir=tmp_path)._probe(url(dl_server, "/tool"))

    assert error.value.status == status
    assert curl_calls == []


def test_curl_fallback_is_skipped_for_a_404_during_transfer(tmp_path, dl_server, monkeypatch):
    """The same rule after a successful probe: a 404 on the bytes is not a dead link."""
    payload = b"present"
    RangeHandler.payloads["/tool"] = payload
    original_get = RangeHandler.do_GET
    curl_calls = []

    def respond(handler):
        rng = handler.headers.get("Range")
        if rng and rng != "bytes=0-0":
            handler.send_error(404)
            return
        original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    monkeypatch.setattr(curl_transport, "probe", lambda url, **kwargs: curl_calls.append(url))
    dest = tmp_path / "tool"

    with pytest.raises(DownloadTransportError):
        Download([Source(url(dl_server, "/tool"), dest)],
                 partials_dir=tmp_path / "partials", connections=1).run()

    assert curl_calls == []
    assert not dest.exists()


def test_curl_fallback_disabled_by_config(tmp_path, dl_server, dead_urllib, monkeypatch):
    """`download.allow_curl_fallback: false` in config.yaml turns it off."""
    from hermes_cli import config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly",
                        lambda: {"download": {"allow_curl_fallback": False}})
    RangeHandler.payloads["/tool"] = b"present"

    with pytest.raises(DownloadTransportError):
        Download([], partials_dir=tmp_path)._probe(url(dl_server, "/tool"))


def test_curl_fallback_skipped_when_curl_is_absent(tmp_path, dl_server, dead_urllib, monkeypatch):
    monkeypatch.setattr(curl_transport, "executable", lambda: None)
    RangeHandler.payloads["/tool"] = b"present"

    with pytest.raises(DownloadTransportError):
        Download([], partials_dir=tmp_path)._probe(url(dl_server, "/tool"))


def test_curl_probe_returns_none_when_the_server_will_not_answer(tmp_path, monkeypatch):
    class _Failing:
        def __init__(self):
            self.calls = 0

        def open(self, request, timeout=None, **kwargs):
            self.calls += 1
            raise urllib.error.URLError(ConnectionResetError("reset"))

    monkeypatch.setattr(downloader, "_OPENER", _Failing())
    monkeypatch.setattr(curl_transport, "executable", lambda: None)
    assert curl_transport.probe("http://127.0.0.1:1/nothing") is None


def test_last_header_block_reads_the_final_hop():
    raw = (
        "HTTP/1.1 302 Found\r\n"
        "Location: https://cdn.example/file\r\n"
        "\r\n"
        "HTTP/1.1 206 Partial Content\r\n"
        "Content-Range: bytes 0-0/1024\r\n"
        'ETag: "abc123"\r\n'
        "Content-Length: 1\r\n"
        "\r\n"
    )
    headers, status = curl_transport._last_header_block(raw)

    assert status == 206
    assert headers["content-range"] == "bytes 0-0/1024"
    assert headers["etag"] == '"abc123"'
    assert "location" not in headers


def test_executable_reports_none_or_an_existing_curl():
    """The contract, not a specific host: a path is only ever returned if it exists."""
    found = curl_transport.executable()
    if found is None:
        assert curl_transport.enabled() is False
    else:
        assert Path(found).is_file()
        assert curl_transport.enabled() is True


def test_enabled_is_false_when_config_disables_it(monkeypatch):
    from hermes_cli import config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly",
                        lambda: {"download": {"allow_curl_fallback": False}})
    assert curl_transport.enabled() is False


@pytest.mark.parametrize("value", ["false", "no", "off", "0"])
def test_enabled_treats_false_shaped_strings_as_off(monkeypatch, value):
    """A YAML `no`/`off` must not silently keep the fallback on."""
    from hermes_cli import config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly",
                        lambda: {"download": {"allow_curl_fallback": value}})
    assert curl_transport._config_allows() is False


def test_enabled_defaults_on_when_config_is_absent(monkeypatch):
    """A missing or unreadable config must not take downloads down."""
    from hermes_cli import config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {})
    assert curl_transport._config_allows() is True


def test_enabled_defaults_on_when_config_is_broken(monkeypatch):
    from hermes_cli import config as config_mod

    def _boom():
        raise RuntimeError("invalid YAML")

    monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
    assert curl_transport._config_allows() is True
