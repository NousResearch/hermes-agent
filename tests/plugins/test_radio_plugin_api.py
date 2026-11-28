"""Tests for the bundled Radio dashboard backend.

Offline tests pin the mapping, paging, logging and outbound-request contract.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "radio"
    / "dashboard"
    / "plugin_api.py"
)
_spec = importlib.util.spec_from_file_location("radio_dashboard_api", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api)


@pytest.fixture(autouse=True)
def _clear_caches():
    for cache in (
        api._query_cache,
        api._stream_cache,
        api._now_playing_cache,
        api._probe_cache,
    ):
        cache.clear()
    yield


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


ROW = {
    "ID": 99426190,
    "Name": "BEST SMOOTH JAZZ - UK (LONDON) HOST ROD LUCAS",
    "Format": "audio/mpeg",
    "Bitrate": 128,
    "Genre": "Smooth Jazz",
    "CurrentTrack": "Host Rod Lucas - Londons Best Smooth Jazz UK",
    "Listeners": 443,
    "IceUrl": "https://bestsmoothjazz.com",
    "StreamUrl": None,
}


class FakeResponse:
    def __init__(
        self,
        *,
        status_code=200,
        headers=None,
        chunks=(),
        payload=None,
        url="https://example.test/",
        is_redirect=False,
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self.chunks = chunks
        self.payload = payload
        self.url = url
        self.is_redirect = is_redirect

    def json(self):
        return self.payload

    def iter_bytes(self):
        return iter(self.chunks)


def test_clean_text_and_stream_validation():
    assert api._clean_text("Tom &amp; Jerry <b>Live</b>") == "Tom & Jerry Live"
    assert api._clean_text(None) == ""
    assert (
        api._clean_stream("http://162.252.171.80:8002/stream")
        == "http://162.252.171.80:8002/stream"
    )
    assert api._clean_stream("file://../../victim") is None
    assert api._clean_stream("http://user:pass@example.test/x") is None
    assert api._clean_stream("http://") is None


def test_station_mapping_and_id_validation():
    assert api._station_id({"ID": 12}) == 12
    assert api._station_id({"ID": "12"}) == 12
    assert api._station_id({"ID": True}) is None
    assert api._station_id({}) is None
    station = api._station(ROW, "http://162.252.171.80:8002/stream")
    assert station["id"] == "sc-99426190"
    assert station["provider"] == "Shoutcast"
    assert station["description"] == "Smooth Jazz · 128 kbps"
    assert station["homepage"] == "https://bestsmoothjazz.com"
    assert station["nowPlaying"] == ROW["CurrentTrack"]


def test_stations_drop_unresolvable_rows_and_dead_streams(monkeypatch):
    rows = [
        dict(ROW, ID=1, Listeners=10),
        dict(ROW, ID=2, Listeners=900),
        dict(ROW, ID=3, Listeners=50),
    ]
    monkeypatch.setattr(
        api,
        "_stream_for",
        lambda station_id: (
            None if station_id == 1 else f"http://example.test/{station_id}"
        ),
    )
    monkeypatch.setattr(
        api,
        "_probe",
        lambda url: {
            "ok": "3" not in url,
            "kind": "audio" if "3" not in url else "html",
            "reason": "",
        },
    )
    logged = []
    monkeypatch.setattr(api, "_append_log", lambda entry: logged.append(entry))
    page = api._stations(rows, limit=10)
    assert [item["id"] for item in page["stations"]] == ["sc-2"]
    assert page["skipped"] == 2
    assert logged and logged[0]["source"] == "shoutcast-resolve"


def test_stations_page_with_offset(monkeypatch):
    rows = [dict(ROW, ID=index) for index in range(1, 13)]
    seen = []
    monkeypatch.setattr(
        api,
        "_stream_for",
        lambda station_id: (
            seen.append(station_id) or None
            if station_id == 5
            else f"http://example.test/{station_id}"
        ),
    )
    monkeypatch.setattr(
        api, "_probe", lambda url: {"ok": True, "kind": "audio", "reason": ""}
    )
    first = api._stations(rows, limit=4)
    assert [item["id"] for item in first["stations"]] == [
        "sc-1",
        "sc-2",
        "sc-3",
        "sc-4",
    ]
    assert first["next"] == 4
    second = api._stations(rows, limit=4, offset=4)
    assert [item["id"] for item in second["stations"]] == ["sc-6", "sc-7", "sc-8"]
    assert second["next"] == 8


def test_first_stream_reads_pls_and_m3u():
    assert (
        api._first_stream("[playlist]\nFile1=http://example.test/live\n")
        == "http://example.test/live"
    )
    assert (
        api._first_stream("#EXTM3U\nhttp://example.test/aac\n")
        == "http://example.test/aac"
    )
    assert api._first_stream("https://example.test/page") == "https://example.test/page"


def test_stream_title_and_freshest_cache():
    assert (
        api._stream_title(b"StreamTitle='Aphex Twin - Xtal';\x00\x00")
        == "Aphex Twin - Xtal"
    )
    assert api._stream_title(b"no title") == ""
    cache = {"k": (api.time.time(), ["old"])}
    assert api._freshest(cache, "k", 60) == ["old"]
    cache["k"] = (0.0, ["old"])
    assert api._freshest(cache, "k", 60) is None
    assert "k" not in cache


def test_probe_rejects_loopback_private_link_local_and_ipv6_ula():
    for url in (
        "http://127.0.0.1:8000/stream",
        "http://localhost/stream",
        "http://192.168.1.9:8000/",
        "http://10.0.0.5/stream",
        "http://169.254.169.254/latest/meta-data",
        "https://[fd00::1]/stream",
    ):
        verdict = api._probe(url)
        assert verdict["kind"] == "invalid", url
    assert api._probe("ftp://example.test/live")["kind"] == "invalid"


def test_safe_target_uses_shared_url_safety_for_resolved_addresses(monkeypatch):
    monkeypatch.setattr(api, "is_safe_url", lambda url: False)
    assert api._safe_target("https://public.example/stream") is None
    monkeypatch.setattr(api, "is_safe_url", lambda url: True)
    assert (
        api._safe_target("https://public.example/stream")
        == "https://public.example/stream"
    )


def test_request_refuses_unsafe_target_before_creating_a_client(monkeypatch):
    monkeypatch.setattr(api, "is_safe_url", lambda url: False)
    created = []
    monkeypatch.setattr(api, "_client", lambda timeout: created.append(timeout))
    with pytest.raises(HTTPException) as error:
        api._request("GET", "https://public.example/stream")
    assert error.value.status_code == 400
    assert created == []


def test_redirect_to_private_address_is_rechecked_and_blocked(monkeypatch):
    monkeypatch.setattr(api, "is_safe_url", lambda url: True)
    response = FakeResponse(
        status_code=302,
        headers={"location": "http://127.0.0.1:8000/admin"},
        url="https://public.example/stream",
        is_redirect=True,
    )
    with pytest.raises(api.UnsafeRedirect):
        api._guard_response(response)
    safe = FakeResponse(
        status_code=200, headers={}, url="https://public.example/stream"
    )
    api._guard_response(safe)


def test_client_registers_redirect_hook_on_ssrf_safe_client(monkeypatch):
    captured = {}
    monkeypatch.setattr(api, "is_safe_url", lambda url: True)

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(api, "create_ssrf_safe_client", fake_client)
    api._client(12.0)
    hooks = captured["event_hooks"]["response"]
    assert captured["follow_redirects"] is True
    assert len(hooks) == 1
    hooks[0](
        FakeResponse(status_code=200, headers={}, url="https://public.example/stream")
    )


def test_shoutcast_directory_request_uses_safe_post(monkeypatch):
    captured = {}
    monkeypatch.setattr(api, "is_safe_url", lambda url: True)

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return FakeResponse(payload=[ROW])

    monkeypatch.setattr(api, "_request", fake_request)
    rows = api._query_directory("/Search/UpdateSearch", "jazz")
    assert rows == [ROW]
    assert captured["method"] == "POST"
    assert captured["url"] == "https://directory.shoutcast.com/Search/UpdateSearch"
    assert captured["content"] == b"query=jazz"


def test_log_path_uses_active_hermes_home(home, monkeypatch):
    assert api._log_path() == home / "plugin-data" / "radio" / "station-log.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(home / "profile"))
    assert (
        api._log_path()
        == home / "profile" / "plugin-data" / "radio" / "station-log.jsonl"
    )


def test_log_route_writes_one_json_line(home):
    result = api.write_station_log(
        api.StationLogEntry(
            name="Bad <b>Station</b>",
            url="http://example.test/live",
            reason="network error",
            at="2026-01-01T00:00:00",
        )
    )
    assert result["status"] == "logged"
    assert result["path"] == str(home / "plugin-data" / "radio" / "station-log.jsonl")
    line = json.loads(
        (home / "plugin-data" / "radio" / "station-log.jsonl")
        .read_text()
        .splitlines()[0]
    )
    assert line["name"] == "Bad Station"
    assert line["url"] == "http://example.test/live"
    assert line["reason"] == "network error"
