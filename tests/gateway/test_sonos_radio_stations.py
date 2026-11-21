from __future__ import annotations

import ipaddress
import json
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from integrations.sonos_radio_stations import (
    MAX_STATIONS,
    add_station,
    get_station,
    list_stations,
    remove_station,
)


def station_file(tmp_path):
    return tmp_path / "private" / "radio-stations.json"


def test_empty_store_lists_without_creating_a_file(tmp_path):
    path = station_file(tmp_path)

    assert list_stations(path) == []
    assert not path.exists()


def test_add_get_list_and_remove_station(tmp_path):
    path = station_file(tmp_path)

    created = add_station("Jazz FM", "https://radio.example/live.mp3", path)

    assert created["name"] == "Jazz FM"
    assert created["url"] == "https://radio.example/live.mp3"
    assert created["id"]
    assert get_station(created["id"], path) == created
    assert list_stations(path) == [created]
    assert remove_station(created["id"], path) is True
    assert list_stations(path) == []
    assert remove_station(created["id"], path) is False


def test_concurrent_adds_do_not_overwrite_each_other(tmp_path, monkeypatch):
    import integrations.sonos_radio_stations as station_store

    path = station_file(tmp_path)
    original_load = station_store._load
    barrier = threading.Barrier(2)

    def synchronized_load(target):
        stations = original_load(target)
        try:
            barrier.wait(timeout=0.25)
        except threading.BrokenBarrierError:
            pass
        return stations

    monkeypatch.setattr(station_store, "_load", synchronized_load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(add_station, "One", "https://one.example/live", path),
            executor.submit(add_station, "Two", "https://two.example/live", path),
        ]
        [future.result(timeout=5) for future in futures]

    assert {station["name"] for station in list_stations(path)} == {"One", "Two"}


def test_default_store_uses_current_hermes_home(tmp_path, monkeypatch):
    import integrations.sonos_radio_stations as station_store

    profile_home = tmp_path / "profile-home"
    monkeypatch.setattr(station_store, "get_hermes_home", lambda: profile_home)
    station = add_station("WFMU", "https://stream.example/freeform", path=None)

    assert station_store.list_stations() == [station]
    assert (profile_home / "sonos-brain-state" / "radio-stations.json").is_file()


def test_station_ids_are_short_random_and_unique(tmp_path):
    path = station_file(tmp_path)

    first = add_station("One", "https://one.example/a", path)
    second = add_station("Two", "https://two.example/b", path)

    assert first["id"] != second["id"]
    assert 6 <= len(first["id"]) <= 16


@pytest.mark.parametrize("name", ["", "  ", "x" * 81, "\nstation\n"])
def test_rejects_invalid_station_names(tmp_path, name):
    with pytest.raises(ValueError):
        add_station(name, "https://radio.example/live", station_file(tmp_path))


def test_trims_station_name(tmp_path):
    station = add_station("  Calm Radio  ", "https://radio.example/live", station_file(tmp_path))
    assert station["name"] == "Calm Radio"


def test_rejects_duplicate_name_case_insensitively(tmp_path):
    path = station_file(tmp_path)
    add_station("Jazz FM", "https://one.example/live", path)

    with pytest.raises(ValueError, match="duplicate"):
        add_station("jAzZ fm", "https://two.example/live", path)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "relative/stream",
        "ftp://radio.example/stream",
        "https:///stream",
        "https://user:pass@radio.example/stream",
        "https://radio.example/stream#fragment",
        "https://radio.example/" + "a" * 2048,
        "https://radio.example/live?access_token=secret",
        "https://radio.example/live?api_key=secret",
        "https://radio.example/live?Signature=secret",
        "http://localhost/live",
        "http://localhost./live",
        "http://radio.local/live",
        "http://radio.private/live",
        "http://127.0.0.1/live",
        "http://10.0.0.2/live",
        "http://172.16.0.2/live",
        "http://192.168.1.2/live",
        "http://169.254.1.2/live",
        "http://[::1]/live",
        "http://[fc00::1]/live",
        "http://[fe80::1]/live",
    ],
)
def test_rejects_unsafe_or_invalid_urls(tmp_path, url):
    with pytest.raises(ValueError):
        add_station("Station", url, station_file(tmp_path))


def test_allows_public_http_stream_and_non_secret_query(tmp_path):
    url = "http://radio.example:8000/live?format=mp3"
    station = add_station("Public Stream", url, station_file(tmp_path))
    assert station["url"] == url


def test_rejects_duplicate_url_case_insensitively(tmp_path):
    path = station_file(tmp_path)
    add_station("First", "https://Radio.Example/Live", path)

    with pytest.raises(ValueError, match="duplicate"):
        add_station("Second", "https://radio.example/live", path)


@pytest.mark.parametrize("url", ["https://radio.example/live?auth=secret", "https://radio.example/live?password=x", "https://radio.example/live?client_secret=x"])
def test_rejects_secret_like_query_parameters(tmp_path, url):
    with pytest.raises(ValueError):
        add_station("Station", url, station_file(tmp_path))


def test_rejects_station_limit(tmp_path):
    path = station_file(tmp_path)
    for index in range(MAX_STATIONS):
        add_station(f"Station {index}", f"https://radio{index}.example/live", path)

    with pytest.raises(ValueError, match="limit"):
        add_station("One too many", "https://overflow.example/live", path)


def test_rejects_corrupt_store_without_overwriting_it(tmp_path):
    path = station_file(tmp_path)
    path.parent.mkdir(mode=0o700, parents=True)
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="store"):
        list_stations(path)
    assert path.read_text(encoding="utf-8") == "not json"


def test_store_directory_and_file_are_private(tmp_path):
    path = station_file(tmp_path)
    add_station("Station", "https://radio.example/live", path)

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_writes_valid_json_and_does_not_leave_temporary_files(tmp_path):
    path = station_file(tmp_path)
    station = add_station("Station", "https://radio.example/live", path)

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == {"stations": [station]}
    assert {entry.name for entry in path.parent.iterdir()} == {path.name, path.with_suffix(".lock").name}
    assert stat.S_IMODE(path.with_suffix(".lock").stat().st_mode) == 0o600


def test_get_unknown_station_returns_none(tmp_path):
    assert get_station("missing", station_file(tmp_path)) is None


def test_list_returns_copies_not_mutable_store_state(tmp_path):
    path = station_file(tmp_path)
    add_station("Station", "https://radio.example/live", path)

    stations = list_stations(path)
    stations[0]["name"] = "changed"
    assert list_stations(path)[0]["name"] == "Station"


def test_rejects_non_string_inputs(tmp_path):
    with pytest.raises(ValueError):
        add_station(None, "https://radio.example/live", station_file(tmp_path))
    with pytest.raises(ValueError):
        add_station("Station", None, station_file(tmp_path))


def test_rejects_ipv4_and_ipv6_addresses_not_globally_routable(tmp_path):
    for address in ("0.0.0.0", "100.64.0.1", "198.18.0.1", "224.0.0.1", "::", "2001:db8::1"):
        parsed = ipaddress.ip_address(address)
        with pytest.raises(ValueError):
            add_station("Station", f"http://[{address}]/live" if parsed.version == 6 else f"http://{address}/live", station_file(tmp_path) / address.replace(":", "_"))


def test_rejects_removal_of_invalid_id_type(tmp_path):
    with pytest.raises(ValueError):
        remove_station(None, station_file(tmp_path))
