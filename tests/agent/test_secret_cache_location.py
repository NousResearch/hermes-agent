"""Secret-source disk caches live directly in the profile home.

A sandbox that hides credential stores by path can hide a file safely only at the top of the
profile home: a writable folder above a hidden file can be renamed to expose it, and the
sandboxed backend has to write ``cache/``. Earlier releases kept these caches in ``cache/``; Hermes
deletes such a copy on first use without reading it, since the agent could have written it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from agent.secret_sources import bitwarden as bw
from agent.secret_sources import onepassword as op
from agent.secret_sources._cache import CachedFetch, DiskCache


@pytest.fixture(autouse=True)
def _reset_caches():
    bw._reset_cache_for_tests()
    op._reset_cache_for_tests()
    yield
    bw._reset_cache_for_tests()
    op._reset_cache_for_tests()


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return home


def _payload(key: str, secrets: dict) -> str:
    return json.dumps({"key": key, "secrets": secrets, "fetched_at": time.time()})


def _previous(home: Path, name: str) -> Path:
    path = home / "cache" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_caches_live_at_the_top_of_the_profile_home(home):
    assert bw._disk_cache_path(home) == home / "bws_cache.json"
    assert bw._encrypted_disk_cache_path(home) == home / "bws_cache.enc.json"
    assert op._STORE.disk.path(home) == home / "op_cache.json"


def test_a_write_lands_in_the_profile_home(home):
    cache = DiskCache("probe.json", key_serializer=str)
    cache.write("k", CachedFetch(secrets={"A": "b"}, fetched_at=time.time()), 60, home)

    assert json.loads((home / "probe.json").read_text(encoding="utf-8"))["secrets"] == {"A": "b"}
    assert not (home / "cache").exists()


def test_the_first_read_deletes_a_previous_copy_unread(home):
    cache = DiskCache("probe.json", key_serializer=str)
    previous = _previous(home, "probe.json")
    previous.write_text(_payload("k", {"A": "b"}), encoding="utf-8")

    assert cache.read("k", 60, home) is None
    assert not previous.exists()
    assert not (home / "probe.json").exists()


def test_a_current_cache_wins_and_the_previous_copy_is_removed(home):
    cache = DiskCache("probe.json", key_serializer=str)
    (home / "probe.json").write_text(_payload("k", {"A": "current"}), encoding="utf-8")
    previous = _previous(home, "probe.json")
    previous.write_text(_payload("k", {"A": "previous"}), encoding="utf-8")

    entry = cache.read("k", 60, home)

    assert entry is not None and entry.secrets == {"A": "current"}
    assert not previous.exists()


def test_a_write_deletes_a_previous_copy(home):
    cache = DiskCache("probe.json", key_serializer=str)
    previous = _previous(home, "probe.json")
    previous.write_text(_payload("k", {"A": "previous"}), encoding="utf-8")

    cache.write("k", CachedFetch(secrets={"A": "c"}, fetched_at=time.time()), 60, home)

    assert not previous.exists()
    assert json.loads((home / "probe.json").read_text(encoding="utf-8"))["secrets"] == {"A": "c"}


def test_a_previous_symlink_is_removed_and_its_target_kept(home):
    cache = DiskCache("probe.json", key_serializer=str)
    target = home / "workspace" / "leak.json"
    target.parent.mkdir()
    target.write_text("{}", encoding="utf-8")
    previous = _previous(home, "probe.json")
    try:
        previous.symlink_to(target)
    except OSError:
        pytest.skip("creating a symlink needs a privilege this host lacks")

    cache.write("k", CachedFetch(secrets={"A": "c"}, fetched_at=time.time()), 60, home)

    assert not os.path.lexists(previous)
    assert target.read_text(encoding="utf-8") == "{}"
    assert json.loads((home / "probe.json").read_text(encoding="utf-8"))["secrets"] == {"A": "c"}


def test_a_previous_directory_is_left_alone(home):
    cache = DiskCache("probe.json", key_serializer=str)
    (_previous(home, "probe.json") / "inner").mkdir(parents=True)

    cache.write("k", CachedFetch(secrets={"A": "c"}, fetched_at=time.time()), 60, home)

    assert (home / "cache" / "probe.json" / "inner").is_dir()
    assert json.loads((home / "probe.json").read_text(encoding="utf-8"))["secrets"] == {"A": "c"}


def test_clear_removes_both_locations(home):
    cache = DiskCache("probe.json", key_serializer=str)
    (home / "probe.json").write_text(_payload("k", {"A": "b"}), encoding="utf-8")
    previous = _previous(home, "probe.json")
    previous.write_text(_payload("k", {"A": "b"}), encoding="utf-8")

    cache.clear(home)

    assert not (home / "probe.json").exists()
    assert not previous.exists()


def test_onepassword_cache_deletes_the_previous_copy_unread(home):
    key = ("auth-fp", "account", str(home), "refs-fp")
    previous = _previous(home, "op_cache.json")
    previous.write_text(_payload(op._disk_key_str(key), {"OP_KEY": "v"}), encoding="utf-8")

    assert op._STORE.disk.read(key, 60, home) is None
    assert not previous.exists()
    assert not (home / "op_cache.json").exists()


def _seed_encrypted(home: Path) -> tuple[tuple[str, str, str], Path]:
    pytest.importorskip("cryptography")
    cache_key = (bw._token_fingerprint("0.t"), "proj-1", "")
    bw._write_encrypted_disk_cache(cache_key=cache_key, access_token="0.t",
                                   entry=CachedFetch(secrets={"K": "v"}, fetched_at=time.time()),
                                   home_path=home)
    previous = _previous(home, "bws_cache.enc.json")
    os.replace(home / "bws_cache.enc.json", previous)
    return cache_key, previous


def test_bitwarden_encrypted_cache_deletes_the_previous_copy_unread(home):
    cache_key, previous = _seed_encrypted(home)

    entry = bw._read_encrypted_disk_cache(cache_key=cache_key, access_token="0.t",
                                          max_age_seconds=3600, home_path=home)

    assert entry is None
    assert not previous.exists()
    assert not (home / "bws_cache.enc.json").exists()


def test_bitwarden_clear_caches_removes_previous_copies(home):
    for name in ("bws_cache.json", "bws_cache.enc.json"):
        _previous(home, name).write_text("{}", encoding="utf-8")
        (home / name).write_text("{}", encoding="utf-8")

    bw.clear_caches(home)

    leftovers = [p for p in (*home.iterdir(), *(home / "cache").iterdir()) if p.is_file()]
    assert leftovers == []
