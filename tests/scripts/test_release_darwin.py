# tests/scripts/test_release_darwin.py — contract tests for the macOS
# electron-updater feed publication (port of tests-js/darwin-feed.test.mjs).
# Feed merge/validation is pure; publication runs against a REAL loopback
# HTTP server (http.server on 127.0.0.1), not a fabricated backend.

from __future__ import annotations

import hashlib
import os
import tempfile

import pytest
import hermes_yaml as yaml

from scripts.releases import darwin, r2
from tests.scripts.test_release_r2 import r2_server  # noqa: F401 — loopback fixture
from scripts.releases.darwin import (
    _darwin_feed,
    mac_feed_references,
    merge_mac_feeds,
    parse_mac_feed,
)


def _inputs(version="0.28.0", light=False):
    channel = "canary" if "-canary." in version else "stable"
    bytes_by_key = {}
    legs = {}
    for i, arch in enumerate(("arm64", "x64")):
        name = f"{'HermesLight' if light else 'HermesBundled'}-{version}-mac-{arch}.zip"
        data = f"test artifact {arch}".encode()
        bytes_by_key[f"releases/tag/v{version}/{name}"] = data
        file_entry = {
            "url": name,
            "size": len(data),
            "sha512": base64_sha512(data),
        }
        legs[f"{arch}-{channel}-mac.yml"] = yaml.safe_dump(
            {
                "version": version,
                "files": [file_entry],
                "path": name,
                "sha512": file_entry["sha512"],
                "releaseDate": f"2026-09-0{i + 1}T00:00:00Z",
                "releaseNotes": "two lines\nof release notes",
            },
            sort_keys=False,
        )
    return legs, bytes_by_key


def base64_sha512(data: bytes) -> str:
    return __import__("base64").b64encode(hashlib.sha512(data).digest()).decode("ascii")



def test_real_prune_protects_live_macos_artifacts_and_fails_closed(r2_server, monkeypatch):
    legs, _bytes = _inputs("0.28.0-canary.20200101000000")
    plan = merge_mac_feeds(legs, "v0.28.0-canary.20200101000000")
    references = mac_feed_references(plan["text"])
    stale = "releases/tag/v0.27.0-canary.20200101000000/unreferenced.zip"
    for key in [plan["key"], *references, stale]:
        r2_server.store[key] = (b"x", '"e"')
    r2_server.store[plan["key"]] = (plan["text"].encode(), '"e"')
    monkeypatch.setattr(r2.time, "time", lambda: 1788547200.0)  # 2026-09-04
    r2.prune(keep_days=14, dry_run=False)
    assert stale not in r2_server.store
    for reference in references:
        assert reference in r2_server.store, reference
    # Fail closed: an unreadable manifest aborts the prune, nothing deleted.
    r2_server.store[stale] = (b"still protected until all feeds are readable", '"e"')
    before = set(r2_server.store)
    r2_server.store["releases/darwin/canary/canary-mac.yml"] = (b"<html>503</html>", '"e"')
    with pytest.raises(ValueError):
        r2.prune(keep_days=14, dry_run=False)
    assert set(r2_server.store) == before


def test_merges_native_legs_and_preserves_release_metadata():
    legs, _bytes = _inputs()
    plan = merge_mac_feeds(legs, "v0.28.0")
    feed = parse_mac_feed(plan["text"])
    assert len(feed["files"]) == 2
    assert feed["releaseNotes"] == "two lines\nof release notes"
    assert plan["key"] == "releases/darwin/stable/stable-mac.yml"
    assert mac_feed_references(plan["text"]) == [
        ref
        for f in feed["files"]
        for ref in (f["url"][1:], f"{f['url'][1:]}.blockmap")
    ]
    assert r2.cache_control_for(plan["key"]) == "no-store"
    selection = _darwin_feed("canary", True)
    light_legs, _ = _inputs("0.29.0-canary.20260906000000", light=True)
    light_plan = merge_mac_feeds(light_legs, "v0.29.0-canary.20260906000000", light=True)
    assert light_plan["key"] == f"{selection['directory']}/{selection['fileName']}"


@pytest.mark.parametrize(
    "kind", ["missing", "version", "variant", "hash", "legacy", "traversal"]
)
def test_rejects_broken_legs_instead_of_publishing(kind):
    legs, _bytes = _inputs()
    key = "arm64-stable-mac.yml"
    feed = yaml.safe_load(legs[key])
    if kind == "missing":
        del legs[key]
    else:
        if kind == "version":
            feed["version"] = "0.27.0"
        if kind == "variant":
            feed["files"][0]["url"] = feed["files"][0]["url"].replace("HermesBundled", "HermesLight")
        if kind == "hash":
            feed["files"][0]["sha512"] = "invalid"
        if kind == "legacy":
            feed["sha512"] = "wrong"
        if kind == "traversal":
            feed["files"][0]["url"] = "../other.zip"
        legs[key] = yaml.safe_dump(feed, sort_keys=False)
    with pytest.raises(Exception):
        merge_mac_feeds(legs, "v0.28.0")


def test_semver_grammar_rejects_non_release_versions():
    # The Python port never imports npm semver; it implements exactly the
    # release grammar (vMAJOR.MINOR.PATCH[-canary.<14 digits>]) and fails
    # loudly on anything else instead of guessing an order.
    from scripts.releases.semver import compare, is_valid_version

    assert is_valid_version("0.28.0") and is_valid_version("0.28.0-canary.20260904101010")
    assert not is_valid_version("0.28") and is_valid_version("2026.7.20")
    assert not is_valid_version("0.28.0-beta.1")
    assert compare("0.28.0", "0.27.9") == 1
    assert compare("0.28.0-canary.20260904101010", "0.28.0") == -1  # prerelease < release
    assert compare("0.28.0-canary.20260904101010", "0.28.0-canary.20260904101011") == -1
    with pytest.raises(ValueError):
        compare("nonsense", "0.28.0")



def test_finalize_uses_the_real_signed_transport_and_publishes_last(r2_server):
    with tempfile.TemporaryDirectory() as dir_path:
        legs, bytes_by_key = _inputs()
        for name, text in legs.items():
            with open(os.path.join(dir_path, name), "w", encoding="utf-8") as handle:
                handle.write(text)
        for key, value in bytes_by_key.items():
            r2_server.store[key] = (value, '"e"')
        darwin.finalize(tag="v0.28.0", dir=dir_path)
    feed_key = "releases/darwin/stable/stable-mac.yml"
    assert feed_key in r2_server.store
    published = r2_server.store[feed_key][0].decode("utf-8")
    assert len(parse_mac_feed(published)["files"]) == 2
    # Order: reads (legs verified) FIRST, then exactly one conditional PUT,
    # then the readback — verify bytes before the pointer write.
    methods = [m for m, _p, _h in r2_server.requests]
    assert methods.count("PUT") == 1
    assert "PUT" not in methods[:3]
    put_headers = [h for m, _p, h in r2_server.requests if m == "PUT"][0]
    assert put_headers["If-None-Match"] == "*"
    assert put_headers["Cache-Control"] == "no-store"
    assert put_headers["Content-Type"] == "application/yaml"
    # Every request was signed.
    for _m, _p, headers in r2_server.requests:
        assert headers["authorization"].startswith("AWS4-HMAC-SHA256 ")


def test_finalize_rejects_a_downgrade_over_the_live_feed(r2_server):
    with tempfile.TemporaryDirectory() as dir_path:
        legs, bytes_by_key = _inputs("0.27.0")
        for name, text in legs.items():
            with open(os.path.join(dir_path, name), "w", encoding="utf-8") as handle:
                handle.write(text)
        for key, value in bytes_by_key.items():
            r2_server.store[key] = (value, '"e"')
        # The live feed is already at 0.28.0.
        newer = merge_mac_feeds(_inputs()[0], "v0.28.0")
        r2_server.store["releases/darwin/stable/stable-mac.yml"] = (
            newer["text"].encode(), '"live"')
        with pytest.raises(ValueError, match="backward"):
            darwin.finalize(tag="v0.27.0", dir=dir_path)
    # The live pointer was not replaced.
    assert parse_mac_feed(
        r2_server.store["releases/darwin/stable/stable-mac.yml"][0].decode()
    )["version"] == "0.28.0"


def test_finalize_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        darwin.finalize(tag="v0.28.0", dir=".", variant="dark")


@pytest.mark.parametrize('fault', ['identical', 'different', 'corrupt-artifact', 'stale-etag', 'readback', 'upgrade'])
def test_finalize_live_feed_faults(tmp_path, r2_server, fault):
    legs, content = _inputs()
    for name, text in legs.items():
        (tmp_path / name).write_text(text, encoding='utf-8')
    r2_server.store.update({key: (value, '"e"') for key, value in content.items()})
    key = 'releases/darwin/stable/stable-mac.yml'
    live = merge_mac_feeds(_inputs('0.28.0' if fault in {'identical', 'different'} else '0.27.0')[0],
                           'v0.28.0' if fault in {'identical', 'different'} else 'v0.27.0')['text']
    if fault == 'different':
        live = live.replace('two lines', 'other notes')
    r2_server.store[key] = (live.encode(), '"live"')
    if fault == 'corrupt-artifact':
        r2_server.store[next(iter(content))] = (b'corrupt', '"e"')
    if fault == 'stale-etag':
        r2_server.race_key = key
    if fault == 'readback':
        r2_server.corrupt_put = key
    errors = {'different': (ValueError, 'different artifacts'),
              'corrupt-artifact': (ValueError, 'checksum mismatch'),
              'stale-etag': (r2.R2RequestError, '412'), 'readback': (ValueError, 'readback differs')}
    if fault in errors:
        error, message = errors[fault]
        with pytest.raises(error, match=message):
            darwin.finalize(tag='v0.28.0', dir=str(tmp_path))
    else:
        darwin.finalize(tag='v0.28.0', dir=str(tmp_path))
    puts = [headers for method, _, headers in r2_server.requests if method == 'PUT']
    if fault in {'identical', 'different', 'corrupt-artifact'}:
        assert puts == [] and r2_server.store[key][0] == live.encode()
    else:
        assert len(puts) == 1 and puts[0]['If-Match'] == '"live"'
        if fault == 'stale-etag':
            assert r2_server.store[key] == (live.encode(), '"raced"')
        elif fault == 'upgrade':
            assert parse_mac_feed(r2_server.store[key][0].decode())['version'] == '0.28.0'
        elif fault == 'readback':
            assert [method for method, path, _ in r2_server.requests if path.endswith(key)] == ['GET', 'PUT', 'HEAD', 'GET']
