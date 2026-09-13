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
    publish_mac_feed,
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


def test_never_overwrites_a_published_macos_artifact_on_same_tag_rerun(r2_server):
    name = "HermesBundled-0.28.0-mac-arm64.zip"
    existing = b"already published artifact"
    key = f"releases/tag/v0.28.0/{name}"
    r2_server.store[key] = (existing, '"etag-1"')
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(existing)
        path = handle.name
    try:
        r2.put("v0.28.0", name, path, immutable=True)
    finally:
        os.unlink(path)
    # The immutable bytes are untouched and the conflict was verified, not ignored.
    assert r2_server.store[key][0] == existing


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


def test_publish_verifies_artifacts_before_conditional_write_and_readback():
    plan = merge_mac_feeds(_inputs()[0], "v0.28.0")
    live = {"text": merge_mac_feeds(_inputs("0.27.0")[0], "v0.27.0")["text"], "etag": "old"}
    events = []

    def read(_key):
        return live

    def verify(key, _file=None):
        events.append(key)

    def write(key, text, etag):
        assert etag == "old"
        events.append(key)
        live.clear()
        live.update({"text": text, "etag": "new"})

    publish_mac_feed(plan, {"read": read, "verify": verify, "write": write})
    assert events[-1] == plan["key"]
    assert len(events) == 3

    write_calls = []

    def fail_write(key, text, etag):
        write_calls.append(key)

    # Downgrade rejection: nothing written.
    with pytest.raises(ValueError, match="backward"):
        publish_mac_feed(
            merge_mac_feeds(_inputs("0.27.0")[0], "v0.27.0"),
            {"read": read, "verify": verify, "write": fail_write},
        )
    assert write_calls == []
    # Corrupt bytes abort before the pointer write.
    def corrupt_verify(_key, _file=None):
        raise ValueError("corrupt bytes")

    with pytest.raises(ValueError, match="corrupt bytes"):
        publish_mac_feed(plan, {"read": lambda _k: None, "verify": corrupt_verify, "write": fail_write})
    assert write_calls == []


def test_same_version_identical_feed_is_a_noop():
    plan = merge_mac_feeds(_inputs()[0], "v0.28.0")
    live = {"text": plan["text"], "etag": "same"}
    calls = []
    publish_mac_feed(
        plan,
        {
            "read": lambda _k: live,
            "verify": lambda k: calls.append(k),
            "write": lambda k, t, e: calls.append(("write", k)),
        },
    )
    assert calls == []  # identical published version → no writes at all


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
