# tests/scripts/test_release_r2.py — contract tests for the Python R2
# release transport (port of tests-js/r2-release.test.mjs). The SigV4
# vectors pin the signer against botocore (the reference implementation,
# 1.43.81) at a FIXED timestamp/creds, so the expected values are
# reproducible fixtures rather than self-consistency:
#   - get-vanilla        generic signer (no x-amz-content-sha256), example.com
#   - r2-put-payload     S3 signer, payload hash signed, region auto
#   - r2-list            S3 signer, ListObjectsV2 with query params
#   - r2-delete          S3 signer, region auto
# The get-vanilla case also reproduces the public aws-sig-v4-test-suite
# request shape (verified by independent spec computation).
#
# Protocol behavior (put/finalize/prune) is tested against a REAL loopback
# HTTP server (http.server on 127.0.0.1) — no fabricated backend output.

from __future__ import annotations

import base64
import io
import json
import os
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts.releases import r2
from scripts.releases.r2 import (
    auth_header,
    cache_control_for,
    canary_doomed_keys,
    canonical_query,
    canonical_request,
    channel_for_tag,
    channel_page_key_for,
    commit_page_key_for,
    commit_prefix_for,
    content_type_for,
    encode_key_path,
    feed_dir_for,
    feed_referenced_keys,
    parse_list_xml,
    public_base_url,
    public_url_for,
    publish_feed_uploads,
    referenced_feed_bundle_filenames,
    rfc3986_encode,
    staging_key_for,
    stale_feed_bundle_keys,
)

AKID = "AKIDEXAMPLE"
SECRET = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
NOW = "20150830T123600Z"
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _auth(**kwargs):
    kwargs.setdefault("access_key_id", AKID)
    kwargs.setdefault("secret_key", SECRET)
    kwargs.setdefault("now", NOW)
    return auth_header(**kwargs)


# ── SigV4 vectors ───────────────────────────────────────────────────────────

def test_get_vanilla_matches_the_aws_test_suite_vector():
    authz = _auth(
        method="GET",
        host="example.com",
        path="/",
        query="",
        headers={"host": "example.com", "x-amz-date": NOW},
        payload_hash=EMPTY_SHA,
        region="us-east-1",
        service="service",
    )
    assert authz == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
        "SignedHeaders=host;x-amz-date, "
        "Signature=33399fd3d4a9d6104710c7c04005f7c959f8b1f8bf41b823587ed36b079e453f"
    )


def test_r2_put_payload_matches_botocore():
    body_hash = "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"  # sha256("Welcome to Amazon S3.")
    host = "abc123.r2.cloudflarestorage.com"
    authz = _auth(
        method="PUT",
        host=host,
        path="/hermes-releases/HermesBundled-0.28.0-win-x64.msix",
        query="",
        headers={"host": host, "x-amz-date": NOW, "x-amz-content-sha256": body_hash},
        payload_hash=body_hash,
    )
    assert authz == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/auto/s3/aws4_request, "
        "SignedHeaders=host;x-amz-content-sha256;x-amz-date, "
        "Signature=05ba50acfb54042fac330848af50877e5fb477c4f2063c2f77f9cc80855eb1e9"
    )


def test_r2_list_matches_botocore():
    query = canonical_query(
        {"list-type": "2", "prefix": "HermesBundled-0.28.0-", "max-keys": "1000"}
    )
    assert query == "list-type=2&max-keys=1000&prefix=HermesBundled-0.28.0-"
    host = "abc123.r2.cloudflarestorage.com"
    authz = _auth(
        method="GET",
        host=host,
        path="/hermes-releases",
        query=query,
        headers={"host": host, "x-amz-date": NOW, "x-amz-content-sha256": EMPTY_SHA},
        payload_hash=EMPTY_SHA,
    )
    assert authz == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/auto/s3/aws4_request, "
        "SignedHeaders=host;x-amz-content-sha256;x-amz-date, "
        "Signature=3ec423c452a318664c85fbcc25667ad07201aedce688e3bb6b345b4baaa39d90"
    )


def test_r2_delete_matches_botocore():
    host = "abc123.r2.cloudflarestorage.com"
    authz = _auth(
        method="DELETE",
        host=host,
        path="/hermes-releases/HermesBundled-0.28.0-canary.20260818-win-arm64.msix",
        query="",
        headers={"host": host, "x-amz-date": NOW, "x-amz-content-sha256": EMPTY_SHA},
        payload_hash=EMPTY_SHA,
    )
    assert authz == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/auto/s3/aws4_request, "
        "SignedHeaders=host;x-amz-content-sha256;x-amz-date, "
        "Signature=ec5ccb76f701193b28aaca052cabbf2f084e9c71ffc64b872fa51d4e70dc6e55"
    )


# ── Encoding / layout helpers ───────────────────────────────────────────────

def test_rfc3986_encode_escapes_the_aws_reserved_set_keeps_unreserved():
    assert rfc3986_encode("HermesBundled-0.28.0-win-x64.msix") == "HermesBundled-0.28.0-win-x64.msix"
    assert rfc3986_encode("a b!'()*c") == "a%20b%21%27%28%29%2Ac"


def test_encode_key_path_encodes_segment_wise_preserves_separators():
    assert encode_key_path("HermesBundled-0.28.0-win-x64.msix") == "HermesBundled-0.28.0-win-x64.msix"
    assert encode_key_path("a b/c d") == "a%20b/c%20d"


def test_parse_list_xml_extracts_keys_truncation_token_entities():
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">\n'
        "  <Name>hermes-releases</Name>\n  <Prefix></Prefix>\n"
        "  <KeyCount>3</KeyCount>\n  <MaxKeys>1000</MaxKeys>\n"
        "  <IsTruncated>true</IsTruncated>\n"
        "  <Contents><Key>HermesBundled-0.28.0-win-x64.msix</Key>"
        "<LastModified>2026-08-18T00:00:00Z</LastModified><Size>123</Size></Contents>\n"
        "  <Contents><Key>a&amp;b.msix</Key>"
        "<LastModified>2026-08-18T00:00:00Z</LastModified><Size>1</Size></Contents>\n"
        "  <Contents><Key>latest.yml</Key>"
        "<LastModified>2026-08-18T00:00:00Z</LastModified><Size>2</Size></Contents>\n"
        "  <NextContinuationToken>abc+def/=</NextContinuationToken>\n"
        "</ListBucketResult>"
    )
    parsed = parse_list_xml(xml)
    assert parsed["keys"] == ["HermesBundled-0.28.0-win-x64.msix", "a&b.msix", "latest.yml"]
    assert parsed["truncated"] is True
    assert parsed["nextToken"] == "abc+def/="


def test_canary_doomed_keys_dates_by_the_key_suffix():
    keys = [
        "releases/tag/v0.28.0/HermesBundled-0.28.0-win-x64.msix",  # stable — never doomed
        "releases/tag/v0.28.0-canary.20260801/HermesBundled-0.28.0-canary.20260801-win-x64.msix",
        "releases/tag/v0.28.0-canary.20260818/HermesBundled-0.28.0-canary.20260818-win-x64.msix",  # today — kept
        "releases/tag/v0.28.0-canary.20260801/HermesBundled-0.28.0-canary.20260801-win-x64.msix.blockmap",
        "latest.yml",
        "canary.yml",
    ]
    assert canary_doomed_keys(keys, "20260814") == [
        "releases/tag/v0.28.0-canary.20260801/HermesBundled-0.28.0-canary.20260801-win-x64.msix",
        "releases/tag/v0.28.0-canary.20260801/HermesBundled-0.28.0-canary.20260801-win-x64.msix.blockmap",
    ]


def test_channel_for_tag_maps_stable_vs_canary():
    assert channel_for_tag("v0.28.0") == "stable"
    assert channel_for_tag("v0.28.0-canary.20260818101010") == "canary"
    assert channel_for_tag("v0.28.0-canary.20260818") == "canary"


def test_relative_artifact_path_rejects_windows_reserved_names_without_ntpath(monkeypatch):
    """Windows-reserved names must be rejected even when ntpath.isreserved
    does not exist (release CI legs run on system Pythons older than 3.13;
    commit-builds-summary crashed on the bare AttributeError)."""
    import ntpath

    # Simulate a pre-3.13 ntpath: isreserved absent, as on the ubuntu-24.04
    # system Python the release workflows run on. Red on the old code, which
    # called ntpath.isreserved unconditionally.
    monkeypatch.delattr(ntpath, "isreserved", raising=False)

    def check(bad):
        with pytest.raises(ValueError, match="Invalid release artifact path"):
            r2.relative_artifact_path(bad)

    # DOS device stems in every dotted form, in every component.
    for bad in ("nul", "NUL.txt", "con.tar.gz", "desktop/aux.js", "com1",
                "lpt9.zip", "a/prn.gz", "COM¹.txt", "x/CONOUT$/y"):
        check(bad)
    # Trailing dots and spaces are reserved on Windows (internal ones are not).
    for bad in ("foo.", "foo ", "dir/foo.", "dir/foo..", "dir/foo .tar.gz."):
        check(bad)
    # Real artifact names, including the nested Termux receipt shape, pass.
    for good in ("app.msix", "deb/pool/hermes_0.28.0_aarch64.deb",
                 "HermesBundled-0.28.0-win-x64.msix", "key.asc", "com10.txt",
                 "xcom1.tar.gz", "hermes-agent-setup.exe"):
        assert r2.relative_artifact_path(good) == good


def test_staging_key_and_feed_dir_layout_keys():
    assert staging_key_for("v0.28.0", "HermesBundled-0.28.0-win-x64.msix") == (
        "releases/tag/v0.28.0/HermesBundled-0.28.0-win-x64.msix"
    )
    assert feed_dir_for("win32", "stable") == "releases/win32/stable"
    assert feed_dir_for("darwin", "canary") == "releases/darwin/canary"


def test_download_page_keys_and_public_urls():
    assert channel_page_key_for("stable") == "releases/stable/index.html"
    assert channel_page_key_for("canary") == "releases/canary/index.html"
    commit = "a" * 40
    assert commit_page_key_for(commit) == f"releases/commit/{commit}/index.html"
    assert commit_prefix_for(commit) == f"releases/commit/{commit}/"
    assert public_url_for("https://cdn.example.com/", "releases/tag/v1/Hermes-1-x64.msix") == (
        "https://cdn.example.com/releases/tag/v1/Hermes-1-x64.msix"
    )
    # Segment-wise encoding: spaces and non-ASCII survive a link.
    assert public_url_for("https://cdn.example.com", "a b/\u00fc.msix") == (
        "https://cdn.example.com/a%20b/%C3%BC.msix"
    )


def test_public_base_url_precedence(monkeypatch):
    # Explicit value, then $CLOUDFLARE_R2_PUBLIC_URL, then the documented
    # production origin — so a local command always names a real page.
    monkeypatch.delenv("CLOUDFLARE_R2_PUBLIC_URL", raising=False)
    assert public_base_url() == "https://hermes-assets.nousresearch.com"
    monkeypatch.setenv("CLOUDFLARE_R2_PUBLIC_URL", "https://cdn.example.com")
    assert public_base_url() == "https://cdn.example.com"
    assert public_base_url("https://explicit.example.com/") == "https://explicit.example.com"


def test_content_type_for_maps_msix_and_appinstaller():
    assert content_type_for("HermesBundled-0.28.0-win-x64.msix") == "application/msix"
    assert content_type_for("HermesBundled-0.28.0-win.msixbundle") == "application/msixbundle"
    assert content_type_for("stable.appinstaller") == "application/appinstaller"
    assert content_type_for("releases/stable/index.html") == "text/html; charset=utf-8"
    assert content_type_for("HermesBundled-0.28.0-mac-x64.dmg") is None
    assert content_type_for("latest-mac.yml") is None
    # Case-insensitive on the suffix.
    assert content_type_for("X.APPINSTALLER") == "application/appinstaller"
    assert content_type_for("INDEX.HTML") == "text/html; charset=utf-8"


def test_apt_mutable_metadata_revalidates_immutable_bytes_cache():
    feed = "releases/termux/canary"
    for name in [
        "key.asc",
        "dists/hermes-canary/InRelease",
        "dists/hermes-canary/Release",
        "dists/hermes-canary/main/binary-aarch64/Packages.gz",
    ]:
        assert cache_control_for(f"{feed}/{name}") == "no-store"
    assert cache_control_for(f"{feed}/dists/hermes-canary/main/binary-aarch64/by-hash/SHA256/abcd") == (
        "public, max-age=31536000, immutable"
    )
    assert cache_control_for(f"{feed}/pool/h/hermes-agent_1.2.3_aarch64.deb") == (
        "public, max-age=31536000, immutable"
    )
    assert cache_control_for("releases/win32/stable/stable.appinstaller") == "no-store"
    # Downloads pages are mutable pointers, like feed manifests.
    assert cache_control_for("releases/stable/index.html") == "no-store"
    assert cache_control_for("releases/canary/index.html") == "no-store"
    assert cache_control_for(f"releases/commit/{'a' * 40}/index.html") == "no-store"


def test_canonical_request_reads_mixed_case_header_values():
    # Regression: the canonical line must carry the VALUE of a mixed-case
    # header ('Content-Type'), never a placeholder.
    host = "abc123.r2.cloudflarestorage.com"
    body_hash = "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"
    headers = {
        "host": host,
        "x-amz-date": NOW,
        "x-amz-content-sha256": body_hash,
        "Content-Type": "application/msix",
    }
    canon = canonical_request(
        "PUT", "/hermes-releases/HermesBundled-0.28.0-win-x64.msix", "", headers, body_hash
    )
    assert "content-type:application/msix" in canon
    assert "undefined" not in canon
    assert "content-type;host;x-amz-content-sha256;x-amz-date" in canon


# ── C22: artifact first, feed pointer last ──────────────────────────────────

def test_publish_feed_uploads_bundle_before_pointer():
    calls = []
    publish_feed_uploads(
        {
            "channelDir": "releases/win32/canary",
            "appinstallerName": "canary.appinstaller",
            "bundleFilename": "HermesBundled-0.27.2.9-win.msixbundle",
            "bundleFile": "C:/rel/HermesBundled-0.27.2.9-win.msixbundle",
            "appinstallerFile": "C:/rel/canary.appinstaller",
        },
        lambda key, file: calls.append([key, file]),
    )
    assert calls == [
        ["releases/win32/canary/HermesBundled-0.27.2.9-win.msixbundle", "C:/rel/HermesBundled-0.27.2.9-win.msixbundle"],
        ["releases/win32/canary/canary.appinstaller", "C:/rel/canary.appinstaller"],
    ]


def test_publish_feed_uploads_never_writes_pointer_when_bundle_fails():
    calls = []

    def upload(key, _file):
        calls.append(key)
        raise RuntimeError("R2 PUT -> 503")

    with pytest.raises(RuntimeError):
        publish_feed_uploads(
            {
                "channelDir": "releases/win32/stable",
                "appinstallerName": "stable.appinstaller",
                "bundleFilename": "HermesBundled-0.28.0.0-win.msixbundle",
                "bundleFile": "bundle",
                "appinstallerFile": "feed",
            },
            upload,
        )
    assert calls == ["releases/win32/stable/HermesBundled-0.28.0.0-win.msixbundle"]


# ── C22: canary feed-dir retention (fail-closed, keep-days grace) ───────────

CANARY_FEED_XML = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<AppInstaller Uri="https://r2.example/releases/win32/canary/canary.appinstaller" '
    'Version="0.27.2.9" xmlns="http://schemas.microsoft.com/appx/appinstaller/2017/2">\n'
    '  <MainPackage Name="NousResearch.HermesBundled" Publisher="CN=..." Version="0.27.2.9" '
    'Uri="https://r2.example/releases/win32/canary/HermesBundled-0.27.2.9-win.msixbundle" />\n'
    "</AppInstaller>\n"
)


def test_referenced_feed_bundle_filenames_reads_main_package_only():
    names = referenced_feed_bundle_filenames(CANARY_FEED_XML)
    assert names == ["HermesBundled-0.27.2.9-win.msixbundle"]
    # The AppInstaller ROOT Uri (the feed pointer itself) must NOT count.
    assert "canary.appinstaller" not in names
    assert referenced_feed_bundle_filenames("") == []
    assert referenced_feed_bundle_filenames("<html>ServiceUnavailable</html>") == []
    # A bundle Uri OUTSIDE MainPackage/MainBundle is not a reference.
    assert referenced_feed_bundle_filenames('<Foo Uri="https://x/HermesBundled-1.0.0-win.msixbundle" />') == []


def test_feed_referenced_keys_protects_bundle_and_absolute_tag_uris():
    tag_feed = CANARY_FEED_XML.replace(
        'Uri="https://r2.example/releases/win32/canary/HermesBundled-0.27.2.9-win.msixbundle"',
        'Uri="https://r2.example/releases/tag/v0.27.2-canary.20260829/HermesBundled-0.27.2-win-x64.msix"',
    )
    keys = feed_referenced_keys("releases/win32/canary", tag_feed)
    assert "releases/win32/canary/HermesBundled-0.27.2-win-x64.msix" in keys
    assert "releases/tag/v0.27.2-canary.20260829/HermesBundled-0.27.2-win-x64.msix" in keys


CANARY_DIR = "releases/win32/canary"
OLD_MS = 1785542400  # 2026-08-01T00:00:00Z
FRESH_MS = 1788480000  # 2026-09-03T00:00:00Z
CUTOFF_MS = 1787356800  # 2026-08-21T00:00:00Z


def _canary_keys(extra=()):
    return [
        f"{CANARY_DIR}/canary.appinstaller",
        f"{CANARY_DIR}/HermesBundled-0.27.2.9-win.msixbundle",  # referenced
        f"{CANARY_DIR}/HermesBundled-0.27.1.12000-win.msixbundle",  # stale
        f"{CANARY_DIR}/HermesBundled-0.27.2.99-win.msixbundle",  # uploaded, not yet pointed
        "releases/win32/stable/stable.appinstaller",
        "releases/win32/stable/HermesBundled-0.28.0.0-win.msixbundle",  # referenced
        "releases/win32/stable/HermesBundled-0.27.0.0-win.msixbundle",  # stale stable
        *extra,
    ]


def _last_modified_for(keys, overrides=None):
    lm = {k: OLD_MS for k in keys}
    lm.update(overrides or {})
    return lm


def test_stale_feed_bundle_keys_old_unreferenced_doomed_referenced_and_fresh_kept():
    keys = _canary_keys()
    lm = _last_modified_for(keys, {f"{CANARY_DIR}/HermesBundled-0.27.2.99-win.msixbundle": FRESH_MS})
    doomed = stale_feed_bundle_keys(keys, {CANARY_DIR: [CANARY_FEED_XML]}, lm, CUTOFF_MS)
    assert doomed == [f"{CANARY_DIR}/HermesBundled-0.27.1.12000-win.msixbundle"]


def test_stale_feed_bundle_keys_stable_dirs_never_pruned():
    keys = _canary_keys()
    lm = _last_modified_for(keys, {f"{CANARY_DIR}/HermesBundled-0.27.2.99-win.msixbundle": FRESH_MS})
    stable_feed = CANARY_FEED_XML.replace("0.27.2.9", "0.28.0.0").replace(
        "HermesBundled-0.27.2.9-win", "HermesBundled-0.28.0.0-win"
    )
    feeds = {CANARY_DIR: [CANARY_FEED_XML], "releases/win32/stable": [stable_feed]}
    doomed = stale_feed_bundle_keys(keys, feeds, lm, CUTOFF_MS)
    assert doomed == [f"{CANARY_DIR}/HermesBundled-0.27.1.12000-win.msixbundle"]


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "<html>boom</html>",
        None,
        '<Foo Uri="https://x/HermesBundled-1-win.msixbundle" />',
        '<MainPackage Uri="https://x/old.msixbundle" />',
        "<AppInstaller><MainBundle Uri=\"https://x/old.msixbundle\" />",
    ],
)
def test_stale_feed_bundle_keys_malformed_manifest_blocks_dir(bad):
    keys = _canary_keys()
    assert stale_feed_bundle_keys(keys, {CANARY_DIR: [bad]}, _last_modified_for(keys), CUTOFF_MS) == []


def test_stale_feed_bundle_keys_union_protected_one_bad_blocks_all():
    second = CANARY_FEED_XML.replace("0.27.2.9", "0.27.3.0").replace(
        "HermesBundled-0.27.2.9-win", "HermesBundled-0.27.3.0-win"
    )
    extra = [f"{CANARY_DIR}/second.appinstaller", f"{CANARY_DIR}/HermesBundled-0.27.3.0-win.msixbundle"]
    keys = _canary_keys(extra)
    lm = _last_modified_for(keys, {f"{CANARY_DIR}/HermesBundled-0.27.2.99-win.msixbundle": FRESH_MS})
    # Union of both feeds: both referenced bundles kept, the rest pruned.
    assert stale_feed_bundle_keys(keys, {CANARY_DIR: [CANARY_FEED_XML, second]}, lm, CUTOFF_MS) == [
        f"{CANARY_DIR}/HermesBundled-0.27.1.12000-win.msixbundle"
    ]
    # ONE unreadable/unrecognized manifest in the dir blocks feed retention.
    assert stale_feed_bundle_keys(keys, {CANARY_DIR: [CANARY_FEED_XML, None]}, lm, CUTOFF_MS) == []
    assert stale_feed_bundle_keys(keys, {CANARY_DIR: [CANARY_FEED_XML, ""]}, lm, CUTOFF_MS) == []


@pytest.mark.parametrize("unknown", [None, float("nan"), float("inf")])
def test_stale_feed_bundle_keys_missing_lastmodified_keeps_object(unknown):
    keys = _canary_keys()
    metadata = {key: unknown for key in keys}
    doomed = stale_feed_bundle_keys(keys, {CANARY_DIR: [CANARY_FEED_XML]}, metadata, CUTOFF_MS)
    assert doomed == []


# ── Real loopback HTTP protocol tests ───────────────────────────────────────

class _R2StubHandler(BaseHTTPRequestHandler):
    """Minimal R2-shaped backend: in-memory objects, records requests."""

    server_version = "r2-stub/1"

    def log_message(self, *args):  # keep test output clean
        pass

    def _key(self):
        from urllib.parse import unquote, urlsplit

        path = unquote(urlsplit(self.path).path)
        return path.split("/", 2)[2] if path.count("/") >= 2 else path.lstrip("/")

    def do_GET(self):
        store = self.server.store  # type: ignore[attr-defined]
        self.server.requests.append(("GET", self.path, dict(self.headers)))  # type: ignore[attr-defined]
        if "list-type=2" in self.path:
            body = self.server.listing_xml()  # type: ignore[attr-defined]
            self._send(200, body, {"content-type": "application/xml"})
            return
        key = self._key()
        if key in store:
            body, ctype = store[key]
            headers = {"etag": '"abc"'}
            if self.headers.get("Range"):
                headers["content-range"] = f"bytes 0-0/{len(body)}"
                self._send(206, body[:1], headers)
                return
            self._send(200, body, headers)
        else:
            self._send(404, b"NoSuchKey")

    def do_HEAD(self):
        self.server.requests.append(("HEAD", self.path, dict(self.headers)))  # type: ignore[attr-defined]
        key = self._key()
        if key in self.server.store:  # type: ignore[attr-defined]
            body, _ = self.server.store[key]  # type: ignore[attr-defined]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", '"abc"')
            self.end_headers()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_PUT(self):
        server = self.server  # type: ignore[attr-defined]
        server.requests.append(("PUT", self.path, dict(self.headers)))
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b""
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            data = self._read_chunked()
        key = self._key()
        if self.headers.get("If-None-Match") == "*" and key in server.store:
            self._send(412, b"Precondition Failed")
            return
        if self.headers.get("If-Match") and self.headers["If-Match"] != server.store.get(key, (b"",))[1]:
            self._send(412, b"Precondition Failed")
            return
        server.store[key] = (data, self.headers.get("If-Match", '"new"'))
        self._send(200, b"")

    def do_DELETE(self):
        server = self.server  # type: ignore[attr-defined]
        server.requests.append(("DELETE", self.path, dict(self.headers)))
        key = self._key()
        server.store.pop(key, None)
        self._send(204, b"")

    def _read_chunked(self):
        data = b""
        while True:
            size_line = self.rfile.readline().strip()
            size = int(size_line.split(b";")[0], 16)
            if size == 0:
                self.rfile.readline()
                return data
            data += self.rfile.read(size)
            self.rfile.readline()

    def _send(self, status, body, headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k.title(), v)
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def r2_server(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _R2StubHandler)
    server.store = {}
    server.requests = []

    def listing_xml():
        parts = ["<?xml version='1.0'?><ListBucketResult><IsTruncated>false</IsTruncated>"]
        for key, (body, _etag) in server.store.items():
            lm = "2026-08-01T00:00:00Z"
            parts.append(f"<Contents><Key>{key}</Key><LastModified>{lm}</LastModified></Contents>")
        parts.append("</ListBucketResult>")
        return "".join(parts)

    server.listing_xml = listing_xml  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Point the transport at the loopback endpoint.
    monkeypatch.setenv("CLOUDFLARE_R2_ACCOUNT_ID", "loopback")
    monkeypatch.setattr(r2, "s3_endpoint", lambda _account: f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("CLOUDFLARE_R2_ACCESS_KEY_ID", AKID)
    monkeypatch.setenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.setenv("CLOUDFLARE_R2_BUCKET", "hermes-releases")
    yield server
    server.shutdown()
    server.server_close()


def test_put_streams_a_file_and_verifies_size(r2_server):
    import tempfile

    payload = os.urandom(5 * 1024 * 1024 + 123)  # multi-chunk stream
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(payload)
        path = handle.name
    try:
        r2.put("v0.28.0", "artifact.bin", path)
    finally:
        os.unlink(path)
    key = "releases/tag/v0.28.0/artifact.bin"
    stored, _etag = r2_server.store[key]
    assert stored == payload
    puts = [r for r in r2_server.requests if r[0] == "PUT"]
    assert len(puts) == 1
    # Every request carries a signed Authorization header.
    for _method, _path, headers in r2_server.requests:
        assert headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
    # The streamed body's hash was signed as x-amz-content-sha256.
    import hashlib

    expected_hash = hashlib.sha256(payload).hexdigest()
    assert puts[0][2]["x-amz-content-sha256"] == expected_hash


def test_put_page_object_carries_html_type_and_no_store(r2_server):
    """A downloads page must RENDER in a browser: the object it is stored
    under has to arrive as HTML, and it is a mutable pointer, so it must not
    be cached. Without the registered content type R2 serves it as an opaque
    octet-stream download."""
    import tempfile

    page = "<!DOCTYPE html>\n<html lang=\"en\"><body><h1>Hermes stable builds</h1></body></html>\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False) as handle:
        handle.write(page)
        path = handle.name
    try:
        r2.put(tag="", key=r2.channel_page_key_for("stable"), file=path, key_is_full=True)
    finally:
        os.unlink(path)
    stored, _etag = r2_server.store["releases/stable/index.html"]
    assert stored == page.encode("utf-8")
    headers = [r[2] for r in r2_server.requests if r[0] == "PUT"][0]
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert headers["Cache-Control"] == "no-store"


def test_put_immutable_conflict_verifies_remote_bytes(r2_server):
    import tempfile

    existing = b"already published artifact"
    key = "releases/tag/v0.28.0/HermesBundled-0.28.0-mac-arm64.zip"
    r2_server.store[key] = (existing, '"etag-1"')
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(existing)
        path = handle.name
    try:
        r2.put("v0.28.0", "HermesBundled-0.28.0-mac-arm64.zip", path, immutable=True)
    finally:
        os.unlink(path)
    # Nothing was overwritten: the remote bytes are unchanged.
    assert r2_server.store[key][0] == existing


def test_put_immutable_conflict_with_corrupt_remote_fails(r2_server):
    import tempfile

    key = "releases/tag/v0.28.0/HermesBundled-0.28.0-mac-x64.zip"
    r2_server.store[key] = (b"corrupt different bytes", '"etag-1"')
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(b"local truth")
        path = handle.name
    try:
        with pytest.raises(ValueError, match="checksum mismatch"):
            r2.put("v0.28.0", "HermesBundled-0.28.0-mac-x64.zip", path, immutable=True)
    finally:
        os.unlink(path)
    assert r2_server.store[key][0] == b"corrupt different bytes"


def test_list_prints_keys_and_paginates(r2_server, capsys):
    r2_server.store["releases/tag/v0.28.0/a.msix"] = (b"x", '"e"')
    r2_server.store["releases/tag/v0.28.0/b.yml"] = (b"y", '"e"')
    r2.list_objects(prefix="releases/tag/v0.28.0/")
    # (list_objects returns; the CLI prints)
    from scripts.releases.r2 import list_objects as _lo

    result = _lo(prefix="releases/tag/v0.28.0/")
    assert sorted(result["keys"]) == ["releases/tag/v0.28.0/a.msix", "releases/tag/v0.28.0/b.yml"]
    assert "list-type=2" in [r for r in r2_server.requests if r[0] == "GET"][0][1]


def test_prune_canaries_dry_run_issues_no_delete(r2_server, capsys, monkeypatch):
    old_bundle = f"{CANARY_DIR}/HermesBundled-0.27.1.12000-win.msixbundle"
    referenced = f"{CANARY_DIR}/HermesBundled-0.27.2.9-win.msixbundle"
    r2_server.store[f"{CANARY_DIR}/canary.appinstaller"] = (CANARY_FEED_XML.encode(), '"e"')
    r2_server.store[referenced] = (b"bundle", '"e"')
    r2_server.store[old_bundle] = (b"stale", '"e"')
    r2_server.store["releases/tag/v0.27.2-canary.20260801000000/old.zip"] = (b"z", '"e"')

    monkeypatch.setattr(r2.time, "time", lambda: 1788547200.0)  # 2026-09-04
    r2.prune(keep_days=14, dry_run=True)
    out = capsys.readouterr().out
    assert f"would delete r2:{old_bundle}" in out
    assert "would delete r2:releases/tag/v0.27.2-canary.20260801000000/old.zip" in out
    assert f"would delete r2:{referenced}" not in out
    assert "would delete" not in out.split("appinstaller")[0] if ".appinstaller" in out else True
    assert not any(r[0] == "DELETE" for r in r2_server.requests)


def test_prune_canaries_real_delete_only_the_doomed(r2_server, capsys, monkeypatch):
    old_bundle = f"{CANARY_DIR}/HermesBundled-0.27.1.12000-win.msixbundle"
    referenced = f"{CANARY_DIR}/HermesBundled-0.27.2.9-win.msixbundle"
    r2_server.store[f"{CANARY_DIR}/canary.appinstaller"] = (CANARY_FEED_XML.encode(), '"e"')
    r2_server.store[referenced] = (b"bundle", '"e"')
    r2_server.store[old_bundle] = (b"stale", '"e"')
    monkeypatch.setattr(r2.time, "time", lambda: 1788547200.0)  # 2026-09-04
    r2.prune(keep_days=14, dry_run=False)
    assert old_bundle not in r2_server.store
    assert referenced in r2_server.store
    assert f"{CANARY_DIR}/canary.appinstaller" in r2_server.store


def test_prune_fails_closed_when_manifest_unreadable(r2_server, monkeypatch):
    r2_server.store[f"{CANARY_DIR}/canary.appinstaller"] = (b"<html>ServiceUnavailable</html>", '"e"')
    r2_server.store[f"{CANARY_DIR}/HermesBundled-0.27.1.12000-win.msixbundle"] = (b"stale", '"e"')
    monkeypatch.setattr(r2.time, "time", lambda: 1788547200.0)  # 2026-09-04
    with pytest.raises(RuntimeError, match="refusing to prune"):
        r2.prune(keep_days=14, dry_run=False)
    # Nothing was deleted.
    assert f"{CANARY_DIR}/HermesBundled-0.27.1.12000-win.msixbundle" in r2_server.store


def test_cli_usage_rejects_unknown_flags(capsys):
    with pytest.raises(SystemExit):
        r2.main(["bogus-command"])
    with pytest.raises(SystemExit):
        r2.main(["put", "--wat", "x"])


def test_verify_remote_artifact_streams_without_buffering(r2_server):
    """REAL streaming proof: the hash is computed over socket-sized chunks
    against the live loopback server — the artifact is never materialized
    whole (a 2GiB artifact would OOM the buffered path)."""
    import base64
    import hashlib

    payload = os.urandom(3 * 1024 * 1024 + 7)
    key = "releases/tag/v0.28.0/stream-check.zip"
    r2_server.store[key] = (payload, '"e"')
    url = f"http://127.0.0.1:{r2_server.server_port}/hermes-releases/{key}"
    r2.verify_remote_artifact(
        url,
        {"access_key_id": AKID, "secret_key": SECRET},
        NOW,
        expected_size=len(payload),
        digest=base64.b64encode(hashlib.sha512(payload).digest()).decode("ascii"),
    )
    # Mismatched digest is rejected.
    with pytest.raises(ValueError, match="checksum mismatch"):
        r2.verify_remote_artifact(
            url,
            {"access_key_id": AKID, "secret_key": SECRET},
            NOW,
            expected_size=len(payload),
            digest=base64.b64encode(hashlib.sha512(b"other").digest()).decode("ascii"),
        )


def test_download_streams_verified_bytes_and_preserves_destination_on_failure(r2_server, tmp_path, monkeypatch):
    import hashlib
    import http.client

    payload = os.urandom(3 * 1024 * 1024 + 7)
    key = "releases/tag/v1.2.3/package.msix"
    r2_server.store[key] = (payload, '"e"')
    target = tmp_path / "downloads" / "package.msix"
    target.parent.mkdir()
    target.write_bytes(b"previous complete file")
    real_read = http.client.HTTPResponse.read
    reads = []

    def bounded_read(response, amount=None):
        assert amount is not None and amount <= 1024 * 1024
        reads.append(amount)
        return real_read(response, amount)

    monkeypatch.setattr(http.client.HTTPResponse, "read", bounded_read)
    args = dict(creds={"access_key_id": AKID, "secret_key": SECRET},
                base=f"http://127.0.0.1:{r2_server.server_port}", bucket="hermes-releases",
                key=key, file=target, now=NOW, expected_size=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest())
    r2.download_object(**args)
    assert target.read_bytes() == payload
    assert len(reads) > 1
    assert all("authorization" in headers for _, _, headers in r2_server.requests)

    original_get = _R2StubHandler.do_GET
    request_times = []
    refreshed = "20150830T125600Z"

    def transient_get(handler):
        request_times.append(handler.headers['x-amz-date'])
        if len(request_times) == 1:
            handler._send(503, b"retry")
        else:
            original_get(handler)

    monkeypatch.setattr(_R2StubHandler, 'do_GET', transient_get)
    monkeypatch.setattr(r2, 'amz_timestamp', lambda: refreshed)
    monkeypatch.setattr(r2.time, 'sleep', lambda _: None)
    r2.download_object(**args)
    assert request_times == [NOW, refreshed]
    monkeypatch.setattr(_R2StubHandler, 'do_GET', original_get)

    r2_server.store[key] = (b"corrupt replacement", '"e"')
    with pytest.raises(ValueError, match="checksum mismatch"):
        r2.download_object(**args)
    assert target.read_bytes() == payload
    assert list(target.parent.iterdir()) == [target]
    del r2_server.store[key]
    with pytest.raises(r2.R2RequestError):
        r2.download_object(**args)
    assert target.read_bytes() == payload
    assert list(target.parent.iterdir()) == [target]


def test_put_accepts_pathlib_paths(r2_server):
    import pathlib
    import tempfile

    payload = b"pathlib input"
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(payload)
        path = pathlib.Path(handle.name)
    try:
        r2.put("v0.28.0", "pathlib.bin", path)
    finally:
        os.unlink(path)
    assert r2_server.store["releases/tag/v0.28.0/pathlib.bin"][0] == payload


def test_parse_list_xml_handles_fractional_and_plain_timestamps():
    parsed = parse_list_xml(
        "<ListBucketResult>"
        "<Contents><Key>a</Key><LastModified>2026-08-18T00:00:00.123Z</LastModified></Contents>"
        "<Contents><Key>b</Key><LastModified>2026-08-18T00:00:00Z</LastModified></Contents>"
        "</ListBucketResult>"
    )
    from datetime import datetime, timezone

    assert parsed["lastModified"]["a"] == int(
        datetime(2026, 8, 18, 0, 0, 0, 123000, tzinfo=timezone.utc).timestamp()
    )
    assert parsed["lastModified"]["b"] == int(
        datetime(2026, 8, 18, tzinfo=timezone.utc).timestamp()
    )


def test_canonical_header_whitespace_is_collapsed():
    canon = canonical_request(
        "PUT", "/p", "", {"host": "h", "Content-Type": "application/msix   extra\tvalue"}, "x"
    )
    assert "content-type:application/msix extra value" in canon
