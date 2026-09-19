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
import hashlib
import html
import http.client
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree as ET

import pytest

from scripts.releases import r2
from scripts.releases.r2 import (
    auth_header,
    canonical_query,
    canonical_request,
    channel_page_key_for,
    commit_page_key_for,
    commit_prefix_for,
    encode_key_path,
    feed_referenced_keys,
    public_base_url,
    public_url_for,
    referenced_feed_bundle_filenames,
    rfc3986_encode,
    stale_feed_bundle_keys,
)

from scripts.releases.r2_scope import R2Scope, channel_public_base

AKID = "AKIDEXAMPLE"
SECRET = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
NOW = "20150830T123600Z"
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _auth(**kwargs):
    kwargs.setdefault("access_key_id", AKID)
    kwargs.setdefault("secret_key", SECRET)
    kwargs.setdefault("now", NOW)
    return auth_header(**kwargs)


def test_disposable_scope_streams_lists_and_never_touches_production(r2_server, tmp_path, monkeypatch):
    monkeypatch.setenv("R2_DISPOSABLE_RUN", "98765-1")
    monkeypatch.setenv("GITHUB_REPOSITORY_ID", "12345")
    scope = R2Scope.configured()
    root = f"http://127.0.0.1:{r2_server.server_port}/hermes-releases"
    monkeypatch.setenv("CLOUDFLARE_R2_PUBLIC_URL", root)
    key = "releases/channel-builds/" + "a" * 32 + "/payload.bin"
    r2_server.store[key] = (b"production sentinel", '"production"')
    path = tmp_path / "payload.bin"
    payload = b"x" * (5 * 1024 * 1024 + 1)
    path.write_bytes(payload)
    creds, base, bucket = r2.credentials()
    r2.put_object(creds, base, bucket, key, path, NOW, "application/octet-stream",
                  {"If-None-Match": "*"}, multipart_part_size=5 * 1024 * 1024)
    # Same immutable bytes are reusable, but only inside this run's prefix.
    r2.put_object(creds, base, bucket, key, path, NOW, None, {"If-None-Match": "*"})
    assert r2.get_object(creds, base, bucket, key, NOW) == payload.decode()
    digest = hashlib.sha256(payload).hexdigest()
    r2.download_object(creds, base, bucket, key, tmp_path / "signed", NOW,
                       expected_size=len(payload), expected_sha256=digest)
    r2.download_public_object(channel_public_base(), key, tmp_path / "public",
                              expected_size=len(payload), expected_sha256=digest)
    assert (tmp_path / "signed").read_bytes() == (tmp_path / "public").read_bytes() == payload
    assert r2.list_objects(prefix="releases/")["keys"] == [key]
    assert r2_server.store[key][0] == b"production sentinel"
    assert r2_server.store[scope.key(key)][0] == payload
    for method, url, headers in r2_server.requests:
        parsed = urlsplit(url)
        if parsed.query and "list-type" in parsed.query:
            assert parsed.path == "/hermes-releases"
            assert parse_qs(parsed.query)["prefix"] == [scope.prefix + "releases/"]
        else:
            assert parsed.path.startswith("/hermes-releases/" + scope.prefix)
        if method != "GET" or headers.get("authorization"):
            assert headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
    before = list(r2_server.requests)
    for bad in ("../escape", "/releases/x", "releases/%2e%2e/x", "releases//x"):
        with pytest.raises(ValueError):
            r2.put_object(creds, base, bucket, bad, b"x", NOW, None)
    with pytest.raises(ValueError, match="bucket"):
        r2.list_objects(creds=creds, base=base, bucket=bucket + "/ci-disposable")
    with pytest.raises(ValueError, match="escaped"):
        scope.logical_key("releases/channels/stable.json")
    for bad in (root, "https://elsewhere.example", root + "/ci-disposable/999/98765-1"):
        with pytest.raises(ValueError, match="authority"):
            channel_public_base(bad)
    monkeypatch.setenv("R2_DISPOSABLE_RUN", "../production")
    with pytest.raises(ValueError):
        r2.put_object(creds, base, bucket, key, b"x", NOW, None)
    monkeypatch.delenv("R2_DISPOSABLE_RUN")
    # Scoping is opt-in via R2_DISPOSABLE_RUN alone: there is no fork isolation
    # to assert after the fork-conditional dispatch was dropped.
    assert r2_server.requests == before
    from scripts.releases.channel_disposable import probe
    from scripts.releases.channels import ChannelPublisher, R2ChannelStore
    unscoped = ChannelPublisher(R2ChannelStore(creds, base, bucket), "fixture/fork", root,
                                authorize=lambda *_: None)
    with pytest.raises(ValueError, match="disposable"):
        probe(unscoped, "a" * 40, "1.2.3", "b" * 40)
    assert r2_server.requests == before


# ── SigV4 vectors ───────────────────────────────────────────────────────────

@pytest.mark.parametrize('method,path,query,payload,scope,signature', [
    ('GET', '/', {}, EMPTY_SHA, 'us-east-1/service',
     '33399fd3d4a9d6104710c7c04005f7c959f8b1f8bf41b823587ed36b079e453f'),
    ('PUT', '/hermes-releases/HermesBundled-0.28.0-win-x64.msix', {},
     '44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072', 'auto/s3',
     '05ba50acfb54042fac330848af50877e5fb477c4f2063c2f77f9cc80855eb1e9'),
    ('GET', '/hermes-releases', {'list-type': '2', 'prefix': 'HermesBundled-0.28.0-', 'max-keys': '1000'},
     EMPTY_SHA, 'auto/s3', '3ec423c452a318664c85fbcc25667ad07201aedce688e3bb6b345b4baaa39d90'),
    ('DELETE', '/hermes-releases/HermesBundled-0.28.0-canary.20260818-win-arm64.msix', {},
     EMPTY_SHA, 'auto/s3', 'ec5ccb76f701193b28aaca052cabbf2f084e9c71ffc64b872fa51d4e70dc6e55'),
])
def test_independent_sigv4_vectors(method, path, query, payload, scope, signature):
    host = 'example.com' if scope == 'us-east-1/service' else 'abc123.r2.cloudflarestorage.com'
    headers = {'host': host, 'x-amz-date': NOW}
    signed = 'host;x-amz-date'
    if scope == 'auto/s3':
        headers['x-amz-content-sha256'] = payload
        signed = 'host;x-amz-content-sha256;x-amz-date'
    encoded = canonical_query(query)
    if query:
        assert encoded == 'list-type=2&max-keys=1000&prefix=HermesBundled-0.28.0-'
    region, service = scope.split('/')
    assert _auth(method=method, host=host, path=path, query=encoded, headers=headers,
                 payload_hash=payload, region=region, service=service) == (
        f'AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/{scope}/aws4_request, '
        f'SignedHeaders={signed}, Signature={signature}')


# ── Encoding / layout helpers ───────────────────────────────────────────────

def test_rfc3986_encode_escapes_the_aws_reserved_set_keeps_unreserved():
    assert rfc3986_encode("HermesBundled-0.28.0-win-x64.msix") == "HermesBundled-0.28.0-win-x64.msix"
    assert rfc3986_encode("a b!'()*c") == "a%20b%21%27%28%29%2Ac"


def test_encode_key_path_encodes_segment_wise_preserves_separators():
    assert encode_key_path("HermesBundled-0.28.0-win-x64.msix") == "HermesBundled-0.28.0-win-x64.msix"
    assert encode_key_path("a b/c d") == "a%20b/c%20d"



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


def test_channel_public_base_defaults_to_production(monkeypatch):
    # Unscoped channel administration names the same documented production
    # origin the commit-build path falls back to; the public URL is not a
    # secret, so a local command should not have to hand-set it.
    monkeypatch.delenv("CLOUDFLARE_R2_PUBLIC_URL", raising=False)
    monkeypatch.delenv("R2_DISPOSABLE_RUN", raising=False)
    assert channel_public_base() == "https://hermes-assets.nousresearch.com"
    monkeypatch.setenv("CLOUDFLARE_R2_PUBLIC_URL", "https://cdn.example.com")
    assert channel_public_base() == "https://cdn.example.com"
    assert channel_public_base("https://explicit.example.com/") == "https://explicit.example.com"


@pytest.mark.parametrize('key,content_type,immutable', [
    ('releases/tag/v1.2.3/app.msix', 'application/msix', None),
    ('releases/tag/v1.2.3/app.msixbundle', 'application/msixbundle', None),
    ('releases/win32/stable/X.APPINSTALLER', 'application/appinstaller', None),
    ('releases/win32/stable/stable.appinstaller', 'application/appinstaller', False),
    ('releases/stable/INDEX.HTML', 'text/html; charset=utf-8', False),
    ('releases/canary/index.html', 'text/html; charset=utf-8', False),
    (f'releases/commit/{"a" * 40}/index.html', 'text/html; charset=utf-8', False),
    ('releases/tag/v1.2.3/app.dmg', None, None),
    ('releases/tag/v1.2.3/latest-mac.yml', None, None),
    ('releases/termux/canary/key.asc', 'text/plain', False),
    ('releases/termux/canary/dists/hermes-canary/InRelease', 'text/plain', False),
    ('releases/termux/canary/dists/hermes-canary/Release', 'text/plain', False),
    ('releases/termux/canary/dists/hermes-canary/main/binary-aarch64/Packages.gz', 'application/gzip', False),
    ('releases/termux/canary/dists/hermes-canary/main/binary-aarch64/by-hash/SHA256/abcd', None, True),
    ('releases/termux/canary/pool/h/package.deb', 'application/vnd.debian.binary-package', True),
])
def test_object_headers_on_real_upload(tmp_path, r2_server, key, content_type, immutable):
    file = tmp_path / 'content'
    file.write_bytes(b'header transport fixture')
    r2.put(tag='', key=key, file=file, key_is_full=True)
    assert r2_server.store[key][0] == file.read_bytes()
    headers = next(headers for method, _, headers in r2_server.requests if method == 'PUT')
    assert headers.get('Content-Type') == content_type
    assert headers.get('Cache-Control') == {None: None, False: 'no-store', True: 'public, max-age=31536000, immutable'}[immutable]


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
CUTOFF_MS = 1787356800  # 2026-08-21T00:00:00Z


@pytest.mark.parametrize("unknown", [None, float("nan"), float("inf")])
def test_stale_feed_bundle_keys_missing_lastmodified_keeps_object(unknown):
    keys = [f'{CANARY_DIR}/unreferenced.msixbundle']
    metadata = {key: unknown for key in keys}
    doomed = stale_feed_bundle_keys(keys, {CANARY_DIR: [CANARY_FEED_XML]}, metadata, CUTOFF_MS)
    assert doomed == []
    assert stale_feed_bundle_keys(keys, {CANARY_DIR: [None]}, {keys[0]: 0}, CUTOFF_MS) == []


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
            body = self.server.listing_xml(parse_qs(urlsplit(self.path).query))  # type: ignore[attr-defined]
            self._send(200, body, {"content-type": "application/xml"})
            return
        key = self._key()
        if key in store:
            body, etag = store[key]
            headers = {"etag": etag}
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
        query = parse_qs(urlsplit(self.path).query)
        if key == getattr(server, 'fail_put', None):
            return self._send(400, b'Interrupted upload')
        if "partNumber" in query:
            state = server.multipart
            assert query["uploadId"] == ["a+/="]
            assert self.headers["x-amz-content-sha256"] == hashlib.sha256(data).hexdigest()
            number = int(query["partNumber"][0])
            state.attempts[number] = state.attempts.get(number, 0) + 1
            if state.failure == "part":
                return self._send(400, b"InvalidRequest")
            if number == 2 and state.attempts[number] == 1:
                return self._send(503, b"")
            etag = '"' + hashlib.md5(data).hexdigest() + '"'
            state.parts[number] = (data, etag)
            return self._send(200, b"", {"ETag": etag})
        if self.headers.get("If-None-Match") == "*" and key in server.store:
            self._send(412, b"Precondition Failed")
            return
        if self.headers.get('If-Match') and key == getattr(server, 'race_key', None):
            server.store[key] = (server.store[key][0], '"raced"')
        if self.headers.get("If-Match") and self.headers["If-Match"] != server.store.get(key, (b"", None))[1]:
            self._send(412, b"Precondition Failed")
            return
        server.store[key] = (data, self.headers.get("If-Match", '"new"'))
        if key == getattr(server, 'corrupt_put', None):
            # Keep HEAD size verification valid; only the readback content lies.
            server.store[key] = (b'x' * len(data), '"corrupt"')
        self._send(200, b"")

    def do_POST(self):
        server, state = self.server, self.server.multipart
        server.requests.append(("POST", self.path, dict(self.headers)))
        query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        assert self.headers["x-amz-content-sha256"] == hashlib.sha256(body).hexdigest()
        key = self._key()
        if "uploads" in query:
            state.created += 1
            state.parts = {}
            if state.reject_existing and key in server.store:
                return self._send(412, b"")
            return self._send(200, b'<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><UploadId>a+/=</UploadId></InitiateMultipartUploadResult>')
        assert query["uploadId"] == ["a+/="]
        state.completed += 1
        if state.failure == "complete" or (state.failure == "transient" and state.completed == 1):
            code = "InvalidPart" if state.failure == "complete" else "InternalError"
            return self._send(200, f"<Error><Code>{code}</Code></Error>")
        if state.failure == "disconnect" and state.completed == 1:
            self.close_connection = True
            return
        if self.headers.get("If-None-Match") == "*" and key in server.store:
            return self._send(412, b"")
        parts = ET.fromstring(body).findall("{*}Part")
        numbers = [int(part.findtext("{*}PartNumber", "0")) for part in parts]
        assert numbers == sorted(state.parts)
        assert [part.findtext("{*}ETag") for part in parts] == [state.parts[n][1] for n in numbers]
        server.store[key] = (b"".join(state.parts[n][0] for n in numbers), '"multipart"')
        if state.failure == "lost-response" and state.completed == 1:
            self.close_connection = True
            return
        self._send(200, b'<CompleteMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><ETag>done</ETag></CompleteMultipartUploadResult>')

    def do_DELETE(self):
        server = self.server  # type: ignore[attr-defined]
        server.requests.append(("DELETE", self.path, dict(self.headers)))
        if "uploadId=" in self.path:
            server.multipart.aborted += 1
            return self._send(204, b"")
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
    server.page_size = 1000
    server.last_modified = {}
    server.multipart = SimpleNamespace(parts={}, attempts={}, failure=None, created=0,
                                       completed=0, aborted=0, reject_existing=False)

    def listing_xml(query):
        keys = sorted(key for key in server.store if key.startswith(query.get('prefix', [''])[0]))
        token = query.get('continuation-token', ['0'])[0]
        start = int(token.removeprefix('page+/='))
        end = start + server.page_size
        truncated = end < len(keys)
        parts = [f"<ListBucketResult><IsTruncated>{str(truncated).lower()}</IsTruncated>"]
        for key in keys[start:end]:
            lm = server.last_modified.get(key, '2026-08-01T00:00:00Z')
            modified = f'<LastModified>{lm}</LastModified>' if lm is not None else ''
            parts.append(f'<Contents><Key>{html.escape(key)}</Key>{modified}</Contents>')
        if truncated:
            parts.append(f'<NextContinuationToken>page+/={end}</NextContinuationToken>')
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
    thread.join(timeout=5)


@pytest.mark.parametrize("failure", [None, "part", "complete", "transient", "disconnect", "lost-response"])
def test_multipart_publication_is_atomic_and_retryable(r2_server, tmp_path, failure):
    path = tmp_path / "bundle.msixbundle"
    payload = b"a" * (5 * 1024 * 1024) + b"b" * (5 * 1024 * 1024) + b"last"
    path.write_bytes(payload)
    state = r2_server.multipart
    state.failure, state.reject_existing = failure, failure is not None
    key = "releases/stable/bundle.msixbundle"
    creds, base, bucket = r2.credentials()

    def upload():
        r2.put_object(creds, base, bucket, key, path, r2.amz_timestamp(),
                      "application/msixbundle", {"If-None-Match": "*"}, multipart_part_size=5 * 1024 * 1024)

    if failure in {"part", "complete"}:
        with pytest.raises(r2.R2RequestError):
            upload()
        assert key not in r2_server.store and state.aborted == 1
        assert not any(method == "HEAD" for method, _, _ in r2_server.requests)
        return
    upload()
    assert r2_server.store[key][0] == payload
    assert state.attempts[2] >= 2
    assert state.created == (2 if failure else 1)
    for method, _, headers in r2_server.requests:
        assert headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
        if method == "POST":
            assert headers["If-None-Match"] == "*"
    metadata = next(headers for method, url, headers in r2_server.requests if "uploads=" in url)
    assert metadata["Content-Type"] == "application/msixbundle" and metadata["Cache-Control"] == "no-store"
    upload()
    path.write_bytes(payload[:-1] + b"!")
    with pytest.raises(ValueError, match="checksum mismatch"):
        upload()
    assert r2_server.store[key][0] == payload


def test_cli_reuses_an_immutable_multipart_object(r2_server, tmp_path):
    path = tmp_path / "bundle.msixbundle"
    payload = b"x" * (64 * 1024 * 1024 + 1)
    path.write_bytes(payload)
    r2_server.store["releases/tag/v1.0.0/bundle.msixbundle"] = (payload, '"existing"')
    r2_server.multipart.reject_existing = True
    # Redirect only sockets: CLI and sibling imports must keep one exception identity.
    script = (
        "import http.client, runpy, sys\n"
        f"http.client.HTTPSConnection = lambda *a, **k: http.client.HTTPConnection('127.0.0.1', {r2_server.server_port})\n"
        "sys.argv = ['r2', *sys.argv[1:]]\n"
        "runpy.run_module('scripts.releases.r2', run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, "put", "--tag", "v1.0.0", "--key", path.name, "--file", str(path), "--immutable"],
        cwd=Path(__file__).resolve().parents[2], env=os.environ, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert [method for method, _, _ in r2_server.requests] == ["POST", "GET", "HEAD"]


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


def test_list_cli_paginates_and_decodes_keys(r2_server, capsys):
    r2_server.page_size = 1
    keys = ['releases/tag/v1.2.3/a&b.msix', 'releases/tag/v1.2.3/nested space/b.yml']
    r2_server.store.update({key: (b'x', '"e"') for key in keys})
    r2_server.store['outside.bin'] = (b'outside', '"e"')
    r2_server.last_modified[keys[0]] = '2026-08-18T00:00:00.123Z'
    result = r2.list_objects(prefix='releases/tag/v1.2.3/')
    assert result == {'keys': keys, 'lastModified': {keys[0]: 1787011200, keys[1]: 1785542400}}
    queries = [parse_qs(urlsplit(path).query) for method, path, _ in r2_server.requests if method == 'GET']
    assert len(queries) == 2 and queries[1]['continuation-token'] == ['page+/=1']
    r2.main(['list', '--prefix', 'releases/tag/v1.2.3/'])
    assert capsys.readouterr().out.splitlines() == keys


@pytest.mark.parametrize('bad_feed', [None, b'', b'<html>boom</html>',
    b'<Foo Uri="https://x/old.msixbundle" />', b'<MainPackage Uri="https://x/old.msixbundle" />',
    b'<AppInstaller><MainBundle Uri="https://x/old.msixbundle" />'])
def test_real_prune_retention_and_second_feed_failure(r2_server, capsys, bad_feed):
    tag_reference = 'releases/tag/v0.27.2-canary.20260801/live.msixbundle'
    second = f'<AppInstaller><MainBundle Uri="https://cdn.example/{tag_reference}" /></AppInstaller>'
    doomed = {f'{CANARY_DIR}/old.msixbundle', 'releases/tag/v0.27.2-canary.20260801000000/old.zip'}
    kept = {f'{CANARY_DIR}/HermesBundled-0.27.2.9-win.msixbundle', f'{CANARY_DIR}/live.msixbundle',
            f'{CANARY_DIR}/fresh.msixbundle', f'{CANARY_DIR}/unknown.msixbundle', tag_reference,
            'releases/win32/stable/old.msixbundle', 'releases/unknown/old.msixbundle',
            'releases/tag/v0.28.0/old.zip', 'releases/tag/v0.28.0-canary.20260904/today.zip'}
    r2_server.store.update({key: (b'artifact', '"e"') for key in doomed | kept})
    r2_server.store[f'{CANARY_DIR}/canary.appinstaller'] = (CANARY_FEED_XML.encode(), '"e"')
    r2_server.store[f'{CANARY_DIR}/second.appinstaller'] = (second.encode() if bad_feed is None else bad_feed, '"e"')
    r2_server.store['releases/win32/stable/stable.appinstaller'] = (CANARY_FEED_XML.encode(), '"e"')
    r2_server.last_modified[f'{CANARY_DIR}/fresh.msixbundle'] = '2026-09-03T00:00:00Z'
    r2_server.last_modified[f'{CANARY_DIR}/unknown.msixbundle'] = None
    r2_server.page_size = 3
    before = dict(r2_server.store)
    if bad_feed is not None:
        with pytest.raises(RuntimeError, match='refusing to prune'):
            r2.prune(keep_days=14, now_epoch=1788547200)
        assert r2_server.store == before
        assert not any(method == 'DELETE' for method, _, _ in r2_server.requests)
        return
    r2.prune(keep_days=14, dry_run=True, now_epoch=1788547200)
    assert {line.removeprefix('(dry-run) would delete r2:') for line in capsys.readouterr().out.splitlines()} == doomed
    assert r2_server.store == before and not any(method == 'DELETE' for method, _, _ in r2_server.requests)
    r2.prune(keep_days=14, now_epoch=1788547200)
    assert set(r2_server.store) == set(before) - doomed


def test_cli_usage_rejects_unknown_flags(capsys):
    with pytest.raises(SystemExit):
        r2.main(["bogus-command"])
    with pytest.raises(SystemExit):
        r2.main(["put", "--wat", "x"])


@pytest.fixture
def bounded_reads(monkeypatch):
    real_read = http.client.HTTPResponse.read
    reads = []

    def bounded_read(response, amount=None):
        assert amount is not None and 0 < amount <= 1024 * 1024
        reads.append(amount)
        return real_read(response, amount)

    monkeypatch.setattr(http.client.HTTPResponse, 'read', bounded_read)
    return reads


def test_verify_remote_artifact_streams_without_buffering(r2_server, bounded_reads):
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
    assert len(bounded_reads) > 1
    # Mismatched digest is rejected.
    with pytest.raises(ValueError, match="checksum mismatch"):
        r2.verify_remote_artifact(
            url,
            {"access_key_id": AKID, "secret_key": SECRET},
            NOW,
            expected_size=len(payload),
            digest=base64.b64encode(hashlib.sha512(b"other").digest()).decode("ascii"),
        )


def test_download_streams_verified_bytes_and_preserves_destination_on_failure(r2_server, tmp_path, monkeypatch, bounded_reads):
    import hashlib
    import http.client

    payload = os.urandom(3 * 1024 * 1024 + 7)
    key = "releases/tag/v1.2.3/package.msix"
    r2_server.store[key] = (payload, '"e"')
    target = tmp_path / "downloads" / "package.msix"
    target.parent.mkdir()
    target.write_bytes(b"previous complete file")

    args = dict(creds={"access_key_id": AKID, "secret_key": SECRET},
                base=f"http://127.0.0.1:{r2_server.server_port}", bucket="hermes-releases",
                key=key, file=target, now=NOW, expected_size=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest())
    r2.download_object(**args)
    assert target.read_bytes() == payload
    assert len(bounded_reads) > 1
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



def test_canonical_header_whitespace_is_collapsed():
    canon = canonical_request(
        "PUT", "/p", "", {"host": "h", "Content-Type": "application/msix   extra\tvalue"}, "x"
    )
    assert "content-type:application/msix extra value" in canon
