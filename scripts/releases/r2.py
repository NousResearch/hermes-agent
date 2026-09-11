"""Stream and verify release objects through the signed R2 transport."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import hmac
import http.client
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Iterable
from urllib.parse import quote, urlparse

REGION = "auto"
SERVICE = "s3"
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

_CONTENT_TYPES_PATH = Path(__file__).resolve().parents[1] / "release-content-types.json"
with _CONTENT_TYPES_PATH.open(encoding="utf-8") as _file:
    _CONTENT_TYPES = tuple(json.load(_file).items())

def content_type_for(filename: str) -> str | None:
    lower = filename.lower()
    base = lower[lower.rfind("/") + 1:]
    for key, mime in _CONTENT_TYPES:
        if "." in key:
            if lower.endswith(key):
                return mime
        elif base == key:
            return mime
    return None


# ---------------------------------------------------------------------------
# SigV4 (pure; tests pin these against botocore-generated vectors)
# ---------------------------------------------------------------------------

_UNRESERVED = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~"
)


def rfc3986_encode(value: str) -> str:
    """RFC3986 encode: everything except unreserved [A-Za-z0-9-_.~]."""
    out: list[str] = []
    for char in value:
        if char in _UNRESERVED:
            out.append(char)
        else:
            out.extend(f"%{b:02X}" for b in char.encode("utf-8"))
    return "".join(out)


def canonical_query(params: dict[str, str]) -> str:
    """Canonical query string: params sorted by encoded key (then value)."""
    pairs = sorted(
        (rfc3986_encode(k), rfc3986_encode(str(v))) for k, v in params.items()
    )
    return "&".join(f"{k}={v}" for k, v in pairs)


def _header_value(headers: dict[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name:
            return str(value)
    raise KeyError(name)


def canonical_request(
    method: str,
    path: str,
    query: str,
    headers: dict[str, str],
    payload_hash: str,
) -> str:
    """Canonical request for one S3-style request. Header names are
    canonicalized lowercase + sorted; values are read case-insensitively
    (the mixed-case 'Content-Type' regression is pinned by a test)."""
    names = sorted(k.lower() for k in headers)
    whitespace = re.compile(r"\s+")

    def _collapsed(name: str) -> str:
        return whitespace.sub(" ", _header_value(headers, name).strip())

    canonical_headers = "\n".join(f"{n}:{_collapsed(n)}" for n in names)
    return "\n".join(
        [method, path, query, canonical_headers, "", ";".join(names), payload_hash]
    )


def string_to_sign(canonical: str, date: str, scope: str) -> str:
    sts_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return "\n".join(["AWS4-HMAC-SHA256", date, scope, sts_hash])


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def signature(sts_text: str, secret_key: str, date: str, region: str, service: str) -> str:
    k_date = _hmac(f"AWS4{secret_key}".encode("utf-8"), date)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, service)
    k_signing = _hmac(k_service, "aws4_request")
    return hmac.new(k_signing, sts_text.encode("utf-8"), hashlib.sha256).hexdigest()


def auth_header(
    *,
    method: str,
    host: str,
    path: str,
    query: str,
    headers: dict[str, str],
    payload_hash: str,
    access_key_id: str,
    secret_key: str,
    now: str,
    region: str = REGION,
    service: str = SERVICE,
) -> str:
    """Full AWS4-HMAC-SHA256 Authorization header value for one request."""
    date = now[:8]
    canonical = canonical_request(method, path, query, headers, payload_hash)
    sts = string_to_sign(canonical, now, f"{date}/{region}/{service}/aws4_request")
    sig = signature(sts, secret_key, date, region, service)
    signed_headers = ";".join(sorted(k.lower() for k in headers))
    return (
        f"AWS4-HMAC-SHA256 Credential={access_key_id}/{date}/{region}/{service}/aws4_request, "
        f"SignedHeaders={signed_headers}, Signature={sig}"
    )


# ---------------------------------------------------------------------------
# R2 request plumbing
# ---------------------------------------------------------------------------

def s3_endpoint(account_id: str) -> str:
    return f"https://{account_id}.r2.cloudflarestorage.com"


def encode_key_path(key: str) -> str:
    """Encode an object key into the URI path, segment by segment."""
    return "/".join(rfc3986_encode(seg) for seg in key.split("/"))


def amz_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class R2RequestError(Exception):
    def __init__(self, method: str, path: str, status: int, body: str = ""):
        self.method, self.path, self.status, self.body = method, path, status, body
        super().__init__(f"R2 {method} {path} -> {status}" + (f": {body[:300]}" if body else ""))


class Response:
    def __init__(self, status: int, headers: list[tuple[str, str]], body: bytes):
        self.status = status
        self.headers = {k.lower(): v for k, v in headers}
        self._body = body

    def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())


def r2_headers(
    method: str,
    host: str,
    path: str,
    query: str,
    body_hash: str,
    now: str,
    creds: dict[str, str],
    content_type: str | None = None,
    content_length: int | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = {
        "host": host,
        "x-amz-date": now,
        "x-amz-content-sha256": body_hash,
    }
    # Content-Type / Content-Length / extras (Range, If-None-Match, ...)
    # join BEFORE signing so they land in SignedHeaders like everything else.
    if content_type:
        headers["Content-Type"] = content_type
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    if extra_headers:
        headers.update(extra_headers)
    headers["authorization"] = auth_header(
        method=method, host=host, path=path, query=query, headers=headers,
        payload_hash=body_hash, access_key_id=creds["access_key_id"],
        secret_key=creds["secret_key"], now=now,
    )
    return headers


def _connection(url: str, timeout: float) -> http.client.HTTPConnection:
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if parsed.scheme == "https":
        import ssl
        return http.client.HTTPSConnection(
            parsed.hostname, port, context=ssl.create_default_context(), timeout=timeout
        )
    return http.client.HTTPConnection(parsed.hostname, port, timeout=timeout)


_RETRYABLE_STATUSES = {500, 502, 503, 504}


def signed_request(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    body_iter_factory: Callable[[], Iterable[bytes]] | None = None,
    body_hash: str = EMPTY_SHA,
    content_length: int | None = None,
    creds: dict[str, str],
    now: str,
    content_type: str | None = None,
    extra_headers: dict[str, str] | None = None,
    tries: int = 3,
) -> Response:
    """Signed request with retries. Path payloads get a FRESH stream per
    attempt (a consumed iterator cannot be replayed). Raises R2RequestError
    on a final non-2xx response."""
    parsed = urlparse(url)
    query = parsed.query
    last: Response | None = None
    for attempt in range(1, tries + 1):
        body_iter = body_iter_factory() if body_iter_factory is not None else None
        headers = r2_headers(
            method, parsed.netloc, parsed.path, query, body_hash, now, creds,
            content_type=content_type, content_length=content_length,
            extra_headers=extra_headers,
        )
        conn = _connection(url, timeout=600.0)
        try:
            conn.request(
                method,
                parsed.path + ("?" + query if query else ""),
                body=body_iter if body_iter is not None else body,
                headers=headers,
            )
            res = conn.getresponse()
            data = res.read()
        except (OSError, http.client.HTTPException):
            if attempt == tries:
                raise
            time.sleep(float(attempt))
            continue
        finally:
            conn.close()
            if body_iter is not None and hasattr(body_iter, "close"):
                body_iter.close()
        response = Response(res.status, res.getheaders(), data)
        if 200 <= response.status < 300:
            return response
        last = response
        if response.status in _RETRYABLE_STATUSES and attempt < tries:
            time.sleep(1.0 * attempt)
            continue
        raise R2RequestError(method, parsed.path, response.status, response.text())
    raise R2RequestError(method, parsed.path, last.status if last else 0)


def stream_file(path: str, chunk_size: int = 1024 * 1024) -> Iterable[bytes]:
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            yield chunk


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    for chunk in stream_file(path):
        digest.update(chunk)
    return digest.hexdigest()


def verify_remote_artifact(
    url: str,
    creds: dict[str, str],
    now: str,
    expected_size: int,
    digest: str,
    algorithm: str = "sha512",
    encoding: str = "base64",
    fetcher: Callable[[str], Response] | None = None,
) -> None:
    """STREAMED GET (release artifacts are ~2GiB — never buffered whole);
    size + digest must both match or the artifact is corrupt."""
    if fetcher is not None:
        response = fetcher(url)
        if response.status >= 400:
            raise R2RequestError("GET", urlparse(url).path, response.status)
        data = response._body  # noqa: SLF001 — loopback-test responses are small
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data)
        size = len(data)
    else:
        hash_obj, size = _stream_and_hash(url, creds, now, algorithm)
    actual = (
        __import__("base64").b64encode(hash_obj.digest()).decode("ascii")
        if encoding == "base64"
        else hash_obj.hexdigest()
    )
    if size != expected_size or actual != digest:
        raise ValueError(f"Artifact checksum mismatch: {urlparse(url).path}")


def download_object(
    creds: dict[str, str], base: str, bucket: str, key: str, file: Path, now: str,
    *, expected_size: int, expected_sha256: str,
) -> None:
    """Publish a download locally only after its exact receipt matches."""
    import tempfile

    if type(expected_size) is not int or expected_size < 0 or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("Invalid artifact size or SHA256")
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base}/{bucket}/{encode_key_path(key)}"
    parsed = urlparse(url)
    for attempt in range(1, 4):
        headers = r2_headers("GET", parsed.netloc, parsed.path, "", EMPTY_SHA,
                             now if attempt == 1 else amz_timestamp(), creds)
        conn = _connection(url, timeout=600.0)
        temporary = None
        try:
            conn.request("GET", parsed.path, headers=headers)
            response = conn.getresponse()
            if response.status != 200:
                raise R2RequestError("GET", parsed.path, response.status)
            digest, size = hashlib.sha256(), 0
            with tempfile.NamedTemporaryFile(dir=file.parent, prefix=f".{file.name}.", delete=False) as output:
                temporary = Path(output.name)
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    output.write(chunk)
            if size != expected_size or digest.hexdigest() != expected_sha256:
                raise ValueError(f"Artifact checksum mismatch: {key}")
            temporary.replace(file)
            return
        except R2RequestError as error:
            if error.status not in _RETRYABLE_STATUSES or attempt == 3:
                raise
        except (OSError, http.client.HTTPException):
            if attempt == 3:
                raise
        finally:
            conn.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        time.sleep(float(attempt))


def _stream_and_hash(url: str, creds: dict[str, str], now: str, algorithm: str):
    """True streaming GET: hash + count bytes as chunks arrive off the
    socket, never materializing the body in memory."""
    parsed = urlparse(url)
    headers = r2_headers(
        "GET", parsed.netloc, parsed.path, parsed.query, EMPTY_SHA, now, creds
    )
    conn = _connection(url, timeout=600.0)
    hash_obj = hashlib.new(algorithm)
    size = 0
    try:
        conn.request("GET", parsed.path + ("?" + parsed.query if parsed.query else ""), headers=headers)
        res = conn.getresponse()
        if res.status >= 400:
            raise R2RequestError("GET", parsed.path, res.status)
        while chunk := res.read(1024 * 1024):
            hash_obj.update(chunk)
            size += len(chunk)
    finally:
        conn.close()
    return hash_obj, size


# ---------------------------------------------------------------------------
# Layout helpers (pure)
# ---------------------------------------------------------------------------

def channel_for_tag(tag: str) -> str:
    """'stable' for a stable tag, 'canary' for a -canary.<ts> tag."""
    return "canary" if re.search(r"-canary\.20\d{6}(?:\d{6})?$", tag) else "stable"


def staging_key_for(tag: str, filename: str) -> str:
    return f"releases/tag/{tag}/{filename}"


_FULL_SHA_RE = re.compile(r"[a-f0-9]{40}")


def is_full_sha(value: str) -> bool:
    return bool(isinstance(value, str) and _FULL_SHA_RE.fullmatch(value))


# Windows reserved names (CPython ntpath parity). COM/LPT superscript forms
# (COM² etc.) are reserved under NTFS namespace rules, same as the digit forms.
_WIN_RESERVED_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"]
    + [f"COM{i}" for i in range(1, 10)] + ["COM¹", "COM²", "COM³"]
    + [f"LPT{i}" for i in range(1, 10)] + ["LPT¹", "LPT²", "LPT³"]
)


def _is_windows_reserved(value: str) -> bool:
    """True if any path component is reserved on Windows.

    Port of ntpath.isreserved() (added in Python 3.13; release CI may run
    older system Pythons, so inline the semantics rather than depend on it).
    """
    for part in reversed(value.split("/")):
        # Trailing dots and spaces are reserved.
        if part[-1:] in (".", " ") and part not in (".", ".."):
            return True
        stem = part.partition(".")[0].rstrip(" ").upper()
        if stem in _WIN_RESERVED_NAMES:
            return True
    return False


def relative_artifact_path(value: str) -> str:
    """Validate the original path before any filesystem normalization."""
    if (not isinstance(value, str) or not value or value.startswith("/")
            or any(part in ("", ".", "..") for part in value.split("/"))
            or any(c in value for c in "\\:%?#") or any(ord(c) < 32 for c in value)
            or _is_windows_reserved(value)):
        raise ValueError("Invalid release artifact path")
    return value


def commit_key_for(commit: str, filename: str) -> str:
    """Keep nested artifacts inside the exact commit namespace."""
    return commit_prefix_for(commit) + relative_artifact_path(filename)


def commit_prefix_for(commit: str) -> str:
    if not is_full_sha(commit):
        raise ValueError("Commit builds require an exact full 40-character SHA")
    return f"releases/commit/{commit}/"


# Public download origin for object keys. CI supplies the authoritative
# value as CLOUDFLARE_R2_PUBLIC_URL; the documented production origin is the
# fallback so a local command can still name a page it is about to publish.
DEFAULT_PUBLIC_URL = "https://hermes-assets.nousresearch.com"


def public_base_url(explicit: str | None = None) -> str:
    return (
        explicit or os.environ.get("CLOUDFLARE_R2_PUBLIC_URL") or DEFAULT_PUBLIC_URL
    ).rstrip("/")


def public_url_for(base_url: str, key: str) -> str:
    """Public download URL of an object key (segment-wise encoded)."""
    return f"{base_url.rstrip('/')}/{quote(key, safe='/')}"


def channel_page_key_for(channel: str) -> str:
    """The mutable per-channel downloads page, replaced by each release."""
    return f"releases/{channel}/index.html"


def commit_page_key_for(commit: str) -> str:
    """The per-commit-build downloads page (every expected binary, built or not)."""
    return commit_prefix_for(commit) + "index.html"


def feed_dir_for(platform: str, channel: str) -> str:
    return f"releases/{platform}/{channel}"


def cache_control_for(key: str) -> str | None:
    """APT indexes are mutable; by-hash indexes and versioned packages are not."""
    if key.endswith((".appinstaller", ".html")) or key.startswith("releases/stable/") or (key.startswith("releases/darwin/") and key.endswith("-mac.yml")):
        return "no-store"
    if not key.startswith("releases/termux/"):
        return None
    return (
        "public, max-age=31536000, immutable"
        if "/by-hash/" in key or "/pool/" in key
        else "no-store"
    )


# ---------------------------------------------------------------------------
# Credentials / env
# ---------------------------------------------------------------------------

def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(
            f"::error::missing env {name} — see the header comment in scripts/releases/r2.py",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return value


def credentials() -> tuple[dict[str, str], str, str]:
    """(creds, base, bucket) from the R2 env vars. No secrets are printed."""
    creds = {
        "access_key_id": required_env("CLOUDFLARE_R2_ACCESS_KEY_ID"),
        "secret_key": required_env("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
    }
    base = s3_endpoint(required_env("CLOUDFLARE_R2_ACCOUNT_ID"))
    bucket = required_env("CLOUDFLARE_R2_BUCKET")
    return creds, base, bucket


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------

def put_object(
    creds: dict[str, str],
    base: str,
    bucket: str,
    key: str,
    payload: str | bytes,
    now: str,
    content_type: str | None,
    conditions: dict[str, str] | None = None,
    fetcher: Callable[..., Response] | None = None,
) -> None:
    """`payload` is a FILE PATH (streamed — the msixbundle is ~2.7GB; str or
    os.PathLike accepted) or a small in-memory bytes body (feed manifests
    from finalize). A 412 on an immutable path PUT verifies the remote bytes
    match before treating the conflict as success."""
    conditions = conditions or {}
    is_path = isinstance(payload, (str, os.PathLike))
    if is_path:
        payload = os.fspath(payload)
        size = os.path.getsize(payload)
        body_hash = file_sha256(payload)

        def body_iter_factory() -> Iterable[bytes]:
            return stream_file(payload)

        body: bytes | None = None
    else:
        size = len(payload)
        body_hash = hashlib.sha256(payload).hexdigest()
        body_iter_factory = None
        body = payload

    cache_control = cache_control_for(key)
    extra = dict(conditions)
    if cache_control:
        extra["Cache-Control"] = cache_control

    url = f"{base}/{bucket}/{encode_key_path(key)}"
    last_error: Exception | None = None
    response: Response | None = None
    for _ in range(3):
        try:
            response = (
                fetcher(method="PUT", url=url, body_hash=body_hash, body=body,
                        content_length=size, extra_headers=extra)
                if fetcher is not None
                else signed_request(
                    "PUT", url, body=body, body_iter_factory=body_iter_factory,
                    body_hash=body_hash, content_length=size, creds=creds, now=now,
                    content_type=content_type, extra_headers=extra,
                )
            )
            break
        except R2RequestError as err:
            last_error = err
            if err.status == 412 and is_path and conditions.get("If-None-Match") == "*":
                # Immutable conflict: verify the remote bytes ARE ours.
                verify_remote_artifact(
                    url, creds, now, size, body_hash, algorithm="sha256", encoding="hex",
                    fetcher=fetcher,
                )
                response = None
                break
            if err.status in _RETRYABLE_STATUSES:
                time.sleep(1.0)
                continue
            raise
    else:
        raise last_error if last_error else R2RequestError("PUT", key, 0)
    if response is not None and response.status >= 400:
        raise R2RequestError("PUT", key, response.status, response.text())

    # HEAD verifies the upload landed at the right size; fall back to a
    # 1-byte ranged GET (Content-Range carries the authoritative total).
    head = (
        fetcher(method="HEAD", url=url, body_hash=EMPTY_SHA)
        if fetcher is not None
        else signed_request("HEAD", url, creds=creds, now=now)
    )
    remote_size = head.header("content-length")
    if remote_size is None:
        ranged = (
            fetcher(method="GET", url=url, body_hash=EMPTY_SHA,
                    extra_headers={"Range": "bytes=0-0"})
            if fetcher is not None
            else signed_request("GET", url, creds=creds, now=now,
                                extra_headers={"Range": "bytes=0-0"})
        )
        content_range = ranged.header("content-range") or ""
        match = re.search(r"bytes 0-0/(\d+)", content_range)
        if not match:
            raise R2RequestError(
                "GET", key, ranged.status,
                f"could not determine remote size (content-range: {content_range!r})",
            )
        remote_size = match.group(1)
    if str(remote_size) != str(size):
        raise R2RequestError("HEAD", key, head.status, f"size mismatch (remote {remote_size}, local {size})")
    print(f"OK r2: {key} ({size} bytes)")


def finalize(tag: str, dir: str, variant: str | None = None) -> None:
    """Lazy re-export of the Darwin finalize so callers can treat
    scripts.releases.r2 as the single transport surface."""
    from . import darwin as darwin_module

    darwin_module.finalize(tag=tag, dir=dir, variant=variant)


def put(
    tag: str,
    key: str,
    file: str,
    key_is_full: bool = False,
    immutable: bool = False,
) -> None:
    """Upload one artifact. `key` is a filename archived under
    releases/tag/<tag>/ unless `key_is_full`. `immutable=True` sends
    If-None-Match: * to reject replacement of existing bytes."""
    creds, base, bucket = credentials()
    now = amz_timestamp()
    key_path = key if key_is_full else staging_key_for(tag, key)
    conditions = {"If-None-Match": "*"} if immutable else {}
    put_object(creds, base, bucket, key_path, file, now, content_type_for(key), conditions)


# ---------------------------------------------------------------------------
# list + prune helpers (pure)
# ---------------------------------------------------------------------------

def parse_list_xml(xml: str) -> dict:
    """Parse a ListObjectsV2 XML body into keys/lastModified/truncated/nextToken."""
    import html

    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    keys = [html.unescape(k) for k in keys]
    last_modified: dict[str, int] = {}
    truncated = "<IsTruncated>true</IsTruncated>" in xml
    token_match = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)

    def _parse_epoch(text: str) -> int:
        # R2 emits both '...Z' and fractional '...mmmZ' forms; accept either.
        text = text.strip()
        if "." in text:
            return int(
                datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
        return int(
            datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

    for block in re.findall(r"<Contents>([\s\S]*?)</Contents>", xml):
        key_match = re.search(r"<Key>([^<]+)</Key>", block)
        lm_match = re.search(r"<LastModified>([^<]+)</LastModified>", block)
        if key_match and lm_match:
            last_modified[html.unescape(key_match.group(1))] = _parse_epoch(lm_match.group(1))
    return {
        "keys": keys,
        "lastModified": last_modified,
        "truncated": truncated,
        "nextToken": html.unescape(token_match.group(1)) if token_match else None,
    }


def list_objects(
    prefix: str = "",
    creds: dict[str, str] | None = None,
    base: str | None = None,
    bucket: str | None = None,
    fetcher: Callable[..., Response] | None = None,
) -> dict:
    if creds is None:
        creds, base, bucket = credentials()
    assert creds is not None and base is not None and bucket is not None
    keys: list[str] = []
    last_modified: dict[str, int] = {}
    token: str | None = None
    while True:
        params: dict[str, str] = {"list-type": "2", "max-keys": "1000"}
        if prefix:
            params["prefix"] = prefix
        if token:
            params["continuation-token"] = token
        query = canonical_query(params)
        url = f"{base}/{bucket}?{query}"
        if fetcher is not None:
            response = fetcher(method="GET", url=url, body_hash=EMPTY_SHA)
            if response.status >= 400:
                raise R2RequestError("GET", f"/{bucket}", response.status)
        else:
            response = signed_request("GET", url, creds=creds, now=amz_timestamp())
        parsed = parse_list_xml(response.text())
        keys.extend(parsed["keys"])
        last_modified.update(parsed["lastModified"])
        if not parsed["truncated"] or not parsed["nextToken"]:
            break
        token = parsed["nextToken"]
    return {"keys": keys, "lastModified": last_modified}


def get_object(
    creds: dict[str, str], base: str, bucket: str, key: str, now: str,
    fetcher: Callable[..., Response] | None = None,
) -> str | None:
    """GET one object's body, or None when it does not exist / cannot be read."""
    url = f"{base}/{bucket}/{encode_key_path(key)}"
    response = (
        fetcher(method="GET", url=url, body_hash=EMPTY_SHA)
        if fetcher is not None
        else signed_request("GET", url, creds=creds, now=now)
    )
    return response.text() if response.status < 400 else None


def canary_doomed_keys(keys: list[str], cutoff: str) -> list[str]:
    """Keys whose own canary date (YYYYMMDD in the name) is before `cutoff`."""
    doomed = []
    for key in keys:
        match = re.search(r"-canary\.(\d{8})", key)
        if match and match.group(1) < cutoff:
            doomed.append(key)
    return doomed


def publish_feed_uploads(plan: dict, upload: Callable[[str, str], None]) -> None:
    """Feed publish order: the immutable .msixbundle FIRST, the pointer LAST."""
    upload(f"{plan['channelDir']}/{plan['bundleFilename']}", plan["bundleFile"])
    upload(f"{plan['channelDir']}/{plan['appinstallerName']}", plan["appinstallerFile"])


def referenced_feed_bundle_filenames(appinstaller_xml: str | None) -> list[str]:
    return [uri.rsplit("/", 1)[-1] for uri in _feed_bundle_uris(appinstaller_xml)]


def feed_referenced_keys(dir_key: str, appinstaller_xml: str | None) -> list[str]:
    """Full bucket keys a manifest references: each referenced bundle inside
    its feed dir, plus any absolute /releases/... path Uri (tag-archive
    targets), protected by exact key."""
    keys: list[str] = []
    for uri in _feed_bundle_uris(appinstaller_xml):
        keys.append(f"{dir_key}/{uri.rsplit('/', 1)[-1]}")
        if "/" in uri and uri.startswith(("http://", "https://")):
            from urllib.parse import urlsplit, unquote
            path = urlsplit(uri).path
            if path.startswith("/releases/"):
                keys.append(unquote(path[1:]))
    return keys


def _feed_bundle_uris(appinstaller_xml: str | None) -> list[str]:
    """Bundle basenames a .appinstaller manifest still references, parsed
    from the KNOWN generated shape (MainPackage/MainBundle Uri attributes
    only — never any Uri=" in the document). Unrecognized/empty -> [] and
    the pruner treats [] as "block this directory" (fail closed)."""
    xml = str(appinstaller_xml or "").strip()
    if not re.fullmatch(
        r"(?:<\?xml[^?]*\?>\s*)?<AppInstaller\b[^>]*>[\s\S]*</AppInstaller>", xml
    ):
        return []
    elements = re.findall(r"<(?:MainPackage|MainBundle)\b[^>]*/>", xml)
    if len(elements) != 1:
        return []
    uri_match = re.search(r"\bUri=\"([^\"]+)\"", elements[0])
    if uri_match and re.search(r"\.(?:msixbundle|msix)$", uri_match.group(1), re.I):
        return [uri_match.group(1)]
    return []


def stale_feed_bundle_keys(
    keys: list[str],
    feed_xml_by_dir: dict[str, list[str | None]],
    last_modified_ms: dict[str, float] | None = None,
    cutoff_ms: float = float("-inf"),
) -> list[str]:
    """Canary feed-dir retention: fail-closed on every unknown. A bundle is
    doomed when its feed dir's manifests were ALL readable AND it is
    referenced by none of them AND its list LastModified predates cutoff.
    Unreadable/unrecognized manifest (zero references) blocks the whole dir;
    stable dirs and unknown dirs are never pruned; a missing LastModified
    keeps the object."""
    doomed: list[str] = []
    for dir_key, manifests in (feed_xml_by_dir or {}).items():
        if not re.search(r"/canary$", dir_key.rstrip("/")):
            continue  # canaries only
        referenced: set[str] = set()
        blocked = False
        for xml in manifests or []:
            names = referenced_feed_bundle_filenames(xml)
            if not names:
                blocked = True
                print(f"::warning::feed manifest unreadable/unrecognized, skipping feed retention for {dir_key}/")
                break
            referenced.update(names)
        if blocked or not referenced:
            continue
        prefix = f"{dir_key.rstrip('/')}/"
        for key in keys:
            if not key.startswith(prefix):
                continue
            if not re.search(r"\.(?:msixbundle|msix)$", key, re.I):
                continue  # pointers + metadata stay
            if key[len(prefix):] in referenced:
                continue
            lm = (last_modified_ms or {}).get(key)
            if lm is None or lm != lm or lm in (float("inf"),) or lm >= cutoff_ms:
                continue  # keep-days grace (fail-closed)
            doomed.append(key)
    return doomed


# ---------------------------------------------------------------------------
# prune-canaries
# ---------------------------------------------------------------------------

def prune(
    keep_days: int,
    dry_run: bool = False,
    fetcher: Callable[..., Response] | None = None,
    now_epoch: float | None = None,
) -> None:
    """Cutoff dated by the canary suffix in the KEY (a re-uploaded old tag
    never resets its clock). Live referenced objects are never deleted."""
    creds, base, bucket = credentials()
    current = now_epoch if now_epoch is not None else time.time()
    cutoff_date = time.strftime(
        "%Y%m%d", time.gmtime(current - keep_days * 86400)
    )
    cutoff_ms = current - keep_days * 86400
    listing = list_objects(creds=creds, base=base, bucket=bucket, fetcher=fetcher)
    keys = listing["keys"]
    last_modified = listing["lastModified"]
    now = amz_timestamp()

    feed_xml_by_dir: dict[str, list[str]] = {}
    protected: set[str] = set()
    for key in [k for k in keys if k.endswith(".appinstaller")]:
        dir_key = key[: key.rfind("/")]
        xml = get_object(creds, base, bucket, key, now, fetcher=fetcher)
        if not _feed_bundle_uris(xml):
            raise RuntimeError(f"Cannot establish live references from {key}; refusing to prune")
        feed_xml_by_dir.setdefault(dir_key, []).append(xml or "")
        for protected_key in feed_referenced_keys(dir_key, xml):
            protected.add(protected_key)

    # Darwin feeds protect their artifacts (key + .blockmap) too.
    from . import darwin as darwin_module

    for key in [k for k in keys if k.startswith("releases/darwin/") and k.endswith("-mac.yml")]:
        text = get_object(creds, base, bucket, key, now, fetcher=fetcher)
        if text is None:
            # Fail closed: an unreadable Darwin feed might reference the very
            # objects the pruner is about to delete.
            raise RuntimeError(f"Cannot read Darwin feed {key}; refusing to prune")
        for reference in darwin_module.mac_feed_references(text):
            protected.add(reference)

    doomed = [
        key
        for key in [
            *canary_doomed_keys(keys, cutoff_date),
            *stale_feed_bundle_keys(keys, feed_xml_by_dir, last_modified, cutoff_ms),
        ]
        if key not in protected
    ]

    if not doomed:
        print(f"OK r2: no canary objects older than {keep_days} days")
        return
    for key in sorted(doomed):
        if dry_run:
            print(f"(dry-run) would delete r2:{key}")
            continue
        url = f"{base}/{bucket}/{encode_key_path(key)}"
        response = (
            fetcher(method="DELETE", url=url, body_hash=EMPTY_SHA)
            if fetcher is not None
            else signed_request("DELETE", url, creds=creds, now=now)
        )
        if response.status >= 400:
            raise R2RequestError("DELETE", key, response.status)
        print(f"deleted r2:{key}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

USAGE = """usage:
  python -m scripts.releases.r2 put --tag vX.Y.Z --key <filename> --file <path> [--key-is-full] [--immutable]
  python -m scripts.releases.r2 finalize --tag vX.Y.Z --dir <staging-dir> [--variant light]
  python -m scripts.releases.r2 list [--prefix <p>]
  python -m scripts.releases.r2 prune-canaries --keep-days <n> [--dry-run]

  --key-is-full: the --key is a FULL object key (e.g. releases/win32/stable/...),
                 not a filename to archive under releases/tag/<tag>/.
  --immutable:   send If-None-Match: * (refuse to overwrite an existing object).
"""

_VALUE_FLAGS = {"--tag", "--key", "--file", "--prefix", "--keep-days", "--dir", "--variant"}
_BOOL_FLAGS = {"--dry-run", "--key-is-full", "--immutable"}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)
    cmd, rest = argv[0], argv[1:]
    args: dict[str, str | bool] = {}
    i = 0
    while i < len(rest):
        flag = rest[i]
        if flag in _VALUE_FLAGS:
            i += 1
            if i >= len(rest):
                print(USAGE, file=sys.stderr)
                raise SystemExit(2)
            args[flag[2:]] = rest[i]
        elif flag in _BOOL_FLAGS:
            args[flag[2:]] = True
        else:
            print(USAGE, file=sys.stderr)
            raise SystemExit(2)
        i += 1

    if cmd == "put":
        tag = args.get("tag") or os.environ.get("HERMES_PAYLOAD_TAG")
        key, file = args.get("key"), args.get("file")
        if not tag or not key or not file:
            print(USAGE, file=sys.stderr)
            raise SystemExit(2)
        immutable = bool(args.get("immutable"))
        if not immutable:
            # Preserve the JS CLI's derivation: darwin artifact staging keys
            # are immutable per-release (rerun-safe via If-None-Match: *).
            key_path = key if args.get("key-is-full") else staging_key_for(str(tag), str(key))
            immutable = key_path.startswith("releases/tag/") and bool(
                re.search(r"-mac-(?:arm64|x64)\.(?:zip|dmg)(?:\.blockmap)?$", key_path)
            )
        put(
            tag=tag,
            key=key,
            file=file,
            key_is_full=bool(args.get("key-is-full")),
            immutable=bool(args.get("immutable")) or immutable,
        )
    elif cmd == "finalize":
        tag, dir_path = args.get("tag"), args.get("dir")
        if not tag or not dir_path:
            print(USAGE, file=sys.stderr)
            raise SystemExit(2)
        from . import darwin as darwin_module

        darwin_module.finalize(tag=tag, dir=dir_path, variant=args.get("variant"))
    elif cmd == "list":
        listing = list_objects(prefix=str(args.get("prefix") or ""))
        for key in listing["keys"]:
            print(key)
    elif cmd == "prune-canaries":
        try:
            keep_days = int(args.get("keep-days", ""))
        except ValueError:
            print(USAGE, file=sys.stderr)
            raise SystemExit(2)
        if keep_days <= 0:
            print(USAGE, file=sys.stderr)
            raise SystemExit(2)
        prune(keep_days, dry_run=bool(args.get("dry-run")))
    else:
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as err:  # pragma: no cover — CLI error surface
        print(f"::error::{err}", file=sys.stderr)
        raise SystemExit(1)
