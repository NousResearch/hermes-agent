"""Client for uploading ``hermes debug share`` bundles to Nous-internal S3.
1. POST {NAS_BASE}/api/diagnostics/upload-url → {uploadUrl, viewUrl, id, ...}; the body carries ``sizeBytes``,
   which NAS signs into the presigned URL's ``ContentLength``, so the PUT must send exactly that many bytes.
2. PUT <uploadUrl> (gzipped bundle, Content-Type application/gzip). NAS is stateless — no confirm step."""

import json
import os
import urllib.request

# Overridable via env so the feature can be pointed at staging / a local dev NAS instance.
NAS_BASE = os.environ.get("HERMES_DIAGNOSTICS_BASE_URL", "https://portal.nousresearch.com")
_REQUEST_TIMEOUT = 30
_UPLOAD_TIMEOUT = 120  # the PUT carries the gzipped log bundle, so a more generous window
_USER_AGENT = "hermes-agent/debug-share"


def _urlopen_checked(url: str, data: bytes, method: str, headers: dict, *, timeout: int, what: str):
    """Send the request; raise ``RuntimeError`` on non-2xx and return the response body bytes."""
    req = urllib.request.Request(url, data=data, method=method, headers={**headers, "User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = getattr(resp, "status", None)
        status = resp.getcode() if status is None else status
        if not (200 <= status < 300):
            raise RuntimeError(f"{what} failed: HTTP {status}")
        return resp.read()


def request_upload_url(content_type: str = "application/gzip", size_bytes: int | None = None) -> dict:
    """Ask NAS to mint a presigned PUT URL for a diagnostics bundle. Returns the parsed JSON, expected to
    carry at least ``uploadUrl``, ``viewUrl`` and ``id``. Raises on non-2xx responses or unparseable JSON."""
    payload: dict = {"contentType": content_type}
    if size_bytes is not None:
        payload["sizeBytes"] = int(size_bytes)
    body = _urlopen_checked(
        f"{NAS_BASE}/api/diagnostics/upload-url", json.dumps(payload).encode("utf-8"), "POST",
        {"Content-Type": "application/json", "Accept": "application/json"},
        timeout=_REQUEST_TIMEOUT, what="diagnostics upload-url request").decode("utf-8")
    try:
        result = json.loads(body)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"diagnostics upload-url returned non-JSON response: {body[:200]}") from exc
    if not isinstance(result, dict) or not result.get("uploadUrl"):
        raise RuntimeError(f"diagnostics upload-url response missing 'uploadUrl': {body[:200]}")
    return result


def put_bundle(upload_url: str, data: bytes, content_type: str = "application/gzip") -> None:
    """PUT the gzipped *data* bundle to a presigned *upload_url*. Raises on non-2xx.
    ``Content-Type`` must match what NAS pinned when signing the URL, otherwise S3 rejects the signature."""
    _urlopen_checked(upload_url, data, "PUT", {"Content-Type": content_type}, timeout=_UPLOAD_TIMEOUT,
                     what="diagnostics bundle PUT")


def share_to_nous(report_bundle: bytes) -> dict:
    """Mint a presigned PUT URL (with the exact ``sizeBytes`` NAS signs), then PUT *report_bundle*."""
    info = request_upload_url(content_type="application/gzip", size_bytes=len(report_bundle))
    put_bundle(info["uploadUrl"], report_bundle, content_type="application/gzip")
    return info
