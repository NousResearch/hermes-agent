"""Client for uploading ``hermes debug share`` bundles to Nous-internal S3.

This is the opt-in (``--nous``) destination for ``hermes debug share``.
Unlike the public paste.rs path, bundles uploaded here go to a Nous-owned
S3 bucket via a short-lived signed URL minted by the Nous account service
(NAS).  The bucket auto-expires objects after 14 days, and the contents are
only viewable by Nous staff (and allowlisted Discord mods) through a
Google-OAuth-gated viewer.

Flow:

    1. POST {NAS_BASE}/api/diagnostics/upload-url  → {uploadUrl, viewUrl, id, ...}
       (the request body carries ``sizeBytes``; NAS signs it into the presigned
       URL's ``ContentLength``, so the PUT must send exactly that many bytes)
    2. PUT <uploadUrl>  (the gzipped bundle, Content-Type application/gzip)

NAS is stateless — the object's existence in S3 is the only state, so there is
no confirm/callback step.

Uses stdlib ``urllib`` only, matching ``debug.py`` style — no third-party deps.
"""

import json
import os
import urllib.request

# Base URL of the Nous account service that mints the signed upload URL.
# Overridable via env so the feature can be pointed at staging / a local dev
# NAS instance during testing.
NAS_BASE = os.environ.get(
    "HERMES_DIAGNOSTICS_BASE_URL", "https://portal.nousresearch.com"
)

# Network timeout for each request (seconds). The upload itself can be larger
# (a gzipped log bundle), so the PUT gets a more generous window.
_REQUEST_TIMEOUT = 30
_UPLOAD_TIMEOUT = 120
_MAX_UPLOAD_URL_RESPONSE_BYTES = 1024 * 1024

_USER_AGENT = "hermes-agent/debug-share"


def _read_bounded_text_response(resp, *, limit: int = _MAX_UPLOAD_URL_RESPONSE_BYTES) -> str:
    headers = getattr(resp, "headers", None)
    raw_length = headers.get("Content-Length") if hasattr(headers, "get") else None
    if isinstance(raw_length, (str, bytes, int)) and int(raw_length) > limit:
        raise RuntimeError("diagnostics upload-url response body is too large")

    body = resp.read(limit + 1)
    if len(body) > limit:
        raise RuntimeError("diagnostics upload-url response body is too large")
    return body.decode("utf-8")


def request_upload_url(
    content_type: str = "application/gzip",
    size_bytes: int | None = None,
) -> dict:
    """Ask NAS to mint a presigned PUT URL for a diagnostics bundle.

    POSTs a small JSON body to ``{NAS_BASE}/api/diagnostics/upload-url`` and
    returns the parsed JSON response, expected to contain at least
    ``uploadUrl``, ``viewUrl`` and ``id`` (plus optional ``expiresAt`` /
    ``uploadExpiresInSeconds``).

    Raises on non-2xx responses or unparseable JSON.
    """
    payload: dict = {"contentType": content_type}
    if size_bytes is not None:
        payload["sizeBytes"] = int(size_bytes)

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{NAS_BASE}/api/diagnostics/upload-url",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        status = getattr(resp, "status", None)
        if status is None:
            status = resp.getcode()
        if not (200 <= status < 300):
            raise RuntimeError(
                f"diagnostics upload-url request failed: HTTP {status}"
            )
        body = _read_bounded_text_response(resp)

    try:
        result = json.loads(body)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"diagnostics upload-url returned non-JSON response: {body[:200]}"
        ) from exc

    if not isinstance(result, dict) or not result.get("uploadUrl"):
        raise RuntimeError(
            "diagnostics upload-url response missing 'uploadUrl': "
            f"{body[:200]}"
        )
    return result


def put_bundle(
    upload_url: str,
    data: bytes,
    content_type: str = "application/gzip",
) -> None:
    """PUT the gzipped *data* bundle to a presigned *upload_url*.

    Sets the ``Content-Type`` header (must match what NAS pinned when signing
    the URL, otherwise S3 rejects the signature). Raises on non-2xx.
    """
    req = urllib.request.Request(
        upload_url,
        data=data,
        method="PUT",
        headers={
            "Content-Type": content_type,
            "User-Agent": _USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=_UPLOAD_TIMEOUT) as resp:
        status = getattr(resp, "status", None)
        if status is None:
            status = resp.getcode()
        if not (200 <= status < 300):
            raise RuntimeError(f"diagnostics bundle PUT failed: HTTP {status}")


def share_to_nous(report_bundle: bytes) -> dict:
    """Orchestrate the full Nous-S3 upload of a gzipped *report_bundle*.

    Two steps: mint a presigned PUT URL (sending the exact ``sizeBytes`` NAS
    signs into the URL's ``ContentLength``), then PUT the bundle. NAS is
    stateless — the object's existence in S3 is the only state, so there is no
    confirm/callback step. Returns the dict from :func:`request_upload_url`
    (which carries ``viewUrl`` / ``id`` / expiry metadata) so the caller can
    print the viewer link. Raises on any failure of either step.
    """
    size_bytes = len(report_bundle)
    info = request_upload_url(
        content_type="application/gzip", size_bytes=size_bytes
    )
    put_bundle(info["uploadUrl"], report_bundle, content_type="application/gzip")

    return info
