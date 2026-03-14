"""
Cloudflare R2 upload client for the Hermes gateway.

Uses the S3-compatible API with AWS Signature Version 4 signing,
implemented via stdlib (hashlib, hmac) — no boto3/aioboto3 required.

Configuration is read from environment variables:
    R2_ACCOUNT_ID        — Cloudflare account ID
    R2_ACCESS_KEY_ID     — R2 API token access key
    R2_SECRET_ACCESS_KEY — R2 API token secret
    R2_BUCKET_NAME       — Bucket name (default: athabasca-media)
    R2_PUBLIC_URL        — Public base URL for the bucket (no trailing slash)

When R2 is not configured (env vars missing), all upload functions return
None gracefully — callers should fall back to the local cache path.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env(key: str) -> str:
    return os.environ.get(key, "").strip()


def _is_configured() -> bool:
    return all([
        _env("R2_ACCOUNT_ID"),
        _env("R2_ACCESS_KEY_ID"),
        _env("R2_SECRET_ACCESS_KEY"),
        _env("R2_BUCKET_NAME"),
        _env("R2_PUBLIC_URL"),
    ])


def _endpoint() -> str:
    account_id = _env("R2_ACCOUNT_ID")
    return f"https://{account_id}.r2.cloudflarestorage.com"


def _public_url(key: str) -> str:
    base = _env("R2_PUBLIC_URL").rstrip("/")
    return f"{base}/{key}"


# ---------------------------------------------------------------------------
# AWS Signature Version 4 (minimal — PUT only)
# ---------------------------------------------------------------------------

def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret: str, date: str, region: str, service: str) -> bytes:
    k_date = _sign(("AWS4" + secret).encode("utf-8"), date)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")
    return k_signing


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_auth_headers(
    method: str,
    endpoint: str,
    bucket: str,
    key: str,
    body: bytes,
    content_type: str,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build Authorization + x-amz-* headers for an S3 PutObject request."""
    access_key = _env("R2_ACCESS_KEY_ID")
    secret_key = _env("R2_SECRET_ACCESS_KEY")
    region = "auto"
    service = "s3"

    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    # Host header (just the host part, no scheme)
    host = endpoint.replace("https://", "").replace("http://", "")
    path = f"/{bucket}/{quote(key, safe='/-._~')}"

    payload_hash = _sha256_hex(body)

    headers: dict[str, str] = {
        "host": host,
        "content-type": content_type,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
        "x-amz-meta-uploaded-by": "athabasca-gateway",
        **(extra_headers or {}),
    }

    # Canonical headers (sorted lowercase)
    canonical_headers_str = "".join(
        f"{k}:{v}\n" for k, v in sorted(headers.items())
    )
    signed_headers = ";".join(sorted(headers.keys()))

    canonical_request = "\n".join([
        method,
        path,
        "",  # query string
        canonical_headers_str,
        signed_headers,
        payload_hash,
    ])

    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        _sha256_hex(canonical_request.encode("utf-8")),
    ])

    signing_key = _signing_key(secret_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    return {
        **{k: v for k, v in headers.items() if k != "host"},
        "Authorization": authorization,
    }


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def make_r2_key(
    project_slug: str,
    category: str,
    filename: str,
) -> str:
    """Build a stable storage key: athabasca/{slug}/{category}/{filename}"""
    safe_filename = re.sub(r"[^a-zA-Z0-9._\-]", "_", filename)
    safe_slug = re.sub(r"[^a-zA-Z0-9._\-]", "_", project_slug)
    return f"athabasca/{safe_slug}/{category}/{safe_filename}"


def r2_public_url(key: str) -> str:
    return _public_url(key)


# ---------------------------------------------------------------------------
# Core upload
# ---------------------------------------------------------------------------

@dataclass
class UploadResult:
    key: str
    public_url: str
    content_type: str
    size_bytes: int


async def upload_bytes(
    key: str,
    data: bytes,
    content_type: str,
    extra_metadata: dict[str, str] | None = None,
) -> Optional[UploadResult]:
    """
    Upload raw bytes to R2 under the given key.
    Returns None (with a warning) if R2 is not configured.
    """
    if not _is_configured():
        logger.warning(
            "R2 not configured — skipping upload of %s. "
            "Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, "
            "R2_BUCKET_NAME, and R2_PUBLIC_URL in .env to enable cloud storage.",
            key,
        )
        return None

    bucket = _env("R2_BUCKET_NAME")
    endpoint = _endpoint()

    extra = {f"x-amz-meta-{k}": v for k, v in (extra_metadata or {}).items()}
    auth_headers = _build_auth_headers(
        method="PUT",
        endpoint=endpoint,
        bucket=bucket,
        key=key,
        body=data,
        content_type=content_type,
        extra_headers=extra,
    )

    url = f"{endpoint}/{bucket}/{quote(key, safe='/-._~')}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.put(
                url,
                content=data,
                headers={**auth_headers, "Content-Type": content_type},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("R2 upload failed for key %s: HTTP %s — %s", key, e.response.status_code, e.response.text[:200])
        return None
    except Exception as e:
        logger.error("R2 upload error for key %s: %s", key, e)
        return None

    return UploadResult(
        key=key,
        public_url=_public_url(key),
        content_type=content_type,
        size_bytes=len(data),
    )


async def upload_file(
    local_path: str | Path,
    key: str,
    content_type: str | None = None,
    extra_metadata: dict[str, str] | None = None,
) -> Optional[UploadResult]:
    """Upload a local file to R2. Returns None if R2 is not configured."""
    path = Path(local_path)
    data = path.read_bytes()

    if content_type is None:
        guessed, _ = mimetypes.guess_type(str(path))
        content_type = guessed or "application/octet-stream"

    return await upload_bytes(key, data, content_type, extra_metadata)


# ---------------------------------------------------------------------------
# Convenience: upload a Telegram-sourced image
# ---------------------------------------------------------------------------

async def upload_telegram_photo(
    local_path: str | Path,
    project_slug: str,
    original_filename: str | None = None,
) -> Optional[UploadResult]:
    """
    Upload a photo that was downloaded from Telegram to R2.

    Uses the project slug to namespace the key under
    athabasca/{slug}/research/{filename}.

    Returns UploadResult with a permanent public_url on success,
    or None if R2 is not configured (caller should use local cache path).
    """
    path = Path(local_path)
    filename = original_filename or path.name
    # Ensure a timestamp suffix to avoid collisions
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem, ext = (filename.rsplit(".", 1) + ["jpg"])[:2]
    unique_filename = f"{stem}_{timestamp}.{ext}"

    key = make_r2_key(project_slug, "research", unique_filename)

    return await upload_file(
        local_path=path,
        key=key,
        extra_metadata={
            "source": "telegram-upload",
            "project-slug": project_slug,
        },
    )


# ---------------------------------------------------------------------------
# Status check (for health endpoints / agent memory)
# ---------------------------------------------------------------------------

def r2_status() -> dict:
    """Return R2 configuration status — safe to log/display."""
    return {
        "configured": _is_configured(),
        "bucket": _env("R2_BUCKET_NAME") or None,
        "public_url": _env("R2_PUBLIC_URL") or None,
        "account_id_set": bool(_env("R2_ACCOUNT_ID")),
        "key_set": bool(_env("R2_ACCESS_KEY_ID")),
    }
