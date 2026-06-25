"""Optional outbound media publishing for gateway adapters.

The default gateway behavior sends local ``MEDIA:`` files as native platform
attachments.  Some platforms (notably Discord) can fail large or flaky uploads.
This module provides a default-off external publishing layer: upload local media
objects to configured storage, verify the public URL, then let the adapter send
links instead of binary attachments.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import hmac
import mimetypes
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote
import urllib.request

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac"}
_SUPPORTED_MODES = {"link_first", "link_on_failure", "attach_and_link"}


@dataclass(frozen=True)
class PublishedMedia:
    local_path: str
    url: str
    display_name: str
    mime_type: str
    size_bytes: int


@dataclass(frozen=True)
class ExternalMediaConfig:
    enabled: bool = False
    provider: str = "r2"
    mode: str = "link_on_failure"
    images: bool = True
    videos: bool = True
    audio: bool = False
    documents: bool = False
    size_threshold_bytes: int = 0
    remote_prefix: str = "hermes/media/%Y/%m/%d/"
    verify: bool = True
    public_base_url: str = ""
    bucket: str = ""
    account_id: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    api_token: str = ""
    wrangler: bool = False

    @classmethod
    def disabled(cls) -> "ExternalMediaConfig":
        return cls(enabled=False)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _merge_dicts(*items: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if isinstance(item, dict):
            merged.update(item)
    return merged


def external_media_config_for(adapter: Any, media_kind: str) -> ExternalMediaConfig:
    """Return the effective external-media config for an adapter/kind.

    Config is intentionally default-off.  Supported shapes:

    ``media_delivery.external_upload`` global config copied into platform extra
    by ``gateway.config``;
    ``media_delivery.external_upload.platforms.<platform>`` platform override;
    ``platforms.<platform>.extra.external_media`` adapter-local override.
    """
    platform = getattr(getattr(adapter, "platform", None), "value", None) or str(getattr(adapter, "platform", ""))
    extra = getattr(getattr(adapter, "config", None), "extra", {}) or {}
    media_delivery = extra.get("media_delivery") if isinstance(extra, dict) else {}
    external = {}
    if isinstance(media_delivery, dict):
        external = media_delivery.get("external_upload") or {}
    platform_override = {}
    if isinstance(external, dict):
        platforms = external.get("platforms") or {}
        if isinstance(platforms, dict):
            platform_override = platforms.get(platform) or {}
    local_override = extra.get("external_media") if isinstance(extra, dict) else {}
    raw = _merge_dicts(external, platform_override, local_override)
    if not raw:
        return ExternalMediaConfig.disabled()

    mode = str(raw.get("mode") or "link_on_failure").strip().lower()
    if mode not in _SUPPORTED_MODES:
        mode = "link_on_failure"

    threshold = _as_int(raw.get("size_threshold_bytes"), -1)
    if threshold < 0:
        threshold_mb = raw.get("size_threshold_mb", 0)
        threshold = max(0, _as_int(threshold_mb, 0)) * 1024 * 1024

    # Secrets may come from env so config.yaml can stay non-secret.
    return ExternalMediaConfig(
        enabled=_as_bool(raw.get("enabled"), False),
        provider=str(raw.get("provider") or "r2").strip().lower(),
        mode=mode,
        images=_as_bool(raw.get("images"), True),
        videos=_as_bool(raw.get("videos"), True),
        audio=_as_bool(raw.get("audio"), False),
        documents=_as_bool(raw.get("documents"), False),
        size_threshold_bytes=threshold,
        remote_prefix=str(raw.get("remote_prefix") or "hermes/media/%Y/%m/%d/"),
        verify=_as_bool(raw.get("verify"), True),
        public_base_url=str(raw.get("public_base_url") or _env("CLOUDFLARE_R2_PUBLIC_BASE_URL") or _env("R2_PUBLIC_BASE_URL")),
        bucket=str(raw.get("bucket") or _env("CLOUDFLARE_R2_BUCKET") or _env("R2_BUCKET")),
        account_id=str(raw.get("account_id") or _env("CLOUDFLARE_ACCOUNT_ID")),
        access_key_id=str(raw.get("access_key_id") or _env("CLOUDFLARE_R2_ACCESS_KEY_ID") or _env("R2_ACCESS_KEY_ID")),
        secret_access_key=str(raw.get("secret_access_key") or _env("CLOUDFLARE_R2_SECRET_ACCESS_KEY") or _env("R2_SECRET_ACCESS_KEY")),
        api_token=str(raw.get("api_token") or _env("CLOUDFLARE_API_TOKEN")),
        wrangler=_as_bool(raw.get("wrangler"), False),
    )


def _kind_for_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in _IMAGE_EXTS:
        return "images"
    if ext in _VIDEO_EXTS:
        return "videos"
    if ext in _AUDIO_EXTS:
        return "audio"
    return "documents"


def _kind_enabled(cfg: ExternalMediaConfig, kind: str) -> bool:
    return bool(getattr(cfg, kind, False))


def _remote_key(cfg: ExternalMediaConfig, path: Path) -> str:
    prefix = _dt.datetime.now().strftime(cfg.remote_prefix).strip("/")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in path.stem)[:80]
    name = f"{safe_stem}-{digest}{path.suffix.lower()}" if safe_stem else f"{digest}{path.suffix.lower()}"
    return f"{prefix}/{name}" if prefix else name


def _public_url(cfg: ExternalMediaConfig, key: str) -> str:
    base = cfg.public_base_url.rstrip("/")
    return f"{base}/{quote(key, safe='/')}"


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _sigv4_authorization(cfg: ExternalMediaConfig, method: str, key: str, payload: bytes, content_type: str, amz_date: str, date_stamp: str) -> str:
    region = "auto"
    service = "s3"
    host = f"{cfg.account_id}.r2.cloudflarestorage.com"
    canonical_uri = f"/{quote(cfg.bucket, safe='')}/{quote(key, safe='/')}"
    payload_hash = hashlib.sha256(payload).hexdigest()
    canonical_headers = (
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join([
        method,
        canonical_uri,
        "",
        canonical_headers,
        signed_headers,
        payload_hash,
    ])
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])
    signing_key = _sign(_sign(_sign(_sign(("AWS4" + cfg.secret_access_key).encode("utf-8"), date_stamp), region), service), "aws4_request")
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    return (
        "AWS4-HMAC-SHA256 "
        f"Credential={cfg.access_key_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )


def _upload_r2_sigv4(cfg: ExternalMediaConfig, local_path: Path, key: str) -> str:
    if not (cfg.account_id and cfg.bucket and cfg.access_key_id and cfg.secret_access_key and cfg.public_base_url):
        raise RuntimeError("R2 external media upload requires account_id, bucket, public_base_url, access_key_id, and secret_access_key")
    payload = local_path.read_bytes()
    content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    now = _dt.datetime.now(_dt.UTC)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    host = f"{cfg.account_id}.r2.cloudflarestorage.com"
    object_url = f"https://{host}/{quote(cfg.bucket, safe='')}/{quote(key, safe='/')}"
    req = urllib.request.Request(object_url, data=payload, method="PUT")
    req.add_header("Content-Type", content_type)
    req.add_header("Host", host)
    req.add_header("X-Amz-Date", amz_date)
    req.add_header("X-Amz-Content-Sha256", hashlib.sha256(payload).hexdigest())
    req.add_header("Authorization", _sigv4_authorization(cfg, "PUT", key, payload, content_type, amz_date, date_stamp))
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - user-configured storage endpoint
        if resp.status not in {200, 201, 204}:
            raise RuntimeError(f"R2 upload failed with HTTP {resp.status}")
    return _public_url(cfg, key)


def _upload_r2_wrangler(cfg: ExternalMediaConfig, local_path: Path, key: str) -> str:
    if not (cfg.account_id and cfg.api_token and cfg.bucket and cfg.public_base_url):
        raise RuntimeError("wrangler R2 external media upload requires account_id, api_token, bucket, and public_base_url")
    env = os.environ.copy()
    env["CLOUDFLARE_ACCOUNT_ID"] = cfg.account_id
    env["CLOUDFLARE_API_TOKEN"] = cfg.api_token
    proc = subprocess.run(
        ["wrangler", "r2", "object", "put", f"{cfg.bucket}/{key}", "--file", str(local_path), "--remote"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        safe = "\n".join(line for line in proc.stdout.splitlines() if "token" not in line.lower())
        raise RuntimeError(f"wrangler R2 upload failed ({proc.returncode}): {safe[-500:]}")
    return _public_url(cfg, key)


def _verify_public_url(url: str) -> None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 - URL was just produced by configured storage
            if resp.status < 200 or resp.status >= 400:
                raise RuntimeError(f"published media URL returned HTTP {resp.status}")
    except Exception:
        # Some object-store custom domains reject HEAD but serve GET.
        req = urllib.request.Request(url, method="GET")
        req.add_header("Range", "bytes=0-0")
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            if resp.status < 200 or resp.status >= 400:
                raise RuntimeError(f"published media URL returned HTTP {resp.status}")


def upload_media_file(cfg: ExternalMediaConfig, local_path: str) -> PublishedMedia:
    path = Path(local_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    key = _remote_key(cfg, path)
    if cfg.provider != "r2":
        raise RuntimeError(f"Unsupported external media provider: {cfg.provider}")
    if cfg.wrangler:
        url = _upload_r2_wrangler(cfg, path, key)
    else:
        url = _upload_r2_sigv4(cfg, path, key)
    if cfg.verify:
        _verify_public_url(url)
    return PublishedMedia(
        local_path=str(path),
        url=url,
        display_name=path.name,
        mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        size_bytes=path.stat().st_size,
    )


async def publish_media_files(adapter: Any, paths: Iterable[str], media_kind: Optional[str] = None) -> list[PublishedMedia]:
    cfg = external_media_config_for(adapter, media_kind or "")
    if not cfg.enabled:
        return []
    published: list[PublishedMedia] = []
    for item in paths:
        path = Path(item).expanduser()
        if not path.is_file():
            continue
        kind = media_kind or _kind_for_path(str(path))
        if not _kind_enabled(cfg, kind):
            continue
        size = path.stat().st_size
        if size < cfg.size_threshold_bytes:
            continue
        published.append(await asyncio.to_thread(upload_media_file, cfg, str(path)))
    return published


def format_published_media_message(items: list[PublishedMedia], *, heading: str = "Media") -> str:
    if not items:
        return ""
    if len(items) == 1:
        item = items[0]
        return f"{heading}: {item.url}"
    lines = [f"{heading} ({len(items)} files):"]
    for item in items:
        lines.append(f"- {item.display_name}: {item.url}")
    return "\n".join(lines)


def example_config_block() -> dict[str, Any]:
    """Return a documentation-friendly default-off config shape."""
    return {
        "media_delivery": {
            "external_upload": {
                "enabled": False,
                "provider": "r2",
                "mode": "link_on_failure",
                "images": True,
                "videos": True,
                "documents": False,
                "size_threshold_mb": 8,
                "remote_prefix": "hermes/media/%Y/%m/%d/",
                "public_base_url": "https://media.example.com",
                "bucket": "hermes-media",
            }
        }
    }
