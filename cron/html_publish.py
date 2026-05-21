"""Publish cron HTML artifacts to a small authenticated hosting endpoint.

The publisher is intentionally boring: it uploads already-sanitized local HTML to
an operator-controlled endpoint and expects the endpoint to return a user-facing
URL.  It never shells out to deployment tools and never treats publishing as
required for cron success.
"""

from __future__ import annotations

import json
import os
import re
import hashlib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import quote, urlparse

_SAFE_KEY_PART_RE = re.compile(r"[^A-Za-z0-9._-]+")


class HtmlArtifactPublishError(RuntimeError):
    """Raised when an enabled artifact publish cannot complete safely."""


def _strict_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off", ""}:
            return False
    return default


def _safe_key_part(value: str) -> str:
    cleaned = _SAFE_KEY_PART_RE.sub("-", value.strip()).strip(".-_")
    while ".." in cleaned:
        cleaned = cleaned.replace("..", ".")
    cleaned = cleaned.replace(".", "-").strip("-")
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned or "artifact"


def artifact_object_key(job_id: str, html_path: Path, content: bytes | None = None) -> str:
    """Return a stable signed artifact object key for an artifact path."""
    stem = _safe_key_part(html_path.stem)
    if content:
        stem = f"{stem}-{hashlib.sha256(content).hexdigest()[:12]}"
    safe_job_id = _safe_key_part(job_id)
    return f"r/{safe_job_id}/{stem}.html"


def _configured_object_key(settings: Mapping[str, Any]) -> str:
    """Return a caller-specified object key for stable public Acta pages."""
    raw = str(settings.get("object_key") or "").strip().lstrip("/")
    if not raw:
        return ""
    if ".." in raw or not re.match(r"^(?:public/[A-Za-z0-9._/-]+|r/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+)\.html$", raw):
        raise HtmlArtifactPublishError("configured object_key is invalid")
    return raw


def resolve_publish_settings(job: Mapping[str, Any], html_settings: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve global + per-job publish settings for one HTML artifact.

    Expected config shape:

    cron.html_artifacts.publish: {enabled, endpoint, upload_token_env, ...}
    cron.html_artifacts.jobs.<job_id>.publish: bool or overrides dict
    """
    global_publish = html_settings.get("publish", {})
    if not isinstance(global_publish, Mapping):
        global_publish = {}

    settings: dict[str, Any] = dict(global_publish)
    global_enabled = _strict_bool(settings.get("enabled"), default=False)
    job_id = str(job.get("id", ""))
    job_cfg = html_settings.get("jobs", {}) if isinstance(html_settings.get("jobs"), Mapping) else {}
    per_job = job_cfg.get(job_id, {}) if isinstance(job_cfg, Mapping) else {}
    if not isinstance(per_job, Mapping):
        per_job = {}
    job_publish = per_job.get("publish")
    if isinstance(job_publish, Mapping):
        settings.update(job_publish)
    elif isinstance(job_publish, bool):
        settings["enabled"] = job_publish

    # Global publish.enabled is the master switch.  A job can opt in only
    # after global publishing is deliberately enabled.
    enabled = global_enabled and _strict_bool(settings.get("enabled"), default=False)
    settings["enabled"] = enabled
    settings.setdefault("upload_token_env", "ACTA_UPLOAD_TOKEN")
    settings.setdefault("max_kb", html_settings.get("max_attachment_kb", 512))
    return settings


def publish_html_artifact(html_path: Path, job: Mapping[str, Any], settings: Mapping[str, Any]) -> Optional[str]:
    """Upload an HTML artifact and return its hosted URL, or None if disabled."""
    if not _strict_bool(settings.get("enabled"), default=False):
        return None

    endpoint = str(settings.get("endpoint") or settings.get("base_url") or "").strip().rstrip("/")
    if not endpoint:
        raise HtmlArtifactPublishError("publish endpoint/base_url is required")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise HtmlArtifactPublishError("publish endpoint must be an https URL")

    token_env = str(settings.get("upload_token_env") or "ACTA_UPLOAD_TOKEN").strip()
    token = os.getenv(token_env, "").strip()
    if not token:
        raise HtmlArtifactPublishError(f"publish token env var {token_env} is not set")

    html_path = Path(html_path)
    body = html_path.read_bytes()
    try:
        max_kb = int(settings.get("max_kb") or 512)
    except (TypeError, ValueError):
        max_kb = 512
    if max_kb > 0 and len(body) > max_kb * 1024:
        raise HtmlArtifactPublishError(f"artifact exceeds publish size cap ({max_kb} KB)")

    key = _configured_object_key(settings) or artifact_object_key(str(job.get("id", "job")), html_path, content=body)
    upload_url = f"{endpoint}/__upload/{quote(key, safe='/') }"
    req = urllib.request.Request(
        upload_url,
        data=body,
        method="PUT",
        headers={
            "Content-Type": "text/html; charset=utf-8",
            "X-Acta-Upload-Token": token,
            "User-Agent": "HermesCronHtmlPublisher/1.0",
        },
    )
    timeout = float(settings.get("timeout_seconds") or 20)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - configured HTTPS endpoint only
            raw = resp.read(65536).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raise HtmlArtifactPublishError(f"publish failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise HtmlArtifactPublishError(f"publish request failed: {exc.reason}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HtmlArtifactPublishError("publish endpoint returned non-JSON response") from exc
    url = str(payload.get("url") or "").strip()
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.netloc != parsed.netloc:
        raise HtmlArtifactPublishError("publish endpoint did not return a valid Acta URL")
    return url
