"""Fetch the public petdex manifest.

``https://petdex.dev/api/manifest`` 307-redirects to a JSON document on R2:
``{"generatedAt", "total", "pets": [{"slug", "displayName", "kind",
"submittedBy", "spritesheetUrl", "petJsonUrl", "zipUrl"}, ...]}``.
Read-only and unauthenticated; no credentials involved.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MANIFEST_URL = "https://petdex.dev/api/manifest"

_DEFAULT_TIMEOUT = 10.0

# In-process cache for the (large, slow, identical-per-call) manifest: a static
# CDN object a single session may ask for many times (every gallery open, plus a
# re-fetch per install/select). A short TTL collapses those into one network hit.
_MANIFEST_TTL = 300.0
_cache: tuple[float, list[ManifestEntry]] | None = None

_prefetch_lock = threading.Lock()
_prefetching = False


def clear_cache() -> None:
    """Drop the cached manifest (forces the next fetch to hit the network)."""
    global _cache
    _cache = None


def _cache_is_warm() -> bool:
    return _cache is not None and time.monotonic() - _cache[0] < _MANIFEST_TTL


def prefetch(*, timeout: float = _DEFAULT_TIMEOUT) -> None:
    """Warm the manifest cache in a daemon thread — idempotent, never blocks.

    The desktop picker calls this when loading the instant local-only gallery so
    the full catalog is usually cached by the time it's requested.
    """
    global _prefetching
    if _cache_is_warm():
        return
    with _prefetch_lock:
        if _prefetching:
            return
        _prefetching = True

    def _run() -> None:
        global _prefetching
        try:
            fetch_manifest(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - best-effort warm
            logger.debug("petdex manifest prefetch failed: %s", exc)
        finally:
            _prefetching = False

    threading.Thread(target=_run, name="petdex-prefetch", daemon=True).start()


@dataclass(frozen=True)
class ManifestEntry:
    """A single pet's row in the manifest."""

    slug: str
    display_name: str
    kind: str
    submitted_by: str
    spritesheet_url: str
    pet_json_url: str
    zip_url: str

    @classmethod
    def from_dict(cls, data: dict) -> "ManifestEntry":
        return cls(
            slug=str(data.get("slug", "")).strip(),
            display_name=str(data.get("displayName", "") or data.get("slug", "")),
            kind=str(data.get("kind", "") or "pet"),
            submitted_by=str(data.get("submittedBy", "") or ""),
            spritesheet_url=str(data.get("spritesheetUrl", "") or ""),
            pet_json_url=str(data.get("petJsonUrl", "") or ""),
            zip_url=str(data.get("zipUrl", "") or ""),
        )


class ManifestError(RuntimeError):
    """Raised when the manifest can't be fetched or parsed."""


def fetch_manifest(*, timeout: float = _DEFAULT_TIMEOUT, force: bool = False) -> list[ManifestEntry]:
    """Every approved pet from the public manifest; cached for ``_MANIFEST_TTL`` s unless *force*. Raises :class:`ManifestError`."""
    global _cache
    if not force and _cache_is_warm():
        return _cache[1]
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx is a core dep
        raise ManifestError("httpx is required to fetch the petdex manifest") from exc
    try:
        resp = httpx.get(MANIFEST_URL, timeout=timeout, follow_redirects=True, headers={"User-Agent": "hermes-agent-petdex"})
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 - normalize to one error type
        raise ManifestError(f"could not fetch petdex manifest: {exc}") from exc

    pets = payload.get("pets") if isinstance(payload, dict) else None
    if not isinstance(pets, list):
        raise ManifestError("petdex manifest had no 'pets' array")
    parsed = (ManifestEntry.from_dict(raw) for raw in pets if isinstance(raw, dict))
    entries = [entry for entry in parsed if entry.slug and entry.spritesheet_url]
    _cache = (time.monotonic(), entries)
    return entries


def find_entry(slug: str, *, timeout: float = _DEFAULT_TIMEOUT) -> ManifestEntry | None:
    """Return the manifest entry for *slug*, or ``None`` if not listed."""
    slug = slug.strip().lower()
    return next((entry for entry in fetch_manifest(timeout=timeout) if entry.slug.lower() == slug), None)
