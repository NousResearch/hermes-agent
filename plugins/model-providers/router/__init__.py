"""Ramp Router (router.com) provider profile: Responses-only LLM gateway (verified live).

``api_mode="codex_responses"`` + the ``api.router.com`` host mandate in ``hermes_cli/providers.py``
keep every path on the native wire. The catalog is account-scoped, so no ``fallback_models``
(picker uses ``fetch_models()``). Router 400s on ``reasoning.effort`` levels outside a model's
published vocabulary and on any reasoning field for non-reasoning models, so the efforts map
from ``GET /v1/models`` is cached (memory + disk mirror, background warmer; never HTTP on the
request hot path) and fed to the codex transport's clamp via ``supported_reasoning_efforts``.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from agent.reasoning_effort import EFFORT_LADDER
from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

ROUTER_DEFAULT_BASE_URL = "https://api.router.com/v1"

#: model id -> accepted effort levels. ``[]`` = model accepts NO reasoning
#: fields; absent = unknown (callers keep their defaults).
_efforts_cache: Optional[dict[str, list[str]]] = None
_efforts_lock = threading.Lock()
_warm_started = False
_disk_checked = False

# A stale verdict beats no verdict: a past-TTL mirror is still served while a
# background refresh runs.
_DISK_TTL_SECONDS = 24 * 60 * 60


def _base_url() -> str:
    return os.getenv("RAMP_ROUTER_BASE_URL", "").strip().rstrip("/") or ROUTER_DEFAULT_BASE_URL


def _resolve_api_key() -> str:
    """Router key (documented var, then alias), preferring dotenv; plain os.environ
    is the fallback when the dotenv resolver is unavailable or raises."""
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv as prefer_dotenv
    except Exception:
        prefer_dotenv = None
    for resolve in filter(None, (prefer_dotenv, os.environ.get)):
        for var in ("RAMP_ROUTER_API_KEY", "ROUTER_API_KEY"):
            try:
                value = str(resolve(var) or "").strip()
            except Exception:
                value = ""
            if value:
                return value
    return ""


def _dig(obj: Any, *keys: str) -> Any:
    """Nested dict lookup; None as soon as a level is missing or not a dict."""
    for key in keys:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj


def _parse_efforts(items: Any) -> Optional[dict[str, list[str]]]:
    """Parse a ``/v1/models`` ``data`` array into the efforts map (None if unusable).

    Ladder-unknown levels are dropped: clamp_effort ignores them, so an all-unknown
    vocabulary would pass the effort through unclamped to a Router 400. ``supported=True``
    with no recognized level leaves the model out (unknown -> transport default clamp).
    """
    if not isinstance(items, list):
        return None
    efforts_by_id: dict[str, list[str]] = {}
    for item in items:
        mid = str(item.get("id") or "").strip() if isinstance(item, dict) else ""
        reasoning = _dig(item, "router", "capabilities", "reasoning")
        if not mid or not isinstance(reasoning, dict):
            continue
        if reasoning.get("supported") is False:
            efforts_by_id[mid] = []
            continue
        values = [str(e.get("value") or "").strip() for e in reasoning.get("efforts") or [] if isinstance(e, dict)]
        levels = [v for v in values if v]
        unknown = [level for level in levels if level not in EFFORT_LADDER]
        if unknown:
            logger.info(
                "router: model %s publishes unrecognized reasoning effort level(s) %s; ignoring them "
                "(update agent/reasoning_effort EFFORT_LADDER to adopt new vendor tiers)", mid, unknown,
            )
            levels = [level for level in levels if level in EFFORT_LADDER]
        if levels:
            efforts_by_id[mid] = levels
    return efforts_by_id or None


def _disk_path() -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "cache" / "router_catalog.json"
    except Exception:
        return None


def _save_disk(efforts_by_id: dict[str, list[str]]) -> None:
    path = _disk_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")  # write-then-rename keeps readers from seeing a torn file
        tmp.write_text(json.dumps({"ts": time.time(), "efforts": efforts_by_id}), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.debug("router: caps disk mirror write failed: %s", exc)


def _load_disk() -> tuple[Optional[dict[str, list[str]]], float]:
    """Disk mirror -> (efforts map or None, age in seconds; TTL when ``ts`` is unparseable)."""
    path = _disk_path()
    if path is None:
        return None, 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        efforts = data.get("efforts")
        if not isinstance(efforts, dict) or not efforts:
            return None, 0.0
        parsed = {str(mid): [str(lv) for lv in levels] for mid, levels in efforts.items() if isinstance(levels, list)}
        try:
            age = max(0.0, time.time() - float(data.get("ts") or 0))
        except (TypeError, ValueError):
            age = float(_DISK_TTL_SECONDS)
        return (parsed or None), age
    except Exception:
        return None, 0.0


def _seed_efforts(items: Any) -> Optional[dict[str, list[str]]]:
    """Seed memory + disk caches from a ``/v1/models`` payload."""
    global _efforts_cache
    parsed = _parse_efforts(items)
    if parsed is not None:
        with _efforts_lock:
            _efforts_cache = parsed
        _save_disk(parsed)
    return parsed


def _fetch_catalog_items(*, api_key: str = "", base_url: str = "", timeout: float = 8.0) -> Optional[list]:
    """Fetch the raw ``/v1/models`` ``data`` array. None on any failure."""
    import urllib.request

    from hermes_cli.urllib_security import open_credentialed_url

    req = urllib.request.Request((base_url or _base_url()).rstrip("/") + "/models")
    key = api_key or _resolve_api_key()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", _profile_user_agent())  # Router's WAF rejects the default urllib UA
    try:
        with open_credentialed_url(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        logger.debug("router: catalog fetch failed: %s", exc)
        return None
    items = data if isinstance(data, list) else data.get("data", [])
    return items if isinstance(items, list) else None


def _efforts_cache_only() -> Optional[dict[str, list[str]]]:
    """Memory, else the disk mirror (checked once per process). Never HTTP (hot-path safe)."""
    global _efforts_cache, _disk_checked
    with _efforts_lock:
        cached = _efforts_cache
    if cached is not None or _disk_checked:
        return cached
    _disk_checked = True
    parsed, age = _load_disk()
    if parsed is None:
        return None
    with _efforts_lock:
        _efforts_cache = cached = _efforts_cache if _efforts_cache is not None else parsed
    if age >= _DISK_TTL_SECONDS:
        _warm_efforts_async()
    return cached


def _warm_efforts_async() -> None:
    """Refresh the efforts cache in the background, at most once per process.

    Skipped under pytest (a mid-suite fetch makes cache state timing-dependent)
    and without a key (it would 401; the first authenticated fetch_models() seeds).
    """
    global _warm_started
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    with _efforts_lock:
        if _warm_started:
            return
        _warm_started = True
    if not _resolve_api_key():
        return

    def _refresh() -> None:
        items = _fetch_catalog_items()
        if items is not None:
            _seed_efforts(items)
    try:
        threading.Thread(target=_refresh, name="router-caps-warm", daemon=True).start()
    except Exception as exc:
        logger.debug("router: caps warmer failed to start: %s", exc)


class RouterProfile(ProviderProfile):
    """Ramp Router — Responses-only gateway with catalog-declared efforts."""

    def fetch_models(
        self, *, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 8.0
    ) -> Optional[list[str]]:
        """Live, key-scoped catalog; the same payload seeds the caps cache.
        Deduped but not sorted: Router's listing order is deliberate presentation."""
        items = _fetch_catalog_items(api_key=api_key or "", base_url=base_url or "", timeout=timeout)
        if items is None:
            return None
        _seed_efforts(items)
        return list(dict.fromkeys(str(i["id"]) for i in items if isinstance(i, dict) and i.get("id"))) or None

    def supported_reasoning_efforts(self, model: Optional[str]) -> Optional[tuple[str, ...]]:
        """Catalog-declared effort vocabulary (cache-only; cold cache -> None + warm)."""
        mid = str(model or "").strip()
        if not mid:
            return None
        efforts_by_id = _efforts_cache_only()
        if efforts_by_id is None:
            _warm_efforts_async()
            return None
        return tuple(efforts_by_id[mid]) if mid in efforts_by_id else None


router = RouterProfile(
    name="router", aliases=("ramp-router", "ramp", "router.com"), api_mode="codex_responses",
    display_name="Ramp Router",
    description="Ramp Router (router.com) — routes each request to the cheapest model that clears your quality bar",
    signup_url="https://app.router.com/keys",
    env_vars=("RAMP_ROUTER_API_KEY", "ROUTER_API_KEY", "RAMP_ROUTER_BASE_URL"), base_url=_base_url(),
    auth_type="api_key",
    # Router attributes coding-agent clients by UA prefix; its WAF rejects default UAs.
    default_headers={"User-Agent": f"Hermes-Agent/{_HERMES_VERSION}"},
    supports_vision=True, default_aux_model="gpt-5.4-mini",
    fallback_models=(),  # account-scoped IDs; the picker uses fetch_models()
)

register_provider(router)
