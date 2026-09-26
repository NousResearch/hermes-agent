"""Account-scoped xAI effort catalog; request-time reads never perform HTTP."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass, field

from agent.reasoning_effort import EFFORT_LADDER, XAI_GROK46_EFFORTS, XAI_LEGACY_EFFORTS
from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)
_TTL = 1200
_RETRY_DELAY = 300
_TIMEOUT = 5


@dataclass
class Catalog:
    efforts: dict[str, tuple[str, ...]] = field(default_factory=dict)
    model_ids: tuple[str, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)
    loaded: bool = False
    refresh_after: float = 0
    loading: bool = False
    ready: threading.Event = field(default_factory=threading.Event)


_catalogs: dict[tuple[str, str, str], Catalog] = {}
_lock = threading.Lock()


def parse_efforts(raw) -> tuple[str, ...] | None:
    """Empty explicitly disables the parameter; malformed is unknown, never disabled."""
    if not isinstance(raw, list) or any(not isinstance(v, str) or v not in EFFORT_LADDER[:-1] for v in raw):
        return None
    return tuple(level for level in EFFORT_LADDER if level in raw)


def _parse_catalog(payload):
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("Malformed model catalog")
    efforts, aliases, ids = {}, {}, []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        mid = item["id"].strip().lower()
        ids.append(mid)
        caps = item.get("capabilities")
        levels = parse_efforts(caps.get("reasoning_effort")) if isinstance(caps, dict) else None
        if levels is not None:
            efforts[mid] = levels
        published_aliases = item.get("aliases")
        for alias in published_aliases if isinstance(published_aliases, list) else []:
            if isinstance(alias, str):
                aliases[alias.strip().lower()] = mid
    # An explicit model ID always takes precedence over another model's alias.
    return efforts, tuple(ids), {k: v for k, v in aliases.items() if k not in ids}


def _fetch(base_url: str, api_key: str):
    from hermes_cli.urllib_security import open_credentialed_url
    from tools.xai_http import hermes_xai_default_headers

    req = urllib.request.Request(base_url + "/models", headers={
        **hermes_xai_default_headers(), "Authorization": "Bearer " + api_key, "Accept": "application/json",
    })
    with open_credentialed_url(req, timeout=_TIMEOUT) as response:
        body = response.read(4 * 1024 * 1024 + 1)
        if len(body) > 4 * 1024 * 1024:
            raise ValueError("Model catalog exceeds size limit")
        return _parse_catalog(json.loads(body))


def _refresh(entry: Catalog, base_url: str, api_key: str) -> None:
    try:
        efforts, ids, aliases = _fetch(base_url, api_key)
    except Exception as exc:
        # URLs, credentials and provider error bodies never belong in this diagnostic.
        logger.debug("xAI reasoning catalog unavailable (%s)", type(exc).__name__)
        with _lock:
            entry.refresh_after = time.monotonic() + _RETRY_DELAY
    else:
        with _lock:
            entry.efforts, entry.model_ids, entry.aliases, entry.loaded = efforts, ids, aliases, True
            entry.refresh_after = time.monotonic() + _TTL
    finally:
        with _lock:
            entry.loading = False
            entry.ready.set()


def prepare_catalog(base_url: str, api_key: str) -> tuple[str, str, str] | None:
    """Cold-load once before inference; stale entries refresh off-thread in their owner's scope."""
    if not isinstance(api_key, str) or not api_key or not base_url:
        return None
    from hermes_cli.config_providers import normalize_route_base_url
    base_url = normalize_route_base_url(base_url)
    fingerprint = hashlib.sha256(api_key.encode()).hexdigest()
    key = (hermes_home_key(), base_url, fingerprint)
    with _lock:
        entry = _catalogs.setdefault(key, Catalog())
        if time.monotonic() < entry.refresh_after:
            return key
        if entry.loading:
            wait = not entry.loaded
            start = False
        else:
            wait, start = False, True
            entry.loading = True
            entry.ready.clear()
        stale = entry.loaded
    if wait:
        if not entry.ready.wait(_TIMEOUT):
            with _lock:
                entry.refresh_after = time.monotonic() + _RETRY_DELAY
    elif start:
        from agent.memory_provider import spawn_context_thread
        spawn_context_thread(lambda: _refresh(entry, base_url, api_key), name="xai-reasoning-catalog").start()
        if not stale and not entry.ready.wait(_TIMEOUT):
            with _lock:
                entry.refresh_after = time.monotonic() + _RETRY_DELAY
    return key


def cached_efforts(key, model: str) -> tuple[str, ...] | None:
    if not key or key[0] != hermes_home_key():
        return None
    name = (model or "").strip().lower()
    with _lock:
        entry = _catalogs.get(key)
        if entry is None:
            return None
        name = name if name in entry.model_ids else name.removeprefix("x-ai/")
        return entry.efforts.get(entry.aliases.get(name, name))


def catalog_knows_model(key, model: str) -> bool:
    if not key or key[0] != hermes_home_key():
        return False
    name = (model or "").strip().lower().removeprefix("x-ai/")
    with _lock:
        entry = _catalogs.get(key)
        return bool(entry and (name in entry.model_ids or name in entry.aliases))


def cached_model_ids(key) -> list[str] | None:
    with _lock:
        entry = _catalogs.get(key)
        return list(entry.model_ids) if entry and entry.loaded else None


def legacy_efforts(model: str) -> tuple[str, ...] | None:
    # Compatibility when the endpoint omits metadata. Never extend by numeric version comparison.
    from agent.model_metadata import grok_supports_reasoning_effort, is_grok_46_family
    if grok_supports_reasoning_effort(model):
        return XAI_GROK46_EFFORTS if is_grok_46_family(model) else XAI_LEGACY_EFFORTS
    return None
