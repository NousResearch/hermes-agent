"""Ramp Router (router.com) provider plugin for Hermes Agent.

Provider profile for `Ramp Router <https://docs.router.com>`_, Ramp's LLM
gateway: one OpenAI Responses-compatible endpoint at
``https://api.router.com/v1`` that routes each request across upstream
providers (OpenAI, Anthropic, xAI, Fireworks, ...) and handles fallbacks and
spend controls server-side.

Wire notes (verified live against api.router.com, Aug 2026):

* **Responses API is the native wire.** Router serves ``GET /v1/models``
  and ``POST /v1/responses``; ``POST /v1/chat/completions`` is only a
  minimal compatibility shim (added Aug 2026) that translates onto
  Responses. Per-model reasoning-effort validation, reasoning summaries,
  and prompt caching are Responses-surface features, so
  ``api_mode="codex_responses"`` plus the ``api.router.com`` host mandate
  in ``hermes_cli/providers.py`` keep every path on the native wire —
  the same shape as the ``api.openai.com`` mandate.
* **Account-scoped catalog.** Valid model IDs are whatever the key's
  ``GET /v1/models`` returns (BYOK accounts see extra entries), so this
  profile ships **no** ``fallback_models`` — the picker relies on the live
  fetch, per Router's own guidance to never hardcode model names.
* **Strict reasoning-effort validation.** Router validates
  ``reasoning.effort`` against each model's catalog-declared vocabulary and
  returns HTTP 400 ``invalid-argument`` on a level the model does not accept
  (e.g. ``max`` on grok-4.6), and 400 ``unsupported_parameter`` when a
  non-reasoning model (gpt-4.1 family, gpt-4o, ...) receives any reasoning
  field. The catalog publishes the vocabulary per model
  (``router.capabilities.reasoning``), so ``supported_reasoning_efforts``
  below feeds the codex transport's clamp from a cached copy of it.
* **Everything else passes through.** ``store: false``, ``prompt_cache_key``,
  ``include: ["reasoning.encrypted_content"]``, and ``reasoning.summary`` are
  accepted on all models (ignored where a backend cannot honor them), tools /
  ``parallel_tool_calls`` / streaming SSE work across backends, and encrypted
  reasoning replay round-trips on OpenAI-served models — so the generic
  Responses transport path needs no Router-specific request surgery.

The capability cache mirrors the OpenRouter reasoning-caps design in
``hermes_cli/models.py``: cache-only lookups on the per-request hot path
(never HTTP), seeded for free whenever ``fetch_models()`` runs (picker,
setup, doctor), hydrated from a disk mirror across processes, and refreshed
by a background warmer when cold or stale.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

ROUTER_DEFAULT_BASE_URL = "https://api.router.com/v1"

#: Efforts-by-model cache: ``model id -> list of accepted effort levels``.
#: ``[]`` means the catalog says the model accepts NO reasoning parameters
#: (``reasoning.supported: false``) — the transport must omit reasoning
#: entirely. A model absent from the dict is unknown (custom/BYOK route or
#: vocabulary not published) and callers fall back to their defaults.
_efforts_cache: Optional[dict[str, list[str]]] = None
_efforts_lock = threading.Lock()
_warm_started = False
_disk_checked = False

#: Disk-mirror staleness bound. Vocabularies change rarely; a stale verdict
#: beats no verdict, so a past-TTL mirror is still served while a background
#: refresh runs (same policy as the OpenRouter caps mirror).
_DISK_TTL_SECONDS = 24 * 60 * 60


def _base_url() -> str:
    """Allow a base-URL override via ``RAMP_ROUTER_BASE_URL``."""
    return os.getenv("RAMP_ROUTER_BASE_URL", "").strip().rstrip("/") or ROUTER_DEFAULT_BASE_URL


def _resolve_api_key() -> str:
    """Resolve the Router key from .env / environment, preferring dotenv.

    ``RAMP_ROUTER_API_KEY`` is Router's documented variable;
    ``ROUTER_API_KEY`` is accepted as a convenience alias. Falls back to the
    raw environment when the hermes_cli helper is unavailable (e.g. stripped
    test environments).
    """
    resolvers = []
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv

        resolvers.append(get_env_value_prefer_dotenv)
    except Exception:
        pass
    resolvers.append(lambda var: os.environ.get(var, ""))
    for resolve in resolvers:
        for var in ("RAMP_ROUTER_API_KEY", "ROUTER_API_KEY"):
            try:
                value = str(resolve(var) or "").strip()
            except Exception:
                value = ""
            if value:
                return value
    return ""


def _parse_efforts(items: Any) -> Optional[dict[str, list[str]]]:
    """Parse a Router ``/v1/models`` ``data`` array into the efforts map.

    Returns None when the array has no usable entries, which callers treat
    as a failed fetch rather than caching an empty verdict.
    """
    if not isinstance(items, list):
        return None
    try:
        from agent.reasoning_effort import EFFORT_LADDER

        known_levels = set(EFFORT_LADDER)
    except Exception:
        known_levels = None
    efforts_by_id: dict[str, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        router_meta = item.get("router")
        reasoning = None
        if isinstance(router_meta, dict):
            capabilities = router_meta.get("capabilities")
            if isinstance(capabilities, dict):
                reasoning = capabilities.get("reasoning")
        if not isinstance(reasoning, dict):
            continue
        if reasoning.get("supported") is False:
            # Definitive negative: any reasoning field 400s on this model.
            efforts_by_id[mid] = []
            continue
        levels = [
            str(entry.get("value") or "").strip()
            for entry in reasoning.get("efforts") or []
            if isinstance(entry, dict) and str(entry.get("value") or "").strip()
        ]
        if known_levels is not None:
            # clamp_effort silently ignores ladder-unknown levels, and an
            # all-unknown vocabulary would pass the requested effort through
            # unclamped straight to a Router 400 — so a new vendor tier is
            # dropped at ingest and fails loudly here instead.
            unknown = [level for level in levels if level not in known_levels]
            if unknown:
                logger.info(
                    "router: model %s publishes unrecognized reasoning effort "
                    "level(s) %s; ignoring them (update agent/reasoning_effort "
                    "EFFORT_LADDER to adopt new vendor tiers)",
                    mid,
                    unknown,
                )
                levels = [level for level in levels if level in known_levels]
        if levels:
            efforts_by_id[mid] = levels
        # supported=True with no (recognized) vocabulary -> leave the model
        # out (unknown), so the transport keeps its default clamp behavior.
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
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"ts": time.time(), "efforts": efforts_by_id}),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception as exc:
        logger.debug("router: caps disk mirror write failed: %s", exc)


def _load_disk() -> tuple[Optional[dict[str, list[str]]], float]:
    path = _disk_path()
    if path is None:
        return None, 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        efforts = data.get("efforts")
        if not isinstance(efforts, dict) or not efforts:
            return None, 0.0
        parsed = {
            str(mid): [str(level) for level in levels]
            for mid, levels in efforts.items()
            if isinstance(levels, list)
        }
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
    if parsed is None:
        return None
    with _efforts_lock:
        _efforts_cache = parsed
    _save_disk(parsed)
    return parsed


def _fetch_catalog_items(
    *, api_key: str = "", base_url: str = "", timeout: float = 8.0
) -> Optional[list]:
    """Fetch the raw ``/v1/models`` ``data`` array. None on any failure."""
    url = (base_url or _base_url()).rstrip("/") + "/models"
    import urllib.request

    from hermes_cli.urllib_security import open_credentialed_url

    req = urllib.request.Request(url)
    key = api_key or _resolve_api_key()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Accept", "application/json")
    # Router sits behind a WAF that rejects the default Python-urllib UA.
    req.add_header("User-Agent", _profile_user_agent())
    try:
        with open_credentialed_url(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        logger.debug("router: catalog fetch failed: %s", exc)
        return None
    items = data if isinstance(data, list) else data.get("data", [])
    return items if isinstance(items, list) else None


def _efforts_cache_only() -> Optional[dict[str, list[str]]]:
    """Memory, else the disk mirror. Never HTTP (hot-path safe)."""
    global _efforts_cache, _disk_checked
    with _efforts_lock:
        cached = _efforts_cache
    if cached is not None:
        return cached
    if _disk_checked:
        return None
    _disk_checked = True
    parsed, age = _load_disk()
    if parsed is None:
        return None
    with _efforts_lock:
        if _efforts_cache is None:
            _efforts_cache = parsed
        cached = _efforts_cache
    if age >= _DISK_TTL_SECONDS:
        _warm_efforts_async()
    return cached


def _warm_efforts_async() -> None:
    """Refresh the efforts cache in the background, at most once per process."""
    global _warm_started
    with _efforts_lock:
        if _warm_started:
            return
        _warm_started = True
    if not _resolve_api_key():
        # Without a key the fetch would 401; the first authenticated
        # fetch_models() (picker/setup/doctor) seeds the cache instead.
        return

    def _refresh() -> None:
        items = _fetch_catalog_items()
        if items is not None:
            _seed_efforts(items)

    try:
        threading.Thread(
            target=_refresh, name="router-caps-warm", daemon=True
        ).start()
    except Exception as exc:
        logger.debug("router: caps warmer failed to start: %s", exc)


class RouterProfile(ProviderProfile):
    """Ramp Router — Responses-only gateway with catalog-declared efforts."""

    def fetch_models(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 8.0,
    ) -> Optional[list[str]]:
        """Fetch the live, key-scoped catalog and seed the caps cache.

        One request serves both consumers: the picker gets the model IDs and
        the reasoning-vocabulary mirror is left warm at no extra network
        cost (the same document carries both).
        """
        items = _fetch_catalog_items(
            api_key=api_key or "", base_url=base_url or "", timeout=timeout
        )
        if items is None:
            return None
        _seed_efforts(items)
        # Deduped but not sorted: Router's listing order is deliberate
        # presentation (featured/current models first), so the picker keeps it.
        ids = list(
            dict.fromkeys(
                str(item["id"])
                for item in items
                if isinstance(item, dict) and item.get("id")
            )
        )
        return ids or None

    def supported_reasoning_efforts(
        self, model: Optional[str]
    ) -> Optional[tuple[str, ...]]:
        """Catalog-declared effort vocabulary for *model* (cache-only).

        Router 400s on efforts outside a model's published set and on any
        reasoning field for non-reasoning models, so the codex transport
        clamps (or suppresses) from this verdict. Cold cache returns None —
        the transport keeps its defaults — and kicks a background warmer so
        the next turn is covered.
        """
        mid = str(model or "").strip()
        if not mid:
            return None
        efforts_by_id = _efforts_cache_only()
        if efforts_by_id is None:
            _warm_efforts_async()
            return None
        levels = efforts_by_id.get(mid)
        if levels is None:
            return None
        return tuple(levels)


router = RouterProfile(
    name="router",
    aliases=("ramp-router", "ramp", "router.com"),
    api_mode="codex_responses",
    display_name="Ramp Router",
    description="Ramp Router (router.com) — routes each request to the cheapest model that clears your quality bar",
    signup_url="https://app.router.com/keys",
    # RAMP_ROUTER_API_KEY is Router's documented variable; ROUTER_API_KEY is
    # a convenience alias. RAMP_ROUTER_BASE_URL overrides the endpoint
    # (auth.py picks it up as the registry's base_url_env_var).
    env_vars=("RAMP_ROUTER_API_KEY", "ROUTER_API_KEY", "RAMP_ROUTER_BASE_URL"),
    base_url=_base_url(),
    auth_type="api_key",
    # Identify Hermes traffic to the gateway (Router attributes coding-agent
    # clients by User-Agent prefix, the way it already recognizes OpenCode's
    # versioned UA) — and Router's WAF rejects blank/default client UAs.
    default_headers={"User-Agent": f"Hermes-Agent/{_HERMES_VERSION}"},
    # Most of the catalog's frontier routes accept image input; capability is
    # still model-dependent and governed by the live catalog.
    supports_vision=True,
    # Cheap, reasoning-capable, and vision-capable — safe for auxiliary tasks
    # (compaction, titles, vision) when Router is the main provider. Also the
    # model Router's own docs use as their example.
    default_aux_model="gpt-5.4-mini",
    # Deliberately empty: model IDs are account-scoped (BYOK accounts see
    # extra entries) and Router's docs say to read the catalog at runtime
    # rather than hardcode names. The picker uses fetch_models() above.
    fallback_models=(),
)

register_provider(router)
