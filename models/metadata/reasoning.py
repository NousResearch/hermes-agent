"""Reasoning capability metadata from OpenRouter-shaped model catalogs.

This module owns catalog facts only: tri-state support, supported effort levels,
and whether reasoning is mandatory. It deliberately does not import Hermes CLI,
agent, provider policy, credentials, or request serializers. Runtime owners may
inject the catalog request function and the Nous catalog URL resolver.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_constants import (
    get_hermes_home,
    get_hermes_home_override,
    hermes_home_key,
    mkdir_under_hermes_home,
)
from models.metadata.types import ReasoningMetadata
from utils import atomic_json_write

logger = logging.getLogger(__name__)

Caps = dict[str, ReasoningMetadata | None]
CatalogRequester = Callable[..., Any]
UrlResolver = Callable[[], str]

# Canonical low-to-high reasoning effort vocabulary. Provider-specific modules
# consume this math; request formatting remains with the transport/application.
EFFORT_LADDER: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
)
CODEX_ASTRA_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
ASTRA_MODEL_IDS: frozenset[str] = frozenset({"gpt-6-astra", "gpt-6-astra-900k"})


def is_astra_model(model: Optional[str]) -> bool:
    return (model or "").strip().lower().rsplit("/", 1)[-1] in ASTRA_MODEL_IDS


def clamp_effort(
    effort: Optional[str],
    supported,
    overrides: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Clamp to the nearest weaker supported canonical effort without escalating cost."""
    requested = str(effort or "").strip().lower()
    if not requested or not supported:
        return effort
    supported_norm = [
        level
        for level in (str(item).strip().lower() for item in supported)
        if level in EFFORT_LADDER
    ]
    if not supported_norm or requested in supported_norm:
        return effort
    if overrides and overrides.get(requested) in supported_norm:
        return overrides[requested]
    if requested not in EFFORT_LADDER:
        return effort
    candidates = [level for level in supported_norm if level != "none"]
    if not candidates:
        return effort
    requested_idx = EFFORT_LADDER.index(requested)
    below = [
        level for level in candidates
        if EFFORT_LADDER.index(level) < requested_idx
    ]
    return (
        max(below, key=EFFORT_LADDER.index)
        if below
        else min(candidates, key=EFFORT_LADDER.index)
    )


def clamp_reasoning_effort_to_supported(
    effort: Optional[str], supported_efforts
) -> Optional[str]:
    """Generic OpenAI-compatible effort clamp used by provider metadata adapters."""
    return clamp_effort(effort, supported_efforts)

_OPENROUTER_CATALOG_URL = "https://openrouter.ai/api/v1/models"
_DEFAULT_NOUS_CATALOG_URL = "https://inference-api.nousresearch.com/v1/models"
_REASONING_CAPS_DISK_TTL_SECONDS = 24 * 3600


def _default_catalog_request(req: urllib.request.Request, *, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout)


# These are deliberately injectable seams. The lower metadata package supplies
# the source protocol; CLI/runtime layers provide Hermes' guarded opener and the
# profile-aware Nous endpoint name without passing credentials into this module.
_urlopen_model_catalog_request: CatalogRequester = _default_catalog_request
_catalog_user_agent = "HermesAgent"
_nous_catalog_url_resolver: UrlResolver | None = None


def configure_reasoning_metadata_sources(
    *, request: CatalogRequester | None = None, nous_url: UrlResolver | None = None,
    user_agent: str | None = None,
) -> None:
    """Inject transport/source naming owned by an upper runtime layer."""
    global _urlopen_model_catalog_request, _nous_catalog_url_resolver, _catalog_user_agent
    if request is not None:
        _urlopen_model_catalog_request = request
    if nous_url is not None:
        _nous_catalog_url_resolver = nous_url
    if user_agent is not None:
        _catalog_user_agent = user_agent


def parse_openrouter_reasoning_capabilities(item: Any) -> ReasoningMetadata | None:
    """Normalize one OpenRouter/Nous catalog entry's reasoning metadata.

    A malformed ``supported_parameters`` field is unknown. A valid list that
    omits ``reasoning`` is a definitive negative. A present reasoning object
    without a mandatory flag keeps the historical optional/False interpretation.
    """
    if not isinstance(item, dict):
        return None
    params = item.get("supported_parameters")
    if not isinstance(params, list):
        return None
    if "reasoning" not in params:
        return ReasoningMetadata(supported=False)
    reasoning = item.get("reasoning")
    if not isinstance(reasoning, dict):
        reasoning = {}
    raw_efforts = reasoning.get("supported_efforts")
    efforts: tuple[str, ...] | None = None
    if isinstance(raw_efforts, list):
        efforts = tuple(dict.fromkeys(str(e).strip().lower() for e in raw_efforts if str(e).strip()))
    return ReasoningMetadata(
        supported=True,
        supported_efforts=efforts,
        mandatory=reasoning.get("mandatory") is True,
    )


def _reasoning_caps_disk_path() -> Path:
    return get_hermes_home() / "cache" / "reasoning_caps.json"


def _read_reasoning_caps_disk() -> dict[str, Any]:
    try:
        with _reasoning_caps_disk_path().open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _serialize_caps(caps: Caps) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for model, value in caps.items():
        if isinstance(value, dict):
            value = _deserialize_cap(value)
        serialized[model] = None if value is None else {
            "supports_reasoning": value.supported,
            "supported_efforts": None if value.supported_efforts is None else list(value.supported_efforts),
            "mandatory": value.mandatory,
        }
    return serialized


def _deserialize_cap(value: Any) -> ReasoningMetadata | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    supported = value.get("supports_reasoning")
    if supported not in (True, False, None):
        return None
    raw_efforts = value.get("supported_efforts")
    efforts = None
    if isinstance(raw_efforts, list):
        efforts = tuple(str(item).strip().lower() for item in raw_efforts if str(item).strip())
    mandatory = value.get("mandatory")
    if mandatory not in (True, False, None):
        mandatory = None
    return ReasoningMetadata(supported=supported, supported_efforts=efforts, mandatory=mandatory)


def _decode_caps(raw: Any) -> Caps | None:
    if not isinstance(raw, dict) or not raw:
        return None
    return {str(model): _deserialize_cap(value) for model, value in raw.items()}


def _load_reasoning_caps_disk(url: str) -> tuple[Caps | None, float]:
    entry = _read_reasoning_caps_disk().get(url)
    if not isinstance(entry, dict):
        return None, 0.0
    caps = _decode_caps(entry.get("caps"))
    if caps is None:
        return None, 0.0
    try:
        age = max(0.0, time.time() - float(entry.get("ts") or 0))
    except (TypeError, ValueError):
        age = float(_REASONING_CAPS_DISK_TTL_SECONDS)
    return caps, age


def _save_reasoning_caps_disk(url: str, caps: Caps) -> None:
    """Merge *url*'s catalog into the shared disk mirror, atomically."""
    try:
        data = _read_reasoning_caps_disk()
        data[url] = {"ts": time.time(), "caps": _serialize_caps(caps)}
        mkdir_under_hermes_home(_reasoning_caps_disk_path().parent)
        atomic_json_write(_reasoning_caps_disk_path(), data, indent=0, separators=(",", ":"))
    except Exception as exc:
        logger.debug("Failed to save reasoning-caps disk cache: %s", exc)


def _warm_reasoning_caps_async(refresh: Callable[[], Any]) -> None:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    threading.Thread(
        target=contextvars.copy_context().run,
        args=(refresh,),
        name="reasoning-caps-warm",
        daemon=True,
    ).start()


def _hydrate_reasoning_caps_from_disk(url: str, refresh: Callable[[], Any]) -> Caps | None:
    caps, age = _load_reasoning_caps_disk(url)
    if caps is not None and age >= _REASONING_CAPS_DISK_TTL_SECONDS:
        _warm_reasoning_caps_async(refresh)
    return caps


def _seed_reasoning_caps(url: str, items: Any) -> Caps | None:
    """Parse a catalog ``data`` array and mirror it for *url* without fetching."""
    if not isinstance(items, list):
        return None
    caps_by_id: Caps = {}
    for item in items:
        model_id = str(item.get("id") or "").strip() if isinstance(item, dict) else ""
        if model_id:
            caps_by_id[model_id] = parse_openrouter_reasoning_capabilities(item)
    if not caps_by_id:
        return None
    _save_reasoning_caps_disk(url, caps_by_id)
    return caps_by_id


def _fetch_reasoning_caps_catalog(url: str, timeout: float) -> Caps | None:
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": _catalog_user_agent},
        )
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return None
    return _seed_reasoning_caps(url, payload.get("data")) if isinstance(payload, dict) else None


@dataclass(frozen=True)
class _CapsSource:
    cache: str
    failed_at: str
    disk_checked: str
    warm_started: str
    url: UrlResolver
    per_profile: bool = False

    def get(self, slot: str):
        module = sys.modules[__name__]
        if get_hermes_home_override() is None or not self.per_profile:
            return getattr(module, getattr(self, slot))
        return _PROFILE_SLOTS.get((hermes_home_key(), getattr(self, slot)),
                                  False if slot in ("disk_checked", "warm_started") else None)

    def set(self, slot: str, value: Any) -> None:
        module = sys.modules[__name__]
        if get_hermes_home_override() is None or not self.per_profile:
            setattr(module, getattr(self, slot), value)
            return
        _PROFILE_SLOTS[(hermes_home_key(), getattr(self, slot))] = value


_PROFILE_SLOTS: dict[tuple[str, str], Any] = {}

_openrouter_reasoning_caps_cache: Caps | None = None
_openrouter_reasoning_caps_failed_at: float | None = None
_openrouter_caps_disk_checked = False
_openrouter_caps_warm_started = False
_nous_reasoning_caps_cache: Caps | None = None
_nous_reasoning_caps_failed_at: float | None = None
_nous_caps_disk_checked = False
_nous_caps_warm_started = False


def _fetch_caps(src: _CapsSource, timeout: float = 6.0, *, force: bool = False) -> Caps | None:
    cached = src.get("cache")
    if cached is not None and not force:
        return cached
    failed_at = src.get("failed_at")
    if failed_at is not None and (time.monotonic() - failed_at) < 60:
        return None
    caps_by_id = _fetch_reasoning_caps_catalog(src.url(), timeout)
    if caps_by_id is None:
        src.set("failed_at", time.monotonic())
        return None
    src.set("cache", caps_by_id)
    return caps_by_id


def _caps_cached(src: _CapsSource) -> Caps | None:
    if src.get("cache") is None and not src.get("disk_checked"):
        src.set("disk_checked", True)
        src.set("cache", _hydrate_reasoning_caps_from_disk(src.url(), lambda: _fetch_caps(src, force=True)))
    return src.get("cache")


def _model_caps(src: _CapsSource, model_id: Optional[str], *, timeout: float, allow_fetch: bool) -> ReasoningMetadata | None:
    model = str(model_id or "").strip()
    if not model:
        return None
    caps_by_id = _caps_cached(src)
    if caps_by_id is None and allow_fetch:
        caps_by_id = _fetch_caps(src, timeout=timeout)
    value = caps_by_id.get(model) if caps_by_id is not None else None
    return _deserialize_cap(value) if isinstance(value, dict) else value


def _warm_caps_async(src: _CapsSource) -> None:
    if src.get("warm_started") or _caps_cached(src) is not None:
        return
    src.set("warm_started", True)
    _warm_reasoning_caps_async(lambda: _fetch_caps(src, force=True))


def refresh_reasoning_caps_async(provider: Optional[str]) -> None:
    source = {
        "nous": _NOUS_CAPS,
        "nous-portal": _NOUS_CAPS,
        "nousresearch": _NOUS_CAPS,
        "openrouter": _OPENROUTER_CAPS,
    }.get(str(provider or "").strip().lower())
    if source is not None:
        _warm_reasoning_caps_async(lambda: _fetch_caps(source, force=True))


def _nous_base_url_override() -> str:
    """Read the active profile's non-secret Nous endpoint override without auth resolution."""
    value = os.environ.get("NOUS_INFERENCE_BASE_URL", "").strip()
    if value:
        return value.strip('"\'')
    try:
        env_path = get_hermes_home() / ".env"
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition("=")
            if separator and key.strip() == "NOUS_INFERENCE_BASE_URL":
                return raw.strip().strip('"\'')
    except (OSError, UnicodeDecodeError):
        pass
    return ""


def nous_catalog_url() -> str:
    """Return the injected Nous catalog URL, or the active profile's non-secret default/env name."""
    if _nous_catalog_url_resolver is not None:
        return _nous_catalog_url_resolver()
    base_url = _nous_base_url_override() or _DEFAULT_NOUS_CATALOG_URL.removesuffix("/v1/models")
    return f"{base_url.rstrip('/').removesuffix('/v1')}/v1/models"


_OPENROUTER_CAPS = _CapsSource(
    "_openrouter_reasoning_caps_cache", "_openrouter_reasoning_caps_failed_at",
    "_openrouter_caps_disk_checked", "_openrouter_caps_warm_started",
    lambda: _OPENROUTER_CATALOG_URL,
)
_NOUS_CAPS = _CapsSource(
    "_nous_reasoning_caps_cache", "_nous_reasoning_caps_failed_at",
    "_nous_caps_disk_checked", "_nous_caps_warm_started",
    lambda: nous_catalog_url(), per_profile=True,
)


def openrouter_model_reasoning_capabilities(
    model_id: Optional[str], *, timeout: float = 6.0, allow_fetch: bool = False,
) -> ReasoningMetadata | None:
    """Cache-only by default; return unknown when the catalog is not available."""
    return _model_caps(_OPENROUTER_CAPS, model_id, timeout=timeout, allow_fetch=allow_fetch)


def nous_model_reasoning_capabilities(
    model_id: Optional[str], *, timeout: float = 6.0, allow_fetch: bool = False,
) -> ReasoningMetadata | None:
    """Nous counterpart of :func:`openrouter_model_reasoning_capabilities`."""
    return _model_caps(_NOUS_CAPS, model_id, timeout=timeout, allow_fetch=allow_fetch)


def warm_openrouter_reasoning_caps_async() -> None:
    _warm_caps_async(_OPENROUTER_CAPS)


def warm_nous_reasoning_caps_async() -> None:
    _warm_caps_async(_NOUS_CAPS)


__all__ = [
    "ReasoningMetadata",
    "EFFORT_LADDER",
    "CODEX_ASTRA_EFFORTS",
    "ASTRA_MODEL_IDS",
    "is_astra_model",
    "clamp_effort",
    "clamp_reasoning_effort_to_supported",
    "_OPENROUTER_CATALOG_URL",
    "_reasoning_caps_disk_path",
    "_save_reasoning_caps_disk",
    "_seed_reasoning_caps",
    "configure_reasoning_metadata_sources",
    "nous_catalog_url",
    "nous_model_reasoning_capabilities",
    "openrouter_model_reasoning_capabilities",
    "parse_openrouter_reasoning_capabilities",
    "refresh_reasoning_caps_async",
    "warm_nous_reasoning_caps_async",
    "warm_openrouter_reasoning_caps_async",
]
