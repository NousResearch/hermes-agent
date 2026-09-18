"""OpenRouter per-model provider-routing screen (second step after picking a model).

Picking ``deepseek/deepseek-v4.1-flash`` on OpenRouter picks a *model*, not who serves it —
that is decided per request by OpenRouter among ~20 upstream providers whose price, context,
quantization and tool support differ. This module lets the user see those providers and pin
the ones they want, right after the model is chosen.

Two surfaces drive it: the interactive CLI ``/model`` picker (``cli_model_switch_mixin`` stages
``routing`` / ``routing_sort``) and the ``hermes model`` console flow via :func:`configure_after_selection`.

Writes ONLY ``provider_routing.models.<model>`` (see
:func:`hermes_constants.resolve_per_model_provider_routing`, consumed by
``agent.chat_completion_helpers._provider_preferences_for_agent``): a pin for one model must
never leak into every other model on the account, which is exactly what the flat
``provider_routing`` keys would do. Existing keys of that entry (``only`` / ``ignore`` /
``require_parameters`` / ``data_collection``) are preserved; this screen owns ``order``
and ``sort``.

Selection maps to ``order`` (a soft preference that keeps OpenRouter's fallbacks), never to
``only``: a hard whitelist breaks the agent loop the moment the pinned provider is down or
does not serve ``tools``.

Endpoint data comes from the public ``/models/{id}/endpoints`` API (no API key needed) and is
cached on disk — a picker screen must not block on a network round-trip.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.curses_ui import curses_checklist, curses_radiolist

logger = logging.getLogger(__name__)

# Keys this screen may write. Anything else in a user's entry is left untouched.
ROUTING_KEYS = ("order", "sort", "only", "ignore", "require_parameters", "data_collection")
# Sort values OpenRouter accepts (mirrors chat_completion_helpers._OPENROUTER_PROVIDER_SORT_VALUES).
SORT_ROWS: Tuple[Tuple[str, str], ...] = (
    ("", "none — keep OpenRouter's default ranking"),
    ("price", "price — cheapest provider first"),
    ("throughput", "throughput — fastest tokens per second"),
    ("latency", "latency — lowest time-to-first-token"),
)
CLEAR_LABEL = "— clear this model's routing (use OpenRouter default) —"
# A single-provider model has nothing to choose: the screen is pure friction there.
MIN_PROVIDERS = 2
_ENDPOINT_TTL = 6 * 3600.0
_ANON_TTL = 900.0  # perf-less (keyless) response: recheck soon in case a key appears
_TIMEOUT = 8.0


# --------------------------------------------------------------------------- endpoint data


def _cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "openrouter_endpoints_cache.json"


def _read_cache() -> Dict[str, Any]:
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache(cache: Dict[str, Any]) -> None:
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache), encoding="utf-8")
    except Exception as exc:  # a cache write must never break the picker
        logger.debug("endpoint cache write failed: %s", exc)


def _endpoints_url(model_id: str, base_url: str = "") -> str:
    from urllib.parse import quote
    from hermes_constants import OPENROUTER_BASE_URL
    base = (base_url or OPENROUTER_BASE_URL).rstrip("/")
    return f"{base}/models/{quote(model_id, safe='/')}/endpoints"


def _api_key() -> str:
    """OpenRouter key when available — only the keyed response carries throughput/latency."""
    try:
        from hermes_cli.config import get_env_value
        return (get_env_value("OPENROUTER_API_KEY") or "").strip()
    except Exception:
        import os
        return (os.getenv("OPENROUTER_API_KEY") or "").strip()


def _get_json(url: str, *, timeout: float = _TIMEOUT, api_key: str = "") -> Optional[dict]:
    headers = {"User-Agent": "hermes-agent", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:
        logger.debug("endpoint fetch failed for %s: %s", url, exc)
        return None


def _has_perf_stats(endpoints: List[dict]) -> bool:
    """True when the response carries throughput/latency (keyed responses only)."""
    return any(isinstance(ep, dict) and ep.get("throughput_last_30m") for ep in endpoints)


def fetch_model_endpoints(
    model_id: str, *, force_refresh: bool = False, base_url: str = "", timeout: float = _TIMEOUT,
) -> List[dict]:
    """Raw ``endpoints`` array for *model_id*; cached on disk, stale-served when offline.

    Pricing/context/uptime come back anonymously; OpenRouter releases throughput and latency
    percentiles only for keyed requests, so the key is sent when one is configured and a
    perf-less response is cached briefly (a later keyed fetch must not sit behind it for hours).
    """
    key = (model_id or "").strip()
    if not key:
        return []
    cached = _read_cache().get(key)
    endpoints = cached.get("endpoints") if isinstance(cached, dict) else None
    ttl = _ENDPOINT_TTL if isinstance(cached, dict) and cached.get("perf") else _ANON_TTL
    if (not force_refresh and isinstance(cached, dict)
            and (time.time() - float(cached.get("at") or 0)) < ttl):
        return endpoints if isinstance(endpoints, list) else []

    payload = _get_json(_endpoints_url(key, base_url), timeout=timeout, api_key=_api_key())
    fresh = ((payload or {}).get("data") or {}).get("endpoints")
    if not isinstance(fresh, list):
        return endpoints if isinstance(endpoints, list) else []
    cache = _read_cache()
    cache[key] = {"at": time.time(), "perf": _has_perf_stats(fresh), "endpoints": fresh}
    _write_cache(cache)
    return fresh


def _per_mtok(raw: Any) -> Optional[float]:
    """Token price string (USD per token) -> USD per million tokens, or None."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value * 1_000_000.0


def _pct(raw: Any, key: str = "p50") -> Optional[float]:
    value = raw.get(key) if isinstance(raw, dict) else raw
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _endpoint_row(tag: str, ep: dict) -> dict:
    pricing: Dict[str, Any] = {}
    if isinstance(ep.get("pricing"), dict):
        pricing = ep["pricing"]
    supported = ep.get("supported_parameters")
    return {
        "tag": tag,
        "name": str(ep.get("provider_name") or tag),
        "prompt": _per_mtok(pricing.get("prompt")),
        "completion": _per_mtok(pricing.get("completion")),
        "cache_read": _per_mtok(pricing.get("input_cache_read")),
        "context": ep.get("context_length") if isinstance(ep.get("context_length"), int) else None,
        "quantization": str(ep.get("quantization") or ""),
        "tool_capable": bool(isinstance(supported, list) and "tools" in supported),
        "uptime": _pct(ep.get("uptime_last_30m")),
        "throughput": _pct(ep.get("throughput_last_30m")),
        "latency": _pct(ep.get("latency_last_30m")),
        "status": ep.get("status"),
    }


def provider_rows(model_id: str, *, force_refresh: bool = False, base_url: str = "") -> List[dict]:
    """One row per selectable provider tag, cheapest first, tool support flagged.

    A tag can appear several times (``baseten/fp8`` at two context lengths); the cheapest
    variant wins so the list never shows the same pin twice.
    """
    best: Dict[str, Tuple[float, dict]] = {}
    for ep in fetch_model_endpoints(model_id, force_refresh=force_refresh, base_url=base_url):
        if not isinstance(ep, dict):
            continue
        tag = str(ep.get("tag") or ep.get("provider_name") or "").strip().lower()
        if not tag:
            continue
        row = _endpoint_row(tag, ep)
        price = row["prompt"] if row["prompt"] is not None else float("inf")
        prev = best.get(tag)
        if prev is None or price < prev[0]:
            best[tag] = (price, row)
    return [row for _price, row in sorted(best.values(), key=lambda item: (item[0], item[1]["tag"]))]


def _money(value: Optional[float]) -> str:
    return f"${value:.2f}" if isinstance(value, float) else "  ?  "


def _ctx_label(context: Optional[int]) -> str:
    if not isinstance(context, int) or context <= 0:
        return "     ?"
    if context >= 1_000_000:
        return f"{context / 1_000_000:.1f}M"
    return f"{context // 1000}K"


def _format_row(row: dict, tag_width: int) -> str:
    """One provider line. Every field is fixed-width: a one-character drift (``100.0%`` vs
    ``99.9%``) used to push rows past the picker panel and wrap them. ``⚠`` marks a provider that
    does not accept ``tools`` — pinning one silently breaks the agent loop."""
    ups = f"{row['uptime']:.1f}%" if isinstance(row["uptime"], float) else "?"
    tps = f"{row['throughput']:.0f}t/s" if isinstance(row["throughput"], float) else "?"
    lat = f"{row['latency'] / 1000:.1f}s" if isinstance(row["latency"], float) else "?"
    mark = "  " if row["tool_capable"] else "⚠ "
    return (
        f"{mark}{row['tag']:<{tag_width}} {_money(row['prompt'])}/{_money(row['completion'])}  "
        f"{_ctx_label(row['context']):>5}  up {ups:>6}  {tps:>8}  p50 {lat:>4}"
    ).rstrip()


def row_label(row: dict) -> str:
    return _format_row(row, 20)


def row_labels(rows: List[dict]) -> List[str]:
    """Labels for a whole list, sharing one tag width so the columns line up."""
    width = max((len(r["tag"]) for r in rows), default=15)
    return [_format_row(r, width) for r in rows]


# --------------------------------------------------------------------------- config plumbing


def existing_model_routing(model_id: str) -> Tuple[str, dict]:
    """``(config key, entry)`` for the entry matching *model_id* (spelling-tolerant).

    Returns ``("", {})`` when the model has no per-model entry.
    """
    from hermes_constants import _canonical_model_variants
    from hermes_cli.config import load_config_readonly
    try:
        pr = load_config_readonly().get("provider_routing")
    except Exception:
        return "", {}
    models = (pr or {}).get("models") if isinstance(pr, dict) else None
    if not isinstance(models, dict):
        return "", {}
    for variant in _canonical_model_variants(model_id):
        entry = models.get(variant)
        if isinstance(entry, dict):
            return variant, entry
    return "", {}


def _clean_entry(entry: dict) -> dict:
    clean: dict[str, Any] = {}
    for key in ROUTING_KEYS:
        value = entry.get(key)
        if value in (None, "", [], {}):
            continue
        if key in ("order", "only", "ignore"):
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                continue
            deduped: list[str] = []
            for item in value:
                slug = str(item).strip().lower()
                if slug and slug not in deduped:
                    deduped.append(slug)
            if not deduped:
                continue
            value = deduped
        if key == "sort":
            value = str(value).strip().lower()
            if value not in {s for s, _ in SORT_ROWS if s}:
                continue
        if key == "require_parameters" and not value:
            continue  # False is the flat default — writing it is noise, not intent
        clean[key] = value
    return clean


def save_model_routing(model_id: str, routing: Optional[dict]) -> dict:
    """Write (or clear, when *routing* is empty) ``provider_routing.models.<model_id>``.

    This screen owns ``order``/``sort`` only: keys it does not manage (``only``, ``ignore``,
    ``require_parameters``, ``data_collection``) survive from the existing entry, and other
    models plus the flat keys are untouched. A ``{}``/``None`` *routing* removes the model's
    entry entirely (spelling variants included) — the only way back to OpenRouter's default.
    """
    from hermes_cli.config import load_config, save_config
    key = (model_id or "").strip()
    if not key:
        return {}
    clean = _clean_entry(routing or {})

    config = load_config()
    pr = config.get("provider_routing")
    if not isinstance(pr, dict):
        pr = {}
    models = pr.get("models")
    if not isinstance(models, dict):
        models = {}

    matched_key, existing = existing_model_routing(key)
    for stale in {key, matched_key} - {""}:
        models.pop(stale, None)

    if clean:
        preserved = {k: v for k, v in existing.items() if k not in ("order", "sort")}
        models[key] = {**preserved, **clean}
    if models:
        pr["models"] = models
    else:
        pr.pop("models", None)
    if pr:
        config["provider_routing"] = pr
    else:
        config.pop("provider_routing", None)
    save_config(config)
    return models.get(key, {}) if clean else {}


def rows_footer(rows: List[dict]) -> str:
    """Column hint printed above the curses list so the numbers are readable without a legend."""
    del rows
    return (f"{'provider':<22} {'$/Mtok in/out':<11} {'ctx':>5}  {'uptime':>9}  {'tput':>8}  "
            f"{'lat':>8}  (⚠ = no tool support)")


def describe_routing(routing: Optional[dict]) -> str:
    """One-line human summary of what was persisted."""
    if routing is None:
        return "routing unchanged"
    if not routing:
        return "routing cleared (OpenRouter default)"
    parts = []
    if routing.get("order"):
        parts.append("order=" + ",".join(routing["order"]))
    if routing.get("sort"):
        parts.append(f"sort={routing['sort']}")
    return " · ".join(parts) or "routing unchanged"


def offloadable_providers(rows: List[dict]) -> List[dict]:
    """Rows that would silently break the agent loop if pinned (no ``tools`` support)."""
    return [r for r in rows if not r.get("tool_capable")]


# --------------------------------------------------------------------------- interactive screen


def _summary(rows: List[dict], chosen: set) -> str:
    picked = [rows[i] for i in sorted(chosen) if 0 <= i < len(rows)]
    if not picked:
        return "no provider selected — Enter skips, routing stays as-is"
    return "  ".join(f"{r['tag']} {_money(r['prompt'])}" for r in picked) + "  $/Mtok"


def prompt_provider_routing(
    model_id: str, *, provider: str = "openrouter", base_url: str = "", force_refresh: bool = False,
) -> Optional[dict]:
    """Second picker step: choose provider order + sort for *model_id*.

    Returns the routing dict to persist, ``{}`` to clear the model's entry, or ``None`` when the
    user skipped / the screen would be pointless (non-OpenRouter provider, single endpoint, no TTY).
    Callers persist the result via :func:`save_model_routing`.
    """
    if (provider or "").strip().lower() != "openrouter":
        return None
    if not sys.stdin.isatty():
        return None
    try:
        rows = provider_rows(model_id, force_refresh=force_refresh, base_url=base_url)
    except Exception as exc:
        logger.debug("provider rows unavailable for %s: %s", model_id, exc)
        return None
    if len(rows) < MIN_PROVIDERS:
        return None

    _key, existing = existing_model_routing(model_id)
    current_order = [str(t).strip().lower() for t in (existing.get("order") or [])]
    clear_idx = len(rows)
    items = row_labels(rows) + [CLEAR_LABEL]
    preselected = {i for i, r in enumerate(rows) if r["tag"] in current_order}
    if existing and not current_order:
        preselected = {i for i, r in enumerate(rows) if r["tag"] in (existing.get("only") or [])}

    title = f"Select providers for {model_id} (SPACE toggles, ENTER confirms)"
    try:
        print()
        print(f"  {rows_footer(rows)}")
        chosen = curses_checklist(
            title, items, preselected, status_fn=lambda chosen: _summary(rows, chosen))
    except Exception as exc:  # curses unavailable mid-flow: never block the model switch
        logger.debug("provider checklist unavailable: %s", exc)
        return None
    print()

    if clear_idx in chosen:
        return {}
    tags = [rows[i]["tag"] for i in sorted(i for i in chosen if 0 <= i < len(rows))]
    if not tags:
        return None

    sort_values = [value for value, _label in SORT_ROWS]
    current_sort = str(existing.get("sort") or "").strip().lower()
    default_sort = sort_values.index(current_sort) if current_sort in sort_values else 0
    try:
        sort_idx = curses_radiolist(
            "Sort eligible providers by (only matters while OpenRouter may fall back)",
            [label for _value, label in SORT_ROWS], selected=default_sort,
            cancel_returns=default_sort)
    except Exception:
        sort_idx = default_sort
    print()

    routing = dict(existing)
    routing["order"] = tags
    sort_value = sort_values[sort_idx] if isinstance(sort_idx, int) and 0 <= sort_idx < len(sort_values) else ""
    if sort_value:
        routing["sort"] = sort_value
    else:
        routing.pop("sort", None)
    return routing


def configure_after_selection(
    model_id: str, *, provider: str = "openrouter", base_url: str = "", force_refresh: bool = False,
    announce: bool = True,
) -> Optional[dict]:
    """Run the screen for a freshly picked model and persist the result.

    Returns the stored routing, ``{}`` when cleared, or ``None`` when skipped/not applicable.
    """
    routing = prompt_provider_routing(
        model_id, provider=provider, base_url=base_url, force_refresh=force_refresh)
    if routing is None:
        return None
    stored = save_model_routing(model_id, routing)
    if announce:
        print(f"  Provider routing saved for {model_id}: {describe_routing(stored)}")
    return stored
