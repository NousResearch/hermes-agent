"""AI Usage Meter backend, mounted at /api/plugins/ai-usage-meter/.

Uses Hermes' normalized provider-account usage adapters and read-only analytics
layer. It never reads prompt content, writes state.db, or exposes credentials.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, UTC
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Query

router = APIRouter()

MODEL_PERIOD_DAYS = 7
MAX_PROVIDERS = 10
MAX_MODELS_PER_PROVIDER = 3
BACKEND_VERSION = "0.2.0"

KNOWN_PROVIDER_ALIASES = {
    "anthropic": ("anthropic",),
    "openai-codex": ("openai-codex", "codex"),
    "openrouter": ("openrouter",),
    "gemini": ("gemini", "google", "google-gemini"),
    "ollama-cloud": ("ollama-cloud",),
}


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


def _clamp_percent(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return round(max(0.0, min(100.0, float(value))), 2)


def _serialize_snapshot(provider: dict[str, Any], snapshot: Any) -> dict[str, Any]:
    if snapshot is None:
        model_count = len(provider.get("models") or ())
        if provider.get("is_ollama"):
            reason = "Local Ollama is reachable. Local providers do not expose an account quota."
            details = [f"{model_count} local model{'s' if model_count != 1 else ''} available."]
        else:
            reason = "Provider is available in Hermes, but it did not expose account quota data."
            details = [f"{model_count} model{'s' if model_count != 1 else ''} available."]
        return {
            "id": provider["id"],
            "name": provider["name"],
            "available": bool(provider.get("available", True)),
            "source": provider.get("source", "HERMES"),
            "provider_source": provider.get("provider_source", "hermes_provider_inventory"),
            "plan": "Local" if provider.get("is_ollama") else None,
            "fetched_at": None,
            "windows": [],
            "details": details,
            "unavailable_reason": reason,
        }

    windows = []
    for window in getattr(snapshot, "windows", ()) or ():
        used = _clamp_percent(getattr(window, "used_percent", None))
        windows.append(
            {
                "label": str(getattr(window, "label", "Usage window")),
                "used_percent": used,
                "remaining_percent": round(100.0 - used, 2) if used is not None else None,
                "reset_at": _iso(getattr(window, "reset_at", None)),
                "detail": getattr(window, "detail", None),
                "scope": getattr(window, "scope", "account") or "account",
                "unit": "percent",
            }
        )

    available = bool(getattr(snapshot, "available", False))
    return {
        "id": provider["id"],
        "name": provider["name"],
        "available": bool(provider.get("available", True)),
        "source": "LIVE" if available else provider.get("source", "HERMES"),
        "provider_source": getattr(snapshot, "source", None),
        "plan": getattr(snapshot, "plan", None),
        "fetched_at": _iso(getattr(snapshot, "fetched_at", None)),
        "windows": windows,
        "details": list(getattr(snapshot, "details", ()) or ()),
        "unavailable_reason": getattr(snapshot, "unavailable_reason", None),
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        number = float(value or 0.0)
        return number if number == number and number not in {float("inf"), float("-inf")} else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _empty_model(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "aliases": [name],
        "matched_models": [name],
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "actual_cost": 0.0,
        "estimated_cost": 0.0,
        "display_cost": 0.0,
        "cost_source": "UNAVAILABLE",
        "api_calls": 0,
        "sessions": 0,
        "source": "HERMES",
        "api_metadata": None,
    }


def _aggregate_models(provider: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    provider_aliases = {
        str(alias).lower()
        for alias in provider.get("aliases", (provider["id"],))
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    names: dict[str, str] = {}
    for row in rows:
        if str(row.get("provider") or "").lower() not in provider_aliases:
            continue
        name = str(row.get("model") or "").strip()
        if not name:
            continue
        key = name.lower()
        names.setdefault(key, name)
        grouped.setdefault(key, []).append(row)

    results = []
    for key, matched in grouped.items():
        item = _empty_model(names[key])
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "reasoning_tokens",
            "api_calls",
            "sessions",
        ):
            item[key] = sum(_safe_int(row.get(key)) for row in matched)
        item["actual_cost"] = round(sum(_safe_float(row.get("actual_cost")) for row in matched), 8)
        item["estimated_cost"] = round(sum(_safe_float(row.get("estimated_cost")) for row in matched), 8)
        item["total_tokens"] = item["input_tokens"] + item["output_tokens"]
        if item["actual_cost"] > 0:
            item["display_cost"] = item["actual_cost"]
            item["cost_source"] = "LIVE"
        elif item["estimated_cost"] > 0:
            item["display_cost"] = item["estimated_cost"]
            item["cost_source"] = "ESTIMATED"
        if item["total_tokens"] > 0:
            results.append(item)
    results.sort(key=lambda item: (-item["total_tokens"], item["name"].lower()))
    return results[:MAX_MODELS_PER_PROVIDER]


def _default_analytics_loader(days: int) -> dict[str, Any]:
    """Read canonical per-route accounting from SessionDB in read-only mode."""
    from hermes_state import SessionDB

    cutoff = time.time() - (days * 86400)
    db = SessionDB(read_only=True)
    try:
        try:
            cursor = db._conn.execute(
                """
                SELECT u.model,
                       u.billing_provider AS provider,
                       SUM(u.input_tokens) AS input_tokens,
                       SUM(u.output_tokens) AS output_tokens,
                       SUM(u.cache_read_tokens) AS cache_read_tokens,
                       SUM(u.reasoning_tokens) AS reasoning_tokens,
                       COALESCE(SUM(u.estimated_cost_usd), 0) AS estimated_cost,
                       COALESCE(SUM(u.actual_cost_usd), 0) AS actual_cost,
                       COUNT(DISTINCT u.session_id) AS sessions,
                       SUM(COALESCE(u.api_call_count, 0)) AS api_calls
                FROM session_model_usage u
                JOIN sessions s ON s.id = u.session_id
                WHERE COALESCE(u.last_seen, s.last_activity_at, s.started_at) > ?
                  AND u.model IS NOT NULL AND u.model != ''
                GROUP BY u.model, u.billing_provider
                """,
                (cutoff,),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            cursor = db._conn.execute(
                """
                SELECT model,
                       billing_provider AS provider,
                       SUM(input_tokens) AS input_tokens,
                       SUM(output_tokens) AS output_tokens,
                       SUM(cache_read_tokens) AS cache_read_tokens,
                       SUM(reasoning_tokens) AS reasoning_tokens,
                       COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost,
                       COALESCE(SUM(actual_cost_usd), 0) AS actual_cost,
                       COUNT(*) AS sessions,
                       SUM(COALESCE(api_call_count, 0)) AS api_calls
                FROM sessions
                WHERE COALESCE(last_activity_at, started_at) > ?
                  AND model IS NOT NULL AND model != ''
                GROUP BY model, billing_provider
                """,
                (cutoff,),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        return {"models": rows, "period_days": days}
    finally:
        db.close()


def _default_ollama_probe(base_url: str) -> Optional[list[str]]:
    """Use Hermes' native Ollama probe; None means the endpoint did not verify as Ollama."""
    from hermes_cli.models_local import fetch_ollama_local_models

    return fetch_ollama_local_models(base_url, timeout=2.0)


def _is_ollama_candidate(row: dict[str, Any]) -> bool:
    slug = str(row.get("slug") or row.get("id") or "").lower()
    name = str(row.get("name") or "").lower()
    base_url = str(row.get("api_url") or row.get("base_url") or "").strip()
    if "ollama" in slug or "ollama" in name:
        return True
    try:
        return urlparse(base_url).port == 11434
    except ValueError:
        return False


def _normalize_discovered_providers(
    rows: list[dict[str, Any]],
    *,
    ollama_probe: Callable[[str], Optional[list[str]]] = _default_ollama_probe,
) -> list[dict[str, Any]]:
    """Normalize Hermes' provider inventory without exposing credentials or raw provider payloads."""
    providers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        provider_id = str(row.get("slug") or row.get("id") or "").strip().lower()
        if not provider_id or provider_id == "moa" or provider_id in seen:
            continue
        seen.add(provider_id)
        name = str(row.get("name") or provider_id).strip() or provider_id
        base_url = str(row.get("api_url") or row.get("base_url") or "").strip()
        configured_models = sorted({str(model).strip() for model in row.get("models", ()) if str(model).strip()})
        is_ollama = False
        source = "HERMES"
        provider_source = "hermes_provider_inventory"
        models = configured_models

        if base_url and _is_ollama_candidate(row):
            try:
                live_models = ollama_probe(base_url)
            except Exception:
                live_models = None
            if live_models is not None:
                is_ollama = True
                source = "LIVE"
                provider_source = "ollama_api_tags"
                models = sorted({str(model).strip() for model in live_models if str(model).strip()})
                if "ollama" not in name.lower():
                    name = "Ollama Local"

        aliases = {
            provider_id,
            str(row.get("provider_id") or "").strip().lower(),
            *KNOWN_PROVIDER_ALIASES.get(provider_id, ()),
        }
        aliases.discard("")
        providers.append(
            {
                "id": provider_id,
                "name": name,
                "models": models,
                "aliases": sorted(aliases),
                "available": True,
                "source": source,
                "provider_source": provider_source,
                "is_ollama": is_ollama,
                "is_current": bool(row.get("is_current")),
            }
        )
    return providers


def _default_provider_discovery(refresh: bool) -> list[dict[str, Any]]:
    """Ask Hermes for providers that are authenticated/configured and probe custom endpoints on refresh."""
    from hermes_cli.inventory import build_model_options_payload, load_picker_context

    inventory = build_model_options_payload(
        load_picker_context(),
        explicit_only=False,
        include_unconfigured=False,
        refresh=bool(refresh),
    )
    return _normalize_discovered_providers(list(inventory.get("providers") or ()))


def _default_usage_fetcher(provider: str):
    from agent.account_usage import fetch_account_usage

    return fetch_account_usage(provider, read_only=True)


def build_usage_payload(
    *,
    days: int = 30,
    refresh: bool = False,
    usage_fetcher: Callable[[str], Any] = _default_usage_fetcher,
    analytics_loader: Callable[[int], dict[str, Any]] = _default_analytics_loader,
    provider_discovery: Callable[[bool], list[dict[str, Any]]] = _default_provider_discovery,
) -> dict[str, Any]:
    """Build a fail-soft, credential-free payload for desktop rendering."""
    requested_days = max(1, min(365, int(days)))
    errors: list[str] = []

    try:
        discovered = list(provider_discovery(bool(refresh)) or ())
    except Exception as exc:
        discovered = []
        errors.append(f"Provider scan unavailable ({type(exc).__name__}).")

    provider_count = len(discovered)
    providers = discovered[:MAX_PROVIDERS]
    omitted = max(0, provider_count - len(providers))
    if omitted:
        errors.append(f"{omitted} additional provider{'s' if omitted != 1 else ''} omitted; the dashboard supports 10 cards.")

    try:
        analytics = analytics_loader(MODEL_PERIOD_DAYS) or {}
        rows = list(analytics.get("models") or [])
        model_period_days = int(analytics.get("period_days") or MODEL_PERIOD_DAYS)
    except Exception as exc:
        rows = []
        model_period_days = MODEL_PERIOD_DAYS
        errors.append(f"Historical usage unavailable ({type(exc).__name__}).")

    snapshots: dict[str, Any] = {}
    if providers:
        with ThreadPoolExecutor(max_workers=min(8, len(providers))) as executor:
            futures = {
                provider["id"]: executor.submit(usage_fetcher, provider["id"])
                for provider in providers
            }
            for provider_id, future in futures.items():
                try:
                    snapshots[provider_id] = future.result()
                except Exception as exc:
                    snapshots[provider_id] = None
                    errors.append(f"{provider_id} usage unavailable ({type(exc).__name__}).")

    provider_payloads = []
    for provider in providers:
        payload = _serialize_snapshot(provider, snapshots.get(provider["id"]))
        models = _aggregate_models(provider, rows)
        payload["models"] = models
        provider_payloads.append(payload)

    return {
        "backend_version": BACKEND_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "period_days": requested_days,
        "model_period_days": model_period_days,
        "providers_discovered": provider_count,
        "providers_omitted": omitted,
        "providers": provider_payloads,
        "errors": errors,
    }


@router.get("/usage")
async def usage(
    days: int = Query(30, ge=1, le=365),
    refresh: bool = Query(False),
):
    return await asyncio.to_thread(build_usage_payload, days=days, refresh=refresh)
