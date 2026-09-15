"""Where a model list came from — the line the pickers used to omit (#110055).

The picker merges live ``/models`` discovery, the hosted ``model-catalog.json`` snapshot, the
in-repo bundled list and the user's own ``models:`` list into one undifferentiated list, so a
missing model gets re-debugged by hand every time. The resolution layer calls :func:`record` at
the point it decides; the pickers read :func:`provider_provenance` (built-in rows) or
:func:`endpoint_provenance` (user-defined endpoints) and render ``label`` next to the provider.

Sources:

``discovered``  live ``/models`` against the provider's own endpoint (or its disk cache).
``live``        provider API / OAuth account catalog, or the models.dev registry.
``catalog``     hosted ``model-catalog.json`` manifest — ``at`` is the snapshot mtime.
``bundled``     the in-repo static list shipped with this build; ``reason`` says why the live
                path didn't produce the list (token missing, endpoint unreachable, ...).
``configured``  a ``models:`` allowlist the user pinned, or the subset shown when discovery
                came back empty.

Display only: nothing here fetches, caches, or changes which models a provider resolves to.
A provider with no entry shows nothing — "unknown" is not worth a row.
"""

from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Optional

DISCOVERED = "discovered"
LIVE = "live"
CATALOG = "catalog"
BUNDLED = "bundled"
CONFIGURED = "configured"

# One entry per provider slug. Resolution runs bottom-up through the fallback chain, so the
# outermost branch that actually produced the list records last and wins.
# entry: {"source", "reason", "detail", "count", "at", "stale"}
_entries: dict[str, dict[str, Any]] = {}


def normalize(provider: Any) -> str:
    """Journal key for a provider slug (case/whitespace-insensitive)."""
    return str(provider or "").strip().lower()


def record(
    provider: Any, source: str, *, reason: str = "", detail: str = "",
    count: Optional[int] = None, at: Optional[float] = None, stale: bool = False,
) -> dict:
    """Journal what just produced ``provider``'s list. Last writer wins."""
    entry = {
        "source": str(source), "reason": str(reason or ""), "detail": str(detail or ""),
        "count": count, "at": time.time() if at is None else float(at), "stale": bool(stale)}
    _entries[normalize(provider)] = entry
    return entry


def recorded(provider: Any) -> Optional[dict]:
    """Raw journal entry for ``provider``, or None when nothing resolved it yet."""
    return _entries.get(normalize(provider))


def reset() -> None:
    """Drop every entry (tests, and provider switches that must not show a stale reason)."""
    _entries.clear()


def age_label(seconds: float) -> str:
    """``2m ago`` / ``3h ago`` / ``2d ago`` — coarse on purpose, this is a hint line."""
    seconds = int(max(0, seconds))
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _snapshot_date(at: float) -> str:
    return _dt.datetime.fromtimestamp(at).strftime("%b %d")


def _label(prov: dict[str, Any]) -> str:
    """One human line, e.g. ``live · 12 models · 2m ago`` (see the module docstring)."""
    source = prov.get("source") or ""
    count = prov.get("count")
    models = f"{count} model{'s' if count != 1 else ''}" if isinstance(count, int) else ""
    if source == CONFIGURED:
        return (f"configured models: list — {prov['reason']}" if prov.get("reason")
                else "configured models: list (discover_models: false)")
    if source == CATALOG:
        label = f"hosted catalog · snapshot {_snapshot_date(prov.get('at') or time.time())}"
        return label + " — stale, run /model --refresh" if prov.get("stale") else label
    if source == BUNDLED:
        return (f"bundled fallback — {prov['reason']}" if prov.get("reason")
                else "bundled in-repo list")
    parts = [prov.get("detail") or ("discovered" if source == DISCOVERED else "live")]
    if models:
        parts.append(models)
    if prov.get("at"):
        parts.append(age_label(prov.get("age_seconds") or 0))
    return " · ".join(parts)


def _finalize(prov: dict[str, Any], *, now: Optional[float] = None) -> dict:
    """Add ``age_seconds`` / ``degraded`` / ``label`` to a provenance dict."""
    out = dict(prov)
    at = out.get("at")
    if at is not None and out.get("age_seconds") is None:
        out["age_seconds"] = max(0, int((now or time.time()) - float(at)))
    out["degraded"] = out.get("source") == BUNDLED and bool(out.get("reason"))
    out["label"] = _label(out)
    return out


def describe(
    source: str, *, reason: str = "", detail: str = "", count: Optional[int] = None,
    at: Optional[float] = None, stale: bool = False, now: Optional[float] = None,
) -> dict:
    """Formatted provenance for a case the caller already knows statically (no journal write)."""
    return _finalize(
        {"source": source, "reason": reason, "detail": detail, "count": count,
         "at": at, "stale": stale, "age_seconds": None},
        now=now)


def provider_provenance(
    provider: Any, *, count: Optional[int] = None, now: Optional[float] = None,
) -> dict:
    """Formatted provenance for ``provider``'s list, ``{}`` when nothing resolved it yet.

    ``count`` overrides the journaled model count when the caller shows a capped list.
    """
    entry = recorded(provider)
    if entry is None:
        return {}
    prov = dict(entry)
    if prov.get("count") is None and count is not None:
        prov["count"] = count
    return _finalize(prov, now=now)


def endpoint_provenance(
    api_url: str, *, discovered: bool, discovery_allowed: bool, count: Optional[int] = None,
    native_empty: bool = False, now: Optional[float] = None,
) -> dict:
    """Provenance for a user-defined / custom endpoint row.

    ``discovered`` — the live probe returned a catalog; ``discovery_allowed`` is False when the
    entry pins its list (``discover_models: false``); neither means the endpoint showed the
    configured ``models:`` subset instead.
    """
    url = str(api_url or "").strip()
    if discovered:
        return describe(
            DISCOVERED, count=count, detail=f"/models on {url}" if url else "/models",
            now=now)
    if not discovery_allowed:
        return describe(CONFIGURED, now=now)
    if native_empty:
        return describe(CONFIGURED, reason="endpoint lists no models", now=now)
    return describe(CONFIGURED, reason="live /models unavailable", now=now)


def catalog_provenance(now: Optional[float] = None) -> dict:
    """The hosted manifest's own status (never fetches — see ``model_catalog.catalog_status``)."""
    from hermes_cli.model_catalog import catalog_status

    status = catalog_status()
    if status.get("origin") == "hosted":
        return describe(
            CATALOG, at=status.get("as_of"), stale=not status.get("fresh", True), now=now)
    return describe(BUNDLED, reason=status.get("reason") or "hosted catalog unavailable", now=now)
