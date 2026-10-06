"""Operator-facing notice when a provider route walls (quota, balance, auth, hard rate limit).

A walled provider is classified, retried, benched and failed-over deep inside the agent, but the
only operator-facing artifact used to be a ``logger.warning`` plus the raw provider error body in
whichever single chat happened to send the turn.  On an unattended fleet — cron batches, kanban
workers, a chat nobody is watching — that is how a quota wall turns into a silently missing report.

This module owns the incident text: one line per provider:model pair that has a problem, so the
operator can see at a glance *which* route is down, *why*, and *until when*.  The agent records the
incident when it leaves a walled primary (``agent.chat_completion_helpers.try_activate_fallback``);
the gateway fans the text out to every served profile's home channels
(``gateway.run_notifications._replay_pending_provider_wall_notice``).

Two invariants, both load-bearing:

* **Never raise.**  This runs on the retry path of a failing turn; a notice bug must never mask the
  provider error the user actually needs to see.
* **One message per incident, not per turn.**  The primary is restored each turn, so a wall that
  persists for hours would otherwise re-notify every turn.  Incidents are keyed by the set of
  unusable routes plus the declared reset instant; a *changed* set (a second route walls) is a new
  incident, a repeated one is not.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

#: Marker the gateway drains; lives in HERMES_HOME beside the restart/update markers.
WALL_MARKER_NAME = ".provider_wall_notice.json"

#: A second, materially different wall inside this window merges into the incident already sent
#: instead of paging the operator twice for one outage.
QUIET_WINDOW_S = 90.0

#: A marker older than this is stale (the wall it describes is long gone); the gateway drops it.
MARKER_TTL_S = 24 * 3600.0

#: Rate-limit windows shorter than this read as a transient throttle (cooling), not a wall.
WALL_MIN_RESET_S = 15 * 60.0

#: How often to re-notify the operator while the wall persists. 0 = disabled.
WALL_RENOTIFY_INTERVAL_S = 3600  # 1 hour (default, overridable via diagnostics.provider_wall_renotify_seconds)

WALLED = "walled"
COOLING = "cooling"
AVAILABLE = "available"
UNKNOWN = "unknown"

_EMOJI = {WALLED: "🔴", COOLING: "🟠", AVAILABLE: "🟢", UNKNOWN: "⚪"}

_BILLING_REASONS = {"billing", "quota", "credits", "insufficient_balance"}
_BILLING_CODES = {402, 403}


@dataclass(frozen=True)
class RouteRow:
    """One provider:model pair and what is wrong (or right) with it."""

    provider: str
    model: str
    status: str = UNKNOWN
    detail: str = ""
    reset_at: Optional[float] = None

    @property
    def emoji(self) -> str:
        return _EMOJI.get(self.status, _EMOJI[UNKNOWN])

    def identity(self) -> tuple[str, str, str]:
        return (self.provider, self.model, self.status)


def _t(key: str, **kwargs: Any) -> str:
    """Localized string; the key itself when i18n is unavailable (never raise from a notice)."""
    try:
        from agent.i18n import t

        return t(key, **kwargs)
    except Exception:  # pragma: no cover - a locale failure must not lose the notice
        logger.debug("provider wall notice: i18n lookup failed for %s", key, exc_info=True)
        return key


def _pool_key(provider: str, base_url: Optional[str]) -> str:
    from agent.credential_pool import resolve_runtime_pool_key

    return resolve_runtime_pool_key(provider, base_url) or provider


def _entry_status(entry: Any, now: float) -> tuple[str, str, Optional[float]]:
    """``(status, detail, reset_at)`` for one credential pool entry."""
    failure = str(getattr(entry, "failure_reason", "") or "").lower()
    code = getattr(entry, "last_error_code", None)
    message = str(getattr(entry, "last_error_message", "") or "").strip()
    reset_at = getattr(entry, "last_error_reset_at", None)
    try:
        reset_at = float(reset_at) if reset_at else None
    except (TypeError, ValueError):
        reset_at = None

    if failure in _BILLING_REASONS or (code in _BILLING_CODES and failure != "rate_limit"):
        return WALLED, _short(message) or "billing/quota refusal", reset_at
    if code == 429:
        if reset_at and reset_at - now > WALL_MIN_RESET_S:
            return WALLED, _short(message) or "rate/quota window exhausted", reset_at
        return COOLING, _short(message) or "rate limited", reset_at
    if failure in {"auth", "auth_error"}:
        return WALLED, _short(message) or "authentication refused", reset_at
    if getattr(entry, "last_status", None) in {"exhausted", "rate_limited"}:
        return COOLING, _short(message) or "cooling down", reset_at
    return AVAILABLE, "", None


def _short(message: str, limit: int = 120) -> str:
    message = " ".join(message.split())
    return message if len(message) <= limit else message[: limit - 1] + "…"


def row_for_route(provider: str, model: str, base_url: Optional[str] = None) -> RouteRow:
    """Pool-derived row for a provider:model pair (never raises; UNKNOWN when unreadable)."""
    try:
        from agent.credential_pool import load_pool

        pool = load_pool(_pool_key(provider, base_url))
        entries = list(pool.entries())
    except Exception:
        logger.debug("provider wall notice: pool read failed for %s", provider, exc_info=True)
        return RouteRow(provider=provider, model=model, status=UNKNOWN, detail="credential state unreadable")

    if not entries:
        return RouteRow(provider=provider, model=model, status=UNKNOWN, detail="no credentials configured")

    now = time.time()
    worst: Optional[tuple[str, str, Optional[float]]] = None
    for entry in entries:
        status, detail, reset_at = _entry_status(entry, now)
        if status == AVAILABLE:
            continue
        # WALLED outranks COOLING: the most actionable line is the one that will not clear itself.
        if worst is None or (status == WALLED and worst[0] != WALLED):
            worst = (status, detail, reset_at)
    if worst is None:
        return RouteRow(provider=provider, model=model, status=AVAILABLE)
    return RouteRow(provider=provider, model=model, status=worst[0], detail=worst[1], reset_at=worst[2])


def incident_rows(
    agent: Any,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    reason: Any = None,
    reset_at: Optional[float] = None,
    detail: Optional[str] = None,
) -> list[RouteRow]:
    """Rows for the incident that just fired: the walled primary plus every unusable chain entry.

    Rows that are healthy are dropped — the notice lists problems, not inventory.  The primary is
    always present so a wall with an empty fallback chain still says something useful.
    """
    primary_provider = str(provider or getattr(agent, "provider", "") or "")
    primary_model = str(model or getattr(agent, "model", "") or "")
    rows: list[RouteRow] = [
        RouteRow(
            provider=primary_provider or "unknown",
            model=primary_model or "unknown",
            status=WALLED,
            detail=_short(str(detail or "")) or _reason_detail(reason),
            reset_at=_as_float(reset_at) or _primary_reset(agent),
        )
    ]
    seen = {(rows[0].provider.lower(), rows[0].model)}
    for entry in _chain_entries(agent):
        fb_provider = str(entry.get("provider") or "").strip()
        fb_model = str(entry.get("model") or "").strip()
        if not fb_provider or not fb_model or (fb_provider.lower(), fb_model) in seen:
            continue
        seen.add((fb_provider.lower(), fb_model))
        row = row_for_route(fb_provider, fb_model, entry.get("base_url"))
        if row.status in {WALLED, COOLING, UNKNOWN}:
            rows.append(row)
    return rows


def _chain_entries(agent: Any) -> Iterable[dict]:
    chain = getattr(agent, "_fallback_chain", None) or []
    return [entry for entry in chain if isinstance(entry, dict)]


def route_pairs(config: Optional[dict]) -> set[tuple[str, str]]:
    """``(provider, model)`` pairs a profile's config routes through: its primary + fallback chain.

    Empty provider means "inherits the install default", which is how an unset
    ``model.provider`` behaves at runtime.
    """
    pairs: set[tuple[str, str]] = set()
    model_cfg = (config or {}).get("model")
    if isinstance(model_cfg, dict):
        model = str(model_cfg.get("default") or "").strip()
        if model:
            pairs.add((str(model_cfg.get("provider") or "").strip().lower(), model))
    for entry in (config or {}).get("fallback_providers") or []:
        if not isinstance(entry, dict):
            continue
        model = str(entry.get("model") or "").strip()
        if model:
            pairs.add((str(entry.get("provider") or "").strip().lower(), model))
    return pairs


def routes_affected(routes: set[tuple[str, str]], incident_routes: Optional[Iterable[dict]]) -> bool:
    """True when any of *routes* is one of the incident's affected ``(provider, model)`` pairs.

    Only the profiles that actually run an affected route are paged: a profile on an unrelated
    model/provider must not be woken for someone else's wall.  Anything unknown fails OPEN —
    an absent route list (older marker) or an unreadable config returns True, because
    over-notifying costs one message while dropping a wall notice costs the incident.
    """
    pairs = {
        (str(route.get("provider") or "").strip().lower(), str(route.get("model") or "").strip())
        for route in (incident_routes or [])
        if isinstance(route, dict)
    }
    pairs.discard(("", ""))
    if not pairs or not routes:
        return True
    incident_models = {model for _provider, model in pairs if model}
    incident_providers = {provider for provider, _model in pairs if provider}
    for provider, model in routes:
        if not model:
            continue
        if (provider, model) in pairs:
            return True
        # An unset provider inherits the install default, so the model alone identifies the route.
        if not provider and model in incident_models:
            return True
        if provider and model in incident_models and provider in incident_providers:
            return True
    return False


def _reason_detail(reason: Any) -> str:
    value = getattr(reason, "value", reason)
    return {
        "billing": "billing/quota refusal",
        "rate_limit": "rate/quota window exhausted",
        "auth": "authentication refused",
    }.get(str(value or "").lower(), str(value or "provider refused the request"))


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value else None
    except (TypeError, ValueError):
        return None


def _primary_reset(agent: Any) -> Optional[float]:
    context = getattr(agent, "_last_api_error_context", None) or {}
    if isinstance(context, dict):
        return _as_float(context.get("reset_at"))
    return None


def _fmt_reset(reset_at: Optional[float]) -> str:
    if not reset_at:
        return ""
    try:
        return time.strftime("%d %b %H:%M %z", time.localtime(float(reset_at)))
    except (OverflowError, OSError, ValueError):  # pragma: no cover - absurd epoch
        return ""


def _fmt_wait(reset_at: Optional[float], now: float) -> str:
    if not reset_at:
        return ""
    minutes = max(1, int((float(reset_at) - now) / 60))
    if minutes >= 90:
        hours = minutes // 60
        return f"{hours}h"
    return f"{minutes}m"


def build_message(rows: list[RouteRow], *, profile: str = "default", now: Optional[float] = None,
                  created_at: Optional[float] = None, expires_at: Optional[float] = None) -> str:
    """The operator notice: a header, one line per problem route, scope and remedy."""
    now = time.time() if now is None else now
    down = [row for row in rows if row.status in {WALLED, COOLING}]
    usable = [row for row in rows if row.status == AVAILABLE]
    header = (
        _t("provider_wall.title_blocked", down=len(down))
        if not usable
        else _t("provider_wall.title_partial", down=len(down), carrying=f"{usable[0].provider} · {usable[0].model}")
    )
    lines = [header, ""]
    # Timing line: when the wall started and when it should clear
    timing_parts = []
    if created_at:
        timing_parts.append(f"Started: {_fmt_reset(created_at)}")
    if expires_at and expires_at > now:
        timing_parts.append(f"Expected reset: {_fmt_reset(expires_at)}")
    elif expires_at:
        timing_parts.append("Expected reset: past due (provider did not specify)")
    if timing_parts:
        lines.append(" · ".join(timing_parts))
        lines.append("")
    for row in rows:
        lines.append(_row_line(row, now))
    cmd = "hermes model" if profile in {"", "default"} else f"hermes -p {profile} model"
    lines.append(_t("provider_wall.scope", profile=profile or "default"))
    lines.append(_t("provider_wall.remedy", cmd=cmd))
    return "\n".join(lines)


def _row_line(row: RouteRow, now: float) -> str:
    until = ""
    if row.status == WALLED and row.reset_at:
        until = _t("provider_wall.until_reset", when=_fmt_reset(row.reset_at))
    elif row.status == COOLING and row.reset_at:
        until = _t("provider_wall.until_cooling", wait=_fmt_wait(row.reset_at, now))
    return _t(
        f"provider_wall.row_{row.status}",
        emoji=row.emoji, provider=row.provider, model=row.model, detail=row.detail, until=until,
    )


def signature(rows: list[RouteRow]) -> str:
    """Stable identity of an incident: which routes are down and when they reopen."""
    payload = json.dumps(
        sorted(
            [[row.provider, row.model, row.status, int(row.reset_at or 0)] for row in rows],
            key=lambda item: (item[0], item[1], item[2], item[3]),
        ),
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def marker_path(home: Optional[Path] = None) -> Path:
    if home is None:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
    return Path(home) / WALL_MARKER_NAME


def read_pending(home: Optional[Path] = None) -> Optional[dict]:
    """Pending notice payload, or None when absent/unreadable/expired. Never raises."""
    path = marker_path(home)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except Exception:
        logger.debug("provider wall notice: unreadable marker %s", path, exc_info=True)
        return None
    if not isinstance(payload, dict) or not payload.get("text"):
        return None
    now = time.time()
    expires_at = _as_float(payload.get("expires_at"))
    created_at = _as_float(payload.get("created_at")) or 0.0
    if (expires_at and now > expires_at) or (created_at and now - created_at > MARKER_TTL_S):
        logger.info("Provider wall notice marker expired (%s) — dropping", path)
        clear_pending(home)
        return None
    return payload


def clear_pending(home: Optional[Path] = None) -> None:
    try:
        marker_path(home).unlink(missing_ok=True)
    except Exception:  # pragma: no cover - best effort
        logger.debug("provider wall notice: could not clear marker", exc_info=True)


def clear_route(provider: str, model: str, home: Optional[Path] = None) -> None:
    """Mark one provider:model pair as recovered in a pending wall notice.

    When the marker exists and lists *provider*:*model* as walled or cooling,
    that route is set to AVAILABLE.  If *every* route is now available the
    marker is removed entirely; otherwise the marker is re-written with the
    updated route list and a fresh signature, so the gateway stops paging
    the operator about a route that recovered.

    A successful call on a route that was never walled leaves the marker
    untouched — a working model must not silence the alarm for a different
    model that is still down.
    """
    try:
        payload = read_pending(home)
        if not payload:
            return
        routes: list[dict] = payload.get("routes") or []
        changed = False
        remaining_walled = False
        for route in routes:
            if route.get("provider") == provider and route.get("model") == model:
                if route.get("status") in {WALLED, COOLING, UNKNOWN}:
                    route["status"] = AVAILABLE
                    route["detail"] = ""
                    changed = True
            if route.get("status") in {WALLED, COOLING, UNKNOWN}:
                remaining_walled = True
        if not changed:
            return
        if not remaining_walled:
            clear_pending(home)
            return
        # Rebuild signature from remaining walled routes
        walled_rows = [
            RouteRow(
                provider=r["provider"],
                model=r["model"],
                status=r.get("status", AVAILABLE),
                detail=r.get("detail", ""),
                reset_at=r.get("reset_at"),
            )
            for r in routes
        ]
        payload["signature"] = signature(walled_rows)
        payload["text"] = build_message(
            walled_rows,
            profile=payload.get("profile", "default"),
            created_at=payload.get("created_at"),
            expires_at=payload.get("expires_at"),
        )
        payload["updated_at"] = time.time()
        _write(payload, home)
        logger.info(
            "Provider wall notice: route %s/%s recovered — %d route(s) still walled",
            provider, model, sum(
                1 for r in routes if r.get("status") in {WALLED, COOLING, UNKNOWN}
            ),
        )
    except Exception:
        logger.debug("provider wall notice: clear_route failed", exc_info=True)


def mark_delivered(targets: Iterable[Iterable[str]], home: Optional[Path] = None) -> None:
    """Record which home channels already received the notice (the gateway's owed-set ledger)."""
    payload = read_pending(home)
    if not payload:
        return
    delivered = {tuple(str(part) if part is not None else "" for part in target)
                 for target in (payload.get("delivered_targets") or [])}
    delivered |= {tuple(str(part) if part is not None else "" for part in target) for target in targets}
    payload["delivered_targets"] = [list(target) for target in sorted(delivered)]
    payload["delivered_at"] = time.time()
    _write(payload, home)


def _write(payload: dict, home: Optional[Path] = None) -> None:
    from utils import atomic_json_write

    atomic_json_write(marker_path(home), payload)


def record_provider_wall(
    agent: Any,
    *,
    reason: Any = None,
    reset_at: Optional[float] = None,
    detail: Optional[str] = None,
    home: Optional[Path] = None,
) -> Optional[str]:
    """Record the incident; return its signature when the notice changed, else None.

    A repeat of the same incident leaves the marker (and its delivered ledger) alone — the gateway
    still drains whatever is pending, but the chat that hit the wall is not told twice.  A
    materially different incident inside ``QUIET_WINDOW_S`` of the last delivery merges into it;
    later than that it starts a fresh ledger and is delivered again.
    """
    try:
        rows = incident_rows(agent, reason=reason, reset_at=reset_at, detail=detail)
        if not rows:
            return None
        sig = signature(rows)
        profile = _profile_name()
        now = time.time()
        previous = read_pending(home)
        resets = [row.reset_at for row in rows if row.reset_at]
        created_ts = _as_float((previous or {}).get("created_at")) or now
        expires_ts = max(resets) if resets else now + MARKER_TTL_S
        text = build_message(rows, profile=profile, created_at=created_ts, expires_at=expires_ts)
        delivered: list = []
        delivered_at: Optional[float] = None
        merge = False
        if previous:
            if previous.get("signature") == sig and previous.get("text") == text:
                return None  # the same wall, already recorded — nothing new to say
            if (_as_float(previous.get("delivered_at")) or 0.0) > now - QUIET_WINDOW_S:
                # Merged incident: keep the delivered ledger (one outage pages once) but refresh the
                # text, so a channel that was unreachable at delivery time still gets the full list.
                merge = True
                delivered = list(previous.get("delivered_targets") or [])
                delivered_at = _as_float(previous.get("delivered_at"))
        payload = {
            "v": 1,
            "signature": sig,
            "profile": profile,
            "text": text,
            "routes": [{"provider": row.provider, "model": row.model, "status": row.status}
                       for row in rows],
            "created_at": _as_float((previous or {}).get("created_at")) or now,
            "updated_at": now,
            "delivered_targets": delivered,
            "delivered_at": delivered_at,
            "expires_at": (max(resets) if resets else now + MARKER_TTL_S),
        }
        _write(payload, home)
        return None if merge else sig
    except Exception:
        logger.debug("provider wall notice: record failed", exc_info=True)
        return None


def _profile_name() -> str:
    try:
        from hermes_constants import get_hermes_home, profile_name_for_home

        return profile_name_for_home(get_hermes_home()) or "default"
    except Exception:
        return "default"
