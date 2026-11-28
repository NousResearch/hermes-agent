"""Provider-endpoint lanes for the kanban dispatcher.

``kanban.max_in_progress_per_endpoint`` caps how many workers may run against
one provider route at once: one local model server (both GTX 1080 Ti behind
``localhost:11434``) or one cloud account (``openrouter``) is a single quota
lane, so a fan-out across profiles still has to queue behind it.

This module owns the lane key (normalisation + resolution from a task row) and
the per-tick counting; the dispatcher only asks it whether a candidate fits.
Keys are what an operator writes in config:

* ``host:port`` from ``model.base_url`` with the scheme and path stripped;
* the provider name when ``base_url`` is empty (one account = one lane);
* ``"*"`` — the fallback lane — when the route cannot be resolved.

Loopback spellings (``localhost``, IPv4 loopback addresses, ``::1``, bracketed
IPv6 forms) all collapse to ``localhost`` so the two address families of one
local server share a counter.
"""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Mapping
from typing import Any, Optional
from urllib.parse import urlsplit

_log = logging.getLogger(__name__)

# Fallback lane: configured with a ``"*"`` key, used for any card whose endpoint
# has no entry of its own — including one that cannot be resolved at all.
FALLBACK_KEY = "*"


def normalise_endpoint_key(value: Optional[str]) -> Optional[str]:
    """Reduce a base URL, provider name or configured key to one lane key.

    ``http://localhost:11434/v1`` → ``localhost:11434``; the host is
    lowercased and loopback spellings collapse to ``localhost``. Returns
    ``None`` for an empty or unparseable value (the caller falls to ``"*"``).
    """
    text = (value or "").strip()
    if not text:
        return None
    if text == FALLBACK_KEY:
        return FALLBACK_KEY
    # A bare ``host:port`` / provider name has no scheme; ``//`` makes urlsplit
    # read it as a netloc instead of a path.
    parsed = urlsplit(text if "://" in text else f"//{text}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return None
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if loopback:
        host = "localhost"
    try:
        port = parsed.port
    except ValueError:
        return None
    return f"{host}:{port}" if port is not None else host


def endpoint_key(provider: Optional[str], base_url: Optional[str]) -> str:
    """Lane key for a resolved route: the normalised base_url, else the provider.

    Returns :data:`FALLBACK_KEY` when neither resolves, or when a base_url is
    present but unparseable (a card whose endpoint cannot be resolved must not
    silently share a server's lane).
    """
    if (base_url or "").strip():
        return normalise_endpoint_key(base_url) or FALLBACK_KEY
    return normalise_endpoint_key(provider) or FALLBACK_KEY


def parse_caps(raw: Any) -> Optional[dict[str, int]]:
    """Parse ``kanban.max_in_progress_per_endpoint``.

    ``None`` when unset (or nothing valid is left) — the caller then applies no
    endpoint cap at all, byte-identical to an install without the key. Keys are
    normalised like every other endpoint key, so an IPv4-loopback spelling and
    ``"localhost:11434"`` configure the same lane; a junk key is dropped with a
    warning rather than silently capping a different lane.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        _log.warning("kanban.max_in_progress_per_endpoint=%r is not a mapping; ignoring", raw)
        return None
    caps: dict[str, int] = {}
    for raw_key, raw_cap in raw.items():
        if isinstance(raw_cap, bool):
            cap = 0
        else:
            try:
                cap = int(raw_cap)
            except (TypeError, ValueError):
                cap = 0
        if cap < 1:
            _log.warning(
                "kanban.max_in_progress_per_endpoint[%r]=%r is not a positive int; ignoring",
                raw_key, raw_cap,
            )
            continue
        key = normalise_endpoint_key(str(raw_key))
        if key is None:
            _log.warning("kanban.max_in_progress_per_endpoint: unusable key %r; ignoring", raw_key)
            continue
        caps[key] = cap
    return caps or None


def cap_for(caps: Mapping[str, int], endpoint: str) -> Optional[int]:
    """Effective cap for one lane: its own entry, else the ``"*"`` fallback."""
    if endpoint in caps:
        return caps[endpoint]
    return caps.get(FALLBACK_KEY)


def lane_is_full(
    caps: Mapping[str, int],
    running: Mapping[str, int],
    endpoint: str,
) -> Optional[int]:
    """``running`` count for *endpoint* when the lane is at (or over) its cap.

    ``None`` means "spawn allowed": the lane has no cap, or it still has room.
    """
    cap = cap_for(caps, endpoint)
    if cap is None:
        return None
    current = running.get(endpoint, 0)
    return current if current >= cap else None


def note_candidate(
    result: Any,
    task_id: str,
    row: Any,
    caps: Optional[Mapping[str, int]],
    running: Mapping[str, int],
    routes: dict[str, tuple[Optional[str], Optional[str]]],
) -> tuple[Optional[str], Optional[int]]:
    """Resolve this candidate's lane and record it on *result*.

    Returns ``(endpoint, running_count_when_full)``; ``(None, None)`` when no
    endpoint cap is configured. Recording happens here so the one place that
    knows the lane key is also the place that reports it on ``DispatchResult``
    (the ``spawned`` event carries it too).
    """
    if caps is None:
        return None, None
    endpoint = endpoint_for_row(row, routes)
    result.endpoint_by_task[task_id] = endpoint
    return endpoint, lane_is_full(caps, running, endpoint)


def profile_route(
    profile: str,
    cache: dict[str, tuple[Optional[str], Optional[str]]],
) -> tuple[Optional[str], Optional[str]]:
    """``(provider, base_url)`` from a profile's own ``config.yaml``.

    Cached for the tick in *cache*: many candidates share one assignee. Fails
    open to ``(None, None)`` — an unreadable profile config must never wedge
    dispatch, it just sends that card to the ``"*"`` lane.
    """
    if profile in cache:
        return cache[profile]
    route: tuple[Optional[str], Optional[str]] = (None, None)
    try:
        from hermes_cli.config_effective import load_user_config_effective
        from hermes_cli.profiles import get_profile_dir

        config = load_user_config_effective(get_profile_dir(profile) / "config.yaml")
        model = config.get("model") if isinstance(config, Mapping) else None
        if isinstance(model, Mapping):
            provider = model.get("provider")
            base_url = model.get("base_url")
            route = (
                provider.strip() if isinstance(provider, str) and provider.strip() else None,
                base_url.strip() if isinstance(base_url, str) and base_url.strip() else None,
            )
    except Exception as exc:  # health: allow BLE001 -- an unreadable profile config must not stop the tick
        _log.debug("kanban endpoint cap: no model route for profile %r: %s", profile, exc)
    cache[profile] = route
    return route


def endpoint_for_row(
    row: Any,
    cache: dict[str, tuple[Optional[str], Optional[str]]],
) -> str:
    """Resolved lane key for one task row (``id, assignee, *_override`` selected).

    A card-level override pins the route its worker will actually call. The
    card stores no base_url, so an override naming a DIFFERENT provider than the
    profile's keys on that provider name (one account = one lane); an override
    that keeps the profile's provider reuses the profile's base_url.
    """
    provider = (row["provider_override"] or "").strip()
    model = (row["model_override"] or "").strip()
    profile_provider, profile_base_url = profile_route(row["assignee"] or "", cache)
    if provider or model:
        if provider and provider.lower() != (profile_provider or "").lower():
            return endpoint_key(provider, None)
        return endpoint_key(provider or profile_provider, profile_base_url)
    return endpoint_key(profile_provider, profile_base_url)


def running_counts(conn: Any, cache: dict[str, tuple[Optional[str], Optional[str]]]) -> dict[str, int]:
    """Per-lane ``running`` counts, recomputed from the DB once per tick."""
    counts: dict[str, int] = {}
    for row in conn.execute(
        "SELECT id, assignee, provider_override, model_override FROM tasks WHERE status = 'running'"
    ):
        key = endpoint_for_row(row, cache)
        counts[key] = counts.get(key, 0) + 1
    return counts
