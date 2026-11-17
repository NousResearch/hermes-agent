"""What Hermes has learned about how each route really caches, across sessions.

The prompt-cache clock assumes a route keeps what its API spec promises (the tier a marked request asks
for, 30 minutes for an automatic cache). Some routes do not: a model that writes every call and never
reads back, one that re-writes the whole prefix on every hit, an automatic cache that is gone in minutes.
This store learns that from the usage Hermes already receives (normal responses and keep-alives; it
never sends a probe) and the warmer and marker policy consult it at decision time.

Key: ``(provider, base_url host, model, explicit|implicit)``, implicit = no ``cache_control`` markers.
Per key: ``reads`` (hits after a known earlier touch), ``attempts`` (a later request inside the tier
that read nothing), ``write_on_hit`` (a keep-alive, an unchanged replay, that hit and still wrote as many
tokens as it read: the same prefix again; an ordinary turn appends new content, so its sizes prove nothing),
the gaps (seconds since the prefix was last touched) of hits and misses, and ``updated``.

Rules (each needs two consistent samples, so one stray miss never downgrades a route):

- ``no-cache``: never read back after >= 2 attempts. No keep-alives; an explicit route stops sending
  markers (they only buy the write premium).
- ``write-on-hit``: >= 2 hits that re-wrote the prefix. No keep-alives (a keep-alive cannot save).
- learned lifetime: >= 2 misses at a gap longer than every hit seen, shorter than the tier. Keep-alives
  fire at 5/6 of the shortest such gap, or stop when that is under a minute. Never longer than the tier.
- ``explicit``: the implicit cache dies early (learned lifetime) while the explicit one reads: prefer
  markers for the model.

Facts expire 7 days after their last update and are learned again. One small JSON file per profile
(``<HERMES_HOME>/cache/prompt_cache_routes.json``), resolved at call time, written atomically, at most
:data:`MAX_ENTRIES` keys (least recently updated dropped first).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_ENTRIES = 500
EXPIRY_SECONDS = 7 * 24 * 3600
MIN_SAMPLES = 2
#: A learned lifetime whose keep-alive would be due sooner than this is not worth warming.
MIN_LEARNED_FIRE_SECONDS = 60.0
#: A hit whose write matches its read within this share re-wrote the very prefix it read (Gemini on Nous:
#: read 8,582 and write 8,582 on every call), not a new tail of a different size.
_SAME_PREFIX_SHARE = 0.05
#: Routine hits rewrite at most this share of what they read (a tail); used to skip unneeded writes.
_REWRITE_SHARE = 0.5
_GAPS_KEPT = 8
_lock = threading.Lock()


def _path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "prompt_cache_routes.json"


def route_key(provider: str, base_url: str, model: str, explicit: bool) -> str:
    from utils import base_url_hostname

    host = base_url_hostname(base_url or "") or ""
    return "|".join((str(provider or ""), host, str(model or ""), "explicit" if explicit else "implicit"))


def _load(path: Path, now: float) -> dict[str, dict[str, Any]]:
    from utils.files import read_json_or_empty

    data = read_json_or_empty(path)
    return {k: v for k, v in data.items()
            if isinstance(v, dict) and now - float(v.get("updated") or 0) < EXPIRY_SECONDS}


def _save(path: Path, data: dict[str, dict[str, Any]]) -> None:
    from utils.files import atomic_json_write

    if len(data) > MAX_ENTRIES:
        keep = sorted(data, key=lambda k: float(data[k].get("updated") or 0))[-MAX_ENTRIES:]
        data = {k: data[k] for k in keep}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, data)


def _push(entry: dict[str, Any], field: str, gap: float) -> None:
    entry[field] = ([*entry.get(field, []), round(gap, 1)])[-_GAPS_KEPT:]


def _routine_hit(entry: dict[str, Any], gap_s: Optional[float], read: int, write: int, now: float) -> bool:
    """A hit that teaches nothing new (enough reads, no rewrite, no longer gap than seen, fresh entry):
    skipped so an ordinary chat does not rewrite the store on every response."""
    return (int(entry.get("reads", 0)) >= MIN_SAMPLES and write < _REWRITE_SHARE * read
            and (gap_s is None or gap_s <= max(entry.get("hit_gaps") or [0.0]))
            and now - float(entry.get("updated") or 0) < 24 * 3600)


def observe(*, provider: str, base_url: str, model: str, explicit: bool, read: int, write: int,
            gap_s: Optional[float], tier_s: Optional[float], now: Optional[float] = None,
            replay: bool = False) -> None:
    """Fold one response's cache usage. ``gap_s`` is the time since the same prefix was last touched
    (None when the clock knows no earlier touch); ``tier_s`` the lifetime the clock assumed for it;
    ``replay`` whether the request re-sent the prefix unchanged (a keep-alive). A response that read and
    wrote nothing still counts as a miss when the prefix had a cache entry."""
    now = time.time() if now is None else now
    hit = read > 0
    known = gap_s is not None and tier_s is not None and gap_s < tier_s
    if not hit and not known:
        return  # a first write: nothing to learn yet
    key = route_key(provider, base_url, model, explicit)
    path = _path()
    try:
        with _lock:
            data = _load(path, now)
            entry = data.setdefault(key, {})
            if hit and _routine_hit(entry, gap_s, read, write, now):
                return
            if hit:
                entry["reads"] = int(entry.get("reads", 0)) + 1
                if gap_s is not None:
                    _push(entry, "hit_gaps", gap_s)
                if replay and abs(write - read) <= _SAME_PREFIX_SHARE * read:
                    entry["write_on_hit"] = int(entry.get("write_on_hit", 0)) + 1
            else:
                entry["attempts"] = int(entry.get("attempts", 0)) + 1
                _push(entry, "miss_gaps", gap_s or 0.0)
            entry["samples"] = int(entry.get("samples", 0)) + 1
            entry["updated"] = now
            _save(path, data)
    except OSError as exc:  # the store is advisory: a read-only home learns nothing
        logger.debug("prompt-cache route store not written (%s)", exc)


def _entry(key: str, now: float) -> dict[str, Any]:
    try:
        return _load(_path(), now).get(key) or {}
    except OSError:
        return {}


def learned_lifetime(entry: dict[str, Any], tier_s: float) -> Optional[float]:
    """The shortest miss gap past every hit and under the tier, when >= 2 misses agree; else None."""
    longest_hit = max(entry.get("hit_gaps") or [0.0])
    early = [g for g in entry.get("miss_gaps") or () if longest_hit < g < tier_s]
    return min(early) if len(early) >= MIN_SAMPLES else None


def rule_for(provider: str, base_url: str, model: str, explicit: bool, *, tier_s: Optional[float] = None,
             now: Optional[float] = None) -> Optional[str]:
    """The learned rule in effect for a route: ``no-cache`` | ``write-on-hit`` | ``short-lived`` |
    ``lifetime:<s>`` (warm on that lifetime instead of the tier), or None."""
    now = time.time() if now is None else now
    entry = _entry(route_key(provider, base_url, model, explicit), now)
    if not entry:
        return None
    if not entry.get("reads") and int(entry.get("attempts", 0)) >= MIN_SAMPLES:
        return "no-cache"
    if int(entry.get("write_on_hit", 0)) >= MIN_SAMPLES:
        return "write-on-hit"
    if tier_s and (life := learned_lifetime(entry, tier_s)) is not None:
        from agent.prompt_cache_lifecycle import cache_warming_delay_s

        delay = cache_warming_delay_s(life)
        return "short-lived" if delay is None or delay < MIN_LEARNED_FIRE_SECONDS else f"lifetime:{int(life)}"
    return None


def marker_preference(provider: str, base_url: str, model: str, *, now: Optional[float] = None) -> Optional[bool]:
    """Learned marker choice for a model: False when its explicit cache never reads back, True when its
    automatic cache dies early while the explicit one reads, else None (the route's policy stands)."""
    now = time.time() if now is None else now
    if rule_for(provider, base_url, model, True, now=now) == "no-cache":
        return False
    implicit = _entry(route_key(provider, base_url, model, False), now)
    explicit = _entry(route_key(provider, base_url, model, True), now)
    from agent.prompt_caching import TTL_SECONDS

    if explicit.get("reads") and implicit and learned_lifetime(implicit, TTL_SECONDS["30m"]) is not None:
        return True
    return None


def implicit_ever_read(provider: str, base_url: str, model: str, *, now: Optional[float] = None) -> bool:
    now = time.time() if now is None else now
    return bool(_entry(route_key(provider, base_url, model, False), now).get("reads"))


def display_rule(rule: Optional[str]) -> Optional[str]:
    """The one short word ``/context`` shows for a learned rule."""
    if not rule:
        return None
    return "short-lived" if rule.startswith("lifetime:") else rule
