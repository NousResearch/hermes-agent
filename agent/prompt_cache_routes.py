"""What Hermes has learned about how each route really caches, across sessions.

The prompt-cache clock assumes a route keeps what its API spec promises (the tier a marked request asks
for, 30 minutes for an automatic cache). Some routes do not: a model that writes every call and never
reads back, one that re-writes the whole prefix on every hit, an automatic cache that is gone in minutes.
This store learns that from the usage Hermes already receives (normal responses and keep-alives; it
never sends a probe) and the warmer and marker policy consult it at decision time.

Key: ``(provider, normalized base_url, model, explicit|implicit)``, implicit = no ``cache_control`` markers.
Per key: ``reads`` (hits after a known earlier touch), ``attempts`` (a later request inside the tier
that read nothing), ``write_on_hit`` (a keep-alive, an unchanged replay, that hit and still wrote as many
tokens as it read: the same prefix again; an ordinary turn appends new content, so its sizes prove nothing),
the gaps (seconds since the prefix was last touched) of hits and misses, and ``updated``.

Rules (each needs two consistent samples from two DIFFERENT sessions, so one stray miss, or one chat's
routing streak, never downgrades a route; a miss served by another upstream than the one that cached the
prefix is never a sample, see ``prompt_cache_clock.response_upstream``):

- ``no-cache``: never read back after >= 2 attempts. No keep-alives; an explicit route stops sending
  markers (they only buy the write premium).
- ``write-on-hit``: >= 2 hits that re-wrote the prefix. No keep-alives (a keep-alive cannot save).
- learned lifetime: >= 2 misses at a gap longer than the longest hit ever seen (kept apart from the
  recent gaps, so it survives their trim), shorter than the tier. Keep-alives fire at 5/6 of the shortest
  such gap, or stop when that is under a minute. Never longer than the tier.
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
    """The endpoint is the normalized base URL (scheme, host, port, path): two local servers on one host
    (llama.cpp on :8080, Ollama on :11434) serving the same model name are different caches."""
    from hermes_cli.route_identity import normalize_route_base_url

    endpoint = normalize_route_base_url(base_url or "")
    return "|".join((str(provider or ""), endpoint, str(model or ""), "explicit" if explicit else "implicit"))


def _load(path: Path, now: float) -> dict[str, dict[str, Any]]:
    from utils.files import read_json_or_empty

    data = read_json_or_empty(path)
    return {k: v for k, v in data.items() if _well_formed(v) and now - v["updated"] < EXPIRY_SECONDS}


def _numbers(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)


def _well_formed(entry: Any) -> bool:
    """An entry every reader can trust: numeric timestamp, counters and gap lists, string session lists.
    A hand-edited or foreign file's bad entry is dropped (and rewritten away on the next save), never raised."""
    if not isinstance(entry, dict) or not isinstance(entry.get("updated"), (int, float)):
        return False
    counters = ("reads", "attempts", "write_on_hit", "samples", "longest_hit")
    gaps = ("hit_gaps", "miss_gaps", "write_on_hit_gaps")
    sessions = ("miss_gaps_sessions", "write_on_hit_gaps_sessions")
    return (all(isinstance(entry.get(k, 0), (int, float)) and not isinstance(entry.get(k, 0), bool) for k in counters)
            and all(_numbers(entry.get(k, [])) for k in gaps)
            and all(isinstance(entry.get(k, []), list) and all(isinstance(s, str) for s in entry.get(k, []))
                    for k in sessions))


def _save(path: Path, data: dict[str, dict[str, Any]]) -> None:
    from utils.files import atomic_json_write

    if len(data) > MAX_ENTRIES:
        keep = sorted(data, key=lambda k: float(data[k].get("updated") or 0))[-MAX_ENTRIES:]
        data = {k: data[k] for k in keep}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, data)


def _push(entry: dict[str, Any], field: str, gap: float, session_id: str = "") -> None:
    """Append ``gap`` (and, for misses, the session it came from in ``<field>_sessions``), keeping the
    last :data:`_GAPS_KEPT`."""
    entry[field] = ([*entry.get(field, []), round(gap, 1)])[-_GAPS_KEPT:]
    if field != "hit_gaps":
        sessions = [*entry.get(f"{field}_sessions", []), session_id][-_GAPS_KEPT:]
        entry[f"{field}_sessions"] = sessions


def _sessions(entry: dict[str, Any], field: str, keep: Optional[list[bool]] = None) -> int:
    """How many distinct sessions the samples in ``field`` came from (only those ``keep`` marks)."""
    sids = list(entry.get(f"{field}_sessions") or ())
    values = list(entry.get(field) or ())
    sids = ([""] * len(values) + sids)[-len(values):] if values else []  # older entries kept no sessions
    return len({sid for i, sid in enumerate(sids) if keep is None or keep[i]})


def _routine_hit(entry: dict[str, Any], gap_s: Optional[float], read: int, write: int, now: float) -> bool:
    """A hit that teaches nothing new (enough reads, no rewrite, no longer gap than seen, fresh entry):
    skipped so an ordinary chat does not rewrite the store on every response."""
    return (int(entry.get("reads", 0)) >= MIN_SAMPLES and write < _REWRITE_SHARE * read
            and (gap_s is None or gap_s <= max(entry.get("hit_gaps") or [0.0]))
            and now - float(entry.get("updated") or 0) < 24 * 3600)


def observe(*, provider: str, base_url: str, model: str, explicit: bool, read: int, write: int,
            gap_s: Optional[float], tier_s: Optional[float], now: Optional[float] = None,
            replay: bool = False, session_id: str = "") -> None:
    """Fold one response's cache usage. ``gap_s`` is the time since the same prefix was last touched
    (None when the clock knows no earlier touch); ``tier_s`` the lifetime the clock assumed for it;
    ``replay`` whether the request re-sent the prefix unchanged (a keep-alive); ``session_id`` the chat it
    came from (a downgrade needs samples from two). A response that read and wrote nothing still counts
    as a miss when the prefix had a cache entry."""
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
                    entry["longest_hit"] = max(float(entry.get("longest_hit") or 0.0), round(gap_s, 1))
                if replay and abs(write - read) <= _SAME_PREFIX_SHARE * read:
                    entry["write_on_hit"] = int(entry.get("write_on_hit", 0)) + 1
                    _push(entry, "write_on_hit_gaps", gap_s or 0.0, session_id)
            else:
                entry["attempts"] = int(entry.get("attempts", 0)) + 1
                _push(entry, "miss_gaps", gap_s or 0.0, session_id)
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
    """The shortest miss gap past the longest hit ever seen and under the tier, when >= 2 such misses
    from >= 2 sessions agree; else None."""
    longest_hit = max([float(entry.get("longest_hit") or 0.0), *(entry.get("hit_gaps") or ())])
    gaps = list(entry.get("miss_gaps") or ())
    keep = [longest_hit < g < tier_s for g in gaps]
    early = [g for g, k in zip(gaps, keep) if k]
    if len(early) < MIN_SAMPLES or _sessions(entry, "miss_gaps", keep) < MIN_SAMPLES:
        return None
    return min(early)


def rule_for(provider: str, base_url: str, model: str, explicit: bool, *, tier_s: Optional[float] = None,
             now: Optional[float] = None) -> Optional[str]:
    """The learned rule in effect for a route: ``no-cache`` | ``write-on-hit`` | ``short-lived`` |
    ``lifetime:<s>`` (warm on that lifetime instead of the tier), or None."""
    now = time.time() if now is None else now
    entry = _entry(route_key(provider, base_url, model, explicit), now)
    if not entry:
        return None
    if (not entry.get("reads") and int(entry.get("attempts", 0)) >= MIN_SAMPLES
            and _sessions(entry, "miss_gaps") >= MIN_SAMPLES):
        return "no-cache"
    if int(entry.get("write_on_hit", 0)) >= MIN_SAMPLES and _sessions(entry, "write_on_hit_gaps") >= MIN_SAMPLES:
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
