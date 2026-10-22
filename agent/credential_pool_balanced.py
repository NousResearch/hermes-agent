"""The ``balanced`` pool strategy: spread sessions by quota headroom without losing warm prompt caches.

Subscription accounts (Claude OAuth, ChatGPT Codex) report their plan windows on every response.
This module keeps, in process memory, the latest report per pooled credential and which sessions
have a warm prompt cache on which credential. With it:

- a new session goes to the credential with the most headroom, divided among the sessions
  already on it; a session counts there from the moment it is placed, so sessions placed or
  rotated in the same moment spread instead of piling onto one account;
- a rotation off a spent Claude credential stays inside its organization while one is
  available: measured, seats of one Claude Team organization read each other's prompt cache, so
  that move costs no cache rewrite (Codex workspaces are not assumed to share one);
- between turns a warm session stays on, or returns to, the credential its cache is on, and a
  cold one is re-placed (``agent.turn_credential_placement``).

The view lives in process memory, keyed by provider and pool entry id. After a restart it is
empty, and sessions spread by count alone until responses report the windows.

Headroom is the fraction left in the tightest window, so plan size needs no table: a bigger plan
fills more slowly and keeps winning new sessions until the percentages even out.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

from agent.credential_pool_quota_window import _UNIFIED_PREFIX
from agent.credential_pool_timestamps import _parse_absolute_timestamp

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

logger = logging.getLogger(__name__)

STRATEGY_BALANCED = "balanced"
# Anthropic cache tiers. Hermes requests no retention on the ChatGPT Codex route, so the server
# default applies: OpenAI documents in-memory prompt caches as kept up to an hour of inactivity.
_CACHE_TTL_S = {"5m": 300.0, "1h": 3600.0}
_CODEX_CACHE_LIFETIME_S = _CACHE_TTL_S["1h"]


@dataclass(frozen=True)
class QuotaWindow:
    used: float  # fraction of the window spent, 0..1
    resets_at: Optional[float]


@dataclass(frozen=True)
class _Observed:
    windows: Mapping[str, QuotaWindow]
    group: Optional[str]


_NOTHING = _Observed({}, None)
_lock = threading.Lock()
_observed: dict[tuple[str, str], _Observed] = {}  # (provider, entry id) -> latest report
_warm: dict[tuple[str, str], tuple[str, float]] = {}  # (provider, session id) -> (entry id, warm until)


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def quota_windows(headers: Mapping[str, str]) -> dict[str, QuotaWindow]:
    """Plan windows reported by one response: Anthropic ``unified-<window>-*`` or Codex
    ``x-codex-<slot>-*`` (slots are named by their length, the primary slot can be the week)."""
    windows: dict[str, QuotaWindow] = {}
    for key in headers.keys():
        name = str(key).lower()
        if name.startswith(_UNIFIED_PREFIX) and name.endswith("-utilization"):
            window = name[len(_UNIFIED_PREFIX):-len("-utilization")]
            used = _number(headers.get(key))
            if used is not None:
                reset = _parse_absolute_timestamp(headers.get(f"{_UNIFIED_PREFIX}{window}-reset"))
                windows[window] = QuotaWindow(used, reset)
    for slot in ("primary", "secondary"):
        minutes = _number(headers.get(f"x-codex-{slot}-window-minutes"))
        used = _number(headers.get(f"x-codex-{slot}-used-percent"))
        if minutes and used is not None:
            windows[f"{int(minutes)}m"] = QuotaWindow(
                used / 100, _parse_absolute_timestamp(headers.get(f"x-codex-{slot}-reset-at")))
    return windows


def observe(
    provider: str, entry_id: str, headers: Mapping[str, str], *,
    session_id: Optional[str] = None, warm_for: float = 0.0, now: Optional[float] = None,
) -> None:
    """Record what is known about *entry_id*: the quota windows and cache group in *headers* (a
    response it served; empty for a placement), and that *session_id* is on it, its prompt cache
    warm there for *warm_for* seconds."""
    now = time.time() if now is None else now
    windows = quota_windows(headers)
    group = headers.get("anthropic-organization-id")
    with _lock:
        previous = _observed.get((provider, entry_id), _NOTHING)
        if windows or group:
            _observed[(provider, entry_id)] = _Observed(windows or previous.windows, group or previous.group)
        if session_id and warm_for > 0:
            _warm[(provider, session_id)] = (entry_id, now + warm_for)
        for key in [key for key, (_entry, until) in _warm.items() if until <= now]:
            del _warm[key]


def warm_entry(provider: str, session_id: Optional[str], *, now: Optional[float] = None) -> Optional[str]:
    """The credential *session_id*'s prompt cache is still warm on, if any."""
    now = time.time() if now is None else now
    with _lock:
        entry_id, until = _warm.get((provider, session_id or ""), ("", 0.0))
    return entry_id if until > now else None


def _headroom(observed: _Observed, now: float) -> Optional[float]:
    """Fraction left in the tightest window that has not reset yet; None when nothing is known."""
    if not observed.windows:
        return None
    live = [w.used for w in observed.windows.values() if w.resets_at is None or w.resets_at > now]
    return max(0.0, 1.0 - max(live, default=0.0))


def _binding_reset(observed: _Observed, now: float) -> float:
    """When the tightest window resets: quota still unused at its reset is lost."""
    pending = [w for w in observed.windows.values() if w.resets_at is not None and w.resets_at > now]
    tightest = max(pending, key=lambda w: w.used, default=None)
    return tightest.resets_at if tightest is not None and tightest.resets_at is not None else float("inf")


def choose(
    provider: str, available: list[PooledCredential], *, rotating_off: Optional[str] = None,
    now: Optional[float] = None,
) -> PooledCredential:
    """Most headroom per session on it; ties go to the window that resets first, then priority.

    With *rotating_off*, candidates in that entry's cache group come first. A credential
    nothing has been heard from yet counts as the median of the ones that have."""
    now = time.time() if now is None else now
    with _lock:
        observed = {entry.id: _observed.get((provider, entry.id), _NOTHING) for entry in available}
        group = _observed.get((provider, rotating_off or ""), _NOTHING).group
        load: dict[str, int] = {}
        for (warm_provider, _session), (entry_id, until) in _warm.items():
            if warm_provider == provider and until > now:
                load[entry_id] = load.get(entry_id, 0) + 1
    if group:
        same_group = [entry for entry in available if observed[entry.id].group == group]
        available = same_group or available
    known = [h for h in (_headroom(observed[e.id], now) for e in available) if h is not None]
    unknown_headroom = statistics.median(known) if known else 1.0

    def _rank(entry: PooledCredential) -> tuple[float, float]:
        headroom = _headroom(observed[entry.id], now)
        share = (unknown_headroom if headroom is None else headroom) / (1 + load.get(entry.id, 0))
        return -share, _binding_reset(observed[entry.id], now)

    return min(available, key=_rank)


def observe_agent_response(agent: Any, http_response: Any) -> None:
    """Feed one response to the view: the serving credential's windows and cache group, and that
    the agent's session is warm on it. Fail-open: never breaks the request that carried it."""
    headers = getattr(http_response, "headers", None)
    if headers:
        _observe_agent(agent, getattr(agent, "_credential_pool_entry_id", None), headers)


def observe_agent_placement(agent: Any, entry_id: Optional[str]) -> None:
    """Count the agent's session as load on *entry_id* from the moment it is placed there."""
    _observe_agent(agent, entry_id, {})


def _observe_agent(agent: Any, entry_id: Optional[str], headers: Mapping[str, str]) -> None:
    pool = getattr(agent, "_credential_pool", None)
    if getattr(pool, "balanced", False) is not True or not isinstance(entry_id, str) or not entry_id:
        return
    try:
        observe(pool.provider, entry_id, headers,
                session_id=getattr(agent, "session_id", None), warm_for=_warm_for_agent(agent))
    except Exception:
        logger.debug("pool quota observation failed", exc_info=True)


def _warm_for_agent(agent: Any) -> float:
    """How long this agent's prompt cache outlives its last request."""
    if str(getattr(agent, "provider", "") or "") == "openai-codex":
        return _CODEX_CACHE_LIFETIME_S
    if not getattr(agent, "_use_prompt_caching", False):
        return 0.0  # nothing cached, so nothing to lose by moving
    from agent.prompt_caching import effective_cache_ttl
    ttl = effective_cache_ttl(getattr(agent, "_cache_ttl", None), provider=getattr(agent, "provider", ""),
                              model=getattr(agent, "model", ""))
    return _CACHE_TTL_S.get(ttl, _CACHE_TTL_S["5m"])


class CredentialPoolBalancedMixin:
    @property
    def balanced(self) -> bool:
        return self._strategy == STRATEGY_BALANCED

    def _choose_balanced(self, available: list[PooledCredential], rotating_off: Optional[str]) -> PooledCredential:
        """``balanced`` selection; a rotation keeps the cache group of the entry it rotates off."""
        return choose(self.provider, available, rotating_off=rotating_off)

    def _choose_balanced_lease(self, candidates: list[PooledCredential]) -> PooledCredential:
        """Subagents spawned together have no warm cache yet: spread them by lease count first,
        then by headroom."""
        fewest = min(self._active_leases.get(entry.id, 0) for entry in candidates)
        return choose(self.provider, [entry for entry in candidates if self._active_leases.get(entry.id, 0) == fewest])

    def best_entry_for_cold_session(
        self, *, model: Optional[str] = None, base_url: Optional[str] = None,
    ) -> Optional[PooledCredential]:
        """The credential a cold session should move to: the ``balanced`` pick, ready to use."""
        from agent.credential_pool import credential_pool_entry_serves_endpoint

        with self._lock:
            available, _pending = self._available_entries(clear_expired=True, refresh=False, model=model)
            if base_url:
                available = [entry for entry in available if credential_pool_entry_serves_endpoint(entry, base_url)]
            if not available:
                return None
            best = choose(self.provider, available)
        return self.reclaim(best.id, model=model)
