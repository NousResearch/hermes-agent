"""Memory-pressure bounds for the gateway's per-session AIAgent cache.

Each cached ``AIAgent`` pins its full live transcript (tens of MB on a tool-heavy
session); the LRU cap counts entries, not bytes, and the idle TTL defers eviction
for busy sessions, so neither sees actual memory use.  This module supplies that
signal — own anonymous RSS against a budget derived from the cgroup limit — and
``GatewayRunner`` sheds LRU transcripts via soft eviction (rebuilt from the
persisted session next turn).  Pure/read-only; config under ``agent.agent_cache``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

# Shed well under the limit: once cgroup ``memory.high`` throttling kicks in (swap full),
# a SIGTERM flush cannot finish inside systemd's stop timeout.
_AUTO_BUDGET_FRACTION = 0.65
# Below this a budget is noise — small containers would evict every pass and never keep a warm prefix.
_AUTO_BUDGET_FLOOR_MB = 512
_DEFAULT_MAX_EVICTIONS_PER_PASS = 16
# Never shed the hottest sessions: their prompt cache is worth the most; evicting them
# just moves the cost to the next turn.
_DEFAULT_PROTECT_RECENT = 8
_BYTES_PER_MB = 1024 * 1024
_OFF_WORDS = frozenset({"", "off", "none", "false", "disabled"})


@dataclass(frozen=True)
class AgentCacheBounds:
    """Operator-facing bounds.  ``max_size``/``idle_ttl_secs`` are ``None`` when unset
    so ``gateway/run.py`` keeps its defaults; ``memory_high_mb`` ``None`` = pressure eviction off."""

    max_size: Optional[int] = None
    idle_ttl_secs: Optional[float] = None
    memory_high_mb: Optional[int] = None
    max_evictions_per_pass: int = _DEFAULT_MAX_EVICTIONS_PER_PASS
    protect_recent: int = _DEFAULT_PROTECT_RECENT


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive(value: Any, cast: Callable[[Any], Any] = int) -> Any:
    """``cast(value)`` if it is a positive number (bools rejected), else None."""
    try:
        parsed = None if isinstance(value, bool) or value is None else cast(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and parsed > 0 else None


def _cgroup_limit_bytes() -> Optional[int]:
    """Memory limit this process runs under, if cgroup-capped.

    Prefers v2 ``memory.high`` (the throttling point) over ``memory.max``, then v1.
    Own cgroup first (where a systemd unit's ``MemoryHigh=``/``MemoryMax=`` lands —
    root reads ``max`` there), then root for container-style limits.  ``max`` and
    the v1 near-2^63 sentinel mean unlimited.
    """
    if sys.platform != "linux":
        return None
    try:
        from gateway.cgroup_cleanup import _own_cgroup_path

        own = _own_cgroup_path()
    except Exception:
        own = None
    roots = ([f"/sys/fs/cgroup{own}"] if own and own != "/" else []) + ["/sys/fs/cgroup"]
    for candidate in [f"{r}/memory.{f}" for r in roots for f in ("high", "max")] + ["/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            limit = int(Path(candidate).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):  # unreadable, empty, or "max"
            continue
        if 0 < limit < (1 << 62):
            return limit
    return None


def _total_memory_bytes() -> Optional[int]:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError, AttributeError):
        pass
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def resolve_memory_high_mb(setting: Any) -> Optional[int]:
    """Absolute MB budget: ``"auto"`` derives from the cgroup limit (or total RAM when
    uncapped); a positive number is literal; anything falsy/off disables the pass."""
    if isinstance(setting, str):
        normalized = setting.strip().lower()
        if normalized != "auto":
            return None if normalized in _OFF_WORDS else _positive(normalized)
    elif setting is False:
        return None
    elif setting is not True:
        return _positive(setting)
    limit = _cgroup_limit_bytes() or _total_memory_bytes()
    if not limit:
        return None
    budget = int(limit * _AUTO_BUDGET_FRACTION / _BYTES_PER_MB)
    return budget if budget >= _AUTO_BUDGET_FLOOR_MB else None


def resolve_agent_cache_bounds(config: Any) -> AgentCacheBounds:
    """Read ``agent.agent_cache`` from the *raw* config: the gateway loader does not
    deep-merge ``DEFAULT_CONFIG``, so callers can tell "operator chose 128" from "unset"."""
    section = (config.get("agent") or {}).get("agent_cache") if isinstance(config, dict) else None
    if not isinstance(section, dict):
        section = {}
    protect_recent = section.get("protect_recent")
    protect_parsed = _positive(protect_recent)
    # 0 means "shed anything" — distinct from unset.  The bool guard keeps `protect_recent: false`
    # (False == 0) on the default instead of silently disabling MRU protection.
    if protect_parsed is None and _is_int(protect_recent) and protect_recent == 0:
        protect_parsed = 0
    return AgentCacheBounds(
        max_size=_positive(section.get("max_size")),
        idle_ttl_secs=_positive(section.get("idle_ttl_secs"), float),
        memory_high_mb=resolve_memory_high_mb(section.get("memory_high_mb", "auto")),
        max_evictions_per_pass=_positive(section.get("max_evictions_per_pass")) or _DEFAULT_MAX_EVICTIONS_PER_PASS,
        protect_recent=_DEFAULT_PROTECT_RECENT if protect_parsed is None else protect_parsed,
    )


def read_anon_rss_mb() -> Optional[int]:
    """Anonymous RSS in MB (where cached transcripts live; file-backed pages are noise),
    or None.  ``/proc/self/status`` first; psutil covers other platforms (total RSS only)."""
    try:
        from hermes_cli.mem_trim import collect_memory_snapshot

        snapshot = collect_memory_snapshot()
        for key in ("rss_anon_kib", "rss_kib"):
            kib = snapshot.get(key)
            if isinstance(kib, int) and kib > 0:
                return kib // 1024
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss / _BYTES_PER_MB)
    except Exception:
        return None


def transcript_persistence_caught_up(agent: Any) -> bool:
    """True when the live transcript is fully on disk.

    Soft eviction rebuilds from the persisted session, so it is only safe once
    ``_last_flushed_db_idx`` (advanced only on a fully successful write) has caught
    up.  Unknown shapes are *not* caught up: a skipped eviction costs memory, a
    wrong one costs the conversation.
    """
    messages, flushed = getattr(agent, "_session_messages", None), getattr(agent, "_last_flushed_db_idx", None)
    return isinstance(messages, list) and _is_int(flushed) and flushed >= len(messages)


def plan_pressure_evictions(
    ordered_entries: Iterable[Tuple[str, Any]], *, is_evictable: Callable[[str, Any], bool],
    max_evictions: int, protect_recent: int = 0,
) -> List[Tuple[str, Any]]:
    """Choose which cached sessions to shed, least-recently-used first.

    ``ordered_entries`` must be LRU→MRU (the cache OrderedDict ``move_to_end``s on
    every hit).  The batch is capped so one pass cannot stall the gateway.
    ``protect_recent`` is clamped to half the cache: a few huge transcripts can
    exhaust the budget alone, and a fixed guard would leave nothing to shed.
    """
    entries = list(ordered_entries)
    if max_evictions <= 0 or not entries:
        return []
    protect = min(max(protect_recent, 0), len(entries) // 2)
    if protect:
        entries = entries[:-protect]
    plan: List[Tuple[str, Any]] = []
    for key, agent in entries:
        if len(plan) >= max_evictions:
            break
        if is_evictable(key, agent):
            plan.append((key, agent))
    return plan
