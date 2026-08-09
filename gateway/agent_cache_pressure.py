"""Memory-pressure bounds for the gateway's per-session AIAgent cache.

The gateway caches one ``AIAgent`` per session so a long-lived conversation
reuses its prompt prefix instead of rebuilding the system prompt every turn.
Each cached agent also pins ``_session_messages`` — the full live transcript,
tool outputs included, which is tens of MB on a tool-heavy session.

``gateway/run.py`` bounds that cache two ways, and both are blind to how much
memory it actually holds:

* the LRU cap counts *entries*, not bytes, and 128 warm transcripts is
  several GB;
* the idle TTL only sheds agents that went quiet for an hour, and it
  deliberately defers eviction for a finalizable session that has not expired
  yet, so a busy gateway hoards every transcript all day.

This module supplies the missing signal: the process's own anonymous RSS,
compared against a budget derived from the cgroup limit the gateway actually
runs under.  ``GatewayRunner._sweep_agent_cache_under_pressure`` uses it to
shed LRU transcripts through the existing soft-eviction path, which rebuilds
from the persisted session on the next turn (#80764).

Everything here is pure or read-only so it can be tested without a gateway.
Config lives under ``agent.agent_cache`` in ``config.yaml``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

# Fraction of the resolved memory limit at which we start shedding
# transcripts.  Deliberately well under the limit: on the reported incident
# the gateway hit cgroup ``memory.high`` throttling with swap full, and a
# SIGTERM flush from there could not finish inside systemd's stop timeout.
# Eviction has to happen while the process still has room to breathe.
_AUTO_BUDGET_FRACTION = 0.65
# Below this a "budget" is noise — small containers would evict on every pass
# and never keep a warm prefix.
_AUTO_BUDGET_FLOOR_MB = 512

_DEFAULT_MAX_EVICTIONS_PER_PASS = 16
# Never let a pressure pass touch the hottest sessions: they are the ones
# whose prompt cache is worth the most, and shedding them just moves the cost
# to the next turn instead of removing it.
_DEFAULT_PROTECT_RECENT = 8

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class AgentCacheBounds:
    """Operator-facing bounds for the per-session agent cache.

    ``max_size`` and ``idle_ttl_secs`` are ``None`` when the operator did not
    set them, so ``gateway/run.py`` keeps using its module-level defaults.
    ``memory_high_mb`` is ``None`` when pressure eviction is switched off.
    """

    max_size: Optional[int] = None
    idle_ttl_secs: Optional[float] = None
    memory_high_mb: Optional[int] = None
    max_evictions_per_pass: int = _DEFAULT_MAX_EVICTIONS_PER_PASS
    protect_recent: int = _DEFAULT_PROTECT_RECENT


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _cgroup_limit_bytes() -> Optional[int]:
    """Return the memory limit this process runs under, if it is cgroup-capped.

    Prefers cgroup v2 ``memory.high`` (the throttling point — passing it is
    what stalled the reported shutdown) over ``memory.max``, and falls back to
    cgroup v1.  ``max`` / absurd sentinel values mean "unlimited".

    Checks the process's *own* cgroup first (where a systemd unit's
    ``MemoryHigh=``/``MemoryMax=`` actually lands — the root files read
    ``max`` on those deployments), then walks up to the root for
    container-style limits.
    """
    if sys.platform != "linux":
        return None
    candidates: list[str] = []
    try:
        from gateway.cgroup_cleanup import _own_cgroup_path

        own = _own_cgroup_path()
    except Exception:
        own = None
    if own and own != "/":
        candidates.extend(
            (
                f"/sys/fs/cgroup{own}/memory.high",
                f"/sys/fs/cgroup{own}/memory.max",
            )
        )
    candidates.extend(
        (
            "/sys/fs/cgroup/memory.high",
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        )
    )
    for candidate in candidates:
        try:
            raw = Path(candidate).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        # cgroup v1 reports "unlimited" as a near-2^63 sentinel.
        if limit <= 0 or limit >= (1 << 62):
            continue
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
    """Resolve the ``memory_high_mb`` setting into an absolute MB budget.

    ``"auto"`` derives a budget from the cgroup limit the gateway runs under
    (or total RAM when uncapped), which is what makes this fix work out of the
    box on the containerised/systemd deployments where the leak bites.  A
    positive number is taken literally; anything falsy disables the pass.
    """
    if isinstance(setting, str):
        normalized = setting.strip().lower()
        if normalized != "auto":
            return (
                None
                if normalized in ("", "off", "none", "false", "disabled")
                else _positive_int(normalized)
            )
    elif isinstance(setting, bool):
        if not setting:
            return None
    else:
        return _positive_int(setting)

    limit = _cgroup_limit_bytes() or _total_memory_bytes()
    if not limit:
        return None
    budget = int(limit * _AUTO_BUDGET_FRACTION / _BYTES_PER_MB)
    return budget if budget >= _AUTO_BUDGET_FLOOR_MB else None


def resolve_agent_cache_bounds(config: Any) -> AgentCacheBounds:
    """Read ``agent.agent_cache`` out of a raw config mapping.

    Reads the *raw* user config (the gateway's loader does not deep-merge
    ``DEFAULT_CONFIG``), so an absent key stays absent and the caller can tell
    "operator chose 128" from "operator said nothing".
    """
    section: Any = None
    if isinstance(config, dict):
        agent_cfg = config.get("agent")
        if isinstance(agent_cfg, dict):
            section = agent_cfg.get("agent_cache")
    if not isinstance(section, dict):
        section = {}

    max_evictions = _positive_int(section.get("max_evictions_per_pass"))
    protect_recent = section.get("protect_recent")
    protect_parsed = _positive_int(protect_recent)
    if (
        protect_parsed is None
        and isinstance(protect_recent, int)
        and not isinstance(protect_recent, bool)
        and protect_recent == 0
    ):
        # 0 means "shed anything" — distinct from unset. The isinstance
        # guards keep `protect_recent: false` (a YAML-typo bool, False == 0)
        # on the default instead of silently disabling MRU protection.
        protect_parsed = 0

    return AgentCacheBounds(
        max_size=_positive_int(section.get("max_size")),
        idle_ttl_secs=_positive_float(section.get("idle_ttl_secs")),
        memory_high_mb=resolve_memory_high_mb(section.get("memory_high_mb", "auto")),
        max_evictions_per_pass=(
            max_evictions if max_evictions is not None else _DEFAULT_MAX_EVICTIONS_PER_PASS
        ),
        protect_recent=(
            protect_parsed if protect_parsed is not None else _DEFAULT_PROTECT_RECENT
        ),
    )


def read_anon_rss_mb() -> Optional[int]:
    """Return the process's anonymous resident memory in MB, or None.

    Anonymous pages are the ones cached transcripts live in — the reported
    incident measured 11.0 GB of anon out of 11.0 GB total, so file-backed
    pages are noise here.  ``collect_memory_snapshot`` already reads
    ``/proc/self/status`` without a dependency; psutil covers everything else,
    where only total RSS is available.
    """
    try:
        from hermes_cli.mem_trim import collect_memory_snapshot

        snapshot = collect_memory_snapshot()
        anon_kib = snapshot.get("rss_anon_kib")
        if isinstance(anon_kib, int) and anon_kib > 0:
            return anon_kib // 1024
        rss_kib = snapshot.get("rss_kib")
        if isinstance(rss_kib, int) and rss_kib > 0:
            return rss_kib // 1024
    except Exception:
        pass

    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss / _BYTES_PER_MB)
    except Exception:
        return None


def transcript_persistence_caught_up(agent: Any) -> bool:
    """True when the agent's live transcript is fully on disk.

    Soft eviction drops ``_session_messages`` and rebuilds it from the
    persisted session next turn, so it is only safe once persistence has
    caught up.  ``_last_flushed_db_idx`` is advanced to ``len(messages)`` by
    ``AIAgent._flush_messages_to_session_db`` and only on a fully successful
    write — the same divergence the FTS write-corruption guard reacts to when
    it preserves live history over a lagging transcript.  Unknown shapes are
    treated as *not* caught up: a skipped eviction costs memory, a wrong one
    costs the user their conversation.
    """
    messages = getattr(agent, "_session_messages", None)
    if not isinstance(messages, list):
        return False
    flushed = getattr(agent, "_last_flushed_db_idx", None)
    if not isinstance(flushed, int) or isinstance(flushed, bool):
        return False
    return flushed >= len(messages)


def plan_pressure_evictions(
    ordered_entries: Iterable[Tuple[str, Any]],
    *,
    is_evictable: Callable[[str, Any], bool],
    max_evictions: int,
    protect_recent: int = 0,
) -> List[Tuple[str, Any]]:
    """Choose which cached sessions to shed, least-recently-used first.

    ``ordered_entries`` must be in LRU→MRU order (the cache is an
    ``OrderedDict`` kept in that order by ``move_to_end`` on every hit).  The
    batch is capped so one pass cannot stall the gateway tearing down clients.

    ``protect_recent`` is an upper bound, clamped to half the cache: a handful
    of sessions can be big enough to exhaust the budget on their own (a single
    tool-heavy transcript runs to hundreds of MB), and a fixed guard would
    then protect the entire cache and leave the gateway climbing toward the
    OOM killer with nothing it is willing to shed.
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
