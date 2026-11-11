"""Content-free gauges for how the previous gateway life ended and live host memory pressure.

Read from the gateway's own home only (the lifecycle sentinel and the 30 s loop heartbeat, via
``gateway.memory_status.collect_memory_status``): two small host-file reads, no probes, reused for
1 s because every gauge callback rebuilds the whole snapshot (once per gauge per export).  Nothing
is emitted when the watchdog fires: the process hard-exits right after and the emitter is
asynchronous, so the record would never leave; the next boot's ``previous_exit`` gauge carries it.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from agent.monitoring.gateway_health import GatewayMetric

PREVIOUS_EXIT_METRIC = "hermes.gateway.previous_exit"
MEMORY_PRESSURE_METRIC = "hermes.gateway.memory_pressure"
# Worst-first: the first matching label names the previous exit.  Host-scope OOM-kill activity
# (any process on the host) is weaker evidence than the gateway's own watchdog verdict.
_PREVIOUS_EXIT_PRECEDENCE = (
    ("oom_kill_cgroup", lambda s: s.get("last_boot_oom_kill") == "cgroup"),
    ("watchdog_memory_starved", lambda s: s.get("last_boot_memory_starved") is True),
    ("watchdog", lambda s: s.get("last_boot_watchdog_exit") is True),
    ("oom_kill_host", lambda s: s.get("last_boot_oom_kill") == "host"),
    ("suspected_oom", lambda s: s.get("last_boot_suspected_oom") is True),
    ("unclean", lambda s: s.get("last_boot_unclean") is True),
)
_STATUS_TTL_S = 1.0  # below the >= 5 s export interval: one read per collection cycle
_status_lock = threading.Lock()
_status_memo: dict[str, Any] = {}


def previous_exit_class(status: dict[str, Any]) -> str:
    """Closed-set label for the previous life: one of the precedence labels above, ``none``
    (no adverse evidence recorded, which includes a first boot) when this life has a sentinel
    claim and no flag is set, else ``unknown``."""
    for label, matches in _PREVIOUS_EXIT_PRECEDENCE:
        if matches(status):
            return label
    return "none" if status.get("boot_id") else "unknown"


def _memory_status() -> dict[str, Any]:
    """``collect_memory_status()`` for the active home, reused for ``_STATUS_TTL_S``."""
    from gateway.lifecycle_ledger import get_lifecycle_sentinel_path
    from gateway.memory_status import collect_memory_status

    key, now = str(get_lifecycle_sentinel_path()), time.monotonic()  # the home collect_memory_status reads
    with _status_lock:
        if _status_memo.get("key") == key and 0 <= now - _status_memo["at"] < _STATUS_TTL_S:
            return _status_memo["status"]
    status = collect_memory_status()
    with _status_lock:
        _status_memo.update(key=key, at=now, status=status)
    return status


def build_memory_health_metrics(base: dict[str, str]) -> list[GatewayMetric]:
    """``previous_exit`` and ``memory_pressure`` gauges (value 1, bounded attribute) carrying
    the snapshot's ``base`` attributes.  ``memory_pressure`` is one of
    ``gateway.memory_status.PRESSURE_LEVELS``: ``collect_memory_status`` sets it only from
    ``classify_pressure`` or its ``unknown`` default."""
    status = _memory_status()
    return [
        GatewayMetric(name=PREVIOUS_EXIT_METRIC, value=1,
                      attributes={**base, "hermes.previous_exit": previous_exit_class(status)}),
        GatewayMetric(name=MEMORY_PRESSURE_METRIC, value=1,
                      attributes={**base, "hermes.memory_pressure": status["pressure"]}),
    ]
