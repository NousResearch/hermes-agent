"""Idle gate for scheduled updates: an unattended slot waits for the gateways to go quiet.

A manual ``hermes update`` drains every gateway it owns (``gateway/restart_when_idle.py``), but
that wait is bounded: past the drain budget a gateway still doing work is left running with its
restart pending and the checkout moves underneath it, so its live turns run old modules against
a new tree (``gateway/code_skew.py``) until they finish. A person who starts an update chose that
moment; a 04:00 timer did not. So before a scheduled slot starts the updater, it asks every
gateway serving this installation's data root (the multiplex host and any standalone profile
gateway) what it is running, waits up to ``updates.auto.idle_wait_minutes`` for all of them to be
idle, and otherwise skips the slot with a ``deferred_busy`` result; the next slot tries again.
``0`` turns the gate off (the updater's own drain still applies). ``run-now`` is never gated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.config import require_readable_config_before_write
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.gateway_lifecycle_guards import live_work
from hermes_cli.update_cmd_drain_report import describe_active_work_unit
from hermes_cli.update_installation import profile_homes
_POLL_S = 15.0


@dataclass(frozen=True)
class IdleVerdict:
    idle: bool
    waited_s: float
    busy: tuple[str, ...]  # one human line per unit still running when the wait ended


def idle_wait_minutes(home: Path) -> float:
    """``updates.auto.idle_wait_minutes`` from the installation data root's config (default 30)."""
    auto = ((require_readable_config_before_write(home / "config.yaml").get("updates") or {}).get("auto") or {})
    value = auto.get("idle_wait_minutes", DEFAULT_CONFIG["updates"]["auto"]["idle_wait_minutes"])
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"updates.auto.idle_wait_minutes must be a number >= 0, got {value!r}")
    return float(value)


def busy_units(home: Path) -> list[str]:
    """Every live work unit on the gateways serving ``home`` and its named profiles."""
    lines: list[str] = []
    for candidate in profile_homes(home):
        label = "default" if candidate == home else candidate.name
        work = live_work(candidate) or {}
        units = [u for u in work.get("active_work") or [] if isinstance(u, dict)]
        # A wedged cron run never finishes, so it must not hold the slot. A restart-safe one does
        # count (unlike for a restart): its worker still imports from the checkout being moved.
        units = [u for u in units if not (u.get("kind") == "cron" and u.get("wedged"))]
        lines += [f"{label}: {describe_active_work_unit(u, candidate)}" for u in units]
        if not units and int(work.get("active_agents") or 0) > 0:
            lines.append(f"{label}: {work['active_agents']} active agent run(s)")
    return lines


def wait_for_idle(home: Path, budget_s: float) -> IdleVerdict:
    """Poll the gateways until none reports work or ``budget_s`` elapses."""
    started = time.monotonic()
    busy = busy_units(home)
    while busy and (elapsed := time.monotonic() - started) < budget_s:
        time.sleep(min(_POLL_S, budget_s - elapsed))
        busy = busy_units(home)
    return IdleVerdict(not busy, round(time.monotonic() - started, 1), tuple(busy))
