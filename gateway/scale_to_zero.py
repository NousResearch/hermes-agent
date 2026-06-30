"""Scale-to-zero idle detection + dormant-quiesce for the gateway (Phase 0).

This is the gateway-side BEHAVIOUR layer that consumes the relay scale-to-zero
PRIMITIVES (gateway-gateway Phase 5: the buffered-flip, the durable per-instance
buffer, the wakeUrl poke, the reconnect supervisor). It owns the *decision* to go
idle and drives the relay transport's ``go_dormant()`` (D12) — it does NOT itself
suspend the machine. On Fly, the now-traffic-idle machine is suspended by
``autostop:"suspend"`` and woken by autostart-on-wakeUrl (decisions.md Q3=C′).

Design constraints (decisions.md):
  - Per-instance enable is gated SOLELY by the NAS "Labs" toggle, carried to the
    gateway as the ``HERMES_SCALE_TO_ZERO`` env stamp (D11/Q8=A). NOT a user
    config key; ``scale_to_zero.idle_timeout_minutes`` IS config.yaml (D2).
  - Arm only when messaging is relay-only or absent (D1/F6) AND a wakeUrl is
    registered (§3.4(1)) AND the flag is set.
  - Idle = no in-flight agent turn AND no inbound for N min AND no live
    background work (D2/D3/F7).
  - The quiesce uses ``go_dormant()`` (socket closed + supervisor preserved),
    NEVER the stop/restart drain or ``disconnect()`` (F12/F14). The process stays
    alive; Fly freezes+resumes it.
  - ``mark_resume_pending`` is deliberately NOT called here (D13 — suspend
    preserves RAM; revive only if we move to autostop:"stop" or see kills).

The pure helpers (``parse_idle_timeout_seconds``, ``scale_to_zero_enabled``,
``messaging_is_relay_only_or_absent``, ``is_idle``, ``should_arm``) take plain
inputs so they unit-test without a live gateway.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

# Env flag stamped by NAS when the scaleToZero Labs toggle is on (D11/Q8=A),
# mirroring how the `relay` feature stamps GATEWAY_RELAY_URL. Truthy values only.
SCALE_TO_ZERO_ENV = "HERMES_SCALE_TO_ZERO"

# config.yaml default (D2). Behavioural setting -> config, not env.
DEFAULT_IDLE_TIMEOUT_MINUTES = 5

_TRUTHY = {"1", "true", "yes", "on"}


def scale_to_zero_enabled(environ: Optional[dict] = None) -> bool:
    """Whether the per-instance Labs toggle is on (the HERMES_SCALE_TO_ZERO stamp).

    D11/Q8=A: this env flag is the SOLE per-instance enable signal reaching the
    gateway. Absent/blank/falsey -> disabled (fail-safe default off).
    """
    env = environ if environ is not None else os.environ
    return str(env.get(SCALE_TO_ZERO_ENV, "")).strip().lower() in _TRUTHY


def parse_idle_timeout_seconds(
    cfg_value: Any, default_minutes: int = DEFAULT_IDLE_TIMEOUT_MINUTES
) -> float:
    """Coerce ``scale_to_zero.idle_timeout_minutes`` (config.yaml, D2) to seconds.

    Degrades to the default on any non-numeric / non-positive value (never raises,
    never returns <= 0 — a zero/negative timeout would make the gateway go dormant
    instantly, which is never the intent).
    """
    try:
        minutes = float(cfg_value)
    except (TypeError, ValueError):
        minutes = float(default_minutes)
    if minutes <= 0:
        minutes = float(default_minutes)
    return minutes * 60.0


def messaging_is_relay_only_or_absent(platforms: Iterable[Any]) -> bool:
    """True iff the only connected messaging platform is RELAY, or there is none
    (a Chronos-only / no-platform agent) — the F6/D1 structural precondition.

    A directly-connected platform (Discord/Telegram/Slack/...) holds a live
    socket and cannot scale to zero, so its presence disarms the feature. We
    compare by the platform's ``.value``/name to avoid importing the enum here
    (keeps this module import-light and unit-testable).
    """
    names = {_platform_name(p) for p in platforms}
    names.discard("relay")
    return not names


def _platform_name(platform: Any) -> str:
    value = getattr(platform, "value", platform)
    return str(value).strip().lower()


def should_arm(
    *,
    enabled: bool,
    relay_only_or_absent: bool,
    wake_url: Optional[str],
) -> bool:
    """Whether to start the idle watcher at all (D1/D11/§3.4(1)).

    ALL must hold: the Labs flag is on, messaging is relay-only/absent, and a
    wakeUrl is registered (a suspended instance with no reachable wake target is
    a black hole — §3.4(1)). Any unmet -> the watcher never starts (no idle
    timer, no dormancy), so a non-opted instance behaves exactly as today.
    """
    return bool(enabled) and bool(relay_only_or_absent) and bool(wake_url)


def is_idle(
    *,
    running_agent_count: int,
    seconds_since_last_inbound: float,
    idle_timeout_seconds: float,
    has_live_background_work: bool,
) -> bool:
    """The idle predicate (D2/D3/F7). Pure — composes the three conjuncts.

    Idle iff: no in-flight agent turn, no inbound within the timeout window, and
    no live background work (backgrounded delegate_task / kanban / bg terminal).
    Any active work keeps the gateway awake — suspending mid-flight would lose it.
    """
    if running_agent_count > 0:
        return False
    if has_live_background_work:
        return False
    return seconds_since_last_inbound >= idle_timeout_seconds
