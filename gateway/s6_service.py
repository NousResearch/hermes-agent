"""s6 leg of the one-gateway-per-host convergence, shared by container boot and in-process migration.

Inside the official image every profile has an s6 slot (``/run/service/gateway-<profile>``). The
container's boot (``hermes_cli/container_boot.py``) registers every NAMED slot down and lets the
root slot inherit their autostart intent; ``hermes gateway migrate --multiplex`` (and the hook
``hermes update`` runs) must be able to do the same thing to a slot that is UP, from inside the
running container, without a container restart. Both paths decide "which named intents fold into
the root slot" through :func:`fold_named_slot_intent` so they can never disagree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

#: Only this desired state autostarts; everything else (startup_failed, starting, stopped, missing)
#: waits for the operator — no crash-loop of a broken gateway across ``docker restart``.
AUTOSTART_STATES = frozenset({"running"})


@dataclass(frozen=True)
class FoldDecision:
    #: Named profiles whose autostart intent the root slot takes over.
    folded: tuple[str, ...]
    #: Whether the root slot must be up after the fold.
    root_should_start: bool


def fold_named_slot_intent(default_prior_state: Optional[str],
                           named_states: Iterable[tuple[str, Optional[str]]]) -> FoldDecision:
    """The ONE rule for folding per-profile s6 slots into the root slot.

    A named slot is never booted from its own intent (a started named slot IS a second gateway on
    this host); its ``running`` intent moves to the root slot, which is the process that serves it.
    Without the fold an image only ever driven as ``hermes -p coder gateway start`` came up with
    ZERO gateways: no root state, every named slot registered down, every action "registered".
    """
    folded = tuple(sorted(name for name, prior in named_states if prior in AUTOSTART_STATES))
    return FoldDecision(folded=folded,
                        root_should_start=default_prior_state in AUTOSTART_STATES or bool(folded))


def named_slot_name(profile: str) -> str:
    from gateway.s6_manager import S6_SERVICE_PREFIX
    return f"{S6_SERVICE_PREFIX}{profile}"


def running_named_slots(profiles: Sequence[str], manager=None) -> list[str]:
    """Named profiles whose s6 slot is UP right now — the only s6 state that is a second gateway.

    A registered-down slot (what boot leaves behind) is a start target for ``hermes -p X gateway
    start``, not a running gateway, and must never veto the multiplex default.
    """
    manager = manager or _manager()
    up: list[str] = []
    for profile in profiles:
        if profile == "default":
            continue
        try:
            if manager.is_running(named_slot_name(profile)):
                up.append(profile)
        except Exception:  # an unprobeable slot is not a running gateway
            logger.debug("could not probe s6 slot for %s", profile, exc_info=True)
    return up


def slot_is_up(profile: str, manager=None) -> bool:
    return bool(running_named_slots([profile], manager))


def park_named_slot(profile: str, manager=None) -> None:
    """Stop a named profile's slot and write its ``down`` file so a supervisor restart does not
    revive it (``s6-svc -d`` alone is undone by the next ``s6-svscanctl -a``/container restart)."""
    manager = manager or _manager()
    slot = named_slot_name(profile)
    if manager.is_running(slot):
        manager.stop(slot)
        _wait_down(manager.scandir / slot)
    (manager.scandir / slot / "down").touch()


def _wait_down(service_dir: Path, timeout_ms: int = 15000) -> None:
    from gateway.s6_manager import _s6_run
    try:
        _s6_run("s6-svwait", "-d", "-t", str(timeout_ms), str(service_dir), timeout=timeout_ms / 1000 + 5)
    except Exception:
        logger.debug("s6-svwait failed for %s", service_dir, exc_info=True)


def bring_root_slot_up(manager=None) -> str:
    """Start (or restart, so it re-reads its config) the root slot; returns a one-line description."""
    manager = manager or _manager()
    slot = named_slot_name("default")
    (manager.scandir / slot / "down").unlink(missing_ok=True)
    if manager.is_running(slot):
        manager.restart(slot)
        return "restarted the root gateway slot (s6)"
    manager.start(slot)
    return "started the root gateway slot (s6)"


def _manager():
    from gateway.s6_manager import S6ServiceManager
    return S6ServiceManager()
