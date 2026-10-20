"""Load admission: refuse a cold load that would not fit in the memory free right now.

The router loads an unloaded model for any request that names it (``--models-autoload``), with no
Hermes code in between. Presets are priced against capacity, so memory another program holds (a
model in LM Studio, a game) is invisible at that point. On Windows the load then runs into the
page file and the whole machine stalls for minutes (#142082). This check runs before the request
leaves Hermes and prices the load against the live budget instead.

Fail-open everywhere: a probe or pricing miss admits the load, as before.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# Statuses that already hold (or are taking) their memory; a request to one of them loads nothing.
_HOLDING = ("loaded", "ready", "loading")


def _footprint(entry, capacity) -> int | None:
    """Bytes the preset's launch holds once loaded, or None when it cannot be priced."""
    from hermes_cli.local_runtime.presets import resident_footprint

    model = (entry.keys or {}).get("model")
    if not model or entry.window <= 0:
        return None
    drafting = (entry.keys or {}).get("spec-type") == "draft-dflash"
    return resident_footprint(Path(model), capacity, entry.window, drafting=drafting)


def admission_refusal(model_id: str, statuses: dict[str, str], decisions: dict, models_max: int | None,
                      capacity, live) -> str | None:
    """Why loading ``model_id`` now would over-commit memory, or None when it fits (or can't be told).

    ``statuses`` is the router's ``GET /models`` view, ``decisions`` the preset INI read back,
    ``models_max`` the cap the router runs with: at the cap the router evicts its LRU child before
    the load, so the smallest resident is credited back (the LRU is not knowable from here).
    """
    if statuses.get(model_id) in _HOLDING or model_id not in decisions:
        return None
    need = _footprint(decisions[model_id], capacity)
    if not need:
        return None
    resident = [m for m, status in statuses.items() if status in _HOLDING]
    freed = 0
    if resident and models_max and len(resident) >= models_max:
        prices = [_footprint(decisions[m], capacity) for m in resident if m in decisions]
        if not prices or None in prices:
            return None
        freed = min(prices)
    free = live.usable_vram_bytes + live.ram_available_bytes + freed
    if need <= free:
        return None
    return (f"loading {model_id} needs ~{need / _GIB:.1f} GiB, but only ~{free / _GIB:.1f} GiB of "
            "GPU memory + RAM is free right now. Close programs holding memory (another model "
            "server, a game, a VM) or pick a smaller model.")


def check_load_admission(model_id: str) -> str | None:
    """Live wrapper of :func:`admission_refusal` against the managed router; None on any probe miss."""
    try:
        from hermes_cli.local_runtime.endpoint import managed_get_json, managed_root
        from hermes_cli.local_runtime.hardware import probe_budget
        from hermes_cli.local_runtime.presets import read_preset_decisions
        from hermes_cli.local_runtime.recovery import read_state

        root = managed_root()
        if root is None:
            return None
        listing = managed_get_json(*root, "/models", timeout_s=3)
        statuses = {m["id"]: m.get("status", {}).get("value", "unknown")
                    for m in (listing or {}).get("data", [])}
        if statuses.get(model_id) in _HOLDING:
            return None  # the common case: skip the hardware probes
        models_max = read_state().get("models_max")
        return admission_refusal(model_id, statuses, read_preset_decisions(),
                                 int(models_max) if models_max else None,
                                 probe_budget(planning=True), probe_budget())
    except Exception:
        logger.debug("load admission check skipped", exc_info=True)
        return None
