"""Which runtime record describes a profile the host multiplexer may serve.

One rule for ``/api/status`` and the messaging card so the two surfaces never pick different records
for the same profile. Read-only: it selects and projects existing ``gateway_state.json`` records and
writes nothing. ``gateway.status`` is resolved at call time so its test seams
(``monkeypatch.setattr(gateway.status, ...)``) still intercept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def local_liveness_pid(liveness: Any) -> Optional[int]:
    """``liveness.pid`` when it was proven on THIS host; None for a health-probe answer, whose PID
    belongs to the remote container and may collide with an unrelated local one."""
    return None if liveness.source == "health" else liveness.pid


def served_profile_runtime(
    home: Path, own_runtime: Optional[dict[str, Any]], *, gateway_pid: Optional[int] = None
) -> Optional[dict[str, Any]]:
    """The multiplexer's record projected onto ``home``'s profile (bare platform keys, the shape a
    standalone gateway writes) when it is that profile's live runtime record, else None.

    - ``own_runtime`` wins whenever it proves a live gateway of ``home`` -- an own ``--force`` one, or
      the host's own record on the host itself, which must stay whole so other profiles'
      ``<profile>:<platform>`` failures remain visible there;
    - a missing/unreadable multiplexer record (``(pid, {})``) is no record;
    - ``gateway_pid`` (a PID the caller proved on this host) must BE the multiplexer, so a gateway of
      the profile's own holding its lock keeps its record.
    """
    from gateway import status

    if own_runtime is not None and status.get_runtime_status_running_pid(own_runtime, expected_home=home) is not None:
        return None
    served = status.multiplexer_liveness_for_profile(home)
    if served is None or not served[1]:
        return None
    mux_pid, record = served
    if gateway_pid is not None and mux_pid != gateway_pid:
        return None
    # Fold on the profile NAME, never ``home.name``: the default root's basename is ``.hermes`` (or
    # any custom HERMES_HOME), so its flat keys never matched (#123088).
    name = status.profile_name_for_home(home) or "default"
    return {**record, "platforms": status.profile_platforms_from_multiplexer(record, name)}
