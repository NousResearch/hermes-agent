"""Which profiles does the LIVE default multiplexer serve? One answer for every CLI/dashboard surface.

``gateway/run_adapters.py::_record_served_profiles`` writes ``served_profiles`` into the default
home's ``gateway_state.json`` at startup. That record is the truth about the running process; the
default ``config.yaml`` plus ``GATEWAY_MULTIPLEX_PROFILES`` as seen by the *CLI* process is only a
guess (``hermes -p coder ...`` loads coder's ``.env``, so an env-only opt-in on the default profile
is invisible to it, and an allowlist edited after start flips the guess before the restart).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def live_default_gateway_pid() -> Optional[int]:
    """PID of the default profile's gateway when its pid record names a live process, else None."""
    from hermes_constants import get_default_hermes_root
    from gateway.status import _pid_exists, _pid_from_record, _read_pid_record
    rec = _read_pid_record(get_default_hermes_root() / "gateway.pid")
    pid = _pid_from_record(rec) if rec else None
    return pid if pid and _pid_exists(pid) else None


def recorded_served_profiles(default_root: Optional[Path] = None) -> Optional[list[str]]:
    """``served_profiles`` the live default gateway recorded, or None when the key is absent (a record
    from before the multiplexer recorded it, or a stopped/absent gateway). Callers fall back to config
    derivation only on None: an empty list is an authoritative "serves nobody else"."""
    from hermes_constants import get_default_hermes_root
    from gateway.status import read_runtime_status
    if live_default_gateway_pid() is None:
        return None
    runtime = read_runtime_status((default_root or get_default_hermes_root()) / "gateway_state.json")
    served = (runtime or {}).get("served_profiles")
    return [str(p) for p in served] if isinstance(served, list) else None


def multiplexer_served_secondaries() -> list[str]:
    """Named profiles the live default multiplexer serves (excludes ``default``); empty when none."""
    return [p for p in (recorded_served_profiles() or []) if p and p != "default"]
