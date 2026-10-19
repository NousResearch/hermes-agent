"""Live-PID evidence that a gateway serves a profile home when its command line cannot say so.

``status._record_matches_live_gateway_pid`` trusts the live command line first. Argv is silent in
two shapes: the ONE host gateway multiplexing every profile under its own (usually bare) command
line, and an environment or sticky-profile launch whose OS argv carries no profile selector.
Facade names are late-imported, so patches on ``gateway.status`` still intercept them.
"""

from __future__ import annotations

import math
import shlex
from pathlib import Path
from typing import Any


def _host_gateway_serves_home(pid: int, profile_home: Path) -> bool:
    """Does the ONE host gateway — PID ``pid`` — serve ``profile_home``'s profile?

    Argv cannot answer this: the host singleton runs ONE home's (usually bare/default) command line
    while multiplexing every profile, so :func:`_command_line_belongs_to_profile` rejects every
    secondary and the profile reads as "not running" while its messages are being served. The live
    served set is the only proof; the argv rule stays as the fallback when no record exists.
    """
    try:
        from gateway.host_attach import host_gateway, profile_name_for_home

        owner = host_gateway()
    except Exception:
        return False
    return owner is not None and owner.pid == pid and owner.serves(profile_name_for_home(profile_home))


def _bare_argv_record_serves_home(record: dict[str, Any], pid: int, live_cmdline: str, expected_home: Path) -> bool:
    """True when a live command line naming no profile belongs to ``expected_home`` by the record's exact
    process fingerprint (start time + gateway argv) and recorded home. An argv naming any profile or
    HERMES_HOME is never overridden: ``_command_line_belongs_to_profile`` already rejected it."""
    from gateway import status

    try:
        tokens = [token.strip("\"'").lower() for token in shlex.split(live_cmdline, posix=False)]
    except ValueError:
        return False
    if any(token in {"-p", "--profile"} or token.startswith(("-p=", "--profile=", "hermes_home=")) for token in tokens):
        return False
    # Environment and sticky-profile launches can have bare OS argv. Their
    # recorded home is usable only while the exact process fingerprint survives.
    home = record.get("hermes_home")
    started = record.get("start_time")
    if (
        not isinstance(home, str) or not home.strip()
        or not isinstance(started, (int, float)) or isinstance(started, bool)
        or not math.isfinite(started) or started <= 0
        or started != status._get_process_start_time(pid)
        or not status._record_looks_like_gateway(record)
    ):
        return False
    return status._same_hermes_home(home, expected_home)
