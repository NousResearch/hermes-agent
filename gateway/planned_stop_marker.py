"""Write the PID-bound planned-stop marker used by systemd ``ExecStop=``.

This deliberately does not stop or signal anything.  systemd runs it before
applying the unit's existing ``KillSignal=SIGTERM`` directive.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Record one valid target PID, or do nothing when systemd has no MAINPID."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        # ExecStop can run after the main process has already exited.  Do not
        # turn that pre-existing result into a planned stop.
        return 0
    if len(args) != 1:
        return 1
    try:
        target_pid = int(args[0])
    except (TypeError, ValueError):
        return 1
    if target_pid <= 0:
        return 1
    try:
        # Keep this import after validation: an unset MAINPID must be entirely
        # inert, including when the status subsystem is unavailable.
        from gateway.status import write_planned_stop_marker

        return 0 if write_planned_stop_marker(target_pid) else 1
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
