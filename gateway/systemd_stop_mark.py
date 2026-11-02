"""Mark a supervisor-ordered stop as planned so SIGTERM exits cleanly.

Runs as systemd ``ExecStop=`` before ``KillSignal=SIGTERM``. A raw
``systemctl restart/stop hermes-gateway`` bypasses the ``hermes gateway``
CLI, so no takeover (``--replace``) or planned-stop marker exists when the
SIGTERM lands; the gateway classifies it as an unexpected kill, exits
non-zero, and the journal records ``Failed with result exit-code`` even
though the restart was intentional.

The unit's ``Environment=HERMES_HOME=`` is inherited here, so the shared
:func:`gateway.status.write_planned_stop_marker` helper lands the marker in
the right profile home with PID + start-time identity. Best-effort and
never blocking: any failure returns 0 so the stop/restart proceeds.
"""

from __future__ import annotations

import os
import sys


def _resolve_target_pid(argv: list[str] | tuple[str, ...]) -> int | None:
    """Explicit ``argv[1]``, else systemd's ``$MAINPID``. None when unknown.

    No PID-file fallback on purpose: marking the recorded PID without proof
    it is the process systemd is about to signal could mislabel an unrelated
    shutdown as planned.
    """
    raw: object = None
    if len(argv) > 1 and argv[1]:
        raw = argv[1]
    else:
        raw = os.environ.get("MAINPID")
    try:
        pid = int(str(raw).strip())
    except (TypeError, ValueError, AttributeError):
        return None
    return pid if pid > 0 else None


def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    """Write the planned-stop marker for the supervised PID. Always 0."""
    try:
        pid = _resolve_target_pid(list(argv) if argv is not None else sys.argv)
        if pid is None:
            return 0
        from gateway.status import write_planned_stop_marker

        write_planned_stop_marker(pid)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
