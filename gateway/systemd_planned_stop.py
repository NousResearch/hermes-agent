"""Mark a systemd-managed gateway stop as intentional.

This module is invoked by the generated unit's ``ExecStop=`` directive before
systemd sends ``SIGTERM`` to the gateway.  The gateway consumes the existing
PID-scoped marker in its signal handler, so direct ``systemctl stop/restart``
uses the same planned-stop path as ``hermes gateway stop``.  Unexpected
``SIGTERM`` signals remain unmarked and keep their non-zero exit behavior.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from gateway.status import write_planned_stop_marker


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        return 2

    try:
        pid = int(args[0])
    except (TypeError, ValueError):
        return 2
    if pid <= 0:
        return 2

    return 0 if write_planned_stop_marker(pid) else 1


if __name__ == "__main__":
    raise SystemExit(main())
