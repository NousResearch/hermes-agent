"""The marker a backend migration leaves in the home, read by ``hermes update``.

``hermes gateway migrate --backends --apply`` writes ``backend_migration.json`` in the home before its first
change and keeps it (contract: ``state`` is ``running`` while it works, then ``done``, ``rolled_back``,
``failed`` or ``restored``). After a migration the old backend's unit is stopped and disabled, but
``hermes update``'s managed-dashboard step (``main_dashboard._restart_managed_dashboard_service``) starts
an inactive ``hermes-dashboard.service`` whose unit file reads as ``static`` (no ``[Install]`` section,
so ``systemctl disable`` leaves it ``static``) or ``generated``/``linked``: it would bring the old backend
back beside the gateway. While a migration retired it (or is still running, or failed without a complete
rollback) the update restarts that unit only if it is already running.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

MANIFEST_NAME = "backend_migration.json"
#: States in which the old backend is meant to stay down (``rolled_back`` and ``restored`` put it back).
_RETIRED_STATES = frozenset({"running", "done", "failed"})


def backend_retired_by_migration(home: Optional[Path] = None) -> bool:
    """Whether this home's backend migration manifest says the old backend must stay down. No manifest
    is False: the update then behaves exactly as without a migration. A manifest that exists but cannot
    be read is True: a migration may have stopped half-way, so nothing starts the old backend on a guess."""
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    path = Path(home) / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return path.exists()
    if not isinstance(manifest, dict) or not isinstance(manifest.get("state"), str):
        return True
    return manifest["state"] in _RETIRED_STATES
