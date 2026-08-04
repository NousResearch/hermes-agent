"""Install identity for gateway monitoring.

The install id is a stable, resettable pseudonymous identifier attached to
exported health signals so an operator can tell instances apart in their
collector. It carries no account identity and can be rotated by clearing
``monitoring.install_id`` in config.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

logger = logging.getLogger(__name__)


def ensure_install_id(config: Dict[str, Any]) -> str:
    """Return a stable install id, minting and persisting one when empty.

    The id must survive gateway restarts (it becomes ``service.instance.id``
    on exported signals), so a freshly minted UUID is written back to
    config.yaml immediately. The write is fail-open: if persisting fails
    (read-only home, managed scope), the ephemeral id is still returned and
    a new one is minted next start.

    Clearing ``monitoring.install_id`` (e.g. ``hermes config set
    monitoring.install_id ""``) rotates the id on the next gateway start.
    """
    mon = config.get("monitoring") if isinstance(config, dict) else None
    existing = (mon or {}).get("install_id") if isinstance(mon, dict) else None
    if isinstance(existing, str) and existing.strip():
        return existing

    minted = str(uuid.uuid4())
    try:
        from hermes_cli.config import load_config, save_config

        fresh = load_config()
        if isinstance(fresh, dict):
            slot = fresh.setdefault("monitoring", {})
            if isinstance(slot, dict) and not str(slot.get("install_id") or "").strip():
                slot["install_id"] = minted
                save_config(fresh)
    except Exception:
        logger.debug("install_id persist failed; using ephemeral id", exc_info=True)
    # Keep the in-memory config consistent for this process either way.
    if isinstance(config, dict):
        config.setdefault("monitoring", {})
        if isinstance(config["monitoring"], dict):
            config["monitoring"]["install_id"] = minted
    return minted


__all__ = [
    "ensure_install_id",
]
