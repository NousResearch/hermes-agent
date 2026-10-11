"""Cron live-adapter delivery helpers split out of ``cron.scheduler_delivery``.

Import names from this module directly. The facade's helpers are reached late-bound (imported
inside the calling function) so there is no module-level facade <-> sibling cycle.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("cron.scheduler")


def _observe_late_live_send(future: Any, job_id: str, where: str) -> None:
    from cron.scheduler_delivery import _confirm_adapter_delivery

    try:
        result = future.result()
    except Exception as exc:  # any adapter error surfaces here; logged with its traceback
        logger.warning(
            "Job '%s': live adapter send to %s failed after confirmation timeout: %r",
            job_id, where, exc, exc_info=True)
        return
    if not _confirm_adapter_delivery(result, job_id):
        logger.warning(
            "Job '%s': live adapter send to %s returned an unconfirmed result "
            "after confirmation timeout", job_id, where)
