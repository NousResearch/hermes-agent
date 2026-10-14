"""Kanban block routing: the configurable unblock-loop limit (``kanban.block_recurrence_limit``)."""
from __future__ import annotations

# Same-reason block -> unblock -> re-block cycles before routing to ``triage``.
# Counts unblock recurrences, NOT dispatcher failures (``DEFAULT_FAILURE_LIMIT``).
BLOCK_RECURRENCE_LIMIT = 2


def block_recurrence_limit() -> int:
    """``kanban.block_recurrence_limit`` from config, else ``BLOCK_RECURRENCE_LIMIT``.

    ``0`` (or negative) disables the loop breaker: repeated same-cause blocks
    stay in ``blocked`` instead of routing to ``triage``. For boards where
    agents resolve every block by comment and no human works triage.
    """
    from hermes_cli.config import load_config

    raw = (load_config().get("kanban") or {}).get("block_recurrence_limit", BLOCK_RECURRENCE_LIMIT)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return BLOCK_RECURRENCE_LIMIT
