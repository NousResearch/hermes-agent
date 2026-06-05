"""Registry hook that lets the existing Hermes cron tick drive the
workflow-YAML scheduler.

Wiring:
- Companion may expose workflows dir to Hermes at a known location
  (env var for companion workflows dir or ``$HERMES_HOME/connected/workflows``
  fallback). Support for the companion env is patched by the adapter.
- The Hermes gateway's per-tick scheduler imports
  ``run_workflow_tick`` and calls it once per tick (no separate process,
  no separate cron line — single scheduler in Hermes).

The fire path is intentionally indirect for the 2-install scale:
appending to ``workflow_engine_runs.jsonl`` is the contract; the
consumer tails that file (or polls it) to execute the actual rule body.
No new IPC surface, no telemetry, no flags.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_workflows_dir() -> Optional[Path]:
    """Locate the (possibly companion-provided) workflows directory.
    Companion AGENTE_* resolution is patched in by the adapter at startup.
    """
    try:
        from hermes_constants import get_hermes_home
    except Exception:  # pragma: no cover
        return None
    fallback = get_hermes_home() / "connected" / "workflows"
    return fallback if fallback.is_dir() else None


def _resolve_state_dir() -> Path:
    """State (next_runs cache + runs jsonl) lives next to Hermes home."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "cron" / "workflows"
    except Exception:  # pragma: no cover
        return Path.home() / ".hermes" / "cron" / "workflows"


def run_workflow_tick(now: Optional[datetime] = None) -> int:
    """Run one workflow-scheduler tick. Returns the count of fires.

    Safe to call every gateway tick (idempotent — dedupes via the
    persisted next_run state file).  Returns 0 silently when there is no
    exposed workflows directory yet.
    """
    yaml_dir = _resolve_workflows_dir()
    if yaml_dir is None:
        return 0
    from cron.workflow_scheduler import tick

    fired = tick(yaml_dir=yaml_dir, state_dir=_resolve_state_dir(), now=now)
    if fired:
        logger.info(
            "workflow scheduler: fired %d workflow(s) this tick: %s",
            len(fired),
            ",".join(r["workflow_id"] for r in fired),
        )
    return len(fired)
