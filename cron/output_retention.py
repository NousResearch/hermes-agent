"""Run-output retention tiers for ``cron/output/<job>/``.

Every run writes one document. A frequent watchdog is silent almost every time, so a single
cap (``cron.output_retention``, default 50) lets its ``[SILENT]`` heartbeats push the few runs
that reported something (or failed) out of the window. Silent runs are therefore written as
``<timestamp>.silent.md`` and pruned on their own, smaller cap (``cron.output_retention_silent``,
default 20); every other run keeps ``cron.output_retention``. A job's own ``output_retention``
overrides both caps. A non-positive cap disables pruning for that tier; a non-integral cap
(``0.5``) is rejected per job and falls back to the default in config, never truncated.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Optional

SILENT_OUTPUT_SUFFIX = ".silent.md"
SILENT_OUTPUT_DEFAULT_KEEP = 20


def _integral(value: Any) -> int:
    """*value* as an int; booleans and non-integral numbers (``0.5``, ``1.9``, NaN, inf) raise
    ValueError instead of being truncated (``0.5`` would otherwise silently disable pruning)."""
    if isinstance(value, bool):
        raise ValueError(value)
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        if not math.isfinite(value) or value != int(value):
            raise ValueError(value)
    return int(value)


def normalize_output_retention(value: Any) -> Optional[int]:
    """Per-job ``output_retention``: None/'' = inherit the global caps; an int (``<= 0`` = never prune)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return _integral(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"output_retention must be an integer or empty (got {value!r})") from exc


def output_keep(job: Optional[dict[str, Any]], *, silent: bool) -> int:
    """The retention cap for one tier of *job*'s run documents."""
    override = job.get("output_retention") if isinstance(job, dict) else None
    if isinstance(override, int) and not isinstance(override, bool):
        return override
    from cron.jobs import _CRON_OUTPUT_DEFAULT_KEEP, _cron_config_number
    if silent:
        return _cron_config_number("output_retention_silent", SILENT_OUTPUT_DEFAULT_KEEP, _integral)
    return _cron_config_number("output_retention", _CRON_OUTPUT_DEFAULT_KEEP, _integral)


def is_silent_output(name: str) -> bool:
    """True for a run document written for a silent run."""
    return name.endswith(SILENT_OUTPUT_SUFFIX)
