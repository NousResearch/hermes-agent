"""Cadence and guard-file policy for file-owned schedules."""
from typing import Any, Mapping
from datetime import datetime
from cron import jobs as _cron_jobs
_UNGUARDED_FLOOR_MINUTES = 15
_GUARDED_FLOOR_MINUTES = 5
_CRON_INTERVAL_SAMPLES = 200

def _effective_interval_minutes(
    parsed: Mapping[str, Any], *, now: datetime
) -> float | None:
    """The smallest gap between consecutive fires; None for one-shots."""

    kind = parsed.get("kind")
    if kind == "interval":
        minutes = int(parsed.get("minutes") or 0)
        return float(minutes) if minutes > 0 else None
    if kind != "cron":
        return None
    expr = str(parsed.get("expr") or "")
    # parse_schedule already proved croniter importable for cron rows; a
    # missing module here still fails open rather than rejecting the file.
    if not expr or not _cron_jobs._ensure_croniter():
        return None
    try:
        iterator = _cron_jobs.croniter(expr, now)
        previous = iterator.get_next(datetime)
    except Exception:
        return None
    minimum: float | None = None
    for _ in range(_CRON_INTERVAL_SAMPLES):
        try:
            current = iterator.get_next(datetime)
        except Exception:
            # A finite (year-bounded) expression runs out of occurrences;
            # the gaps measured so far still bound its cadence.
            break
        gap = (current - previous).total_seconds() / 60.0
        previous = current
        if minimum is None or gap < minimum:
            minimum = gap
    return minimum

def _guard_rejection(
    prefix: str,
    parsed: Mapping[str, Any],
    declaration: Mapping[str, Any],
    package_scripts: tuple[str, ...] | None,
    *,
    now: datetime,
) -> str | None:
    """The cadence-floor or missing-script warning for a declaration, or None.

    The strings are the spec-locked reconciliation contract
    (docs/specs/guarded-schedules/wording-diffs.md); each names the next
    change the author should make.
    """

    script = declaration.get("script")
    display = str(parsed.get("display") or "")
    interval = (
        None
        if parsed.get("kind") == "once"
        else _effective_interval_minutes(parsed, now=now)
    )
    if interval is not None:
        if script is None and interval < _UNGUARDED_FLOOR_MINUTES:
            return (
                f"{prefix}: {display} is below the 15m floor for agent runs "
                "— not armed. Add script: to watch cheaply (5m floor), or "
                "widen the schedule."
            )
        if script is not None and interval < _GUARDED_FLOOR_MINUTES:
            return (
                f"{prefix}: {display} is below the 5m floor for guarded "
                "ticks — not armed. Widen the schedule."
            )
        if interval < _UNGUARDED_FLOOR_MINUTES and not (
            isinstance(declaration.get("repeat"), int)
            and declaration["repeat"] > 0
        ):
            return (
                f"{prefix}: intervals under 15m require repeat: — watching "
                "is bounded, not permanent. Set repeat, or widen to 15m+. "
                "Not armed."
            )
    return _missing_script_rejection(prefix, script, package_scripts)

def _missing_script_rejection(
    prefix: str,
    script: Any,
    package_scripts: tuple[str, ...] | None,
) -> str | None:
    """The missing-guard-script warning, or None.

    Split from the declaration-shaped checks because script existence is
    the one guard input that can drift while the YAML is untouched — a
    deleted script must unfreeze nothing and arm nothing on the unchanged
    content-hash reconcile path too.
    """

    if script is None or package_scripts is None:
        return None
    filename = str(script).partition("/")[2]
    if filename not in package_scripts:
        return (
            f"{prefix}: script {script} not found in the package — not "
            "armed. Add the file, or remove the script field."
        )
    return None
