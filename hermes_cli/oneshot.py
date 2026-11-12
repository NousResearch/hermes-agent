"""Output contract shared by the top-level ``hermes -z`` one-shot run.

The run itself goes through the gateway (``hermes_cli.gateway_chat_startup.launch_gateway_chat``);
this module keeps the pieces that path and the gateway's CLI turns reuse: the exit-code mapping, the
``--usage-file`` ledger (with its auxiliary-usage delta) and ``--toolsets`` flag normalization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

# Not used here: re-exported only so ``hermes_cli.oneshot.get_fallback_chain`` stays importable and
# patchable as on main (external operator scripts set it). The ``-z`` run reads the fallback chain
# through the gateway from ``hermes_cli.fallback_config``, so patching this name does not change it.
from hermes_cli.fallback_config import get_fallback_chain

# Keys copied from the run result into the ``--usage-file`` report. ``service_tier`` is a
# billing-audit field: the tier REQUESTED via request_overrides.extra_body (None when unset), so
# batch pipelines can verify the tier they pay for went out on the wire. ``partial`` /
# ``interrupted`` / ``turn_exit_reason`` say WHY ``completed`` is false, so a pipeline can tell
# an iteration-budget stop from a Ctrl-C without parsing stderr (#111770).
_USAGE_KEYS = (
    "estimated_cost_usd", "cost_status", "cost_source", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "total_tokens", "api_calls",
    "model", "provider", "session_id", "completed", "partial", "interrupted", "turn_exit_reason",
)

# Counters summed per auxiliary task (vision, compression, title_generation, ...) into the
# ``auxiliary`` block of the report. The main-loop keys above stay main-loop-only (backward
# compatible); ``total_including_auxiliary`` carries the grand total pipelines bill on (#112848).
_AUX_COUNTERS = (
    "api_calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens", "estimated_cost_usd",
)


def _auxiliary_usage(session_db, session_id: Optional[str]) -> dict[str, dict]:
    """Per-task aux usage recorded for *session_id*'s lineage (``{}`` without a store / session)."""
    if session_db is None or not session_id:
        return {}
    try:
        return session_db.auxiliary_usage_by_task(session_id)
    except Exception:
        logging.debug("oneshot: auxiliary usage read failed", exc_info=True)
        return {}


def _attach_auxiliary_usage(result: dict, session_db, before: dict[str, dict],
                            fallback_session_id: Optional[str] = None) -> None:
    """Store this run's auxiliary usage on *result* as the delta against the pre-turn snapshot
    (a resumed session already carries earlier runs' aux rows). Waits (bounded) for the auto-title
    thread first: it bills from a daemon thread and can still be in flight when the turn returns.
    Failed-turn dicts carry no ``session_id``; *fallback_session_id* keeps the delta readable then."""
    from hermes_agent.agent.title_generator import wait_for_title_upgrades

    wait_for_title_upgrades()
    after = _auxiliary_usage(session_db, result.get("session_id") or fallback_session_id)
    by_task: dict[str, dict] = {}
    for task, counters in after.items():
        prior = before.get(task, {})
        delta = {key: (counters.get(key) or 0) - (prior.get(key) or 0) for key in _AUX_COUNTERS}
        if any(delta.values()):
            by_task[task] = delta
    result["auxiliary_usage"] = by_task


def _auxiliary_report(report: dict, by_task: dict[str, dict]) -> None:
    """Add the ``auxiliary`` breakdown and ``total_including_auxiliary`` to the ledger."""
    totals = {key: sum(t.get(key) or 0 for t in by_task.values()) for key in _AUX_COUNTERS}
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    report["auxiliary"] = {**totals, "by_task": by_task}
    main_cost = report.get("estimated_cost_usd")
    report["total_including_auxiliary"] = {
        "estimated_cost_usd": None if main_cost is None else main_cost + totals["estimated_cost_usd"],
        "total_tokens": (report.get("total_tokens") or 0) + totals["total_tokens"],
        "api_calls": (report.get("api_calls") or 0) + totals["api_calls"],
    }

# Exit code for a turn stopped by an interrupt (SIGINT convention, same as ``chat -Q``).
_INTERRUPTED_EXIT_CODE = 130


def _oneshot_exit_code(response: Optional[str], result: dict) -> int:
    """Map a finished ``-z`` turn onto its exit code: ``0`` only when the turn completed;
    ``130`` interrupted; ``2`` failed or stopped partway (``partial``, ``completed: False`` such
    as the iteration budget); ``1`` a completed turn that produced no text at all.

    The outcome is judged from the result, not from whether text was printed: a partial or
    failed turn usually leaves an explanation on stdout, and exiting 0 for it made scripts treat
    a half-done job (or a provider error summary) as success (#111770).
    """
    if result.get("interrupted"):
        return _INTERRUPTED_EXIT_CODE
    if result.get("failed") or result.get("partial") or result.get("completed") is False:
        return 2
    if not (response or "").strip():
        return 1
    return 0


def _normalize_toolsets(toolsets: object = None) -> list[str] | None:
    """Split repeated/comma-separated toolset flags into a clean list (``None`` when empty)."""
    if not toolsets:
        return None
    items = toolsets if isinstance(toolsets, (list, tuple)) else [toolsets]
    parts = [str(item).split(",") if isinstance(item, str) else [str(item)] for item in items]
    return [p.strip() for chunk in parts for p in chunk if p.strip()] or None


def _write_usage_file(path: Optional[str], result: dict, failure: Optional[str] = None) -> None:
    """Best-effort JSON usage report for pipelines (``-z --usage-file``).

    Written even on failure so callers can always account for spend. Never raises — a broken usage
    write must not mask the run's own outcome.
    """
    if not path:
        return
    try:
        report = {key: result.get(key) for key in _USAGE_KEYS}
        report["failed"] = bool(result.get("failed")) or failure is not None
        report["service_tier"] = result.get("service_tier")
        if isinstance(result.get("auxiliary_usage"), dict):
            _auxiliary_report(report, result["auxiliary_usage"])
        if failure is not None:
            report["failure"] = failure
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
