"""Failure-path integration for the loop-diagnostics subsystem.

This module is the glue between the graph recorder
(``loop_diagnostics_recorder.py``) + the diagnosis engine
(``loop_diagnostics_engine.py``) and the Kanban worker failure lifecycle.

On a *terminal attempt failure* (worker ``crashed`` / ``timed_out`` /
``spawn_failed`` / blocked with an open run, ``gave_up``) the dispatcher-side
integration:

1. finalizes the recorder trace (the recorder is process-local to the worker,
   so this is a no-op unless the trace file already exists on disk);
2. runs the deterministic diagnosis engine on the run's trace;
3. persists the ``DiagnosisResult`` to ``<run_id>.diagnosis.json`` in the
   same directory as the trace (contract §10);
4. appends a ``diagnosis`` task event carrying a concise machine-readable
   AND human-readable report (root-cause candidate, propagation path,
   evidence, suggested intervention);
5. records structured diagnostics metrics (outcome + latency) in a small
   redacted JSONL sink so operators can see diagnosis health.

Guarantees (contract §2 non-goals + §13 graceful behavior):

* **Never masks the original failure.** Every entry point is wrapped;
  a diagnosis failure (missing trace, engine hiccup, write error) is
  swallowed, debug-logged, and a ``diagnosis_failed`` / ``diagnosis_skipped``
  event is emitted so operators can see the fallback. The caller's error
  handling is untouched.
* **No dispatcher semantics change.** Retry / circuit-breaker / failure
  accounting are not influenced. Diagnostics are advisory; the only new
  side effect is the event + diagnosis file + metrics.
* **Zero behavior when disabled.** ``kanban.loop_diagnostics.enabled`` is
  checked via the same loader the recorder uses. When False, nothing is
  imported beyond a config read and no event is emitted. The
  ``kanban.loop_diagnostics.diagnose_on_failure`` flag (default True)
  independently gates the failure-time diagnosis so operators can keep
  recording traces while disabling the diagnostic step.
* **Deterministic + bounded.** Diagnosis is the engine's own guarantee; the
  integration adds only a file write + one event insert, both best-effort.

The module imports NO third-party dependencies (stdlib only), matching the
engine. It deliberately does not import ``kanban_db`` at module load to avoid
import cycles — the caller passes the connection + resolved paths in.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_ENABLED = False


def _load_enabled() -> Dict[str, Any]:
    """Read ``kanban.loop_diagnostics.enabled`` with the recorder's loader.

    Returns the full config dict so callers can also inspect
    ``diagnose_on_failure``.
    """
    try:
        from hermes_cli.observability.loop_diagnostics_recorder import (
            load_recorder_config,
        )

        return load_recorder_config()
    except Exception as exc:
        logger.debug("loop-diagnostics: config unavailable (%s)", exc)
        return {}


def _diagnosis_dir() -> Path:
    """Return the diagnosis sink directory (redacted metrics JSONL)."""
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "governance" / "telemetry"


def _metrics_path() -> Path:
    return _diagnosis_dir() / "loop-diagnostics.jsonl"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def attach_failure_diagnosis(
    conn: Any,
    task_id: str,
    *,
    run_id: Optional[int] = None,
    outcome: Optional[str] = None,
    error: Optional[str] = None,
    failed_action_id: Optional[str] = None,
    board: Optional[str] = None,
    trace_path: Optional[Path] = None,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run diagnosis for a failed worker run and attach the report.

    Designed to be called from the dispatcher-side failure paths
    (``detect_crashed_workers`` / timeout reclaim / spawn-failure /
    ``_record_task_failure``) *after* the run has been closed. The caller is
    responsible for having the run id available.

    Args:
        conn: kanban DB connection (used only to emit the event; may be
            a closed/None connection when the caller already emitted).
        task_id: the kanban task id.
        run_id: the attempt id (``task_runs.id``). When None, diagnosis is
            skipped (no trace can be located) — the fallback event is emitted
            if the task exists.
        outcome: the run outcome (crashed / timed_out / spawn_failed /
            blocked / gave_up) — recorded in metrics + fallback messaging.
        error: the original error string (never masked; carried in the
            event payload + metrics).
        failed_action_id: optional explicit failure site for the engine.
        board: board slug. Defaults to the current board resolution.
        trace_path: optional explicit trace file path (tests use this).
        force: when True, run diagnosis even if the config flag is disabled
            (used by tests and explicit operator tooling).

    Returns:
        The ``DiagnosisResult`` dict when diagnosis ran (even if the result
        is ``unknown`` / ``malformed_trace``), or None when skipped (disabled,
        missing identity, or the run id is absent).

    Never raises. All failures degrade to a ``diagnosis_failed`` /
    ``diagnosis_skipped`` event + metrics row.
    """
    if not force:
        cfg = _load_enabled()
        if not cfg.get("enabled"):
            logger.debug(
                "loop-diagnostics: disabled; skipping diagnosis for %s run %s",
                task_id, run_id,
            )
            return None
        if not cfg.get("diagnose_on_failure", True):
            logger.debug(
                "loop-diagnostics: diagnose_on_failure=false; skipping for %s run %s",
                task_id, run_id,
            )
            return None

    # Missing identity — no trace can be located.
    if not task_id or run_id is None:
        _emit_skipped(
            conn, task_id,
            reason="missing run identity",
            outcome=outcome,
            error=error,
        )
        return None

    t0 = time.monotonic()

    # 1. Finalize — the recorder trace is process-local; there is nothing to
    #    close from here. The trace file (if any) is already on disk.

    # 2. Run the engine.
    try:
        from hermes_cli.observability.loop_diagnostics_engine import (
            diagnose,
            trace_path_for,
        )

        path = trace_path if trace_path is not None else trace_path_for(
            task_id, run_id, board
        )
        result = diagnose(
            task_id,
            run_id,
            failed_action_id=failed_action_id,
            board=board,
            trace_path=path,
        )
    except Exception as exc:
        # Diagnosis must never mask the worker error.
        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.debug(
            "loop-diagnostics: diagnosis failed for %s run %s (%s)",
            task_id, run_id, exc,
        )
        _record_metric(
            task_id=task_id,
            run_id=run_id,
            outcome=outcome,
            status="diagnosis_error",
            category="",
            latency_ms=latency_ms,
            error=error,
        )
        _emit_failure(
            conn, task_id, run_id,
            reason=f"diagnosis engine error: {exc}",
            outcome=outcome,
            error=error,
        )
        return None

    # 3. Persist the diagnosis next to the trace (contract §10).
    try:
        diag_path = _persist_diagnosis(path, result)
    except Exception as exc:
        # Persisting is best-effort; the event still carries the report.
        diag_path = None
        logger.debug(
            "loop-diagnostics: diagnosis persist failed for %s run %s (%s)",
            task_id, run_id, exc,
        )

    # 4. Emit the event (machine + human readable).
    latency_ms = int((time.monotonic() - t0) * 1000)
    try:
        _emit_diagnosis_event(
            conn, task_id, run_id,
            result=result,
            outcome=outcome,
            error=error,
            diag_path=diag_path,
        )
    except Exception as exc:
        logger.debug(
            "loop-diagnostics: event emit failed for %s run %s (%s)",
            task_id, run_id, exc,
        )

    # 5. Metrics.
    _record_metric(
        task_id=task_id,
        run_id=run_id,
        outcome=outcome,
        status=result.get("status", "unknown"),
        category=result.get("category", ""),
        latency_ms=latency_ms,
        error=error,
    )

    logger.info(
        "loop-diagnostics: %s run %s diagnosis=%s category=%s latency=%dms",
        task_id, run_id, result.get("status"), result.get("category"), latency_ms,
    )
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist_diagnosis(
    trace_path: Path,
    result: Dict[str, Any],
) -> Path:
    """Write the DiagnosisResult to ``<run_id>.diagnosis.json``.

    Writes next to the *resolved trace file* so the diagnosis always lives in
    the same directory as the trace (contract §10) — including when the
    caller passed an explicit trace path. Returns the written path.

    The recorder's ``TraceWriter.prune`` already unlinks ``*.diagnosis.json``
    when pruning a run directory, so retention stays consistent.
    """
    diag_path = trace_path.with_suffix(".diagnosis.json")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    # Compact but not minified — keeps it greppable.
    diag_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return diag_path


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


def _emit_event(conn: Any, task_id: str, kind: str, payload: dict, run_id: Optional[int] = None) -> None:
    """Best-effort event append. Never raises.

    ``conn`` may be None (caller already owns the txn and emitted the event
    themselves, or tests pass a stub). When a connection is provided, the
    event is appended inside the caller's open transaction — matching the
    pattern of ``_append_event`` in ``kanban_db``.
    """
    if conn is None:
        return
    try:
        from hermes_cli.kanban_db import _append_event

        _append_event(conn, task_id, kind, payload, run_id=run_id)
    except Exception as exc:
        logger.debug("loop-diagnostics: event append failed (%s)", exc)


def _human_readable(result: Dict[str, Any]) -> str:
    """One-line human-readable summary of a DiagnosisResult."""
    status = result.get("status", "unknown")
    category = result.get("category", "unknown")
    roots = result.get("root_cause_action_ids") or []
    path = result.get("propagation_path") or []
    interventions = result.get("interventions") or []
    expl = (result.get("explanation") or "").strip()

    parts = [f"diagnosis={status} category={category}"]
    if roots:
        parts.append(f"root={','.join(str(r) for r in roots[:3])}")
    if path:
        parts.append(f"path={','.join(str(p) for p in path[:6])}")
    if interventions:
        kinds = ",".join(
            str(i.get("kind", "")) for i in interventions[:3] if isinstance(i, dict)
        )
        if kinds:
            parts.append(f"intervention={kinds}")
    if expl:
        # One line, bounded.
        expl_one = " ".join(expl.split())[:300]
        parts.append(f"detail={expl_one}")
    return " · ".join(parts)


def _emit_diagnosis_event(
    conn: Any,
    task_id: str,
    run_id: int,
    *,
    result: Dict[str, Any],
    outcome: Optional[str],
    error: Optional[str],
    diag_path: Optional[Path],
) -> None:
    """Append a ``diagnosis`` event with machine + human readable report."""
    roots = result.get("root_cause_action_ids") or []
    path = result.get("propagation_path") or []
    interventions = result.get("interventions") or []
    payload = {
        "run_id": run_id,
        "outcome": outcome,
        "status": result.get("status", "unknown"),
        "category": result.get("category", "unknown"),
        "confidence": result.get("confidence", 0.0),
        "root_cause_action_ids": roots,
        "propagation_path": path,
        "interventions": [
            {
                "kind": i.get("kind"),
                "action_id": i.get("action_id"),
                "rationale": (i.get("rationale") or "")[:300],
                "payload": i.get("payload"),
            }
            for i in interventions[:5]
            if isinstance(i, dict)
        ],
        "evidence": result.get("evidence") or {},
        "explanation": (result.get("explanation") or "")[:1024],
        # Human-readable one-liner for the board UI / notifier.
        "summary": _human_readable(result),
        # The original error is preserved, never masked.
        "original_error": (error or "")[:1024] or None,
        # Trace path for operators.
        "trace_path": str(diag_path) if diag_path else None,
    }
    _emit_event(conn, task_id, "diagnosis", payload, run_id=run_id)


def _emit_skipped(
    conn: Any,
    task_id: str,
    *,
    reason: str,
    outcome: Optional[str],
    error: Optional[str],
) -> None:
    """Emit a ``diagnosis_skipped`` event (no run id / disabled path)."""
    if not task_id:
        return
    payload = {
        "reason": reason[:300],
        "outcome": outcome,
        "original_error": (error or "")[:1024] or None,
    }
    _emit_event(conn, task_id, "diagnosis_skipped", payload, run_id=None)


def _emit_failure(
    conn: Any,
    task_id: str,
    run_id: int,
    *,
    reason: str,
    outcome: Optional[str],
    error: Optional[str],
) -> None:
    """Emit a ``diagnosis_failed`` event when the engine itself errored."""
    payload = {
        "run_id": run_id,
        "reason": reason[:300],
        "outcome": outcome,
        "original_error": (error or "")[:1024] or None,
    }
    _emit_event(conn, task_id, "diagnosis_failed", payload, run_id=run_id)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _record_metric(
    *,
    task_id: str,
    run_id: Optional[int],
    outcome: Optional[str],
    status: str,
    category: str,
    latency_ms: int,
    error: Optional[str],
) -> None:
    """Append one redacted diagnostics metric line. Best-effort, never raises."""
    try:
        p = _metrics_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": int(time.time()),
            "task_id": task_id,
            "run_id": run_id,
            "outcome": (outcome or "")[:32],
            "status": (status or "")[:32],
            "category": (category or "")[:48],
            "latency_ms": int(latency_ms),
            # Error presence only — never the raw text (redaction by construction).
            "error": bool(error),
        }
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as exc:
        logger.debug("loop-diagnostics: metric write failed (%s)", exc)
