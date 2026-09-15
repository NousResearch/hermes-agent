"""Bridge into the standalone decision-hud plugin's db.py.

Independent gates live here:
  - ``check_batch_approval`` — Rule-1 dispatch-side batch-approval gate (F1).
  - ``push_task_missing_constraint`` / ``check_constraint_resolved`` —
    Rule-4 retry-cap missing-constraint escalation.
  - ``push_problem_report`` — Rule 6 EARS-ification-gate refusal path (files
    a raw problem report instead of inventing a plausible-looking answer).

Minimal cut note: this module intentionally does NOT yet implement every
decision-hud call — see decision-hub-first-work/plans/02-... for follow-ups
deferred to their own PRs/branches.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from typing import Optional

_MODULE_CACHE_KEY = "_hermes_decision_hud_db_bridge"
_MIN_DECISION_HUD_SCHEMA_VERSION = 6


def _decision_hud_db_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "plugins" / "decision-hud" / "db.py"


def _assert_supported_schema(module: object) -> None:
    try:
        database_path = Path(module.db_path())  # type: ignore[attr-defined]
    except Exception as exc:
        raise ImportError(f"decision-hud database path unavailable: {exc}") from exc
    if not database_path.exists() or database_path.stat().st_size == 0:
        return
    conn = sqlite3.connect(str(database_path))
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
        version = int(row[0]) if row else 0
    finally:
        conn.close()
    if version < _MIN_DECISION_HUD_SCHEMA_VERSION:
        raise ImportError(
            f"decision-hud schema too old: {version}; "
            f"bridge requires >= {_MIN_DECISION_HUD_SCHEMA_VERSION}"
        )


def _load_decision_hud_db():
    """Import, cached by (path, mtime) via sys.modules — a real mtime check,
    not just identity: an in-place edit to db.py while the dispatcher process
    is alive invalidates the cache and reloads, so a live-patched decision-hud
    policy fix is picked up on the next call rather than silently ignored
    until process restart."""
    db_path = _decision_hud_db_path().expanduser().resolve()
    if not db_path.is_file():
        raise ImportError(f"decision-hud not installed: {db_path} not found")
    mtime = db_path.stat().st_mtime_ns
    module_cache_key = f"{_MODULE_CACHE_KEY}:{db_path}"
    cached = sys.modules.get(module_cache_key)
    cached_mtime = getattr(cached, "_hermes_bridge_cached_mtime_ns", None)
    if cached is not None and cached_mtime == mtime:
        _assert_supported_schema(cached)
        return cached
    spec = importlib.util.spec_from_file_location(module_cache_key, db_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load decision-hud db module from {db_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_cache_key] = module
    try:
        spec.loader.exec_module(module)
        _assert_supported_schema(module)
    except Exception:
        sys.modules.pop(module_cache_key, None)
        raise
    module._hermes_bridge_cached_mtime_ns = mtime  # type: ignore[attr-defined]
    return module


def check_batch_approval(*, project: str, batch_id: str) -> tuple[bool, str]:
    """(True, "") iff decision-hud has an approve-resolved batch_approval
    decision for (project, batch_id). (False, reason) for
    pending/rejected/missing/any error. Never raises — the dispatch-side
    caller's fail-closed default applies uniformly on any (False, _)."""
    try:
        db = _load_decision_hud_db()
    except Exception as exc:
        return False, f"decision-hud unavailable: {exc}"
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = db.connect()
        db.require_batch_approval(conn, project_id=project, batch_id=batch_id)
        return True, ""
    except Exception as exc:
        # BatchNotApproved is resolved as an attribute on the loaded module,
        # not imported statically — a genuinely broken/partial decision-hud
        # install without that class would otherwise let AttributeError
        # escape from an `except db.BatchNotApproved` clause, breaking this
        # function's own "never raises" contract. getattr with a sentinel
        # base class (never matches, since real exceptions never subclass
        # object() directly) keeps this a plain isinstance check either way.
        not_approved_cls = getattr(db, "BatchNotApproved", ())
        if not_approved_cls and isinstance(exc, not_approved_cls):
            return False, str(exc)
        return False, f"batch approval check errored: {exc}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def push_problem_report(
    *, project: str, problem: str, context: Optional[str] = None, reporter: Optional[str] = None,
) -> tuple[bool, str]:
    """File a raw problem report in decision-hud (Rule 6 EARS-gate refusal
    path). ``(True, report_id)`` on success, ``(False, reason)`` on any
    failure — mirrors :func:`check_batch_approval`'s never-raises contract
    so a decision-hud outage never crashes the caller (the decomposer
    proceeds with ``ears_sentence=None`` regardless; see
    ``kanban_decompose.py``'s EARS-gate docstring for the fail-open
    rationale).
    """
    try:
        db = _load_decision_hud_db()
    except Exception as exc:
        return False, f"decision-hud unavailable: {exc}"
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = db.connect()
        row = db.push_problem_report(conn, project_id=project, problem=problem, context=context, reporter=reporter)
        return True, str(row.get("id", ""))
    except Exception as exc:
        return False, f"problem report push failed: {exc}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def push_task_missing_constraint(
    *, project: str, task_id: str, question: str, urgency: str = "normal",
) -> tuple[bool, str]:
    """Push a Rule-4 ``missing_constraint`` decision-hud card for one task.

    ``(True, "")`` on a successful push. ``(False, reason)`` for every
    failure mode, INCLUDING decision-hud's own duplicate-open-constraint
    ``ValueError`` (a task that already has an unresolved missing_constraint
    row for this ``(project, task_id)`` must not get a second card — the
    caller (the retry-cap dispatch gate) treats that as "already escalated,
    nothing new to do" rather than an error to surface). Never raises —
    matches :func:`check_batch_approval`'s never-raises contract so the
    dispatch-side caller's fail-closed default applies uniformly.
    """
    try:
        db = _load_decision_hud_db()
    except Exception as exc:
        return False, f"decision-hud unavailable: {exc}"
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = db.connect()
        db.push_missing_constraint(
            conn, project_id=project, task_id=task_id, question=question, urgency=urgency,
        )
        return True, ""
    except ValueError as exc:
        # Duplicate open constraint for this (project, task_id) — not an
        # error, just "already escalated, don't push a second card".
        return False, str(exc)
    except Exception as exc:
        return False, f"push_missing_constraint errored: {exc}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def check_constraint_resolved(*, project: str, task_id: str) -> bool:
    """``True`` iff decision-hud has a RESOLVED ``missing_constraint``
    decision for this exact ``(project, task_id)`` (the most recent row —
    see decision-hud's ``require_constraint_resolved`` docstring: an old
    resolved escalation never masks a fresh unresolved one). ``False`` for
    pending/missing/any error — every failure mode (plugin not installed, DB
    locked/corrupt, schema mismatch) folds into ``False`` so the dispatch-
    side caller's fail-closed default applies uniformly, mirroring
    :func:`check_batch_approval`'s never-raises contract.
    """
    try:
        db = _load_decision_hud_db()
    except Exception:
        return False
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = db.connect()
        db.require_constraint_resolved(conn, project_id=project, task_id=task_id)
        return True
    except Exception as exc:
        # ConstraintNotResolved is resolved as an attribute on the loaded
        # module, not imported statically — mirrors check_batch_approval's
        # getattr-with-sentinel pattern so a partial/broken decision-hud
        # install can't leak an AttributeError through this "never raises"
        # boundary.
        not_resolved_cls = getattr(db, "ConstraintNotResolved", ())
        if not_resolved_cls and isinstance(exc, not_resolved_cls):
            return False
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
