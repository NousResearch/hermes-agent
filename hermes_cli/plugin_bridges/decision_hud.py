"""Bridge into the standalone decision-hud plugin's db.py — the Rule-1
dispatch-side batch-approval gate (F1). Minimal cut: only
check_batch_approval(). See decision-hub-first-work/plans/02-... for the
follow-up functions (push_task_missing_constraint, check_constraint_resolved,
push_problem_report) intentionally deferred to their own PRs.
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
