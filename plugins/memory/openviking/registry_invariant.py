#!/usr/bin/env python3
"""
registry_invariant — schema-level rule that COMMITTED implies verified_ov_dir=1.

Lives at: apps/hermes-agent/plugins/memory/openviking/registry_invariant.py

This is the Phase 1 deliverable from the 2026-06-01 OV orphan 3xstanbrain audit
(/home/openclaw/audit-ov-orphan-2026-06-01-1505/CONSOLIDATED-PLAN.md).

The original design (registry_schema_invariant.py in the audit folder) is
preserved verbatim — same triggers, same mark_verified() semantics, same
verify_ingestion() cross-reference. The only adaptation is import path
(plugin location) and removal of the dev-mode fallback so we fail loud if
the plugin package is not importable.

Public surface:
    install_invariant()          — call from registry.ensure_schema() (1-line patch)
    mark_verified(session_id)   — set verified_ov_dir based on live OV dir check
    verify_ingestion()           — cross-reference: count of guaranteed empty shells
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

# Reuse the same lock + DB path that registry.py exposes.
try:
    from plugins.memory.openviking.registry import (
        _lock,
        _SESSION_DB_PATH,
        ensure_schema,
        _now,
    )
except Exception as exc:  # pragma: no cover — fail loud, not silent
    raise SystemExit(
        f"registry_invariant: cannot import registry internals ({exc!r}). "
        "This module must be deployed alongside the plugin, not standalone."
    ) from exc


# Defence in depth: wrap the registry Lock as an RLock so that nested
# acquisition (e.g. mark_verified -> install_invariant from the same
# thread) cannot deadlock. The fix for the 2026-06-05 verifier-deadlock
# class. Registry.py itself can be migrated independently later.
class _RLockWrapper:
    """Wraps a Lock() in RLock-like semantics using a re-entrant check.

    The registry's _lock is a threading.Lock(). mark_verified() calls
    install_invariant() which also takes _lock — re-entering a Lock()
    from the same thread deadlocks. We detect re-entry by thread identity
    and let the same thread pass; foreign threads block as before.
    """

    def __init__(self, inner):
        import threading as _t
        self._inner = inner
        self._local = _t.local()

    def __enter__(self):
        if getattr(self._local, "depth", 0) > 0:
            self._local.depth += 1
            return self
        self._inner.acquire()
        self._local.depth = 1
        return self

    def __exit__(self, *exc):
        self._local.depth -= 1
        if self._local.depth == 0:
            self._inner.release()


_lock = _RLockWrapper(_lock)  # noqa: F811 — replace the imported Lock() with our wrapper


# OV session directory — the canonical "the session is in OV" ground truth.
# Same path the rest of the OV plugin uses.
OV_SESSION_DIR = Path(
    os.environ.get(
        "OPENVIKING_SESSION_DIR",
        os.path.expanduser("~/data/viking/default/session"),
    )
)


# A coarse thread lock for the schema-installation phase. registry._lock
# protects per-row writes; this protects the one-shot CREATE TRIGGER calls
# (which are themselves idempotent, but we serialize to keep the log clean).
_invariant_lock = threading.Lock()


def install_invariant() -> None:
    """Add the verified_ov_dir column and the state-update triggers.

    Idempotent — safe to call on every registry startup. Each ALTER and each
    CREATE TRIGGER IF NOT EXISTS is a no-op on the second call.
    """
    ensure_schema()
    with _invariant_lock:
        with _lock:
            with sqlite3.connect(_SESSION_DB_PATH, timeout=10.0) as conn:
                # Add columns (idempotent — sqlite ALTER has no IF NOT EXISTS,
                # so we swallow the "duplicate column" error and move on).
                for col, typedef in (
                    ("verified_ov_dir", "INTEGER NOT NULL DEFAULT 0"),
                    ("verified_at", "TEXT"),
                ):
                    try:
                        conn.execute(
                            f"ALTER TABLE sessions ADD COLUMN {col} {typedef}"
                        )
                    except Exception:
                        # Column already exists, or another error. The
                        # subsequent DDL will tell us if this is a real problem.
                        pass

                # Primary trigger: COMMITTED requires verified_ov_dir=1.
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS sessions_committed_requires_verified
                    BEFORE UPDATE OF state ON sessions
                    FOR EACH ROW
                    WHEN NEW.state = 'COMMITTED' AND NEW.verified_ov_dir = 0
                    BEGIN
                        SELECT RAISE(ABORT, 'invariant: state=COMMITTED requires verified_ov_dir=1; call mark_verified() first');
                    END;
                    """
                )
                # Companion trigger: any non-COMMITTED state clears the flag.
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS sessions_non_committed_resets_verified
                    BEFORE UPDATE OF state ON sessions
                    FOR EACH ROW
                    WHEN NEW.state != 'COMMITTED' AND OLD.verified_ov_dir = 1
                    BEGIN
                        UPDATE sessions SET verified_ov_dir = 0 WHERE session_id = OLD.session_id;
                    END;
                    """
                )
                conn.commit()


def mark_verified(session_id: str) -> Dict[str, Any]:
    """Mark a session verified in OV (filesystem dir present).

    Returns a small dict describing the result. The caller is expected to
    use this BEFORE the next update_state(state='COMMITTED', ...), in the
    same transaction if possible, to satisfy the schema invariant.
    """
    install_invariant()
    has_dir = (OV_SESSION_DIR / session_id).is_dir()
    now = _now()
    with _lock:
        with sqlite3.connect(_SESSION_DB_PATH, timeout=10.0) as conn:
            conn.execute(
                """
                UPDATE sessions
                SET verified_ov_dir = ?, verified_at = ?
                WHERE session_id = ?
                """,
                (1 if has_dir else 0, now, session_id),
            )
            conn.commit()
    return {
        "session_id": session_id,
        "verified_ov_dir": int(has_dir),
        "verified_at": now,
    }


def verify_ingestion() -> Dict[str, Any]:
    """Cross-reference the registry against the schema invariant.

    Returns:
        total_sessions, committed_total, committed_verified_ov_dir,
        guaranteed_empty_shells, guaranteed_empty_shell_ids (first 100).
    """
    install_invariant()
    with _lock:
        with sqlite3.connect(_SESSION_DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            committed = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE state='COMMITTED'"
            ).fetchone()[0]
            verified = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE state='COMMITTED' AND verified_ov_dir=1"
            ).fetchone()[0]
            shells: List[Dict[str, Any]] = [
                dict(r) for r in conn.execute(
                    "SELECT session_id, verified_at FROM sessions "
                    "WHERE state='COMMITTED' AND verified_ov_dir=0 LIMIT 100"
                ).fetchall()
            ]
    return {
        "ts": time.time(),
        "total_sessions": total,
        "committed_total": committed,
        "committed_verified_ov_dir": verified,
        "guaranteed_empty_shells": committed - verified,
        "guaranteed_empty_shell_ids": [s["session_id"] for s in shells],
    }


if __name__ == "__main__":  # pragma: no cover — CLI for ops
    import json as _json
    r = verify_ingestion()
    print(_json.dumps(r, indent=2))
    sys.exit(0 if r["guaranteed_empty_shells"] == 0 else 1)
