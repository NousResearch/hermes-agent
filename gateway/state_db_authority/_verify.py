from __future__ import annotations

import sqlite3
import stat
from pathlib import Path

import hermes_state

from ._model import (
    IntegrityVerdict,
    StateDBIntegrityReport,
    canonical_state_db_path,
    identity_from_stat,
    is_repairable,
    problem_verdict,
    same_identity,
    sqlite_read_only_uri,
    stat_identity,
)


def verify_state_db_integrity(path: Path | str) -> StateDBIntegrityReport:
    """Run the canonical Hermes health contract against one exact file."""
    resolved = canonical_state_db_path(path)
    try:
        before = resolved.stat()
    except FileNotFoundError:
        return StateDBIntegrityReport(
            path=resolved,
            verdict=IntegrityVerdict.ABSENT,
            checked="absent",
        )
    except OSError as exc:
        return StateDBIntegrityReport(
            path=resolved,
            verdict=IntegrityVerdict.ENVIRONMENT_ERROR,
            checked="stat",
            problems=(f"stat failed: {exc}",),
        )

    identity = identity_from_stat(before)
    if not stat.S_ISREG(before.st_mode):
        return StateDBIntegrityReport(
            path=resolved,
            verdict=IntegrityVerdict.UNSUPPORTED_OBJECT,
            checked="stat",
            problems=("state.db is not a regular file",),
            identity=identity,
        )
    if before.st_size == 0:
        return StateDBIntegrityReport(
            path=resolved,
            verdict=IntegrityVerdict.EMPTY,
            checked="empty",
            identity=identity,
        )

    try:
        from hermes_cli.sqlite_safe_read import has_live_connection

        if has_live_connection(resolved):
            return StateDBIntegrityReport(
                path=resolved,
                verdict=IntegrityVerdict.BUSY,
                checked="live_connection",
                problems=(
                    "a state.db connection is already live in this process; "
                    "an independent probe could cancel its POSIX locks",
                ),
                identity=identity,
            )
    except ImportError:
        pass
    except Exception as exc:
        return StateDBIntegrityReport(
            path=resolved,
            verdict=IntegrityVerdict.ENVIRONMENT_ERROR,
            checked="live_connection",
            problems=(f"live-connection probe failed: {exc}",),
            identity=identity,
        )

    conn = None
    try:
        conn = sqlite3.connect(
            sqlite_read_only_uri(resolved),
            uri=True,
            timeout=5.0,
            isolation_level=None,
        )
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA schema_version").fetchone()
    except sqlite3.Error as exc:
        return StateDBIntegrityReport(
            path=resolved,
            verdict=problem_verdict(str(exc)),
            checked="read_only_open",
            problems=(str(exc),),
            identity=identity,
        )
    except Exception as exc:
        return StateDBIntegrityReport(
            path=resolved,
            verdict=IntegrityVerdict.ENVIRONMENT_ERROR,
            checked="read_only_open",
            problems=(str(exc),),
            identity=identity,
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    try:
        problem = hermes_state._db_opens_cleanly(resolved)
    except Exception as exc:
        problem = str(exc)
    if problem:
        return StateDBIntegrityReport(
            path=resolved,
            verdict=problem_verdict(problem),
            checked="canonical_full",
            problems=(str(problem),),
            identity=identity,
        )

    try:
        after_identity = stat_identity(resolved)
    except OSError as exc:
        return StateDBIntegrityReport(
            path=resolved,
            verdict=IntegrityVerdict.ENVIRONMENT_ERROR,
            checked="post_stat",
            problems=(f"post-verification stat failed: {exc}",),
            identity=identity,
        )
    if not same_identity(identity, after_identity):
        return StateDBIntegrityReport(
            path=resolved,
            verdict=IntegrityVerdict.BUSY,
            checked="generation_changed",
            problems=("state.db was replaced during verification",),
            identity=after_identity,
        )
    return StateDBIntegrityReport(
        path=resolved,
        verdict=IntegrityVerdict.VERIFIED,
        checked="canonical_full",
        identity=after_identity,
        may_open_writer=True,
    )


def repair_and_reverify(
    path: Path,
    report: StateDBIntegrityReport,
) -> StateDBIntegrityReport:
    if report.verdict is not IntegrityVerdict.CORRUPT or not is_repairable(report):
        return report
    result = hermes_state.repair_state_db_schema(path)
    if result.get("repaired"):
        return verify_state_db_integrity(path)
    detail = result.get("error") or "canonical repair did not establish health"
    return StateDBIntegrityReport(
        path=path,
        verdict=IntegrityVerdict.CORRUPT,
        checked="canonical_repair",
        problems=(*report.problems, str(detail)),
        identity=(stat_identity(path) if path.exists() else report.identity),
    )
