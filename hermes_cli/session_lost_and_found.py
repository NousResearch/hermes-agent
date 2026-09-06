"""Last-resort page-level salvage for an unreadable session database schema, via the sqlite3 shell's
``.recover`` (rows it cannot attribute to a schema land in ``lost_and_found`` tables:
``rootpgno, pgno, nfield, id, c0..cN``)."""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from hermes_cli.session_recovery import (
    _AUXILIARY_TABLE_SCHEMAS, _AUXILIARY_TABLES, _CANONICAL_TABLES, _count_rows, _immediate_transaction,
    _placeholder_titles, _quoted_columns, _table_columns,
)

logger = logging.getLogger(__name__)

# Hermes session ids are timestamps (20260812_135332_ab12cd): the strongest sentinel for schema-less rows.
SESSION_ID_PATTERN = re.compile(r"^\d{8}_\d{6}_")
MESSAGE_ROLES = frozenset({"user", "assistant", "tool", "system"})

# Values observed in sessions.source across gateway platforms and tooling.
KNOWN_SOURCES = frozenset({
    "cli", "telegram", "discord", "slack", "whatsapp", "signal", "matrix",
    "irc", "email", "x", "twitter", "api", "gateway", "web", "dashboard",
    "tool", "subagent", "cron", "recovered", "imported", "acp",
})

# Historical sessions layouts. Columns are only ever appended, so an older record is a strict prefix.
SESSIONS_LAYOUT_NFIELDS = frozenset({55, 54, 52})
SESSIONS_LEGACY_MINIMAL_NFIELD = 14
SESSION_MODEL_USAGE_NFIELD = 18

# Plausible unix-epoch window for started_at heuristics on legacy layouts.
_EPOCH_LOW = 1_000_000_000.0   # 2001
_EPOCH_HIGH = 4_000_000_000.0  # 2096

# Title label/prefix of every session row this lane synthesises (legacy-layout rows and stubbed parents).
# The recovery verifier keys on the prefix to tell synthesised rows from positionally mapped ones.
_STUB_TITLE_LABEL = "best-effort recovered"
STUB_TITLE_PREFIX = f"[{_STUB_TITLE_LABEL}"

SQLITE3_CLI_GUIDANCE = (
    "A last-resort page-level salvage is available when a `.recover`-capable `sqlite3` command-line shell is "
    "installed: its `.recover` command can rebuild rows into lost_and_found tables even when the table schemas are "
    "unreadable (this is a CLI-only feature, not part of Python's sqlite3 module, and some distro builds lack it — "
    "the shell must include the sqlite_dbpage extension, as the official builds from sqlite.org do). Install such a "
    "sqlite3 CLI (e.g. `brew install sqlite` or the precompiled sqlite-tools from sqlite.org) so it is on PATH, then "
    "re-run with --allow-partial."
)

# SQLite's WAL-reset bug (https://sqlite.org/wal.html#walresetbug) lets a
# fresh opener unlink a live WAL/SHM sidecar pair and split the database into
# two concurrent generations whose acknowledged writes can silently vanish.
# It is real in CLI builds up to 3.51.2; fixed in 3.51.3+ with backports
# 3.50.7 and 3.44.6 — the same version gate hermes_state applies to the
# embedded library (#69784). The system `sqlite3` CLI on Debian/Ubuntu is
# routinely in the vulnerable band (e.g. 3.45.1), and #100368's forensics
# caught exactly this shell converting a live Hermes state.db into two
# generations. A salvage shell must therefore be version-gated, not just
# capability-gated, before it is pointed at (a copy of) a Hermes database.
#
# The predicate lives in hermes_cli.sqlite_runtime (stdlib-only, shared with
# the installer/update gates) so the embedded runtime and the salvage shell
# can never disagree about which versions are safe.
from hermes_cli.sqlite_runtime import is_sqlite_wal_reset_vulnerable as _wal_reset_vulnerable  # noqa: E502

_WAL_RESET_VULNERABLE_GUIDANCE = (
    "salvage against a Hermes database with the WAL-reset bug "
    "(https://sqlite.org/wal.html#walresetbug, fixed in 3.51.3+ / backports "
    "3.50.7 / 3.44.6; the vulnerable fresh-opener can unlink a live WAL/SHM "
    "pair and split the database into two generations, losing acknowledged "
    "writes — #100368). Install a fixed sqlite3 CLI (3.51.3+, e.g. `brew "
    "install sqlite` or the precompiled sqlite-tools from sqlite.org)"
)


class LostAndFoundError(RuntimeError):
    """Raised when the CLI .recover pass cannot produce a usable database."""


def _parse_sqlite3_cli_version(binary: str) -> Optional[tuple[int, int, int]]:
    """Version of the sqlite3 CLI at *binary* via ``--version``, or None when it cannot run or be parsed."""
    try:
        probe = subprocess.run([binary, "--version"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if probe.returncode != 0:
        return None
    match = re.search(rb"(\d+)\.(\d+)\.(\d+)", probe.stdout)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


_last_cli_refusal: dict[str, Any] = {}


def find_sqlite3_cli_refusal() -> dict[str, Any]:
    """Why the last :func:`find_sqlite3_cli` call in this process refused: ``{"reason": ...}`` with reason in
    ``missing``, ``no_dbpage`` (shell cannot run ``.recover``), ``wal_reset_vulnerable``; empty if it succeeded."""
    return dict(_last_cli_refusal)


def find_sqlite3_cli() -> Optional[str]:
    """A salvage-safe ``.recover``-capable sqlite3 CLI path, or None.

    PATH presence is not enough, and neither is ``.recover`` support alone: (1) distro builds can lack the
    ``sqlite_dbpage`` virtual table ``.recover`` needs — probed once on a scratch DB; (2) a capable CLI can still
    carry the WAL-reset opener bug (fixed 3.51.3+ / backports 3.50.7 / 3.44.6). The salvage lane only runs it on a
    snapshot copy, but refusing it keeps vulnerable shells out of the documented workflow. Refusals are recorded
    for :func:`find_sqlite3_cli_refusal` so callers can say exactly what to install.
    """
    global _last_cli_refusal
    _last_cli_refusal = {}
    binary = shutil.which("sqlite3")
    if binary is None:
        _last_cli_refusal = {"reason": "missing"}
        return None
    if not _cli_supports_recover(binary):
        _last_cli_refusal = {"reason": "no_dbpage", "binary": binary}
        return None
    version = _parse_sqlite3_cli_version(binary)
    if version is not None and _wal_reset_vulnerable(version):
        version_str = ".".join(str(part) for part in version)
        logger.warning(
            "sqlite3 CLI %s reports version %s, which still carries the "
            "WAL-reset opener bug; refusing to use it for salvage",
            binary,
            version_str,
        )
        _last_cli_refusal = {
            "reason": "wal_reset_vulnerable",
            "binary": binary,
            "version": version_str,
            "detail": f"reports version {version_str}, which has " + _WAL_RESET_VULNERABLE_GUIDANCE,
        }
        return None
    return binary


def _cli_supports_recover(binary: str) -> bool:
    """True when ``binary`` can run ``.recover`` (has sqlite_dbpage)."""
    scratch_dir = tempfile.mkdtemp(prefix="hermes-recover-probe-")
    scratch = Path(scratch_dir) / "probe.db"
    try:
        conn = sqlite3.connect(str(scratch))
        try:
            conn.execute("CREATE TABLE t (x)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
        finally:
            conn.close()
        probe = subprocess.run([binary, "-readonly", str(scratch), ".recover"], capture_output=True, timeout=30)
        return probe.returncode == 0 and b"sqlite_dbpage" not in probe.stderr
    except (OSError, subprocess.SubprocessError, sqlite3.Error):
        return False
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def run_cli_lost_and_found_recover(
    source: Path, lf_path: Path, sqlite3_bin: str, *, timeout: float = 3600.0,
) -> dict[str, Any]:
    """Run ``sqlite3 <source> .recover`` streamed into a fresh scratch DB."""
    attempts: list[dict[str, Any]] = []
    for command in (".recover --ignore-freelist", ".recover"):
        if lf_path.exists():
            lf_path.unlink()
        dump = subprocess.Popen(
            [sqlite3_bin, "-readonly", str(source), command], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        load = subprocess.Popen(
            [sqlite3_bin, str(lf_path)], stdin=dump.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        assert dump.stdout is not None
        dump.stdout.close()  # let dump receive SIGPIPE if load dies
        try:
            _, load_err = load.communicate(timeout=timeout)
            dump_err = dump.stderr.read() if dump.stderr is not None else b""
            dump.wait(timeout=60)
        except subprocess.TimeoutExpired:
            dump.kill()
            load.kill()
            raise LostAndFoundError(f"sqlite3 .recover timed out after {timeout:.0f}s")
        attempts.append({
            "command": command, "dump_returncode": dump.returncode, "load_returncode": load.returncode,
            "dump_stderr_tail": dump_err.decode("utf-8", "replace")[-2000:],
            "load_stderr_tail": load_err.decode("utf-8", "replace")[-2000:],
            "usable": _lost_and_found_db_usable(lf_path),
        })
        if attempts[-1]["usable"]:
            return {"binary": sqlite3_bin, "attempts": attempts}
    details = "; ".join(
        f"[{a['command']}] dump rc={a['dump_returncode']} load rc={a['load_returncode']} "
        f"{a['dump_stderr_tail'] or a['load_stderr_tail']}".strip()
        for a in attempts
    )
    raise LostAndFoundError(f"sqlite3 .recover did not produce a usable lost_and_found database: {details}")


def _lost_and_found_db_usable(lf_path: Path) -> bool:
    if not lf_path.exists() or lf_path.stat().st_size == 0:
        return False
    try:
        conn = sqlite3.connect(str(lf_path))
        try:
            return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone() is not None
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False


def _notnull_defaults(conn: sqlite3.Connection, table: str) -> dict[int, Any]:
    """Column index -> substitute for NOT NULL columns. Salvage can return NULLs where the schema says
    NOT NULL (torn cells, old rows); dropping a row over one damaged counter would defeat the lane, so
    such NULLs get the schema default (or '' / 0 when none is declared)."""
    substitutes: dict[int, Any] = {}
    for index, row in enumerate(conn.execute(f'PRAGMA table_info("{table}")')):
        if not row[3]:  # notnull flag
            continue
        if row[4] is not None:
            substitutes[index] = _parse_sql_default(str(row[4]))
        else:
            declared = str(row[2] or "").upper()
            substitutes[index] = 0 if ("INT" in declared or "REAL" in declared) else ""
    return substitutes


def _parse_sql_default(text: str) -> Any:
    """Coerce a ``PRAGMA table_info`` default literal: quoted string, int, float, or raw."""
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _is_session_id(value: Any) -> bool:
    return isinstance(value, str) and bool(SESSION_ID_PATTERN.match(value))


def _looks_like_source(value: Any) -> bool:
    return bool(value) and isinstance(value, str) and (
        value in KNOWN_SOURCES or bool(re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value))
    )


def classify_lost_and_found_row(nfield: int, cells: tuple[Any, ...]) -> Optional[str]:
    """Classify one lost_and_found record by field count + sentinel values."""
    if len(cells) >= 3 and cells[0] is None:
        # Rowid-alias tables store their INTEGER PRIMARY KEY as NULL; messages is the only canonical
        # table shaped like that with a session id second and a role third.
        is_message = (
            isinstance(cells[1], str) and cells[1] and isinstance(cells[2], str) and cells[2] in MESSAGE_ROLES
        )
        return "messages" if is_message else None
    if not _is_session_id(cells[0] if cells else None):
        return None
    second = cells[1] if len(cells) > 1 else None
    if nfield == SESSION_MODEL_USAGE_NFIELD:  # session id first, model string second
        return "session_model_usage" if isinstance(second, str) and second else None
    # Known sessions layouts, or an unknown historical one (>= 30 fields): session id + source is enough.
    if (
        nfield in SESSIONS_LAYOUT_NFIELDS or nfield == SESSIONS_LEGACY_MINIMAL_NFIELD or nfield >= 30
    ) and _looks_like_source(second):
        return "sessions"
    return None


def _heuristic_started_at(cells: tuple[Any, ...]) -> float:
    for value in cells:
        if isinstance(value, (int, float)) and _EPOCH_LOW <= float(value) <= _EPOCH_HIGH:
            return float(value)
    return 0.0


def _insert_prefix_row(
    dest: sqlite3.Connection, table: str, dest_columns: list[str], values: list[Any],
    notnull_substitutes: Optional[dict[int, Any]] = None,
) -> bool:
    if notnull_substitutes:
        values = [
            notnull_substitutes[index] if value is None and index in notnull_substitutes else value
            for index, value in enumerate(values)
        ]
    quoted, placeholders = _quoted_columns(dest_columns[: len(values)])
    cursor = dest.execute(f'INSERT OR IGNORE INTO "{table}" ({quoted}) VALUES ({placeholders})', values)
    return cursor.rowcount == 1


def _copy_direct_tables(lf_conn: sqlite3.Connection, dest: sqlite3.Connection) -> dict[str, int]:
    """Copy rows .recover managed to attribute to real canonical tables."""
    copied: dict[str, int] = {}
    for table in (*_CANONICAL_TABLES, *_AUXILIARY_TABLES):
        source_columns = _table_columns(lf_conn, table)
        if not source_columns:
            continue
        dest_columns = _table_columns(dest, table)
        if not dest_columns and table in _AUXILIARY_TABLE_SCHEMAS:  # lazily-created gateway table
            _AUXILIARY_TABLE_SCHEMAS[table](dest)
            dest_columns = _table_columns(dest, table)
        columns = [c for c in dest_columns if c in source_columns]
        if not columns:
            continue
        quoted, placeholders = _quoted_columns(columns)
        rows = lf_conn.execute(f'SELECT {quoted} FROM "{table}"').fetchall()
        if not rows:
            copied[table] = 0
            continue
        before = _count_rows(dest, table)
        dest.executemany(f'INSERT OR IGNORE INTO "{table}" ({quoted}) VALUES ({placeholders})', rows)
        copied[table] = _count_rows(dest, table) - before
    return copied


def map_lost_and_found_rows(lf_conn: sqlite3.Connection, dest: sqlite3.Connection) -> dict[str, Any]:
    """Best-effort mapping of a .recover output DB into a fresh SessionDB."""
    report: dict[str, Any] = {
        "direct_table_rows": {}, "mapped": {"sessions": 0, "messages": 0, "session_model_usage": 0},
        "legacy_minimal_sessions": 0, "unmapped_rows": 0, "insert_conflicts": 0, "lost_and_found_tables": [],
    }
    with _immediate_transaction(dest):
        report["direct_table_rows"] = _copy_direct_tables(lf_conn, dest)

        # Per-kind destination columns + NOT NULL substitutes. Identity fields are never fabricated:
        # rows with a NULL session id / role / source were already rejected by classify_lost_and_found_row.
        targets: dict[str, tuple[list[str], dict[int, Any]]] = {}
        for kind_name, protected in (("sessions", (0, 1)), ("messages", (1, 2)), ("session_model_usage", (0, 1))):
            defaults = _notnull_defaults(dest, kind_name)
            for index in protected:
                defaults.pop(index, None)
            targets[kind_name] = (_table_columns(dest, kind_name), defaults)
        lf_tables = [
            str(row[0]) for row in
            lf_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'lost_and_found%'")
        ]
        report["lost_and_found_tables"] = lf_tables
        for lf_table in lf_tables:
            if _table_columns(lf_conn, lf_table)[:3] != ["rootpgno", "pgno", "nfield"]:
                continue
            for row in lf_conn.execute(f'SELECT * FROM "{lf_table}"'):
                try:
                    nfield = int(row[2]) if row[2] is not None else 0
                except (TypeError, ValueError):
                    report["unmapped_rows"] += 1
                    continue
                lf_rowid = row[3]
                cells = tuple(row[4 : 4 + max(nfield, 0)])
                kind = classify_lost_and_found_row(nfield, cells)
                if kind is None:
                    report["unmapped_rows"] += 1
                    continue
                columns, defaults = targets[kind]
                try:
                    if kind == "sessions" and nfield == SESSIONS_LEGACY_MINIMAL_NFIELD:
                        # Pre-modern layout with unknown column order: salvage identity + timing only.
                        row_values = (
                            cells[0], cells[1] if _looks_like_source(cells[1]) else "recovered",
                            _heuristic_started_at(cells),
                            f"{STUB_TITLE_PREFIX}] legacy session row (layout unknown)",
                        )
                        inserted = dest.execute(
                            "INSERT OR IGNORE INTO sessions (id, source, started_at, title) VALUES (?, ?, ?, ?)",
                            row_values,
                        ).rowcount == 1
                        report["legacy_minimal_sessions"] += int(inserted)
                    else:
                        # messages: the rowid-alias PK is NULL in the record; use the lost_and_found rowid.
                        values = list(cells[:len(columns)])
                        if kind == "messages":
                            values[0] = lf_rowid
                        inserted = _insert_prefix_row(dest, kind, columns, values, defaults)
                except sqlite3.DatabaseError:
                    report["unmapped_rows"] += 1
                    continue
                if inserted:
                    report["mapped"][kind] += 1
                else:
                    report["insert_conflicts"] += 1
    return report


def stub_missing_parent_sessions(dest: sqlite3.Connection) -> dict[str, Any]:
    """Fabricate clearly-marked placeholder parents for salvaged child rows: children (messages,
    model-usage rows) are NEVER deleted for FK cleanup — a stub parent beats losing the only copy."""
    result: dict[str, Any] = {"sessions_stubbed": 0, "messages_retained": 0, "usage_rows_retained": 0}
    with _immediate_transaction(dest):
        orphan_ids: dict[str, dict[str, Any]] = {}
        for session_id, first_ts, count in dest.execute(
            "SELECT m.session_id, MIN(m.timestamp), COUNT(*) FROM messages AS m WHERE m.session_id IS NOT NULL AND "
            "NOT EXISTS (SELECT 1 FROM sessions WHERE sessions.id = m.session_id) GROUP BY m.session_id"
        ):
            orphan_ids[str(session_id)] = {
                "started_at": float(first_ts) if first_ts is not None else 0.0,
                "message_count": int(count),
            }
        for (session_id,) in dest.execute(
            "SELECT DISTINCT u.session_id FROM session_model_usage AS u WHERE u.session_id IS NOT NULL AND NOT "
            "EXISTS (SELECT 1 FROM sessions WHERE sessions.id = u.session_id)"
        ):
            orphan_ids.setdefault(str(session_id), {"started_at": 0.0, "message_count": 0})
        titles = _placeholder_titles(dest, _STUB_TITLE_LABEL)
        for session_id, info in sorted(orphan_ids.items()):
            title = next(titles)
            dest.execute(
                "INSERT INTO sessions (id, source, started_at, title, message_count) "
                "VALUES (?, 'recovered', ?, ?, ?)",
                (session_id, info["started_at"], title, info["message_count"]),
            )
            result["sessions_stubbed"] += 1
            result["messages_retained"] += info["message_count"]
        result["usage_rows_retained"] = int(dest.execute("SELECT COUNT(*) FROM session_model_usage").fetchone()[0])

        # Repair dangling intra-sessions references without deleting rows.
        dest.execute(
            "UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id IS NOT NULL AND NOT EXISTS (SELECT "
            "1 FROM sessions AS p WHERE p.id = sessions.parent_session_id)"
        )
        dest.execute(
            "UPDATE sessions SET system_prompt_hash = NULL WHERE system_prompt_hash IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM system_prompts WHERE system_prompts.hash = sessions.system_prompt_hash)"
        )
    return result


def rebuild_fts_indexes(dest: sqlite3.Connection) -> dict[str, str]:
    """Rebuild derived FTS indexes from the salvaged canonical rows."""
    results: dict[str, str] = {}
    for table in ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"):
        if not _table_columns(dest, table):
            continue
        try:
            dest.execute(f'INSERT INTO "{table}" ("{table}") VALUES (\'rebuild\')')
            results[table] = "rebuilt"
        except sqlite3.DatabaseError as exc:
            results[table] = f"rebuild failed: {exc}"
    return results
