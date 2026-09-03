"""Last-resort page-level salvage for an unreadable session database schema.

``hermes sessions recover --allow-partial`` normally copies rows through SQL,
which requires the ``sessions`` and ``messages`` table *schemas* to be
readable. When the schema page itself is damaged, SQL-level salvage is
impossible — but the row payloads frequently survive on their b-tree pages.

The SQLite command-line shell ships a page-level ``.recover`` command that
walks raw pages and rebuilds rows it cannot attribute to a schema into
``lost_and_found`` tables of the shape::

    lost_and_found(rootpgno, pgno, nfield, id, c0, c1, ..., cN)

This module shells out to that CLI (it is a shell feature, NOT available via
the Python ``sqlite3`` module) and then heuristically maps ``lost_and_found``
rows back into a fresh current-schema Hermes session database.

Everything produced through this lane is explicitly **best effort**: column
mapping is heuristic (field counts plus sentinel values), fabricated parent
sessions are stubbed for orphaned child rows rather than deleting salvaged
data, and derived FTS indexes are rebuilt from scratch.
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Hermes session ids are timestamps: 20260812_135332_ab12cd. This is the
# strongest sentinel available for classifying schema-less rows.
SESSION_ID_PATTERN = re.compile(r"^\d{8}_\d{6}_")

MESSAGE_ROLES = frozenset({"user", "assistant", "tool", "system"})

# Values observed in sessions.source across gateway platforms and tooling.
KNOWN_SOURCES = frozenset({
    "cli", "telegram", "discord", "slack", "whatsapp", "signal", "matrix",
    "irc", "email", "x", "twitter", "api", "gateway", "web", "dashboard",
    "tool", "subagent", "cron", "recovered", "imported", "acp",
})

# Field counts observed for historical sessions records. A record's field
# count is the column count of the table when the row was last written, so
# these only classify a record — they do NOT imply its cells follow the
# current *declared* column order. Live stores gain columns through
# ``_reconcile_columns()`` (ALTER TABLE ADD COLUMN), which appends them in
# physical add order, while SCHEMA_SQL declares several of them
# mid-definition; the two orders diverge (#101409). Cells are mapped by
# name through PHYSICAL_LAYOUTS below, never zipped onto declared order.
SESSIONS_LAYOUT_NFIELDS = frozenset({55, 54, 52})
SESSIONS_LEGACY_MINIMAL_NFIELD = 14
SESSION_MODEL_USAGE_NFIELD = 18

# Physical column order of a store that has been upgraded in place rather
# than created at the current schema: ``_reconcile_columns()`` APPENDS every
# newly declared column with ALTER TABLE ADD COLUMN, so the table keeps the
# order in which columns were added, not the order SCHEMA_SQL declares them.
# A record with N fields carries the first N names of the matching list
# (SQLite does not rewrite existing rows when a column is added).
#
# The sessions/messages orders below are the ones reported and verified
# against real salvaged rows in #101409, extended with the columns declared
# after that report (they are appended in the same way on any store upgraded
# since). Stores that also carry a column which was declared and then removed
# again (sessions.handoff_pending, sessions.icon,
# messages.anthropic_content_blocks — each shipped for days) are shifted from
# both known orders; such rows fail layout validation below and fall back to
# the conservative path rather than being mis-mapped.
PHYSICAL_LAYOUTS: dict[str, tuple[str, ...]] = {
    "sessions": (
        "id", "source", "user_id", "model", "model_config",
        "system_prompt", "parent_session_id", "started_at", "ended_at",
        "end_reason", "message_count", "tool_call_count", "input_tokens",
        "output_tokens", "cache_read_tokens", "cache_write_tokens",
        "reasoning_tokens", "billing_provider", "billing_base_url",
        "billing_mode", "estimated_cost_usd", "actual_cost_usd",
        "cost_status", "cost_source", "pricing_version", "title",
        "api_call_count", "handoff_state", "handoff_platform",
        "handoff_error", "cwd", "rewind_count", "archived",
        "session_key", "chat_id", "chat_type", "thread_id", "git_branch",
        "git_repo_root", "compression_failure_cooldown_until",
        "compression_failure_error", "display_name", "origin_json",
        "expiry_finalized", "compression_fallback_streak",
        "profile_name", "compression_ineffective_count", "pinned",
        "system_prompt_hash", "last_activity_at",
        "last_activity_description", "last_activity_provenance",
        "git_metadata_generation", "title_source", "hidden",
        "last_read_at", "compression_recovery_deadline", "tool_names",
    ),
    "messages": (
        "id", "session_id", "role", "content", "tool_call_id",
        "tool_calls", "tool_name", "timestamp", "token_count",
        "finish_reason", "reasoning", "reasoning_content",
        "reasoning_details", "codex_reasoning_items",
        "codex_message_items", "platform_message_id", "observed",
        "active", "compacted", "effect_disposition", "api_content",
        "display_kind", "display_metadata", "_compressed_summary",
    ),
    # Same class on the usage table: ``task`` is declared sixth but was added
    # last, and ``billing_mode``/cost columns are declared before the counters
    # they were added after.
    "session_model_usage": (
        "session_id", "model", "billing_provider", "billing_base_url",
        "api_call_count", "input_tokens", "output_tokens",
        "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
        "estimated_cost_usd", "first_seen", "last_seen", "billing_mode",
        "actual_cost_usd", "cost_status", "cost_source", "task",
    ),
}

# Columns whose salvaged value must look right before a candidate layout is
# accepted for a record. They are the cells that differ hardest between the
# declared and the upgraded-physical order, so a wrong layout is rejected
# instead of silently shifting every field.
_LAYOUT_SENTINELS: dict[str, tuple[str, ...]] = {
    "sessions": ("id", "source", "started_at"),
    "messages": ("session_id", "role", "timestamp"),
    "session_model_usage": ("session_id", "model"),
}

# Plausible unix-epoch window for started_at heuristics on legacy layouts.
_EPOCH_LOW = 1_000_000_000.0   # 2001
_EPOCH_HIGH = 4_000_000_000.0  # 2096

# Title prefix of every session row this lane synthesises (legacy-layout rows
# and stubbed parents). The recovery verifier keys on it to tell synthesised
# rows from positionally mapped ones.
STUB_TITLE_PREFIX = "[best-effort recovered"

SQLITE3_CLI_GUIDANCE = (
    "A last-resort page-level salvage is available when a `.recover`-capable "
    "`sqlite3` command-line shell is installed: its `.recover` command can "
    "rebuild rows into lost_and_found tables even when the table schemas are "
    "unreadable (this is a CLI-only feature, not part of Python's sqlite3 "
    "module, and some distro builds lack it — the shell must include the "
    "sqlite_dbpage extension, as the official builds from sqlite.org do). "
    "Install such a sqlite3 CLI (e.g. `brew install sqlite` or the "
    "precompiled sqlite-tools from sqlite.org) so it is on PATH, then re-run "
    "with --allow-partial."
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
    """Parse the reporting version of the sqlite3 CLI at *binary*.

    Returns ``None`` when the CLI cannot be executed or its version line
    cannot be understood (older shells print the version only in
    interactive mode; the modern ``--version`` flag covers every build in
    the supported range).
    """
    try:
        probe = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            timeout=30,
        )
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
    """Why the last :func:`find_sqlite3_cli` call in this process refused.

    ``{"reason": ...}`` with ``reason`` in ``missing``, ``no_dbpage`` (the
    shell cannot run ``.recover``), or ``wal_reset_vulnerable``; empty when
    the last probe found a usable shell or never ran.
    """
    return dict(_last_cli_refusal)


def find_sqlite3_cli() -> Optional[str]:
    """Return a salvage-safe ``.recover``-capable sqlite3 CLI path, or None.

    PATH presence is not enough, and neither is `.recover` support alone:

    1. Distro builds (e.g. Ubuntu's) can ship a sqlite3 shell compiled
       without the ``sqlite_dbpage`` virtual table that ``.recover``
       requires — those fail every recovery with ``no such table:
       sqlite_dbpage``. Capability is probed on a scratch DB once.
    2. A `.recover`-capable CLI can still carry the WAL-reset opener bug
       (fixed 3.51.3+ / backports 3.50.7 / 3.44.6). The salvage lane runs
       the CLI against a *snapshot copy* of the source, so it cannot hit
       the live sidecars itself; but the same binary is what operators
       reach for when following the old guidance, and refusing it here
       keeps the vulnerable shells out of the documented workflow
       entirely. Probe the version once.

    Refusals are recorded for :func:`find_sqlite3_cli_refusal` so callers
    can explain exactly what to install instead of a generic "not found".
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
            "detail": (
                f"reports version {version_str}, which has "
                + _WAL_RESET_VULNERABLE_GUIDANCE
            ),
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
        probe = subprocess.run(
            [binary, "-readonly", str(scratch), ".recover"],
            capture_output=True,
            timeout=30,
        )
        if probe.returncode != 0:
            return False
        return b"sqlite_dbpage" not in probe.stderr
    except (OSError, subprocess.SubprocessError, sqlite3.Error):
        return False
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def run_cli_lost_and_found_recover(
    source: Path,
    lf_path: Path,
    sqlite3_bin: str,
    *,
    timeout: float = 3600.0,
) -> dict[str, Any]:
    """Run ``sqlite3 <source> .recover`` streamed into a fresh scratch DB.

    ``--ignore-freelist`` avoids resurrecting deleted rows; older shells
    without that option fall back to a plain ``.recover``.
    """

    attempts: list[dict[str, Any]] = []
    for command in (".recover --ignore-freelist", ".recover"):
        if lf_path.exists():
            lf_path.unlink()
        dump = subprocess.Popen(
            [sqlite3_bin, "-readonly", str(source), command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        load = subprocess.Popen(
            [sqlite3_bin, str(lf_path)],
            stdin=dump.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
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
            raise LostAndFoundError(
                f"sqlite3 .recover timed out after {timeout:.0f}s"
            )
        attempt = {
            "command": command,
            "dump_returncode": dump.returncode,
            "load_returncode": load.returncode,
            "dump_stderr_tail": dump_err.decode("utf-8", "replace")[-2000:],
            "load_stderr_tail": load_err.decode("utf-8", "replace")[-2000:],
        }
        attempts.append(attempt)
        if _lost_and_found_db_usable(lf_path):
            attempt["usable"] = True
            return {"binary": sqlite3_bin, "attempts": attempts}
        attempt["usable"] = False

    raise LostAndFoundError(
        "sqlite3 .recover did not produce a usable lost_and_found database: "
        + "; ".join(
            f"[{a['command']}] dump rc={a['dump_returncode']} "
            f"load rc={a['load_returncode']} "
            f"{a['dump_stderr_tail'] or a['load_stderr_tail']}".strip()
            for a in attempts
        )
    )


def _lost_and_found_db_usable(lf_path: Path) -> bool:
    if not lf_path.exists() or lf_path.stat().st_size == 0:
        return False
    try:
        conn = sqlite3.connect(str(lf_path))
        try:
            tables = [
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False
    return bool(tables)


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]


def _notnull_defaults(conn: sqlite3.Connection, table: str) -> dict[int, Any]:
    """Map column index -> substitute value for NOT NULL columns.

    Page-level salvage can hand back records with NULL in positions that the
    live schema declares NOT NULL (torn cells, historical rows). Dropping the
    whole row over one damaged optional counter would defeat the lane, so
    NULLs in such positions are replaced by the schema default (or '' / 0
    when no default is declared).
    """

    substitutes: dict[int, Any] = {}
    for index, row in enumerate(conn.execute(f'PRAGMA table_info("{table}")')):
        if not row[3]:  # notnull flag
            continue
        default = row[4]
        if default is None:
            declared = str(row[2] or "").upper()
            substitutes[index] = 0 if ("INT" in declared or "REAL" in declared) else ""
            continue
        text = str(default)
        if text.startswith("'") and text.endswith("'"):
            substitutes[index] = text[1:-1]
        else:
            try:
                substitutes[index] = int(text)
            except ValueError:
                try:
                    substitutes[index] = float(text)
                except ValueError:
                    substitutes[index] = text
    return substitutes


def _is_session_id(value: Any) -> bool:
    return isinstance(value, str) and bool(SESSION_ID_PATTERN.match(value))


def _looks_like_source(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return value in KNOWN_SOURCES or bool(re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value))


def classify_lost_and_found_row(
    nfield: int,
    cells: tuple[Any, ...],
) -> Optional[str]:
    """Classify one lost_and_found record by field count + sentinel values.

    Returns 'sessions', 'messages', 'session_model_usage', or None.
    """

    if len(cells) >= 3 and cells[0] is None:
        # Rowid-alias tables store their INTEGER PRIMARY KEY as NULL in the
        # record; messages is the only canonical table shaped like that with
        # a session id second and a role third.
        if (
            isinstance(cells[1], str)
            and cells[1]
            and isinstance(cells[2], str)
            and cells[2] in MESSAGE_ROLES
        ):
            return "messages"
        return None

    if not _is_session_id(cells[0] if cells else None):
        return None

    if nfield == SESSION_MODEL_USAGE_NFIELD:
        # 18 fields, session id first, model string second.
        if len(cells) > 1 and isinstance(cells[1], str) and cells[1]:
            return "session_model_usage"
        return None

    if nfield in SESSIONS_LAYOUT_NFIELDS or nfield == SESSIONS_LEGACY_MINIMAL_NFIELD:
        if len(cells) > 1 and _looks_like_source(cells[1]):
            return "sessions"
        return None

    # Unknown historical sessions layout: session-id first cell plus a
    # recognizable source string is still strong enough for a prefix map.
    if nfield >= 30 and len(cells) > 1 and _looks_like_source(cells[1]):
        return "sessions"
    return None


def _heuristic_started_at(cells: tuple[Any, ...]) -> float:
    for value in cells:
        if isinstance(value, (int, float)) and _EPOCH_LOW <= float(value) <= _EPOCH_HIGH:
            return float(value)
    return 0.0


def _insert_prefix_row(
    dest: sqlite3.Connection,
    table: str,
    dest_columns: list[str],
    values: list[Any],
    notnull_substitutes: Optional[dict[int, Any]] = None,
) -> bool:
    if notnull_substitutes:
        values = [
            notnull_substitutes[index]
            if value is None and index in notnull_substitutes
            else value
            for index, value in enumerate(values)
        ]
    columns = dest_columns[: len(values)]
    quoted = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    cursor = dest.execute(
        f'INSERT OR IGNORE INTO "{table}" ({quoted}) VALUES ({placeholders})',
        values,
    )
    return cursor.rowcount == 1


def _declared_types(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    return {
        str(row[1]): str(row[2] or "")
        for row in conn.execute(f'PRAGMA table_info("{table}")')
    }


def _type_conflicts(value: Any, declared: str) -> bool:
    """True when a salvaged cell cannot belong to a column of this type."""

    if value is None:
        return False
    affinity = declared.upper()
    if "INT" in affinity or "REAL" in affinity or "FLOA" in affinity or "DOUB" in affinity:
        return not isinstance(value, (int, float))
    if "CHAR" in affinity or "CLOB" in affinity or "TEXT" in affinity:
        return not isinstance(value, str)
    return False


def _sentinel_holds(name: str, value: Any) -> bool:
    if name == "id":  # sessions.id; messages.id is a rowid alias, never a sentinel
        return _is_session_id(value)
    if name == "source":
        return _looks_like_source(value)
    if name in ("started_at", "timestamp"):
        return (
            isinstance(value, (int, float))
            and _EPOCH_LOW <= float(value) <= _EPOCH_HIGH
        )
    if name == "session_id":
        return isinstance(value, str) and bool(value)
    if name == "role":
        return value in MESSAGE_ROLES
    if name == "model":
        return isinstance(value, str) and bool(value)
    return True


def _layout_fits_cells(
    kind: str,
    layout: list[str],
    cells: tuple[Any, ...],
    dest_types: dict[str, str],
) -> bool:
    positions = {name: index for index, name in enumerate(layout)}
    for name in _LAYOUT_SENTINELS[kind]:
        if name not in positions or positions[name] >= len(cells):
            return False
        if not _sentinel_holds(name, cells[positions[name]]):
            return False
    for name, value in zip(layout, cells):
        declared = dest_types.get(name)
        if declared is not None and _type_conflicts(value, declared):
            return False
    return True


def select_physical_layout(
    kind: str,
    cells: tuple[Any, ...],
    dest_columns: list[str],
    dest_types: dict[str, str],
) -> Optional[list[str]]:
    """Return the source column names for ``cells``, or None if unknown.

    A salvaged record carries no schema, so the only way to know which cell
    is ``started_at`` is to recognise the layout that produced it. Two are
    known: the destination's own declared order (a store created at the
    current schema, never ALTERed) and the upgraded-in-place physical order
    (#101409). Both are checked against the record's sentinel cells and
    column affinities; an unrecognised layout returns None so the caller can
    fall back conservatively instead of shifting every field.
    """

    candidates: list[list[str]] = []
    for full in (dest_columns, list(PHYSICAL_LAYOUTS.get(kind, ()))):
        candidate = full[: len(cells)]
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if _layout_fits_cells(kind, candidate, cells, dest_types):
            return candidate
    return None


def _insert_named_row(
    dest: sqlite3.Connection,
    table: str,
    layout: list[str],
    cells: tuple[Any, ...],
    dest_columns: list[str],
    notnull_substitutes: dict[str, Any],
    overrides: Optional[dict[str, Any]] = None,
) -> bool:
    """INSERT salvaged cells by source column name, never by position."""

    known = set(dest_columns)
    mapped: dict[str, Any] = {}
    for name, value in zip(layout, cells):
        if name not in known:
            continue  # column dropped since the source was written
        if value is None and name in notnull_substitutes:
            value = notnull_substitutes[name]
        mapped[name] = value
    mapped.update(overrides or {})
    columns = list(mapped)
    quoted = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    cursor = dest.execute(
        f'INSERT OR IGNORE INTO "{table}" ({quoted}) VALUES ({placeholders})',
        [mapped[column] for column in columns],
    )
    return cursor.rowcount == 1


def _copy_direct_tables(
    lf_conn: sqlite3.Connection,
    dest: sqlite3.Connection,
) -> dict[str, int]:
    """Copy rows .recover managed to attribute to real canonical tables."""

    # Lazy import: session_recovery imports this module inside a function, so
    # a module-level import here would be circular.
    from hermes_cli.session_recovery import (
        _AUXILIARY_TABLE_SCHEMAS,
        _AUXILIARY_TABLES,
        _CANONICAL_TABLES,
    )

    copied: dict[str, int] = {}
    for table in (*_CANONICAL_TABLES, *_AUXILIARY_TABLES):
        source_columns = _table_columns(lf_conn, table)
        if not source_columns:
            continue
        dest_columns = _table_columns(dest, table)
        if not dest_columns and table in _AUXILIARY_TABLE_SCHEMAS:
            # Lazily-created gateway table: base SessionDB never made it on
            # the fresh destination, so create it before copying.
            _AUXILIARY_TABLE_SCHEMAS[table](dest)
            dest_columns = _table_columns(dest, table)
        columns = [c for c in dest_columns if c in source_columns]
        if not columns:
            continue
        quoted = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join("?" for _ in columns)
        rows = lf_conn.execute(f'SELECT {quoted} FROM "{table}"').fetchall()
        if not rows:
            copied[table] = 0
            continue
        before = int(dest.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        dest.executemany(
            f'INSERT OR IGNORE INTO "{table}" ({quoted}) VALUES ({placeholders})',
            rows,
        )
        after = int(dest.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        copied[table] = after - before
    return copied


def map_lost_and_found_rows(
    lf_conn: sqlite3.Connection,
    dest: sqlite3.Connection,
) -> dict[str, Any]:
    """Best-effort mapping of a .recover output DB into a fresh SessionDB.

    Handles both rows .recover attributed to real tables and unattributed
    ``lost_and_found`` rows classified by field count + sentinel columns.
    """

    report: dict[str, Any] = {
        "direct_table_rows": {},
        "mapped": {"sessions": 0, "messages": 0, "session_model_usage": 0},
        "legacy_minimal_sessions": 0,
        "mapped_by_layout": 0,
        "unrecognized_layout_rows": 0,
        "unmapped_rows": 0,
        "insert_conflicts": 0,
        "lost_and_found_tables": [],
    }

    dest.execute("BEGIN IMMEDIATE")
    try:
        report["direct_table_rows"] = _copy_direct_tables(lf_conn, dest)

        sessions_columns = _table_columns(dest, "sessions")
        messages_columns = _table_columns(dest, "messages")
        usage_columns = _table_columns(dest, "session_model_usage")
        sessions_defaults = _notnull_defaults(dest, "sessions")
        messages_defaults = _notnull_defaults(dest, "messages")
        usage_defaults = _notnull_defaults(dest, "session_model_usage")
        # Never fabricate identity fields: a row whose session id / role /
        # source cell is genuinely NULL was already rejected by
        # classify_lost_and_found_row, so these substitutions only fill
        # NOT NULL bookkeeping counters and flag columns.
        for defaults, protected in (
            (sessions_defaults, (0, 1)),
            (messages_defaults, (1, 2)),
            (usage_defaults, (0, 1)),
        ):
            for index in protected:
                defaults.pop(index, None)

        dest_columns = {
            "sessions": sessions_columns,
            "messages": messages_columns,
            "session_model_usage": usage_columns,
        }
        dest_types = {
            table: _declared_types(dest, table) for table in dest_columns
        }
        # The same substitutes, keyed by name for the mapped-by-name path.
        named_defaults = {
            table: {
                dest_columns[table][index]: value
                for index, value in defaults.items()
                if index < len(dest_columns[table])
            }
            for table, defaults in (
                ("sessions", sessions_defaults),
                ("messages", messages_defaults),
                ("session_model_usage", usage_defaults),
            )
        }

        lf_tables = [
            str(row[0])
            for row in lf_conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name LIKE 'lost_and_found%'"
            )
        ]
        report["lost_and_found_tables"] = lf_tables

        for lf_table in lf_tables:
            lf_columns = _table_columns(lf_conn, lf_table)
            if lf_columns[:3] != ["rootpgno", "pgno", "nfield"]:
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
                layout = select_physical_layout(
                    kind, cells, dest_columns[kind], dest_types[kind]
                )
                if layout is None and not (
                    kind == "sessions"
                    and nfield == SESSIONS_LEGACY_MINIMAL_NFIELD
                ):
                    # No known layout produced these cells (a store that also
                    # carries since-removed columns, or a torn sentinel). The
                    # positional prefix below is a guess, so count it: the
                    # recovery verifier's plausibility gate is what keeps such
                    # a salvage from being reported as verified.
                    report["unrecognized_layout_rows"] += 1
                try:
                    if layout is not None:
                        # The source's own column names are known: map cell to
                        # column by name (#101409). Zipping onto the fresh
                        # destination's declared order would shift every field
                        # of a store upgraded via ALTER TABLE ADD COLUMN.
                        inserted = _insert_named_row(
                            dest,
                            kind,
                            layout,
                            cells,
                            dest_columns[kind],
                            named_defaults[kind],
                            # messages.id is a rowid alias: NULL in the record,
                            # carried by the lost_and_found row id instead.
                            {"id": lf_rowid} if kind == "messages" else None,
                        )
                        if inserted:
                            report["mapped_by_layout"] += 1
                    elif kind == "messages":
                        values = [lf_rowid, *cells[1 : min(nfield, len(messages_columns))]]
                        inserted = _insert_prefix_row(
                            dest, "messages", messages_columns, values,
                            messages_defaults,
                        )
                    elif kind == "session_model_usage":
                        values = list(cells[: len(usage_columns)])
                        inserted = _insert_prefix_row(
                            dest, "session_model_usage", usage_columns, values,
                            usage_defaults,
                        )
                    elif nfield == SESSIONS_LEGACY_MINIMAL_NFIELD:
                        # A pre-modern layout whose column order is unknown:
                        # salvage identity + timing rather than guessing 14
                        # positional meanings.
                        inserted = bool(
                            dest.execute(
                                "INSERT OR IGNORE INTO sessions "
                                "(id, source, started_at, title) "
                                "VALUES (?, ?, ?, ?)",
                                (
                                    cells[0],
                                    cells[1] if _looks_like_source(cells[1])
                                    else "recovered",
                                    _heuristic_started_at(cells),
                                    f"{STUB_TITLE_PREFIX}] legacy session "
                                    "row (layout unknown)",
                                ),
                            ).rowcount
                            == 1
                        )
                        if inserted:
                            report["legacy_minimal_sessions"] += 1
                    else:
                        values = list(cells[: min(nfield, len(sessions_columns))])
                        inserted = _insert_prefix_row(
                            dest, "sessions", sessions_columns, values,
                            sessions_defaults,
                        )
                except sqlite3.DatabaseError:
                    report["unmapped_rows"] += 1
                    continue
                if inserted:
                    report["mapped"][
                        "sessions" if kind == "sessions" else kind
                    ] += 1
                else:
                    report["insert_conflicts"] += 1
        dest.execute("COMMIT")
    except BaseException:
        dest.execute("ROLLBACK")
        raise
    return report


def stub_missing_parent_sessions(dest: sqlite3.Connection) -> dict[str, Any]:
    """Fabricate placeholder parents for salvaged child rows.

    Salvaged children (messages, model-usage rows) are NEVER deleted for
    foreign-key cleanup — a fabricated parent is cheaper than losing the only
    surviving copy of the user's data. Stubs are clearly marked.
    """

    result: dict[str, Any] = {
        "sessions_stubbed": 0,
        "messages_retained": 0,
        "usage_rows_retained": 0,
    }
    dest.execute("BEGIN IMMEDIATE")
    try:
        orphan_ids: dict[str, dict[str, Any]] = {}
        for session_id, first_ts, count in dest.execute(
            "SELECT m.session_id, MIN(m.timestamp), COUNT(*) FROM messages AS m "
            "WHERE m.session_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM sessions WHERE sessions.id = m.session_id) "
            "GROUP BY m.session_id"
        ):
            orphan_ids[str(session_id)] = {
                "started_at": float(first_ts) if first_ts is not None else 0.0,
                "message_count": int(count),
            }
        for (session_id,) in dest.execute(
            "SELECT DISTINCT u.session_id FROM session_model_usage AS u "
            "WHERE u.session_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM sessions WHERE sessions.id = u.session_id)"
        ):
            orphan_ids.setdefault(
                str(session_id), {"started_at": 0.0, "message_count": 0}
            )

        sequence = 1
        for session_id, info in sorted(orphan_ids.items()):
            while True:
                title = (
                    f"{STUB_TITLE_PREFIX} {sequence}] session metadata "
                    "was unreadable"
                )
                sequence += 1
                if (
                    dest.execute(
                        "SELECT 1 FROM sessions WHERE title = ? LIMIT 1",
                        (title,),
                    ).fetchone()
                    is None
                ):
                    break
            dest.execute(
                "INSERT INTO sessions (id, source, started_at, title, "
                "message_count) VALUES (?, 'recovered', ?, ?, ?)",
                (
                    session_id,
                    info["started_at"],
                    title,
                    info["message_count"],
                ),
            )
            result["sessions_stubbed"] += 1
            result["messages_retained"] += info["message_count"]

        result["usage_rows_retained"] = int(
            dest.execute("SELECT COUNT(*) FROM session_model_usage").fetchone()[0]
        )

        # Repair dangling intra-sessions references without deleting rows.
        dest.execute(
            "UPDATE sessions SET parent_session_id = NULL "
            "WHERE parent_session_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM sessions AS p WHERE p.id = sessions.parent_session_id)"
        )
        dest.execute(
            "UPDATE sessions SET system_prompt_hash = NULL "
            "WHERE system_prompt_hash IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM system_prompts "
            "WHERE system_prompts.hash = sessions.system_prompt_hash)"
        )
        dest.execute("COMMIT")
    except BaseException:
        dest.execute("ROLLBACK")
        raise
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
