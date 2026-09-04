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

One corruption shape defeats the CLI lane before it starts: when page 1
itself is damaged (SIGKILL mid-write after an OOM), SQLite refuses the file
outright — ``file is not a database`` — and ``.recover`` opens files like any
other client, so it fails too. The pages *after* page 1 usually survive
intact. This module therefore first splices a valid 100-byte SQLite header
onto a private copy of such a file (page size discovered from the damaged
bytes when plausible, else a candidate sweep), then runs the normal
``.recover`` flow against the repaired copy. The source is never modified.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

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

# Historical physical layouts of the sessions table. Columns are only ever
# appended (ALTER TABLE ADD COLUMN), so an older record is a strict prefix of
# the current column order.
SESSIONS_LAYOUT_NFIELDS = frozenset({55, 54, 52})
SESSIONS_LEGACY_MINIMAL_NFIELD = 14
SESSION_MODEL_USAGE_NFIELD = 18

# Plausible unix-epoch window for started_at heuristics on legacy layouts.
_EPOCH_LOW = 1_000_000_000.0   # 2001
_EPOCH_HIGH = 4_000_000_000.0  # 2096

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


class LostAndFoundError(RuntimeError):
    """Raised when the CLI .recover pass cannot produce a usable database."""


def find_sqlite3_cli() -> Optional[str]:
    """Return a ``.recover``-capable sqlite3 CLI path, or None.

    PATH presence is not enough: distro builds (e.g. Ubuntu's) can ship a
    sqlite3 shell compiled without the ``sqlite_dbpage`` virtual table that
    ``.recover`` requires — those fail every recovery with
    ``no such table: sqlite_dbpage``. Probe capability on a scratch DB once
    instead of discovering it mid-recovery.
    """

    binary = shutil.which("sqlite3")
    if binary is None:
        return None
    return binary if _cli_supports_recover(binary) else None


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


# ── header salvage (SQLite refuses a page-1-damaged file outright) ─────────


SQLITE_HEADER_LENGTH = 100

# The version-number bytes of a SQLite header are file-format revisions (1 =
# rollback journal, 2 = WAL), not user data. A damaged pair makes SQLite
# reject the file with 'unsupported file format', so a spliced header always
# sanitizes them; WAL content without its -wal sidecar is lost anyway and the
# rollback-mode bytes let SQLite read the main file directly.
_SQLITE_VERSION_ROLLBACK = (1, 1)

# Page sizes to try when the damaged bytes 16-17 are not a legal page size.
# 4096 is the SQLite default and what Hermes state.db files use in practice.
_HEADER_PAGE_SIZE_CANDIDATES = (4096, 8192, 16384, 65536, 2048, 1024, 512)


_HEADER_REJECTION_HINTS = ("not a database", "unsupported file format")
# SQLITE_NOTADB — raised when the 16-byte magic (or the version bytes) in
# page 1 are damaged, i.e. the file is not recognized as a database at all.
_SQLITE_NOTADB = 26


def header_version_bytes(path: Path) -> tuple[int, int]:
    """The write/read format version pair from the file's own header.

    1 = rollback journal, 2 = WAL. Returns the pair only when the damaged
    file still shows a consistent, legal value; the rollback default is the
    safe choice for anything else (unmatched pairs make SQLite refuse the
    file with 'unsupported file format', and a WAL header without its -wal
    sidecar buys nothing).
    """

    try:
        with path.open("rb") as handle:
            pair = handle.read(20)[18:20]
    except OSError:
        pair = b""
    if len(pair) == 2 and pair[0] == pair[1] and pair[0] in (1, 2):
        return (pair[0], pair[1])
    return _SQLITE_VERSION_ROLLBACK


def detect_database_open_error(path: Path) -> Optional[str]:
    """Return the header-rejection error when SQLite refuses ``path``, else None.

    ``PRAGMA journal_mode`` is the first statement that parses the database
    header (matching ``hermes_state._db_opens_cleanly``). Only rejections
    that indict the header itself — ``file is not a database`` /
    ``unsupported file format`` — are reported here; ``database disk image
    is malformed`` means the file opens and the schema is damaged, which is
    the existing page-level lane's normal input, not a header salvage case.
    Operational errors (busy/locked) are reported as None so the plain
    flow's own retries stay in charge.
    """

    code: Optional[int] = None
    try:
        conn = sqlite3.connect(str(path), isolation_level=None, timeout=1.0)
    except sqlite3.DatabaseError as exc:
        text = str(exc)
        code = getattr(exc, "sqlite_errorcode", None)
    else:
        try:
            conn.execute("PRAGMA journal_mode").fetchone()
            return None
        except sqlite3.DatabaseError as exc:
            text = str(exc)
            code = getattr(exc, "sqlite_errorcode", None)
        except sqlite3.Error:
            return None
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass
    lowered = text.lower()
    if code == _SQLITE_NOTADB or any(hint in lowered for hint in _HEADER_REJECTION_HINTS):
        return text
    return None


def _donor_header(
    scratch_dir: Path,
    page_size: int,
    version_bytes: tuple[int, int],
) -> bytes:
    """Return a fresh valid 100-byte header for ``page_size``.

    Built by letting SQLite itself write a minimal database (one table) at
    the requested page size; only the version bytes are overridden.
    """

    donor = scratch_dir / f"header-donor-{page_size}.db"
    if donor.exists():
        donor.unlink()
    conn = sqlite3.connect(str(donor), isolation_level=None)
    try:
        conn.execute(f"PRAGMA page_size={page_size}")
        conn.execute("CREATE TABLE hermes_header_donor (x)")
        conn.execute("INSERT INTO hermes_header_donor VALUES (1)")
    finally:
        conn.close()
    header = bytearray(donor.read_bytes()[:SQLITE_HEADER_LENGTH])
    if len(header) != SQLITE_HEADER_LENGTH:
        raise LostAndFoundError(
            f"header donor at page size {page_size} produced a short file"
        )
    header[18] = version_bytes[0]
    header[19] = version_bytes[1]
    donor.unlink(missing_ok=True)
    return bytes(header)


def build_sqlite_header(
    page_size: int,
    version_bytes: tuple[int, int] = _SQLITE_VERSION_ROLLBACK,
) -> bytes:
    """Public wrapper around the donor header for tests and reuse."""

    with tempfile.TemporaryDirectory(prefix="hermes-header-donor-") as tmp:
        return _donor_header(Path(tmp), page_size, version_bytes)


def header_page_size_candidates(path: Path) -> list[int]:
    """Plausible page sizes for a header-damaged file, best guess first.

    bytes 16-17 of the file are the page size (1 means 65536); when those
    bytes themselves are damaged the standard sizes are swept, default first.
    """

    candidates: list[int] = []
    try:
        with path.open("rb") as handle:
            size_bytes = handle.read(18)[16:18]
    except OSError:
        size_bytes = b""
    if len(size_bytes) == 2:
        raw = int.from_bytes(size_bytes, "big")
        page_size = 65536 if raw == 1 else raw
        if 512 <= page_size <= 65536 and page_size & (page_size - 1) == 0:
            candidates.append(page_size)
    for standard in _HEADER_PAGE_SIZE_CANDIDATES:
        if standard not in candidates:
            candidates.append(standard)
    return candidates


def _splice_header(
    source: Path,
    repaired: Path,
    header: bytes,
) -> None:
    """Write ``source`` with its first 100 bytes replaced by ``header``.

    Streamed so a 335 MB damaged database costs one streamed copy, not two
    in-memory loads.
    """

    with source.open("rb") as handle:
        tail = handle.read(SQLITE_HEADER_LENGTH)
        if len(tail) < SQLITE_HEADER_LENGTH:
            raise LostAndFoundError(
                "source file is shorter than a SQLite header; it is not a "
                "page-1-damaged database"
            )
    with source.open("rb") as read_handle, repaired.open("wb") as write_handle:
        read_handle.seek(SQLITE_HEADER_LENGTH)
        write_handle.write(header[:SQLITE_HEADER_LENGTH])
        shutil.copyfileobj(read_handle, write_handle, length=1024 * 1024)


def salvage_header_damaged_source(
    source: Path,
    lf_path: Path,
    sqlite3_bin: str,
    *,
    timeout: float,
) -> Optional[dict[str, Any]]:
    """Splice a valid header onto a copy and retry ``.recover``.

    Returns a report dict when the source cannot be opened at all and one of
    the candidate headers makes the existing ``.recover`` flow produce a
    usable lost_and_found database; ``None`` when the source opens fine
    (nothing to do) or no candidate worked (the caller keeps its own error).
    Never touches ``source``.
    """

    open_error = detect_database_open_error(source)
    if open_error is None:
        return None

    salvage_dir = Path(
        tempfile.mkdtemp(prefix="hermes-header-salvage-", dir=str(lf_path.parent))
    )
    try:
        # Keep the file's own journal-format version pair when its header
        # bytes survived; otherwise default to rollback mode.
        version_bytes = header_version_bytes(source)
        # The repaired copy lives beside lost_and_found.db, inside the
        # snapshot directory that the caller owns and removes when recovery
        # finishes; the transient donor files are removed here.
        repaired = lf_path.with_name(f"{source.name}.header-fixed")
        recover_error: Optional[str] = None
        for page_size in header_page_size_candidates(source):
            try:
                header = _donor_header(salvage_dir, page_size, version_bytes)
            except (LostAndFoundError, OSError, sqlite3.Error):
                continue
            try:
                _splice_header(source, repaired, header)
            except (LostAndFoundError, OSError):
                return None  # too short to be a damaged SQLite file
            try:
                report = _run_cli_recover_attempts(
                    repaired, lf_path, sqlite3_bin, timeout=timeout
                )
            except LostAndFoundError as exc:
                if "not a database" not in str(exc).lower():
                    # The header was accepted but the surviving pages are
                    # damaged beyond .recover — a different page size will
                    # not change that. Keep the real error for the report.
                    recover_error = str(exc)
                    break
                continue
            report["header_salvage"] = {
                "triggered": True,
                "open_error": open_error,
                "page_size": page_size,
                "repaired_source": str(repaired),
            }
            return report
        return {
            "triggered": False,
            "open_error": open_error,
            "candidates_tried": header_page_size_candidates(source),
            "recover_error": recover_error,
        }
    finally:
        shutil.rmtree(salvage_dir, ignore_errors=True)


def _run_cli_recover_attempts(
    source: Path,
    lf_path: Path,
    sqlite3_bin: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    """The ``.recover`` command attempts, extracted for header retry."""

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

    When SQLite refuses to open ``source`` at all (damaged page 1 reports
    ``file is not a database``), a valid header is spliced onto a private
    copy first — the source is never modified — and the copy is recovered.
    """

    salvage = salvage_header_damaged_source(
        source, lf_path, sqlite3_bin, timeout=timeout
    )
    if salvage is not None and salvage.get("header_salvage", {}).get("triggered"):
        return salvage

    try:
        report = _run_cli_recover_attempts(
            source, lf_path, sqlite3_bin, timeout=timeout
        )
    except LostAndFoundError as exc:
        if salvage is not None:
            raise LostAndFoundError(
                f"{exc}; a valid header was also spliced onto a copy of the "
                f"source (open error: {salvage['open_error']!r}; page-size "
                f"candidates tried: {salvage['candidates_tried']}) and the "
                "recovered copy still produced no usable rows"
            ) from None
        raise
    if salvage is not None:
        report["header_salvage"] = salvage
    return report


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


def _copy_direct_tables(
    lf_conn: sqlite3.Connection,
    dest: sqlite3.Connection,
) -> dict[str, int]:
    """Copy rows .recover managed to attribute to real canonical tables."""

    copied: dict[str, int] = {}
    for table in (
        "system_prompts",
        "sessions",
        "messages",
        "session_model_usage",
        "compression_locks",
        "gateway_routing",
        "async_delegations",
    ):
        source_columns = _table_columns(lf_conn, table)
        if not source_columns:
            continue
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
                try:
                    if kind == "messages":
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
                                    "[best-effort recovered] legacy session "
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
                    f"[best-effort recovered {sequence}] session metadata "
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
