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
from typing import Any, Callable, Optional, Sequence

from hermes_cli.session_schema_history import SCHEMA_HISTORY, reachable_physical_layouts

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

# A record's field count is the column count of the table when the row was last written. It classifies
# the record but says nothing about which column each cell belongs to — live stores gain columns through
# ``_reconcile_columns()`` (ALTER TABLE ADD COLUMN), appended in physical add order, while SCHEMA_SQL
# declares several mid-definition (#101409). Cells are mapped by name through the layout
# ``infer_physical_layouts`` recovers, never zipped onto declared order.
SESSIONS_LEGACY_MINIMAL_NFIELD = 14
SESSION_MODEL_USAGE_NFIELD = 18

# Per-table cells whose salvaged value must look like what the column name
# says. On stores created at the original schema these sit in the shared
# base prefix, so they veto a width outright (-> unrecognized) rather than
# pick between candidates; the per-column type and text-shape rules below
# carry the discrimination between layouts.
_LAYOUT_SENTINELS: dict[str, tuple[str, ...]] = {
    "sessions": ("id", "source", "started_at"),
    "messages": ("session_id", "role", "timestamp"),
    "session_model_usage": ("session_id", "model"),
}

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
    # Any historical sessions width: session id first + recognizable source second (every sessions
    # layout ever shipped has at least the 14 original columns).
    if nfield >= SESSIONS_LEGACY_MINIMAL_NFIELD and _looks_like_source(second):
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
    return _execute_insert(dest, table, dest_columns[: len(values)], values)


def _declared_types(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    return {str(row[1]): str(row[2] or "") for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _type_conflicts(value: Any, declared: str) -> bool:
    """True when a salvaged cell cannot have come from a column of this type.

    Stricter than SQLite affinity on purpose: SQLite would happily keep a
    non-numeric string in an INTEGER column, but Hermes never writes one, so
    text sitting where a counter is declared means the layout is wrong. A
    TEXT column always yields ``str`` (numbers are coerced on write), a REAL
    column always yields ``float``, an INTEGER column yields ``int``.
    """

    if value is None:
        return False
    affinity = declared.upper()
    if "CHAR" in affinity or "CLOB" in affinity or "TEXT" in affinity:
        return not isinstance(value, str)
    if "REAL" in affinity or "FLOA" in affinity or "DOUB" in affinity:
        return not isinstance(value, float)
    if "INT" in affinity:
        return not isinstance(value, int)
    return False


# Shapes of the text columns that move between the declared and physical
# orders. A salvaged string that does not fit the shape rules out the
# column for that position; a string that fits does not prove it. Only
# invariants every writer in this repo honours belong here (this is how the
# columns are actually written, not how they could theoretically be).
_SESSION_KEY_PATTERN = re.compile(r"^agent:[^:]+:")
# Machine tokens (``compression``, ``agent.compression_timeout``, ``api_key``,
# ``compute_host_shutdown``): never spaces, never mixed case.
_TOKEN_PATTERN = re.compile(r"^[a-z][a-z0-9_.:-]*$")
# The closed sets SessionDB writes (hermes_state.py: request/claim/complete/
# fail_handoff; TITLE_SOURCE_* + _title_rank). chat_type is platform-supplied
# and open-ended, so it only gets the identifier shape.
_HANDOFF_STATES = frozenset({"pending", "running", "completed", "failed"})
_TITLE_SOURCES = frozenset({"derived", "llm", "user"})


def _is_epoch(value: Any) -> bool:
    return isinstance(value, (int, float)) and _EPOCH_LOW <= float(value) <= _EPOCH_HIGH


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


# Sentinel columns: the cells that differ hardest between candidate layouts, so a wrong layout is
# rejected instead of silently shifting every field. messages.id is a rowid alias, never a sentinel.
_SENTINEL_RULES: dict[str, Callable[[Any], bool]] = {
    "id": _is_session_id,
    "source": _looks_like_source,
    "started_at": _is_epoch,
    "timestamp": _is_epoch,
    "session_id": _is_nonempty_str,
    "role": lambda value: value in MESSAGE_ROLES,
    "model": _is_nonempty_str,
}


def _sentinel_holds(name: str, value: Any) -> bool:
    rule = _SENTINEL_RULES.get(name)
    return rule(value) if rule else True


def _is_token(value: str) -> bool:
    return bool(_TOKEN_PATTERN.match(value))


def _is_json_start(value: str) -> bool:
    return value[:1] in "{["


def _is_path(value: str) -> bool:
    return value[:1] in "/~" or (len(value) > 1 and value[1] == ":")


def _is_url(value: str) -> bool:
    return "://" in value


def _blank_or(rule: Callable[[str], bool]) -> Callable[[str], bool]:
    return lambda value: value == "" or rule(value)


# Cheap per-column shape rules for text cells, by table (see module comment above).
_TEXT_SHAPE_RULES: dict[str, dict[str, Callable[[str], bool]]] = {
    "sessions": {
        "session_key": lambda value: bool(_SESSION_KEY_PATTERN.match(value)),
        **dict.fromkeys(("chat_type", "end_reason", "cost_status", "cost_source", "billing_mode",
                         "last_activity_provenance", "handoff_platform"), _is_token),
        "pricing_version": lambda value: bool(re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value)),
        "title_source": lambda value: value in _TITLE_SOURCES,
        "handoff_state": lambda value: value in _HANDOFF_STATES,
        "parent_session_id": _is_session_id,
        "system_prompt_hash": lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value)),
        "model_config": _is_json_start, "origin_json": _is_json_start,
        "cwd": _is_path, "git_repo_root": _is_path,
        "billing_base_url": _is_url,
    },
    "messages": {
        **dict.fromkeys(("effect_disposition", "finish_reason", "display_kind"), _is_token),
        **dict.fromkeys(("tool_calls", "reasoning_details", "codex_reasoning_items", "codex_message_items",
                         "api_content", "display_metadata"), _blank_or(_is_json_start)),
    },
    "session_model_usage": {
        "billing_base_url": _blank_or(_is_url),
        **dict.fromkeys(("billing_mode", "cost_status", "cost_source"), _blank_or(_is_token)),
    },
}


def _text_shape_holds(kind: str, name: str, value: str) -> bool:
    rule = _TEXT_SHAPE_RULES.get(kind, {}).get(name)
    return rule(value) if rule else True


def _cell_fits(kind: str, name: str, value: Any, dest_types: dict[str, str]) -> bool:
    if value is None:
        return True
    if name in _LAYOUT_SENTINELS[kind] and not _sentinel_holds(name, value):
        return False
    declared = dest_types.get(name)
    # Columns since removed from the schema have no destination type; any
    # value is admissible there (and is dropped on insert).
    if declared is None:
        return True
    if _type_conflicts(value, declared):
        return False
    return not isinstance(value, str) or _text_shape_holds(kind, name, value)


def _row_invariants_hold(kind: str, layout: tuple[str, ...], rows: Sequence[tuple[Any, ...]]) -> bool:
    """Cross-column invariants every writer honours, checked per record.

    ``handoff_error`` is only ever written together with ``handoff_state``
    (``mark_handoff_failed``); a record with a value at the position a
    candidate calls ``handoff_error`` but NULL where it calls
    ``handoff_state`` cannot have come from that layout. This is what
    separates ``cwd`` from ``handoff_error`` when a store's rows never
    handed off — the two are otherwise both free-form TEXT.
    """

    if kind != "sessions":
        return True
    positions = {name: index for index, name in enumerate(layout)}
    error_at = positions.get("handoff_error")
    state_at = positions.get("handoff_state")
    if error_at is None or state_at is None:
        return True
    # rows are exactly len(layout) wide (bucketed by width), so both positions are in range.
    return all(row[error_at] is None or row[state_at] is not None for row in rows)


_SAMPLE_CAP = 512


class LayoutEvidence:
    """What layout inference needs from a population, gathered in one streaming pass.

    Distinct non-NULL values per position (a column's admissibility is a property of the value, not the
    row, and salvaged populations repeat values heavily — a few hundred distinct values per position
    discriminate as well as 150k) plus, for ``sessions`` only, the rows themselves for the cross-column
    invariant. Keeping the whole population — every ``messages.content`` included — until pass 2 would
    hold the entire corrupted store in memory.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.widths: set[int] = set()
        self.by_position: list[set[Any]] = []
        self.rows_by_width: dict[int, list[tuple[Any, ...]]] = {}

    def add(self, cells: tuple[Any, ...]) -> None:
        self.widths.add(len(cells))
        while len(self.by_position) < len(cells):
            self.by_position.append(set())
        for index, value in enumerate(cells):
            if value is not None and len(self.by_position[index]) < _SAMPLE_CAP:
                self.by_position[index].add(value)
        if self.kind == "sessions":
            self.rows_by_width.setdefault(len(cells), []).append(cells)


def infer_physical_layouts(evidence: LayoutEvidence, dest_types: dict[str, str]) -> dict[int, list[Optional[str]]]:
    """Infer which source column each record position holds, per field count.

    Salvaged records carry no schema. Every layout a real store can have is,
    however, a known chain over the shipped schema history (see
    ``session_schema_history``): the declared order at creation plus every
    later column appended at upgrade time. This walks that graph, pruning
    every branch whose column names contradict the salvaged cells (sentinel
    columns must look like what they name; every cell must fit its column's
    type affinity), and keeps the layouts that fit ALL records at once — the
    whole population of a table was written by one store, so one physical
    order produced every record of a given width.

    Returns, for each distinct record width, the column name each position
    resolves to. A position is ``None`` when the surviving layouts disagree
    on it (the cells there did not discriminate — typically all NULL/zero
    counters) so the caller can leave that column alone rather than guess.
    Returns an empty dict when no known layout fits the records.
    """

    kind = evidence.kind
    if kind not in SCHEMA_HISTORY or not evidence.widths:
        return {}
    by_position = evidence.by_position
    verdicts: dict[tuple[str, Any], bool] = {}

    def fits(name: str, value: Any) -> bool:
        key = (name, value)
        verdict = verdicts.get(key)
        if verdict is None:
            verdict = verdicts[key] = _cell_fits(kind, name, value, dest_types)
        return verdict

    # The invariant verdict depends only on where a candidate puts the two handoff columns, and most
    # candidates of one width agree on that — memoise so the rows are not rescanned per candidate.
    invariant_verdicts: dict[tuple[int, Optional[int], Optional[int]], bool] = {}

    def invariants_hold(layout: tuple[str, ...]) -> bool:
        same_width = evidence.rows_by_width.get(len(layout))
        if not same_width:
            return True
        positions = {name: index for index, name in enumerate(layout)}
        key = (len(layout), positions.get("handoff_error"), positions.get("handoff_state"))
        verdict = invariant_verdicts.get(key)
        if verdict is None:
            verdict = invariant_verdicts[key] = _row_invariants_hold(kind, layout, same_width)
        return verdict

    def accept(layout: tuple[str, ...], first_new: int) -> bool:
        for index in range(first_new, min(len(layout), len(by_position))):
            name = layout[index]
            for value in by_position[index]:
                if not fits(name, value):
                    return False
        return invariants_hold(layout)

    # Enumerate once and bucket by width. A record of width ``k`` was written
    # while the table had exactly ``k`` columns (ADD COLUMN runs at startup,
    # before any row is written), so its layout is a chain state of exactly
    # that length.
    survivors_by_width: dict[int, list[tuple[str, ...]]] = {w: [] for w in sorted(evidence.widths)}
    for layout in reachable_physical_layouts(kind, accept):
        bucket = survivors_by_width.get(len(layout))
        if bucket is not None and layout not in bucket:
            bucket.append(layout)

    result: dict[int, list[Optional[str]]] = {}
    for width, survivors in survivors_by_width.items():
        if not survivors:
            continue
        # Per-position consensus. Where the survivors disagree the cells did
        # not discriminate (they fit every candidate column), so no name is
        # evidence — the destination's own declared order included. Leave
        # such positions alone rather than guess; the cells there are, by
        # construction, values several columns could legitimately hold.
        consensus: list[Optional[str]] = []
        for index in range(width):
            names = {layout[index] for layout in survivors}
            consensus.append(names.pop() if len(names) == 1 else None)
        result[width] = consensus
    return result


def _insert_named_row(
    dest: sqlite3.Connection,
    table: str,
    layout: Sequence[Optional[str]],
    cells: tuple[Any, ...],
    dest_columns: list[str],
    notnull_substitutes: dict[int, Any],
    overrides: Optional[dict[str, Any]] = None,
) -> bool:
    """INSERT salvaged cells by source column name, never by position.

    ``layout[i]`` names the source column of ``cells[i]``; ``None`` positions
    (ambiguous or since-removed columns) are skipped and take the destination
    default. ``notnull_substitutes`` is index-keyed on ``dest_columns`` as
    returned by ``_notnull_defaults``.
    """

    dest_index = {name: index for index, name in enumerate(dest_columns)}
    mapped: dict[str, Any] = {}
    for name, value in zip(layout, cells):
        index = dest_index.get(name) if name is not None else None
        if index is None:
            continue
        if value is None and index in notnull_substitutes:
            value = notnull_substitutes[index]
        mapped[name] = value
    # NOT NULL columns the layout could not place (ambiguous or absent) take
    # the same substitute the positional path uses, so the row still lands;
    # the recovery verifier's plausibility gate audits the result.
    for index, substitute in notnull_substitutes.items():
        mapped.setdefault(dest_columns[index], substitute)
    mapped.update(overrides or {})
    return _execute_insert(dest, table, list(mapped), list(mapped.values()))


def _execute_insert(
    dest: sqlite3.Connection, table: str, columns: list[str], values: list[Any]
) -> bool:
    quoted, placeholders = _quoted_columns(columns)
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
        "legacy_minimal_sessions": 0, "mapped_by_layout": 0, "unrecognized_layout_rows": 0,
        "unrecognized_layout_widths": {}, "inferred_layouts": {},
        "unmapped_rows": 0, "insert_conflicts": 0, "lost_and_found_tables": [],
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
        dest_types = {table: _declared_types(dest, table) for table in targets}
        lf_tables = [
            str(row[0]) for row in
            lf_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'lost_and_found%'")
        ]
        report["lost_and_found_tables"] = lf_tables

        def records():
            """Yield (kind, lf_rowid, nfield, cells) for every classifiable lost_and_found row."""
            for lf_table in lf_tables:
                if _table_columns(lf_conn, lf_table)[:3] != ["rootpgno", "pgno", "nfield"]:
                    continue
                for row in lf_conn.execute(f'SELECT * FROM "{lf_table}"'):
                    try:
                        nfield = int(row[2]) if row[2] is not None else 0
                    except (TypeError, ValueError):
                        yield None, None, 0, ()
                        continue
                    cells = tuple(row[4 : 4 + max(nfield, 0)])
                    yield classify_lost_and_found_row(nfield, cells), row[3], nfield, cells

        # Pass 1: stream the population once, keeping only what layout inference needs. The physical
        # layout is a property of the whole population (one store wrote all of them), so it is inferred
        # once per kind, not per row.
        evidence = {kind: LayoutEvidence(kind) for kind in targets}
        for kind, _, _, cells in records():
            if kind is None:
                report["unmapped_rows"] += 1
            else:
                evidence[kind].add(cells)
        layouts = {kind: infer_physical_layouts(evidence[kind], dest_types[kind]) for kind in targets}
        # Per kind and record width, the column each position resolved to (None where the surviving
        # layouts disagreed and the cell was left to the destination default).
        report["inferred_layouts"] = {
            kind: {str(width): list(layout) for width, layout in by_width.items()}
            for kind, by_width in layouts.items() if by_width
        }

        # Pass 2: insert. Records whose width resolved to a layout are mapped by column name (#101409);
        # the rest take the historical positional prefix, audited by the recovery verifier's plausibility gate.
        for kind, lf_rowid, nfield, cells in records():
            if kind is None:
                continue  # counted in pass 1
            columns, defaults = targets[kind]
            layout = layouts[kind].get(len(cells))
            legacy_minimal = kind == "sessions" and nfield == SESSIONS_LEGACY_MINIMAL_NFIELD
            if layout is None and not legacy_minimal:
                report["unrecognized_layout_rows"] += 1
                widths = report["unrecognized_layout_widths"].setdefault(kind, [])
                if len(cells) not in widths:
                    widths.append(len(cells))
            try:
                if layout is not None:
                    # messages.id is a rowid alias: NULL in the record, carried by the lost_and_found row id.
                    inserted = _insert_named_row(
                        dest, kind, layout, cells, columns, defaults,
                        {"id": lf_rowid} if kind == "messages" else None,
                    )
                    report["mapped_by_layout"] += int(inserted)
                elif legacy_minimal:
                    # A 14-field record matching no known layout (torn cells, or a pre-history store):
                    # salvage identity + timing rather than guessing 14 positional meanings.
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
