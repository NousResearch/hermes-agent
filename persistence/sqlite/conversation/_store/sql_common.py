"""The SQL fragments, builders and statement helpers the store functions share."""


from __future__ import annotations

from persistence.conversation.messages import QUEUED_PROMPT_METADATA_KEY
from persistence.sqlite._schema_ddl import _BRANCH_CHILD_SQL, _RESET_CHILD_SQL, _sql_json_extract, _sql_literal
from persistence.sqlite.conversation._store.rules import (
    AUTO_LOAD_SCAFFOLD_SQL_LIKE, EPOCH_MAX, EPOCH_MIN, SKILL_SCAFFOLD_SQL_LIKE,
)
from session_rules.turns import (
    LEGACY_SUMMARY_PREFIX, SKILL_EXCERPT_JOINT, SUMMARY_PREFIX, _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)


# Session preview = head of the first user message (shown when a session has no title).  A /skill invocation
# embeds the whole skill body, so scaffolded rows take a wider excerpt (whole message under budget, else head +
# tail where the typed instruction lands) and ``_shape_preview`` recovers the typed command from it.
_PREVIEW_HEAD_CHARS = 63


_PREVIEW_SCAFFOLD_WINDOW = 400


def escape_like(text: str) -> str:
    """Escape LIKE wildcards (``%``, ``_``) so derived text matches literally; pair with ``ESCAPE '\\'``.
    ``_`` is common in branch names/titles/paths and a substring match must not silently widen."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_PREVIEW_CONTENT_SQL = "REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' ')"


# User-invoked and gateway auto-load scaffolds both get the head+tail excerpt window:
# the typed request sits at the tail (auto-load) or behind the instruction marker.
# No LIKE wildcards in either prefix, so no ESCAPE clause is needed.
_PREVIEW_SCAFFOLDED_SQL = (
    f"(m.content LIKE '{SKILL_SCAFFOLD_SQL_LIKE}'"
    f" OR m.content LIKE '{AUTO_LOAD_SCAFFOLD_SQL_LIKE}')")


_SQL_WHITESPACE = "CHAR(9) || CHAR(10) || CHAR(13) || CHAR(32)"


def _sql_ltrim_whitespace(expression: str) -> str:
    return f"LTRIM({expression}, {_SQL_WHITESPACE})"


def _sql_trim_whitespace(expression: str) -> str:
    return f"TRIM({expression}, {_SQL_WHITESPACE})"


def _sql_starts_with(expression: str, prefixes: tuple[str, ...]) -> str:
    trimmed = _sql_ltrim_whitespace(expression)
    return "(" + " OR ".join(f"SUBSTR({trimmed}, 1, {len(p)}) = {_sql_literal(p)}" for p in prefixes) + ")"


def _sql_after_marker(marker: str) -> str:
    """``m.content`` after the first occurrence of *marker*."""
    return f"SUBSTR(m.content, INSTR(m.content, {_sql_literal(marker)}) + {len(marker)})"


# Match the whole introduction shared by current and legacy prefixes, so a message that merely starts with the
# bracketed label is not a compaction carrier.
_PREVIEW_LONG_FORM_PREFIX = SUMMARY_PREFIX.split("Do NOT answer", 1)[0]


_PREVIEW_SUMMARY_PREFIXES = (_PREVIEW_LONG_FORM_PREFIX, LEGACY_SUMMARY_PREFIX)


_PREVIEW_STANDALONE_SUMMARY_SQL = _sql_starts_with("m.content", _PREVIEW_SUMMARY_PREFIXES)


_PREVIEW_MERGED_AFTER_SQL = _sql_after_marker(_MERGED_SUMMARY_DELIMITER)


_PREVIEW_MERGED_SUMMARY_SQL = (f"(INSTR(m.content, {_sql_literal(_MERGED_SUMMARY_DELIMITER)}) > 0"
    f" AND {_sql_starts_with(_PREVIEW_MERGED_AFTER_SQL, _PREVIEW_SUMMARY_PREFIXES)})")


_PREVIEW_MERGED_PRIOR_SQL = _sql_trim_whitespace(
    f"SUBSTR(m.content, 1, INSTR(m.content, {_sql_literal(_MERGED_SUMMARY_DELIMITER)}) - 1)")


_PREVIEW_MERGED_PRIOR_LTRIMMED_SQL = _sql_ltrim_whitespace(_PREVIEW_MERGED_PRIOR_SQL)


_PREVIEW_MERGED_PRIOR_UNWRAPPED_SQL = (f"CASE WHEN SUBSTR({_PREVIEW_MERGED_PRIOR_LTRIMMED_SQL}, 1,"
    f" {len(_MERGED_PRIOR_CONTEXT_HEADER)}) = {_sql_literal(_MERGED_PRIOR_CONTEXT_HEADER)}"
    f" THEN {_sql_ltrim_whitespace(f'SUBSTR({_PREVIEW_MERGED_PRIOR_LTRIMMED_SQL}, {len(_MERGED_PRIOR_CONTEXT_HEADER) + 1})')}"
    f" ELSE {_PREVIEW_MERGED_PRIOR_SQL} END")


_PREVIEW_FORCE_USER_REMAINDER_SQL = _sql_after_marker(_SUMMARY_END_MARKER)


# Pure compaction rows are ineligible; force-user-leading and merged carriers only when authentic content survives.
# A display_kind="hidden" row is model-facing scaffolding the gateway never paints; the preview must not paint it either.
_PREVIEW_ELIGIBLE_SQL = (f"(COALESCE(m.display_kind, '') <> 'hidden'"
    f" AND ((NOT {_PREVIEW_STANDALONE_SUMMARY_SQL} AND NOT {_PREVIEW_MERGED_SUMMARY_SQL})"
    f" OR ({_PREVIEW_STANDALONE_SUMMARY_SQL} AND INSTR(m.content, {_sql_literal(_SUMMARY_END_MARKER)}) > 0"
    f" AND LENGTH({_sql_trim_whitespace(_PREVIEW_FORCE_USER_REMAINDER_SQL)}) > 0)"
    f" OR ({_PREVIEW_MERGED_SUMMARY_SQL}"
    f" AND LENGTH({_sql_trim_whitespace(_PREVIEW_MERGED_PRIOR_UNWRAPPED_SQL)}) > 0)))")


# ``_preview_raw`` SELECT for every listing query (scaffolded rows: head + tail around SKILL_EXCERPT_JOINT).
_PREVIEW_RAW_SELECT = (
    f"CASE WHEN {_PREVIEW_STANDALONE_SUMMARY_SQL} THEN {_PREVIEW_FORCE_USER_REMAINDER_SQL}"
    f" WHEN {_PREVIEW_MERGED_SUMMARY_SQL} THEN {_PREVIEW_MERGED_PRIOR_UNWRAPPED_SQL}"
    f" WHEN {_PREVIEW_SCAFFOLDED_SQL} AND LENGTH(m.content) > {_PREVIEW_SCAFFOLD_WINDOW * 2}"
    f" THEN SUBSTR({_PREVIEW_CONTENT_SQL}, 1, {_PREVIEW_SCAFFOLD_WINDOW}) || '{SKILL_EXCERPT_JOINT}'"
    f" || SUBSTR({_PREVIEW_CONTENT_SQL}, -{_PREVIEW_SCAFFOLD_WINDOW})"
    f" WHEN {_PREVIEW_SCAFFOLDED_SQL} THEN SUBSTR({_PREVIEW_CONTENT_SQL}, 1, {_PREVIEW_SCAFFOLD_WINDOW * 2})"
    f" ELSE SUBSTR({_PREVIEW_CONTENT_SQL}, 1, {_PREVIEW_HEAD_CHARS}) END")


def _sql_preview_raw(session_id_expr: str = "s.id") -> str:
    """Correlated ``_preview_raw`` column: the first eligible user message of *session_id_expr*.
    ``INDEXED BY`` walks the session in timestamp order and stops at the first eligible row; with planner
    stats older than idx_messages_session_id, SQLite otherwise reads and sorts the whole session (#119403)."""
    return (f"COALESCE((SELECT {_PREVIEW_RAW_SELECT} FROM messages m INDEXED BY idx_messages_session"
        f" WHERE m.session_id = {session_id_expr} AND m.role = 'user' AND m.content IS NOT NULL"
        f" AND {_PREVIEW_ELIGIBLE_SQL} ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw")


# Accidental end reasons recovery treats as resumable. superseded_by_resume = sentinel-parked runtime replaced
# by a fresh session.resume; startup_orphan_reap = dead-gateway sweep, same class as ws_orphan_reap but kept
# distinct for forensics.
_RECOVERABLE_END_REASONS = ("agent_close", "ws_orphan_reap", "superseded_by_resume", "startup_orphan_reap")


_RECOVERABLE_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in _RECOVERABLE_END_REASONS)


def _non_continuation_child_sql(child: str = "", parent: str = "?") -> str:
    """``  AND ...`` clauses rejecting children that are NOT compression continuations of *parent*
    (branch/delegate/reset forks, tool sessions).  Markers are bound to the parent id: continuations
    inherit ``model_config`` verbatim, so a marker naming another row is inherited, not a fork.
    ``child`` is the column prefix (``""``, ``"c."``); single owner so prune and compression agree."""
    return "".join(
        f"  AND COALESCE({_sql_json_extract(f'{child}model_config', f'$.{marker}')}, '') != {parent}\n"
        for marker in ("_branched_from", "_delegate_from", "_reset_from")
    ) + f"  AND COALESCE({child}source, '') != 'tool'\n"


# Picker-visible rows: roots + branch/reset children (not subagent runs or compression continuations).
_LISTABLE_CHILD_SQL = (f"(s.parent_session_id IS NULL OR {_BRANCH_CHILD_SQL.format(a='s')}"
    f" OR {_RESET_CHILD_SQL.format(a='s')})")


_SQL_IN_WINDOW = f"BETWEEN {EPOCH_MIN!r} AND {EPOCH_MAX!r}"


def _sql_in_window(expr: str) -> str:
    """*expr* when it is inside the ``coerce_epoch`` window, else NULL."""
    return f"(SELECT _win.v FROM (SELECT {expr} AS v) _win WHERE _win.v {_SQL_IN_WINDOW})"


def _sql_freshest_of(activity: str, session_id_expr: str, started: str) -> str:
    """Freshest of *activity* and the latest message timestamp for *session_id_expr*, else *started*.
    Heartbeats are rate-limited (~60s) so ``last_activity_at`` can lag a newer message; never use it alone.
    Cells outside the ``coerce_epoch`` window (garbage doubles salvaged from a damaged page, TEXT) are
    skipped, fallback included, or one bad row pins the session's recency to ``5e+246`` (#91536); a
    session with no trusted cell at all is NULL."""
    # Pinned for the same reason as _sql_preview_raw (#119403).
    msg_max = (f"(SELECT MAX(_act_m.timestamp) FROM messages _act_m INDEXED BY idx_messages_session"
        f" WHERE _act_m.session_id = {session_id_expr}"
        f" AND _act_m.timestamp {_SQL_IN_WINDOW})")
    return (f"COALESCE((SELECT MAX(_act_v.v) FROM (SELECT {activity} AS v UNION ALL SELECT {msg_max}) _act_v"
        f" WHERE _act_v.v {_SQL_IN_WINDOW}), {_sql_in_window(started)})")


def _sql_session_last_active(alias: str = "s") -> str:
    """Session recency expression for a ``sessions {alias}`` row."""
    return _sql_freshest_of(f"{alias}.last_activity_at", f"{alias}.id", f"{alias}.started_at")


def _sql_session_last_active_by_id(session_id_expr: str) -> str:
    """Same freshest-of expression keyed by a session-id SQL expression."""
    return _sql_freshest_of(
        f"(SELECT last_activity_at FROM sessions _act_s WHERE _act_s.id = {session_id_expr})", session_id_expr,
        f"(SELECT started_at FROM sessions _act_s WHERE _act_s.id = {session_id_expr})")


# Cap on user-controlled FTS5 query input before sanitizer processing.
MAX_FTS5_QUERY_CHARS = 2_048


# Row probes the message and compression functions share.
_ENDED_ROW_SQL = "SELECT ended_at, end_reason FROM sessions WHERE id = ?"


_COMPRESSION_LOCK_ROW_SQL = "SELECT holder, expires_at FROM compression_locks WHERE session_id = ?"


def _placeholders(items) -> str:
    """``?,?,?`` for one bound parameter per element of *items* (a sequence or an int count)."""
    return ",".join("?" for _ in range(items if isinstance(items, int) else len(items)))


# A first prompt is durable in ``session_admissions`` from admission on, while its transcript row lands
# only at turn start; a restart in between leaves the admission ``unknown`` and the session at 0
# messages. Only a session with neither messages nor an admission is an empty launch stub.
_NOT_A_LAUNCH_STUB_SQL = (
    "(s.message_count > 0 OR EXISTS (SELECT 1 FROM session_admissions a WHERE a.target_session_id = s.id))")


def launch_stub_filter_sql(db) -> str:
    """``sessions s`` predicate that drops empty launch stubs (the CLI crashed or a resume failed before
    the first turn). A read-only open does not migrate, so a store from before the runtime tables may
    lack ``session_admissions``; messages alone decide there."""
    if db._read_all("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'session_admissions'"):
        return _NOT_A_LAUNCH_STUB_SQL
    return "s.message_count > 0"


def _source_match_clause(sources, local_created_sources=None) -> tuple[str, list]:
    """``sessions s`` source predicate. With ``local_created_sources``, a ``source='local'`` row
    matches only when ``created_source`` is one of those surfaces; the placeholder ``'unknown'``
    is not a surface and stays unmatched until an upsert stamps the real creator."""
    labels = list(sources or [])
    if not labels:
        return "", []
    surfaces = list(local_created_sources or ())
    if not surfaces:
        return f"s.source IN ({_placeholders(labels)})", labels
    non_local = [item for item in labels if item != "local"]
    local_sql = f"(s.source = 'local' AND s.created_source IN ({_placeholders(surfaces)}))"
    if not non_local:
        return local_sql, list(surfaces)
    return (
        f"(s.source IN ({_placeholders(non_local)}) OR {local_sql})",
        [*non_local, *surfaces],
    )


# Ids per ``IN (?,...)`` list: SQLite caps bound parameters at SQLITE_MAX_VARIABLE_NUMBER (999 on builds
# < 3.32, 32766 after); a bulk prune of a cron-heavy store bound tens of thousands of ids into one list and
# died with "too many SQL variables". Every IN-list over session ids goes through ``_id_chunks``.
_SQL_IN_CHUNK = 900


def _id_chunks(ids, size: int = _SQL_IN_CHUNK):
    """Yield *ids* (any iterable) as lists of at most *size* elements."""
    ids = list(ids)
    for start in range(0, len(ids), size):
        yield ids[start:start + size]
