"""SQLite persistence for blackbox per-turn telemetry."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from agent.redact import redact_sensitive_text
from hermes_constants import get_hermes_home
from plugins.blackbox.record import TurnRecord, tools_summary

logger = logging.getLogger(__name__)


def _db_path() -> Path:
    return get_hermes_home() / "blackbox" / "turns.db"


def scrub_and_truncate(text: Any, n: int = 2000) -> str:
    """Redact secrets before truncating persisted text previews."""
    if text is None:
        return ""
    scrubbed = redact_sensitive_text(str(text), force=True)
    return scrubbed[:n]


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS turns (
            turn_id TEXT PRIMARY KEY,
            parent_turn_id TEXT,
            is_subagent INT,
            ts_start REAL,
            ts_end REAL,
            profile TEXT,
            provider TEXT,
            model TEXT,
            platform TEXT,
            chat_id TEXT,
            chat_name TEXT,
            api_calls INT,
            tools TEXT,
            input_tokens INT,
            output_tokens INT,
            cache_read INT,
            cache_write INT,
            reasoning INT,
            context_used INT,
            context_length INT,
            last_cache_read INT,
            last_cache_write INT,
            last_uncached INT,
            comp_sys_tokens INT,
            comp_tool_schema_tokens INT,
            comp_history_tokens INT,
            comp_history_message_count INT,
            comp_tool_result_tokens INT,
            comp_tool_arg_tokens INT,
            comp_tool_result_count INT,
            comp_skills_tokens INT,
            comp_skills_count INT,
            comp_framing_tokens INT,
            comp_calls_json TEXT,
            cost_usd REAL,
            cost_status TEXT,
            cost_uncached_usd REAL,
            cost_cache_read_usd REAL,
            cost_cache_write_usd REAL,
            cost_output_usd REAL,
            interrupted INT,
            alerted INT DEFAULT 0,
            user_text TEXT,
            final_text TEXT
        );

        CREATE TABLE IF NOT EXISTS turn_tool_calls (
            turn_id TEXT,
            seq INT,
            name TEXT,
            args_preview TEXT,
            result_preview TEXT,
            PRIMARY KEY(turn_id, seq)
        );

        CREATE TABLE IF NOT EXISTS last_turn (
            platform TEXT,
            chat_id TEXT,
            turn_id TEXT,
            PRIMARY KEY(platform, chat_id)
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_blackbox_turns_chat_end
            ON turns(platform, chat_id, ts_end);
        CREATE INDEX IF NOT EXISTS idx_blackbox_turns_cost
            ON turns(cost_usd);
        """
    )
    # Additive migration for DBs created before the last-call cache split
    # columns existed. CREATE TABLE IF NOT EXISTS won't add columns to an
    # existing table, so ALTER each missing one. Guarded per-column so a
    # partially-migrated DB (or a second writer that already added them)
    # never raises "duplicate column name".
    _existing = {row[1] for row in conn.execute("PRAGMA table_info(turns)").fetchall()}
    for _col in ("last_cache_read", "last_cache_write", "last_uncached"):
        if _col not in _existing:
            try:
                conn.execute(f"ALTER TABLE turns ADD COLUMN {_col} INT")
            except sqlite3.OperationalError:
                pass  # raced with another writer; column now exists
    # Request-composition columns (fixed vs non-fixed breakdown of the final
    # call). Same guarded additive pattern. INT for the token buckets, TEXT for
    # the per-call composition JSON blob.
    for _col in (
        "comp_sys_tokens", "comp_tool_schema_tokens", "comp_history_tokens",
        "comp_history_message_count",
        "comp_tool_result_tokens", "comp_tool_arg_tokens", "comp_tool_result_count",
        "comp_skills_tokens", "comp_framing_tokens",
        "comp_skills_count",
    ):
        if _col not in _existing:
            try:
                conn.execute(f"ALTER TABLE turns ADD COLUMN {_col} INT")
            except sqlite3.OperationalError:
                pass
    if "comp_calls_json" not in _existing:
        try:
            conn.execute("ALTER TABLE turns ADD COLUMN comp_calls_json TEXT")
        except sqlite3.OperationalError:
            pass
    # Per-class cost columns (SPEC-C). REAL, nullable. Same guarded additive
    # pattern so a DB created before these existed gains them on next open, and
    # a partially-migrated / concurrently-written DB never raises.
    for _col in (
        "cost_uncached_usd", "cost_cache_read_usd",
        "cost_cache_write_usd", "cost_output_usd",
    ):
        if _col not in _existing:
            try:
                conn.execute(f"ALTER TABLE turns ADD COLUMN {_col} REAL")
            except sqlite3.OperationalError:
                pass
    conn.commit()


def _int(value: Any) -> int:
    return int(value or 0)


def _int_or_none(value: Any) -> int | None:
    """Preserve NULL for columns that are genuinely absent (old rows / no data).

    Unlike ``_int`` (which coerces None→0), this keeps None as SQL NULL so a
    missing last-call split reads back as None and the renderer can fall back to
    the plain Context line instead of showing a misleading ``0`` split.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_int(value: Any) -> int:
    return 1 if bool(value) else 0


def _cost_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    data["is_subagent"] = bool(data.get("is_subagent"))
    data["interrupted"] = bool(data.get("interrupted"))
    data["alerted"] = bool(data.get("alerted"))
    data["cache_read_tokens"] = data.pop("cache_read")
    data["cache_write_tokens"] = data.pop("cache_write")
    data["reasoning_tokens"] = data.pop("reasoning")
    # Last-call cache split — preserve None (old rows predate these columns).
    # Expose under the _tokens names for renderer consistency, keeping the raw
    # column keys too so direct SELECT * consumers (e.g. /context) still match.
    data["last_cache_read_tokens"] = data.get("last_cache_read")
    data["last_cache_write_tokens"] = data.get("last_cache_write")
    data["last_uncached_tokens"] = data.get("last_uncached")
    try:
        data["tools"] = json.loads(data.get("tools") or "[]")
    except json.JSONDecodeError:
        data["tools"] = []
    data["tools_summary"] = tools_summary(data["tools"])
    return data


def insert_turn(record: TurnRecord) -> None:
    """Persist one turn. Telemetry failures are logged but never raised."""
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO turns (
                    turn_id, parent_turn_id, is_subagent, ts_start, ts_end,
                    profile, provider, model, platform, chat_id, chat_name,
                    api_calls, tools, input_tokens, output_tokens, cache_read,
                    cache_write, reasoning, context_used, context_length,
                    last_cache_read, last_cache_write, last_uncached,
                    comp_sys_tokens, comp_tool_schema_tokens, comp_history_tokens,
                    comp_history_message_count,
                    comp_tool_result_tokens, comp_tool_arg_tokens, comp_tool_result_count,
                    comp_skills_tokens, comp_framing_tokens,
                    comp_skills_count,
                    comp_calls_json,
                    cost_usd, cost_status,
                    cost_uncached_usd, cost_cache_read_usd,
                    cost_cache_write_usd, cost_output_usd,
                    interrupted, alerted, user_text,
                    final_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.turn_id,
                    record.parent_turn_id,
                    _bool_int(record.is_subagent),
                    float(record.ts_start or 0.0),
                    float(record.ts_end or 0.0),
                    record.profile or "",
                    record.provider or "",
                    record.model or "",
                    record.platform or "",
                    record.chat_id or "",
                    record.chat_name or "",
                    _int(record.api_calls),
                    json.dumps(list(record.tools or [])),
                    _int(record.input_tokens),
                    _int(record.output_tokens),
                    _int(record.cache_read_tokens),
                    _int(record.cache_write_tokens),
                    _int(record.reasoning_tokens),
                    _int(record.context_used),
                    _int(record.context_length),
                    _int_or_none(record.last_cache_read_tokens),
                    _int_or_none(record.last_cache_write_tokens),
                    _int_or_none(record.last_uncached_tokens),
                    _int_or_none(record.comp_sys_tokens),
                    _int_or_none(record.comp_tool_schema_tokens),
                    _int_or_none(record.comp_history_tokens),
                    _int_or_none(record.comp_history_message_count),
                    _int_or_none(record.comp_tool_result_tokens),
                    _int_or_none(record.comp_tool_arg_tokens),
                    _int_or_none(record.comp_tool_result_count),
                    _int_or_none(record.comp_skills_tokens),
                    _int_or_none(record.comp_framing_tokens),
                    _int_or_none(record.comp_skills_count),
                    record.comp_calls_json,
                    _cost_float(record.cost_usd),
                    record.cost_status or "unknown",
                    _cost_float(record.cost_uncached_usd),
                    _cost_float(record.cost_cache_read_usd),
                    _cost_float(record.cost_cache_write_usd),
                    _cost_float(record.cost_output_usd),
                    _bool_int(record.interrupted),
                    _bool_int(record.alerted),
                    scrub_and_truncate(record.user_text),
                    scrub_and_truncate(record.final_text),
                ),
            )
            conn.execute("DELETE FROM turn_tool_calls WHERE turn_id = ?", (record.turn_id,))
            for seq, call in enumerate(record.tool_calls or []):
                conn.execute(
                    """
                    INSERT INTO turn_tool_calls (
                        turn_id, seq, name, args_preview, result_preview
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.turn_id,
                        seq,
                        str(call.get("name", "")),
                        scrub_and_truncate(call.get("args_preview", "")),
                        scrub_and_truncate(call.get("result_preview", "")),
                    ),
                )
            conn.execute(
                """
                INSERT INTO last_turn(platform, chat_id, turn_id)
                VALUES (?, ?, ?)
                ON CONFLICT(platform, chat_id) DO UPDATE SET turn_id = excluded.turn_id
                """,
                (record.platform or "", record.chat_id or "", record.turn_id),
            )
    except Exception:
        logger.warning("blackbox telemetry insert failed", exc_info=True)


def mark_alerted(turn_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE turns SET alerted = 1 WHERE turn_id = ? AND alerted = 0",
            (turn_id,),
        )
        return cur.rowcount == 1


def get_turn(turn_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
    return _row_to_dict(row)


def get_tool_calls(turn_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT seq, name, args_preview, result_preview
            FROM turn_tool_calls
            WHERE turn_id = ?
            ORDER BY seq
            """,
            (turn_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_last_turn(platform: str, chat_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT t.*
            FROM last_turn lt
            JOIN turns t ON t.turn_id = lt.turn_id
            WHERE lt.platform = ? AND lt.chat_id = ?
            """,
            (platform or "", chat_id or ""),
        ).fetchone()
    return _row_to_dict(row)


def session_rollup(platform: str, chat_id: str, limit: int = 50) -> dict[str, Any]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT turn_id, cost_usd, is_subagent
            FROM turns
            WHERE platform = ? AND chat_id = ?
            ORDER BY ts_end DESC
            LIMIT ?
            """,
            (platform or "", chat_id or "", int(limit)),
        ).fetchall()
    costs = [float(row["cost_usd"] or 0.0) for row in rows]
    total = sum(costs)
    max_row = max(rows, key=lambda row: float(row["cost_usd"] or 0.0), default=None)
    # Split main vs subagent turns so the caller can show an honest breakdown
    # (the total already INCLUDES subagent spend — they are real rows here).
    sub_rows = [r for r in rows if int(r["is_subagent"] or 0) == 1]
    sub_total = sum(float(r["cost_usd"] or 0.0) for r in sub_rows)
    return {
        "total_usd": total,
        "count": len(rows),
        "avg_usd": total / len(rows) if rows else 0.0,
        "max_turn": dict(max_row) if max_row else None,
        "subagent_count": len(sub_rows),
        "subagent_usd": sub_total,
    }


def top_turns(n: int = 5, since_days: int = 30) -> list[dict[str, Any]]:
    cutoff = time.time() - (int(since_days) * 86400)
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM turns
            WHERE ts_end >= ?
            ORDER BY cost_usd DESC
            LIMIT ?
            """,
            (cutoff, int(n)),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def subagent_rollup(platform: str, chat_id: str, limit: int = 200) -> dict[str, Any]:
    """Aggregate subagent turns spawned within a channel (platform + chat_id).

    Subagent turns are recorded as their own rows with ``is_subagent = 1`` and
    carry the PARENT's channel identity (``platform``/``chat_id`` are stamped
    from ``_blackbox_parent_platform``/``_blackbox_parent_chat_id`` by
    delegate_tool). They are NOT reliably linkable to a single parent *turn* —
    ``parent_turn_id`` holds the parent's session KEY, and parent turns don't
    store their own session key — so we roll up by channel, the same axis every
    other /cost view (session/latest/turn) resolves on.

    Returns counts + summed cost/tokens across the channel's subagent turns,
    plus how many are unpriced (cost_usd IS NULL) so the caller can show an
    honest "+N unpriced" note instead of silently undercounting.
    """
    if not platform or not chat_id:
        return {"count": 0, "total_usd": 0.0, "unpriced": 0,
                "input_tokens": 0, "output_tokens": 0, "models": []}
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT cost_usd, input_tokens, output_tokens, cache_read, model
            FROM turns
            WHERE is_subagent = 1 AND platform = ? AND chat_id = ?
            ORDER BY ts_end DESC
            LIMIT ?
            """,
            (platform or "", chat_id or "", int(limit)),
        ).fetchall()
    total = 0.0
    unpriced = 0
    in_tok = 0
    out_tok = 0
    models: list[str] = []
    for r in rows:
        c = r["cost_usd"]
        if c is None:
            unpriced += 1
        else:
            total += float(c or 0.0)
        in_tok += int(r["input_tokens"] or 0)
        out_tok += int(r["output_tokens"] or 0)
        m = r["model"]
        if m and m not in models:
            models.append(str(m))
    return {
        "count": len(rows),
        "total_usd": total,
        "unpriced": unpriced,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "models": models,
    }


def sweep(retention_days: int, max_deletes: int = 10000) -> int:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    cutoff = time.time() - (int(retention_days) * 86400)
    max_deletes = max(0, int(max_deletes))
    if max_deletes == 0:
        return 0

    with _connect() as conn:
        last_sweep = conn.execute(
            "SELECT value FROM meta WHERE key = 'last_sweep_date'"
        ).fetchone()
        if last_sweep and last_sweep["value"] == today:
            return 0

        rows = conn.execute(
            """
            SELECT turn_id
            FROM turns
            WHERE ts_end < ?
            ORDER BY ts_end
            LIMIT ?
            """,
            (cutoff, max_deletes),
        ).fetchall()
        turn_ids = [row["turn_id"] for row in rows]
        if turn_ids:
            placeholders = ",".join("?" for _ in turn_ids)
            conn.execute(
                f"DELETE FROM turn_tool_calls WHERE turn_id IN ({placeholders})",
                turn_ids,
            )
            conn.execute(
                f"DELETE FROM turns WHERE turn_id IN ({placeholders})",
                turn_ids,
            )
            conn.execute(
                f"DELETE FROM last_turn WHERE turn_id IN ({placeholders})",
                turn_ids,
            )
        deleted = len(turn_ids)
        # Atomic: deletes + sentinel commit together so a crash can't leave the
        # rows deleted without the sentinel (or vice-versa). The sentinel is
        # written BEFORE the single commit; sqlite3's context manager commits on
        # clean exit and rolls back on exception.
        conn.execute(
            """
            INSERT INTO meta(key, value)
            VALUES ('last_sweep_date', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (today,),
        )
        conn.commit()
        return deleted


def debug_stats() -> dict[str, Any]:
    """Operational snapshot for the /cost debug command.

    Surfaces the on-disk DB path, whether it exists, row/side-table counts,
    alerted count, oldest/newest turn timestamps, and the last sweep date.
    Read-only; never raises — returns an ``error`` key on failure so the
    debug command can show *why* telemetry looks empty.
    """
    path = _db_path()
    out: dict[str, Any] = {
        "db_path": str(path),
        "db_exists": path.exists(),
        "db_size_bytes": path.stat().st_size if path.exists() else 0,
    }
    try:
        with _connect() as conn:
            out["turns"] = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            out["tool_calls"] = conn.execute(
                "SELECT COUNT(*) FROM turn_tool_calls"
            ).fetchone()[0]
            out["alerted"] = conn.execute(
                "SELECT COUNT(*) FROM turns WHERE alerted = 1"
            ).fetchone()[0]
            out["subagent_turns"] = conn.execute(
                "SELECT COUNT(*) FROM turns WHERE is_subagent = 1"
            ).fetchone()[0]
            row = conn.execute(
                "SELECT MIN(ts_end), MAX(ts_end) FROM turns"
            ).fetchone()
            out["oldest_ts"] = row[0]
            out["newest_ts"] = row[1]
            sweep_row = conn.execute(
                "SELECT value FROM meta WHERE key = 'last_sweep_date'"
            ).fetchone()
            out["last_sweep_date"] = sweep_row["value"] if sweep_row else None
    except Exception as exc:  # pragma: no cover - defensive
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out
