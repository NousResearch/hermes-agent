"""token-consumption-tracker plugin.

Hooks into ``post_api_request`` to record per-request token consumption
(prompt_tokens, completion_tokens, total_tokens, model, provider, session_id)
into a local SQLite database at ``~/.hermes/personal/token-usage.db``.

Provides ``generate_report()`` for daily/weekly summary generation.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---- paths ------------------------------------------------------------------

def _read_data_dir_from_observability_config(obs: Any) -> str | None:
    """Extract ``data_dir`` from an ``observability`` config dict.

    Recognises two structures (in priority order):
      observability:
        token-consumption-tracker:
          data_dir: <path>          # 1. Plugin-specific (highest in-file)
        default:
          data_dir: <path>          # 2. All-plugins default

    Returns ``None`` when no ``data_dir`` is found.
    """
    if not obs or not isinstance(obs, dict):
        return None

    # 1. Plugin-specific override (new format)
    plugin_cfg = obs.get("token-consumption-tracker")
    if isinstance(plugin_cfg, dict):
        val = plugin_cfg.get("data_dir")
        if val and isinstance(val, str):
            return val

    # 2. All-plugins default (new format)
    default_cfg = obs.get("default")
    if isinstance(default_cfg, dict):
        val = default_cfg.get("data_dir")
        if val and isinstance(val, str):
            return val

    return None


def _read_config_yaml(config_path: Path | None) -> dict:
    """Safely read and parse a YAML config file. Returns empty dict on failure."""
    if not config_path or not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _resolve_data_dir() -> Path:
    """Resolve the observability data directory with multi-layer priority.

    Priority chain (highest to lowest):
      1. ``TOKEN_CONSUMPTION_DATA_DIR`` env var (plugin-specific)
      2. ``OBSERVABILITY_DATA_DIR`` env var (generic)
      3. Per-profile config:  ``observability.token-consumption-tracker.data_dir``
      4. Per-profile config:  ``observability.default.data_dir``
      5. Global config (from hermes root): same structure as steps 3-4
      6. Fallback:  ``~/.hermes``
    """
    # 1-2. Env var overrides
    env_val = os.environ.get("TOKEN_CONSUMPTION_DATA_DIR", "").strip()
    if env_val:
        return Path(env_val).expanduser()
    env_val = os.environ.get("OBSERVABILITY_DATA_DIR", "").strip()
    if env_val:
        return Path(env_val).expanduser()

    # 3-5. Per-profile config (from ``HERMES_HOME``)
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    profile_config_path = Path(hermes_home) / "config.yaml" if hermes_home else None
    data_dir = None
    if profile_config_path:
        config = _read_config_yaml(profile_config_path)
        data_dir = _read_data_dir_from_observability_config(
            config.get("observability")
        )

    # 6. Global config (from hermes root — shared by all profiles)
    if not data_dir:
        # Avoid re-reading the same file when ``HERMES_HOME`` *is* the root
        try:
            from hermes_constants import get_default_hermes_root
        except ImportError:
            get_default_hermes_root = None

        if get_default_hermes_root is not None:
            global_config_path = get_default_hermes_root() / "config.yaml"
            if (profile_config_path is None
                    or global_config_path.resolve() != profile_config_path.resolve()):
                config = _read_config_yaml(global_config_path)
                data_dir = _read_data_dir_from_observability_config(
                    config.get("observability")
                )

    # 7. Hard-coded fallback
    if not data_dir:
        data_dir = "~/.hermes"

    return Path(data_dir).expanduser()


_HERMES_PERSONAL = _resolve_data_dir()
_DB_PATH = _HERMES_PERSONAL / "token-usage.db"
_REPORT_DIR = _HERMES_PERSONAL / "token-usage"

# ---- thread-safe write queue ------------------------------------------------
# Post_API_request fires synchronously in the LLM call path — we must NOT block
# the caller with a SQLite write.  Queue the record and flush asynchronously.

_lock = threading.Lock()
_queue: list[dict[str, Any]] = []
_flush_timer: threading.Timer | None = None
_FLUSH_INTERVAL = 3.0  # flush every 3 seconds (or on session end)


def _init_db() -> None:
    """Create the database and table if they don't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS token_usage (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                turn_id     TEXT,
                api_request_id TEXT,
                model       TEXT NOT NULL,
                provider    TEXT NOT NULL,
                profile     TEXT DEFAULT '',
                prompt_tokens   INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                cache_read_tokens  INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                workspace       TEXT DEFAULT '',
                worker          TEXT DEFAULT '',
                total_tokens    INTEGER DEFAULT 0,
                api_duration    REAL DEFAULT 0.0,
                finish_reason   TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_token_usage_created_at
            ON token_usage(created_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_token_usage_session
            ON token_usage(session_id)
            """
        )
        # Add columns if missing (schema migration)
        for col, col_type in (("raw_usage", "TEXT"), ("cache_read_tokens", "INTEGER DEFAULT 0"),
                               ("cache_write_tokens", "INTEGER DEFAULT 0"),
                               ("workspace", "TEXT DEFAULT ''"), ("worker", "TEXT DEFAULT ''"),
                               ("profile", "TEXT DEFAULT ''")):
            try:
                conn.execute(f"ALTER TABLE token_usage ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass
        conn.commit()
    finally:
        conn.close()


def _flush_queue() -> None:
    """Write all queued records to the database in a single transaction."""
    global _flush_timer
    _flush_timer = None

    with _lock:
        records = list(_queue)
        _queue.clear()

    if not records:
        return

    try:
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            conn.executemany(
                """
                INSERT INTO token_usage
                    (session_id, turn_id, api_request_id, model, provider, profile,
                     prompt_tokens, completion_tokens, total_tokens,
                     cache_read_tokens, cache_write_tokens,
                     workspace, worker,
                     api_duration, finish_reason, created_at, raw_usage)
                VALUES
                    (:session_id, :turn_id, :api_request_id, :model, :provider, :profile,
                     :prompt_tokens, :completion_tokens, :total_tokens,
                     :cache_read_tokens, :cache_write_tokens,
                     :workspace, :worker,
                     :api_duration, :finish_reason, :created_at, :raw_usage)
                """,
                records,
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.warning("token-consumption-tracker: _flush_queue failed", exc_info=True)


def _schedule_flush() -> None:
    """Start or restart the async flush timer."""
    global _flush_timer

    if _flush_timer is not None and _flush_timer.is_alive():
        # Already scheduled — cancel and restart
        _flush_timer.cancel()

    _flush_timer = threading.Timer(_FLUSH_INTERVAL, _flush_queue)
    _flush_timer.daemon = True
    _flush_timer.start()


def flush_now() -> None:
    """Force an immediate flush.  Safe to call from any thread."""
    if _flush_timer is not None:
        _flush_timer.cancel()
    _flush_queue()


# ---- hook handler -----------------------------------------------------------


def _has_token_usage(usage: Any) -> bool:
    """Check if the usage dict has meaningful token counts."""
    if usage is None:
        return False
    if isinstance(usage, dict):
        return bool(usage.get("prompt_tokens") or usage.get("total_tokens"))
    if isinstance(usage, (int, float)):
        return usage > 0
    return True


def _on_post_api_request(**kw: Any) -> None:
    """Record token usage per API request."""
    usage: Any = kw.get("usage")

    if not _has_token_usage(usage):
        return
    # After _has_token_usage returns True, usage is guaranteed non-None
    usage_dict: dict = usage if isinstance(usage, dict) else {}

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record = {
        "session_id": kw.get("session_id", ""),
        "turn_id": kw.get("turn_id", ""),
        "api_request_id": kw.get("api_request_id", ""),
        "model": kw.get("model", "unknown"),
        "provider": kw.get("provider", "unknown"),
        "profile": kw.get("profile", os.environ.get("HERMES_PROFILE", "")),
        "prompt_tokens": int(usage_dict.get("prompt_tokens", 0)),
        "completion_tokens": int(usage_dict.get("output_tokens", 0)),
        "total_tokens": int(usage_dict.get("total_tokens", 0)),
        "cache_read_tokens": int(usage_dict.get("cache_read_tokens", 0)),
        "cache_write_tokens": int(usage_dict.get("cache_write_tokens", 0)),
        "workspace": os.environ.get("HERMES_KANBAN_WORKSPACE", ""),
        "worker": os.environ.get("HERMES_KANBAN_TASK", ""),
        "api_duration": float(kw.get("api_duration", 0)),
        "finish_reason": kw.get("finish_reason", ""),
        "created_at": now,
        "raw_usage": json.dumps(usage_dict, ensure_ascii=False),
    }

    with _lock:
        _queue.append(record)

    _schedule_flush()


def _on_session_end(**kw: Any) -> None:
    """Flush any remaining records when the session ends."""
    flush_now()


# ---- report generation ------------------------------------------------------


def _db_connect() -> sqlite3.Connection:
    """Open a read-only connection to the token usage DB."""
    _init_db()
    return sqlite3.connect(str(_DB_PATH))


def generate_report(date_str: str | None = None) -> str:
    """Generate a daily usage report in markdown format.

    Args:
        date_str: Date in ``YYYY-MM-DD`` format.  Defaults to yesterday.

    Returns:
        Markdown report string.
    """
    # Ensure any pending records are written before querying
    flush_now()

    if date_str is None:
        date_str = (
            datetime.datetime.now() - datetime.timedelta(days=1)
        ).strftime("%Y-%m-%d")

    conn = _db_connect()
    try:
        cursor = conn.cursor()

        # ---- totals (with cache) ----
        cursor.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(prompt_tokens), 0),
                   COALESCE(SUM(completion_tokens), 0),
                   COALESCE(SUM(total_tokens), 0),
                   COALESCE(SUM(cache_read_tokens), 0),
                   COALESCE(SUM(cache_write_tokens), 0)
            FROM token_usage
            WHERE created_at >= ? AND created_at < ?
            """,
            (f"{date_str} 00:00:00", f"{date_str} 23:59:59"),
        )
        total_count, total_prompt, total_completion, total_all, total_cache_read, total_cache_write = cursor.fetchone()

        # ---- by model ----
        cursor.execute(
            """
            SELECT model,
                   COUNT(*),
                   SUM(prompt_tokens),
                   SUM(completion_tokens),
                   SUM(total_tokens),
                   ROUND(AVG(api_duration), 2)
            FROM token_usage
            WHERE created_at >= ? AND created_at < ?
            GROUP BY model
            ORDER BY SUM(total_tokens) DESC
            """,
            (f"{date_str} 00:00:00", f"{date_str} 23:59:59"),
        )
        by_model = cursor.fetchall()

        # ---- by session ----
        cursor.execute(
            """
            SELECT session_id,
                   COUNT(*),
                   SUM(prompt_tokens),
                   SUM(completion_tokens),
                   SUM(total_tokens)
            FROM token_usage
            WHERE created_at >= ? AND created_at < ?
            GROUP BY session_id
            ORDER BY SUM(total_tokens) DESC
            LIMIT 10
            """,
            (f"{date_str} 00:00:00", f"{date_str} 23:59:59"),
        )
        by_session = cursor.fetchall()

        # ---- by provider ----
        cursor.execute(
            """
            SELECT provider,
                   COUNT(*),
                   SUM(total_tokens)
            FROM token_usage
            WHERE created_at >= ? AND created_at < ?
            GROUP BY provider
            ORDER BY SUM(total_tokens) DESC
            """,
            (f"{date_str} 00:00:00", f"{date_str} 23:59:59"),
        )
        by_provider = cursor.fetchall()

        # ---- hourly breakdown ----
        cursor.execute(
            """
            SELECT strftime('%H', created_at) as hour,
                   COUNT(*),
                   SUM(total_tokens)
            FROM token_usage
            WHERE created_at >= ? AND created_at < ?
            GROUP BY hour
            ORDER BY hour
            """,
            (f"{date_str} 00:00:00", f"{date_str} 23:59:59"),
        )
        by_hour = cursor.fetchall()

    finally:
        conn.close()

    # ---- build report ----
    lines: list[str] = []
    lines.append(f"# Token Usage Report — {date_str}")
    lines.append("")

    # Summary header
    lines.append("## Daily Summary")
    lines.append("")
    actual_input = total_prompt - total_cache_read - total_cache_write
    lines.append(f"| Metric | Value |")
    lines.append(f"|------|------|")
    lines.append(f"| API Requests | {total_count} |")
    lines.append(f"| Input (New) | {actual_input:,} |")
    lines.append(f"| Cache Read | {total_cache_read:,} |")
    if total_cache_write:
        lines.append(f"| Cache Write | {total_cache_write:,} |")
    lines.append(f"| Input (Total) | {total_prompt:,} |")
    lines.append(f"| Output | {total_completion:,} |")
    lines.append(f"| Total | {total_all:,} |")
    if total_count > 0:
        avg_tokens = total_all // total_count
        lines.append(f"| Avg per Request | {avg_tokens:,} |")
    lines.append("")

    # By model
    if by_model:
        lines.append("## By Model")
        lines.append("")
        lines.append("| Model | Requests | Input | Output | Total | Avg Time(s) |")
        lines.append("|------|--------|-------|--------|------|------------|")
        for row in by_model:
            model_name, cnt, inp, out, tot, avg_dur = row
            lines.append(f"| {model_name} | {cnt} | {inp:,} | {out:,} | {tot:,} | {avg_dur} |")
        lines.append("")

    # By provider
    if by_provider:
        lines.append("## By Provider")
        lines.append("")
        lines.append("| Provider | Requests | Total Tokens |")
        lines.append("|----------|--------|-----------|")
        for row in by_provider:
            prov, cnt, tot = row
            lines.append(f"| {prov} | {cnt} | {tot:,} |")
        lines.append("")

    # By session
    if by_session:
        lines.append("## By Session (Top 10)")
        lines.append("")
        lines.append("| Session | Requests | Input | Output | Total |")
        lines.append("|---------|--------|-------|--------|------|")
        for row in by_session:
            sid, cnt, inp, out, tot = row
            short_sid = sid[:16] + "..." if len(sid) > 20 else sid
            lines.append(f"| {short_sid} | {cnt} | {inp:,} | {out:,} | {tot:,} |")
        lines.append("")

    # Hourly
    if by_hour:
        lines.append("## Hourly Distribution")
        lines.append("")
        lines.append("| Hour | Requests | Tokens |")
        lines.append("|------|--------|--------|")
        for row in by_hour:
            hour, cnt, tot = row
            lines.append(f"| {hour}:00 | {cnt} | {tot:,} |")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by token-consumption-tracker at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    return "\n".join(lines)


def save_report_to_file(date_str: str | None = None) -> str:
    """Generate and save a daily report.

    Returns the file path of the saved report.
    """
    if date_str is None:
        date_str = (
            datetime.datetime.now() - datetime.timedelta(days=1)
        ).strftime("%Y-%m-%d")

    report = generate_report(date_str)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _REPORT_DIR / f"{date_str}.md"
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)


# ---- slash command -----------------------------------------------------------


def _resolve_date(args: str, rest: str) -> tuple[str | None, str | None]:
    """Parse date argument from rest of args.

    Returns ``(date_str, error_msg)`` — one is None.
    """
    rest = rest.strip()
    if not rest:
        # Default: today
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d"), None
    if rest == "yesterday":
        return None, None  # generate_report defaults to yesterday
    if rest.startswith("20") and len(rest) >= 10:
        return rest[:10], None
    return None, (
        f"Unrecognized date: {rest!r}\n"
        "  Usage: /token show [yesterday|2026-06-17]\n"
    )


def _handle_slash_command(args: str) -> str:
    """Handle the ``/token`` slash command.

    Subcommands:
      /token list              — list saved reports
      /token show [date]       — generate & print (default today)
      /token save [date]       — generate & save to file (default today)
      /token status            — DB location, size, record count
    """
    parts = args.strip().split(None, 1)
    cmd = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    if cmd == "list":
        return _handle_token_list()

    if cmd == "show":
        date_str, err = _resolve_date(args, rest)
        if err:
            return err
        report = generate_report(date_str)
        if date_str is None:
            from datetime import datetime, timedelta
            date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        return f"# Token Usage Report — {date_str} (generated on-the-fly, not saved)\n\n" + report

    if cmd == "save":
        date_str, err = _resolve_date(args, rest)
        if err:
            return err
        path = save_report_to_file(date_str)
        report = generate_report(date_str)
        if date_str is None:
            from datetime import datetime, timedelta
            date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        return f"Report saved to: `{path}`\n\n" + report

    if cmd == "status":
        return _get_status()

    # Fallback: help
    return (
        "Usage:\n"
        "  /token list              — List saved reports\n"
        "  /token show [yesterday|2026-06-17]  — Generate & display (default: today)\n"
        "  /token save [yesterday|2026-06-17]  — Generate & save (default: today)\n"
        "  /token status            — Database status\n"
    )


def _handle_token_list() -> str:
    """List existing daily report files in the report directory."""
    if not _REPORT_DIR.exists():
        return "Report directory does not exist."

    files = sorted(_REPORT_DIR.glob("20*.md"), reverse=True)
    if not files:
        return "No saved reports."

    lines = [f"Saved reports ({len(files)}):\n"]
    for f in files:
        name = f.stem
        size = f.stat().st_size
        size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
        lines.append(f"  {name}  ({size_str})")

    return "\n".join(lines)


def _get_status() -> str:
    """Return DB status: path, size, record count."""
    _init_db()
    try:
        size = _DB_PATH.stat().st_size
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            count = conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]
            earliest = conn.execute(
                "SELECT MIN(created_at) FROM token_usage"
            ).fetchone()[0] or "—"
            latest = conn.execute(
                "SELECT MAX(created_at) FROM token_usage"
            ).fetchone()[0] or "—"
        finally:
            conn.close()
    except Exception:
        return f"Cannot read database: {_DB_PATH}"

    size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
    return (
        f"**Token Usage Database**\n\n"
        f"| Item | Value |\n"
        f"|------|-----|\n"
        f"| Path | `{_DB_PATH}` |\n"
        f"| Size | {size_str} |\n"
        f"| Records | {count:,} |\n"
        f"| Earliest | {earliest} |\n"
        f"| Latest | {latest} |\n"
    )

# ---- plugin entry point -----------------------------------------------------


def register(ctx: Any) -> None:
    """Register the post_api_request hook for token tracking."""
    _init_db()

    try:
        _log_path = _HERMES_PERSONAL / "logs" / "token-consumption-tracker.log"
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        # Don't configure root logger — just use the module logger
        logger.info("token-consumption-tracker: plugin loaded, DB at %s", _DB_PATH)
    except Exception:
        pass

    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("on_session_end", _on_session_end)

    # ── Slash command: /token ──
    ctx.register_command(
        name="token",
        handler=_handle_slash_command,
        description="Token usage: list/show/save/status",
        args_hint="list | show [date] | save [date] | status",
    )
