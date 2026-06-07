"""Import legacy Hermes WebUI JSON sessions into the canonical SessionDB.

The standalone ``hermes-webui`` project stored conversations as JSON sidecars
under ``$HERMES_HOME/webui/sessions``.  Hermes Desktop and the dashboard API read
the canonical ``state.db`` instead, so old WebUI conversations are invisible
until they are copied into that store.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home
from hermes_state import SessionDB


VALID_ROLES = {"assistant", "system", "tool", "user"}
MAX_SESSION_ID_LENGTH = 160


@dataclass
class WebUISessionImportIssue:
    path: str
    reason: str
    session_id: str | None = None


@dataclass
class WebUISessionImportReport:
    sessions_dir: str
    dry_run: bool
    profile: str | None = None
    scanned: int = 0
    imported: int = 0
    refreshed: int = 0
    skipped_existing: int = 0
    skipped_empty: int = 0
    skipped_invalid: int = 0
    skipped_foreign_source: int = 0
    skipped_other_profile: int = 0
    errors: list[WebUISessionImportIssue] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return self.imported + self.refreshed


@dataclass
class WebUIProfilesImportReport:
    sessions_dir: str
    dry_run: bool
    reports: list[WebUISessionImportReport] = field(default_factory=list)
    errors: list[WebUISessionImportIssue] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return sum(report.changed for report in self.reports)


def resolve_webui_sessions_dir(
    *,
    state_dir: str | os.PathLike[str] | None = None,
    sessions_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the legacy WebUI ``sessions`` directory."""
    if sessions_dir:
        return Path(sessions_dir).expanduser()
    if state_dir:
        return Path(state_dir).expanduser() / "sessions"
    env_state_dir = os.environ.get("HERMES_WEBUI_STATE_DIR", "").strip()
    base = Path(env_state_dir).expanduser() if env_state_dir else get_hermes_home() / "webui"
    return base / "sessions"


def import_webui_sessions(
    db: SessionDB,
    *,
    state_dir: str | os.PathLike[str] | None = None,
    sessions_dir: str | os.PathLike[str] | None = None,
    dry_run: bool = True,
    include_empty: bool = False,
    profile: str | None = None,
) -> WebUISessionImportReport:
    """Import legacy WebUI session JSON files into ``db``.

    The importer is intentionally conservative:

    * it is dry-run by default;
    * empty sessions are skipped unless ``include_empty`` is true;
    * existing non-WebUI rows are never overwritten;
    * existing WebUI rows are refreshed only when the JSON transcript is longer.
    """
    root = resolve_webui_sessions_dir(state_dir=state_dir, sessions_dir=sessions_dir)
    profile_filter = _normalized_profile(profile)
    report = WebUISessionImportReport(sessions_dir=str(root), dry_run=dry_run, profile=profile_filter)
    if not root.exists() or not root.is_dir():
        report.errors.append(WebUISessionImportIssue(path=str(root), reason="sessions directory not found"))
        return report

    for path in _iter_webui_session_files(root):
        report.scanned += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            report.skipped_invalid += 1
            report.errors.append(WebUISessionImportIssue(path=str(path), reason=f"invalid JSON: {exc}"))
            continue
        if not isinstance(payload, dict):
            report.skipped_invalid += 1
            report.errors.append(WebUISessionImportIssue(path=str(path), reason="session JSON must be an object"))
            continue

        session_id = _session_id_for_payload(path, payload)
        if not _valid_session_id(session_id):
            report.skipped_invalid += 1
            report.errors.append(
                WebUISessionImportIssue(path=str(path), session_id=session_id, reason="invalid session id")
            )
            continue

        payload_profile = _payload_profile(payload)
        if profile_filter and payload_profile != profile_filter:
            report.skipped_other_profile += 1
            continue

        messages = _normalized_messages(payload.get("messages"))
        if not messages and not include_empty:
            report.skipped_empty += 1
            continue

        existing = db.get_session(session_id)
        if existing:
            source = str(existing.get("source") or "").strip().lower()
            if source and source != "webui":
                report.skipped_foreign_source += 1
                continue
            existing_count = _actual_message_count(db, session_id)
            if existing_count >= len(messages):
                report.skipped_existing += 1
                continue
            report.refreshed += 1
            if not dry_run:
                _write_webui_session(db, session_id, payload, messages, existing=existing)
            continue

        report.imported += 1
        if not dry_run:
            _write_webui_session(db, session_id, payload, messages, existing=None)

    return report


def import_webui_sessions_by_profile(
    *,
    state_dir: str | os.PathLike[str] | None = None,
    sessions_dir: str | os.PathLike[str] | None = None,
    dry_run: bool = True,
    include_empty: bool = False,
) -> WebUIProfilesImportReport:
    """Import legacy WebUI sessions into the matching Hermes profile DB.

    Standalone WebUI stored all profile transcripts under one sessions
    directory, with each JSON carrying a ``profile`` field.  Desktop profile
    switching reads each profile's own ``state.db``, so importing everything
    into the active/default database makes legacy conversations show under the
    wrong profile.
    """
    from hermes_cli import profiles

    root = resolve_webui_sessions_dir(state_dir=state_dir, sessions_dir=sessions_dir)
    report = WebUIProfilesImportReport(sessions_dir=str(root), dry_run=dry_run)

    profile_dirs = {info.name: info.path for info in profiles.list_profiles()}
    needed_profiles = _profiles_in_webui_sessions(root)
    for name in sorted(needed_profiles, key=lambda value: (value != "default", value)):
        home = profile_dirs.get(name)
        if not home:
            report.errors.append(
                WebUISessionImportIssue(
                    path=str(root),
                    session_id=name,
                    reason="profile referenced by WebUI session JSON does not exist",
                )
            )
            continue
        db = SessionDB(db_path=Path(home) / "state.db")
        try:
            report.reports.append(
                import_webui_sessions(
                    db,
                    state_dir=state_dir,
                    sessions_dir=sessions_dir,
                    dry_run=dry_run,
                    include_empty=include_empty,
                    profile=name,
                )
            )
        finally:
            db.close()

    return report


def format_webui_import_report(report: WebUISessionImportReport) -> str:
    mode = "dry-run" if report.dry_run else "applied"
    profile = f" [{report.profile}]" if report.profile else ""
    lines = [
        f"WebUI session import {mode}{profile}: {report.sessions_dir}",
        f"  scanned: {report.scanned}",
        f"  would import: {report.imported}" if report.dry_run else f"  imported: {report.imported}",
        f"  would refresh: {report.refreshed}" if report.dry_run else f"  refreshed: {report.refreshed}",
        f"  skipped existing: {report.skipped_existing}",
        f"  skipped empty: {report.skipped_empty}",
        f"  skipped non-WebUI existing rows: {report.skipped_foreign_source}",
        f"  skipped other profiles: {report.skipped_other_profile}",
        f"  skipped invalid: {report.skipped_invalid}",
    ]
    if report.errors:
        lines.append("  errors:")
        for issue in report.errors[:10]:
            sid = f" [{issue.session_id}]" if issue.session_id else ""
            lines.append(f"    - {issue.path}{sid}: {issue.reason}")
        if len(report.errors) > 10:
            lines.append(f"    - ... {len(report.errors) - 10} more")
    if report.dry_run and report.changed:
        lines.append("Re-run with --apply to write these sessions into state.db.")
    return "\n".join(lines)


def format_webui_profiles_import_report(report: WebUIProfilesImportReport) -> str:
    mode = "dry-run" if report.dry_run else "applied"
    lines = [f"WebUI profile import {mode}: {report.sessions_dir}"]
    if report.errors:
        lines.append("  profile errors:")
        for issue in report.errors:
            lines.append(f"    - {issue.session_id}: {issue.reason}")
    for child in report.reports:
        lines.append("")
        lines.append(format_webui_import_report(child))
    if report.dry_run and report.changed:
        lines.append("")
        lines.append("Re-run with --apply to write these sessions into profile state.db files.")
    return "\n".join(lines)


def _iter_webui_session_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*.json"), key=lambda p: p.name):
        if path.name.startswith("_"):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        yield path


def _session_id_for_payload(path: Path, payload: dict[str, Any]) -> str:
    raw = payload.get("session_id") or payload.get("id") or path.stem
    return str(raw).strip()


def _normalized_profile(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value or "").strip().lower()
    return text or None


def _payload_profile(payload: dict[str, Any]) -> str:
    return _normalized_profile(payload.get("profile")) or "default"


def _profiles_in_webui_sessions(root: Path) -> set[str]:
    profiles: set[str] = set()
    if not root.exists() or not root.is_dir():
        return profiles
    for path in _iter_webui_session_files(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            profiles.add(_payload_profile(payload))
    return profiles


def _valid_session_id(session_id: str) -> bool:
    if not session_id or len(session_id) > MAX_SESSION_ID_LENGTH:
        return False
    return re.search(r"[\r\n\x00]", session_id) is None


def _normalized_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role == "human":
            role = "user"
        if role not in VALID_ROLES:
            continue
        msg = dict(item)
        msg["role"] = role
        out.append(msg)
    return out


def _actual_message_count(db: SessionDB, session_id: str) -> int:
    try:
        with db._lock:
            row = db._conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND active = 1",
                (session_id,),
            ).fetchone()
        return int(row["n"] if hasattr(row, "keys") else row[0])
    except sqlite3.Error:
        return 0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return default


def _message_timestamp(message: dict[str, Any], fallback: float) -> float:
    return _safe_float(message.get("timestamp")) or _safe_float(message.get("_ts")) or fallback


def _session_timestamp(payload: dict[str, Any], key: str, fallback: float | None = None) -> float | None:
    return _safe_float(payload.get(key)) or fallback


def _safe_title(db: SessionDB, session_id: str, raw_title: Any) -> str | None:
    text = str(raw_title or "").strip()
    if text.lower() == "untitled":
        return None
    try:
        cleaned = SessionDB.sanitize_title(text)
    except ValueError:
        cleaned = SessionDB.sanitize_title(text[: SessionDB.MAX_TITLE_LENGTH])
    if not cleaned:
        return None

    existing = db.get_session_by_title(cleaned)
    if not existing or existing.get("id") == session_id:
        return cleaned

    suffix = f" ({session_id[:8]})"
    base = cleaned[: max(1, SessionDB.MAX_TITLE_LENGTH - len(suffix))].rstrip()
    candidate = f"{base}{suffix}"
    existing = db.get_session_by_title(candidate)
    if not existing or existing.get("id") == session_id:
        return candidate
    return None


def _json_field(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def _tool_call_count(messages: list[dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            count += len(tool_calls)
        elif tool_calls:
            count += 1
    return count


def _write_webui_session(
    db: SessionDB,
    session_id: str,
    payload: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    existing: dict[str, Any] | None,
) -> None:
    now = time.time()
    created_at = _session_timestamp(payload, "created_at") or _first_message_ts(messages) or now
    title = _safe_title(db, session_id, payload.get("title"))
    model = payload.get("model")
    cwd = payload.get("workspace") or payload.get("cwd")
    archived = 1 if bool(payload.get("archived")) else 0
    input_tokens = _safe_int(payload.get("input_tokens"))
    output_tokens = _safe_int(payload.get("output_tokens"))
    cache_read_tokens = _safe_int(payload.get("cache_read_tokens"))
    cache_write_tokens = _safe_int(payload.get("cache_write_tokens"))
    estimated_cost = payload.get("estimated_cost")
    try:
        estimated_cost = float(estimated_cost) if estimated_cost is not None else None
    except (TypeError, ValueError):
        estimated_cost = None
    parent_session_id = payload.get("parent_session_id")
    if parent_session_id and not db.get_session(str(parent_session_id)):
        parent_session_id = None

    def _do(conn):
        conn.execute(
            """INSERT OR IGNORE INTO sessions (
                   id, source, model, parent_session_id, started_at, message_count,
                   tool_call_count, input_tokens, output_tokens, cache_read_tokens,
                   cache_write_tokens, cwd, title, archived, estimated_cost_usd
               ) VALUES (?, 'webui', ?, ?, ?, 0, 0, 0, 0, 0, 0, ?, ?, ?, ?)""",
            (
                session_id,
                str(model) if model else None,
                str(parent_session_id) if parent_session_id else None,
                created_at,
                str(cwd) if cwd else None,
                title,
                archived,
                estimated_cost,
            ),
        )
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

        last_ts = created_at
        for idx, msg in enumerate(messages):
            ts = _message_timestamp(msg, last_ts + 1e-6 if idx else created_at)
            last_ts = ts
            tool_calls = msg.get("tool_calls")
            reasoning_details = msg.get("reasoning_details") if msg.get("role") == "assistant" else None
            codex_reasoning_items = msg.get("codex_reasoning_items") if msg.get("role") == "assistant" else None
            codex_message_items = msg.get("codex_message_items") if msg.get("role") == "assistant" else None
            conn.execute(
                """INSERT INTO messages (
                       session_id, role, content, tool_call_id, tool_calls,
                       tool_name, timestamp, token_count, finish_reason, reasoning,
                       reasoning_content, reasoning_details, codex_reasoning_items,
                       codex_message_items, platform_message_id, observed
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    msg.get("role"),
                    db._encode_content(msg.get("content")),
                    msg.get("tool_call_id"),
                    _json_field(tool_calls) if tool_calls else None,
                    msg.get("tool_name"),
                    ts,
                    msg.get("token_count"),
                    msg.get("finish_reason"),
                    msg.get("reasoning") if msg.get("role") == "assistant" else None,
                    msg.get("reasoning_content") if msg.get("role") == "assistant" else None,
                    _json_field(reasoning_details),
                    _json_field(codex_reasoning_items),
                    _json_field(codex_message_items),
                    msg.get("platform_message_id") or msg.get("message_id"),
                    1 if msg.get("observed") else 0,
                ),
            )

        conn.execute(
            """UPDATE sessions SET
                   source = 'webui',
                   model = COALESCE(model, ?),
                   parent_session_id = COALESCE(parent_session_id, ?),
                   started_at = ?,
                   message_count = ?,
                   tool_call_count = ?,
                   input_tokens = ?,
                   output_tokens = ?,
                   cache_read_tokens = ?,
                   cache_write_tokens = ?,
                   cwd = COALESCE(cwd, ?),
                   title = COALESCE(title, ?),
                   archived = ?,
                   estimated_cost_usd = COALESCE(?, estimated_cost_usd)
               WHERE id = ?""",
            (
                str(model) if model else None,
                str(parent_session_id) if parent_session_id else None,
                created_at if not existing else (existing.get("started_at") or created_at),
                len(messages),
                _tool_call_count(messages),
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
                str(cwd) if cwd else None,
                title,
                archived,
                estimated_cost,
                session_id,
            ),
        )

    db._execute_write(_do)


def _first_message_ts(messages: list[dict[str, Any]]) -> float | None:
    for msg in messages:
        ts = _message_timestamp(msg, 0)
        if ts:
            return ts
    return None
