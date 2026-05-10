#!/usr/bin/env python3
"""Backfill or repair SQLite session rows from Hermes JSON session logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_state import SessionDB


SUMMARY_KEYS = (
    "scanned",
    "backfilled",
    "would_backfill",
    "skipped_existing",
    "skipped_active",
    "skipped_missing_messages",
    "created_sessions",
    "forced",
    "errors",
)


def _empty_summary() -> dict[str, Any]:
    summary = {key: 0 for key in SUMMARY_KEYS}
    summary["reports"] = []
    return summary


def _report(
    summary: dict[str, Any],
    *,
    session_id: str | None,
    path: Path,
    status: str,
    message_count: int | None = None,
    reason: str | None = None,
) -> None:
    item: dict[str, Any] = {
        "session_id": session_id,
        "path": str(path),
        "status": status,
    }
    if message_count is not None:
        item["message_count"] = message_count
    if reason:
        item["reason"] = reason
    summary["reports"].append(item)


def _session_path(sessions_dir: Path, session_id: str) -> Path:
    return sessions_dir / f"session_{session_id}.json"


def _candidate_paths(sessions_dir: Path, session_id: str | None) -> list[Path]:
    if session_id:
        return [_session_path(sessions_dir, session_id)]
    return sorted(sessions_dir.glob("session_*.json"))


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("session log must contain a JSON object")
    return payload


def _payload_session_id(path: Path, payload: dict[str, Any]) -> str:
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session log is missing string session_id")
    expected = path.name.removeprefix("session_").removesuffix(".json")
    if expected != session_id:
        raise ValueError(
            f"session_id mismatch: file implies {expected!r}, JSON has {session_id!r}"
        )
    return session_id


def _payload_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("session log is missing messages list")
    normalized_messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} is not an object")
        if not isinstance(message.get("role"), str):
            raise ValueError(f"message {index} is missing string role")
        normalized = dict(message)
        # Older JSON session logs store tool result names in OpenAI's `name`
        # field, while SessionDB stores/searches them as `tool_name`.
        if (
            normalized.get("role") == "tool"
            and not normalized.get("tool_name")
            and isinstance(normalized.get("name"), str)
        ):
            normalized["tool_name"] = normalized["name"]
        normalized_messages.append(normalized)
    return normalized_messages


def _ensure_session_row(
    db: SessionDB,
    payload: dict[str, Any],
    session_id: str,
) -> bool:
    if db.get_session(session_id) is not None:
        return False
    source = payload.get("platform") or payload.get("source") or "unknown"
    model = payload.get("model")
    db.create_session(session_id=session_id, source=str(source), model=model)
    # A missing DB row has no live-session marker to protect. JSON transcripts
    # from the normal agent log format do not include `ended_at`, so mark rows
    # created by this manual repair as ended to avoid creating active ghosts.
    db.end_session(
        session_id,
        end_reason=str(payload.get("end_reason") or "json_backfill"),
    )
    return True


def _process_one(
    db: SessionDB,
    path: Path,
    summary: dict[str, Any],
    *,
    dry_run: bool,
    force: bool,
    include_active: bool,
) -> None:
    try:
        payload = _load_payload(path)
        session_id = _payload_session_id(path, payload)
        messages = _payload_messages(payload)
    except Exception as exc:
        summary["errors"] += 1
        _report(
            summary,
            session_id=None,
            path=path,
            status="error",
            reason=str(exc),
        )
        return

    if not messages:
        summary["skipped_missing_messages"] += 1
        _report(
            summary,
            session_id=session_id,
            path=path,
            status="skipped_missing_messages",
            message_count=0,
            reason="JSON transcript has no messages",
        )
        return

    existing = db.get_session(session_id)
    created = False
    if existing is None:
        if not dry_run:
            created = _ensure_session_row(db, payload, session_id)
            existing = db.get_session(session_id)
        else:
            existing = {
                "id": session_id,
                # Missing DB rows are treated like repair candidates, not
                # active sessions, because there is no DB active marker. Keep
                # dry-run parity with the actual path, which creates and ends
                # the repair row before replacing messages.
                "ended_at": 1,
                "message_count": 0,
            }

    if existing is not None and existing.get("ended_at") is None and not include_active:
        summary["skipped_active"] += 1
        _report(
            summary,
            session_id=session_id,
            path=path,
            status="skipped_active",
            message_count=len(messages),
            reason="session is active; pass --include-active to backfill",
        )
        return

    current_count = int((existing or {}).get("message_count") or 0)
    if current_count > 0 and not force:
        summary["skipped_existing"] += 1
        _report(
            summary,
            session_id=session_id,
            path=path,
            status="skipped_existing",
            message_count=current_count,
            reason="session already has DB messages; pass --force to replace",
        )
        return

    if dry_run:
        summary["would_backfill"] += 1
        _report(
            summary,
            session_id=session_id,
            path=path,
            status="would_backfill",
            message_count=len(messages),
        )
        return

    if created:
        summary["created_sessions"] += 1

    try:
        db.replace_messages(session_id, messages)
    except Exception as exc:
        summary["errors"] += 1
        _report(
            summary,
            session_id=session_id,
            path=path,
            status="error",
            message_count=len(messages),
            reason=f"database write failed: {exc}",
        )
        return
    summary["backfilled"] += 1
    if force and current_count > 0:
        summary["forced"] += 1
        status = "forced"
    else:
        status = "backfilled"
    _report(
        summary,
        session_id=session_id,
        path=path,
        status=status,
        message_count=len(messages),
    )


def backfill_sessions_from_json(
    db_path: Path | str | None = None,
    sessions_dir: Path | str | None = None,
    session_id: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    include_active: bool = False,
) -> dict[str, Any]:
    """Backfill SQLite session messages from Hermes ``session_*.json`` logs."""
    home = get_hermes_home()
    db_path = Path(db_path) if db_path is not None else home / "state.db"
    sessions_dir = Path(sessions_dir) if sessions_dir is not None else home / "sessions"

    summary = _empty_summary()
    paths = _candidate_paths(sessions_dir, session_id)
    db = SessionDB(db_path=db_path)
    try:
        for path in paths:
            summary["scanned"] += 1
            if not path.exists():
                summary["errors"] += 1
                _report(
                    summary,
                    session_id=session_id,
                    path=path,
                    status="error",
                    reason="session JSON file does not exist",
                )
                continue
            _process_one(
                db,
                path,
                summary,
                dry_run=dry_run,
                force=force,
                include_active=include_active,
            )
    finally:
        db.close()
    return summary


def _build_parser() -> argparse.ArgumentParser:
    home = get_hermes_home()
    parser = argparse.ArgumentParser(
        description="Backfill or repair Hermes SQLite session messages from JSON logs.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=home / "state.db",
        help="Path to state.db (default: $HERMES_HOME/state.db).",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=home / "sessions",
        help="Directory containing session_*.json logs (default: $HERMES_HOME/sessions).",
    )
    parser.add_argument("--session-id", help="Backfill only one session ID.")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing DB messages with the JSON transcript.",
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="Allow backfilling sessions whose DB row has no ended_at.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the summary as JSON instead of human-readable lines.",
    )
    return parser


def _print_text_summary(summary: dict[str, Any]) -> None:
    for key in SUMMARY_KEYS:
        print(f"{key}: {summary[key]}")
    for report in summary["reports"]:
        suffix = ""
        if report.get("reason"):
            suffix = f" ({report['reason']})"
        print(f"{report['status']}: {report['session_id'] or report['path']}{suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = backfill_sessions_from_json(
        db_path=args.db,
        sessions_dir=args.sessions_dir,
        session_id=args.session_id,
        dry_run=args.dry_run,
        force=args.force,
        include_active=args.include_active,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_text_summary(summary)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
