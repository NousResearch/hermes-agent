"""Safe conversation-history reset for ``hermes memory reset``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ``SessionDB.delete_sessions`` discovers delegate children with a query that
# binds each input ID twice. 250 stays below SQLite's legacy 999-variable limit.
_SESSION_DELETE_BATCH = 250
_UNKNOWN_RUNNING_PID = 0


def _get_running_gateway_pid(hermes_home: Path) -> int | None:
    """Return a live gateway PID that may write to this profile's state."""
    from gateway.status import resolve_gateway_liveness
    from hermes_constants import get_default_hermes_root

    homes: list[Path] = []
    for candidate in (hermes_home, get_default_hermes_root()):
        canonical = candidate.expanduser().resolve(strict=False)
        if canonical not in homes:
            homes.append(canonical)

    for candidate in homes:
        liveness = resolve_gateway_liveness(profile_dir=candidate, use_cache=False)
        if liveness.running:
            return liveness.pid if liveness.pid is not None else _UNKNOWN_RUNNING_PID
        if liveness.probe_error:
            raise RuntimeError(
                f"gateway liveness probe was inconclusive for {candidate}"
            )
    return None


def _collect_session_ids(db: Any, expected_count: int) -> list[str]:
    """Read every session ID once and fail if the snapshot is inconsistent."""
    if expected_count <= 0:
        return []

    rows = db.list_sessions_rich(
        limit=expected_count,
        include_children=True,
        project_compression_tips=False,
        include_archived=True,
        compact_rows=True,
    )
    session_ids = [
        row.get("id")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]
    if len(session_ids) != expected_count or len(set(session_ids)) != len(session_ids):
        raise RuntimeError(
            "session listing changed while reset was preparing; "
            "stop all Hermes processes and retry"
        )
    return session_ids


def _delete_conversations(db: Any, sessions_dir: Path, session_ids: list[str]) -> None:
    """Delete the captured sessions through the existing SessionDB contract."""
    for start in range(0, len(session_ids), _SESSION_DELETE_BATCH):
        db.delete_sessions(
            session_ids[start : start + _SESSION_DELETE_BATCH],
            sessions_dir=sessions_dir,
        )

    remaining_sessions = db.session_count(include_archived=True)
    remaining_messages = db.message_count()
    if remaining_sessions or remaining_messages:
        raise RuntimeError(
            f"{remaining_sessions} session(s) and {remaining_messages} message(s) "
            "remained; stop all Hermes processes and retry"
        )


def cmd_memory_reset(args: Any) -> int:
    """Clear persisted conversation history while preserving built-in memory."""
    from hermes_constants import display_hermes_home, get_hermes_home

    target = getattr(args, "target", None)
    if target != "conversations":
        print(f"\n  ✗ Unsupported conversation reset target: {target!r}\n")
        return 2

    hermes_home = Path(get_hermes_home())
    db_path = hermes_home / "state.db"
    if not db_path.is_file():
        print("\n  Nothing to reset.\n")
        return 0

    try:
        running_pid = _get_running_gateway_pid(hermes_home)
    except Exception as exc:
        print(f"\n  ✗ Could not verify gateway status: {exc}\n")
        return 1
    if running_pid is not None:
        pid_detail = f" (PID {running_pid})" if running_pid else ""
        print(
            f"\n  ✗ A gateway that may use this profile is running{pid_detail}. "
            "Stop it before clearing conversation history:\n"
            "      hermes gateway stop\n"
        )
        return 1

    from hermes_state import SessionDB

    db = None
    try:
        db = SessionDB(db_path)
        session_count = db.session_count(include_archived=True)
        message_count = db.message_count()
        session_ids = _collect_session_ids(db, session_count)
        if message_count and not session_ids:
            raise RuntimeError(
                "state.db contains messages without sessions; refusing a partial reset"
            )
    except Exception as exc:
        if db is not None:
            db.close()
        print(f"\n  ✗ Could not inspect conversation history: {exc}\n")
        return 1

    if not session_count and not message_count:
        db.close()
        print("\n  Nothing to reset.\n")
        return 0

    print(
        "\n  This will permanently erase conversation history — "
        f"{session_count:,} sessions, {message_count:,} messages."
    )
    print("  Note: stop all other Hermes CLI/TUI/cron processes first.")

    if not getattr(args, "yes", False):
        try:
            answer = input("\n  Type 'yes' to confirm: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer != "yes":
            db.close()
            print("  Cancelled.\n")
            return 0

    try:
        _delete_conversations(db, hermes_home / "sessions", session_ids)
    except Exception as exc:
        print(f"  ✗ Failed to clear conversation history: {exc}")
        return 1
    finally:
        db.close()

    print(
        "  ✓ Cleared conversation history "
        f"({session_count:,} sessions, {message_count:,} messages)"
    )
    print(f"  Hermes home: {display_hermes_home()}\n")
    return 0
