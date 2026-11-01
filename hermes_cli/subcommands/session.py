"""``hermes session`` subcommand parser — export/import session checkpoints.

Provides the CLI interface for cross-session context bridging:

    hermes session export <session-id>    # export session as JSON (stdout)
    hermes session export <session-id> -o file.json   # write to file
    hermes session import <file.json>      # import a session checkpoint

This is the MVP for issue #63748 — session checkpoint export/import for
cross-session context bridging. The export captures the full session record
and all messages. The import replays them back into the database under a
fresh session id so the context can be /resumed or searched.

Usage examples::

    # Export the current session (from a running agent, capture the session id)
    hermes session export abc123-def456
    hermes session export abc123-def456 -o ~/checkpoint.json

    # Import a checkpoint into a new session
    hermes session import ~/checkpoint.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ── helpers ──────────────────────────────────────────────────────────────────


def _get_db() -> Any:
    """Lazy-import and return a writable SessionDB instance."""
    from hermes_state import SessionDB

    return SessionDB()


def _resolve_session(db: Any, session_id: str) -> Optional[str]:
    """Resolve an exact or uniquely-prefixed session id, or None."""
    resolved = db.resolve_session_id(session_id)
    if resolved:
        return resolved
    # Try exact match fallback (resolve_session_id already does this)
    return None


def _fmt_bytes(n: int) -> str:
    units = ("B", "KB", "MB", "GB")
    size = float(n or 0)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _collect_context_sessions(
    db: Any, session_id: str, lineage_only: bool
) -> List[Dict[str, Any]]:
    """Collect one or more sessions to export.

    When *lineage_only* is True and the session has a compression lineage,
    returns the full lineage (all segments merged). Otherwise returns a
    single-session export.
    """
    if lineage_only:
        export = db.export_session_lineage(session_id)
        if export:
            return [export]
    export = db.export_session(session_id)
    return [export] if export else []


def _format_session_summary(session: Dict[str, Any]) -> str:
    """Return a one-line summary of an exported session dict."""
    sid = session.get("id", "?")
    title = session.get("title") or "(untitled)"
    msg_count = session.get("message_count", 0)
    segments = session.get("segments")
    if segments:
        return (
            f"{sid} — \"{title}\" — {msg_count} messages "
            f"({len(segments)} compressed segments)"
        )
    return f"{sid} — \"{title}\" — {msg_count} messages"


# ── command handlers ─────────────────────────────────────────────────────────


def cmd_export(args: Any) -> int:
    """Export one or more sessions to a JSON checkpoint file or stdout."""
    session_id = args.session_id.strip()
    output_path: Optional[str] = args.output
    lineage_only = args.lineage

    db = _get_db()
    resolved = _resolve_session(db, session_id)
    if not resolved:
        print(
            f"Session not found: {session_id}",
            file=sys.stderr,
        )
        return 1

    sessions = _collect_context_sessions(db, resolved, lineage_only=lineage_only)
    if not sessions:
        print(
            f"No data found for session: {resolved}",
            file=sys.stderr,
        )
        return 1

    # Build the checkpoint payload
    checkpoint = {
        "hermes_checkpoint_version": 1,
        "created_at": time.time(),
        "source_session_ids": [s.get("id", resolved) for s in sessions],
        "sessions": sessions,
    }

    # Summary
    total_msgs = sum(s.get("message_count", 0) for s in sessions)
    print(f"Exported {len(sessions)} session(s) with {total_msgs} messages:")
    for s in sessions:
        print(f"  {_format_session_summary(s)}")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(checkpoint, indent=2, default=str, ensure_ascii=False)
        )
        print(f"Written to: {out.resolve()}")
        print(f"Size: {_fmt_bytes(out.stat().st_size)}")
    else:
        json.dump(checkpoint, sys.stdout, indent=2, default=str, ensure_ascii=False)
        sys.stdout.write("\n")

    return 0


def cmd_import(args: Any) -> int:
    """Import a checkpoint file into the session database.

    Creates new sessions with fresh timestamps and appends all messages.
    Keeps the original session metadata (model, title, system_prompt, etc.)
    so the imported context is discoverable via /sessions and session_search.
    """
    import_path = Path(args.file)
    if not import_path.exists():
        print(f"File not found: {import_path}", file=sys.stderr)
        return 1

    try:
        checkpoint = json.loads(import_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Failed to read checkpoint file: {exc}", file=sys.stderr)
        return 1

    # Validate checkpoint format
    version = checkpoint.get("hermes_checkpoint_version")
    if version != 1:
        print(
            f"Unknown checkpoint version: {version}. Expected 1.",
            file=sys.stderr,
        )
        return 1

    sessions_data = checkpoint.get("sessions", [])
    if not sessions_data:
        print("Checkpoint contains no sessions.", file=sys.stderr)
        return 1

    import uuid

    db = _get_db()
    total_imported = 0
    total_messages = 0

    for session_data in sessions_data:
        # Strip messages from the session dict — we insert them separately
        messages: List[Dict[str, Any]] = session_data.pop("messages", [])
        segments: List[Dict[str, Any]] = session_data.pop("segments", [])
        lineage_ids: List[str] = session_data.pop("lineage_session_ids", [])

        # If this is a lineage export, the flattened messages are already in
        # session_data["messages"] from export_session_lineage, but we popped
        # them above. Merge raw segment messages back if needed.
        if segments and not messages:
            for seg in segments:
                messages.extend(seg.get("messages") or [])

        # Generate a fresh session id for the imported copy
        new_id = str(uuid.uuid4())
        source = session_data.get("source", "cli-checkpoint")
        title = session_data.get("title")
        model = session_data.get("model")
        model_config = session_data.get("model_config")
        system_prompt = session_data.get("system_prompt")

        # Create the new session with available metadata
        # Note: create_session does NOT accept title, started_at, etc.
        # Those are set via a separate UPDATE after message insertion.
        kw: Dict[str, Any] = {}
        if model:
            kw["model"] = model
        if model_config:
            kw["model_config"] = model_config
        if system_prompt:
            kw["system_prompt"] = system_prompt

        try:
            db.create_session(new_id, source=source, **kw)
        except Exception as exc:
            print(
                f"Failed to create session: {exc}",
                file=sys.stderr,
            )
            continue

        # Append messages preserving original role/content order
        msg_count = 0
        for msg in messages:
            role = msg.get("role")
            if not role:
                continue
            try:
                db.append_message(
                    new_id,
                    role=role,
                    content=msg.get("content"),
                    tool_name=msg.get("tool_name"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                    token_count=msg.get("token_count"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning"),
                    reasoning_content=msg.get("reasoning_content"),
                    reasoning_details=msg.get("reasoning_details"),
                    codex_reasoning_items=msg.get("codex_reasoning_items"),
                    codex_message_items=msg.get("codex_message_items"),
                    platform_message_id=msg.get("platform_message_id"),
                    observed=bool(msg.get("observed", False)),
                    effect_disposition=msg.get("effect_disposition"),
                    # Preserve original timestamp so message ordering is retained
                    timestamp=msg.get("timestamp"),
                )
                msg_count += 1
            except Exception as exc:
                print(
                    f"  Warning: skipped message {msg.get('id', '?')}: {exc}",
                    file=sys.stderr,
                )
                continue

        total_imported += 1
        total_messages += msg_count
        print(
            f"Imported: {new_id[:12]}… — "
            f"\"{title or '(untitled)'}\" — {msg_count} messages"
        )

    print()
    print(
        f"Done. Imported {total_imported} session(s) with "
        f"{total_messages} message(s) total."
    )
    if total_imported > 0:
        print("Use `/sessions` or `hermes sessions` to find the imported sessions.")
    return 0


# ── parser registration ─────────────────────────────────────────────────────


def build_session_parser(
    subparsers: Any,
    *,
    cmd_session_export: Callable = cmd_export,
    cmd_session_import: Callable = cmd_import,
) -> None:
    """Attach ``hermes session`` to the top-level CLI parser."""
    parser = subparsers.add_parser(
        "session",
        help="Export/import session checkpoints for cross-session context bridging",
        description="Export session context to a JSON file, or import a previously "
        "exported checkpoint into this instance's database. "
        "Useful for transferring context between Hermes instances "
        "or persisting a session's state before cleanup.",
    )
    subs = parser.add_subparsers(
        dest="session_command", metavar="COMMAND", required=True
    )

    # ── export ──────────────────────────────────────────────────────────
    export_parser = subs.add_parser(
        "export",
        help="Export session(s) to a JSON checkpoint file or stdout",
        description="Export a session (and optionally its compression lineage) "
        "as a JSON checkpoint. When --lineage is set and the session has "
        "been compressed, the full chain of compressed segments is merged "
        "into one logical export. The checkpoint can be imported later "
        "via `hermes session import <file>`.",
    )
    export_parser.add_argument(
        "session_id",
        help="Session ID (full or unique prefix) to export",
    )
    export_parser.add_argument(
        "-o", "--output",
        help="Output file path (default: write to stdout)",
        default=None,
    )
    export_parser.add_argument(
        "--lineage",
        action="store_true",
        help="Export the full compression lineage instead of just the current segment",
    )
    export_parser.set_defaults(func=cmd_session_export)

    # ── import ──────────────────────────────────────────────────────────
    import_parser = subs.add_parser(
        "import",
        help="Import a session checkpoint from a JSON file",
        description="Import a previously-exported session checkpoint into "
        "the local session database. The imported session gets a fresh ID "
        "and its original metadata (title, model, system prompt) are "
        "preserved, making it discoverable via /sessions and session_search.",
    )
    import_parser.add_argument(
        "file",
        help="Path to the checkpoint JSON file (.json)",
    )
    import_parser.set_defaults(func=cmd_session_import)
