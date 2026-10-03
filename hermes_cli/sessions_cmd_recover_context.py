"""Offline recovery of model context from an operator-reviewed summary."""

import sqlite3
from pathlib import Path


def cmd_recover_context(db, args) -> int:
    from agent.context_compressor import (
        COMPRESSED_SUMMARY_METADATA_KEY,
        SUMMARY_PREFIX,
        ContextCompressor,
        _SUMMARY_END_MARKER,
    )

    summary_path = Path(args.summary_file).expanduser()
    try:
        summary = ContextCompressor._with_summary_prefix(
            summary_path.read_text(encoding="utf-8-sig").strip(),
        )
    except (OSError, UnicodeError) as exc:
        print(f"Error: could not read the UTF-8 summary file: {exc}")
        return 2
    if summary == SUMMARY_PREFIX:
        print("Error: the summary file must contain a non-empty continuation summary.")
        return 2
    if not args.session_id.strip():
        print("Error: supply a session ID or a unique prefix.")
        return 2
    session_id = db.resolve_session_id(args.session_id)
    if session_id is None:
        print(f"No unique session matches '{args.session_id}'. Run: hermes sessions list")
        return 1
    target = db.resolve_resume_session_id(session_id)
    expected_ids = db.get_active_message_ids(target)
    if not expected_ids:
        print(f"Session '{target}' has no active context to recover.")
        return 1

    print(f"{'Applying' if args.apply else 'Preview'} context recovery for {target} in {db.db_path}")
    print(f"Archive {len(expected_ids)} active message(s) and replace model context with the reviewed summary.")
    print("Original history remains available for display and search; the stored system prompt is preserved.")
    if not args.apply:
        print(f"\nSummary:\n{summary}\n\nRe-run with --apply after stopping clients using this profile.")
        return 0

    # A client retaining the old context can replay it later even when no turn is
    # running. Use the native holder scan as well as the transactional lease fence.
    from hermes_state_holders import foreign_state_db_holders
    holders = foreign_state_db_holders(db.db_path)
    if holders:
        print("Refusing context recovery: the store is in use or its holder scan is incomplete.")
        for pid, target_path in holders:
            print(f"  {pid}: {target_path}")
        print("Stop the gateway, quit Desktop/TUI/CLI clients, and pause cron for this profile before retrying.")
        return 1

    messages = [
        {
            "role": "user",
            "content": summary + "\n\n" + _SUMMARY_END_MARKER,
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        {
            "role": "assistant",
            "content": "Understood. I will use the summary as context and wait for the next request.",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
            "display_kind": "hidden",
        },
    ]
    try:
        db.archive_and_compact(target, messages, expected_active_ids=expected_ids)
    except (RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"Error: context recovery was not applied: {exc}")
        print("Stop active writers and preview the session again before retrying.")
        return 1
    print(f"Context recovered. Resume with: hermes --resume {target}")
    return 0
