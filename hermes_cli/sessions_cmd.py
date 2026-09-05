"""``hermes sessions`` command.

``cmd_sessions`` routes ``args.sessions_action`` through ``_PRE_DB_HANDLERS`` (repair / recover /
import — must run without opening ``SessionDB()``, which a malformed schema prevents) and
``_DB_HANDLERS`` (everything else, sharing one ``SessionDB``). ``get_hermes_home`` is resolved through
``hermes_cli.main`` at call time so monkeypatches keep working. Picker: :mod:`hermes_cli.sessions_cmd_browse`.
"""

import json
import os
import shutil
import sys
from functools import partial
from pathlib import Path

from hermes_cli.sessions_cmd_browse import _relative_time, _session_browse_picker


def get_hermes_home():
    from hermes_cli import main
    return main.get_hermes_home()


def _sessions_dir() -> Path:
    return get_hermes_home() / "sessions"


def _size_mb(path) -> float:
    return os.path.getsize(path) / (1024 * 1024) if path.exists() else 0.0


def _size_delta_label(saved_mb: float) -> str:
    """A negative delta means the file GREW (concurrent writes during a long optimize); "reclaimed
    -163.0 MB" reads as data loss, so say "grew by"."""
    return f"reclaimed {saved_mb:.1f} MB" if saved_mb >= 0 else f"grew by {-saved_mb:.1f} MB"


def _confirm_prompt(prompt: str) -> bool:
    """Prompt for y/N confirmation, safe against non-TTY environments."""
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def _not_found(session_id) -> int:
    print(f"Session '{session_id}' not found.")
    return 1


def _print_dry_run_preview(candidates, filters) -> None:
    from hermes_cli.session_filters import describe_filters
    print(f"Would export {len(candidates)} session(s) ({describe_filters(filters)}).")
    for row in candidates[:100]:
        print(f"  {row.get('id')}  {row.get('source', '')}")
    if len(candidates) > 100:
        print(f"  ... {len(candidates) - 100} more")


_FILTER_ARGS = (
    "older_than", "newer_than", "before", "after", "source", "title", "end_reason", "cwd", "min_messages",
    "max_messages", "model", "provider", "user", "chat_id", "chat_type", "branch", "min_tokens", "max_tokens",
    "min_cost", "max_cost", "min_tool_calls", "max_tool_calls",
)


def _any_filter_args(args) -> bool:
    return any(getattr(args, a, None) is not None for a in _FILTER_ARGS)


def _export_dir(output) -> Path:
    """``--output`` dir for multi-file exports; ``~/.hermes/session-exports`` when empty or ``-``."""
    return Path(output).expanduser() if output and output != "-" else get_hermes_home() / "session-exports"


def _write_output(output, text, summary) -> None:
    """Write to stdout when *output* is empty or ``-``; else to the file + print *summary*."""
    if not output or output == "-":
        sys.stdout.write(text)
        return
    with open(output, "w", encoding="utf-8") as f:
        f.write(text)
    print(summary)


# -- handlers that must run BEFORE SessionDB() is opened ----------------------

def _cmd_repair(args):
    from hermes_state import DEFAULT_DB_PATH as db_path, SessionDB
    from hermes_state_repair import _db_opens_cleanly, repair_state_db_schema
    if not db_path.exists():
        print(f"No session database at {db_path} (nothing to repair).")
        return
    reason = _db_opens_cleanly(db_path)
    if reason is None:
        print(f"✓ {db_path} opens cleanly — no repair needed.")
        return
    print(f"✗ {db_path} does not open cleanly: {reason}")
    if getattr(args, "check_only", False):
        return 1
    print("Repairing (a backup copy is made first)…")
    report = repair_state_db_schema(db_path, backup=not getattr(args, "no_backup", False))
    if report.get("repaired"):
        if report.get("backup_path"):
            print(f"  backup: {report['backup_path']}")
        print(f"  strategy: {report.get('strategy')}")
        try:
            with SessionDB() as _repair_db:
                n = _repair_db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            print(f"✓ Repaired — {n} sessions recovered.")
        except Exception:
            print("✓ Repaired.")
        return
    print(f"✗ Repair failed: {report.get('error')}")
    if report.get("backup_path"):
        print(f"  A backup is preserved at: {report['backup_path']}")
    # Without this pointer the user is at a dead end; lead with --inspect-only before writing.
    source_hint = report.get("backup_path") or db_path
    print(
        "  Keep state.db and the backup; do not delete them.\n"
        "\n  Next step — offline recovery (never modifies the source):\n"
        f"    hermes sessions recover --source {source_hint} \\\n"
        "        --inspect-only\n"
        "  If that reports the data is recoverable, rebuild it into\n"
        "  a NEW database (the active one is left untouched):\n"
        f"    hermes sessions recover --source {source_hint} \\\n"
        "        --output recovered-state.db"
    )


def _cmd_recover(args):
    """Offline recovery: works on a disposable copy of the source; never touches the active database."""
    import sqlite3
    from hermes_cli.session_recovery import (
        SessionRecoveryError, inspect_session_database, recover_session_database, write_recovery_report,
    )
    source, output = args.source, getattr(args, "output", None)
    inspect_only = bool(getattr(args, "inspect_only", False))
    allow_partial = bool(getattr(args, "allow_partial", False))
    report_path = getattr(args, "report", None)
    if not inspect_only and output is not None and report_path is None:
        report_path = output.with_name(output.name + ".recovery.json")
    usage_errors = (
        (inspect_only and output is not None, "--output cannot be used with --inspect-only."),
        (inspect_only and allow_partial, "--allow-partial cannot be used with --inspect-only."),
        (not inspect_only and output is None, "--output is required unless --inspect-only is used."),
        (
            report_path is not None and os.path.lexists(report_path.expanduser()),
            f"refusing to overwrite existing report: {report_path}",
        ),
    )
    for bad, msg in usage_errors:
        if bad:
            print(f"Error: {msg}")
            return 2
    work_dir = getattr(args, "work_dir", None)
    try:
        if inspect_only:
            report = inspect_session_database(source, work_dir=work_dir)
        else:
            print("Recovering canonical session data into a new database…")
            progress = _RecoveryProgress()
            report = recover_session_database(
                source, output, work_dir=work_dir, chunk_size=getattr(args, "chunk_size", 1000),
                progress_cb=progress, allow_partial=allow_partial,
            )
            progress.finish()
    except (SessionRecoveryError, OSError, sqlite3.DatabaseError) as exc:
        print(f"Error: session recovery failed: {exc}\nThe supplied source database was not replaced or deleted.")
        return 1
    if report_path is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        try:
            print(f"Recovery report: {write_recovery_report(report_path, report)}")
        except (FileExistsError, OSError) as exc:
            print(f"Error: could not write recovery report: {exc}")
            return 1
    if inspect_only:
        return 0 if report.get("recoverable") else 1
    return _print_recovery_verdict(report, output, allow_partial)


class _RecoveryProgress:
    """`recover` progress printer: one live-updating ``  <table>: n/total`` line per table."""

    table = None

    def __call__(self, info):
        table = info.get("table")
        if table != self.table:
            self.finish()
            print(f"  {table}: ", end="", flush=True)
            self.table = table
        total = info.get("source_rows")
        suffix = f"/{int(total):,}" if total is not None else ""
        print(f"\r  {table}: {int(info.get('copied_rows') or 0):,}{suffix}", end="", flush=True)

    def finish(self):
        if self.table is not None:
            print()


def _print_recovery_verdict(report, output, allow_partial) -> int:
    if report.get("complete"):
        print(
            f"✓ Recovered database verified at: {output}\n"
            "  The active session database was not changed.\n"
            "  Review the JSON report before installing this database."
        )
        return 0
    if allow_partial and report.get("verified"):
        counts = report.get("verification", {}).get("table_counts", {})
        if report.get("best_effort"):
            print(
                f"✓ BEST-EFFORT page-level salvage verified at: {output}\n"
                "  The source table schemas were unreadable; rows were rebuilt from raw pages "
                "via sqlite3 .recover and mapped heuristically."
            )
        else:
            print(f"✓ Partial recovery output verified at: {output}")
        sessions_n, messages_n = int(counts.get("sessions") or 0), int(counts.get("messages") or 0)
        print(
            f"  Recovered {sessions_n:,} sessions and {messages_n:,} messages.\n"
            "  The active session database was not changed.\n"
            "  This output is incomplete. Review every skipped range and orphan count in the "
            "JSON report before installing it."
        )
        return 0
    print(
        "✗ Recovery output did not pass every verification check.\n"
        "  Do not install it. Review the JSON report for partial data or errors."
    )
    return 1


def _cmd_import(args):
    from hermes_cli.foreign_sessions import run_sessions_import
    # Explicit path but nothing imported -> non-zero for scripts. Picker cancel (no path) -> exit 0.
    if run_sessions_import(args) is None and getattr(args, "path", None):
        return 1


# -- handlers that receive an open SessionDB ----------------------------------

def _default_exclude(args):
    """Hide third-party tool sessions by default, but honour explicit --source."""
    return None if getattr(args, "source", None) else ["tool"]


def _cmd_list(db, args):
    from hermes_state_sessions import workspace_key as _ws_key
    sessions = db.list_sessions_rich(source=args.source, exclude_sources=_default_exclude(args), limit=args.limit)

    # Workspace filter: workspace key (git repo root, else cwd) — path substring or exact basename.
    _ws_filter = (getattr(args, "workspace", None) or "").strip()
    if _ws_filter:
        _needle = _ws_filter.lower()
        keyed = ((s, (_ws_key(s) or "").lower()) for s in sessions)
        sessions = [
            s for s, key in keyed if key and (_needle in key or _needle == os.path.basename(key.rstrip("/\\")))
        ]
    if not sessions:
        print("No sessions found.")
        return

    # Workspace column only when some session carries a key (or when filtering): unbound listings read as before.
    has_ws = bool(_ws_filter) or any(_ws_key(s) for s in sessions)
    has_titles = any(s.get("title") for s in sessions)

    def _ws(s):  # repo/dir basename, "—" when unbound
        key = _ws_key(s)
        return ((os.path.basename(key.rstrip("/\\")) or key) if key else "—")[:16]
    _title = lambda s, n: (s.get("title") or "—")[:n]  # noqa: E731
    _preview = lambda s, n: s.get("preview", "")[:n]  # noqa: E731
    _ago = lambda s: _relative_time(s.get("last_active"))  # noqa: E731
    layouts = {  # (has_ws, has_titles): header, rule width, row formatter
        (True, True): (f"{'Title':<28} {'Workspace':<18} {'Last Active':<13} {'ID'}", 110,
                       lambda s: f"{_title(s, 26):<28} {_ws(s):<18} {_ago(s):<13} {s['id']}"),
        (True, False): (f"{'Preview':<38} {'Workspace':<18} {'Last Active':<13} {'Src':<6} {'ID'}", 100,
                        lambda s: f"{_preview(s, 36):<38} {_ws(s):<18} {_ago(s):<13} {s['source']:<6} {s['id']}"),
        (False, True): (f"{'Title':<32} {'Preview':<40} {'Last Active':<13} {'ID'}", 110,
                        lambda s: f"{_title(s, 30):<32} {_preview(s, 38):<40} {_ago(s):<13} {s['id']}"),
        (False, False): (f"{'Preview':<50} {'Last Active':<13} {'Src':<6} {'ID'}", 95,
                         lambda s: f"{_preview(s, 48):<50} {_ago(s):<13} {s['source']:<6} {s['id']}"),
    }
    header, rule, fmt = layouts[(has_ws, has_titles)]
    print(header + "\n" + "─" * rule)
    for s in sessions:
        print(fmt(s))


# -- export -----------------------------------------------------------------

def _cmd_export(db, args):
    from hermes_cli.session_filters import build_prune_filters
    filters = None
    if _any_filter_args(args):
        try:
            filters = build_prune_filters(args)
        except ValueError as e:
            print(f"Error: {e}")
            return
        # Unlike prune/archive, export includes archived sessions.
        filters["archived"] = None

    def _redact(data):
        if not args.redact or data is None:
            return data
        from hermes_cli.session_export_md import redact_session_data
        return redact_session_data(data)

    def _collect_sessions():
        """--session-id / filters / bare export -> redacted session dicts, or None after printing an error."""
        if args.session_id:
            resolved = db.resolve_session_id(args.session_id)
            data = _redact(db.export_session(resolved)) if resolved else None
            if not data:
                _not_found(args.session_id)
                return None
            return [data]
        if filters:
            candidates = db.list_prune_candidates(**filters)
            if args.dry_run:
                return _print_dry_run_preview(candidates, filters)
            return [s for s in (_redact(db.export_session(row["id"])) for row in candidates) if s]
        if args.dry_run:
            return print("--dry-run requires at least one filter.")
        return [_redact(s) for s in db.export_all(source=None)]
    if getattr(args, "only", None):
        return _export_flat("only", args, _collect_sessions)
    if args.format == "trace":
        return _export_trace(db, args, filters)
    if args.format in _FLAT_EXPORTERS:
        return _export_flat(args.format, args, _collect_sessions)
    return _export_markdown(db, args, filters, _redact)


def _render_only(args, sessions):
    """--only user-prompts: one prompt record per line (jsonl) or headed sections (md)."""
    from hermes_cli.session_export import export_record_count, render_sessions_export
    rendered = render_sessions_export(sessions, fmt="markdown" if args.format == "md" else "jsonl", only=args.only)
    count, noun = export_record_count(sessions, only=args.only)
    return rendered, f"Exported {count} {noun}{'' if count == 1 else 's'} to {args.output}"


def _render_html(args, sessions):
    """One self-contained file (single session, or multi-session with sidebar)."""
    from hermes_cli.session_export_html import generate_html_export, generate_multi_session_html_export
    single = len(sessions) == 1
    content = generate_html_export(sessions[0]) if single else generate_multi_session_html_export(sessions)
    return content, f"Exported {len(sessions)} {'session' if single else 'sessions'} to {args.output} (HTML)"


def _render_jsonl(args, sessions):
    lines = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in sessions)
    return lines, f"Exported {len(sessions)} {'session' if args.session_id else 'sessions'} to {args.output}"


#: kind -> (usage error when the --output/--format combination is unusable, renderer)
_FLAT_EXPORTERS = {
    "only": (
        lambda a: a.format not in ("jsonl", "md"), "--only user-prompts supports --format jsonl or md.", _render_only
    ),
    "html": (lambda a: not a.output or a.output == "-", "HTML export requires an output file path.", _render_html),
    "jsonl": (lambda a: not a.output, "JSONL export requires an output path (use - for stdout).", _render_jsonl),
}


def _export_flat(kind, args, collect):
    """Single-file export: validate the output target, collect sessions, render, write."""
    unusable, message, render = _FLAT_EXPORTERS[kind]
    if unusable(args):
        print(message)
        return
    sessions = collect()
    if sessions is not None:
        _write_output(args.output, *render(args, sessions))


def _export_trace(db, args, filters):
    """Claude Code JSONL trace export — local file or HF upload. Redaction is ON by default (traces
    leave the machine with --upload); --no-redact opts out."""
    session_id = args.session_id
    if not session_id and not filters:  # "the last thing I did"
        rows = db.list_sessions_rich(limit=1, order_by_last_active=True)
        session_id = rows[0].get("id") if rows else None
        if not session_id:
            print("No session found to export. Pass --session-id.")
            return
    if session_id and not db.resolve_session_id(session_id):
        _not_found(session_id)
        return
    from agent.trace_upload import TraceRedactionError, build_trace_jsonl, upload_session_trace
    redact_trace = not getattr(args, "no_redact", False)
    if getattr(args, "upload", False):
        if not session_id:
            print("--upload exports one session: pass --session-id (or drop filters to use the most recent).")
            return
        resolved = db.resolve_session_id(session_id)
        db.close()
        print(upload_session_trace(resolved, cwd="", redact=redact_trace, private=not getattr(args, "public", False)))
        return
    if session_id:
        ids = [db.resolve_session_id(session_id)]
    else:
        candidates = db.list_prune_candidates(**filters)
        if args.dry_run:
            return _print_dry_run_preview(candidates, filters)
        ids = [row["id"] for row in candidates]

    def _render_trace(sid):
        meta = db.get_session(sid) or {}
        messages = db.get_messages_as_conversation(sid)
        if not messages:
            return None
        return build_trace_jsonl(messages, session_id=sid, model=meta.get("model") or "", cwd="", redact=redact_trace)
    try:
        if len(ids) == 1:
            jsonl = _render_trace(ids[0])
            if not jsonl:
                print(f"No transcript to export for session '{ids[0]}'.")
                return
            _write_output(args.output, jsonl, f"Exported 1 session trace to {args.output}")
        else:
            out_dir = _export_dir(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            exported = 0
            for sid in ids:
                jsonl = _render_trace(sid)
                if jsonl:
                    (out_dir / f"{sid}.trace.jsonl").write_text(jsonl, encoding="utf-8")
                    exported += 1
            print(f"Exported {exported} session trace(s) to {out_dir}")
    except TraceRedactionError:
        print("Redaction failed; refusing to export unredacted trace content.")


def _export_markdown(db, args, filters, redact):
    """Markdown / QMD export: one file per session plus a manifest entry."""
    from hermes_cli.session_export_md import append_manifest_entry, write_session_markdown
    if args.output == "-":
        print("Markdown/QMD export writes files; stdout (-) is only supported with --format jsonl.")
        return
    output_dir = _export_dir(args.output)

    def _export_one(session_id: str, *, include_lineage: bool = False):
        data = db.export_session_lineage(session_id) if include_lineage else db.export_session(session_id)
        if not data:
            return None, None
        data = redact(data)
        path = write_session_markdown(data, output_dir, fmt=args.format, force=args.force)
        append_manifest_entry(output_dir, data, path, fmt=args.format)
        return data, path
    if args.delete_after_verified and not args.yes:
        print("--delete-after-verified requires --yes.")
        return
    if args.delete_after_verified and not args.session_id:
        print("--delete-after-verified is only supported with --session-id.")
        return
    lineage_is_logical = getattr(args, "lineage", "single") == "logical"
    if args.session_id:
        return _export_markdown_single(db, args, _export_one, output_dir, lineage_is_logical)
    if not filters:
        print("Refusing bulk export without a filter. Pass --session-id or "
              "at least one filter (e.g. --older-than 90, --source telegram).")
        return
    candidates = db.list_prune_candidates(**filters)
    if args.dry_run:
        return _print_dry_run_preview(candidates, filters)
    exported = 0
    for row in candidates:
        try:
            data, exported_path = _export_one(row["id"], include_lineage=lineage_is_logical)
        except FileExistsError as e:
            print(f"Skipping existing export: {e}. Pass --force to overwrite.")
            continue
        if data and exported_path:
            exported += 1
    print(f"Exported {exported} session(s) to {output_dir}")


def _export_markdown_single(db, args, export_one, output_dir, lineage_is_logical):
    """--session-id markdown export, optionally + verified delete of it and its delegates."""
    from hermes_cli.session_export_md import verify_export_file
    resolved_session_id = db.resolve_session_id(args.session_id)
    if not resolved_session_id:
        _not_found(args.session_id)
        return
    delete_target_ids = (
        db.get_session_delete_targets(resolved_session_id) if args.delete_after_verified else [resolved_session_id]
    )
    exported_items = []
    for target_id in delete_target_ids:
        try:
            data, exported_path = export_one(
                target_id, include_lineage=(target_id == resolved_session_id and lineage_is_logical),
            )
        except FileExistsError as e:
            print(f"Export already exists: {e}. Pass --force to overwrite.")
            return
        if not data or not exported_path:
            print(f"Session '{target_id}' disappeared during export; nothing was deleted.")
            return
        exported_items.append((data, exported_path))
    message_count = sum(len(data.get("messages") or []) for data, _path in exported_items)
    n = len(exported_items)
    print(f"Exported {n} session{'' if n == 1 else 's'} ({message_count} message{'' if message_count == 1 else 's'}) "
          f"to {exported_items[0][1] if n == 1 else output_dir}")
    if not args.delete_after_verified:
        return
    for data, exported_path in exported_items:
        ok, reason = verify_export_file(exported_path, data)
        if not ok:
            print(f"Export verification failed; not deleting session '{data.get('id')}': {reason}")
            return
    if not db.delete_session(
        resolved_session_id, sessions_dir=_sessions_dir(), expected_delete_ids=delete_target_ids
    ):
        print(f"Exported, but session '{resolved_session_id}' was not deleted because its delegate set changed.")
        return
    delegates = len(delete_target_ids) - 1
    delegate_suffix = f" and {delegates} delegate session{'' if delegates == 1 else 's'}" if delegates else ""
    print(f"Deleted exported session '{resolved_session_id}'{delegate_suffix}.")


# -- delete / prune / archive -------------------------------------------------

def _cmd_delete(db, args):
    resolved_session_id = db.resolve_session_id(args.session_id)
    if not resolved_session_id:
        return _not_found(args.session_id)
    # The delete is honored (explicit id), but a pin is a "keep" flag: say so instead of silently destroying it.
    _pinned_note = " (this session is PINNED)" if (db.get_session(resolved_session_id) or {}).get("pinned") else ""
    if not args.yes:
        if not _confirm_prompt(f"Delete session '{resolved_session_id}'{_pinned_note} and all its messages? [y/N] "):
            print("Cancelled.")
            return
    elif _pinned_note:
        print(f"Warning: deleting a pinned session '{resolved_session_id}'.")
    if not db.delete_session(resolved_session_id, sessions_dir=_sessions_dir()):
        return _not_found(args.session_id)
    print(f"Deleted session '{resolved_session_id}'.")


#: Age floor for `prune --never-active`; generous: a young never-active row may be a chat nobody replied to yet.
_NEVER_ACTIVE_DEFAULT_DAYS = 30.0


def _prune_never_active_keyed(db, args):
    """`prune --never-active`: drop keyed gateway rows opened and never used (mostly escaped test
    fixtures). Separate from the shared prune/archive selector, which is pinned to `ended_at IS NOT
    NULL` — never-closed rows sit outside it by construction.

    The population is dominated by escaped test fixtures (#82770), which the hermetic-isolation guard can
    only stop from being *created* — rows already written to a developer's state.db need a sweep to leave.
    """
    from hermes_cli.session_filters import format_epoch, parse_duration_seconds
    older_than = getattr(args, "older_than", None)
    days = _NEVER_ACTIVE_DEFAULT_DAYS
    if older_than is not None:
        seconds = parse_duration_seconds(str(older_than))
        if seconds is None:
            print(f"Error: --older-than '{older_than}' is not a duration. "
                  "Use a bare number of days or a form like '2d' / '1w'.")
            return
        days = seconds / 86400.0
    candidates = db.list_never_active_keyed_sessions(older_than_days=days)
    if not candidates:
        print(f"No never-active keyed sessions older than {days:g} day(s).")
        return
    shown = candidates if args.dry_run else candidates[:15]
    print(f"{len(candidates)} never-active keyed session(s) older than {days:g} day(s) "
          "— no messages, tokens, tool calls or title:")
    for s in shown:
        print(f"  {s['id']}  {format_epoch(s.get('started_at')):<17} {(s.get('source') or '-'):<10} "
              f"{s.get('session_key') or '-'}")
    if len(candidates) > len(shown):
        print(f"  … {len(candidates) - len(shown)} more")
    if args.dry_run:
        print("Dry run — nothing deleted.")
        return
    if not args.yes and not _confirm_prompt(f"Delete {len(candidates)} session(s)? [y/N] "):
        print("Aborted.")
        return
    deleted, routing_deleted = db.prune_never_active_keyed_sessions(
        older_than_days=days, sessions_dir=_sessions_dir()
    )
    print(f"Deleted {deleted} never-active session(s) and {routing_deleted} stale routing entr(ies).")


def _note_pinned_skipped(db, filters, action):
    """Tell the user how many pinned rows bulk prune/archive spared (pin = durable keep; only
    `prune --include-pinned` opts in, archive always spares them)."""
    _base = {k: v for k, v in filters.items() if k != "include_pinned"}
    with_pinned, without = (int(db.count_prune_matches(**_base, include_pinned=flag)) for flag in (True, False))
    skipped = max(with_pinned - without, 0)
    if not skipped:
        return
    suffix = "" if skipped == 1 else "s"
    if action == "prune":
        verb = "deleted"
        optin = "Pass --include-pinned to delete them anyway, or unpin first with `hermes sessions unpin <id>`."
    else:
        verb, optin = "archived", "Unpin first with `hermes sessions unpin <id>` to include them."
    print(f"Note: {skipped} pinned session{suffix} also match these filters but will NOT be {verb} "
          f"(pin is a keep flag). {optin}")


def _cmd_prune_or_archive(db, args, action):
    prune = action == "prune"
    if prune and getattr(args, "never_active", False):
        return _prune_never_active_keyed(db, args)
    from hermes_cli.session_filters import build_prune_filters, describe_filters, format_epoch
    # Bare `prune` keeps the historical "older than 90 days" default. ANY filter — including --source —
    # suppresses the implicit cutoff (`prune --source cron` matches ALL cron sessions); the preview +
    # confirmation below is the safety net.
    if prune and not _any_filter_args(args):
        args.older_than = "90"
    try:
        filters = build_prune_filters(args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    if not prune and not any(v for k, v in filters.items() if k != "older_than_days"):
        print("Refusing to archive every ended session: pass at least one "
              "filter (e.g. --newer-than 5h, --source cli, --title codex).")
        return

    # Prune skips archived rows unless --include-archived; archive only targets not-yet-archived rows.
    filters["archived"] = None if prune and getattr(args, "include_archived", False) else False
    filters["include_pinned"] = getattr(args, "include_pinned", False)
    if not filters["include_pinned"]:
        _note_pinned_skipped(db, filters, action)
    candidates = db.list_prune_candidates(**filters)
    # Archive expands each row to its compression lineage (may include open continuations), so a
    # direct-open count would misdescribe its effect.
    skipped_open = db.count_open_prune_matches(**filters) if prune else 0
    if skipped_open:
        print(f"Note: {skipped_open} open session{'' if skipped_open == 1 else 's'} also match these filters but "
              "will be skipped because prune only deletes ended sessions. Use `hermes sessions delete <id>` "
              "to remove one explicitly.")
    if not candidates:
        print(f"No sessions match ({describe_filters(filters)}).")
        return
    # Candidates are oldest-activity-first; show the span so a long-lived but recently used
    # conversation cannot look old merely by creation date.
    _span = (
        f"oldest activity {format_epoch(candidates[0].get('last_active'))}, "
        f"newest activity {format_epoch(candidates[-1].get('last_active'))}"
    )
    if args.dry_run or not args.yes:
        shown = candidates if args.dry_run else candidates[:15]
        print(f"{len(candidates)} session(s) match ({describe_filters(filters)}; {_span}):")
        for s in shown:
            model = (s.get("model") or "-").split("/")[-1][:24]
            print(f"  {s['id']}  {format_epoch(s.get('last_active')):<17} {s['source']:<10} {model:<24} "
                  f"{s['message_count']:>4} msgs  {(s.get('title') or '')[:36]}")
        if len(candidates) > len(shown):
            print(f"  … and {len(candidates) - len(shown)} more")
        if args.dry_run:
            print(f"Dry run — nothing {'deleted' if prune else 'archived'}.")
            return
    verb = "Delete" if prune else "Archive"
    if not args.yes and not _confirm_prompt(f"{verb} these {len(candidates)} session(s) ({_span})? [y/N] "):
        print("Cancelled.")
        return
    if prune:
        print(f"Pruned {db.prune_sessions(sessions_dir=_sessions_dir(), **filters)} session(s).")
    else:
        print(f"Archived {db.archive_sessions(**filters)} session(s). They're hidden from listings "
              "but fully recoverable (nothing was deleted).")


# -- titles / pins -----------------------------------------------------------

def _cmd_rename(db, args):
    resolved_session_id = db.resolve_session_id(args.session_id)
    if not resolved_session_id:
        return _not_found(args.session_id)
    title = " ".join(args.title)
    # Empty titles render as "—" and newlines corrupt the `list` table; length is validated in set_session_title.
    if not title.strip():
        print("Error: title cannot be empty or whitespace-only.")
        return 1
    if "\n" in title or "\r" in title:
        print("Error: title cannot contain newlines.")
        return 1
    try:
        if not db.set_session_title(resolved_session_id, title):
            return _not_found(args.session_id)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    print(f"Session '{resolved_session_id}' renamed to: {title}")


def _cmd_pin(db, args, pinning):
    """Durable "keep" flag (exempt from sessions.auto_archive, always listed); every surface shares the store."""
    failures = 0
    # Pinned sessions are exempt from the sessions.auto_archive stale sweep and always surface in listings;
    # until now only the Desktop sidebar could write the flag. Inspired by Perplexity Computer's
    # conversational session management (pin/archive from any surface): pin state is operational
    # infrastructure, so every surface — GUI, TUI, CLI, scripts — needs read/write access to the same store.
    # See #52955.
    for raw_id in args.session_ids:
        resolved = db.resolve_session_id(raw_id)
        if resolved and db.set_session_pinned(resolved, pinning):
            title = db.get_session_title(resolved)
            print(f"{'Pinned' if pinning else 'Unpinned'} session '{resolved}'.{f'  ({title})' if title else ''}")
        else:
            failures += _not_found(raw_id)
    return 1 if failures else None


def _cmd_pinned(db, args):
    # limit=1 keeps the recency page minimal; include_pinned back-fills ALL pinned rows the page missed.
    rows = db.list_sessions_rich(limit=1, include_pinned=True, exclude_sources=_default_exclude(args))
    pinned_rows = [s for s in rows if s.get("pinned")]
    if getattr(args, "json", False):
        keys = ("title", "source", "last_active", "message_count")
        print(json.dumps([{"id": s["id"], **{k: s.get(k) for k in keys}} for s in pinned_rows], indent=2))
        return
    if not pinned_rows:
        print("No pinned sessions. Pin one with: hermes sessions pin <session_id>")
        return
    print(f"{'Title':<32} {'Last Active':<13} {'Src':<9} {'ID'}\n" + "─" * 100)
    for s in pinned_rows:
        title = (s.get("title") or s.get("preview", "") or "—")[:30]
        print(f"{title:<32} {_relative_time(s.get('last_active')):<13} {(s.get('source') or '-'):<9} {s['id']}")


def _cmd_retitle_skills(db, args):
    from agent.skill_commands import describe_skill_invocation
    from agent.title_generator import generate_title
    limit = max(1, int(getattr(args, "limit", 200) or 200))
    apply_changes = bool(getattr(args, "apply", False))
    candidates = db.list_skill_scaffolded_sessions(limit=limit)
    if not candidates:
        print("No sessions were titled from a /skill invocation.")
        return
    mode = "" if apply_changes else " (dry run — pass --apply to write)"
    print(f"{len(candidates)} session(s) opened with a /skill{mode}:")
    changed = 0
    for row in candidates:
        session_id = row["id"]
        typed = describe_skill_invocation(row["content"]) or ""
        new_title = generate_title(typed)
        if not new_title or new_title == row["title"]:
            continue
        if not new_title[0].isalnum():
            # Non-title: an auxiliary model occasionally answers the prompt instead of titling it
            # ('$ df -h /'). This is a REPAIR — never replace a serviceable title with that.
            print(f"  {session_id}\n    kept {row['title']!r} — got {new_title!r}")
            continue
        print(f"  {session_id}\n    {row['title']!r}\n    → {new_title!r}")
        changed += 1
        if not apply_changes:
            continue
        try:
            db.set_session_title(session_id, new_title)
        except ValueError:  # unique-title collision: dedupe like the live auto-titler (base #2, #3, ...)
            deduped = db.get_next_title_in_lineage(new_title)
            try:
                db.set_session_title(session_id, deduped)
                print(f"    (renamed to {deduped!r} — title was taken)")
            except ValueError as e:
                print(f"    skipped: {e}")
                changed -= 1
    if not changed:
        print("  every title already reflects the user's request.")
    elif apply_changes:
        print(f"✓ Re-titled {changed} session(s).")


def _cmd_browse(db, args):
    limit = getattr(args, "limit", 500) or 500
    sessions = db.list_sessions_rich(
        source=getattr(args, "source", None), exclude_sources=_default_exclude(args), limit=limit
    )
    if not sessions:
        db.close()
        print("No sessions found.")
        return
    try:  # keep the DB open: the picker uses it for status tags and 'd' delete
        selected_id = _session_browse_picker(sessions, session_db=db)
    finally:
        db.close()
    if not selected_id:
        print("Cancelled.")
        return
    print(f"Resuming session: {selected_id}")
    from hermes_cli.relaunch import relaunch
    relaunch(["--resume", selected_id])  # won't return after execvp


# -- storage maintenance -----------------------------------------------------

def _print_size_change(db, before_mb, prefix=""):
    """Report before/after size, preferring SQLite's page accounting over stat(): in WAL mode a VACUUM's
    rewrite sits in the -wal file until a checkpoint (refused while a live gateway holds a read-mark),
    so the main file lags and stat() can even go negative."""
    logical_after = db.logical_size_bytes()
    after_mb = logical_after / (1024 * 1024) if logical_after is not None else _size_mb(db.db_path)
    delta = _size_delta_label(before_mb - after_mb)
    print(f"{prefix}Database size: {before_mb:.1f} MB -> {after_mb:.1f} MB ({delta})")


def _cmd_optimize(db, args):
    before_mb = _size_mb(db.db_path)
    print("Optimizing session store (FTS merge + VACUUM)…")
    try:
        n = db.vacuum()  # merges FTS5 segments then VACUUMs; returns indexes merged
    except Exception as e:
        print(f"Error: optimization failed: {e}")
        return
    print(f"Optimized {n} FTS index(es).")
    _print_size_change(db, before_mb)


def _cmd_clean_markers(db, args):
    print(f"{'Dry run — scanning' if args.dry_run else 'Scanning'} for stale tool-call marker rows (#78148)…")
    report = db.purge_stale_tool_call_markers(dry_run=args.dry_run, backup=not args.no_backup)
    if report["rows_affected"] == 0:
        print("✓ No affected rows found — nothing to clean.")
    elif args.dry_run:
        print(f"Would clear {report['rows_affected']} row(s): ids {report['row_ids']}")
    else:
        if report["backup_path"]:
            print(f"  backup: {report['backup_path']}")
        print(f"✓ Cleared {report['rows_affected']} row(s).")


def _cmd_optimize_storage(db, args):
    db_path = db.db_path
    if not db.fts_optimize_available():
        print("Search index is already on the compact layout — nothing to do.")
        return
    before_bytes = os.path.getsize(db_path) if db_path.exists() else 0
    before_mb = before_bytes / (1024 * 1024)
    # Disk preflight: the new index is built before the old is torn down, and VACUUM needs a full
    # second copy — require headroom ≈ current file size.
    do_vacuum = not getattr(args, "no_vacuum", False)
    try:
        free_bytes = shutil.disk_usage(db_path.parent).free
    except Exception:
        free_bytes = None
    need_bytes = before_bytes if do_vacuum else int(before_bytes * 0.3)
    print(f"Search-index optimization for {db_path}\n  Current database size: {before_mb:.1f} MB")
    if free_bytes is not None:
        print(f"  Free disk: {free_bytes / (1024*1024):.0f} MB "
              f"(need ~{need_bytes / (1024*1024):.0f} MB to complete{' incl. VACUUM' if do_vacuum else ''})")
        if free_bytes < need_bytes:
            print("\n⚠ Not enough free disk to complete safely. Free up space, or run with --no-vacuum "
                  "(rebuilds the index but doesn't reclaim space until a later VACUUM).")
            return
    if before_mb > 500:
        print("  This may take a while on a large database. It runs in the foreground with progress below; "
              "safe to Ctrl-C and re-run (it resumes).")
    if not getattr(args, "yes", False):
        try:
            resp = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            resp = ""
        if resp not in ("y", "yes"):
            print("Cancelled.")
            return
    _last = {"phase": None}
    labels = {"teardown": "Reclaiming old index", "vacuum": "Compacting database (VACUUM)", "done": "Done"}

    def _progress(info):
        phase = info.get("phase")
        if phase == "backfill":
            print(f"\r  Rebuilding index: {info.get('percent', 0):3d}% "
                  f"({info.get('indexed',0):,}/{info.get('total',0):,})", end="", flush=True)
        elif phase != _last["phase"]:
            print(f"\n  {labels.get(phase, phase)}…", flush=True)
        _last["phase"] = phase
    print("Optimizing search-index storage…")
    try:
        result = db.optimize_fts_storage(progress_cb=_progress, vacuum=do_vacuum)
    except Exception as e:
        print(f"\nError: optimization failed: {e}\nNo data was lost. Re-run to resume.")
        return
    if not result.get("ok"):
        print(f"\nCould not optimize: {result.get('reason', 'unknown')}")
        return
    print("\n✓ Search index optimized.")
    _print_size_change(db, before_mb, prefix="  ")
    if result.get("vacuumed") is False:
        print("  (VACUUM was skipped or failed — run `hermes sessions optimize` later to reclaim freed space.)")


def _cmd_repair_routing(db, args):
    records = db.find_orphaned_gateway_sessions(max_gap_s=getattr(args, "max_gap_seconds", None))
    adoptable = [r for r in records if r["adoptable"]]
    for record in records:
        print(f"{record['orphan_id']}  ({record['source']}, {record['message_count']} messages)")
        if record["adoptable"]:
            print(f"  → adopt into {record['session_key']} (from {record['donor_id']}, "
                  f"evidence: {record['evidence']})")
        else:
            print(f"  ✗ not repairable — {record['reason']}")
    if not records:
        print("✓ No gateway sessions are missing their routing identity.")
        return
    if not adoptable:
        print(f"\n{len(records)} orphaned session(s) found, none unambiguously repairable. Nothing to do.")
        return
    if not getattr(args, "apply", False):
        print(f"\n{len(adoptable)} of {len(records)} orphaned session(s) can be repaired. "
              "Re-run with --apply to perform them.")
        return
    print("\nStop the gateway before applying — a running gateway still holds the old routing mapping in memory.")
    if not _confirm_prompt(f"Adopt {len(adoptable)} orphaned session(s)? [y/N] "):
        print("Aborted — nothing was changed.")
        return
    repaired = 0
    for record in adoptable:
        if db.adopt_orphaned_gateway_session(record["orphan_id"], record["donor_id"]):
            repaired += 1
            print(f"✓ {record['orphan_id']} now owns {record['session_key']}")
        else:
            print(f"✗ {record['orphan_id']} was not adopted (the row changed since it was reported)")
    print(f"\nRepaired {repaired} of {len(adoptable)} session(s).")


def _cmd_stats(db, args):
    print(f"Total sessions: {db.session_count()}\nTotal messages: {db.message_count()}")
    for src in ("cli", "telegram", "discord", "whatsapp", "slack"):
        if (c := db.session_count(source=src)) > 0:
            print(f"  {src}: {c} sessions")
    if db.db_path.exists():
        print(f"Database size: {_size_mb(db.db_path):.1f} MB")


# -- dispatch -----------------------------------------------------------------

_PRE_DB_HANDLERS = {"repair": _cmd_repair, "recover": _cmd_recover, "import": _cmd_import}
_DB_HANDLERS = {
    "list": _cmd_list, "export": _cmd_export, "delete": _cmd_delete, "rename": _cmd_rename, "pinned": _cmd_pinned,
    "prune": partial(_cmd_prune_or_archive, action="prune"), "pin": partial(_cmd_pin, pinning=True),
    "archive": partial(_cmd_prune_or_archive, action="archive"), "unpin": partial(_cmd_pin, pinning=False),
    "retitle-skills": _cmd_retitle_skills, "browse": _cmd_browse, "optimize": _cmd_optimize,
    "clean-markers": _cmd_clean_markers, "optimize-storage": _cmd_optimize_storage,
    "repair-routing": _cmd_repair_routing, "stats": _cmd_stats,
}


def cmd_sessions(args, sessions_parser=None):
    action = args.sessions_action
    pre = _PRE_DB_HANDLERS.get(action)
    if pre is not None:
        return pre(args)
    try:
        from hermes_state import SessionDB
        db = SessionDB()
    except Exception as e:
        print(f"Error: Could not open session database: {e}")
        return 1
    try:
        handler = _DB_HANDLERS.get(action)
        if handler is None:
            sessions_parser.print_help()
            return
        return handler(db, args)
    finally:
        db.close()
