"""``hermes sessions`` command — extracted from ``hermes_cli/main.py``.

Mechanical move (main.py decomposition): ``cmd_sessions`` was a ``def`` nested
inside ``main()``'s body; its dispatch on ``args.sessions_action`` is lifted
byte-identical. A symtable/AST closure check found exactly two free variables:

* ``_confirm_prompt`` — a sibling nested def with zero captures of its own;
  moved here to module level, byte-identical.
* ``sessions_parser`` — a ``main()``-local (the argparse subparser, used only
  for ``sessions_parser.print_help()`` in the fallthrough branch). It is
  threaded as a keyword parameter via ``functools.partial`` at the
  ``set_defaults(func=...)`` wiring site in ``main()``.

Helpers that stay in ``hermes_cli.main`` (``get_hermes_home``,
``_relative_time``, ``_session_browse_picker``, ``_size_delta_label``) are
delegated through call-time wrappers below so existing test monkeypatches on
``hermes_cli.main.<name>`` keep reaching this code path, and so imports stay
one-way (main.py imports this module; the reverse happens only lazily at call
time — no import cycle).
"""

import os
import sys
from pathlib import Path


def _m():
    """Lazy ``hermes_cli.main`` reference (call-time, keeps patches working)."""
    from hermes_cli import main

    return main


def get_hermes_home():
    return _m().get_hermes_home()


def _relative_time(ts):
    return _m()._relative_time(ts)


def _session_browse_picker(sessions, session_db=None):
    return _m()._session_browse_picker(sessions, session_db=session_db)


def _size_delta_label(saved_mb):
    return _m()._size_delta_label(saved_mb)


def _confirm_prompt(prompt: str) -> bool:
    """Prompt for y/N confirmation, safe against non-TTY environments."""
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


#: Default age floor for `hermes sessions prune --never-active`.  Deliberately
#: generous: the rows are worthless but harmless, and a young never-active row
#: may simply be a chat that nobody has replied to yet.
_NEVER_ACTIVE_DEFAULT_DAYS = 30.0


def _prune_never_active_keyed(db, args):
    """`hermes sessions prune --never-active` — drop leaked/dead keyed rows.

    Targets keyed gateway rows that were opened and never used at all.  The
    population is dominated by escaped test fixtures (#82770), which the
    hermetic-isolation guard can only stop from being *created* — rows already
    written to a developer's state.db need a sweep to leave.
    """
    from hermes_cli.session_filters import format_epoch, parse_duration_seconds

    older_than = getattr(args, "older_than", None)
    if older_than is None:
        days = _NEVER_ACTIVE_DEFAULT_DAYS
    else:
        seconds = parse_duration_seconds(str(older_than))
        if seconds is None:
            print(
                f"Error: --older-than '{older_than}' is not a duration. "
                "Use a bare number of days or a form like '2d' / '1w'."
            )
            return
        days = seconds / 86400.0

    candidates = db.list_never_active_keyed_sessions(older_than_days=days)
    if not candidates:
        print(f"No never-active keyed sessions older than {days:g} day(s).")
        return

    shown = candidates if args.dry_run else candidates[:15]
    print(
        f"{len(candidates)} never-active keyed session(s) older than "
        f"{days:g} day(s) — no messages, tokens, tool calls or title:"
    )
    for s in shown:
        print(
            f"  {s['id']}  {format_epoch(s.get('started_at')):<17} "
            f"{(s.get('source') or '-'):<10} {s.get('session_key') or '-'}"
        )
    if len(candidates) > len(shown):
        print(f"  … {len(candidates) - len(shown)} more")

    if args.dry_run:
        print("Dry run — nothing deleted.")
        return
    if not args.yes and not _confirm_prompt(
        f"Delete {len(candidates)} session(s)? [y/N] "
    ):
        print("Aborted.")
        return

    sessions_dir = get_hermes_home() / "sessions"
    deleted, routing_deleted = db.prune_never_active_keyed_sessions(
        older_than_days=days, sessions_dir=sessions_dir
    )
    print(
        f"Deleted {deleted} never-active session(s) and {routing_deleted} "
        "stale routing entr(ies)."
    )


def cmd_sessions(args, sessions_parser=None):
    import json as _json

    action = args.sessions_action

    # 'repair' and 'recover' must run BEFORE opening SessionDB(): a
    # malformed schema is exactly the case where SessionDB() can't open.
    # Recovery additionally promises never to open the supplied source
    # directly, so it operates through its own disposable source copy.
    if action == "repair":
        from hermes_state import (
            DEFAULT_DB_PATH,
            _db_opens_cleanly,
            repair_state_db_schema,
        )

        db_path = DEFAULT_DB_PATH
        if not db_path.exists():
            print(f"No session database at {db_path} (nothing to repair).")
            return
        reason = _db_opens_cleanly(db_path)
        if reason is None:
            print(f"✓ {db_path} opens cleanly — no repair needed.")
            return
        print(f"✗ {db_path} does not open cleanly: {reason}")
        if getattr(args, "check_only", False):
            return
        print("Repairing (a backup copy is made first)…")
        report = repair_state_db_schema(
            db_path, backup=not getattr(args, "no_backup", False)
        )
        if report.get("repaired"):
            if report.get("backup_path"):
                print(f"  backup: {report['backup_path']}")
            print(f"  strategy: {report.get('strategy')}")
            try:
                from hermes_state import SessionDB

                _repair_db = SessionDB()
                try:
                    n = _repair_db._conn.execute(
                        "SELECT COUNT(*) FROM sessions"
                    ).fetchone()[0]
                    print(f"✓ Repaired — {n} sessions recovered.")
                finally:
                    _repair_db.close()
            except Exception:
                print("✓ Repaired.")
        else:
            print(f"✗ Repair failed: {report.get('error')}")
            if report.get("backup_path"):
                print(f"  A backup is preserved at: {report['backup_path']}")
            print("  Keep state.db and the backup; do not delete them.")
            # Without this pointer the user is at a dead end: in-place
            # repair has failed and nothing tells them the non-destructive
            # offline recovery path exists. Lead with --inspect-only so
            # they confirm the data is readable before writing anything.
            print("")
            print("  Next step — offline recovery (never modifies the source):")
            source_hint = report.get("backup_path") or db_path
            print(f"    hermes sessions recover --source {source_hint} \\")
            print("        --inspect-only")
            print("  If that reports the data is recoverable, rebuild it into")
            print("  a NEW database (the active one is left untouched):")
            print(f"    hermes sessions recover --source {source_hint} \\")
            print("        --output recovered-state.db")
        return

    if action == "recover":
        import sqlite3 as _sqlite3

        from hermes_cli.session_recovery import (
            SessionRecoveryError,
            inspect_session_database,
            recover_session_database,
            write_recovery_report,
        )

        source = args.source
        output = getattr(args, "output", None)
        inspect_only = bool(getattr(args, "inspect_only", False))
        allow_partial = bool(getattr(args, "allow_partial", False))
        report_path = getattr(args, "report", None)
        if inspect_only and output is not None:
            print("Error: --output cannot be used with --inspect-only.")
            return 2
        if inspect_only and allow_partial:
            print("Error: --allow-partial cannot be used with --inspect-only.")
            return 2
        if not inspect_only and output is None:
            print("Error: --output is required unless --inspect-only is used.")
            return 2
        if not inspect_only and report_path is None:
            report_path = output.with_name(output.name + ".recovery.json")
        if (
            report_path is not None
            and os.path.lexists(report_path.expanduser())
        ):
            print(f"Error: refusing to overwrite existing report: {report_path}")
            return 2

        try:
            if inspect_only:
                report = inspect_session_database(
                    source,
                    work_dir=getattr(args, "work_dir", None),
                )
            else:
                last_progress = {"table": None}

                def _recovery_progress(info):
                    table = info.get("table")
                    copied = int(info.get("copied_rows") or 0)
                    total = info.get("source_rows")
                    if table != last_progress["table"]:
                        if last_progress["table"] is not None:
                            print()
                        print(f"  {table}: ", end="", flush=True)
                        last_progress["table"] = table
                    suffix = f"/{int(total):,}" if total is not None else ""
                    print(f"\r  {table}: {copied:,}{suffix}", end="", flush=True)

                print("Recovering canonical session data into a new database…")
                report = recover_session_database(
                    source,
                    output,
                    work_dir=getattr(args, "work_dir", None),
                    chunk_size=getattr(args, "chunk_size", 1000),
                    progress_cb=_recovery_progress,
                    allow_partial=allow_partial,
                )
                if last_progress["table"] is not None:
                    print()
        except (SessionRecoveryError, OSError, _sqlite3.DatabaseError) as exc:
            print(f"Error: session recovery failed: {exc}")
            print("The supplied source database was not replaced or deleted.")
            return 1

        if report_path is not None:
            try:
                written_report = write_recovery_report(report_path, report)
            except (FileExistsError, OSError) as exc:
                print(f"Error: could not write recovery report: {exc}")
                return 1
            print(f"Recovery report: {written_report}")
        else:
            print(_json.dumps(report, indent=2, sort_keys=True))

        if inspect_only:
            return 0 if report.get("recoverable") else 1
        if report.get("complete"):
            print(f"✓ Recovered database verified at: {output}")
            print("  The active session database was not changed.")
            print("  Review the JSON report before installing this database.")
            return 0
        if allow_partial and report.get("verified"):
            counts = report.get("verification", {}).get("table_counts", {})
            if report.get("best_effort"):
                print(f"✓ BEST-EFFORT page-level salvage verified at: {output}")
                print(
                    "  The source table schemas were unreadable; rows were "
                    "rebuilt from raw pages via sqlite3 .recover and mapped "
                    "heuristically."
                )
            else:
                print(f"✓ Partial recovery output verified at: {output}")
            print(
                "  Recovered "
                f"{int(counts.get('sessions') or 0):,} sessions and "
                f"{int(counts.get('messages') or 0):,} messages."
            )
            print("  The active session database was not changed.")
            print(
                "  This output is incomplete. Review every skipped range "
                "and orphan count in the JSON report before installing it."
            )
            return 0
        print("✗ Recovery output did not pass every verification check.")
        print("  Do not install it. Review the JSON report for partial data or errors.")
        return 1

    if action == "import":
        from hermes_cli.foreign_sessions import run_sessions_import

        result = run_sessions_import(args)
        # A path was explicitly given but nothing imported → real error (bad
        # path, unknown source, no turns). Propagate a non-zero exit so
        # scripts can detect the failure (SES-04/SES-10). An interactive
        # picker cancel (no path) returning None is a normal no-op → exit 0.
        if result is None and getattr(args, "path", None):
            return 1
        return

    try:
        from hermes_state import SessionDB

        db = SessionDB()
    except Exception as e:
        print(f"Error: Could not open session database: {e}")
        return 1

    # Hide third-party tool sessions by default, but honour explicit --source
    _source = getattr(args, "source", None)
    _exclude = None if _source else ["tool"]

    if action == "list":
        from hermes_state import workspace_key as _ws_key

        sessions = db.list_sessions_rich(
            source=args.source,
            exclude_sources=_exclude,
            limit=args.limit,
            include_hidden=getattr(args, "include_hidden", False),
        )

        # Workspace filter: match a session by its workspace key (git repo
        # root, else cwd) — path substring or exact basename.
        _ws_filter = (getattr(args, "workspace", None) or "").strip()
        if _ws_filter:
            _needle = _ws_filter.lower()

            def _in_workspace(s):
                key = (_ws_key(s) or "").lower()
                return bool(key) and (
                    _needle in key or _needle == os.path.basename(key.rstrip("/\\"))
                )

            sessions = [s for s in sessions if _in_workspace(s)]

        if not sessions:
            print("No sessions found.")
            return

        # Short workspace label: the repo/dir basename, "—" when unbound. The
        # Workspace column only appears once at least one session carries one
        # (or when filtering), so all-unbound listings read as before.
        def _ws_label(s):
            key = _ws_key(s)
            return (os.path.basename(key.rstrip("/\\")) or key) if key else "—"

        has_ws = bool(_ws_filter) or any(_ws_key(s) for s in sessions)
        has_titles = any(s.get("title") for s in sessions)

        if has_ws:
            if has_titles:
                print(f"{'Title':<28} {'Workspace':<18} {'Last Active':<13} {'ID'}")
                print("─" * 110)
            else:
                print(f"{'Preview':<38} {'Workspace':<18} {'Last Active':<13} {'Src':<6} {'ID'}")
                print("─" * 100)
            for s in sessions:
                last_active = _relative_time(s.get("last_active"))
                ws = _ws_label(s)[:16]
                if has_titles:
                    title = (s.get("title") or "—")[:26]
                    print(f"{title:<28} {ws:<18} {last_active:<13} {s['id']}")
                else:
                    preview = s.get("preview", "")[:36]
                    print(f"{preview:<38} {ws:<18} {last_active:<13} {s['source']:<6} {s['id']}")
            return

        if has_titles:
            print(f"{'Title':<32} {'Preview':<40} {'Last Active':<13} {'ID'}")
            print("─" * 110)
        else:
            print(f"{'Preview':<50} {'Last Active':<13} {'Src':<6} {'ID'}")
            print("─" * 95)
        for s in sessions:
            last_active = _relative_time(s.get("last_active"))
            preview = (
                s.get("preview", "")[:38]
                if has_titles
                else s.get("preview", "")[:48]
            )
            if has_titles:
                title = (s.get("title") or "—")[:30]
                sid = s["id"]
                print(f"{title:<32} {preview:<40} {last_active:<13} {sid}")
            else:
                sid = s["id"]
                print(f"{preview:<50} {last_active:<13} {s['source']:<6} {sid}")

    elif action == "export":
        from hermes_cli.session_filters import (
            build_prune_filters,
            describe_filters,
        )

        _filter_arg_names = (
            "older_than", "newer_than", "before", "after",
            "source", "title", "end_reason", "cwd",
            "min_messages", "max_messages", "model", "provider",
            "user", "chat_id", "chat_type", "branch",
            "min_tokens", "max_tokens", "min_cost", "max_cost",
            "min_tool_calls", "max_tool_calls",
        )
        _any_filters = any(
            getattr(args, a, None) is not None for a in _filter_arg_names
        )
        filters = None
        if _any_filters:
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
            """Resolve --session-id / filters / bare export into a list
            of redacted session dicts, or None after printing an error."""
            if args.session_id:
                resolved = db.resolve_session_id(args.session_id)
                data = _redact(db.export_session(resolved)) if resolved else None
                if not data:
                    print(f"Session '{args.session_id}' not found.")
                    return None
                return [data]
            if filters:
                candidates = db.list_prune_candidates(**filters)
                if args.dry_run:
                    print(
                        f"Would export {len(candidates)} session(s) "
                        f"({describe_filters(filters)})."
                    )
                    for row in candidates[:100]:
                        print(f"  {row.get('id')}  {row.get('source', '')}")
                    if len(candidates) > 100:
                        print(f"  ... {len(candidates) - 100} more")
                    return None
                return [
                    s
                    for s in (
                        _redact(db.export_session(row["id"])) for row in candidates
                    )
                    if s
                ]
            if args.dry_run:
                print("--dry-run requires at least one filter.")
                return None
            return [_redact(s) for s in db.export_all(source=None)]

        # Prompt-only export (--only user-prompts): one prompt record per
        # line (jsonl) or headed sections (md). Delegates rendering to
        # hermes_cli.session_export.
        if getattr(args, "only", None):
            if args.format not in ("jsonl", "md"):
                print("--only user-prompts supports --format jsonl or md.")
                return
            from hermes_cli.session_export import (
                export_record_count,
                render_sessions_export,
            )

            sessions = _collect_sessions()
            if sessions is None:
                db.close()
                return
            rendered = render_sessions_export(
                sessions,
                fmt="markdown" if args.format == "md" else "jsonl",
                only=args.only,
            )
            if not args.output or args.output == "-":
                sys.stdout.write(rendered)
                db.close()
                return
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(rendered)
            count, noun = export_record_count(sessions, only=args.only)
            suffix = "" if count == 1 else "s"
            print(f"Exported {count} {noun}{suffix} to {args.output}")
            db.close()
            return

        # Standalone HTML export: one self-contained file (single session
        # or multi-session with sidebar navigation).
        if args.format == "html":
            if not args.output or args.output == "-":
                print("HTML export requires an output file path.")
                return
            from hermes_cli.session_export_html import (
                generate_html_export,
                generate_multi_session_html_export,
            )

            sessions = _collect_sessions()
            if sessions is None:
                db.close()
                return
            if len(sessions) == 1:
                content = generate_html_export(sessions[0])
            else:
                content = generate_multi_session_html_export(sessions)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(content)
            suffix = "" if len(sessions) == 1 else "s"
            print(f"Exported {len(sessions)} session{suffix} to {args.output} (HTML)")
            db.close()
            return

        # Claude Code JSONL trace export — local file or HF upload.
        # Redaction is ON by default for traces (they leave the machine
        # when --upload is used); --no-redact opts out after review.
        if args.format == "trace":
            if getattr(args, "only", None):
                print("--only user-prompts supports --format jsonl or md.")
                db.close()
                return
            session_id = args.session_id
            if not session_id and not filters:
                # Match the shell's common intent: "the last thing I did".
                rows = db.list_sessions_rich(limit=1, order_by_last_active=True)
                session_id = rows[0].get("id") if rows else None
                if not session_id:
                    print("No session found to export. Pass --session-id.")
                    db.close()
                    return
            if session_id and not db.resolve_session_id(session_id):
                print(f"Session '{session_id}' not found.")
                db.close()
                return

            from agent.trace_upload import (
                TraceRedactionError,
                build_trace_jsonl,
                upload_session_trace,
            )

            redact_trace = not getattr(args, "no_redact", False)

            if getattr(args, "upload", False):
                if not session_id:
                    print("--upload exports one session: pass --session-id (or drop filters to use the most recent).")
                    db.close()
                    return
                resolved = db.resolve_session_id(session_id)
                db.close()
                status = upload_session_trace(
                    resolved,
                    cwd="",
                    redact=redact_trace,
                    private=not getattr(args, "public", False),
                )
                print(status)
                return

            # Local trace file(s)
            def _trace_ids():
                if session_id:
                    return [db.resolve_session_id(session_id)]
                candidates = db.list_prune_candidates(**filters)
                if args.dry_run:
                    print(
                        f"Would export {len(candidates)} session(s) "
                        f"({describe_filters(filters)})."
                    )
                    for row in candidates[:100]:
                        print(f"  {row.get('id')}  {row.get('source', '')}")
                    if len(candidates) > 100:
                        print(f"  ... {len(candidates) - 100} more")
                    return None
                return [row["id"] for row in candidates]

            ids = _trace_ids()
            if ids is None:
                db.close()
                return

            def _render_trace(sid):
                meta = db.get_session(sid) or {}
                messages = db.get_messages_as_conversation(sid)
                if not messages:
                    return None
                return build_trace_jsonl(
                    messages,
                    session_id=sid,
                    model=meta.get("model") or "",
                    cwd="",
                    redact=redact_trace,
                )

            try:
                if len(ids) == 1:
                    jsonl = _render_trace(ids[0])
                    if not jsonl:
                        print(f"No transcript to export for session '{ids[0]}'.")
                        db.close()
                        return
                    if not args.output or args.output == "-":
                        sys.stdout.write(jsonl)
                    else:
                        with open(args.output, "w", encoding="utf-8") as f:
                            f.write(jsonl)
                        print(f"Exported 1 session trace to {args.output}")
                else:
                    out_dir = (
                        Path(args.output).expanduser()
                        if args.output and args.output != "-"
                        else get_hermes_home() / "session-exports"
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    exported = 0
                    for sid in ids:
                        jsonl = _render_trace(sid)
                        if not jsonl:
                            continue
                        (out_dir / f"{sid}.trace.jsonl").write_text(
                            jsonl, encoding="utf-8"
                        )
                        exported += 1
                    print(f"Exported {exported} session trace(s) to {out_dir}")
            except TraceRedactionError:
                print("Redaction failed; refusing to export unredacted trace content.")
            db.close()
            return

        if args.format == "jsonl":
            if not args.output:
                print("JSONL export requires an output path (use - for stdout).")
                return
            if args.session_id:
                resolved_session_id = db.resolve_session_id(args.session_id)
                if not resolved_session_id:
                    print(f"Session '{args.session_id}' not found.")
                    return
                data = _redact(db.export_session(resolved_session_id))
                if not data:
                    print(f"Session '{args.session_id}' not found.")
                    return
                line = _json.dumps(data, ensure_ascii=False) + "\n"
                if args.output == "-":

                    sys.stdout.write(line)
                else:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(line)
                    print(f"Exported 1 session to {args.output}")
            else:
                if filters:
                    candidates = db.list_prune_candidates(**filters)
                    if args.dry_run:
                        print(
                            f"Would export {len(candidates)} session(s) "
                            f"({describe_filters(filters)})."
                        )
                        for row in candidates[:100]:
                            print(f"  {row.get('id')}  {row.get('source', '')}")
                        if len(candidates) > 100:
                            print(f"  ... {len(candidates) - 100} more")
                        return
                    sessions = [
                        s
                        for s in (
                            db.export_session(row["id"]) for row in candidates
                        )
                        if s
                    ]
                else:
                    if args.dry_run:
                        print("--dry-run requires at least one filter.")
                        return
                    sessions = db.export_all(source=None)
                if args.output == "-":

                    for s in sessions:
                        sys.stdout.write(
                            _json.dumps(_redact(s), ensure_ascii=False) + "\n"
                        )
                else:
                    with open(args.output, "w", encoding="utf-8") as f:
                        for s in sessions:
                            f.write(
                                _json.dumps(_redact(s), ensure_ascii=False) + "\n"
                            )
                    print(f"Exported {len(sessions)} sessions to {args.output}")
            return

        # Markdown / QMD export
        from hermes_cli.session_export_md import (
            append_manifest_entry,
            verify_export_file,
            write_session_markdown,
        )

        if args.output == "-":
            print("Markdown/QMD export writes files; stdout (-) is only supported with --format jsonl.")
            db.close()
            return
        output_dir = Path(args.output).expanduser() if args.output else get_hermes_home() / "session-exports"

        def _export_one(session_id: str, *, include_lineage: bool = False):
            data = (
                db.export_session_lineage(session_id)
                if include_lineage
                else db.export_session(session_id)
            )
            if not data:
                return None, None
            data = _redact(data)
            path = write_session_markdown(
                data,
                output_dir,
                fmt=args.format,
                force=args.force,
            )
            append_manifest_entry(output_dir, data, path, fmt=args.format)
            return data, path

        if args.delete_after_verified and not args.yes:
            print("--delete-after-verified requires --yes.")
            db.close()
            return
        if args.delete_after_verified and not args.session_id:
            print("--delete-after-verified is only supported with --session-id.")
            db.close()
            return

        lineage_is_logical = getattr(args, "lineage", "single") == "logical"

        if args.session_id:
            resolved_session_id = db.resolve_session_id(args.session_id)
            if not resolved_session_id:
                print(f"Session '{args.session_id}' not found.")
                db.close()
                return
            delete_target_ids = [resolved_session_id]
            if args.delete_after_verified:
                delete_target_ids = db.get_session_delete_targets(
                    resolved_session_id
                )

            exported_items = []
            for target_id in delete_target_ids:
                try:
                    data, exported_path = _export_one(
                        target_id,
                        include_lineage=(
                            target_id == resolved_session_id
                            and lineage_is_logical
                        ),
                    )
                except FileExistsError as e:
                    print(
                        f"Export already exists: {e}. "
                        "Pass --force to overwrite."
                    )
                    db.close()
                    return
                if not data or not exported_path:
                    print(
                        f"Session '{target_id}' disappeared during export; "
                        "nothing was deleted."
                    )
                    db.close()
                    return
                exported_items.append((data, exported_path))

            message_count = sum(
                len(data.get("messages") or [])
                for data, _path in exported_items
            )
            suffix = "" if message_count == 1 else "s"
            if len(exported_items) == 1:
                print(
                    f"Exported 1 session ({message_count} message{suffix}) "
                    f"to {exported_items[0][1]}"
                )
            else:
                print(
                    f"Exported {len(exported_items)} sessions "
                    f"({message_count} message{suffix}) to {output_dir}"
                )
            if args.delete_after_verified:
                for data, exported_path in exported_items:
                    ok, reason = verify_export_file(exported_path, data)
                    if not ok:
                        print(
                            "Export verification failed; not deleting "
                            f"session '{data.get('id')}': {reason}"
                        )
                        db.close()
                        return
                sessions_dir = get_hermes_home() / "sessions"
                if db.delete_session(
                    resolved_session_id,
                    sessions_dir=sessions_dir,
                    expected_delete_ids=delete_target_ids,
                ):
                    delegate_count = len(delete_target_ids) - 1
                    delegate_suffix = (
                        ""
                        if not delegate_count
                        else f" and {delegate_count} delegate session"
                        f"{'' if delegate_count == 1 else 's'}"
                    )
                    print(
                        f"Deleted exported session '{resolved_session_id}'"
                        f"{delegate_suffix}."
                    )
                else:
                    print(
                        f"Exported, but session '{resolved_session_id}' was "
                        "not deleted because its delegate set changed."
                    )
            db.close()
            return

        if not filters:
            print(
                "Refusing bulk export without a filter. Pass --session-id or "
                "at least one filter (e.g. --older-than 90, --source telegram)."
            )
            db.close()
            return
        candidates = db.list_prune_candidates(**filters)
        if args.dry_run:
            print(
                f"Would export {len(candidates)} session(s) "
                f"({describe_filters(filters)})."
            )
            for row in candidates[:100]:
                print(f"  {row.get('id')}  {row.get('source', '')}")
            if len(candidates) > 100:
                print(f"  ... {len(candidates) - 100} more")
            db.close()
            return
        exported = 0
        for row in candidates:
            try:
                data, exported_path = _export_one(
                    row["id"],
                    include_lineage=lineage_is_logical,
                )
            except FileExistsError as e:
                print(f"Skipping existing export: {e}. Pass --force to overwrite.")
                continue
            if data and exported_path:
                exported += 1
        print(f"Exported {exported} session(s) to {output_dir}")

    elif action == "delete":
        resolved_session_id = db.resolve_session_id(args.session_id)
        if not resolved_session_id:
            print(f"Session '{args.session_id}' not found.")
            return 1
        # Note when the explicit target is pinned — the user named this id
        # directly so we honor the delete, but a pin is a "keep" flag and
        # silently destroying it (round-3 QA SES-01) is surprising.
        _get_session = getattr(db, "get_session", None)
        _meta = (_get_session(resolved_session_id) or {}) if callable(_get_session) else {}
        _pinned_note = " (this session is PINNED)" if _meta.get("pinned") else ""
        if not args.yes:
            if not _confirm_prompt(
                f"Delete session '{resolved_session_id}'{_pinned_note} "
                "and all its messages? [y/N] "
            ):
                print("Cancelled.")
                return
        elif _pinned_note:
            print(f"Warning: deleting a pinned session '{resolved_session_id}'.")
        sessions_dir = get_hermes_home() / "sessions"
        if db.delete_session(resolved_session_id, sessions_dir=sessions_dir):
            print(f"Deleted session '{resolved_session_id}'.")
        else:
            print(f"Session '{args.session_id}' not found.")
            return 1

    elif action == "prune" and getattr(args, "never_active", False):
        # Separate branch on purpose: the shared prune/archive selector is
        # pinned to `ended_at IS NOT NULL`, so never-closed rows sit outside
        # it by construction and cannot be expressed as one more filter.
        _prune_never_active_keyed(db, args)

    elif action in ("prune", "archive"):
        from hermes_cli.session_filters import (
            build_prune_filters,
            describe_filters,
            format_epoch,
        )

        # Preserve the historical default ONLY for a truly bare
        # `hermes sessions prune`: no time window and no filters at all
        # means "older than 90 days". ANY filter — including --source —
        # suppresses the implicit cutoff, so `prune --source cron`
        # matches ALL cron sessions regardless of age. The preview +
        # confirmation below (count, oldest/newest) is the safety net.
        _non_time_filters = any(
            getattr(args, a, None) is not None
            for a in (
                "source", "title", "end_reason", "cwd",
                "min_messages", "max_messages", "model", "provider",
                "user", "chat_id", "chat_type", "branch",
                "min_tokens", "max_tokens", "min_cost", "max_cost",
                "min_tool_calls", "max_tool_calls",
            )
        )
        if (
            action == "prune"
            and args.older_than is None
            and args.newer_than is None
            and args.before is None
            and args.after is None
            and not _non_time_filters
        ):
            args.older_than = "90"

        try:
            filters = build_prune_filters(args)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        if action == "archive" and not any(
            v for k, v in filters.items() if k != "older_than_days"
        ):
            print(
                "Refusing to archive every ended session: pass at least one "
                "filter (e.g. --newer-than 5h, --source cli, --title codex)."
            )
            return

        # Prune skips archived sessions unless --include-archived;
        # archive only targets not-yet-archived rows (idempotent).
        if action == "prune":
            filters["archived"] = (
                None if getattr(args, "include_archived", False) else False
            )
        else:
            filters["archived"] = False

        # Pinned sessions are excluded by default from bulk prune/archive
        # (pin = durable keep). `prune --include-pinned` opts in; archive has
        # no such flag, so archive always spares pinned rows. Surface a count
        # of pinned matches being skipped so the user knows they were spared.
        _include_pinned = getattr(args, "include_pinned", False)
        filters["include_pinned"] = _include_pinned
        _count_matches = getattr(db, "count_prune_matches", None)
        if not _include_pinned and callable(_count_matches):
            _base = {k: v for k, v in filters.items() if k != "include_pinned"}
            try:
                _with_pinned = int(_count_matches(**_base, include_pinned=True))
                _without_pinned = int(_count_matches(**_base, include_pinned=False))
                _pinned_skipped = max(_with_pinned - _without_pinned, 0)
            except TypeError:
                # A db double without include_pinned support — skip the note.
                _pinned_skipped = 0
            if _pinned_skipped:
                _suffix = "" if _pinned_skipped == 1 else "s"
                _verb_word = "deleted" if action == "prune" else "archived"
                _optin = (
                    "Pass --include-pinned to delete them anyway, or unpin "
                    "first with `hermes sessions unpin <id>`."
                    if action == "prune"
                    else "Unpin first with `hermes sessions unpin <id>` to include them."
                )
                print(
                    f"Note: {_pinned_skipped} pinned session{_suffix} also match "
                    f"these filters but will NOT be {_verb_word} (pin is a keep "
                    f"flag). {_optin}"
                )

        candidates = db.list_prune_candidates(**filters)
        # Archive expands each selected row to its compression lineage, which
        # can include open continuations; a direct-open count would therefore
        # describe the eventual archive effect inaccurately.
        skipped_open = (
            db.count_open_prune_matches(**filters) if action == "prune" else 0
        )
        if skipped_open:
            suffix = "" if skipped_open == 1 else "s"
            print(
                f"Note: {skipped_open} open session{suffix} also match these "
                "filters but will be skipped because prune only deletes ended "
                "sessions. Use `hermes sessions delete <id>` "
                "to remove one explicitly."
            )
        verb = "Delete" if action == "prune" else "Archive"
        if not candidates:
            print(f"No sessions match ({describe_filters(filters)}).")
            return

        # Candidates are ordered by activity oldest-first. Surface that
        # span so a long-lived but recently used conversation cannot look
        # old merely because of its creation date.
        _oldest = candidates[0].get("last_active")
        _newest = candidates[-1].get("last_active")
        _span = (
            f"oldest activity {format_epoch(_oldest)}, "
            f"newest activity {format_epoch(_newest)}"
        )

        if args.dry_run or not args.yes:
            shown = candidates if args.dry_run else candidates[:15]
            print(
                f"{len(candidates)} session(s) match "
                f"({describe_filters(filters)}; {_span}):"
            )
            for s in shown:
                title = (s.get("title") or "")[:36]
                model = (s.get("model") or "-").split("/")[-1][:24]
                print(
                    f"  {s['id']}  {format_epoch(s.get('last_active')):<17} "
                    f"{s['source']:<10} {model:<24} "
                    f"{s['message_count']:>4} msgs  {title}"
                )
            if len(candidates) > len(shown):
                print(f"  … and {len(candidates) - len(shown)} more")
            if args.dry_run:
                print(f"Dry run — nothing {'deleted' if action == 'prune' else 'archived'}.")
                return

        if not args.yes:
            if not _confirm_prompt(
                f"{verb} these {len(candidates)} session(s) ({_span})? [y/N] "
            ):
                print("Cancelled.")
                return

        if action == "prune":
            sessions_dir = get_hermes_home() / "sessions"
            count = db.prune_sessions(sessions_dir=sessions_dir, **filters)
            print(f"Pruned {count} session(s).")
        else:
            count = db.archive_sessions(**filters)
            print(
                f"Archived {count} session(s). They're hidden from listings "
                "but fully recoverable (nothing was deleted)."
            )

    elif action == "rename":
        resolved_session_id = db.resolve_session_id(args.session_id)
        if not resolved_session_id:
            print(f"Session '{args.session_id}' not found.")
            return 1
        title = " ".join(args.title)
        # Reject blank / whitespace-only / newline-bearing titles (SES-05):
        # an empty title renders as "—" and embedded newlines corrupt the
        # `list` table. length is validated in set_session_title; guard
        # emptiness + control chars here.
        if not title.strip():
            print("Error: title cannot be empty or whitespace-only.")
            return 1
        if "\n" in title or "\r" in title:
            print("Error: title cannot contain newlines.")
            return 1
        try:
            if db.set_session_title(resolved_session_id, title):
                print(f"Session '{resolved_session_id}' renamed to: {title}")
            else:
                print(f"Session '{args.session_id}' not found.")
                return 1
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif action == "unhide":
        # The `hidden` flag is a durable "don't show in the global Sessions
        # sidebar" marker set by surfaces that own their sessions (Bot Mode
        # plumbing rows, plugins, the REST PATCH on api_server). Until now
        # there was no way back out: a stale or wrongly-adopted pointer left
        # an ordinary conversation hidden from every listing with no CLI, UI
        # or documented recovery short of raw SQL on state.db. Unhide is the
        # recovery affordance for that bug class — the DB setter flips the
        # whole compression lineage as a unit (set_session_hidden), so one
        # id per conversation is enough.
        failures = 0
        for raw_id in args.session_ids:
            resolved = db.resolve_session_id(raw_id)
            if not resolved:
                print(f"Session '{raw_id}' not found.")
                failures += 1
                continue
            if db.set_session_hidden(resolved, False):
                title = db.get_session_title(resolved)
                suffix = f"  ({title})" if title else ""
                print(f"Unhidden session '{resolved}'.{suffix}")
            else:
                # resolve_session_id() already proved the row exists, so a
                # False here is the setter's "no rows changed" return — the
                # session (and its whole lineage) was already visible.
                # Reporting "not found" would be wrong AND would bump
                # failures toward exit 1, making an idempotent second unhide
                # of the same id look like a failure. set_session_hidden
                # contract: True only when at least one row changed.
                print(f"Session '{resolved}' is already visible — nothing to unhide.")
        if failures:
            return 1

    elif action in ("pin", "unpin"):
        # CLI surface for the durable "keep" flag (issue #52955). Pinned
        # sessions are exempt from the sessions.auto_archive stale sweep and
        # always surface in listings; until now only the Desktop sidebar
        # could write the flag. Inspired by Perplexity Computer's
        # conversational session management (pin/archive from any surface):
        # pin state is operational infrastructure, so every surface — GUI,
        # TUI, CLI, scripts — needs read/write access to the same store.
        pinning = action == "pin"
        failures = 0
        for raw_id in args.session_ids:
            resolved = db.resolve_session_id(raw_id)
            if not resolved:
                print(f"Session '{raw_id}' not found.")
                failures += 1
                continue
            if db.set_session_pinned(resolved, pinning):
                verb = "Pinned" if pinning else "Unpinned"
                title = db.get_session_title(resolved)
                suffix = f"  ({title})" if title else ""
                print(f"{verb} session '{resolved}'.{suffix}")
            else:
                print(f"Session '{raw_id}' not found.")
                failures += 1
        if failures:
            return 1

    elif action == "pinned":
        # List every pinned conversation regardless of age. limit=1 keeps the
        # recency page minimal; include_pinned back-fills ALL pinned rows the
        # page missed (bounded by pin count, see list_sessions_rich), so old
        # pins can't fall off a paging window.
        rows = db.list_sessions_rich(
            limit=1, include_pinned=True, exclude_sources=_exclude
        )
        pinned_rows = [s for s in rows if s.get("pinned")]
        if getattr(args, "json", False):
            payload = [
                {
                    "id": s["id"],
                    "title": s.get("title"),
                    "source": s.get("source"),
                    "last_active": s.get("last_active"),
                    "message_count": s.get("message_count"),
                }
                for s in pinned_rows
            ]
            print(_json.dumps(payload, indent=2))
            return
        if not pinned_rows:
            print(
                "No pinned sessions. Pin one with: hermes sessions pin <session_id>"
            )
            return
        print(f"{'Title':<32} {'Last Active':<13} {'Src':<9} {'ID'}")
        print("─" * 100)
        for s in pinned_rows:
            title = (s.get("title") or s.get("preview", "") or "—")[:30]
            last_active = _relative_time(s.get("last_active"))
            print(
                f"{title:<32} {last_active:<13} {(s.get('source') or '-'):<9} {s['id']}"
            )

    elif action == "retitle-skills":
        from agent.skill_commands import describe_skill_invocation
        from agent.title_generator import generate_title

        limit = max(1, int(getattr(args, "limit", 200) or 200))
        apply_changes = bool(getattr(args, "apply", False))

        def _is_titlelike(candidate: str) -> bool:
            """Reject a candidate that isn't a title at all.

            An auxiliary model occasionally answers the prompt instead of
            titling it and echoes the assistant's output ('$ df -h /'). The
            live path has no alternative and takes what it gets, but this is
            a REPAIR — replacing a serviceable title with command output
            would make things worse, so keep the old one.
            """
            return bool(candidate) and candidate[0].isalnum()

        candidates = db.list_skill_scaffolded_sessions(limit=limit)
        if not candidates:
            print("No sessions were titled from a /skill invocation.")
            return

        print(
            f"{len(candidates)} session(s) opened with a /skill"
            f"{'' if apply_changes else ' (dry run — pass --apply to write)'}:"
        )
        changed = 0
        for row in candidates:
            session_id = row["id"]
            typed = describe_skill_invocation(row["content"]) or ""
            new_title = generate_title(typed)
            if not new_title or new_title == row["title"]:
                continue
            if not _is_titlelike(new_title):
                print(f"  {session_id}\n    kept {row['title']!r} — got {new_title!r}")
                continue
            print(f"  {session_id}\n    {row['title']!r}\n    → {new_title!r}")
            changed += 1
            if not apply_changes:
                continue
            try:
                db.set_session_title(session_id, new_title)
            except ValueError:
                # Unique-title collision. Dedupe the same way the live
                # auto-titler does (base #2, base #3, ...) rather than
                # leaving the leaked title in place.
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

    elif action == "browse":
        limit = getattr(args, "limit", 500) or 500
        source = getattr(args, "source", None)
        _browse_exclude = None if source else ["tool"]
        sessions = db.list_sessions_rich(
            source=source, exclude_sources=_browse_exclude, limit=limit
        )
        if not sessions:
            db.close()
            print("No sessions found.")
            return

        # Keep the DB open: the picker uses it for lifecycle status tags and
        # the 'd' delete-with-confirmation action.
        try:
            selected_id = _session_browse_picker(sessions, session_db=db)
        finally:
            db.close()
        if not selected_id:
            print("Cancelled.")
            return

        # Launch hermes --resume <id> by replacing the current process
        print(f"Resuming session: {selected_id}")
        from hermes_cli.relaunch import relaunch

        relaunch(["--resume", selected_id])
        return  # won't reach here after execvp

    elif action == "optimize":
        db_path = db.db_path
        before_mb = (
            os.path.getsize(db_path) / (1024 * 1024)
            if db_path.exists()
            else 0.0
        )
        print("Optimizing session store (FTS merge + VACUUM)…")
        try:
            # vacuum() merges FTS5 segments (optimize_fts) then VACUUMs,
            # and returns the number of indexes it merged.
            n = db.vacuum()
        except Exception as e:
            print(f"Error: optimization failed: {e}")
            db.close()
            return
        after_mb = (
            os.path.getsize(db_path) / (1024 * 1024)
            if db_path.exists()
            else 0.0
        )
        # Same WAL caveat as optimize-storage: after a VACUUM the main file
        # on disk lags until the WAL is checkpointed back (refused while a
        # live gateway holds a read-mark), so stat() understates the win and
        # can go negative. SQLite's page accounting is correct immediately.
        logical_after = db.logical_size_bytes()
        if logical_after is not None:
            after_mb = logical_after / (1024 * 1024)
        saved = before_mb - after_mb
        print(f"Optimized {n} FTS index(es).")
        print(
            f"Database size: {before_mb:.1f} MB -> {after_mb:.1f} MB "
            f"({_size_delta_label(saved)})"
        )

    elif action == "clean-markers":
        if args.dry_run:
            print("Dry run — scanning for stale tool-call marker rows (#78148)…")
        else:
            print("Scanning for stale tool-call marker rows (#78148)…")
        report = db.purge_stale_tool_call_markers(
            dry_run=args.dry_run, backup=not args.no_backup
        )
        if report["rows_affected"] == 0:
            print("✓ No affected rows found — nothing to clean.")
        elif args.dry_run:
            print(
                f"Would clear {report['rows_affected']} row(s): "
                f"ids {report['row_ids']}"
            )
        else:
            if report["backup_path"]:
                print(f"  backup: {report['backup_path']}")
            print(f"✓ Cleared {report['rows_affected']} row(s).")

    elif action == "optimize-storage":
        db_path = db.db_path
        if not db.fts_optimize_available():
            print("Search index is already on the compact layout — nothing to do.")
            db.close()
            return

        before_bytes = os.path.getsize(db_path) if db_path.exists() else 0
        before_mb = before_bytes / (1024 * 1024)

        # Disk preflight: the rebuild adds the new index before the old is
        # torn down, and the final VACUUM needs a full second copy of the
        # file. Require headroom ≈ current file size to finish cleanly.
        do_vacuum = not getattr(args, "no_vacuum", False)
        try:
            import shutil as _shutil
            free_bytes = _shutil.disk_usage(db_path.parent).free
        except Exception:
            free_bytes = None
        need_bytes = before_bytes if do_vacuum else int(before_bytes * 0.3)
        print(f"Search-index optimization for {db_path}")
        print(f"  Current database size: {before_mb:.1f} MB")
        if free_bytes is not None:
            print(f"  Free disk: {free_bytes / (1024*1024):.0f} MB "
                  f"(need ~{need_bytes / (1024*1024):.0f} MB to complete"
                  f"{' incl. VACUUM' if do_vacuum else ''})")
            if free_bytes < need_bytes:
                print()
                print("⚠ Not enough free disk to complete safely. Free up "
                      "space, or run with --no-vacuum (rebuilds the index "
                      "but doesn't reclaim space until a later VACUUM).")
                db.close()
                return
        if before_mb > 500:
            print("  This may take a while on a large database. It runs in "
                  "the foreground with progress below; safe to Ctrl-C and "
                  "re-run (it resumes).")
        if not getattr(args, "yes", False):
            try:
                resp = input("Proceed? [y/N] ").strip().lower()
            except EOFError:
                resp = ""
            if resp not in ("y", "yes"):
                print("Cancelled.")
                db.close()
                return

        _last = {"phase": None}

        def _progress(info):
            phase = info.get("phase")
            pct = info.get("percent", 0)
            if phase == "backfill":
                print(f"\r  Rebuilding index: {pct:3d}% "
                      f"({info.get('indexed',0):,}/{info.get('total',0):,})",
                      end="", flush=True)
            elif phase != _last["phase"]:
                label = {"teardown": "Reclaiming old index",
                         "vacuum": "Compacting database (VACUUM)",
                         "done": "Done"}.get(phase, phase)
                print(f"\n  {label}…", flush=True)
            _last["phase"] = phase

        print("Optimizing search-index storage…")
        try:
            result = db.optimize_fts_storage(
                progress_cb=_progress, vacuum=do_vacuum
            )
        except Exception as e:
            print(f"\nError: optimization failed: {e}")
            print("No data was lost. Re-run to resume.")
            db.close()
            return
        if not result.get("ok"):
            print(f"\nCould not optimize: {result.get('reason', 'unknown')}")
            db.close()
            return
        after_mb = (
            os.path.getsize(db_path) / (1024 * 1024) if db_path.exists() else 0.0
        )
        # Prefer SQLite's own page accounting over stat(). In WAL mode a
        # VACUUM's rewrite sits in the -wal file until a checkpoint folds it
        # back, and that checkpoint is refused while another connection (a
        # live gateway) holds a read-mark — so the main file on disk still
        # reads at its pre-VACUUM size and keeps growing. stat()ing it here
        # reported "reclaimed -3820.1 MB" on a DB that had actually shrunk
        # 60%. page_count * page_size is correct immediately.
        logical_after = db.logical_size_bytes()
        if logical_after is not None:
            after_mb = logical_after / (1024 * 1024)
        saved = before_mb - after_mb
        print(f"\n✓ Search index optimized.")
        print(
            f"  Database size: {before_mb:.1f} MB -> {after_mb:.1f} MB "
            f"({_size_delta_label(saved)})"
        )
        if result.get("vacuumed") is False:
            print("  (VACUUM was skipped or failed — run "
                  "`hermes sessions optimize` later to reclaim freed space.)")

    elif action == "repair-routing":
        records = db.find_orphaned_gateway_sessions(
            max_gap_s=getattr(args, "max_gap_seconds", None)
        )
        adoptable = [r for r in records if r["adoptable"]]
        for record in records:
            print(f"{record['orphan_id']}  ({record['source']}, "
                  f"{record['message_count']} messages)")
            if record["adoptable"]:
                print(f"  → adopt into {record['session_key']} "
                      f"(from {record['donor_id']}, "
                      f"evidence: {record['evidence']})")
            else:
                print(f"  ✗ not repairable — {record['reason']}")

        if not records:
            print("✓ No gateway sessions are missing their routing identity.")
        elif not adoptable:
            print(f"\n{len(records)} orphaned session(s) found, none "
                  "unambiguously repairable. Nothing to do.")
        elif not getattr(args, "apply", False):
            print(f"\n{len(adoptable)} of {len(records)} orphaned session(s) "
                  "can be repaired. Re-run with --apply to perform them.")
        else:
            # A running gateway holds the old routing mapping in memory and
            # would write it back over the repair on its next save.
            print("\nStop the gateway before applying — a running gateway "
                  "still holds the old routing mapping in memory.")
            if _confirm_prompt(
                f"Adopt {len(adoptable)} orphaned session(s)? [y/N] "
            ):
                repaired = 0
                for record in adoptable:
                    if db.adopt_orphaned_gateway_session(
                        record["orphan_id"], record["donor_id"]
                    ):
                        repaired += 1
                        print(f"✓ {record['orphan_id']} now owns "
                              f"{record['session_key']}")
                    else:
                        print(f"✗ {record['orphan_id']} was not adopted "
                              "(the row changed since it was reported)")
                print(f"\nRepaired {repaired} of {len(adoptable)} session(s).")
            else:
                print("Aborted — nothing was changed.")

    elif action == "stats":
        total = db.session_count()
        msgs = db.message_count()
        print(f"Total sessions: {total}")
        print(f"Total messages: {msgs}")
        for src in ["cli", "telegram", "discord", "whatsapp", "slack"]:
            c = db.session_count(source=src)
            if c > 0:
                print(f"  {src}: {c} sessions")
        db_path = db.db_path
        if db_path.exists():
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            print(f"Database size: {size_mb:.1f} MB")

    else:
        sessions_parser.print_help()

    db.close()
