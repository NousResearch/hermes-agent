"""``hermes sessions`` subcommand parser."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag, add_yes_flag


def _flag(parser, *names, help, **kw):
    parser.add_argument(*names, action="store_true", help=help, **kw)


def build_sessions_parser(subparsers, *, cmd_sessions: Callable) -> None:
    """Attach the ``sessions`` subcommand to ``subparsers``."""
    sessions_parser = subparsers.add_parser(
        "sessions", help="Manage session history (list, rename, export, prune, delete)",
        description="View and manage the SQLite session store")
    sessions_subparsers = sessions_parser.add_subparsers(dest="sessions_action")

    sessions_list = sessions_subparsers.add_parser("list", help="List recent sessions")
    sessions_list.add_argument("--source", help="Filter by source (cli, telegram, discord, etc.)")
    sessions_list.add_argument("--limit", type=int, default=20, help="Max sessions to show")
    sessions_list.add_argument("--workspace", metavar="NEEDLE",
        help="Only sessions in one workspace: a git repo root or project dir "
        "(matched by path substring or basename).")

    _filter_args = (
        ("--newer-than", dict(metavar="AGE", help="Only match sessions active within the last AGE "
            "(e.g. '5h', '2d') or after an ISO timestamp")),
        ("--before", dict(metavar="TIME", help="Only match sessions started before TIME "
            "(duration ago like '5h', or ISO timestamp like '2026-07-05 14:30')")),
        ("--after", dict(metavar="TIME", help="Only match sessions started at/after TIME "
            "(duration ago like '5h', or ISO timestamp)")),
        ("--source", dict(help="Only match sessions from this source")),
        ("--title", dict(help="Only match sessions whose title contains this substring")),
        ("--end-reason", dict(help="Only match sessions with this end reason")),
        ("--cwd", dict(help="Only match sessions whose working directory is under this path")),
        ("--min-messages", dict(type=int, help="Only match sessions with >= N messages")),
        ("--max-messages", dict(type=int, help="Only match sessions with <= N messages")),
        ("--model", dict(help="Only match sessions whose model name contains this substring "
            "(e.g. 'sonnet', 'gpt-5', 'hermes')")),
        ("--provider", dict(help="Only match sessions billed through this provider "
            "(e.g. openrouter, anthropic, nous)")),
        ("--user", dict(help="Only match sessions from this user ID")),
        ("--chat-id", dict(help="Only match sessions from this chat/channel ID")),
        ("--chat-type", dict(help="Only match sessions with this chat type (e.g. dm, group)")),
        ("--branch", dict(help="Only match sessions whose git branch contains this substring")),
        ("--min-tokens", dict(type=int,
            help="Only match sessions with >= N total tokens (input+output)")),
        ("--max-tokens", dict(type=int,
            help="Only match sessions with <= N total tokens (input+output)")),
        ("--min-cost", dict(type=float,
            help="Only match sessions costing >= N USD (actual or estimated)")),
        ("--max-cost", dict(type=float,
            help="Only match sessions costing <= N USD (actual or estimated)")),
        ("--min-tool-calls", dict(type=int, help="Only match sessions with >= N tool calls")),
        ("--max-tool-calls", dict(type=int, help="Only match sessions with <= N tool calls")),
        ("--dry-run", dict(action="store_true",
            help="List matching sessions without changing anything")))

    def _add_session_filter_args(p, default_older_help):
        p.add_argument("--older-than", metavar="AGE", help=default_older_help)
        for flag, kw in _filter_args:
            p.add_argument(flag, **kw)
        add_yes_flag(p, "Skip confirmation")

    sessions_export = sessions_subparsers.add_parser(
        "export", help="Export sessions to JSONL, Markdown, or QMD")
    sessions_export.add_argument("output", nargs="?",
        help="Output path. JSONL: file path (use - for stdout, required). "
            "md/qmd: output directory (default: <hermes home>/session-exports)")
    sessions_export.add_argument(
        "--format", choices=["jsonl", "md", "qmd", "html", "trace"], default="jsonl",
        help="Export format (default: jsonl). 'trace' emits Claude Code JSONL "
            "for the Hugging Face Agent Trace Viewer")
    _flag(sessions_export, "--upload",
        help="trace only: upload to your Hugging Face traces dataset instead "
            "of writing a local file (needs HF_TOKEN)")
    _flag(sessions_export, "--public",
        help="trace --upload only: create/update a public dataset instead of private")
    _flag(sessions_export, "--no-redact",
        help=("trace only: skip the forced secret redaction; only use after manual review"))
    sessions_export.add_argument("--only", choices=["user-prompts"],
        help="Export only a filtered view (user-prompts: one prompt record "
            "per line for jsonl, headed sections for md)")
    sessions_export.add_argument("--session-id", help="Session ID or unique prefix to export")
    _add_session_filter_args(
        sessions_export, "Only export sessions older than AGE (duration like '5h'/'2d', "
        "bare number of days, or an ISO timestamp)")
    _flag(sessions_export, "--redact",
        help="Redact secrets (API keys, tokens, credentials) from exported content")
    sessions_export.add_argument("--lineage", choices=["single", "logical"], default="single",
        help="md/qmd only: export one row or its compression lineage")
    _flag(sessions_export, "--delete-after-verified",
        help="md/qmd only: after verified single-session export, delete that session (needs --yes)")
    _flag(sessions_export, "--force", help="md/qmd only: overwrite an existing export file")

    sessions_delete = sessions_subparsers.add_parser("delete", help="Delete a specific session")
    sessions_delete.add_argument("session_id", help="Session ID to delete")
    add_yes_flag(sessions_delete, "Skip confirmation")

    sessions_prune = sessions_subparsers.add_parser(
        "prune", help="Delete old sessions (filterable by time window, source, title, ...)")
    _add_session_filter_args(
        sessions_prune, "Delete sessions older than AGE — days if bare number, or a duration "
        "like '5h'/'2d'/'1w', or an ISO timestamp (bare prune with no filters "
        "defaults to 90 days; any filter matches all ages)")
    _flag(sessions_prune, "--include-archived",
        help="Also delete archived sessions (excluded by default)")
    _flag(sessions_prune, "--include-pinned",
        help="Also delete pinned sessions (excluded by default — pin is a keep flag)")
    _flag(sessions_prune, "--never-active",
        help="Instead of ended sessions, delete keyed gateway rows that were "
            "opened and never used (no messages, tokens, tool calls or title) "
            "and are older than AGE (default 30 days). Ordinary prune can "
            "never reach these — it only ever selects ended sessions")

    sessions_archive = sessions_subparsers.add_parser(
        "archive", help="Bulk-archive (soft-hide) sessions matching filters — no deletion")
    _add_session_filter_args(
        sessions_archive, "Only archive sessions older than AGE (duration like '5h'/'2d', "
        "bare number of days, or ISO timestamp)")

    sessions_subparsers.add_parser(
        "optimize", help="Reclaim disk space: merge FTS5 segments + VACUUM (no data change)")

    sessions_clean_markers = sessions_subparsers.add_parser("clean-markers",
        help="Permanently clear stale tool-call marker content left by sessions from before #78148",
        description="Before the #78148 fix, a local tool-call template could persist a "
            "bare bracketed marker (e.g. \"[memory]\") as an assistant turn's "
            "content instead of real text. This is already repaired in memory "
            "on every session load, so running this is optional — it rewrites "
            "the affected rows once, in place, so long-lived sessions stop "
            "re-scanning/re-repairing the same rows on every resume. Only the "
            "content column is touched; tool_calls and every other column on "
            "the row are left untouched.")
    _flag(sessions_clean_markers, "--dry-run", default=False,
        help="Report the affected row count without writing")
    _flag(sessions_clean_markers, "--no-backup", default=False,
        help="Skip the timestamped state.db backup taken before writing (not recommended)")

    sessions_optimize_storage = sessions_subparsers.add_parser("optimize-storage",
        help="Migrate the search index to the compact v23 layout (reclaims disk on large DBs)",
        description="Rebuild the full-text search index in the compact v23 "
            "external-content layout. On large databases this reclaims a "
            "large fraction of state.db (the old layout stored duplicate "
            "copies of every message and indexed tool output). Runs "
            "foreground with a progress bar, throttles so a running gateway "
            "stays responsive, and VACUUMs at the end. Safe to interrupt and "
            "re-run — it resumes where it left off. No conversation data is "
            "changed; only the search index is rebuilt.")
    _flag(sessions_optimize_storage, "--no-vacuum", default=False,
        help="Skip the final VACUUM (index is rebuilt but freed pages aren't returned to the OS until a later VACUUM)")
    _flag(sessions_optimize_storage, "--yes", "-y", default=False,
        help="Skip the disk-space confirmation prompt")

    sessions_repair = sessions_subparsers.add_parser(
        "repair", help="Repair a malformed state.db schema so hidden sessions reappear",
        description="Recover a state.db whose schema is malformed (e.g. 'table "
            "messages_fts already exists'), which makes Desktop/Dashboard show "
            "no sessions. A backup is made first; sessions and messages are "
            "preserved and the FTS search index is rebuilt if needed.")
    _flag(sessions_repair, "--check-only",
        help="Only report whether the database opens cleanly; do not modify it")
    _flag(sessions_repair, "--no-backup", help="Skip the timestamped backup copy (not recommended)")

    sessions_repair_routing = sessions_subparsers.add_parser(
        "repair-routing", help="Re-stamp gateway sessions that lost their routing identity",
        description="Find gateway conversations stranded in session rows whose "
            "routing identity (session_key/chat_id/origin) was never "
            "written — the damage a corrupt state.db write path leaves "
            "behind (#82616). Such a row is invisible to restart recovery, "
            "so the chat resumes an older session instead. Re-stamps each "
            "orphan from the keyed predecessor it continues, and only when "
            "that predecessor is unambiguous. Reports without touching the "
            "database unless --apply is given.")
    _flag(sessions_repair_routing, "--apply", help="Perform the adoptions (default: report only)")
    sessions_repair_routing.add_argument("--max-gap-seconds", type=float, default=None,
        help="Window between a keyed predecessor's last activity and an "
            "orphan's start for them to count as the same conversation "
            "(default: 900)")

    sessions_recover = sessions_subparsers.add_parser(
        "recover", help="Rebuild canonical session data into a separate clean database",
        description="Offline, non-destructive recovery for a damaged state.db. The "
            "source database and its WAL/SHM/rollback-journal sidecars are "
            "copied before SQLite opens anything. Canonical rows are rebuilt "
            "into a new output database; derived search indexes are recreated "
            "and the active database is never replaced automatically.")
    sessions_recover.add_argument("--source", type=Path, required=True,
        help="Source state.db or preserved backup to inspect/recover")
    sessions_recover.add_argument(
        "--output", type=Path, help="New recovery database path (required unless --inspect-only)")
    _flag(sessions_recover, "--inspect-only",
        help="Only report canonical table readability; do not create an output database")
    sessions_recover.add_argument("--work-dir", type=Path,
        help="Existing directory for the disposable source copy (defaults beside the output)")
    sessions_recover.add_argument("--chunk-size", type=int, default=1000,
        help="Rows committed per recovery batch (default: 1000)")
    _flag(sessions_recover, "--allow-partial",
        help="Best-effort salvage across damaged row ranges; the output remains "
            "separate and every skipped range is recorded")
    sessions_recover.add_argument(
        "--report", type=Path, help="JSON report path (defaults to <output>.recovery.json)")

    sessions_subparsers.add_parser("stats", help="Show session store statistics")

    sessions_rename = sessions_subparsers.add_parser(
        "rename", help="Set or change a session's title")
    sessions_rename.add_argument("session_id", help="Session ID to rename")
    sessions_rename.add_argument("title", nargs="+", help="New title for the session")

    sessions_pin = sessions_subparsers.add_parser(
        "pin", help="Pin session(s) — durable keep flag, exempt from auto-archive",
        description="Set the durable 'keep' flag on one or more sessions. Pinned "
            "sessions are exempt from the sessions.auto_archive stale sweep "
            "and always appear in listings. The same flag drives the Desktop "
            "sidebar's Pinned section — pin from either surface, both see it.")
    sessions_pin.add_argument(
        "session_ids", nargs="+", help="Session ID(s) or unique prefix(es) to pin")

    sessions_unpin = sessions_subparsers.add_parser(
        "unpin", help="Remove the pin (durable keep flag) from session(s)")
    sessions_unpin.add_argument(
        "session_ids", nargs="+", help="Session ID(s) or unique prefix(es) to unpin")

    sessions_pinned = sessions_subparsers.add_parser("pinned", help="List pinned sessions")
    add_json_flag(sessions_pinned, "Emit machine-readable JSON (for backup/restore scripting)")

    sessions_retitle = sessions_subparsers.add_parser(
        "retitle-skills", help="Re-title sessions whose auto-title came from a /skill's own text",
        description="Sessions opened with a /skill were auto-titled from the expanded "
            "message, which embeds the whole skill body — so the title "
            "describes the SKILL, not the request. This regenerates those "
            "titles from what the user actually typed. Lists what it would "
            "change unless --apply is passed.")
    _flag(sessions_retitle, "--apply", help="Write the new titles (default: dry run)")
    sessions_retitle.add_argument(
        "--limit", type=int, default=200, help="Maximum sessions to examine (default: 200)")

    sessions_browse = sessions_subparsers.add_parser(
        "browse", help="Interactive session picker — browse, search, and resume sessions")
    sessions_browse.add_argument("--source", help="Filter by source (cli, telegram, discord, etc.)")
    sessions_browse.add_argument(
        "--limit", type=int, default=500, help="Max sessions to load (default: 500)")

    sessions_import = sessions_subparsers.add_parser(
        "import", help="Import a Claude Code or Codex CLI session into Hermes",
        description="Pull a conversation started in Claude Code (~/.claude/projects) "
            "or Codex CLI (~/.codex/sessions) into the Hermes session store "
            "so it can be resumed with 'hermes --resume <id>'. The foreign "
            "files are only read, never modified.")
    sessions_import.add_argument("--from", dest="from_source", choices=["claude", "codex"],
        help="Which tool to import from (default: pick across both)")
    sessions_import.add_argument(
        "path", nargs="?", help="Path to a specific session JSONL file (skips the picker)")


    # cmd_sessions lives in hermes_cli/sessions_cmd.py; the parser is threaded
    # in because the fallthrough branch calls sessions_parser.print_help().
    # main() injects a lazy indirection so sessions_cmd is only imported when
    # the subcommand runs.
    def _dispatch_sessions(_args, *, sessions_parser=sessions_parser):
        return cmd_sessions(_args, sessions_parser=sessions_parser)

    sessions_parser.set_defaults(func=_dispatch_sessions)
