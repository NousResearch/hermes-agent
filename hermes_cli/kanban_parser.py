"""Argparse tree for ``hermes kanban …`` (``build_parser``).

The subcommand tree is declared as data — one ``_cmd(...)`` record per
subcommand holding its ``add_parser`` kwargs and an ordered tuple of
``add_argument`` specs — and materialised by ``_add_commands``. Order of
records and arguments is the order argparse renders in ``--help``.
"""

from __future__ import annotations

import argparse

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_notify as kbn


def _arg(*flags: str, **kw):
    return (flags, kw)


def _cmd(name: str, args=(), *, children=None, **parser_kw):
    """``children`` = ``(dest, [specs])`` for a nested subparser group."""
    return (name, parser_kw, tuple(args), children)


def _add_commands(sub: argparse._SubParsersAction, specs) -> None:
    for name, parser_kw, args, children in specs:
        p = sub.add_parser(name, **parser_kw)
        for flags, kw in args:
            p.add_argument(*flags, **kw)
        if children:
            dest, child_specs = children
            _add_commands(p.add_subparsers(dest=dest), child_specs)


def _json_flag(**kw):
    return _arg("--json", action="store_true", **kw)


def _reason(help: str):
    return _arg("--reason", help=help)


def _run_state_args(type_help: str):
    return (
        _arg("--state-type", choices=("status", "outcome"), help=f"With --state-name: {type_help}"),
        _arg("--state-name", metavar="VALUE",
             help="With --state-type: keep runs whose column equals this value"),
    )


def _triage_sweep_args(verb: str, Verb: str, noun: str):
    """Shared ``specify`` / ``decompose`` arguments."""
    return (
        _arg("task_id", nargs="?", help=f"Task id to {verb} (required unless --all is given)"),
        _arg("--all", dest="all_triage", action="store_true", help=f"{Verb} every task currently in the triage column"),
        _arg("--tenant", help="When used with --all, restrict the sweep to this tenant"),
        _arg("--author",
             help=f"Author name recorded on the audit comment (default: $HERMES_PROFILE or '{noun}')"),
        _json_flag(help="Emit one JSON object per task on stdout"),
    )


def _bulk_ids(verb: str):
    return _arg("--ids", nargs="+", help=f"Additional task ids to {verb} with the same reason (bulk mode)")


_TASK_ID = _arg("task_id")
_TASK_IDS = _arg("task_ids", nargs="+")
_SLUG = _arg("slug")
_TENANT = _arg("--tenant", help="Tenant namespace")
_PRIORITY = _arg("--priority", type=int, default=0, help="Priority tiebreaker")
_RECLAIM_REASON = _reason("Human-readable reason (recorded on the reclaimed event)")
_NOTIFY_TARGET = (
    _arg("--platform", required=True),
    _arg("--chat-id", required=True),
    _arg("--thread-id"),
)
_STEP_HANDOFF = (
    _arg("--summary", help="Structured handoff summary. Falls back to --result if omitted."),
    _arg("--metadata", help="JSON dict of structured facts to store on the latest completed run."),
)

_BOARD_SPECS = [
    _cmd("list", [
        _json_flag(),
        _arg("--all", action="store_true", help="Include archived boards too"),
    ], aliases=["ls"], help="List all boards with task counts"),
    _cmd("create", [
        _arg("slug", help="Board slug (kebab-case, e.g. atm10-server)"),
        _arg("--name", help="Human-readable display name (defaults to Title Case of slug)"),
        _arg("--description", help="Optional description"),
        _arg("--icon", help="Optional emoji or single-character icon for the dashboard"),
        _arg("--color", help="Optional hex color (e.g. '#8b5cf6') for the dashboard"),
        _arg("--switch", action="store_true", help="Switch to the new board after creating it"),
        _arg("--default-workdir", help="Default workspace path for tasks created on this board"),
    ], aliases=["new"], help="Create a new board"),
    _cmd("rm", [
        _SLUG,
        _arg("--delete", action="store_true",
             help="Hard-delete the board directory instead of archiving it. "
                  "Default is to move it to boards/_archived/ so it's recoverable."),
    ], aliases=["remove", "delete"], help="Archive (default) or delete a board"),
    _cmd("switch", [_SLUG], aliases=["use"], help="Set the active board for subsequent CLI calls"),
    _cmd("show", aliases=["current"], help="Print the currently-active board slug"),
    _cmd("rename", [_SLUG, _arg("name", help="New display name")],
         help="Change a board's human-readable display name (slug is immutable)"),
    _cmd("set-default-workdir", [
        _SLUG,
        _arg("path", nargs="?", help="Absolute path to use as default workdir. Omit to clear."),
    ], help="Set the default workspace path for tasks on a board"),
    _cmd("export", [
        _arg("slug", nargs="?", help="Board to export (default: the current board)"),
        _arg("-o", "--output", help="Archive path (default: ./<slug>.tar.gz)"),
        _arg("--no-attachments", action="store_true", help="Skip attachment files, keeping the archive small"),
        _arg("--include-logs", action="store_true", help="Include per-task worker logs"),
        _json_flag(),
    ], help="Export a board to a portable .tar.gz archive", description=(
        "Package a board's tasks, comments, links, history, and file attachments into one archive "
        "that can be imported on another machine. Claims, worker PIDs, chat subscriptions, and "
        "paths belonging to this machine are stripped. Workspaces are never included — they are "
        "rebuilt on demand."
    )),
    _cmd("import", [
        _arg("archive", help="Path to the .tar.gz archive"),
        _arg("--as", dest="as_slug", help="Slug for the imported board (default: from the archive)"),
        _arg("--switch", action="store_true", help="Switch to the imported board afterwards"),
        _json_flag(),
    ], help="Import a board archive as a new board", description=(
        "Import a .tar.gz produced by `hermes kanban boards export`. The board always lands as a "
        "NEW board — the slug gains a numeric suffix if it is already taken — so an import can "
        "never overwrite or merge into a board you already have."
    )),
]

# Top-level ``hermes kanban <action>`` records, in ``--help`` order.
_SPECS = [
    _cmd("init", help="Create kanban.db if missing (idempotent)"),
    _cmd("boards", children=("boards_action", _BOARD_SPECS),
         help="Manage kanban boards (one board per project / workstream)",
         description=(
             "Boards let you separate unrelated streams of work (projects, repos, domains) into "
             "isolated queues. Each board has its own DB, workspaces directory, and dispatcher "
             "loop — tasks on one board cannot collide with tasks on another. The first board is "
             "'default' and always exists."
         )),
    _cmd("create", [
        _arg("title", help="Task title"),
        _arg("--body", help="Optional opening post"),
        _arg("--assignee", help="Profile name to assign"),
        _arg("--parent", action="append", default=[], help="Parent task id (repeatable)"),
        _arg("--workspace", default="scratch",
             help="scratch | worktree | worktree:<path> | dir:<path> (default: scratch)"),
        _arg("--branch", help="Branch name for worktree tasks, e.g. wt/t6-wire"),
        _arg("--project",
             help="Link to a project (id or slug). Anchors the task's "
                  "worktree under the project's primary repo with a "
                  "deterministic branch. See `hermes project list`."),
        _TENANT,
        _PRIORITY,
        _arg("--triage", action="store_true",
             help="Park in triage — a specifier will flesh out the spec and promote to todo"),
        _arg("--idempotency-key",
             help="Dedup key. If a non-archived task with this key exists, "
                  "its id is returned instead of creating a duplicate."),
        _arg("--max-runtime",
             help="Per-task runtime cap. Accepts seconds (300) or durations (90s, "
                  "30m, 2h, 1d). When exceeded, the dispatcher SIGTERMs (then "
                  "SIGKILLs) the worker and re-queues the task."),
        _arg("--created-by", default="user", help="Author name recorded on the task (default: user)"),
        _arg("--skill", action="append", default=[], dest="skills",
             help="Skill to force-load into the worker (repeatable). The kanban "
                  "lifecycle is already injected automatically. Example: --skill "
                  "translation --skill github-code-review"),
        _arg("--max-retries", type=int, metavar="N",
             help="Per-task override for the consecutive-failure "
                  f"circuit breaker. Trip on the Nth failure — e.g. --max-retries 1 blocks on the "
                  f"first failure (no retries), --max-retries 3 allows two retries. Omit to use "
                  f"the dispatcher's kanban.failure_limit config (default "
                  f"{kb.DEFAULT_FAILURE_LIMIT})."),
        _arg("--model", dest="model_override",
             help="Pin the worker to this model (passed as -m <model>) without "
                  "changing the profile's configured model. Combine with --provider "
                  "when the model belongs to a different backend than the profile's default."),
        _arg("--provider", dest="provider_override",
             help="Provider the --model belongs to (passed as --provider <name> to "
                  "the worker). Requires --model."),
        _arg("--goal", action="store_true", dest="goal_mode",
             help="Run the worker in a goal loop: after each turn a judge checks the "
                  "response against the card title/body and, if not done, the worker "
                  "keeps going in the same session until the judge agrees it's "
                  "complete (or the turn budget runs out, which blocks the card for "
                  "review). Best for open-ended cards one shot rarely finishes."),
        _arg("--goal-max-turns", type=int, metavar="N", dest="goal_max_turns",
             help="Turn budget for --goal workers (default 20). Ignored without --goal."),
        _arg("--initial-status", choices=sorted(kb.VALID_INITIAL_STATUSES), default="running",
             help="Initial card status. Use 'blocked' for cards "
                  "that require immediate human ops (R3 gate) "
                  "to skip the brief running-to-blocked transition."),
        _json_flag(help="Emit JSON output"),
    ], help="Create a new task"),
    _cmd("swarm", [
        _arg("goal", help="Swarm goal / final outcome"),
        _arg("--worker", action="append", default=[], metavar="PROFILE:TITLE[:SKILL,SKILL]",
             help="Parallel worker card (repeatable)"),
        _arg("--verifier", required=True, help="Verifier profile"),
        _arg("--synthesizer", required=True, help="Synthesizer/writer profile"),
        _TENANT,
        _PRIORITY,
        _arg("--created-by", help="Creator/anchor profile"),
        _arg("--idempotency-key", help="Dedup key for the root card"),
        _json_flag(help="Emit JSON output"),
    ], help="Create a Kanban Swarm v1 graph (parallel workers → verifier → synthesizer)"),
    _cmd("list", [
        _arg("--mine", action="store_true", help="Filter by $HERMES_PROFILE as assignee"),
        _arg("--assignee"),
        _arg("--status", choices=sorted(kb.VALID_STATUSES)),
        _arg("--tenant"),
        _arg("--session",
             help="Filter by originating chat/agent session id (set on tasks created from inside an ACP loop)"),
        _arg("--archived", action="store_true", help="Include archived tasks"),
        _json_flag(),
        _arg("--sort", choices=sorted(kb.VALID_SORT_ORDERS.keys()),
             help="Sort order for listed tasks (default: priority)"),
        _arg("--workflow-template-id", metavar="ID", help="Restrict to tasks with this workflow_template_id"),
        _arg("--step-key", dest="current_step_key", metavar="KEY",
             help="Restrict to tasks with this current_step_key"),
    ], aliases=["ls"], help="List tasks"),
    _cmd("show", [_TASK_ID, _json_flag(), *_run_state_args("filter listed runs by task_runs column")],
         help="Show a task with comments + events"),
    _cmd("assign", [_TASK_ID, _arg("profile", help="Profile name (or 'none' to unassign)")],
         help="Assign or reassign a task"),
    _cmd("set-model", [
        _TASK_ID,
        _arg("model", nargs="?", help="Model to pin the worker to (or 'none' to clear the override)"),
        _arg("--provider",
             help="Provider the model belongs to (worker is spawned with "
                  "--provider <name>). Cleared together with the model."),
    ], help="Set or clear a task's model/provider override (takes effect on the next dispatch)"),
    _cmd("reclaim", [_TASK_ID, _RECLAIM_REASON], help="Release an active worker claim on a running task"),
    _cmd("reassign", [
        _TASK_ID,
        _arg("profile", help="New profile name (or 'none' to unassign)"),
        _arg("--reclaim", action="store_true",
             help="Release any active claim before reassigning (required if task is running)"),
        _RECLAIM_REASON,
    ], help="Reassign a task to a different profile, optionally reclaiming first"),
    _cmd("diagnostics", [
        _arg("--severity", choices=["warning", "error", "critical"],
             help="Only show diagnostics at or above this severity"),
        _arg("--task", help="Only show diagnostics for one task id"),
        _json_flag(help="Emit JSON (structured) instead of the default human table"),
    ], aliases=["diag"], help="List active diagnostics on the current board"),
    _cmd("link", [_arg("parent_id"), _arg("child_id")], help="Add a parent->child dependency"),
    _cmd("unlink", [_arg("parent_id"), _arg("child_id")], help="Remove a parent->child dependency"),
    _cmd("claim", [
        _TASK_ID,
        _arg("--ttl", type=int, default=kb.DEFAULT_CLAIM_TTL_SECONDS, help="Claim TTL in seconds (default: 900)"),
    ], help="Atomically claim a ready task (prints resolved workspace path)"),
    _cmd("comment", [
        _TASK_ID,
        _arg("text", nargs="+", help="Comment body"),
        _arg("--author", help="Author name (default: $HERMES_PROFILE or 'user')"),
        _arg("--max-len", type=int, help="Trim the stored comment body to this many characters"),
    ], help="Append a comment"),
    _cmd("attach", [
        _TASK_ID,
        _arg("path", help="Path to the local file to attach"),
        _arg("--content-type", help="MIME type (default: guessed from the file extension)"),
        _arg("--name", help="Stored filename (default: the source file's basename)"),
        _arg("--author", help="uploaded_by label (default: $HERMES_PROFILE or 'user')"),
    ], help="Attach a local file to a task"),
    _cmd("attachments", [_TASK_ID, _json_flag()], help="List a task's attachments"),
    _cmd("attach-rm", [_arg("attachment_id", type=int)], help="Delete an attachment by id"),
    _cmd("complete", [
        _arg("task_ids", nargs="+", help="One or more task ids (only --result applies to all of them)"),
        _arg("--result", help="Result summary"),
        _arg("--summary",
             help="Structured handoff summary for downstream tasks. Falls back to --result if omitted."),
        _arg("--metadata",
             help='JSON dict of structured facts (e.g. \'{"changed_files": [...], '
                  '"tests_run": 12}\'). Stored on the closing run.'),
    ], help="Mark one or more tasks done"),
    _cmd("edit", [
        _TASK_ID,
        _arg("--result", required=True, help="Backfilled task result text for a done task"),
        *_STEP_HANDOFF,
    ], help="Edit recovery fields on an already-completed task"),
    _cmd("block", [
        _TASK_ID,
        _arg("reason", nargs="*", help="Reason (also appended as a comment)"),
        _bulk_ids("block"),
        _arg("--kind", choices=sorted(kb.VALID_BLOCK_KINDS),
             help="Typed block reason. 'dependency' waits in todo (auto-promoted when "
                  "parents finish, no human); 'needs_input'/'capability' go to "
                  "blocked for a human; 'transient' marks a maybe-flaky failure. "
                  "Repeated same-kind re-blocks after unblock route the task to "
                  "triage to break unblock loops. Omit for a generic block."),
    ], help="Mark one or more tasks blocked"),
    _cmd("schedule", [
        _TASK_ID,
        _arg("reason", nargs="*", help="Reason/timing note (also appended as a comment)"),
        _bulk_ids("schedule"),
    ], help="Park one or more tasks in Scheduled (waiting on time, not human input)"),
    _cmd("unblock", [
        _reason("Optional reason/note — recorded as a comment before unblocking. Quote multi-word reasons."),
        _TASK_IDS,
    ], help="Return blocked/scheduled tasks to ready, or todo while parents remain open"),
    _cmd("request-review", [
        _TASK_ID,
        _arg("--summary", help="What was implemented and how it was verified — shown to the reviewer."),
        _arg("--reviewer", help="Optional reviewer profile; reassigns the task before review dispatch."),
        _arg("--metadata", help="JSON object with structured reviewer handoff facts."),
        _arg("--force", action="store_true",
             help="Override the live-claim guard: move a running, claimed "
                  "task to review even without owning its run (clears the worker's claim)."),
    ], help="Move a task to 'review' (implementation done, awaiting review) — NOT a block"),
    _cmd("request-changes", [_TASK_ID, _arg("reason", nargs="+", help="Concrete changes required before re-review")],
         help="Reviewer verdict: return the active review run to its implementer"),
    _cmd("reopen-review", [
        _TASK_IDS,
        _reason("Optional reason/note — recorded as a comment before reopening. Quote multi-word reasons."),
    ], help="Send one or more review tasks back for changes (review -> ready/todo)"),
    _cmd("promote", [
        _TASK_ID,
        _arg("reason", nargs="*", help="Audit-trail reason (recorded on the task_events row)"),
        _bulk_ids("promote"),
        _arg("--force", action="store_true", help="Promote even if parent dependencies are not yet done/archived"),
        _arg("--dry-run", action="store_true", help="Validate the promotion without mutating state"),
        _arg("--json", dest="json", action="store_true", help="Emit machine-readable JSON result"),
    ], help="Manually move one or more todo/blocked tasks to ready (recovery path)"),
    _cmd("archive", [
        _arg("task_ids", nargs="*", help="Task ids to archive (default mode)"),
        _arg("--rm", dest="purge_ids", nargs="+",
             help="Permanently delete already-archived task ids from the board"),
    ], help="Archive one or more tasks"),
    _cmd("tail", [_TASK_ID, _arg("--interval", type=float, default=1.0)], help="Follow a task's event stream"),
    _cmd("dispatch", [
        _arg("--dry-run", action="store_true", help="Don't actually spawn processes; just print what would happen"),
        _arg("--max", type=int, help="Cap number of spawns this pass"),
        _arg("--failure-limit", type=int, default=kbd.DEFAULT_FAILURE_LIMIT,
             help=f"Auto-block a task after this many consecutive non-success attempts "
                  f"(spawn_failed, timed_out, or crashed; default: {kbd.DEFAULT_FAILURE_LIMIT})"),
        _json_flag(),
    ], help="One dispatcher pass: reclaim stale, promote ready, spawn workers"),
    _cmd("daemon", [
        _arg("--interval", type=float, default=60.0, help="Seconds between dispatch ticks (default: 60)"),
        _arg("--max", type=int, help="Cap number of spawns per tick"),
        _arg("--failure-limit", type=int, default=kbd.DEFAULT_FAILURE_LIMIT),
        _arg("--pidfile", help="Write the daemon's PID to this file on start"),
        _arg("--verbose", "-v", action="store_true", help="Log each tick's outcome to stdout"),
        # Escape hatch for hosts that truly cannot run the gateway; hidden from
        # --help so nobody casually keeps the double-dispatcher pattern alive.
        _arg("--force", action="store_true", help=argparse.SUPPRESS),
    ], help="DEPRECATED — dispatcher now runs in the gateway. Use `hermes gateway start`."),
    _cmd("watch", [
        _arg("--assignee", help="Only show events for tasks assigned to this profile"),
        _arg("--tenant", help="Only show events from tasks in this tenant"),
        _arg("--kinds",
             help="Comma-separated event kinds to include (e.g. 'completed,blocked,gave_up,crashed,timed_out')"),
        _arg("--interval", type=float, default=0.5, help="Poll interval in seconds (default: 0.5)"),
    ], help="Live-stream task_events to the terminal (Ctrl+C to exit)"),
    _cmd("stats", [_json_flag()], help="Per-status + per-assignee counts + oldest-ready age"),
    _cmd("notify-subscribe", [
        _TASK_ID,
        *_NOTIFY_TARGET,
        _arg("--user-id"),
        _arg("--user-id-alt"),
        _arg("--chat-type", choices=("dm", "group", "channel", "thread"),
             help="Originating source chat_type, recorded so the active-wake delivery "
                  "modes resolve the operator's real session. Omit to leave an "
                  "existing sub unchanged (new subs default to 'dm')."),
        _arg("--notifier-profile",
             help="Profile gateway that owns/delivers this subscription (default: active profile)"),
        # choices: single source of truth shared with the DB/watcher enum.
        _arg("--delivery-mode", choices=kbn._NOTIFY_DELIVERY_MODES,
             help="How the kanban-notifier reacts to terminal events for this "
                  "subscription: 'notify' (passive message only; default), "
                  "'notify+wake' (message AND wake the destination gateway agent so "
                  "it reads the full board context and replies in its own voice), or "
                  "'wake' (wake the agent only, no passive message). Omit to leave an "
                  "existing subscription's mode unchanged (new subs default to 'notify')."),
    ], help="Subscribe a gateway source to a task's terminal events (used by /kanban subscribe in the gateway adapter)"),
    _cmd("notify-list", [_arg("task_id", nargs="?"), _json_flag()],
         help="List notification subscriptions (optionally for a single task)"),
    _cmd("notify-unsubscribe", [_TASK_ID, *_NOTIFY_TARGET], help="Remove a gateway subscription from a task"),
    _cmd("log", [_TASK_ID, _arg("--tail", type=int, help="Only print the last N bytes")],
         help="Print the worker log for a task (from <kanban-root>/kanban/logs/)"),
    _cmd("runs", [_TASK_ID, _json_flag(), *_run_state_args("filter runs by task_runs column")],
         help="Show attempt history for a task (one row per run: profile, outcome, elapsed, summary)"),
    _cmd("heartbeat", [
        _TASK_ID,
        _arg("--note", help="Optional short note attached to the heartbeat event"),
    ], help="Emit a heartbeat event for a running task (worker liveness signal)"),
    _cmd("assignees", [_json_flag()],
         help="List known profiles + per-profile task counts (union of ~/.hermes/profiles/ and current assignees on the board)"),
    _cmd("context", [_TASK_ID],
         help="Print the full context a worker sees for a task (title + body + parent results + comments)."),
    _cmd("specify", _triage_sweep_args("specify", "Specify", "specifier"),
         help="Flesh out a triage-column task into a concrete spec (title + "
              "body) and promote it to todo. Uses the auxiliary LLM "
              "configured under auxiliary.triage_specifier."),
    _cmd("decompose", _triage_sweep_args("decompose", "Decompose", "decomposer"),
         help="Decompose a triage-column task into a graph of child tasks "
              "routed to specialist profiles by description. Falls back "
              "to specify-style single-task promotion when the task "
              "doesn't benefit from fan-out. Uses auxiliary.kanban_decomposer."),
    _cmd("gc", [
        _arg("--event-retention-days", type=int, default=30,
             help="Delete task_events older than N days for terminal tasks (default: 30)"),
        _arg("--log-retention-days", type=int, default=30, help="Delete worker log files older than N days (default: 30)"),
    ], help="Garbage-collect archived-task workspaces, old events, and old logs"),
    _cmd("repair", [_json_flag(help="Emit the repair report as JSON")],
         help="Check kanban.db integrity and auto-repair index-only corruption",
         description=(
             "Runs PRAGMA integrity_check on the board's DB and reports the result. When the "
             "failure consists only of index-scoped errors ('wrong # of entries in index <name>' / "
             "'row N missing from index <name>'), the corrupt file is quarantined to a "
             ".corrupt.<hash>.bak sibling first and the damaged indexes are rebuilt with REINDEX — "
             "the same narrow auto-repair the connect-time guard applies. Any other corruption "
             "class is reported and left untouched (fail-closed). Exits 0 when the DB is healthy "
             "or was repaired, non-zero when it is still corrupt."
         )),
]


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Attach the ``kanban`` subcommand tree; returns the ``kanban`` parser."""
    kanban_parser = parent_subparsers.add_parser(
        "kanban",
        help="Multi-profile collaboration board (tasks, links, comments)",
        description="Durable SQLite-backed task board shared across Hermes profiles. "
                    "Tasks are claimed atomically, can depend on other tasks, and "
                    "are executed by a named profile in an isolated workspace. "
                    "See https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban "
                    "or docs/hermes-kanban-v1-spec.pdf for the full design.",
    )
    # --board scopes every subcommand to one board's DB; when omitted the
    # resolution is HERMES_KANBAN_BOARD, then the persisted current-board
    # file, then "default" (kanban_db.get_current_board()).
    kanban_parser.add_argument("--board", default=None, metavar="<slug>",
                               help="Board slug to operate on. Defaults to the current board (set "
                                    "via `hermes kanban boards switch <slug>` or the "
                                    "HERMES_KANBAN_BOARD env var). Use `hermes kanban boards "
                                    "list` to see all boards.")
    _add_commands(kanban_parser.add_subparsers(dest="kanban_action"), _SPECS)
    kanban_parser.set_defaults(_kanban_parser=kanban_parser)
    return kanban_parser
