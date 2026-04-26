"""CLI for the Hermes Kanban board — ``hermes kanban …`` subcommand.

Exposes the full 15-verb surface documented in the design spec
(``docs/hermes-kanban-v1-spec.pdf``).  All DB work is delegated to
``kanban_db``.  This module adds:

  * Argparse subcommand construction (``build_parser``).
  * Argument dispatch (``kanban_command``).
  * Output formatting (plain text + ``--json``).
  * A short shared helper that parses a single slash-style string
    (used by ``/kanban …`` in CLI and gateway) and forwards it to the
    argparse surface.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "todo":     "◻",
    "ready":    "▶",
    "running":  "●",
    "blocked":  "⊘",
    "done":     "✓",
    "archived": "—",
}


def _fmt_ts(ts: Optional[int]) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _fmt_task_line(t: kb.Task) -> str:
    icon = _STATUS_ICONS.get(t.status, "?")
    assignee = t.assignee or "(unassigned)"
    tenant = f" [{t.tenant}]" if t.tenant else ""
    return f"{icon} {t.id}  {t.status:8s}  {assignee:20s}{tenant}  {t.title}"


def _task_to_dict(t: kb.Task) -> dict[str, Any]:
    return {
        "id": t.id,
        "title": t.title,
        "body": t.body,
        "assignee": t.assignee,
        "status": t.status,
        "priority": t.priority,
        "tenant": t.tenant,
        "workspace_kind": t.workspace_kind,
        "workspace_path": t.workspace_path,
        "created_by": t.created_by,
        "created_at": t.created_at,
        "started_at": t.started_at,
        "completed_at": t.completed_at,
        "result": t.result,
    }


def _parse_workspace_flag(value: str) -> tuple[str, Optional[str]]:
    """Parse ``--workspace`` into ``(kind, path|None)``.

    Accepts: ``scratch``, ``worktree``, ``dir:<path>``.
    """
    if not value:
        return ("scratch", None)
    v = value.strip()
    if v in ("scratch", "worktree"):
        return (v, None)
    if v.startswith("dir:"):
        path = v[len("dir:"):].strip()
        if not path:
            raise argparse.ArgumentTypeError(
                "--workspace dir: requires a path after the colon"
            )
        return ("dir", os.path.expanduser(path))
    raise argparse.ArgumentTypeError(
        f"unknown --workspace value {value!r}: use scratch, worktree, or dir:<path>"
    )


# ---------------------------------------------------------------------------
# Argparse builder
# ---------------------------------------------------------------------------

def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Attach the ``kanban`` subcommand tree under an existing subparsers.

    Returns the top-level ``kanban`` parser so caller can ``set_defaults``.
    """
    kanban_parser = parent_subparsers.add_parser(
        "kanban",
        help="Multi-profile collaboration board (tasks, links, comments)",
        description=(
            "Durable SQLite-backed task board shared across Hermes profiles. "
            "Tasks are claimed atomically, can depend on other tasks, and "
            "are executed by a named profile in an isolated workspace. "
            "See https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban "
            "or docs/hermes-kanban-v1-spec.pdf for the full design."
        ),
    )
    sub = kanban_parser.add_subparsers(dest="kanban_action")

    # --- init ---
    sub.add_parser("init", help="Create kanban.db if missing (idempotent)")

    # --- create ---
    p_create = sub.add_parser("create", help="Create a new task")
    p_create.add_argument("title", help="Task title")
    p_create.add_argument("--body", default=None, help="Optional opening post")
    p_create.add_argument("--assignee", default=None, help="Profile name to assign")
    p_create.add_argument("--parent", action="append", default=[],
                          help="Parent task id (repeatable)")
    p_create.add_argument("--workspace", default="scratch",
                          help="scratch | worktree | dir:<path> (default: scratch)")
    p_create.add_argument("--tenant", default=None, help="Tenant namespace")
    p_create.add_argument("--priority", type=int, default=0, help="Priority tiebreaker")
    p_create.add_argument("--created-by", default="user",
                          help="Author name recorded on the task (default: user)")
    p_create.add_argument("--json", action="store_true", help="Emit JSON output")

    # --- list ---
    p_list = sub.add_parser("list", aliases=["ls"], help="List tasks")
    p_list.add_argument("--mine", action="store_true",
                        help="Filter by $HERMES_PROFILE as assignee")
    p_list.add_argument("--assignee", default=None)
    p_list.add_argument("--status", default=None,
                        choices=sorted(kb.VALID_STATUSES))
    p_list.add_argument("--tenant", default=None)
    p_list.add_argument("--archived", action="store_true",
                        help="Include archived tasks")
    p_list.add_argument("--json", action="store_true")

    # --- show ---
    p_show = sub.add_parser("show", help="Show a task with comments + events")
    p_show.add_argument("task_id")
    p_show.add_argument("--json", action="store_true")

    # --- assign ---
    p_assign = sub.add_parser("assign", help="Assign or reassign a task")
    p_assign.add_argument("task_id")
    p_assign.add_argument("profile", help="Profile name (or 'none' to unassign)")

    # --- link / unlink ---
    p_link = sub.add_parser("link", help="Add a parent->child dependency")
    p_link.add_argument("parent_id")
    p_link.add_argument("child_id")
    p_unlink = sub.add_parser("unlink", help="Remove a parent->child dependency")
    p_unlink.add_argument("parent_id")
    p_unlink.add_argument("child_id")

    # --- claim ---
    p_claim = sub.add_parser(
        "claim",
        help="Atomically claim a ready task (prints resolved workspace path)",
    )
    p_claim.add_argument("task_id")
    p_claim.add_argument("--ttl", type=int, default=kb.DEFAULT_CLAIM_TTL_SECONDS,
                         help="Claim TTL in seconds (default: 900)")

    # --- comment / complete / block / unblock / archive ---
    p_comment = sub.add_parser("comment", help="Append a comment")
    p_comment.add_argument("task_id")
    p_comment.add_argument("text", nargs="+", help="Comment body")
    p_comment.add_argument("--author", default=None,
                           help="Author name (default: $HERMES_PROFILE or 'user')")

    p_complete = sub.add_parser("complete", help="Mark a task done")
    p_complete.add_argument("task_id")
    p_complete.add_argument("--result", default=None, help="Result summary")

    p_block = sub.add_parser("block", help="Mark a task blocked (needs input)")
    p_block.add_argument("task_id")
    p_block.add_argument("reason", nargs="*", help="Reason (also appended as a comment)")

    p_unblock = sub.add_parser("unblock", help="Return a blocked task to ready")
    p_unblock.add_argument("task_id")

    p_archive = sub.add_parser("archive", help="Archive a task (hide from default list)")
    p_archive.add_argument("task_id")

    # --- tail ---
    p_tail = sub.add_parser("tail", help="Follow a task's event stream")
    p_tail.add_argument("task_id")
    p_tail.add_argument("--interval", type=float, default=1.0)

    # --- dispatch ---
    p_disp = sub.add_parser(
        "dispatch",
        help="One dispatcher pass: reclaim stale, promote ready, spawn workers",
    )
    p_disp.add_argument("--dry-run", action="store_true",
                        help="Don't actually spawn processes; just print what would happen")
    p_disp.add_argument("--max", type=int, default=None,
                        help="Cap number of spawns this pass")
    p_disp.add_argument("--json", action="store_true")

    # --- context --- (for spawned workers)
    p_ctx = sub.add_parser(
        "context",
        help="Print the full context a worker sees for a task "
             "(title + body + parent results + comments).",
    )
    p_ctx.add_argument("task_id")

    # --- gc ---
    sub.add_parser(
        "gc", help="Garbage-collect workspaces of archived tasks"
    )

    kanban_parser.set_defaults(_kanban_parser=kanban_parser)
    return kanban_parser


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

def kanban_command(args: argparse.Namespace) -> int:
    """Entry point from ``hermes kanban …`` argparse dispatch.

    Returns a shell-style exit code (0 on success, non-zero on error).
    """
    action = getattr(args, "kanban_action", None)
    if not action:
        # No subaction given: print help via the stored parser reference.
        parser = getattr(args, "_kanban_parser", None)
        if parser is not None:
            parser.print_help()
        else:
            print(
                "usage: hermes kanban <action> [options]\n"
                "Run 'hermes kanban --help' for the full list of actions.",
                file=sys.stderr,
            )
        return 0

    handlers = {
        "init":     _cmd_init,
        "create":   _cmd_create,
        "list":     _cmd_list,
        "ls":       _cmd_list,
        "show":     _cmd_show,
        "assign":   _cmd_assign,
        "link":     _cmd_link,
        "unlink":   _cmd_unlink,
        "claim":    _cmd_claim,
        "comment":  _cmd_comment,
        "complete": _cmd_complete,
        "block":    _cmd_block,
        "unblock":  _cmd_unblock,
        "archive":  _cmd_archive,
        "tail":     _cmd_tail,
        "dispatch": _cmd_dispatch,
        "context":  _cmd_context,
        "gc":       _cmd_gc,
    }
    handler = handlers.get(action)
    if not handler:
        print(f"kanban: unknown action {action!r}", file=sys.stderr)
        return 2
    try:
        return int(handler(args) or 0)
    except (ValueError, RuntimeError) as exc:
        print(f"kanban: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _profile_author() -> str:
    """Best-effort author name for an interactive CLI call."""
    for env in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
        v = os.environ.get(env)
        if v:
            return v
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "user"
    except Exception:
        return "user"


def _cmd_init(args: argparse.Namespace) -> int:
    path = kb.init_db()
    print(f"Kanban DB initialized at {path}")
    return 0


def _cmd_create(args: argparse.Namespace) -> int:
    ws_kind, ws_path = _parse_workspace_flag(args.workspace)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title=args.title,
            body=args.body,
            assignee=args.assignee,
            created_by=args.created_by or _profile_author(),
            workspace_kind=ws_kind,
            workspace_path=ws_path,
            tenant=args.tenant,
            priority=args.priority,
            parents=tuple(args.parent or ()),
        )
        task = kb.get_task(conn, task_id)
    if getattr(args, "json", False):
        print(json.dumps(_task_to_dict(task), indent=2, ensure_ascii=False))
    else:
        print(f"Created {task_id}  ({task.status}, assignee={task.assignee or '-'})")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    assignee = args.assignee
    if args.mine and not assignee:
        assignee = _profile_author()
    with kb.connect() as conn:
        # Cheap "mini-dispatch": recompute ready so list output reflects
        # dependencies that may have cleared since the last dispatcher tick.
        kb.recompute_ready(conn)
        tasks = kb.list_tasks(
            conn,
            assignee=assignee,
            status=args.status,
            tenant=args.tenant,
            include_archived=args.archived,
        )
    if getattr(args, "json", False):
        print(json.dumps([_task_to_dict(t) for t in tasks], indent=2, ensure_ascii=False))
        return 0
    if not tasks:
        print("(no matching tasks)")
        return 0
    for t in tasks:
        print(_fmt_task_line(t))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with kb.connect() as conn:
        task = kb.get_task(conn, args.task_id)
        if not task:
            print(f"no such task: {args.task_id}", file=sys.stderr)
            return 1
        comments = kb.list_comments(conn, args.task_id)
        events = kb.list_events(conn, args.task_id)
        parents = kb.parent_ids(conn, args.task_id)
        children = kb.child_ids(conn, args.task_id)

    if getattr(args, "json", False):
        payload = {
            "task": _task_to_dict(task),
            "parents": parents,
            "children": children,
            "comments": [
                {"author": c.author, "body": c.body, "created_at": c.created_at}
                for c in comments
            ],
            "events": [
                {"kind": e.kind, "payload": e.payload, "created_at": e.created_at}
                for e in events
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"Task {task.id}: {task.title}")
    print(f"  status:    {task.status}")
    print(f"  assignee:  {task.assignee or '-'}")
    if task.tenant:
        print(f"  tenant:    {task.tenant}")
    print(f"  workspace: {task.workspace_kind}" +
          (f" @ {task.workspace_path}" if task.workspace_path else ""))
    print(f"  created:   {_fmt_ts(task.created_at)} by {task.created_by or '-'}")
    if task.started_at:
        print(f"  started:   {_fmt_ts(task.started_at)}")
    if task.completed_at:
        print(f"  completed: {_fmt_ts(task.completed_at)}")
    if parents:
        print(f"  parents:   {', '.join(parents)}")
    if children:
        print(f"  children:  {', '.join(children)}")
    if task.body:
        print()
        print("Body:")
        print(task.body)
    if task.result:
        print()
        print("Result:")
        print(task.result)
    if comments:
        print()
        print(f"Comments ({len(comments)}):")
        for c in comments:
            print(f"  [{_fmt_ts(c.created_at)}] {c.author}: {c.body}")
    if events:
        print()
        print(f"Events ({len(events)}):")
        for e in events[-20:]:
            pl = f" {e.payload}" if e.payload else ""
            print(f"  [{_fmt_ts(e.created_at)}] {e.kind}{pl}")
    return 0


def _cmd_assign(args: argparse.Namespace) -> int:
    profile = None if args.profile.lower() in ("none", "-", "null") else args.profile
    with kb.connect() as conn:
        ok = kb.assign_task(conn, args.task_id, profile)
    if not ok:
        print(f"no such task: {args.task_id}", file=sys.stderr)
        return 1
    print(f"Assigned {args.task_id} to {profile or '(unassigned)'}")
    return 0


def _cmd_link(args: argparse.Namespace) -> int:
    with kb.connect() as conn:
        kb.link_tasks(conn, args.parent_id, args.child_id)
    print(f"Linked {args.parent_id} -> {args.child_id}")
    return 0


def _cmd_unlink(args: argparse.Namespace) -> int:
    with kb.connect() as conn:
        ok = kb.unlink_tasks(conn, args.parent_id, args.child_id)
    if not ok:
        print(f"No such link: {args.parent_id} -> {args.child_id}", file=sys.stderr)
        return 1
    print(f"Unlinked {args.parent_id} -> {args.child_id}")
    return 0


def _cmd_claim(args: argparse.Namespace) -> int:
    with kb.connect() as conn:
        task = kb.claim_task(conn, args.task_id, ttl_seconds=args.ttl)
        if task is None:
            # Report why
            existing = kb.get_task(conn, args.task_id)
            if existing is None:
                print(f"no such task: {args.task_id}", file=sys.stderr)
                return 1
            print(
                f"cannot claim {args.task_id}: status={existing.status} "
                f"lock={existing.claim_lock or '(none)'}",
                file=sys.stderr,
            )
            return 1
        workspace = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, task.id, str(workspace))
    print(f"Claimed {task.id}")
    print(f"Workspace: {workspace}")
    return 0


def _cmd_comment(args: argparse.Namespace) -> int:
    body = " ".join(args.text).strip()
    author = args.author or _profile_author()
    with kb.connect() as conn:
        kb.add_comment(conn, args.task_id, author, body)
    print(f"Comment added to {args.task_id}")
    return 0


def _cmd_complete(args: argparse.Namespace) -> int:
    with kb.connect() as conn:
        ok = kb.complete_task(conn, args.task_id, result=args.result)
    if not ok:
        print(f"cannot complete {args.task_id} (unknown id or terminal state)", file=sys.stderr)
        return 1
    print(f"Completed {args.task_id}")
    return 0


def _cmd_block(args: argparse.Namespace) -> int:
    reason = " ".join(args.reason).strip() if args.reason else None
    author = _profile_author()
    with kb.connect() as conn:
        if reason:
            kb.add_comment(conn, args.task_id, author, f"BLOCKED: {reason}")
        ok = kb.block_task(conn, args.task_id, reason=reason)
    if not ok:
        print(f"cannot block {args.task_id}", file=sys.stderr)
        return 1
    print(f"Blocked {args.task_id}" + (f": {reason}" if reason else ""))
    return 0


def _cmd_unblock(args: argparse.Namespace) -> int:
    with kb.connect() as conn:
        ok = kb.unblock_task(conn, args.task_id)
    if not ok:
        print(f"cannot unblock {args.task_id} (not blocked?)", file=sys.stderr)
        return 1
    print(f"Unblocked {args.task_id}")
    return 0


def _cmd_archive(args: argparse.Namespace) -> int:
    with kb.connect() as conn:
        ok = kb.archive_task(conn, args.task_id)
    if not ok:
        print(f"cannot archive {args.task_id}", file=sys.stderr)
        return 1
    print(f"Archived {args.task_id}")
    return 0


def _cmd_tail(args: argparse.Namespace) -> int:
    last_id = 0
    print(f"Tailing events for {args.task_id}. Ctrl-C to stop.")
    try:
        while True:
            with kb.connect() as conn:
                events = kb.list_events(conn, args.task_id)
            for e in events:
                if e.id > last_id:
                    pl = f" {e.payload}" if e.payload else ""
                    print(f"[{_fmt_ts(e.created_at)}] {e.kind}{pl}", flush=True)
                    last_id = e.id
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        print("\n(stopped)")
        return 0


def _cmd_dispatch(args: argparse.Namespace) -> int:
    with kb.connect() as conn:
        res = kb.dispatch_once(
            conn,
            dry_run=args.dry_run,
            max_spawn=args.max,
        )
    if getattr(args, "json", False):
        print(json.dumps({
            "reclaimed": res.reclaimed,
            "promoted": res.promoted,
            "spawned": [
                {"task_id": tid, "assignee": who, "workspace": ws}
                for (tid, who, ws) in res.spawned
            ],
            "skipped_unassigned": res.skipped_unassigned,
        }, indent=2))
        return 0
    print(f"Reclaimed:  {res.reclaimed}")
    print(f"Promoted:   {res.promoted}")
    print(f"Spawned:    {len(res.spawned)}")
    for tid, who, ws in res.spawned:
        tag = " (dry)" if args.dry_run else ""
        print(f"  - {tid}  ->  {who}  @ {ws or '-'}{tag}")
    if res.skipped_unassigned:
        print(f"Skipped (unassigned): {', '.join(res.skipped_unassigned)}")
    return 0


def _cmd_context(args: argparse.Namespace) -> int:
    with kb.connect() as conn:
        text = kb.build_worker_context(conn, args.task_id)
    print(text)
    return 0


def _cmd_gc(args: argparse.Namespace) -> int:
    """Remove scratch workspaces of archived tasks.

    Only touches directories under the default scratch root; leaves user
    ``dir:`` workspaces and ``worktree`` dirs alone (user owns those).
    """
    import shutil
    scratch_root = kb.workspaces_root()
    removed = 0
    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT id, workspace_kind, workspace_path FROM tasks WHERE status = 'archived'"
        ).fetchall()
    for row in rows:
        if row["workspace_kind"] != "scratch":
            continue
        path = Path(row["workspace_path"] or (scratch_root / row["id"]))
        try:
            path = path.resolve()
        except OSError:
            continue
        try:
            scratch_root.resolve().relative_to(scratch_root.resolve())
            path.relative_to(scratch_root.resolve())
        except ValueError:
            # Safety: never delete outside the scratch root.
            continue
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    print(f"GC complete: removed {removed} scratch workspace(s)")
    return 0


# ---------------------------------------------------------------------------
# Slash-command entry point (used by /kanban from CLI and gateway)
# ---------------------------------------------------------------------------

def run_slash(rest: str) -> str:
    """Execute a ``/kanban …`` string and return captured stdout/stderr.

    ``rest`` is everything after ``/kanban`` (may be empty).  Used from
    both the interactive CLI (``self._handle_kanban_command``) and the
    gateway (``_handle_kanban_command``) so formatting is identical.
    """
    import io
    import contextlib

    tokens = shlex.split(rest) if rest and rest.strip() else []

    parser = argparse.ArgumentParser(prog="/kanban", add_help=False)
    parser.exit_on_error = False  # type: ignore[attr-defined]
    sub = parser.add_subparsers(dest="kanban_action")
    # Reuse the argparse builder -- call it with a throwaway parent
    # subparsers via a wrapping top-level parser.
    wrap = argparse.ArgumentParser(prog="/", add_help=False)
    wrap.exit_on_error = False  # type: ignore[attr-defined]
    wrap_sub = wrap.add_subparsers(dest="_top")
    build_parser(wrap_sub)

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        # Prepend the "kanban" token so our top-level subparser routes here.
        argv = ["kanban", *tokens] if tokens else ["kanban"]
        args = wrap.parse_args(argv)
    except SystemExit as exc:
        return f"(usage error: {exc})"
    except argparse.ArgumentError as exc:
        return f"(usage error: {exc})"

    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        try:
            kanban_command(args)
        except SystemExit:
            pass
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)

    out = buf_out.getvalue().rstrip()
    err = buf_err.getvalue().rstrip()
    if err and out:
        return f"{out}\n{err}"
    return err if err else (out or "(no output)")
