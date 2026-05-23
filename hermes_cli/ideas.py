"""CLI for markdown ideas — ``hermes ideas …``."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from typing import Any, Optional

from hermes_cli import ideas_db as db

_SLASH_IDEAS_HELP = """\
/ideas — markdown idea drafts (not Kanban tasks)

  /ideas list [--all-boards] [--board <slug>] [--status draft|active|…]
  /ideas boards
  /ideas show <i_…>
  /ideas create "title" [--body …] [--board default]
  /ideas update <i_…> [--title …] [--body …] [--status …]
  /ideas delete <i_…>
  /ideas convert <i_…> [--assignee <profile>] [--no-triage]

Agents: use ideas_list(all_boards=true) — not read_file on ~/.hermes/ideas.
"""


def _fmt_ts(ts: Optional[int]) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _fmt_line(idea: dict[str, Any]) -> str:
    tags = " ".join(f"#{t}" for t in (idea.get("tags") or [])[:4])
    tag_s = f"  {tags}" if tags else ""
    return f"{idea['id']}  {idea['status']:9s}  {idea['title']}{tag_s}"


def _board_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--board",
        default=None,
        metavar="<slug>",
        help="Board slug (default: 'default').",
    )
    return p


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    board_args = _board_parent()
    ideas_parser = parent_subparsers.add_parser(
        "ideas",
        help="Markdown idea drafts scoped by Kanban board",
        description=(
            "Lightweight markdown ideas stored under ~/.hermes/ideas. "
            "Each board slug is a separate project list. Use "
            "`hermes ideas convert` to promote an idea into a Kanban task."
        ),
        parents=[board_args],
    )
    sub = ideas_parser.add_subparsers(dest="ideas_action")

    p_list = sub.add_parser(
        "list", aliases=["ls"], help="List ideas", parents=[board_args],
    )
    p_list.add_argument("--status", default=None)
    p_list.add_argument("--q", default=None)
    p_list.add_argument("--tag", default=None)
    p_list.add_argument("--all", action="store_true", help="Include archived")
    p_list.add_argument(
        "--all-boards", action="store_true",
        help="List ideas from every board (not just --board)",
    )
    p_list.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="Show one idea", parents=[board_args])
    p_show.add_argument("idea_id")
    p_show.add_argument("--json", action="store_true")

    p_create = sub.add_parser("create", help="Create an idea", parents=[board_args])
    p_create.add_argument("title")
    p_create.add_argument("--body", default="")
    p_create.add_argument("--summary", default=None)
    p_create.add_argument("--status", default=db.DEFAULT_STATUS)
    p_create.add_argument("--tags", default="", help="Comma-separated tags")
    p_create.add_argument("--json", action="store_true")

    p_update = sub.add_parser("update", help="Update an idea", parents=[board_args])
    p_update.add_argument("idea_id")
    p_update.add_argument("--title", default=None)
    p_update.add_argument("--body", default=None)
    p_update.add_argument("--summary", default=None)
    p_update.add_argument("--status", default=None)
    p_update.add_argument("--tags", default=None, help="Comma-separated tags")
    p_update.add_argument("--json", action="store_true")

    p_delete = sub.add_parser(
        "delete", aliases=["rm"], help="Delete an idea", parents=[board_args],
    )
    p_delete.add_argument("idea_id")
    p_delete.add_argument("--keep-file", action="store_true")
    p_delete.add_argument("--json", action="store_true")

    p_dup = sub.add_parser(
        "duplicate", aliases=["dup"], help="Duplicate an idea", parents=[board_args],
    )
    p_dup.add_argument("idea_id")
    p_dup.add_argument("--json", action="store_true")

    p_convert = sub.add_parser(
        "convert", help="Convert idea to Kanban task", parents=[board_args],
    )
    p_convert.add_argument("idea_id")
    p_convert.add_argument("--assignee", default=None)
    p_convert.add_argument("--priority", type=int, default=0)
    p_convert.add_argument(
        "--no-triage", action="store_true", help="Create task in todo, not triage",
    )
    p_convert.add_argument("--tenant", default=None)
    p_convert.add_argument("--json", action="store_true")

    p_boards = sub.add_parser("boards", help="List boards with idea counts")
    p_boards.add_argument("--json", action="store_true")
    b_sub = p_boards.add_subparsers(dest="boards_action")
    b_list = b_sub.add_parser("list", aliases=["ls"], help="List boards", parents=[board_args])
    b_list.add_argument("--json", action="store_true")

    return ideas_parser


def _wants_json(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "json", False))


def ideas_command(args: argparse.Namespace) -> int:
    board = getattr(args, "board", None)
    action = getattr(args, "ideas_action", None)
    if action is None:
        print("usage: hermes ideas <list|show|create|update|delete|convert|boards>", file=sys.stderr)
        return 1

    try:
        if action in {"list", "ls"}:
            if getattr(args, "all_boards", False):
                result = db.list_ideas_all_boards(
                    status=args.status,
                    q=args.q,
                    tag=args.tag,
                    include_archived=args.all,
                )
                if _wants_json(args):
                    print(json.dumps(result, indent=2))
                else:
                    print(f"all boards  ({result['count']} ideas)")
                    for idea in result["ideas"]:
                        print(f"[{idea['board']}] {_fmt_line(idea)}")
                return 0
            result = db.list_ideas(
                board=board,
                status=args.status,
                q=args.q,
                tag=args.tag,
                include_archived=args.all,
            )
            if _wants_json(args):
                print(json.dumps(result, indent=2))
            else:
                print(f"board: {result['board']}  ({len(result['ideas'])} ideas)")
                for idea in result["ideas"]:
                    print(_fmt_line(idea))
            return 0

        if action == "show":
            idea = db.get_idea(args.idea_id)
            if _wants_json(args):
                print(json.dumps({"idea": idea}, indent=2))
            else:
                print(f"{idea['id']}  {idea['status']}  board={idea['board']}")
                print(f"title: {idea['title']}")
                if idea.get("summary"):
                    print(f"summary: {idea['summary']}")
                if idea.get("task_id"):
                    print(f"task: {idea['task_id']}")
                print(f"file: {idea['file_path']}")
                print(f"updated: {_fmt_ts(idea.get('updated_at'))}")
                print()
                print(idea.get("body") or "")
            return 0

        if action == "create":
            tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
            idea = db.create_idea(
                title=args.title,
                body=args.body,
                summary=args.summary,
                status=args.status,
                tags=tags or None,
                board=board,
            )
            if _wants_json(args):
                print(json.dumps({"idea": idea}, indent=2))
            else:
                print(f"created {idea['id']} on board {idea['board']}: {idea['title']}")
            return 0

        if action == "update":
            tags = None
            if args.tags is not None:
                tags = [t.strip() for t in args.tags.split(",") if t.strip()]
            idea = db.update_idea(
                args.idea_id,
                title=args.title,
                body=args.body,
                summary=args.summary,
                status=args.status,
                tags=tags,
            )
            if _wants_json(args):
                print(json.dumps({"idea": idea}, indent=2))
            else:
                print(f"updated {idea['id']}: {idea['title']}")
            return 0

        if action in {"delete", "rm"}:
            db.delete_idea(args.idea_id, delete_file=not args.keep_file)
            if _wants_json(args):
                print(json.dumps({"ok": True, "deleted": args.idea_id}))
            else:
                print(f"deleted {args.idea_id}")
            return 0

        if action in {"duplicate", "dup"}:
            idea = db.duplicate_idea(args.idea_id)
            if _wants_json(args):
                print(json.dumps({"idea": idea}, indent=2))
            else:
                print(f"duplicated -> {idea['id']}")
            return 0

        if action == "convert":
            result = db.convert_to_task(
                args.idea_id,
                assignee=args.assignee,
                priority=args.priority,
                triage=not args.no_triage,
                tenant=args.tenant,
            )
            if _wants_json(args):
                print(json.dumps(result, indent=2))
            else:
                print(
                    f"converted {args.idea_id} -> task {result['task_id']} "
                    f"on board {result['board']}"
                )
            return 0

        if action == "boards":
            boards_action = getattr(args, "boards_action", None)
            if boards_action in {None, "list", "ls"}:
                result = db.list_boards()
                if _wants_json(args):
                    print(json.dumps(result, indent=2))
                else:
                    for b in result["boards"]:
                        cur = " *" if b.get("is_current") else ""
                        print(
                            f"{b.get('slug'):20s}  {int(b.get('idea_count', 0)):3d} ideas{cur}"
                        )
                return 0

    except db.IdeaNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except db.IdeasError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Unknown ideas subcommand: {action}", file=sys.stderr)
    return 1


def run_slash(rest: str) -> str:
    import contextlib
    import io

    tokens = shlex.split(rest) if rest and rest.strip() else []
    if not tokens or tokens[0] in {"help", "--help", "-h", "?"}:
        return _SLASH_IDEAS_HELP

    _wrap = argparse.ArgumentParser(prog="/ideas-wrap", add_help=False)
    _wrap.exit_on_error = False  # type: ignore[attr-defined]
    _top_sub = _wrap.add_subparsers(dest="_top")
    ideas_parser = build_parser(_top_sub)
    ideas_parser.prog = "/ideas"

    argv = ["ideas", *tokens]
    try:
        args = _wrap.parse_args(argv)
    except argparse.ArgumentError as exc:
        return f"(._.) ideas error: {exc}"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = ideas_command(args)
    out = buf.getvalue().strip()
    if rc != 0 and out:
        return out
    if rc != 0:
        return f"(._.) ideas command failed (exit {rc})"
    return out or "(done)"
