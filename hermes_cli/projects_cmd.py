"""``hermes project`` CLI — manage first-class, multi-folder Projects.

A Project is a human-named workspace spanning one or more folders, with one
designated primary repo. Projects anchor desktop session grouping and (when
bound to a kanban board) give kanban tasks a deterministic worktree + branch
convention. State lives in the per-profile ``$HERMES_HOME/projects.db`` store
(see :mod:`hermes_cli.projects_db`).

This is a footprint-ladder rung-2 capability: a CLI command + gateway RPC,
with zero model-tool schema cost.
"""

from __future__ import annotations

import argparse
import functools
import sys

from hermes_cli import projects_db as pdb


def build_parser(
    parent_subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Attach the ``project`` subcommand tree. Returns the top parser."""
    parser = parent_subparsers.add_parser(
        "project",
        help="Manage projects (named, multi-folder workspaces)",
        description=(
            "Projects are human-named workspaces that can span multiple "
            "folders / repos. They anchor desktop session grouping and, when "
            "bound to a kanban board, give tasks a deterministic worktree + "
            "branch convention. State is per-profile."
        ),
    )
    sub = parser.add_subparsers(dest="project_action")

    p_create = sub.add_parser("create", help="Create a new project")
    p_create.add_argument("name", help="Human name, e.g. 'Hermes Agent'")
    p_create.add_argument(
        "folders", nargs="*", help="Folder paths to include (first = primary)"
    )
    p_create.add_argument("--slug", default=None, help="Explicit slug override")
    p_create.add_argument(
        "--primary", default=None, metavar="PATH", help="Primary repo path"
    )
    p_create.add_argument("--description", default=None)
    p_create.add_argument("--icon", default=None)
    p_create.add_argument("--color", default=None)
    p_create.add_argument(
        "--board", default=None, metavar="SLUG", help="Bind a kanban board"
    )
    p_create.add_argument(
        "--use", action="store_true", help="Set as the active project"
    )

    p_list = sub.add_parser("list", aliases=["ls"], help="List projects")
    p_list.add_argument(
        "--all", action="store_true", dest="include_archived",
        help="Include archived projects",
    )

    p_show = sub.add_parser("show", help="Show a project's details")
    p_show.add_argument("project", help="Project id or slug")

    p_add = sub.add_parser("add-folder", help="Add a folder to a project")
    p_add.add_argument("project", help="Project id or slug")
    p_add.add_argument("path", help="Folder path")
    p_add.add_argument("--label", default=None)
    p_add.add_argument(
        "--primary", action="store_true", help="Mark as primary repo"
    )

    p_rm = sub.add_parser("remove-folder", help="Remove a folder from a project")
    p_rm.add_argument("project", help="Project id or slug")
    p_rm.add_argument("path", help="Folder path")

    p_mv = sub.add_parser(
        "move-folder",
        help="Move/redirect a project's folder to a new absolute path",
        description=(
            "Rewrites a folder's path on a project without removing it. "
            "The folder's is_primary / label / added_at are preserved; when the "
            "moved folder is the project's primary, projects.primary_path is "
            "updated too. Use this after a folder was renamed or moved on disk."
        ),
    )
    p_mv.add_argument("project", help="Project id or slug")
    p_mv.add_argument("path", help="Current folder path")
    p_mv.add_argument("new_path", help="New folder path (absolute)")

    p_rename = sub.add_parser("rename", help="Rename a project")
    p_rename.add_argument("project", help="Project id or slug")
    p_rename.add_argument("name", help="New name")

    p_primary = sub.add_parser("set-primary", help="Set the primary folder")
    p_primary.add_argument("project", help="Project id or slug")
    p_primary.add_argument("path", help="Folder path (must already be in project)")

    p_use = sub.add_parser("use", help="Set the active project")
    p_use.add_argument(
        "project", nargs="?", default=None,
        help="Project id or slug (omit to clear)",
    )

    p_archive = sub.add_parser("archive", help="Archive a project")
    p_archive.add_argument("project", help="Project id or slug")

    p_restore = sub.add_parser("restore", help="Restore an archived project")
    p_restore.add_argument("project", help="Project id or slug")

    p_bind = sub.add_parser("bind-board", help="Bind a kanban board to a project")
    p_bind.add_argument("project", help="Project id or slug")
    p_bind.add_argument(
        "board", nargs="?", default="", help="Board slug (omit to unbind)"
    )

    parser.set_defaults(_project_parser=parser)
    return parser


def projects_command(args: argparse.Namespace) -> int:
    """Entry point from ``hermes project …`` argparse dispatch."""
    action = getattr(args, "project_action", None)
    if not action:
        parser = getattr(args, "_project_parser", None)
        if parser is not None:
            parser.print_help()
        else:
            print(
                "usage: hermes project <action> [options]\n"
                "Run 'hermes project --help' for the full list.",
                file=sys.stderr,
            )
        return 0

    handlers = {
        "create": _cmd_create,
        "list": _cmd_list,
        "ls": _cmd_list,
        "show": _cmd_show,
        "add-folder": _cmd_add_folder,
        "remove-folder": _cmd_remove_folder,
        "move-folder": _cmd_move_folder,
        "rename": _cmd_rename,
        "set-primary": _cmd_set_primary,
        "use": _cmd_use,
        "archive": _cmd_archive,
        "restore": _cmd_restore,
        "bind-board": _cmd_bind_board,
    }
    handler = handlers.get(action)
    if handler is None:
        print(f"Unknown project action: {action}", file=sys.stderr)
        return 1
    return handler(args)


def _resolve(conn, ident: str):
    proj = pdb.get_project(conn, ident)
    if proj is None:
        print(f"project: no such project: {ident}", file=sys.stderr)
    return proj


def _with_project(fn):
    """Open the DB, resolve ``args.project``, and run ``fn(args, conn, proj)``.

    Collapses the connect / resolve / not-found(1) / bad-arg(2) boilerplate every
    project-scoped subcommand repeated.
    """

    @functools.wraps(fn)
    def wrapper(args: argparse.Namespace) -> int:
        with pdb.connect_closing() as conn:
            proj = _resolve(conn, args.project)
            if proj is None:
                return 1
            try:
                return fn(args, conn, proj)
            except ValueError as exc:
                print(f"project: {exc}", file=sys.stderr)
                return 2

    return wrapper


def _print_project(proj) -> None:
    flags = " (archived)" if proj.archived else ""
    print(f"{proj.slug}  [{proj.id}]{flags}")
    print(f"  name:    {proj.name}")
    if proj.description:
        print(f"  about:   {proj.description}")
    if proj.board_slug:
        print(f"  board:   {proj.board_slug}")
    if proj.primary_path:
        print(f"  primary: {proj.primary_path}")
    if proj.folders:
        print("  folders:")
        for f in proj.folders:
            mark = " *" if f.is_primary else "  "
            label = f" ({f.label})" if f.label else ""
            print(f"   {mark} {f.path}{label}")


def _cmd_create(args: argparse.Namespace) -> int:
    try:
        with pdb.connect_closing() as conn:
            pid = pdb.create_project(
                conn,
                name=args.name,
                slug=args.slug,
                folders=args.folders,
                primary_path=args.primary,
                description=args.description,
                icon=args.icon,
                color=args.color,
                board_slug=args.board,
            )
            if args.use:
                pdb.set_active(conn, pid)
            proj = pdb.get_project(conn, pid)
    except ValueError as exc:
        print(f"project: {exc}", file=sys.stderr)
        return 2
    if proj is None:
        print("project: vanished after create", file=sys.stderr)
        return 2
    print(f"Created project {proj.slug} ({pid})")
    _print_project(proj)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    with pdb.connect_closing() as conn:
        active = pdb.get_active_id(conn)
        projs = pdb.list_projects(
            conn, include_archived=getattr(args, "include_archived", False)
        )
    if not projs:
        print("No projects yet. Create one with `hermes project create <name>`.")
        return 0
    for p in projs:
        marker = "*" if p.id == active else " "
        flags = " (archived)" if p.archived else ""
        nfolders = len(p.folders)
        print(f"{marker} {p.slug:<24} {p.name}{flags}  [{nfolders} folder(s)]")
    return 0


@_with_project
def _cmd_show(args, conn, proj) -> int:
    _print_project(proj)
    return 0


@_with_project
def _cmd_add_folder(args, conn, proj) -> int:
    path = pdb.add_folder(conn, proj.id, args.path, label=args.label, is_primary=args.primary)
    print(f"Added {path} to {proj.slug}")
    return 0


@_with_project
def _cmd_remove_folder(args, conn, proj) -> int:
    if not pdb.remove_folder(conn, proj.id, args.path):
        print(f"project: folder not in project: {args.path}", file=sys.stderr)
        return 1
    print(f"Removed {args.path} from {proj.slug}")
    return 0


@_with_project
def _cmd_move_folder(args, conn, proj) -> int:
    """Rewrite a folder's path on this project without removing it.

    The folder's is_primary / label / added_at are preserved; when the moved
    folder is the project's primary, projects.primary_path is updated too.
    ValueError (unknown folder, collision, bad path) bubbles to ``_with_project``,
    which prints ``project: <reason>`` and returns 2.
    """
    new_path = pdb.move_folder(conn, proj.id, args.path, args.new_path)
    print(f"Moved {args.path} -> {new_path} on {proj.slug}")
    return 0


@_with_project
def _cmd_rename(args, conn, proj) -> int:
    pdb.update_project(conn, proj.id, name=args.name)
    print(f"Renamed {proj.slug} -> {args.name}")
    return 0


@_with_project
def _cmd_set_primary(args, conn, proj) -> int:
    if not pdb.set_primary(conn, proj.id, args.path):
        print(
            f"project: '{args.path}' is not a folder of {proj.slug}; "
            f"add it first with `hermes project add-folder`.",
            file=sys.stderr,
        )
        return 1
    print(f"Set primary of {proj.slug} -> {args.path}")
    return 0


def _cmd_use(args: argparse.Namespace) -> int:
    with pdb.connect_closing() as conn:
        if not args.project:
            pdb.set_active(conn, None)
            print("Cleared active project")
            return 0
        proj = _resolve(conn, args.project)
        if proj is None:
            return 1
        pdb.set_active(conn, proj.id)
    print(f"Active project: {proj.slug}")
    return 0


@_with_project
def _cmd_archive(args, conn, proj) -> int:
    pdb.archive_project(conn, proj.id)
    print(f"Archived {proj.slug}")
    return 0


@_with_project
def _cmd_restore(args, conn, proj) -> int:
    pdb.restore_project(conn, proj.id)
    print(f"Restored {proj.slug}")
    return 0


@_with_project
def _cmd_bind_board(args, conn, proj) -> int:
    pdb.update_project(conn, proj.id, board_slug=args.board)
    if args.board.strip():
        print(f"Bound {proj.slug} -> board {args.board}")
        _sync_board_default_workdir(proj, args.board)
    else:
        print(f"Unbound board from {proj.slug}")
    return 0


def _sync_board_default_workdir(proj, board_slug: str) -> None:
    """Best-effort: point the bound board's default_workdir at the primary repo.

    Keeps kanban task worktrees anchored to the project's repo. Failures here
    are non-fatal — the binding itself already succeeded.
    """
    if not proj.primary_path:
        return
    try:
        from hermes_cli import kanban_db as kb

        slug = kb._normalize_board_slug(board_slug)
        if not slug:
            return
        if slug != kb.DEFAULT_BOARD and not kb.board_exists(slug):
            return
        kb.write_board_metadata(slug, default_workdir=proj.primary_path)
    except Exception:
        pass
