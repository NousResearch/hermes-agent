"""``hermes kanban boards …`` — board directories, the ``current`` pointer and ``board.json``.
Filesystem-only, so every action works before ``kanban init`` and must ignore the shared
``--board`` task-routing override.
"""

from __future__ import annotations

import argparse
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_output import _err, _fmt_counts, _json_out


def _dispatch_boards(args: argparse.Namespace) -> int:
    """``hermes kanban boards <action>`` — filesystem-only, so it works before ``kanban init``."""
    sub = getattr(args, "boards_action", None) or "list"
    handler = _BOARD_HANDLERS.get(sub)
    if handler is None:
        return _err(f"kanban boards: unknown action {sub!r}", 2)
    return handler(args)


def _board_task_counts(slug: str) -> dict[str, int]:
    """``{status: count}`` for a board. Safe to call on an empty DB."""
    try:
        if not kb.kanban_db_path(board=slug).exists():
            return {}
        with kbc.connect_closing(board=slug) as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall()
        return {r["status"]: int(r["n"]) for r in rows}
    except Exception:
        return {}


def _board_slug_arg(args: argparse.Namespace, cmd: str, *, must_exist: bool) -> tuple[Optional[str], int]:
    """Normalize ``args.slug`` for a ``boards`` subcommand; ``(slug, 0)`` or ``(None, rc)``."""
    try:
        normed = kb._normalize_board_slug(args.slug)
    except ValueError as exc:
        return None, _err(f"kanban boards {cmd}: {exc}", 2)
    if must_exist:
        if not normed or not kb.board_exists(normed):
            return None, _err(f"kanban boards {cmd}: board {args.slug!r} does not exist")
    elif not normed:
        return None, _err(f"kanban boards {cmd}: slug is required", 2)
    return normed, 0


def _cmd_boards_list(args: argparse.Namespace) -> int:
    boards = kb.list_boards(include_archived=bool(getattr(args, "all", False)))
    current = kb.get_current_board()
    for b in boards:
        b["is_current"] = (b["slug"] == current)
        b["counts"] = _board_task_counts(b["slug"])
        b["total"] = sum(b["counts"].values())
    if _json_out(args, boards):
        return 0
    if not boards:
        print("(no boards — create one with `hermes kanban boards create <slug>`)")
        return 0
    print(f"{'':2s}  {'SLUG':24s}  {'NAME':28s}  COUNTS")
    for b in boards:
        marker = "●" if b["is_current"] else " "
        name = (b.get("name") or "") + (" [archived]" if b.get("archived") else "")
        print(f"{marker:2s}  {b['slug']:24s}  {name:28s}  {_fmt_counts(b['counts'] or {}, '(empty)')}")
    print(f"\nCurrent board: {current}")
    if len(boards) > 1:
        print("Switch boards with `hermes kanban boards switch <slug>`.")
    return 0


def _cmd_boards_create(args: argparse.Namespace) -> int:
    normed, rc = _board_slug_arg(args, "create", must_exist=False)
    if rc:
        return rc
    already = kb.board_exists(normed) and normed != kb.DEFAULT_BOARD
    meta = kb.create_board(
        normed, name=args.name, description=args.description, icon=args.icon, color=args.color,
        default_workdir=args.default_workdir,
    )
    print(f"Board {meta['slug']!r} {'already exists' if already else 'created'}.\n"
          f"  Display name: {meta.get('name', '')}\n"
          f"  DB path:      {meta['db_path']}")
    if getattr(args, "switch", False):
        kb.set_current_board(meta["slug"])
        print(f"  Switched to {meta['slug']!r}.")
    else:
        print(f"  Use `hermes kanban boards switch {meta['slug']}` to make it current.")
    return 0


def _cmd_boards_rm(args: argparse.Namespace) -> int:
    # `boards delete <slug>` (alias) never sets args.delete because --delete belongs to the 'rm'
    # subparser only; treat the alias as `rm --delete`.
    # See #23139.
    force_delete = getattr(args, "delete", False) or getattr(args, "boards_action", "") == "delete"
    try:
        res = kb.remove_board(args.slug, archive=not force_delete)
    except ValueError as exc:
        return _err(f"kanban boards rm: {exc}")
    if res["action"] == "archived":
        print(f"Board {res['slug']!r} archived → {res['new_path']}\n"
              "Recover by moving the directory back to <root>/kanban/boards/<slug>/.")
    else:
        print(f"Board {res['slug']!r} deleted.")
    return 0


def _cmd_boards_switch(args: argparse.Namespace) -> int:
    normed, rc = _board_slug_arg(args, "switch", must_exist=False)
    if rc:
        return rc
    if not kb.board_exists(normed):
        return _err(
            f"kanban boards switch: board {normed!r} does not exist. "
            f"Create it with `hermes kanban boards create {normed}`."
        )
    kb.set_current_board(normed)
    print(f"Active board is now {normed!r}.")
    return 0


def _cmd_boards_show(args: argparse.Namespace) -> int:
    current = kb.get_current_board()
    meta = kb.read_board_metadata(current)
    counts = _board_task_counts(current)
    print(f"Current board: {current}\n  Display name: {meta.get('name', '')}")
    if meta.get("description"):
        print(f"  Description:  {meta['description']}")
    print(f"  DB path:      {meta['db_path']}\n"
          f"  Tasks:        {sum(counts.values())} total" + (f" ({_fmt_counts(counts)})" if counts else ""))
    return 0


def _cmd_boards_rename(args: argparse.Namespace) -> int:
    normed, rc = _board_slug_arg(args, "rename", must_exist=True)
    if rc:
        return rc
    meta = kb.write_board_metadata(normed, name=args.name)
    print(f"Board {normed!r} renamed to {meta['name']!r}.")
    return 0


def _cmd_boards_set_default_workdir(args: argparse.Namespace) -> int:
    normed, rc = _board_slug_arg(args, "set-default-workdir", must_exist=True)
    if rc:
        return rc
    new_val = kb.write_board_metadata(normed, default_workdir=args.path).get("default_workdir")
    if new_val:
        print(f"Board {normed!r} default workdir set to {new_val!r}.")
    else:
        print(f"Board {normed!r} default workdir cleared.")
    return 0


def _cmd_boards_export(args: argparse.Namespace) -> int:
    from hermes_cli import kanban_transfer
    from hermes_cli.sizefmt import format_bytes

    slug = args.slug or kb.get_current_board()
    output = args.output or f"{slug}.tar.gz"
    try:
        res = kanban_transfer.export_board(
            slug, output, include_attachments=not args.no_attachments, include_logs=args.include_logs,
        )
    except (OSError, ValueError) as exc:
        return _err(f"kanban boards export: {exc}")
    if _json_out(args, res):
        return 0
    counts = res["counts"]
    print(f"Exported board {res['board']!r} → {res['archive']}\n"
          f"  Size:        {format_bytes(res['size'])}\n"
          f"  Tasks:       {counts['tasks']}\n"
          f"  Comments:    {counts['task_comments']}\n"
          f"  Attachments: {counts['attachment_files']}\n"
          "Import it with `hermes kanban boards import <archive>`.")
    return 0


def _cmd_boards_import(args: argparse.Namespace) -> int:
    from hermes_cli import kanban_transfer

    try:
        res = kanban_transfer.import_board(args.archive, args.as_slug, activate=args.switch)
    except (OSError, ValueError) as exc:
        return _err(f"kanban boards import: {exc}")
    if _json_out(args, res):
        return 0
    print(f"Imported board {res['board']!r} ({res['name']}).")
    if res["renamed"]:
        print(f"  Renamed from {res['requested_board']!r} — that slug was taken.")
    print(f"  Path:  {res['path']}\n  Tasks: {res['counts']['tasks']}")
    for warning in res["warnings"]:
        print(f"  Note:  {warning}")
    if res["activated"]:
        print(f"  Active board is now {res['board']!r}.")
    else:
        print(f"  Switch to it with `hermes kanban boards switch {res['board']}`.")
    return 0


_BOARD_HANDLERS = {
    "list": _cmd_boards_list, "ls": _cmd_boards_list,
    "create": _cmd_boards_create, "new": _cmd_boards_create,
    "rm": _cmd_boards_rm, "remove": _cmd_boards_rm, "delete": _cmd_boards_rm,
    "switch": _cmd_boards_switch, "use": _cmd_boards_switch,
    "show": _cmd_boards_show, "current": _cmd_boards_show,
    "rename": _cmd_boards_rename,
    "set-default-workdir": _cmd_boards_set_default_workdir,
    "export": _cmd_boards_export,
    "import": _cmd_boards_import,
}
