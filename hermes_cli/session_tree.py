"""``/tree``: the session-family navigator's words and its classic-CLI picker.

The family itself comes from :mod:`hermes_state_family`; this module owns the labels and the
curses picker, and the same text is what the messaging gateway and ``slash.exec`` print. The Ink
overlay draws the same nodes over ``session.family`` — only the drawing is written twice.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from agent.i18n import t
from hermes_cli.curses_ui import curses_single_select
from hermes_state_family import (
    KIND_BRANCH, KIND_COMPRESSION, KIND_ROOT, KIND_SUBAGENT, KIND_UNKNOWN, SessionFamily,
)

_KIND_KEYS = {
    KIND_ROOT: "cli.tree.kind_root",
    KIND_BRANCH: "cli.tree.kind_branch",
    KIND_COMPRESSION: "cli.tree.kind_compression",
    KIND_SUBAGENT: "cli.tree.kind_subagent",
    KIND_UNKNOWN: "cli.tree.kind_unknown",
}

_INDENT = "  "
_ALL_ARGS = {"--all", "all"}


def parse_show_all(arg: str) -> Optional[bool]:
    """``/tree [--all]`` argument: True for ``--all``, False for none, ``None`` for anything else."""
    tokens = (arg or "").lower().split()
    if any(token not in _ALL_ARGS for token in tokens):
        return None
    return bool(tokens)


def usage_line() -> str:
    return t("cli.tree.usage")


def empty_line() -> str:
    return t("cli.tree.empty")


def node_title(node: dict[str, Any]) -> str:
    """The row's title, or an honest ``untitled <id>`` — never a made-up name."""
    return (node.get("title") or "").strip() or t("cli.tree.untitled", id=node["session_id"][:8])


def _when(created_at: float) -> str:
    return datetime.fromtimestamp(created_at).strftime("%d/%m %H:%M") if created_at else ""


def node_line(node: dict[str, Any]) -> str:
    """One row: title, kind, size, creation time, then the facts that qualify it.

    ``after msg N`` is the parent's sealed message number, printed only where the store can prove
    it (a compression continuation). ``+N compressed`` is the segments folded into this row.
    """
    count = node.get("message_count") or 0
    parts = [
        node_title(node),
        t(_KIND_KEYS.get(node["kind"], "cli.tree.kind_unknown")),
        t("cli.tree.messages_one" if count == 1 else "cli.tree.messages", count=count),
        _when(node.get("created_at") or 0),
    ]
    if node.get("branch_from") is not None:
        parts.append(t("cli.tree.after_message", number=node["branch_from"]))
    if node.get("folded_segments"):
        parts.append(t("cli.tree.compressed_segments", count=node["folded_segments"]))
    if node.get("is_current"):
        parts.append(t("cli.tree.current"))
    return _INDENT * int(node.get("depth") or 0) + " · ".join(part for part in parts if part)


def tree_lines(family: SessionFamily) -> list[str]:
    """The family as indented text. *family* is already reduced to the caller's view."""
    lines = [node_line(node) for node in family.nodes]
    if family.truncated:
        lines.append(t("cli.tree.truncated"))
    return lines


def tree_text(family: SessionFamily) -> str:
    """Header plus rows, or why there is nothing to show — what every text surface prints."""
    if not family.nodes:
        return empty_line()
    root = next((node for node in family.nodes if node["session_id"] == family.root_id), None)
    header = t("cli.tree.header", title=node_title(root) if root else family.root_id,
               count=len(family.nodes))
    return "\n".join([header, *tree_lines(family)])


def load_family(db: Any, session_id: str, *, show_all: bool = False) -> SessionFamily:
    """The family around *session_id*; ``show_all`` also lists compressions and subagent runs.

    Goes through ``SessionDB.session_family`` so the classic CLI's thin-client store can answer
    it from its own read-only handle (a free function over ``_read_one`` is refused there).
    """
    payload = db.session_family(session_id, show_compressions=show_all, show_subagents=show_all)
    return SessionFamily(nodes=list(payload.get("nodes") or []),
                         root_id=str(payload.get("root_id") or ""),
                         truncated=bool(payload.get("truncated")))


def pick_session(family: SessionFamily) -> Optional[str]:
    """Pick a session from *family*; ``None`` when cancelled or there is nothing to switch to.

    ``curses_single_select`` gives arrows, ``/`` search, Enter and Esc. It cannot fold at a
    keypress (its item list is fixed for the life of the menu), so ``--all`` is the CLI's view
    toggle and the TUI overlay folds live.
    """
    nodes = family.nodes
    if len(nodes) < 2:
        return None
    current = next((index for index, node in enumerate(nodes) if node["is_current"]), 0)
    choice = curses_single_select(
        t("cli.tree.title"), [node_line(node) for node in nodes], default_index=current,
        cancel_label=t("cli.tree.cancel"), searchable=True)
    return nodes[choice]["session_id"] if choice is not None else None
