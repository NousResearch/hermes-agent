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

# What a kind is called in a row. The identifiers stay English; only the label is localized.
_KIND_KEYS = {
    KIND_ROOT: "cli.tree.kind_root",
    KIND_BRANCH: "cli.tree.kind_branch",
    KIND_COMPRESSION: "cli.tree.kind_compression",
    KIND_SUBAGENT: "cli.tree.kind_subagent",
    KIND_UNKNOWN: "cli.tree.kind_unknown",
}

_INDENT = "  "


def _t(key: str, default: str, **kwargs: Any) -> str:
    """``t(key, **kwargs)`` with *default* when the catalog has no such key (an untranslated install)."""
    value = t(key, **kwargs)
    return default if value == key else value


def node_title(node: dict[str, Any]) -> str:
    """The row's title, or an honest ``untitled <id>`` — never a made-up name."""
    title = (node.get("title") or "").strip()
    return title or _t("cli.tree.untitled", "untitled {id}", id=node["session_id"][:8])


def usage_line() -> str:
    """The one usage string every surface prints for a bad /tree argument."""
    return _t("cli.tree.usage", "Usage: /tree [--all]")


def empty_line() -> str:
    """What every surface says when the session has no stored family yet."""
    return _t("cli.tree.empty", "No stored family for this session yet — send a message first.")


def _when(created_at: float) -> str:
    if not created_at:
        return ""
    return datetime.fromtimestamp(created_at).strftime("%d/%m %H:%M")


def node_line(node: dict[str, Any]) -> str:
    """One row: title, kind, size, creation time, then the facts that qualify it.

    ``after msg N`` is the parent's sealed message number, printed only where the store can prove
    it (a compression continuation). ``+N compressed`` is the segments folded into this row.
    """
    parts = [node_title(node), _t(_KIND_KEYS.get(node["kind"], "cli.tree.kind_unknown"),
                                 node["kind"])]
    count = node.get("message_count") or 0
    parts.append(_t("cli.tree.messages", "{count} msgs", count=count) if count != 1
                 else _t("cli.tree.messages_one", "1 msg", count=count))
    when = _when(node.get("created_at") or 0)
    if when:
        parts.append(when)
    if node.get("branch_from") is not None:
        parts.append(_t("cli.tree.after_message", "after msg {number}", number=node["branch_from"]))
    if node.get("folded_segments"):
        parts.append(_t("cli.tree.compressed_segments", "+{count} compressed",
                        count=node["folded_segments"]))
    if node.get("is_current"):
        parts.append(_t("cli.tree.current", "current"))
    return _INDENT * int(node.get("depth") or 0) + " · ".join(part for part in parts if part)


def tree_lines(family: SessionFamily) -> list[str]:
    """The family as indented text — the gateway's ``/tree`` and every non-curses CLI path.

    *family* is already reduced to the caller's view (``SessionDB.session_family`` applies the
    flags), so this only draws: re-applying the view here would re-parent it a second time.
    """
    lines = [node_line(node) for node in family.nodes]
    if family.truncated:
        lines.append(_t("cli.tree.truncated", "(family truncated: more sessions than this view lists)"))
    return lines


def tree_header(family: SessionFamily) -> str:
    """Title line for the text views: whose family this is, and how many rows it shows.

    The count is the rows the caller asked for (a folded continuation is one row, a hidden subagent
    run is none), so it never promises a row the view does not draw.
    """
    root = next((node for node in family.nodes if node["session_id"] == family.root_id), None)
    name = node_title(root) if root else family.root_id
    return _t("cli.tree.header", "Session tree · {title} · {count} session(s)",
              title=name, count=len(family.nodes))


def tree_text(family: SessionFamily) -> str:
    """The whole text answer for a text surface: header, rows, or why there is nothing to show.

    The messaging gateway's ``/tree`` and ``slash.exec`` print exactly this, so their output
    reads like the classic CLI's.
    """
    if not family.nodes:
        return empty_line()
    return "\n".join([tree_header(family), *tree_lines(family)])


def family_view(db: Any, session_id: str, *, show_compressions: bool = False,
                show_subagents: bool = False) -> tuple[SessionFamily, list[str]]:
    """``(family, lines)`` for *session_id* in this store, ready to print or pick from.

    Goes through ``SessionDB.session_family``, so the classic CLI's thin client store can answer it
    from its own read-only handle (a free function over ``_read_one`` would be refused there).
    """
    family = _family_from(db.session_family(session_id, show_compressions=show_compressions,
                                            show_subagents=show_subagents))
    return family, tree_lines(family)


def _family_from(payload: dict[str, Any]) -> SessionFamily:
    """The store's ``{root_id, nodes, truncated}`` as a :class:`SessionFamily`."""
    return SessionFamily(nodes=list(payload.get("nodes") or []),
                         root_id=str(payload.get("root_id") or ""),
                         truncated=bool(payload.get("truncated")))


def pick_session(db: Any, session_id: str, *, show_compressions: bool = False,
                 show_subagents: bool = False) -> Optional[str]:
    """Pick a session in the family around *session_id*; ``None`` when cancelled or there is no choice.

    The rows are the ones :func:`tree_lines` prints, so a picker row and a printed row read the
    same. ``curses_single_select`` gives arrows, ``/`` search, Enter and Esc; it cannot fold at a
    keypress (its item list is fixed for the life of the menu), so folding is a view argument here
    — the defaults fold compressions and hide subagents, and the TUI overlay toggles them live.
    """
    family, _lines = family_view(db, session_id, show_compressions=show_compressions,
                                 show_subagents=show_subagents)
    if not family.nodes:
        return None
    nodes = family.nodes
    if len(nodes) < 2:
        return None
    labels = [node_line(node) for node in nodes]
    current = next((index for index, node in enumerate(nodes) if node["is_current"]), 0)
    choice = curses_single_select(
        _t("cli.tree.title", "Session tree"), labels, default_index=current,
        cancel_label=_t("cli.tree.cancel", "Cancel"), searchable=True)
    return nodes[choice]["session_id"] if choice is not None else None
