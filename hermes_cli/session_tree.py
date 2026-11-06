"""``/tree``: the session-family navigator's words and the row layout every surface draws.

The family itself comes from :mod:`hermes_state_family` (kinds, fold rules, tree guides, previews);
this module turns a node into the cells each surface shows — guide, kind glyph, title, kind,
message weight, age — and the text the messaging gateway and ``slash.exec`` print. The classic
CLI's full-screen browser is :mod:`hermes_cli.session_tree_browser`; the Ink overlay draws the same
cells from the same ``session.family`` nodes, so only the drawing is written twice.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from agent.i18n import t
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
# One mark per kind, the same in every surface: the original is solid, a branch hollow.
KIND_GLYPHS = {
    KIND_ROOT: "◆", KIND_BRANCH: "◇", KIND_COMPRESSION: "≋", KIND_SUBAGENT: "✦", KIND_UNKNOWN: "·",
}
HERE_GLYPH = "☤"
_WEIGHT_BARS = "▁▂▃▄▅▆▇█"
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


def kind_label(kind: str) -> str:
    return t(_KIND_KEYS.get(kind, "cli.tree.kind_unknown"))


def kind_glyph(kind: str) -> str:
    return KIND_GLYPHS.get(kind, KIND_GLYPHS[KIND_UNKNOWN])


def row_glyph(node: dict[str, Any]) -> str:
    """The row's mark: the caduceus where you are, otherwise its kind."""
    return HERE_GLYPH if node.get("is_current") else kind_glyph(node["kind"])


def messages_label(count: int) -> str:
    return t("cli.tree.messages_one" if count == 1 else "cli.tree.messages", count=count)


def weight_bar(count: int, heaviest_count: int) -> str:
    """A one-cell bar for a row's message count, on a log scale against the family's largest."""
    if count <= 0 or heaviest_count <= 0:
        return " "
    level = round((len(_WEIGHT_BARS) - 1) * math.log1p(count) / math.log1p(heaviest_count))
    return _WEIGHT_BARS[min(level, len(_WEIGHT_BARS) - 1)]


def heaviest(nodes: list[dict[str, Any]]) -> int:
    return max((int(node.get("message_count") or 0) for node in nodes), default=0)


def row_notes(node: dict[str, Any]) -> list[str]:
    """What qualifies a row after its age: where it continues, what is folded, where you are."""
    notes = []
    if node.get("branch_from") is not None:
        notes.append(t("cli.tree.after_message", number=node["branch_from"]))
    if node.get("folded_segments"):
        notes.append(t("cli.tree.compressed_segments", count=node["folded_segments"]))
    if node.get("folded"):
        notes.append(t("cli.tree.folded"))
    if node.get("is_current"):
        notes.append(t("cli.tree.here"))
    return notes


def node_line(node: dict[str, Any], heaviest_count: int = 0) -> str:
    """One row as plain text: guide, glyph and title, then kind, size, age and notes."""
    count = int(node.get("message_count") or 0)
    size = messages_label(count)
    if heaviest_count:
        size = f"{weight_bar(count, heaviest_count)} {size}"
    head = f"{node.get('guide') or ''}{row_glyph(node)} {node_title(node)}"
    parts = [kind_label(node["kind"]), size, node.get("age") or "", *row_notes(node)]
    return " · ".join([head, *(part for part in parts if part)])


def tree_lines(family: SessionFamily) -> list[str]:
    """The family as text rows. *family* is already reduced to the caller's view."""
    top = heaviest(family.nodes)
    lines = [node_line(node, top) for node in family.nodes]
    if family.truncated:
        lines.append(t("cli.tree.truncated"))
    return lines


def tree_header(family: SessionFamily) -> str:
    root = next((node for node in family.nodes if node["session_id"] == family.root_id), None)
    return t("cli.tree.header", title=node_title(root) if root else family.root_id)


def count_label(family: SessionFamily) -> str:
    return t("cli.tree.count", count=len(family.nodes))


def tree_text(family: SessionFamily) -> str:
    """Header plus rows, or why there is nothing to show — what every text surface prints."""
    if not family.nodes:
        return empty_line()
    return "\n".join([f"{tree_header(family)} · {count_label(family)}", *tree_lines(family)])


def preview_lines(node: dict[str, Any]) -> list[tuple[str, str]]:
    """The preview pane for one row as ``(label, text)`` pairs: its facts, then its last exchange."""
    if node.get("parent_session_id"):
        parent = node.get("parent_title") or t("cli.tree.untitled", id=node["parent_session_id"][:8])
        facts = [t("cli.tree.from_parent", kind=kind_label(node["kind"]), title=parent)]
    else:
        facts = [kind_label(node["kind"])]
    facts.append(messages_label(int(node.get("message_count") or 0)))
    if node.get("age"):
        facts.append(t("cli.tree.last_used", age=node["age"]))
    lines = [(f"{kind_glyph(node['kind'])} {node_title(node)}", " · ".join(facts))]
    if node.get("last_prompt"):
        lines.append((t("cli.tree.you"), node["last_prompt"]))
    if node.get("last_reply"):
        lines.append((t("cli.tree.hermes"), node["last_reply"]))
    if len(lines) == 1:
        lines.append(("", t("cli.tree.no_messages")))
    return lines


def hint_pairs() -> list[tuple[str, str]]:
    """Footer key hints as ``(key, action)`` — the TUI overlay shows the same keys."""
    return [("↑↓", t("cli.tree.hint_move")), ("←→", t("cli.tree.hint_fold")),
            ("⏎", t("cli.tree.hint_switch")), ("/", t("cli.tree.hint_search")),
            ("c", t("cli.tree.hint_compressed")), ("s", t("cli.tree.hint_subagents")),
            ("esc", t("cli.tree.hint_close"))]


def load_family(db: Any, session_id: str, *, show_all: bool = False,
                show_compressions: Optional[bool] = None, show_subagents: Optional[bool] = None,
                folded: tuple[str, ...] = ()) -> SessionFamily:
    """The family around *session_id* under one view (``show_all`` sets both list toggles).

    Goes through ``SessionDB.session_family`` so the classic CLI's thin-client store can answer
    it from its own read-only handle (a free function over ``_read_one`` is refused there).
    """
    payload = db.session_family(
        session_id,
        show_compressions=show_all if show_compressions is None else show_compressions,
        show_subagents=show_all if show_subagents is None else show_subagents,
        folded=list(folded))
    nodes = list(payload.get("nodes") or [])
    for node in nodes:
        node["folded"] = node["session_id"] in folded and bool(node.get("has_children"))
    return SessionFamily(nodes=nodes, root_id=str(payload.get("root_id") or ""),
                         truncated=bool(payload.get("truncated")))
