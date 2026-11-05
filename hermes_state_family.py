"""Session-family reads: the ancestry walk behind ``/tree``.

Read-only. Every statement here goes through ``SessionDB._read_all`` / ``_read_one``: nothing
writes, flushes or mutates, so the same builder is safe inside a live turn and behind an RPC.
One builder serves the classic CLI, the Ink TUI (``session.family``) and the messaging gateway,
so all three draw the same family with the same kinds.

Kinds come only from markers the store already writes (``sessions.parent_session_id`` plus the
``model_config`` markers ``_branched_from`` / ``_delegate_from`` / ``_reset_from``, ``source``
and the parent's ``end_reason``). A node whose marker does not bind to its parent is ``unknown``:
this module never guesses a relationship it cannot prove.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional

KIND_ROOT = "root"
KIND_BRANCH = "branch"
KIND_COMPRESSION = "compression"
KIND_SUBAGENT = "subagent"
KIND_UNKNOWN = "unknown"

# Every kind a node can carry, in the order a reader should read them.
KINDS = (KIND_ROOT, KIND_BRANCH, KIND_COMPRESSION, KIND_SUBAGENT, KIND_UNKNOWN)

# Hard stops: a corrupt store can point a row at itself, or two rows at each other.
MAX_DEPTH = 200
MAX_NODES = 500

# Only what the tree draws: no prompts, no message bodies, no cost totals.
_COLUMNS = "id, parent_session_id, title, started_at, message_count, end_reason, source, model_config"


@dataclass(frozen=True)
class SessionFamily:
    """One session's family: nodes in depth-first pre-order, the root first.

    ``truncated`` is True when the walk hit ``MAX_NODES`` / ``MAX_DEPTH`` and stopped early —
    the caller must say so rather than present a partial family as the whole one.
    """

    nodes: list[dict[str, Any]]
    root_id: str
    truncated: bool = False


class _Rows:
    """Per-call row cache so the walk reads each session once, and kind checks reuse its parents."""

    def __init__(self, db: Any) -> None:
        self._db = db
        self._rows: dict[str, Optional[dict[str, Any]]] = {}

    def get(self, session_id: Optional[str]) -> Optional[dict[str, Any]]:
        if not session_id:
            return None
        if session_id not in self._rows:
            row = self._db._read_one(f"SELECT {_COLUMNS} FROM sessions WHERE id = ?", (session_id,))
            self._rows[session_id] = dict(row) if row else None
        return self._rows[session_id]

    def children(self, parent_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = list(parent_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._db._read_all(
            f"SELECT {_COLUMNS} FROM sessions WHERE parent_session_id IN ({placeholders}) "
            "ORDER BY started_at ASC, id ASC",
            tuple(ids),
        )
        return [dict(row) for row in rows]


def _markers(row: dict[str, Any]) -> dict[str, Any]:
    """``model_config`` as a dict; a missing or unparseable value carries no markers."""
    config = row.get("model_config")
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except (TypeError, ValueError):
            return {}
    return config if isinstance(config, dict) else {}


def _marker_points_at(row: dict[str, Any], name: str, parent_id: Optional[str]) -> bool:
    """Whether *name* names this row's parent.

    Same binding as ``SessionDB._is_explicit_fork_child_row``: a marker only counts when it
    equals ``parent_session_id``, because compression copies ``model_config`` onto its
    continuation. With no parent at all, any marker value is foreign and does not count as a link.
    """
    value = _markers(row).get(name)
    return value == parent_id if parent_id else False


def _is_explicit_fork(row: dict[str, Any], parent_id: Optional[str]) -> bool:
    """Branch, delegate or reset child of *parent_id* — never a compression continuation."""
    if row.get("source") == "tool":
        return True
    return any(_marker_points_at(row, marker, parent_id)
               for marker in ("_branched_from", "_delegate_from", "_reset_from"))


def classify_kind(db: Any, row: dict[str, Any], parent: Optional[dict[str, Any]]) -> str:
    """The kind the stored markers prove for *row*, or ``unknown``.

    Order matters and mirrors the store's own helpers: a tool/delegate row is a subagent before
    it can be anything else; an explicit branch marker wins over the compression rule because
    ``_is_compression_child_row`` excludes explicit forks itself.
    """
    parent_id = row.get("parent_session_id")
    if parent is None:
        return KIND_ROOT
    if row.get("source") == "tool" or _marker_points_at(row, "_delegate_from", parent_id):
        return KIND_SUBAGENT
    if _marker_points_at(row, "_branched_from", parent_id):
        return KIND_BRANCH
    # The store's rule (``_is_compression_child_row``), read against the parent row already in
    # hand: a sealed segment of ``end_reason == 'compression'`` continues into this child.
    if parent.get("end_reason") == "compression" and not _is_explicit_fork(row, parent_id):
        return KIND_COMPRESSION
    return KIND_UNKNOWN


def _branch_from(kind: str, parent: Optional[dict[str, Any]]) -> Optional[int]:
    """The parent's message number this node continues from, when the store can prove it.

    Only a compression continuation can: its parent is sealed, so the parent's message count is
    exactly where the segment ended. A branch copies a prefix and then grows, and the copied
    count is not persisted, so its branch point is not knowable — reported as ``None``, never
    estimated from today's counts.
    """
    if kind != KIND_COMPRESSION or parent is None:
        return None
    return int(parent.get("message_count") or 0)


def _node(db: Any, row: dict[str, Any], parent: Optional[dict[str, Any]], depth: int,
          current_id: str) -> dict[str, Any]:
    kind = classify_kind(db, row, parent)
    return {
        "session_id": row["id"],
        "parent_session_id": row.get("parent_session_id"),
        "depth": depth,
        "title": row.get("title") or "",
        "created_at": float(row.get("started_at") or 0),
        "message_count": int(row.get("message_count") or 0),
        "kind": kind,
        "branch_from": _branch_from(kind, parent),
        "is_current": row["id"] == current_id,
        "has_children": False,
        "folded_segments": 0,
    }


def _ancestors(rows: _Rows, session_id: str) -> list[dict[str, Any]]:
    """*session_id* then each parent up to the root, stopping on a cycle or ``MAX_DEPTH``."""
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = rows.get(session_id)
    while current is not None and len(chain) < MAX_DEPTH:
        if current["id"] in seen:
            break
        seen.add(current["id"])
        chain.append(current)
        current = rows.get(current.get("parent_session_id"))
    return chain


def _children_by_parent(rows: _Rows, root_id: str, max_nodes: int) -> tuple[dict[str, list], bool]:
    """Every descendant of *root_id*, one query per level, truncated at *max_nodes*."""
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frontier, seen, total, truncated = [root_id], {root_id}, 1, False
    while frontier and not truncated:
        level = rows.children(frontier)
        frontier = []
        for row in level:
            parent_id = row.get("parent_session_id")
            if parent_id is None or row["id"] in seen:  # a row with no link, or a cycle back onto the path
                continue
            seen.add(row["id"])
            if total >= max_nodes:
                truncated = True
                break
            children[parent_id].append(row)
            frontier.append(row["id"])
            total += 1
    return children, truncated


def build_session_family(db: Any, session_id: str, *, max_nodes: int = MAX_NODES) -> SessionFamily:
    """The family around *session_id*: walk up to the root, then every descendant.

    Returns an empty family when *session_id* is not in this store — which is also how a
    request for another profile's session resolves, because each profile reads its own DB.
    """
    rows = _Rows(db)
    ancestors = _ancestors(rows, session_id)
    if not ancestors:
        return SessionFamily(nodes=[], root_id=session_id)
    root = ancestors[-1]
    children, truncated = _children_by_parent(rows, root["id"], max_nodes)
    nodes: list[dict[str, Any]] = []
    stack: list[tuple[dict[str, Any], Optional[dict[str, Any]], int]] = [(root, None, 0)]
    while stack:
        row, parent, depth = stack.pop()
        if depth > MAX_DEPTH or len(nodes) >= max_nodes:
            truncated = True
            break
        nodes.append(_node(db, row, parent, depth, session_id))
        for child in reversed(children.get(row["id"], [])):
            stack.append((child, row, depth + 1))
    return SessionFamily(nodes=nodes, root_id=root["id"], truncated=truncated)


def visible_nodes(
    nodes: list[dict[str, Any]], *, show_compressions: bool = False,
    show_subagents: bool = False, folded: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Apply the client's view to a family: the default view, then its toggles.

    Default (both flags off): a compression continuation folds into its nearest shown ancestor
    and is replaced by a ``folded_segments`` count on that ancestor, so a compressed conversation
    reads as one line; subagent runs are hidden. ``folded`` hides the descendants of those ids.
    ``depth`` is recomputed, so a caller only ever indents by the value it is given.
    """
    folded_ids = set(folded)
    depth_of: dict[str, int] = {}
    anchor_of: dict[Optional[str], Optional[str]] = {}
    folded_segments: dict[str, int] = {}
    # Ids that carry at least one child in the family, whatever the view hides.
    family_parents = {node.get("parent_session_id") for node in nodes}
    out: list[dict[str, Any]] = []

    for node in nodes:
        session_id, parent_id, kind = node["session_id"], node.get("parent_session_id"), node["kind"]
        hidden = (kind == KIND_SUBAGENT and not show_subagents) or (
            kind == KIND_COMPRESSION and not show_compressions)
        if hidden:
            anchor = anchor_of.get(parent_id)
            anchor_of[session_id] = anchor
            if kind == KIND_COMPRESSION and anchor is not None:
                folded_segments[anchor] = folded_segments.get(anchor, 0) + 1
            continue
        anchor = anchor_of.get(parent_id)
        if anchor is not None and anchor in folded_ids:
            anchor_of[session_id] = anchor
            continue
        visible = dict(node)
        visible["depth"] = 0 if anchor is None else depth_of[anchor] + 1
        out.append(visible)
        depth_of[session_id] = visible["depth"]
        anchor_of[session_id] = session_id

    for index, node in enumerate(out):
        node["folded_segments"] = folded_segments.get(node["session_id"], 0)
        following = out[index + 1] if index + 1 < len(out) else None
        # A folded row has nothing drawn under it, so its own family children cannot show it is
        # foldable — count the fold itself, or the ▸ arrow would appear on leaf rows only.
        node["has_children"] = bool(following and following["depth"] > node["depth"]) or (
            node["session_id"] in folded_ids and node["session_id"] in family_parents)
    return out


class SessionFamilyMixin:
    """``/tree``'s family read, bound onto ``SessionDB``.

    It is a store METHOD, not a free function over a handle, because the classic CLI's session
    store is a thin client: only names in ``gateway_session_store._READS`` reach its read-only
    ``state.db`` reader, and anything else refuses. One method therefore serves all three
    surfaces — the CLI's local reader, the gateway's store, and the TUI's RPC.
    """

    def session_family(
        self, session_id: str, *, show_compressions: bool = False,
        show_subagents: bool = False, folded: Iterable[str] = (),
    ) -> dict[str, Any]:
        """``{root_id, nodes, truncated}`` for *session_id*, already reduced to the caller's view.

        Read-only: the walk reads rows and the view flags only drop rows. A session this store
        does not hold yields empty ``nodes``, which is how another profile's request resolves.
        """
        family = build_session_family(self, session_id)
        nodes = visible_nodes(family.nodes, show_compressions=show_compressions,
                              show_subagents=show_subagents, folded=folded)
        return {"root_id": family.root_id, "nodes": nodes, "truncated": family.truncated}
