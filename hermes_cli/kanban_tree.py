"""Kanban task-tree renderer — trace every subtask back to its original parent.

Invoked by ``hermes kanban tree [task_id]``. Reads the durable ``task_links``
edges (the same parent/child graph that ``show`` reports and ``decompose``
writes) and renders it as:

  - a Jira-style ASCII forest/subtree (default),
  - a nested JSON structure (``--json``), or
  - a Mermaid ``flowchart TD`` definition (``--mermaid``), which the Hermes
    desktop app already renders from ```mermaid`` fences.

Read-only. No state mutation; the tree is derived entirely from ``task_links``
plus the ``tasks`` table. Sibling module mirrors ``kanban_decompose`` /
``kanban_specify`` — holds the logic, while ``hermes_cli/kanban.py`` holds the
thin argparse/exit-code handler.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from hermes_cli import kanban_db as kb

# Box-drawing branch glyphs (Jira-style hierarchy).
_TREE_LAST = "└─"
_TREE_MID = "├─"
_TREE_CONT = "│  "
_TREE_SPACE = "   "

_STATUS_ICON = {  # mirrors kanban_output._STATUS_ICONS (kept local to avoid a circular import)
    "todo": "◻", "ready": "▶", "running": "●", "scheduled": "⏱",
    "blocked": "⊘", "done": "✓", "archived": "—",
}


def _load_adjacency(
    conn,
    *,
    include_archived: bool = False,
) -> Tuple[Dict[str, kb.Task], Dict[str, List[str]], Dict[str, List[str]]]:
    """Bulk-load the board's tasks and ``task_links`` edges in two queries.

    Returns ``(tasks_by_id, children_by_id, parents_by_id)``. Only non-archived
    tasks are walked unless ``include_archived`` is set. Every edge participates
    in the parent map even when the other endpoint is filtered out, so roots
    are computed against the full link set (a task whose parent is archived is
    still not a root).
    """
    rows = conn.execute(
        "SELECT * FROM tasks"
        + ("" if include_archived else " WHERE status != 'archived'")
    ).fetchall()
    tasks = {r["id"]: kb.Task.from_row(r) for r in rows}

    children: Dict[str, List[str]] = {}
    parents: Dict[str, List[str]] = {}
    for r in conn.execute(
        "SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id"
    ).fetchall():
        p, c = r["parent_id"], r["child_id"]
        children.setdefault(p, []).append(c)
        parents.setdefault(c, []).append(p)
    return tasks, children, parents


def _node(
    task_id: str,
    tasks: Dict[str, kb.Task],
    children: Dict[str, List[str]],
    *,
    _stack: Optional[set] = None,
) -> Dict[str, Any]:
    """Recursively nest one task and its descendants into a tree node dict.

    ``_stack`` guards against cycles — the DB link guards forbid them, but the
    renderer terminates regardless so a corrupt board cannot hang the CLI. A
    node already on the recursion stack is rendered as a leaf (its subtrees
    are skipped) rather than recursing forever.
    """
    _stack = set() if _stack is None else _stack
    task = tasks[task_id]
    node: Dict[str, Any] = {
        "id": task_id,
        "title": task.title,
        "status": task.status,
        "assignee": task.assignee,
        "children": [],
    }
    if task_id in _stack:
        return node
    _stack.add(task_id)
    # Deterministic ordering: priority desc, then created_at asc (matches list --sort priority).
    cids = sorted(children.get(task_id, ()), key=lambda c: (
        -tasks[c].priority, tasks[c].created_at, c,
    ))
    node["children"] = [
        _node(cid, tasks, children, _stack=_stack) for cid in cids
        if cid in tasks
    ]
    _stack.discard(task_id)
    return node


def build_forest(
    conn,
    *,
    root_id: Optional[str] = None,
    include_archived: bool = False,
) -> List[Dict[str, Any]]:
    """Return a list of root tree nodes for ``hermes kanban tree``.

    With ``root_id``: exactly that task's subtree (raises if the task is
    unknown or archived-excluded). Without: every task with no parent edge —
    the board as a forest. Each node is ``{id, title, status, assignee,
    children: [...]}``.
    """
    tasks, children, parents = _load_adjacency(conn, include_archived=include_archived)

    if root_id is not None:
        if root_id not in tasks:
            raise ValueError(
                f"no such task {root_id}"
                + (" (it is archived; pass --archived to include archived tasks)"
                   if kb.get_task(conn, root_id) and not include_archived else "")
            )
        return [_node(root_id, tasks, children)]

    roots = [tid for tid in tasks if not parents.get(tid)]
    roots.sort(key=lambda t: (-tasks[t].priority, tasks[t].created_at, t))
    return [_node(tid, tasks, children) for tid in roots]


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def render_ascii(forest: List[Dict[str, Any]]) -> str:
    """Render a forest of tree nodes as a Jira-style indented hierarchy."""
    lines: List[str] = []

    def fmt_label(node: Dict[str, Any]) -> str:
        icon = _STATUS_ICON.get(node["status"], "?")
        assignee = f"  [{node['assignee']}]" if node.get("assignee") else ""
        return f"{node['id']}  {icon} {node['status']:8s}  {node['title']}{assignee}"

    def emit(node: Dict[str, Any], prefix: str, is_last: bool) -> None:
        branch = _TREE_LAST if is_last else _TREE_MID
        lines.append(f"{prefix}{branch} {fmt_label(node)}")
        kids = node.get("children", [])
        child_prefix = prefix + (_TREE_SPACE if is_last else _TREE_CONT)
        for i, kid in enumerate(kids):
            emit(kid, child_prefix, i == len(kids) - 1)

    for root in forest:
        # Board roots render at depth 0 with no leading branch glyph.
        lines.append(fmt_label(root))
        kids = root.get("children", [])
        for j, kid in enumerate(kids):
            emit(kid, "", j == len(kids) - 1)

    return "\n".join(lines)


def render_json(forest: List[Dict[str, Any]]) -> str:
    """Return the forest as pretty-printed JSON (the node dicts natively nest)."""
    import json
    return json.dumps(forest, indent=2)


def _mermaid_label(text: Any) -> str:
    """Escape a string for use inside a Mermaid node label.

    Mermaid renders at ``securityLevel: strict`` in the desktop — a raw quote,
    bracket, or newline can break the diagram or inject node text, so every
    label is escaped to a single-line double-quoted form. Mirrors the exchange
    that protects the existing ```mermaid`` embed path.
    """
    s = str(text) if text is not None else ""
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("[", "\\[").replace("]", "\\]")
    return s.replace("\n", " ").replace("\r", " ").strip()


def render_mermaid(forest: List[Dict[str, Any]]) -> str:
    """Render a forest as a Mermaid ``flowchart TD`` graph.

    Node ids equal task ids so the graph is addressable and can be clicked /
    cross-referenced; labels are the task id + title. Descendant edges follow
    the parent/child nesting, so lineage is explicit in the diagram.
    """
    lines: List[str] = ["flowchart TD"]
    for node in forest:
        _emit_mermaid_node(lines, node)
    return "\n".join(lines)


def _emit_mermaid_node(lines: List[str], node: Dict[str, Any]) -> None:
    nid = str(node["id"])
    label = _mermaid_label(f"{nid}: {node['title']}")
    lines.append(f'  {nid}["{label}"]')
    for kid in node.get("children", []):
        lines.append(f"  {nid} --> {kid['id']}")
        _emit_mermaid_node(lines, kid)