"""Parser for Roadmaps plan documents (T5c, Vision session).

``parse_plan`` converts the Vision agent's plan document into a validated,
normalized payload ready for ``RoadmapsWriter.create_plan``.

Two input forms are accepted:

- **Strict JSON** (REQUIRED of the agent): the document is the JSON object
  ``{"title", "purpose"?, "nodes": [...], "relations": [...], "todos": [...]}``
  exactly as the planning rules specify, either as a pure JSON text or inside
  a ```json ... ``` code fence.  This is the reliable path.
- **Structured Markdown** (documented fallback, LESS reliable): the agent
  MUST emit JSON; Markdown is tolerated only so a rough draft can be parsed.
  Grammar: ``# Title`` plan title; ``## <kind>: <title>`` nodes (kind
  objective|phase|milestone|step|decision — otherwise heading level maps
  2→phase, 3→milestone, 4→step, 5→decision), parents from heading nesting;
  ``- [ ]`` / ``- [x]`` todos attached to the most recent node;
  ``- <relation_kind>: <From> -> <To>`` relations resolved by unique node
  title.

Validation (both forms): node ids non-empty and unique, kinds valid, titles
non-empty, parents referenced / never self / acyclic; relations from/to
referenced and distinct, kinds valid; todos reference an existing node or
null, positions integers >= 0.

Normalization: ``state``/``progress`` defaults (planned/0), relation state
defaults to ``active``, todo state ``open`` + position 0; provided ids are
kept as-is; MISSING ids are generated deterministically with the ``n_`` /
``r_`` / ``t_`` prefixes; identical relations (same from/to/kind) and
identical todos (same node/title) are deduplicated keeping the first.

Hardening limits (DoS gate, 2026-08-15): the module constants
``MAX_PLAN_TEXT_BYTES`` (2 MB) caps the raw document size and
``MAX_PLAN_NODES`` / ``MAX_PLAN_RELATIONS`` / ``MAX_PLAN_TODOS`` (2000
each) cap the element counts — exceeded inputs raise
:class:`PlanParseError` with ``field="input"``.  ``MAX_JSON_NESTING_DEPTH``
(512) gates ``json.loads`` with a linear, string-aware depth pre-scan so
pathologically nested documents raise a clean ``PlanParseError`` instead of
a raw ``RecursionError`` (CPython's C scanner overflows ~1000 levels deep);
an ``except RecursionError`` around ``json.loads`` remains as a safety net.
Cycle detection over parent references is an iterative 3-color DFS (explicit
stack), so a 3000-node parent chain cannot overflow the Python stack, and
reports ``field="parent_node_id"`` with the node's list index.

Errors raise :class:`PlanParseError` with a structured message: the
concerned field (``nodes[2].kind``), the list index, and — for JSON syntax
errors — line/column in the original text.
"""

from __future__ import annotations

import json
import re
from typing import Any

from hermes_cli.roadmaps_writer import (
    NODE_KINDS,
    NODE_STATES,
    RELATION_KINDS,
    TODO_STATES,
)

NODE_KINDS_VALID = NODE_KINDS
RELATION_KINDS_VALID = RELATION_KINDS
NODE_STATES_VALID = NODE_STATES
TODO_STATES_VALID = TODO_STATES

_ID_PREFIXES = {"nodes": "n_", "relations": "r_", "todos": "t_"}

# ── DoS hardening limits (gate request_changes, 2026-08-15) ──────────────────
# Module-level so tests and callers can tune them; violations raise
# PlanParseError(field="input") with a clear message.
MAX_PLAN_TEXT_BYTES = 2_000_000  # raw document size cap (2 MB)
MAX_PLAN_NODES = 2000  # node cap
MAX_PLAN_RELATIONS = 2000  # relation cap (proportional to nodes)
MAX_PLAN_TODOS = 2000  # todo cap (proportional to nodes)
# JSON nesting gate: a linear, string-aware pre-scan rejects documents nested
# deeper than this BEFORE json.loads (CPython's C scanner raises a raw
# RecursionError ~1000 levels deep); the except around json.loads stays as a
# safety net.  Legitimate plan documents nest ~10 levels at most.
MAX_JSON_NESTING_DEPTH = 512

_FENCE_RE = re.compile(r"```\s*json\s*\n(.*?)```", re.DOTALL)
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_MARKDOWN_KIND_RE = re.compile(
    r"^(objective|phase|milestone|step|decision)\s*:\s*(.+?)\s*$", re.IGNORECASE
)
_MARKDOWN_TODO_RE = re.compile(r"^[-*]\s+\[( |x|X)\]\s+(.+?)\s*$")
_MARKDOWN_RELATION_RE = re.compile(
    r"^[-*]\s+(" + "|".join(sorted(RELATION_KINDS_VALID)) + r")\s*:\s*(.+?)\s*->\s*(.+?)\s*$"
)

_HEADING_LEVEL_KIND = {2: "phase", 3: "milestone", 4: "step", 5: "decision"}


class PlanParseError(ValueError):
    """Structured plan-parsing failure the Vision agent can act on.

    Attributes mirror the payload position: ``field`` (e.g. ``nodes[2].kind``),
    ``index`` (list position, when inside a list), and ``line``/``column``
    (source position, populated for JSON syntax errors).
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        index: int | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.index = index
        self.line = line
        self.column = column

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.field:
            parts.append(f"field: {self.field}")
        if self.line is not None:
            loc = f"line {self.line}"
            if self.column is not None:
                loc += f", column {self.column}"
            parts.append(loc)
        return " — ".join(parts)


def _generated_id(prefix: str, used: set[str], counter: list[int]) -> str:
    """Next deterministic ``prefix`` + zero-padded counter id, collision-free."""
    while True:
        counter[0] += 1
        candidate = f"{prefix}{counter[0]:04d}"
        if candidate not in used:
            return candidate


def _require(value: Any, field: str, *, index: int | None = None) -> Any:
    if value is None:
        raise PlanParseError("required field missing", field=field, index=index)
    if isinstance(value, str) and not value.strip():
        raise PlanParseError("must be a non-empty string", field=field, index=index)
    return value


def _validate_ids(items: list[dict[str, Any]], kind: str) -> None:
    seen: set[str] = set()
    for i, item in enumerate(items):
        raw = item.get(f"{kind}_id")
        if isinstance(raw, str) and not raw.strip():
            raise PlanParseError(
                f"{kind}_id must be non-empty", field=f"{kind}s[{i}].{kind}_id", index=i
            )
        if raw is None:
            continue
        if raw in seen:
            raise PlanParseError(
                f"duplicate {kind}_id {raw!r}", field=f"{kind}s[{i}].{kind}_id", index=i
            )
        seen.add(raw)


def _normalize_nodes(nodes: Any) -> list[dict[str, Any]]:
    if not isinstance(nodes, list):
        raise PlanParseError("nodes must be a list", field="nodes")
    normalized: list[dict[str, Any]] = []
    used: set[str] = set()
    counter: list[int] = [0]
    for i, item in enumerate(nodes):
        field = f"nodes[{i}]"
        if not isinstance(item, dict):
            raise PlanParseError("node must be an object", field=field, index=i)
        node_id = item.get("node_id")
        if isinstance(node_id, str) and not node_id.strip():
            raise PlanParseError(
                "node_id must be non-empty", field=f"{field}.node_id", index=i
            )
        if node_id is None:
            node_id = _generated_id("n_", used, counter)
        if node_id in used:
            raise PlanParseError(
                f"duplicate node_id {node_id!r}", field=f"{field}.node_id", index=i
            )
        used.add(node_id)
        kind = item.get("kind")
        if kind not in NODE_KINDS_VALID:
            raise PlanParseError(
                f"invalid kind {kind!r} (expected one of {sorted(NODE_KINDS_VALID)})",
                field=f"{field}.kind", index=i,
            )
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            raise PlanParseError("title must be a non-empty string", field=f"{field}.title", index=i)
        state = item.get("state", "planned")
        if state not in NODE_STATES_VALID:
            raise PlanParseError(
                f"invalid state {state!r} (expected one of {sorted(NODE_STATES_VALID)})",
                field=f"{field}.state", index=i,
            )
        progress = item.get("progress", 0)
        if isinstance(progress, bool) or not isinstance(progress, int) or not 0 <= progress <= 100:
            raise PlanParseError(
                "progress must be an integer between 0 and 100",
                field=f"{field}.progress", index=i,
            )
        parent = item.get("parent_node_id")
        if isinstance(parent, str) and not parent.strip():
            raise PlanParseError(
                "parent_node_id must be non-empty", field=f"{field}.parent_node_id", index=i
            )
        normalized.append({
            "node_id": node_id,
            "kind": kind,
            "title": title.strip(),
            "description": item.get("description"),
            "parent_node_id": parent,
            "state": state,
            "progress": progress,
            "owner_agent": item.get("owner_agent"),
            "block_reason": item.get("block_reason"),
        })
    # Parent references: must exist, never self, acyclic.
    by_id = {n["node_id"]: n for n in normalized}
    for i, node in enumerate(normalized):
        parent = node["parent_node_id"]
        if parent is None:
            continue
        if parent == node["node_id"]:
            raise PlanParseError(
                "parent_node_id cannot reference the node itself",
                field=f"nodes[{i}].parent_node_id", index=i,
            )
        if parent not in by_id:
            raise PlanParseError(
                f"parent_node_id {parent!r} does not reference a node of this plan",
                field=f"nodes[{i}].parent_node_id", index=i,
            )
    _detect_cycle(
        {n["node_id"]: n["parent_node_id"] for n in normalized},
        "parent",
        index_of={n["node_id"]: i for i, n in enumerate(normalized)},
    )
    return normalized


def _normalize_relations(relations: Any, node_ids: set[str]) -> list[dict[str, Any]]:
    if relations is None:
        return []
    if not isinstance(relations, list):
        raise PlanParseError("relations must be a list", field="relations")
    normalized: list[dict[str, Any]] = []
    used: set[str] = set()
    counter: list[int] = [0]
    seen_content: set[tuple[Any, Any, Any]] = set()
    for i, item in enumerate(relations):
        field = f"relations[{i}]"
        if not isinstance(item, dict):
            raise PlanParseError("relation must be an object", field=field, index=i)
        relation_id = item.get("relation_id")
        if isinstance(relation_id, str) and not relation_id.strip():
            raise PlanParseError(
                "relation_id must be non-empty", field=f"{field}.relation_id", index=i
            )
        if relation_id is None:
            relation_id = _generated_id("r_", used, counter)
        if relation_id in used:
            raise PlanParseError(
                f"duplicate relation_id {relation_id!r}",
                field=f"{field}.relation_id", index=i,
            )
        used.add(relation_id)
        from_node = item.get("from_node_id")
        to_node = item.get("to_node_id")
        if from_node not in node_ids:
            raise PlanParseError(
                f"from_node_id {from_node!r} does not reference a node of this plan",
                field=f"{field}.from_node_id", index=i,
            )
        if to_node not in node_ids:
            raise PlanParseError(
                f"to_node_id {to_node!r} does not reference a node of this plan",
                field=f"{field}.to_node_id", index=i,
            )
        if from_node == to_node:
            raise PlanParseError(
                "from_node_id and to_node_id must differ",
                field=f"{field}.to_node_id", index=i,
            )
        kind = item.get("kind")
        if kind not in RELATION_KINDS_VALID:
            raise PlanParseError(
                f"invalid kind {kind!r} (expected one of {sorted(RELATION_KINDS_VALID)})",
                field=f"{field}.kind", index=i,
            )
        content = (from_node, to_node, kind)
        if content in seen_content:
            continue  # identical relation already declared — dedupe, keep first
        seen_content.add(content)
        normalized.append({
            "relation_id": relation_id,
            "from_node_id": from_node,
            "to_node_id": to_node,
            "kind": kind,
            "state": item.get("state", "active"),
            "reason": item.get("reason"),
        })
    return normalized


def _normalize_todos(todos: Any, node_ids: set[str]) -> list[dict[str, Any]]:
    if todos is None:
        return []
    if not isinstance(todos, list):
        raise PlanParseError("todos must be a list", field="todos")
    normalized: list[dict[str, Any]] = []
    used: set[str] = set()
    counter: list[int] = [0]
    seen_content: set[tuple[str | None, str]] = set()
    for i, item in enumerate(todos):
        field = f"todos[{i}]"
        if not isinstance(item, dict):
            raise PlanParseError("todo must be an object", field=field, index=i)
        todo_id = item.get("todo_id")
        if isinstance(todo_id, str) and not todo_id.strip():
            raise PlanParseError(
                "todo_id must be non-empty", field=f"{field}.todo_id", index=i
            )
        if todo_id is None:
            todo_id = _generated_id("t_", used, counter)
        if todo_id in used:
            raise PlanParseError(
                f"duplicate todo_id {todo_id!r}", field=f"{field}.todo_id", index=i
            )
        used.add(todo_id)
        node_id = item.get("node_id")
        if node_id is not None and node_id not in node_ids:
            raise PlanParseError(
                f"node_id {node_id!r} does not reference a node of this plan",
                field=f"{field}.node_id", index=i,
            )
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            raise PlanParseError("title must be a non-empty string", field=f"{field}.title", index=i)
        position = item.get("position", 0)
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise PlanParseError(
                "position must be an integer >= 0", field=f"{field}.position", index=i
            )
        content = (node_id, title.strip())
        if content in seen_content:
            continue  # identical todo already declared — dedupe, keep first
        seen_content.add(content)
        normalized.append({
            "todo_id": todo_id,
            "node_id": node_id,
            "title": title.strip(),
            "state": item.get("state", "open"),
            "position": position,
        })
    return normalized


def _detect_cycle(
    parent_of: dict[str, str | None],
    label: str,
    index_of: dict[str, int] | None = None,
) -> None:
    """Raise on any cycle in the reference graph (parents or relations).

    Iterative 3-color DFS with an explicit stack: a 3000-node parent chain
    (or any input within the node cap) cannot overflow the Python stack,
    unlike the previous recursive ``visit``.  On a cycle, raises a structured
    :class:`PlanParseError` with ``field="parent_node_id"`` and the list
    index of the node where the cycle closes (when ``index_of`` is given).
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {node: WHITE for node in parent_of}
    for start in parent_of:
        if color[start] != WHITE:
            continue
        stack: list[tuple[str, bool]] = [(start, False)]  # (node, expanded)
        while stack:
            node, expanded = stack.pop()
            if expanded:
                color[node] = BLACK
                continue
            color[node] = GRAY
            stack.append((node, True))
            parent = parent_of.get(node)
            if parent is None:
                continue
            pcolor = color.get(parent, WHITE)
            if pcolor == GRAY:
                raise PlanParseError(
                    f"cycle detected in {label} references at {node!r}",
                    field="parent_node_id",
                    index=index_of.get(node) if index_of else None,
                )
            if pcolor == WHITE:
                stack.append((parent, False))


# ── document extraction ─────────────────────────────────────────────────────


def _max_json_nesting(text: str) -> int:
    """Max structural nesting depth of ``{}`` / ``[]`` in a JSON document.

    Linear, string-aware scan (no parsing): braces inside JSON strings are
    skipped.  Used as a cheap gate before ``json.loads`` so deeply nested
    documents raise a structured error instead of a raw ``RecursionError``.
    """
    depth = 0
    max_depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch in "}]":
            depth -= 1
    return max_depth


def _extract_json_document(text: str) -> tuple[dict[str, Any], int | None, int | None]:
    """Return (document, line, column) for a strict-JSON document.

    A ```json fence wins over surrounding prose; otherwise the whole text is
    parsed as JSON.  Raises :class:`PlanParseError` on syntax errors with the
    source position of the failure, and on pathologically nested documents
    (``field="input"``, before ``json.loads`` can overflow its stack).
    """
    fence = _FENCE_RE.search(text)
    candidate = fence.group(1) if fence else text.strip()
    base_line = text[: fence.start(1)].count("\n") + 1 if fence else 1
    if _max_json_nesting(candidate) > MAX_JSON_NESTING_DEPTH:
        raise PlanParseError(
            "plan document is nested too deeply",
            field="input",
            line=base_line,
        )
    try:
        parsed = json.loads(candidate)
    except RecursionError:
        # Safety net: the depth pre-scan above should fire first, but keep
        # the contract "PlanParseError only" even if the C scanner trips.
        raise PlanParseError(
            "plan document is nested too deeply",
            field="input",
            line=base_line,
        ) from None
    except json.JSONDecodeError as exc:
        raise PlanParseError(
            f"invalid JSON: {exc.msg}",
            line=base_line + exc.lineno - 1,
            column=exc.colno,
        ) from None
    if not isinstance(parsed, dict):
        raise PlanParseError(
            "plan document must be a JSON object", line=base_line, column=1
        )
    return parsed, base_line, 1


def _parse_markdown(text: str) -> dict[str, Any]:
    """Minimal structured-Markdown fallback (documented, less reliable)."""
    title: str | None = None
    nodes: list[dict[str, Any]] = []
    todos: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    by_title: dict[str, str] = {}
    stack: list[tuple[int, str]] = []  # (level, node_id)
    # Persistent id bookkeeping: the old code rebuilt ``set(by_title.values())``
    # (and reset the counter) for every node, an O(n^2) total.  With persistent
    # sets + counters each generated id is O(1) amortized.
    used_node_ids: set[str] = set()
    node_counter: list[int] = [0]
    used_todo_ids: set[str] = set()
    todo_counter: list[int] = [0]
    used_relation_ids: set[str] = set()
    relation_counter: list[int] = [0]

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = _MARKDOWN_HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            rest = heading.group(2).strip()
            kind_match = _MARKDOWN_KIND_RE.match(rest)
            if kind_match:
                kind = kind_match.group(1).lower()
                node_title = kind_match.group(2).strip()
            else:
                kind = _HEADING_LEVEL_KIND.get(level)
                node_title = rest
            if level == 1 and kind is None:
                title = rest
                continue
            if kind is None:
                continue  # unknown heading level, ignore
            if not node_title:
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1] if stack else None
            node_id = _generated_id("n_", used_node_ids, node_counter)
            nodes.append({
                "node_id": node_id, "kind": kind, "title": node_title,
                "description": None, "parent_node_id": parent,
                "state": "planned", "progress": 0,
                "owner_agent": None, "block_reason": None,
            })
            used_node_ids.add(node_id)
            by_title[node_title] = node_id
            stack.append((level, node_id))
            if len(nodes) > MAX_PLAN_NODES:
                raise PlanParseError(
                    f"plan exceeds the limit of {MAX_PLAN_NODES} nodes",
                    field="input",
                )
            continue
        todo = _MARKDOWN_TODO_RE.match(line)
        if todo:
            done = todo.group(1).lower() == "x"
            node_id = stack[-1][1] if stack else None
            todos.append({
                "todo_id": _generated_id("t_", used_todo_ids, todo_counter),
                "node_id": node_id,
                "title": todo.group(2).strip(),
                "state": "done" if done else "open",
                "position": len(todos),
            })
            used_todo_ids.add(todos[-1]["todo_id"])
            if len(todos) > MAX_PLAN_TODOS:
                raise PlanParseError(
                    f"plan exceeds the limit of {MAX_PLAN_TODOS} todos",
                    field="input",
                )
            continue
        relation = _MARKDOWN_RELATION_RE.match(line)
        if relation:
            kind = relation.group(1).lower()
            from_title = relation.group(2).strip()
            to_title = relation.group(3).strip()
            from_id = by_title.get(from_title)
            to_id = by_title.get(to_title)
            if from_id is None:
                raise PlanParseError(
                    f"relation target {from_title!r} does not match any node title",
                    field="relations", index=len(relations),
                )
            if to_id is None:
                raise PlanParseError(
                    f"relation target {to_title!r} does not match any node title",
                    field="relations", index=len(relations),
                )
            relations.append({
                "relation_id": _generated_id("r_", used_relation_ids, relation_counter),
                "from_node_id": from_id, "to_node_id": to_id, "kind": kind,
                "state": "active", "reason": None,
            })
            used_relation_ids.add(relations[-1]["relation_id"])
            if len(relations) > MAX_PLAN_RELATIONS:
                raise PlanParseError(
                    f"plan exceeds the limit of {MAX_PLAN_RELATIONS} relations",
                    field="input",
                )
    return {"title": title, "nodes": nodes, "relations": relations, "todos": todos}


def parse_plan(
    text: str,
    *,
    source: str = "vision",
    default_actor: str = "user",
) -> dict[str, Any]:
    """Parse a plan document into a payload ready for ``plans.create``.

    Accepts strict JSON (pure or in a ```json fence) or the documented
    Markdown fallback.  Validates, normalizes defaults, generates missing
    ids (``n_``/``r_``/``t_``), dedupes identical relations/todos, and
    returns ``{"title", "purpose"?, "source", "actor", "nodes",
    "relations", "todos"}`` where ``nodes``/``relations``/``todos`` match the
    exact shape ``RoadmapsWriter.create_plan`` validates.
    """
    if not isinstance(text, str) or not text.strip():
        raise PlanParseError("plan text is required", field="text")
    size = len(text.encode("utf-8"))
    if size > MAX_PLAN_TEXT_BYTES:
        raise PlanParseError(
            f"plan document too large: {size} bytes exceeds the limit "
            f"of {MAX_PLAN_TEXT_BYTES} bytes",
            field="input",
        )
    stripped = text.strip()
    has_fence = _FENCE_RE.search(text) is not None
    looks_like_json = stripped.startswith("{") or stripped.startswith("[")

    document: dict[str, Any]
    if has_fence or looks_like_json:
        document, _line, _col = _extract_json_document(text)
    else:
        document = _parse_markdown(text)
    for key, cap in (
        ("nodes", MAX_PLAN_NODES),
        ("relations", MAX_PLAN_RELATIONS),
        ("todos", MAX_PLAN_TODOS),
    ):
        items = document.get(key)
        if isinstance(items, list) and len(items) > cap:
            raise PlanParseError(
                f"plan declares {len(items)} {key}, exceeding the limit of {cap}",
                field="input",
            )

    title = document.get("title")
    if not isinstance(title, str) or not title.strip():
        raise PlanParseError(
            "plan document must declare a non-empty title", field="title"
        )

    nodes = _normalize_nodes(document.get("nodes"))
    if not nodes:
        raise PlanParseError(
            "no plan nodes found; expected strict JSON (```json ... ```) "
            "or structured markdown with at least one node",
            field="nodes",
        )
    node_ids = {n["node_id"] for n in nodes}
    relations = _normalize_relations(document.get("relations"), node_ids)
    todos = _normalize_todos(document.get("todos"), node_ids)

    purpose = document.get("purpose")
    if purpose is not None and (not isinstance(purpose, str) or not purpose.strip()):
        raise PlanParseError("purpose must be a non-empty string", field="purpose")

    return {
        "title": title.strip(),
        "purpose": purpose.strip() if isinstance(purpose, str) else None,
        "source": source,
        "actor": default_actor,
        "nodes": nodes,
        "relations": relations,
        "todos": todos,
    }
