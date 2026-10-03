"""Deterministic Agent OS primitives for future debt inventories.

These helpers extend the accepted prior-work-first baseline with focused DAG
checks, ready-frontier selection, and one-writer-safe scheduling.
"""

from __future__ import annotations

from collections import defaultdict, deque
from functools import lru_cache
from typing import Any, Mapping, Sequence

REQUIRED_NODE_FIELDS = (
    "CAPABILITY_ID",
    "STATE",
    "DEPS",
    "RISK",
    "WEIGHT",
    "FIRST_UNPROVEN_BOUNDARY",
    "BASE_HEAD",
    "BASE_TREE",
    "CHANGED_PATHS",
    "SEMANTIC_OWNER",
    "RUNTIME_IMPACT",
    "RELEASE_IMPACT",
    "BLOCK_LIVE",
    "MERGE_PRIORITY",
)
VALID_STATES = frozenset({"DONE", "READY", "RUNNING", "BLOCKED", "FUTURE", "INVALIDATED"})
ACTIVE_STATES = frozenset({"READY", "RUNNING", "BLOCKED"})
PARITY_EXACT = "EXACT"
PARITY_HEAD_MOVED_TREE_STABLE = "HEAD_MOVED_TREE_STABLE"
PARITY_TREE_CHANGED = "TREE_CHANGED"
PARITY_STALE_BASELINE = "STALE_BASELINE"
DEFAULT_PASS_CHECKS = (
    "DAG_VALIDATION",
    "READY_FRONTIER_PASS",
    "CRITICAL_PATH_PASS",
    "PARITY_CLASSIFICATION_PASS",
    "ONE_WRITER_PASS",
)


def normalize_node(node: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(node)
    normalized["CAPABILITY_ID"] = str(normalized.get("CAPABILITY_ID") or "").strip()
    normalized["STATE"] = str(normalized.get("STATE") or "").strip().upper()
    normalized["FIRST_UNPROVEN_BOUNDARY"] = str(
        normalized.get("FIRST_UNPROVEN_BOUNDARY") or ""
    ).strip()
    normalized["BASE_HEAD"] = str(normalized.get("BASE_HEAD") or "").strip()
    normalized["BASE_TREE"] = str(normalized.get("BASE_TREE") or "").strip()
    normalized["SEMANTIC_OWNER"] = str(normalized.get("SEMANTIC_OWNER") or "").strip()
    normalized["RUNTIME_IMPACT"] = str(normalized.get("RUNTIME_IMPACT") or "").strip()
    normalized["RELEASE_IMPACT"] = str(normalized.get("RELEASE_IMPACT") or "").strip()
    normalized["DEPS"] = [str(item).strip() for item in _coerce_list(normalized.get("DEPS"))]
    normalized["CHANGED_PATHS"] = [
        str(item).replace("\\", "/").strip() for item in _coerce_list(normalized.get("CHANGED_PATHS"))
    ]
    normalized["RISK"] = _coerce_number(normalized.get("RISK"))
    normalized["WEIGHT"] = _coerce_number(normalized.get("WEIGHT"))
    normalized["MERGE_PRIORITY"] = _coerce_number(normalized.get("MERGE_PRIORITY"))
    normalized["BLOCK_LIVE"] = bool(normalized.get("BLOCK_LIVE"))
    return normalized


def validate_inventory(nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_node(node) for node in nodes]
    errors: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []

    for index, node in enumerate(normalized):
        missing = [field for field in REQUIRED_NODE_FIELDS if _missing_value(node.get(field), field)]
        if missing:
            errors.append(
                f"node[{index}]::{node.get('CAPABILITY_ID') or '<missing>'} missing fields: {', '.join(missing)}"
            )
        capability_id = node["CAPABILITY_ID"]
        if capability_id in by_id:
            duplicates.append(capability_id)
        elif capability_id:
            by_id[capability_id] = node
        if node["STATE"] not in VALID_STATES:
            errors.append(
                f"node[{index}]::{capability_id or '<missing>'} invalid STATE={node['STATE']!r}"
            )

    if duplicates:
        errors.append("duplicate CAPABILITY_ID values: " + ", ".join(sorted(set(duplicates))))

    for node in normalized:
        missing_deps = [dep for dep in node["DEPS"] if dep and dep not in by_id]
        if missing_deps:
            errors.append(
                f"{node['CAPABILITY_ID']} references unknown DEPS: {', '.join(sorted(missing_deps))}"
            )

    cycle = _detect_cycle(by_id)
    if cycle:
        errors.append("dependency cycle detected: " + " -> ".join(cycle))

    writer = validate_one_writer(normalized)
    return {
        "DAG_VALIDATION": not errors,
        "ERRORS": errors,
        "NODE_COUNT": len(normalized),
        "EDGE_COUNT": sum(len(node["DEPS"]) for node in normalized),
        "ONE_WRITER_SEMANTICS": writer["ONE_WRITER_SEMANTICS"],
        "ONE_WRITER_VIOLATIONS": writer["VIOLATIONS"],
        "NODES": normalized,
    }


def classify_parity(base_head: str, base_tree: str, current_head: str, current_tree: str) -> str:
    base_head = str(base_head or "").strip()
    base_tree = str(base_tree or "").strip()
    current_head = str(current_head or "").strip()
    current_tree = str(current_tree or "").strip()
    if base_head == current_head and base_tree == current_tree:
        return PARITY_EXACT
    if base_tree and base_tree == current_tree:
        return PARITY_HEAD_MOVED_TREE_STABLE
    if base_head == current_head and base_tree and base_tree != current_tree:
        return PARITY_TREE_CHANGED
    return PARITY_STALE_BASELINE


def parity_classification(
    nodes: Sequence[Mapping[str, Any]], current_head: str, current_tree: str
) -> dict[str, str]:
    return {
        node["CAPABILITY_ID"]: classify_parity(
            node["BASE_HEAD"], node["BASE_TREE"], current_head, current_tree
        )
        for node in (normalize_node(item) for item in nodes)
        if node["CAPABILITY_ID"]
    }


def ready_frontier(nodes: Sequence[Mapping[str, Any]]) -> list[str]:
    normalized, by_id = _normalized_lookup(nodes)
    ready = [
        node
        for node in normalized
        if node["STATE"] == "READY" and _deps_satisfied(node, by_id)
    ]
    ready.sort(key=_priority_sort_key)
    return [node["CAPABILITY_ID"] for node in ready]


def parallel_safe_frontier(nodes: Sequence[Mapping[str, Any]]) -> list[str]:
    normalized, by_id = _normalized_lookup(nodes)
    running_owners = {
        node["SEMANTIC_OWNER"]
        for node in normalized
        if node["STATE"] == "RUNNING" and node["SEMANTIC_OWNER"]
    }
    selected: list[str] = []
    seen_owners: set[str] = set()
    for node in sorted(
        (node for node in normalized if node["STATE"] == "READY" and _deps_satisfied(node, by_id)),
        key=_priority_sort_key,
    ):
        owner = node["SEMANTIC_OWNER"] or node["CAPABILITY_ID"]
        if owner in running_owners or owner in seen_owners:
            continue
        seen_owners.add(owner)
        selected.append(node["CAPABILITY_ID"])
    return selected


def critical_path(nodes: Sequence[Mapping[str, Any]]) -> list[str]:
    normalized, by_id = _normalized_lookup(nodes)
    active = {
        node["CAPABILITY_ID"]: node
        for node in normalized
        if node["CAPABILITY_ID"] and node["STATE"] in ACTIVE_STATES
    }
    if not active:
        return []

    dependents: dict[str, list[str]] = defaultdict(list)
    active_deps: dict[str, list[str]] = {}
    for capability_id, node in active.items():
        deps = [dep for dep in node["DEPS"] if dep in active]
        active_deps[capability_id] = deps
        for dep in deps:
            dependents[dep].append(capability_id)

    roots = sorted([cap for cap, deps in active_deps.items() if not deps])
    if not roots:
        roots = sorted(active)

    @lru_cache(maxsize=None)
    def best_from(capability_id: str) -> tuple[str, ...]:
        children = sorted(dependents.get(capability_id, []))
        if not children:
            return (capability_id,)
        best = (capability_id,)
        best_score = _path_score([active[capability_id]])
        for child in children:
            candidate = (capability_id,) + best_from(child)
            score = _path_score([active[item] for item in candidate])
            if score > best_score:
                best = candidate
                best_score = score
        return best

    best_path = max((best_from(root) for root in roots), key=lambda path: _path_score([active[item] for item in path]))
    return list(best_path)


def validate_one_writer(nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_node(node) for node in nodes]
    violations: list[str] = []
    owners: dict[str, list[str]] = defaultdict(list)
    for node in normalized:
        if node["STATE"] != "RUNNING":
            continue
        owner = node["SEMANTIC_OWNER"] or node["CAPABILITY_ID"]
        if not owner:
            continue
        owners[owner].append(node["CAPABILITY_ID"])
    for owner, capability_ids in sorted(owners.items()):
        if len(capability_ids) > 1:
            violations.append(
                f"semantic owner {owner!r} has multiple RUNNING nodes: {', '.join(capability_ids)}"
            )
    return {
        "ONE_WRITER_SEMANTICS": not violations,
        "RUNNING_BY_OWNER": {owner: caps for owner, caps in sorted(owners.items())},
        "VIOLATIONS": violations,
    }


def inventory_report(
    nodes: Sequence[Mapping[str, Any]], current_head: str, current_tree: str
) -> dict[str, Any]:
    validation = validate_inventory(nodes)
    parity = parity_classification(validation["NODES"], current_head, current_tree)
    ready = ready_frontier(validation["NODES"]) if validation["DAG_VALIDATION"] else []
    critical = critical_path(validation["NODES"]) if validation["DAG_VALIDATION"] else []
    parallel = parallel_safe_frontier(validation["NODES"]) if validation["DAG_VALIDATION"] else []
    return {
        **validation,
        "PARITY_CLASSIFICATION": parity,
        "READY_FRONTIER": ready,
        "CRITICAL_PATH": critical,
        "PARALLEL_SAFE_FRONTIER": parallel,
        "READY_FRONTIER_PASS": bool(ready),
        "CRITICAL_PATH_PASS": bool(critical),
        "PARITY_CLASSIFICATION_PASS": bool(parity),
    }


def durable_pass_gate(
    checks: Mapping[str, Any], *, required: Sequence[str] = DEFAULT_PASS_CHECKS
) -> dict[str, Any]:
    failed = [name for name in required if not bool(checks.get(name))]
    return {
        "DURABLE_PASS_GATE": not failed,
        "FAILED_CHECKS": failed,
        "REQUIRED_CHECKS": list(required),
    }


def _normalized_lookup(
    nodes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    normalized = [normalize_node(node) for node in nodes]
    by_id = {node["CAPABILITY_ID"]: node for node in normalized if node["CAPABILITY_ID"]}
    return normalized, by_id


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _coerce_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _missing_value(value: Any, field: str) -> bool:
    if field in {"DEPS", "CHANGED_PATHS"}:
        return value is None
    if field == "BLOCK_LIVE":
        return value is None
    if field in {"RISK", "WEIGHT", "MERGE_PRIORITY"}:
        return value is None or value == ""
    return str(value or "").strip() == ""


def _deps_satisfied(node: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    for dep in node.get("DEPS", []):
        dep_node = by_id.get(dep)
        if dep_node is None or dep_node.get("STATE") != "DONE":
            return False
    return True


def _detect_cycle(by_id: Mapping[str, Mapping[str, Any]]) -> list[str]:
    indegree = {cap: 0 for cap in by_id}
    dependents: dict[str, list[str]] = defaultdict(list)
    for cap, node in by_id.items():
        for dep in node.get("DEPS", []):
            if dep not in by_id:
                continue
            indegree[cap] += 1
            dependents[dep].append(cap)
    queue = deque(sorted(cap for cap, degree in indegree.items() if degree == 0))
    visited: list[str] = []
    while queue:
        cap = queue.popleft()
        visited.append(cap)
        for child in sorted(dependents.get(cap, [])):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(visited) == len(by_id):
        return []
    remaining = sorted(cap for cap, degree in indegree.items() if degree > 0)
    return remaining


def _priority_sort_key(node: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(bool(node.get("BLOCK_LIVE"))),
        -float(node.get("MERGE_PRIORITY", 0.0)),
        -float(node.get("WEIGHT", 0.0)),
        -float(node.get("RISK", 0.0)),
        str(node.get("CAPABILITY_ID") or ""),
    )


def _path_score(path: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    return (
        int(any(bool(node.get("BLOCK_LIVE")) for node in path)),
        sum(float(node.get("MERGE_PRIORITY", 0.0)) for node in path),
        sum(float(node.get("WEIGHT", 0.0)) for node in path),
        sum(float(node.get("RISK", 0.0)) for node in path),
        len(path),
        tuple(str(node.get("CAPABILITY_ID") or "") for node in path),
    )
