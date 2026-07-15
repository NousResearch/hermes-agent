"""Deterministic Ship's Crew risk classification and graph compilation.

This module owns the model-free part of Quest planning. It validates structured
risk inputs, chooses the smallest lawful governance graph, and can persist that
graph atomically to a Kanban board. Ambiguous or malformed inputs fail closed;
no model or provider is consulted.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Mapping

from hermes_cli import kanban_db as kb


RISK_DIMENSIONS = (
    "blast_radius",
    "reversibility",
    "authority",
    "data_security",
    "uncertainty",
    "operational_cost",
)
GOVERNANCE_RANK = {"lite": 0, "standard": 1, "constitutional": 2}
HARD_TRIGGERS = frozenset(
    {
        "credential-access",
        "security-policy-change",
        "external-publication",
        "external-side-effect",
        "irreversible-write",
        "spending-commitment",
        "sensitive-data",
        "broad-filesystem-scope",
        "broad-network-scope",
    }
)
RULE_VERSION = "ship-crew-risk/v1"


class GraphCompilationError(ValueError):
    """Raised before or during graph persistence; failed compiles leave no rows."""


@dataclass(frozen=True)
class RiskClassification:
    governance_class: str
    score: int
    hard_triggers: tuple[str, ...]
    reviewer: str | None
    rationale_codes: tuple[str, ...]
    rule_version: str = RULE_VERSION


@dataclass(frozen=True)
class GraphNode:
    key: str
    title: str
    role: str
    contract_type: str
    parent_keys: tuple[str, ...] = ()
    user_gate: bool = False


@dataclass(frozen=True)
class CompiledGraph:
    quest_id: str
    capsule_sha256: str
    classification: RiskClassification
    graph_template: str
    nodes: tuple[GraphNode, ...]
    task_ids: Mapping[str, str] | None = None


_GRAPH_TEMPLATES: dict[str, tuple[GraphNode, ...]] = {
    "lite": (
        GraphNode("owner", "Quest owner", "captain", "captain-disposition/v1"),
        GraphNode("verification", "Deterministic verification", "engineer", "engineer-delivery/v1", ("owner",)),
        GraphNode("closeout", "Mission closeout", "captain", "captain-disposition/v1", ("verification",)),
    ),
    "standard": (
        GraphNode("proposal", "Scoped proposal", "engineer", "engineer-delivery/v1"),
        GraphNode("review", "Independent risk review", "pirate", "pirate-review/v1", ("proposal",)),
        GraphNode("decision", "Captain decision", "captain", "captain-disposition/v1", ("review",)),
        GraphNode("execution", "Substantive execution", "engineer", "engineer-delivery/v1", ("decision",)),
        GraphNode("verification", "Deterministic verification", "engineer", "engineer-delivery/v1", ("execution",)),
        GraphNode("closeout", "Mission closeout", "captain", "captain-disposition/v1", ("verification",)),
    ),
    "constitutional": (
        GraphNode("council", "Full Council review", "navigator", "navigator-evidence/v1"),
        GraphNode("pirate_review", "Independent Pirate review", "pirate", "pirate-review/v1", ("council",)),
        GraphNode("decision", "Captain decision", "captain", "captain-disposition/v1", ("pirate_review",)),
        GraphNode("user_sail", "User sail gate", "user", "captain-disposition/v1", ("decision",), True),
        GraphNode("execution", "Substantive execution", "engineer", "engineer-delivery/v1", ("user_sail",)),
        GraphNode("verification", "Deterministic verification", "engineer", "engineer-delivery/v1", ("execution",)),
        GraphNode("closeout", "Mission closeout", "captain", "captain-disposition/v1", ("verification",)),
    ),
}

_REVIEWER_BY_RISK = {
    "blast_radius": "pirate",
    "reversibility": "engineer",
    "authority": "navigator",
    "data_security": "pirate",
    "uncertainty": "navigator",
    "operational_cost": "engineer",
}


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GraphCompilationError(f"capsule is not canonical JSON: {exc}") from exc


def capsule_sha256(capsule: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(capsule)).hexdigest()


def validate_quest_capsule(capsule: Mapping[str, Any]) -> None:
    if not isinstance(capsule, Mapping):
        raise GraphCompilationError("Quest Capsule must be an object")
    required = {"schema_version", "quest_id", "risk_inputs", "scope", "acceptance_evidence"}
    missing = sorted(required - set(capsule))
    if missing:
        raise GraphCompilationError(f"capsule missing required fields: {', '.join(missing)}")
    if capsule["schema_version"] != "quest-capsule/v1":
        raise GraphCompilationError("capsule schema_version must be quest-capsule/v1")
    quest_id = capsule["quest_id"]
    if not isinstance(quest_id, str) or not quest_id.startswith("Q-") or not quest_id[2:]:
        raise GraphCompilationError("capsule quest_id must be a non-empty Q-* identifier")
    inputs = capsule["risk_inputs"]
    if not isinstance(inputs, Mapping):
        raise GraphCompilationError("risk_inputs must be an object")
    unknown = sorted(set(inputs) - set(RISK_DIMENSIONS) - {"hard_triggers"})
    if unknown:
        raise GraphCompilationError(f"risk_inputs has unknown fields: {', '.join(unknown)}")
    for name in RISK_DIMENSIONS:
        value = inputs.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 3:
            raise GraphCompilationError(f"risk_inputs.{name} must be an integer from 0 through 3")
    triggers = inputs.get("hard_triggers")
    if not isinstance(triggers, list) or len(set(triggers)) != len(triggers):
        raise GraphCompilationError("risk_inputs.hard_triggers must be a unique array")
    unknown_triggers = sorted(set(triggers) - HARD_TRIGGERS)
    if unknown_triggers:
        raise GraphCompilationError(f"unknown hard trigger(s): {', '.join(unknown_triggers)}")
    scope = capsule["scope"]
    if not isinstance(scope, Mapping):
        raise GraphCompilationError("scope must be an object")
    if not isinstance(scope.get("external_effect"), bool):
        raise GraphCompilationError("scope.external_effect must be boolean")
    if scope.get("write_scope") not in {"read-only", "scoped-internal", "broad"}:
        raise GraphCompilationError("scope.write_scope is invalid")
    if scope.get("data_class") not in {"ordinary", "sensitive"}:
        raise GraphCompilationError("scope.data_class is invalid")
    if (scope["external_effect"] or scope["write_scope"] == "broad" or scope["data_class"] == "sensitive") and not triggers:
        raise GraphCompilationError("scope risk must be represented by an explicit hard trigger")
    evidence = capsule["acceptance_evidence"]
    if not isinstance(evidence, list) or not evidence or any(not isinstance(item, str) or not item.strip() for item in evidence):
        raise GraphCompilationError("acceptance_evidence must be a non-empty list of strings")


def classify_risk(capsule: Mapping[str, Any]) -> RiskClassification:
    validate_quest_capsule(capsule)
    inputs = capsule["risk_inputs"]
    triggers = tuple(sorted(inputs["hard_triggers"]))
    score = sum(int(inputs[name]) for name in RISK_DIMENSIONS)
    rationale: list[str] = []
    if triggers:
        governance = "constitutional"
        rationale.append("hard-trigger")
    elif score >= 10:
        governance = "constitutional"
        rationale.append("score-critical")
    elif score >= 5 or capsule["scope"]["write_scope"] == "scoped-internal":
        governance = "standard"
        rationale.append("score-standard" if score >= 5 else "scoped-write")
    else:
        governance = "lite"
        rationale.append("score-lite")
    reviewer = None
    if governance == "standard":
        # RISK_DIMENSIONS order is the deterministic tie-break order.
        dominant = max(RISK_DIMENSIONS, key=lambda name: (int(inputs[name]), -RISK_DIMENSIONS.index(name)))
        reviewer = _REVIEWER_BY_RISK[dominant]
        rationale.append(f"reviewer-{dominant}")
    if governance == "constitutional":
        rationale.append("full-council")
    return RiskClassification(governance, score, triggers, reviewer, tuple(rationale))


def graph_for_classification(classification: RiskClassification) -> tuple[GraphNode, ...]:
    try:
        return _GRAPH_TEMPLATES[classification.governance_class]
    except KeyError as exc:
        raise GraphCompilationError(f"unknown governance class {classification.governance_class!r}") from exc


def compile_graph_plan(capsule: Mapping[str, Any], *, captain_classification: str | None = None) -> CompiledGraph:
    classification = classify_risk(capsule)
    selected = captain_classification or classification.governance_class
    if selected not in GOVERNANCE_RANK:
        raise GraphCompilationError(f"unknown governance class {selected!r}")
    if GOVERNANCE_RANK[selected] < GOVERNANCE_RANK[classification.governance_class]:
        raise GraphCompilationError("Captain cannot lower the deterministic risk classification")
    if selected != classification.governance_class:
        classification = RiskClassification(selected, classification.score, classification.hard_triggers, classification.reviewer, classification.rationale_codes + ("captain-promoted",))
    return CompiledGraph(
        quest_id=str(capsule["quest_id"]),
        capsule_sha256=capsule_sha256(capsule),
        classification=classification,
        graph_template=f"{selected}-v1",
        nodes=graph_for_classification(classification),
    )


def _node_idempotency(prefix: str, node: GraphNode) -> str:
    return f"ship-crew:{prefix}:{node.key}"


def _existing_graph_rows(conn: sqlite3.Connection, keys: list[str]) -> dict[str, str]:
    if not keys:
        return {}
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"SELECT id, idempotency_key FROM tasks WHERE idempotency_key IN ({placeholders}) AND status != 'archived'",
        keys,
    ).fetchall()
    return {str(row["idempotency_key"]): str(row["id"]) for row in rows}


def compile_graph(
    conn: sqlite3.Connection,
    capsule: Mapping[str, Any],
    *,
    captain_classification: str | None = None,
    created_by: str = "captain",
) -> CompiledGraph:
    """Compile and persist one graph atomically, or return its existing graph."""
    plan = compile_graph_plan(capsule, captain_classification=captain_classification)
    prefix = f"{plan.quest_id}:{plan.capsule_sha256[:16]}:{plan.graph_template}"
    keys = [_node_idempotency(prefix, node) for node in plan.nodes]
    existing = _existing_graph_rows(conn, keys)
    if existing:
        if len(existing) != len(keys):
            raise GraphCompilationError("partial graph exists for idempotency prefix; refusing to duplicate or repair silently")
        return CompiledGraph(plan.quest_id, plan.capsule_sha256, plan.classification, plan.graph_template, plan.nodes, {node.key: existing[_node_idempotency(prefix, node)] for node in plan.nodes})

    task_ids: dict[str, str] = {}
    with kb.write_txn(conn):
        # Re-check under the write lock so two compilers cannot both create the graph.
        if _existing_graph_rows(conn, keys):
            raise GraphCompilationError("graph appeared concurrently; retry the idempotent compile")
        now = int(__import__("time").time())
        for index, node in enumerate(plan.nodes):
            task_id = kb._new_task_id()
            parent_ids = [task_ids[parent] for parent in node.parent_keys]
            if node.user_gate:
                status = "blocked"
                block_kind = "needs_input"
            elif parent_ids:
                status = "todo"
                block_kind = None
            else:
                status = "ready"
                block_kind = None
            task_ids[node.key] = task_id
            body = json.dumps(
                {
                    "schema_version": "crew-task/v1",
                    "mission_id": plan.quest_id,
                    "graph_template": plan.graph_template,
                    "node_key": node.key,
                    "contract_type": node.contract_type,
                    "governance_class": plan.classification.governance_class,
                    "rule_version": RULE_VERSION,
                },
                sort_keys=True,
            )
            conn.execute(
                """INSERT INTO tasks (
                    id, title, body, assignee, status, block_kind, priority,
                    created_by, created_at, workspace_kind, workspace_path,
                    branch_name, project_id, tenant, idempotency_key,
                    max_runtime_seconds, skills, max_retries, goal_mode,
                    goal_max_turns, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, 'scratch', NULL, NULL, NULL, ?, ?, NULL, NULL, NULL, 0, NULL, NULL)""",
                (
                    task_id,
                    f"{plan.quest_id} — {node.title}",
                    body,
                    node.role,
                    status,
                    block_kind,
                    created_by,
                    now,
                    plan.quest_id,
                    _node_idempotency(prefix, node),
                ),
            )
            for parent_id in parent_ids:
                conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (parent_id, task_id))
            kb._append_event(
                conn,
                task_id,
                "created",
                {
                    "source": "ship-crew-graph-compiler",
                    "mission_id": plan.quest_id,
                    "graph_template": plan.graph_template,
                    "node_key": node.key,
                    "parents": parent_ids,
                    "contract_type": node.contract_type,
                    "user_gate": node.user_gate,
                    "classification": {
                        "governance_class": plan.classification.governance_class,
                        "score": plan.classification.score,
                        "hard_triggers": list(plan.classification.hard_triggers),
                        "reviewer": plan.classification.reviewer,
                        "rationale_codes": list(plan.classification.rationale_codes),
                        "rule_version": RULE_VERSION,
                    },
                },
            )
            if node.user_gate:
                kb._append_event(conn, task_id, "blocked", {"kind": "needs_input", "reason": "await user sail"})
    return CompiledGraph(plan.quest_id, plan.capsule_sha256, plan.classification, plan.graph_template, plan.nodes, task_ids)


def sail_before_execution(conn: sqlite3.Connection, graph: CompiledGraph) -> bool:
    """Return whether every execution node is downstream of a satisfied gate."""
    if not graph.task_ids:
        return True
    execution_id = graph.task_ids.get("execution")
    if not execution_id:
        return True
    rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id=? ORDER BY parent_id", (execution_id,)
    ).fetchall()
    parent_ids = {row["parent_id"] for row in rows}
    gate_id = graph.task_ids.get("user_sail")
    if gate_id:
        return gate_id in parent_ids and conn.execute("SELECT status FROM tasks WHERE id=?", (gate_id,)).fetchone()["status"] == "blocked"
    return True
