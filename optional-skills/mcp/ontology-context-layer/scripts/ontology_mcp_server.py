#!/usr/bin/env python3
"""
Ontology Context Layer — a lightweight MCP server for Hermes Agent.

A local, ontology-based semantic context layer: it gives the agent a structured
knowledge graph (entities + relationships) plus a business-rule engine, so it can
retrieve reliable context, validate facts against business logic, and explain
decisions with evidence.

Data model:
  - Entity:   {id, type, name, properties{...}, source, confidence, verified}
  - Relation: {from, to, type, properties{...}}
  - Rule:     {name, if[...], then, severity, mode}

Rules DSL (JSON, see references/rules-dsl.md in this skill):
  {
    "name": "High-value account",
    "if": [{"property": "annual_revenue", "op": "gt", "value": 100000}],
    "then": "qualified",
    "severity": "info",
    "mode": "all"
  }
  ops: eq, ne, gt, gte, lt, lte, in, contains, exists, not_exists
  conditions may also check relationships:
  {"relationship": "works_at", "target_type": "Company", "count_op": "gte", "count": 1}

Storage: JSON file at $HERMES_HOME/ontology_store.json (default ~/.hermes/).
Override with the ONTOLOGY_STORE env var. No network, no external services.

Usage:
  python ontology_mcp_server.py            # stdio MCP server (connect via `hermes mcp add`)
  python ontology_mcp_server.py --seed     # seed demo data, then serve
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Storage layer
# ---------------------------------------------------------------------------

DEFAULT_STORE = os.path.join(
    os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")),
    "ontology_store.json",
)
STORE_PATH = os.environ.get("ONTOLOGY_STORE", DEFAULT_STORE)


def _empty_store() -> dict[str, Any]:
    return {"entities": {}, "relations": [], "rules": [], "meta": {"created": time.time()}}


def load_store() -> dict[str, Any]:
    p = Path(STORE_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return _empty_store()
    return _empty_store()


def save_store(store: dict[str, Any]) -> None:
    Path(STORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(STORE_PATH).write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def _eid() -> str:
    return uuid.uuid4().hex[:10]


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

def _eval_op(op: str, actual: Any, expected: Any) -> tuple[bool, str]:
    if op == "eq":
        return actual == expected, f"expected == {expected!r}"
    if op == "ne":
        return actual != expected, f"expected != {expected!r}"
    if op in ("gt", "gte", "lt", "lte"):
        try:
            a, e = float(actual), float(expected)
        except (TypeError, ValueError):
            return False, f"'{actual}' not numeric"
        table = {"gt": a > e, "gte": a >= e, "lt": a < e, "lte": a <= e}
        return table[op], f"expected {op} {expected}"
    if op == "in":
        return actual in expected, f"expected one of {expected}"
    if op == "contains":
        return expected in actual, f"expected to contain {expected!r}"
    if op == "exists":
        return actual is not None and actual != "", "expected to exist"
    if op == "not_exists":
        return actual is None or actual == "", "expected to be absent"
    return False, f"unknown op {op}"


def _rule_condition_satisfied(
    condition: dict[str, Any], entity: dict[str, Any], store: dict[str, Any]
) -> tuple[bool, str]:
    """Evaluate one rule condition against an entity."""
    if "property" in condition:
        prop = condition["property"]
        # properties first, then top-level entity fields (verified, confidence, source...)
        if prop in entity.get("properties", {}):
            actual = entity["properties"][prop]
        else:
            actual = entity.get(prop)
        return _eval_op(condition.get("op", "eq"), actual, condition.get("value"))
    if "relationship" in condition:
        rel_type = condition["relationship"]
        matches = [r for r in store["relations"] if r["from"] == entity["id"] and r["type"] == rel_type]
        if condition.get("target_type"):
            matches = [
                r for r in matches
                if store["entities"].get(r["to"], {}).get("type") == condition["target_type"]
            ]
        return _eval_op(condition.get("count_op", "gte"), len(matches), condition.get("count", 1))
    return False, "condition missing 'property' or 'relationship'"


def evaluate_rule(rule: dict[str, Any], entity: dict[str, Any], store: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a rule against an entity. Returns pass/fail with per-condition reasons."""
    conditions = rule.get("if")
    if not conditions:
        return {"rule": rule.get("name", "?"), "passed": True, "reasons": [], "then": rule.get("then", "ok")}
    if isinstance(conditions, dict):
        conditions = [conditions]
    mode = rule.get("mode", "all")  # all | any
    satisfied = 0
    reasons: list[str] = []
    for cond in conditions:
        ok, detail = _rule_condition_satisfied(cond, entity, store)
        label = cond.get("property") or f"relationship:{cond.get('relationship')}"
        reasons.append(f"{'PASS' if ok else 'FAIL'} {label} {detail}")
        if ok:
            satisfied += 1
    passed = satisfied == len(conditions) if mode == "all" else satisfied > 0
    return {
        "rule": rule.get("name", "?"),
        "passed": passed,
        "mode": mode,
        "reasons": reasons,
        "then": rule.get("then", "ok"),
        "severity": rule.get("severity", "info"),
        "description": rule.get("description", ""),
    }


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "hermes-ontology",
    instructions=(
        "Semantic context layer: manage entities, relationships and business rules; "
        "query the knowledge graph; validate facts against rules; explain decisions. "
        "Use ontology_query before answering factual questions about known entities, "
        "ontology_validate to check facts against business logic, and ontology_explain "
        "to show why a decision passed or failed."
    ),
)


@mcp.tool()
def ontology_ingest_entity(
    type: str,
    name: str,
    properties: dict[str, Any] = {},
    source: str = "manual",
    confidence: float = 1.0,
    verified: bool = True,
    entity_id: str = "",
) -> dict[str, Any]:
    """Add or update an entity (a 'thing' the agent should know about) in the ontology graph."""
    store = load_store()
    eid = entity_id or name.lower().replace(" ", "_")[:64]
    if eid in store["entities"]:
        existing = store["entities"][eid]
        existing["properties"] = {**existing.get("properties", {}), **properties}
        existing.update({
            "name": name, "type": type, "source": source,
            "confidence": confidence, "verified": verified, "updated": time.time(),
        })
        action = "updated"
    else:
        store["entities"][eid] = {
            "id": eid, "type": type, "name": name, "properties": dict(properties),
            "source": source, "confidence": confidence, "verified": verified,
            "created": time.time(), "updated": time.time(),
        }
        action = "created"
    save_store(store)
    return {"action": action, "id": eid, "entity": store["entities"][eid]}


@mcp.tool()
def ontology_add_relationship(
    from_entity: str, to_entity: str, type: str, properties: dict[str, Any] = {}
) -> dict[str, Any]:
    """Link two entities with a typed relationship (e.g. from=acme_corp, to=jane_doe, type='employs')."""
    store = load_store()
    if from_entity not in store["entities"] or to_entity not in store["entities"]:
        missing = [e for e in (from_entity, to_entity) if e not in store["entities"]]
        return {"error": f"Unknown entities: {missing}. Ingest them first."}
    rel = {
        "id": _eid(), "from": from_entity, "to": to_entity, "type": type,
        "properties": dict(properties),
    }
    store["relations"].append(rel)
    save_store(store)
    return {"action": "added", "relationship": rel}


@mcp.tool()
def ontology_query(
    query: str = "",
    type: str = "",
    property_name: str = "",
    property_value: Any = None,
    verified_only: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    """Search entities by type, name keyword, or property. Use to look up context before answering."""
    store = load_store()
    results = []
    q = query.lower().strip()
    for e in store["entities"].values():
        if type and e["type"].lower() != type.lower():
            continue
        if verified_only and not e.get("verified", False):
            continue
        if property_name and e["properties"].get(property_name) != property_value:
            continue
        if q:
            hay = (e.get("name", "") + " " + json.dumps(e.get("properties", {}))).lower()
            if q not in hay:
                continue
        results.append(e)
    return {"count": len(results), "results": results[:limit]}


@mcp.tool()
def ontology_get_entity(entity_id: str) -> dict[str, Any]:
    """Get one entity by ID, including its outgoing relationships."""
    store = load_store()
    e = store["entities"].get(entity_id)
    if not e:
        return {"error": f"Entity '{entity_id}' not found. Use ontology_query to search."}
    rels = [r for r in store["relations"] if r["from"] == entity_id or r["to"] == entity_id]
    return {"entity": e, "relationships": rels}


@mcp.tool()
def ontology_traverse(
    entity_id: str, relationship_type: str = "", max_depth: int = 1
) -> dict[str, Any]:
    """Traverse the knowledge graph from an entity along relationship types (BFS, max_depth hops)."""
    store = load_store()
    if entity_id not in store["entities"]:
        return {"error": f"Entity '{entity_id}' not found."}
    visited = {entity_id}
    frontier = [entity_id]
    levels: list[dict[str, Any]] = []
    for _ in range(max(1, min(max_depth, 5))):
        level: dict[str, Any] = {}
        nxt = []
        for cur in frontier:
            for r in store["relations"]:
                if r["from"] == cur and (not relationship_type or r["type"] == relationship_type):
                    target = store["entities"].get(r["to"])
                    if target and r["to"] not in visited:
                        visited.add(r["to"])
                        nxt.append(r["to"])
                        level.setdefault(cur, []).append({
                            "relationship": r["type"], "to": r["to"],
                            "name": target.get("name"), "type": target.get("type"),
                        })
        if not level:
            break
        levels.append({"depth": len(levels) + 1, "connections": level})
        frontier = nxt
    return {"root": entity_id, "relationship_type": relationship_type or "any", "levels": levels}


@mcp.tool()
def ontology_add_rule(
    name: str,
    if_conditions: list[dict[str, Any]] | dict[str, Any],
    then: str = "ok",
    severity: str = "info",
    description: str = "",
    mode: str = "all",
) -> dict[str, Any]:
    """Add a business rule. Ops: eq,ne,gt,gte,lt,lte,in,contains,exists,not_exists over entity
    properties, or {'relationship': type, 'count_op': 'gte', 'count': N} over relations."""
    store = load_store()
    rule = {
        "name": name, "if": if_conditions, "then": then, "severity": severity,
        "description": description, "mode": mode, "created": time.time(),
    }
    store["rules"] = [r for r in store["rules"] if r["name"] != name]  # upsert by name
    store["rules"].append(rule)
    save_store(store)
    return {"action": "added", "rule": rule, "total_rules": len(store["rules"])}


@mcp.tool()
def ontology_validate(entity_id: str) -> dict[str, Any]:
    """Validate an entity against ALL business rules. Returns pass/fail per rule with reasons."""
    store = load_store()
    e = store["entities"].get(entity_id)
    if not e:
        return {"error": f"Entity '{entity_id}' not found."}
    if not store["rules"]:
        return {"entity_id": entity_id, "note": "No rules defined yet. Use ontology_add_rule.", "results": []}
    results = [evaluate_rule(r, e, store) for r in store["rules"]]
    return {
        "entity_id": entity_id,
        "entity_name": e.get("name"),
        "entity_type": e.get("type"),
        "overall": "PASS" if all(r["passed"] for r in results) else "FAIL",
        "results": results,
    }


@mcp.tool()
def ontology_explain(entity_id: str) -> dict[str, Any]:
    """Explain the current state of an entity: what's known, what rules pass/fail and why."""
    store = load_store()
    e = store["entities"].get(entity_id)
    if not e:
        return {"error": f"Entity '{entity_id}' not found."}
    rels = [r for r in store["relations"] if r["from"] == entity_id]
    validation = ontology_validate(entity_id)
    return {
        "entity": e,
        "outgoing_relationships": [
            {"type": r["type"], "to": r["to"], "name": store["entities"].get(r["to"], {}).get("name")}
            for r in rels
        ],
        "rule_check": validation.get("overall", "no rules"),
        "rule_details": validation.get("results", []),
        "confidence": e.get("confidence"),
        "verified": e.get("verified"),
        "source": e.get("source"),
    }


@mcp.tool()
def ontology_stats() -> dict[str, Any]:
    """Show ontology graph statistics: entity counts by type, relation counts, rule count."""
    store = load_store()
    by_type: dict[str, int] = {}
    for e in store["entities"].values():
        by_type[e.get("type", "?")] = by_type.get(e.get("type", "?"), 0) + 1
    rel_types: dict[str, int] = {}
    for r in store["relations"]:
        rel_types[r.get("type", "?")] = rel_types.get(r.get("type", "?"), 0) + 1
    return {
        "entities": len(store["entities"]),
        "by_type": by_type,
        "relations": len(store["relations"]),
        "by_relationship_type": rel_types,
        "rules": len(store["rules"]),
        "store_path": STORE_PATH,
    }


@mcp.tool()
def ontology_export() -> str:
    """Export the full ontology store as JSON (for backup / sharing)."""
    return json.dumps(load_store(), indent=2, ensure_ascii=False)


@mcp.tool()
def ontology_import(json_blob: str) -> dict[str, Any]:
    """Replace the ontology store from a JSON blob produced by ontology_export."""
    try:
        data = json.loads(json_blob)
    except Exception as exc:
        return {"error": f"Invalid JSON: {exc}"}
    if "entities" not in data or "relations" not in data or "rules" not in data:
        return {"error": "Not a valid ontology export (missing entities/relations/rules)."}
    save_store(data)
    return {"action": "imported", "entities": len(data["entities"]), "relations": len(data["relations"]), "rules": len(data["rules"])}


def seed_demo(store: dict[str, Any]) -> dict[str, Any]:
    """Load a small demo knowledge graph so the tools are immediately useful."""
    now = time.time()
    store["entities"] = {
        "acme_corp": {"id": "acme_corp", "type": "Company", "name": "Acme Corp",
                      "properties": {"annual_revenue": 250000, "industry": "Construction", "region": "ON"},
                      "source": "seed", "confidence": 1.0, "verified": True, "created": now, "updated": now},
        "jane_doe": {"id": "jane_doe", "type": "Person", "name": "Jane Doe",
                     "properties": {"role": "CFO", "email": "jane@acme.test"},
                     "source": "seed", "confidence": 1.0, "verified": True, "created": now, "updated": now},
        "bob_smith": {"id": "bob_smith", "type": "Person", "name": "Bob Smith",
                      "properties": {"role": "Procurement", "email": "bob@acme.test"},
                      "source": "seed", "confidence": 0.9, "verified": True, "created": now, "updated": now},
        "proj_fence": {"id": "proj_fence", "type": "Project", "name": "Fence Replacement",
                       "properties": {"value": 4500, "status": "Proposal", "deposit_paid": False},
                       "source": "seed", "confidence": 1.0, "verified": True, "created": now, "updated": now},
    }
    store["relations"] = [
        {"id": _eid(), "from": "jane_doe", "to": "acme_corp", "type": "works_at", "properties": {}},
        {"id": _eid(), "from": "bob_smith", "to": "acme_corp", "type": "works_at", "properties": {}},
        {"id": _eid(), "from": "proj_fence", "to": "acme_corp", "type": "sold_to", "properties": {}},
    ]
    store["rules"] = [
        {"name": "Large deal needs approval",
         "if": [{"property": "value", "op": "gt", "value": 10000}],
         "then": "requires_manager_approval", "severity": "warning",
         "description": "Deals over $10k need manager sign-off.", "mode": "all"},
        {"name": "Deposit required before work",
         "if": [{"property": "status", "op": "in", "value": ["Scheduled", "In Progress"]},
                {"property": "deposit_paid", "op": "eq", "value": True}],
         "then": "ok", "severity": "info",
         "description": "Scheduled projects must have deposit paid.", "mode": "all"},
        {"name": "Enterprise customer",
         "if": [{"property": "annual_revenue", "op": "gt", "value": 100000}],
         "then": "enterprise_tier", "severity": "info",
         "description": "Revenue over $100k = enterprise tier.", "mode": "all"},
    ]
    return store


if __name__ == "__main__":
    if "--seed" in sys.argv:
        st = load_store()
        if not st["entities"]:
            save_store(seed_demo(st))
            print(f"[hermes-ontology] demo data seeded to {STORE_PATH}", file=sys.stderr)
    mcp.run(transport="stdio")
