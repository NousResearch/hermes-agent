---
name: ontology-context-layer
description: Give Hermes a knowledge graph + business rules for explainable decisions.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ontology, knowledge-graph, MCP, context, rules, explainability]
    related_skills: [hermes-agent, fastmcp, mcporter]
prerequisites:
  commands: [python3]
---

# Ontology Context Layer

A local, ontology-based semantic context layer for Hermes, delivered as an MCP server. It combines a knowledge graph (entities + typed relationships) with a business-rule engine so the agent can retrieve reliable context, validate facts against business logic, and explain decisions with evidence — without any cloud dependency.

## When to Use

Use this skill when the task is to:

- install or re-connect the ontology MCP server (`hermes mcp add` → tools appear as `ontology_*`)
- seed demo data and verify the server works end to end
- define entities, relationships, or business rules (see `references/rules-dsl.md`)
- teach the agent to look up context before answering (`ontology_query`, `ontology_get_entity`, `ontology_traverse`)
- validate a fact or proposed action against business rules (`ontology_validate`) and justify the outcome (`ontology_explain`)
- back up or migrate the ontology store (`ontology_export` / `ontology_import`)

Use `fastmcp` when building a *new* MCP server from scratch. Use `mcporter` for ad-hoc CLI access to an existing MCP server. Use this skill when the goal is an ontology/graph + rules context layer specifically.

## Included Files

- `scripts/ontology_mcp_server.py` — the MCP server (stdio transport, FastMCP)
- `scripts/test_mcp.py` — protocol smoke test (list tools, ingest, rule, validate, export)
- `references/rules-dsl.md` — full rules DSL reference with worked examples

## Install

The server ships in this skill; register it with Hermes' native MCP client:

```bash
hermes mcp add ontology --command python --args "<path-to>/scripts/ontology_mcp_server.py"
```

Then **start a new session** (MCP tools have no hot-reload). The 11 tools appear as `ontology_*` in every conversation. Verify with:

```bash
hermes mcp list
python "<path-to>/scripts/test_mcp.py"     # protocol smoke test
python "<path-to>/scripts/ontology_mcp_server.py" --seed   # optional demo data
```

The store is a JSON file at `$HERMES_HOME/ontology_store.json` (default `~/.hermes/`); override with the `ONTOLOGY_STORE` env var. No network, no external services.

## Data Model

- **Entity** — a thing the agent should know: `{id, type, name, properties{...}, source, confidence, verified}`. `verified=false` marks unconfirmed facts so downstream rules can flag them.
- **Relation** — a typed link: `{from, to, type}` (e.g. `jane_doe --works_at--> acme_corp`).
- **Rule** — business logic: `{name, if:[conditions], then, severity, mode}`. Validation runs every rule against an entity and reports per-condition PASS/FAIL reasons.

## Tool Reference

| Tool | Purpose |
|---|---|
| `ontology_ingest_entity` | Add/update an entity (upsert by id or slugified name) |
| `ontology_add_relationship` | Link two entities with a type |
| `ontology_query` | Search entities by type, keyword, or property |
| `ontology_get_entity` | One entity + its relationships |
| `ontology_traverse` | BFS graph traversal along relationship types |
| `ontology_add_rule` | Add/upsert a business rule by name |
| `ontology_validate` | Run all rules against an entity → PASS/FAIL + reasons |
| `ontology_explain` | Entity state + rule outcomes + evidence (confidence, source) |
| `ontology_stats` | Graph statistics |
| `ontology_export` / `ontology_import` | JSON backup / restore |

## Usage Patterns

### 1. Install and smoke-test

```bash
hermes mcp add ontology --command python --args "$(pwd)/scripts/ontology_mcp_server.py"
python scripts/test_mcp.py
```

Completion criterion: `ALL SMOKE TESTS PASSED` and `hermes mcp list` shows `ontology ... ✓ enabled`.

### 2. Build context before answering

When the user asks about an entity the agent knows, query the graph first, then answer from verified entities. Ingest new facts as they surface, marking unconfirmed ones `verified=false`.

### 3. Validate and explain a decision

```text
ontology_validate(entity_id="acme_corp")   → overall PASS/FAIL + per-rule reasons
ontology_explain(entity_id="acme_corp")    → entity + rule evidence + confidence/source
```

Completion criterion: every claim in the final answer traces to a PASS rule or a cited entity property.

### 4. Backup before migrating

Call `ontology_export`, save the JSON, `ontology_import` on the target machine.

## Common Pitfalls

1. **Forgetting to restart Hermes after `hermes mcp add`.** MCP servers load at session start — tools stay invisible until a new session.
2. **Rules can't see nested properties.** A condition checks `properties.<key>` first, then top-level entity fields (`verified`, `confidence`, `source`, `id`, `type`, `name`). Store rule-relevant facts at the top level of `properties` or as entity fields.
3. **Hardcoding the store path.** The store is `$HERMES_HOME/ontology_store.json`; always read `ontology_stats` (it reports the path) instead of assuming `~/.hermes`.
4. **Ingesting unverified facts as truth.** Leave `verified=false` for uncertain data so `ontology_validate` can flag it; set `verified_only=true` in queries when the answer must be confirmed.
5. **Rule conditions are AND by default.** Use `"mode": "any"` when at least one condition should satisfy the rule.
6. **Traversal depth limit.** `ontology_traverse` caps at 5 hops by design — use `ontology_query` for broad searches instead of deep walks.

## Verification Checklist

- [ ] `python scripts/test_mcp.py` → `ALL SMOKE TESTS PASSED`
- [ ] `hermes mcp list` shows the server with status `✓ enabled`
- [ ] Tools appear in a fresh session as `ontology_*` (new session required after add)
- [ ] A rule with a top-level field (`verified`) and a rule with a `properties` field both evaluate correctly
- [ ] `ontology_stats` shows the expected store path and counts
- [ ] `ontology_export` output round-trips through `ontology_import`
