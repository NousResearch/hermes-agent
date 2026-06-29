---
name: gitnexus-guide
description: "Use when the user asks about GitNexus itself — available tools, how to query the knowledge graph, MCP resources, graph schema, or workflow reference. Examples: \"What GitNexus tools are available?\", \"How do I use GitNexus?\""
---

# GitNexus Guide

Quick reference for all GitNexus MCP tools, resources, and the knowledge graph schema.

## Always Start Here

For any task involving code understanding, debugging, impact analysis, or refactoring:

1. **Read `gitnexus://repo/{name}/context`** — codebase overview + check index freshness
2. **Match your task to a skill below** and **read that skill file**
3. **Follow the skill's workflow and checklist**

> If step 1 warns the index is stale, run `node .gitnexus/run.cjs analyze` in the terminal first.

## Skills

| Task                                         | Skill to read       |
| -------------------------------------------- | ------------------- |
| Understand architecture / "How does X work?" | `gitnexus-exploring`         |
| Blast radius / "What breaks if I change X?"  | `gitnexus-impact-analysis`   |
| Trace bugs / "Why is X failing?"             | `gitnexus-debugging`         |
| Rename / extract / split / refactor          | `gitnexus-refactoring`       |
| Tools, resources, schema reference           | `gitnexus-guide` (this file) |
| Index, status, clean, wiki CLI commands      | `gitnexus-cli`               |

## Tools Reference

| Tool             | What it gives you                                                        |
| ---------------- | ------------------------------------------------------------------------ |
| `query`          | Process-grouped code intelligence — execution flows related to a concept |
| `context`        | 360-degree symbol view — categorized refs, processes it participates in  |
| `impact`         | Symbol blast radius — what breaks at depth 1/2/3 with confidence         |
| `trace`          | Shortest path between two symbols — "how does A reach B?" in one call     |
| `detect_changes` | Git-diff impact — what do your current changes affect                    |
| `rename`         | Multi-file coordinated rename with confidence-tagged edits               |
| `cypher`         | Raw graph queries (read `gitnexus://repo/{name}/schema` first)           |
| `explain`        | Persisted taint findings — source→sink data flows (needs `analyze --pdg`) |
| `pdg_query`      | Control/data dependence — what gates X (CDG) / where Y flows (REACHING_DEF); needs `analyze --pdg` |
| `check`          | Check graph invariants such as circular imports                          |
| `list_repos`     | Discover indexed repos (paginated — `limit`/`offset`)                    |

### Paginating `list_repos`

`list_repos` is paginated so a large registry is not truncated by MCP/LLM token limits. It takes optional `limit` (default **50**, max **200**) and `offset`, and returns:

```jsonc
{
  "repositories": [
    { "name": "...", "path": "...", "indexedAt": "...", "lastCommit": "...", "stats": { } }
  ],
  "pagination": {
    "total": 437,
    "limit": 50,
    "offset": 0,
    "returned": 50,
    "hasMore": true,
    "nextOffset": 50
  }
}
```

To enumerate **every** repository, keep calling with `offset` set to `pagination.nextOffset` until `hasMore` is `false`:

```text
list_repos {}               → repos 1–50,    nextOffset 50,  hasMore true
list_repos { offset: 50 }   → repos 51–100,  nextOffset 100, hasMore true
…
list_repos { offset: 400 }  → repos 401–437,                 hasMore false   (done)
```

Notes: `offset` ≥ `total` returns an empty page (with `total` still reported). Out-of-range or malformed `limit`/`offset` (non-integer, `limit` outside `[1, 200]`, `offset < 0`) are rejected with a clear error — `limit` above the max is rejected, not silently capped. The order is deterministic (lower-cased name, then path), so paging never skips or duplicates an entry while the registry is unchanged.

### Taint findings (`explain`)

`explain` returns intra-procedural taint findings (`TAINTED` edges) recorded by `gitnexus analyze --pdg` — each with a sink category (command-injection, code-injection, path-traversal, sql-injection, xss), source/sink lines, and the ordered hop path with the variable carried on each hop.

- `explain {}` — enumerate all findings for the repo (bounded by `limit`, deterministic order)
- `explain { target: "src/vuln.ts" }` — findings in a file (suffix path match accepted)
- `explain { target: "runUserCommand" }` — findings in a function (resolved like `context`; ambiguous names return ranked candidates)

A repo indexed without `--pdg` returns a clear "no taint layer" note. Caveats: findings are intra-procedural only — cross-function, closure/callback, property/field, and implicit flows are not modeled, so the absence of a finding is **not** proof of safety. `SANITIZES` (sanitizer-kill) edges are queryable via `cypher`.

### Control & data dependence (`pdg_query`)

`pdg_query` reads the control/data-dependence layers `gitnexus analyze --pdg` records (CDG + REACHING_DEF, basic-block granular) — the control/data analog of `explain`. It is **always anchored** (a `target` file path or symbol, resolved like `context`) and has two modes:

- `pdg_query { mode: "controls", target: "..." }` — CDG: "under what condition does X run?". Each edge is a controlling predicate block → dependent block with the branch sense (`'T'`/`'F'`) in `reason`; an edge into an early `return`/`throw` is flagged `guard: true` (guard-clause discovery — the sense depends on the predicate, so don't filter guards by a fixed label).
- `pdg_query { mode: "flows", target: "...", variable?: "..." }` — REACHING_DEF def→use edges within the function; pass `variable` to trace one binding.

A repo indexed without `--pdg` returns a "no PDG layer" note (or "status unknown" when the layer can't be confirmed). Intra-procedural only — cross-function flow is taint's domain (`explain`). The raw CDG/REACHING_DEF edges are also queryable via `cypher`. See the `gitnexus-pdg-query` skill for the full query surface.

### Shortest path between two symbols (`trace`)

`trace` answers "how does A reach B?" in one call — the shortest directed path over `CALLS` (plus `HAS_METHOD`, so a class-rooted trace descends into its methods) instead of chaining 3–8 `context`/`impact` hops by hand.

- `trace { from: "validateUser", to: "executeQuery" }` — shortest path between two symbols.
- Disambiguate common names with `from_uid`/`to_uid` (zero-ambiguity) or `from_file`/`to_file`; an ambiguous name returns ranked candidates.
- `maxDepth` (default 10, max 30) bounds the search; `includeTests` (default false) lets the traversal pass through test-file symbols.

Returns ordered `hops` (each `{ name, filePath, startLine }`) and an aligned `edges[]` of `{ relType, confidence }`, so call hops and containment (`HAS_METHOD`) hops stay distinguishable. When no path exists it reports the **furthest** reachable node (where the chain breaks) and sets `truncated: true` if a traversal cap was hit first. Every result carries a `status`: `ok` / `no_path` / `ambiguous` / `not_found` / `error`.

## Resources Reference

Lightweight reads (~100-500 tokens) for navigation:

| Resource                                       | Content                                   |
| ---------------------------------------------- | ----------------------------------------- |
| `gitnexus://repo/{name}/context`               | Stats, staleness check                    |
| `gitnexus://repo/{name}/clusters`              | All functional areas with cohesion scores |
| `gitnexus://repo/{name}/cluster/{clusterName}` | Area members                              |
| `gitnexus://repo/{name}/processes`             | All execution flows                       |
| `gitnexus://repo/{name}/process/{processName}` | Step-by-step trace                        |
| `gitnexus://repo/{name}/schema`                | Graph schema for Cypher                   |

## Graph Schema

**Nodes:** File, Function, Class, Interface, Method, Community, Process
**Edges (via CodeRelation.type):** CALLS, IMPORTS, EXTENDS, IMPLEMENTS, DEFINES, MEMBER_OF, STEP_IN_PROCESS

```cypher
MATCH (caller)-[:CodeRelation {type: 'CALLS'}]->(f:Function {name: "myFunc"})
RETURN caller.name, caller.filePath
```
