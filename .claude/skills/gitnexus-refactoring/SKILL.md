---
name: gitnexus-refactoring
description: "Use when the user wants to rename, extract, split, move, or restructure code safely. Examples: \"Rename this function\", \"Extract this into a module\", \"Refactor this class\", \"Move this to a separate file\""
---

# Refactoring with GitNexus

## When to Use

- "Rename this function safely"
- "Extract this into a module"
- "Split this service"
- "Move this to a new file"
- Any task involving renaming, extracting, splitting, or restructuring code

## Bind the repository first

Refactoring writes to disk. `rename` with `dry_run: false` edits files in
whichever repository was resolved, so binding identity here is a safety gate,
not bookkeeping.

Call `list_repos {}` before the first tool call. With one indexed repository,
use the examples below as written. With more than one, pass `repo` on every
call: an omitted `repo` normally errors, but under an MCP policy with a
configured default it resolves to that default silently. If you cannot tell
which repository is meant, stop and ask. Never run `rename` with
`dry_run: false` until the preview in the same bound repository has been
reviewed — its returned `file_path` values show which checkout is about to be
written, so read them as a confirmation of identity.

`list_repos` is paginated, so page with `offset: pagination.nextOffset` until
`hasMore` is false before concluding a repository is absent.

`detect_changes` takes `worktree` when you are editing a linked worktree the
MCP server was not launched from; otherwise `git diff` runs in the wrong
checkout and reports nothing changed, which reads as a verified refactor.

## Workflow

```
0. list_repos {}                                  → Bind repo (and worktree)
1. impact({target: "X", direction: "upstream"})  → Map all dependents
2. query({search_query: "X"})                            → Find execution flows involving X
3. context({name: "X"})                           → See all incoming/outgoing refs
4. Plan update order: interfaces → implementations → callers → tests
```

> If "Index is stale" → run `node .gitnexus/run.cjs analyze` in terminal.

## Checklists

### Rename Symbol

```
- [ ] list_repos {} — bind repo; explicit repo when >1 indexed, ask if ambiguous
- [ ] rename({symbol_name: "oldName", new_name: "newName", dry_run: true}) — preview all edits
- [ ] Confirm the previewed file paths are in the bound repository/worktree
- [ ] Review graph edits (high confidence) and text_search edits (review carefully)
- [ ] If satisfied: rename({..., dry_run: false}) — apply edits
- [ ] detect_changes() — verify only expected files changed
- [ ] Run tests for affected processes
```

### Extract Module

```
- [ ] list_repos {} — bind repo; explicit repo when >1 indexed, ask if ambiguous
- [ ] context({name: target}) — see all incoming/outgoing refs
- [ ] impact({target, direction: "upstream"}) — find all external callers
- [ ] Define new module interface
- [ ] Extract code, update imports
- [ ] detect_changes() — verify affected scope
- [ ] Run tests for affected processes
```

### Split Function/Service

```
- [ ] list_repos {} — bind repo; explicit repo when >1 indexed, ask if ambiguous
- [ ] context({name: target}) — understand all callees
- [ ] Group callees by responsibility
- [ ] impact({target, direction: "upstream"}) — map callers to update
- [ ] Create new functions/services
- [ ] Update callers
- [ ] detect_changes() — verify affected scope
- [ ] Run tests for affected processes
```

## Tools

**rename** — automated multi-file rename:

```
rename({symbol_name: "validateUser", new_name: "authenticateUser", repo: "my-app", dry_run: true})
→ 12 edits across 8 files
→ 10 graph edits (high confidence), 2 text_search edits (review)
→ Changes: [{file_path, edits: [{line, old_text, new_text, confidence}]}]
```

**impact** — map all dependents first:

```
impact({target: "validateUser", repo: "my-app", direction: "upstream"})
→ d=1: loginHandler, apiMiddleware, testUtils
→ Affected Processes: LoginFlow, TokenRefresh
```

**detect_changes** — verify your changes after refactoring:

```
detect_changes({scope: "all"})
→ Changed: 8 files, 12 symbols
→ Affected processes: LoginFlow, TokenRefresh
→ Risk: MEDIUM
```

`partial: true` (a graph query failed) or `truncated: true` (the changed-symbol
listing was capped) means the result is short of the truth: a short or empty
list is not proof that only the expected files changed. Re-run it rather than
treat the refactor as verified.

A wrong-worktree zero carries neither flag and is indistinguishable from a
clean verification, so confirm the diffed checkout is the one you edited.

**cypher** — custom reference queries:

```cypher
MATCH (caller)-[:CodeRelation {type: 'CALLS'}]->(f:Function {name: "validateUser"})
RETURN caller.name, caller.filePath ORDER BY caller.filePath
```

## Risk Rules

| Risk Factor         | Mitigation                                |
| ------------------- | ----------------------------------------- |
| Many callers (>5)   | Use rename for automated updates |
| Cross-area refs     | Use detect_changes after to verify scope  |
| String/dynamic refs | query to find them               |
| External/public API | Version and deprecate properly            |
| Same name in another indexed repo | Bind `repo`; verify previewed paths before applying |

## Example: Rename `validateUser` to `authenticateUser`

```
0. list_repos {}
   → total: 2 (my-app, billing-api) — both define validateUser, so bind explicitly

1. rename({symbol_name: "validateUser", new_name: "authenticateUser", repo: "my-app", dry_run: true})
   → 12 edits: 10 graph (safe), 2 text_search (review)
   → Files: validator.ts, login.ts, middleware.ts, config.json...

2. Review text_search edits (config.json: dynamic reference!)

3. rename({symbol_name: "validateUser", new_name: "authenticateUser", repo: "my-app", dry_run: false})
   → Applied 12 edits across 8 files

4. detect_changes({scope: "all", repo: "my-app"})
   → Affected: LoginFlow, TokenRefresh
   → Risk: MEDIUM — run tests for these flows
   Repository: my-app (/abs/path/my-app)  Worktree: same  Index: current
```

With a single indexed repository, step 0 returns `total: 1` and the `repo`
argument drops out of every call above.
