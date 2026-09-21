---
name: gitnexus-impact-analysis
description: "Use when the user wants to know what will break if they change something, or needs safety analysis before editing code. Examples: \"Is it safe to change X?\", \"What depends on this?\", \"What will break?\""
---

# Impact Analysis with GitNexus

## When to Use

- "Is it safe to change this function?"
- "What will break if I modify X?"
- "Show me the blast radius"
- "Who uses this code?"
- Before making non-trivial code changes
- Before committing — to understand what your changes affect

## Bind the repository first

Impact analysis is the gate that authorizes an edit, so it must answer for the
repository you are about to edit.

Call `list_repos {}` before the first tool call. With one indexed repository,
use the examples below as written. With more than one, pass `repo` on every
call: an omitted `repo` normally errors, but under an MCP policy with a
configured default it resolves to that default silently. If you cannot tell
which repository is meant, stop and ask — every result below an ambiguous
identity inherits the ambiguity. `list_repos` is paginated, so page with
`offset: pagination.nextOffset` until `hasMore` is false before concluding a
repository is absent.

`detect_changes` takes `worktree` when your changes are in a linked worktree
the MCP server was not launched from. The server auto-detects a worktree only
when it was launched from inside one; otherwise `git diff` runs in the wrong
checkout and reports zero changed symbols — a false clean check that carries
none of the degradation flags described below. In the CLI fallbacks, `--repo .`
means the current checkout; pass the intended repository path instead when you
are not standing in it.

State the bound identity with your risk report:

```
Repository: <name> (<path>)   Worktree: <path>   Index: <commit>, <n> behind HEAD
```

## Workflow

```
0. list_repos {}                                           → Bind repo (and worktree)
1. impact({target: "X", direction: "upstream"}) or `node .gitnexus/run.cjs impact "X" --direction upstream --repo .`
2. READ gitnexus://repo/{name}/processes                   → Check affected execution flows
3. detect_changes({scope: "all"}) or `node .gitnexus/run.cjs detect-changes --scope all --repo .`
4. Assess risk and report to user, echoing repo/worktree/index identity
```

> If "Index is stale" → run `node .gitnexus/run.cjs analyze` in terminal.
> If `.gitnexus/run.cjs` is missing, replace `node .gitnexus/run.cjs` with `npx gitnexus` in the fallback commands.

## Checklist

```
- [ ] list_repos {} — bind repo; explicit repo when >1 indexed, ask if ambiguous
- [ ] impact({target, direction: "upstream"}) or CLI fallback to find dependents
- [ ] Review d=1 items first (these WILL BREAK)
- [ ] Check high-confidence (>0.8) dependencies
- [ ] READ processes to check affected execution flows
- [ ] detect_changes({scope: "all"}) or CLI fallback for pre-commit check
- [ ] Confirm the checkout you edited is the checkout that was diffed
- [ ] Assess risk level and report, stating repo/worktree/index identity
```

## Understanding Output

| Depth | Risk Level       | Meaning                  |
| ----- | ---------------- | ------------------------ |
| d=1   | **WILL BREAK**   | Direct callers/importers |
| d=2   | LIKELY AFFECTED  | Indirect dependencies    |
| d=3   | MAY NEED TESTING | Transitive effects       |

## Risk Assessment

| Affected                       | Risk     |
| ------------------------------ | -------- |
| <5 symbols, few processes      | LOW      |
| 5-15 symbols, 2-5 processes    | MEDIUM   |
| >15 symbols or many processes  | HIGH     |
| Critical path (auth, payments) | CRITICAL |
| **Zero callers found**         | **UNKNOWN** |

`UNKNOWN` is not a low rung on this scale — it means the walk could not answer.
An empty caller set is equally consistent with "genuinely unused" and "the
callers are not resolvable by the index" (plain-object property access, dynamic
dispatch, cross-language calls), so few-callers ⇒ LOW does **not** apply. The
result carries a `riskNote` saying so. Confirm with a text search before
treating the symbol as safe to change or delete.

## Tools

**impact** — the primary tool for symbol blast radius. If MCP is unavailable, use `node .gitnexus/run.cjs impact <symbol> --direction upstream --repo .` instead:

```
impact({
  target: "validateUser",
  repo: "my-app",          // required once >1 repository is indexed
  direction: "upstream",
  minConfidence: 0.8,
  maxDepth: 3
})

→ d=1 (WILL BREAK):
  - loginHandler (src/auth/login.ts:42) [CALLS, 100%]
  - apiMiddleware (src/api/middleware.ts:15) [CALLS, 100%]

→ d=2 (LIKELY AFFECTED):
  - authRouter (src/routes/auth.ts:22) [CALLS, 95%]
```

**detect_changes** — git-diff based impact analysis. If MCP is unavailable, use `node .gitnexus/run.cjs detect-changes --scope all --repo .` instead:

```
detect_changes({scope: "all"})

→ Changed: 5 symbols in 3 files
→ Affected: LoginFlow, TokenRefresh, APIMiddlewarePipeline
→ Risk: MEDIUM
```

Add `repo` once more than one repository is indexed, and `worktree: "<abs
path>"` when your changes are in a linked worktree the server was not launched
from.

`partial: true` (a graph query failed) or `truncated: true` (the changed-symbol
listing was capped) means the result is short of the truth, and reads like
`UNKNOWN` above: a zero there means unseen, not unaffected. Re-run it rather
than tick the pre-commit check.

A wrong-worktree zero carries neither flag and is shape-identical to a genuine
clean result, so confirm the checkout you edited is the one that was diffed
before treating an empty change set as a passed check.

## Example: "What breaks if I change validateUser?"

```
0. list_repos {}
   → total: 2 (my-app, billing-api) — both define validateUser, so bind explicitly

1. impact({target: "validateUser", repo: "my-app", direction: "upstream"}) or `node .gitnexus/run.cjs impact "validateUser" --direction upstream --repo .`
   → d=1: loginHandler, apiMiddleware (WILL BREAK)
   → d=2: authRouter, sessionManager (LIKELY AFFECTED)

2. READ gitnexus://repo/my-app/processes
   → LoginFlow and TokenRefresh touch validateUser

3. Risk: 2 direct callers, 2 processes = MEDIUM
   Repository: my-app (/abs/path/my-app)  Worktree: same  Index: current
```

With a single indexed repository, step 0 returns `total: 1` and the `repo`
argument drops out of every call above.
