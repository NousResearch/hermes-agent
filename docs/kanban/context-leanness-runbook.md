# Kanban context-leanness runbook

Audit: task `t_4bf79a41` · Guardrail commit: `57904454c` (re-landed on this tree).

## What this guards

Kanban worker handoff must stay **lean and task-scoped**. A worker spawn is a
fresh `hermes -p <assignee> --cli chat -q "work kanban task <id>"` process;
it receives **no** full-session context. Working Context is derived at runtime
by the worker's first `kanban_show()` → `build_worker_context()`, which is
hard-capped so even a pathological board (retry-storm, comment-storm, giant
bodies) collapses to a bounded prompt.

## The caps (hermes_cli/kanban_db.py)

| Constant                  | Value | Applies to                               |
|---------------------------|-------|------------------------------------------|
| `_CTX_MAX_PRIOR_ATTEMPTS` | 10    | prior runs shown in full                 |
| `_CTX_MAX_COMMENTS`       | 30    | comments shown in full                   |
| `_CTX_MAX_FIELD_BYTES`    | 4 KB  | per summary/error/metadata/result        |
| `_CTX_MAX_BODY_BYTES`     | 8 KB  | per task body (opening post)             |
| `_CTX_MAX_COMMENT_BYTES`  | 2 KB  | per comment                              |

Plus: assignee role history limited to the 5 most-recent completed runs
(`LIMIT 5`), first 200 chars of each summary; done-parent handoffs per-field
capped.

## Instrumentation

`build_worker_context()` logs the rendered size at the return point:

    _log.debug("kanban worker context for %s: %d chars (%d lines)",
               task_id, len(text), len(lines))

Context is derived at runtime on first `kanban_show`, so this is the
authoritative observation point for context-size drift.

## Regression test

`tests/plugins/test_kanban_context_leanness.py` enforces:

1. `build_worker_context` output stays bounded (< 512 KB) under pathological
   inputs (64 KB body, 200-comment storm, 128 KB summary), and per-field caps
   truncate with visible `… [truncated, N chars omitted]` markers.
2. No full-session-history section can appear in the render.
3. The spawned worker env carries no content/session/history blob key, and the
   spawn argv is exactly the fixed `work kanban task <id>` prompt.

Run locally:

    cd /usr/local/lib/hermes-agent
    python -m pytest tests/plugins/test_kanban_context_leanness.py -v

## Levers on prompt size (for operators)

Because kanban context is derive-on-demand from the board DB, the real lever
on prompt size is **board-DB hygiene** — pruning comments, collapsing giant
summaries — not passing more/fewer session turns at spawn. Watch the
`kanban worker context for <id>: N chars` debug line for drift.

## Re-land note (this tree)

The original guardrail commit `57904454c` was swept off `main` when a
`hermes-update` reset the branch to `origin/main`, orphaning the commit
(preserved at
`.git/refs/hermes-update-backups/orphan-main-20260906-185016-57904454cec0`).
The three source changes (instrumentation, doc-comment fix, regression test)
were re-applied directly to this working tree so the guardrail is live here.
Any commit landing on a local `main` must be pushed to `origin` or it will be
orphaned again by the next update reset.