# High-autonomy approval prompt audit for t_374a6388

Sources searched:
- session_search query: "requires approval" OR "Approval required" OR "approval_required" OR "pending approval" (newest, user/assistant/tool).
- session_search query: "Do you approve" OR "approval prompt" OR "approve" "deny" "command" (newest, user/assistant/tool).
- session_search query: Medium-risk approval prompt surfaced Yunuen approval prompts.
- Hermes Maintenance Kanban task logs under ~/.hermes/kanban/boards/hermes-maintenance/logs were enumerated for approval-related task logs.

## Classified recent prompt classes

| Class | Evidence | Classification | Fix / policy |
|---|---|---|---|
| Shell/script wrapper prompts | t_533eba26 prior issue and command class python -c / bash -lc. | Medium-risk but routine/scoped. | Prior bridge covers exact wrapper descriptions: High/Full auto-approves; Medium still prompts. |
| Scoped scratch recursive delete | Session 20260606_143634: task t_533eba26 command hit pending_approval while preserving artifacts and removing a task-local artifacts dir before rebase/tests. Session 20260606_141129: temporary deployment-test tree cleanup in /tmp/hm-runtime-v202606063-testtree hit pending_approval. | Medium-risk but routine/scoped only when target is under task workspace or a scratch /tmp subdir. Genuinely HIGH-risk when target is root/home/system, globbed, parent-traversal, or otherwise outside scope. | Added deterministic scoped target parser for recursive delete warnings; High/Full auto-approves scoped scratch cleanup only. |
| GitHub evidence write prompts | Session 20260606_091810: gh pr comment hit local GitHub authority guard pending_approval. Sessions 20260605_223424 and 20260606_072842 recorded git push yellow-operation approval gates. | Medium-risk but routine/scoped for branch push and PR/issue evidence comments/creates; still high/held when command is merge/review/close/workflow/release/secrets/protected/force push. | Added High/Full auto-handling for GitHub authority yellow warnings only in bounded families/reasons; added gh issue read/write classification. |
| Tirith mixed/high findings | Session 20260606_085648: command with hermes output piped to python produced Security scan including [MEDIUM] schemeless URL and [HIGH] pipe-to-interpreter. | Genuinely HIGH-risk or avoidable command-shape issue. | Kept gated; do not auto-approve Tirith policy warnings. Rewrite to avoid pipe-to-interpreter and inspect output first. |
| Deployment/restart/secrets/config | Task scope explicitly preserves deployment/live restart, secrets/access, destructive/global operations. Existing regression covers ~/.hermes/config.yaml and hermes gateway restart. | Genuinely HIGH-risk. | Kept gated under High autonomy. |

## Representative session-search evidence snippets

- 20260606_143634_b15603: pending_approval command for t_533eba26 rebase/test workflow; description "recursive delete" / "delete in root path" on task-local cleanup before tests.
- 20260606_141129_bbe5ce: pending_approval command for temporary deployment test tree cleanup under /tmp/hm-runtime-v202606063-testtree.
- 20260606_091810_b3ce19: pending_approval command for gh pr comment 47; local GitHub authority guard surfaced a yellow-operation approval.
- 20260606_085648_33a729: pending_approval with Tirith Security scan; included [HIGH] pipe-to-interpreter and [MEDIUM] schemeless URL. Kept as high/avoidable, not auto-handled.
- 20260605_223424_95ddb5 and 20260606_072842_3c33d125: summaries mention git push yellow-operation approval gates blocking GitHub branch/PR progression.

## Code-path trace

- tools/approval.py check_all_command_guards path: hardline/sudo guard -> yolo/off -> CLI/gateway context -> Tirith -> detect_dangerous_command -> GitHub authority -> warning collection -> High/Full project autonomy bridge -> smart/manual/gateway approval.
- _get_project_autonomy_level reads active board metadata via hermes_cli.kanban_db.read_board_metadata(HERMES_KANBAN_BOARD), defaulting to Medium on failure.
- Gateway/Kanban worker propagation uses HERMES_KANBAN_BOARD/HERMES_KANBAN_WORKSPACE/HERMES_SESSION_KEY; tests monkeypatch these deterministically.
- No LLM decision is used for High-autonomy suppression; smart approval remains later and is skipped for GitHub authority policy warnings.

## Regression coverage added/extended

- High autonomy auto-approves scoped scratch recursive delete.
- High autonomy does not auto-approve unscoped recursive delete.
- High autonomy auto-approves scoped GitHub PR evidence comment.
- Medium autonomy still prompts for the same GitHub PR evidence comment.
- High autonomy does not auto-approve GitHub PR merge.
- GitHub authority now classifies gh issue comment as yellow scoped write.

