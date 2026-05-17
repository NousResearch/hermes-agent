# MSIG Codex Kanban Card Template

Use this shape for cards assigned to `codex`. The card body is the execution contract; Codex runs non-interactively, so the card must be scoped enough to execute without live steering.

```markdown
Goal:
<One clear end state. Treat this as the Codex goal.>

Context:
- Repo/workspace: <path or worktree policy>
- Current branch/base: <staging/main/etc.>
- Relevant files/surfaces: <paths, routes, UI surfaces, jobs>
- Background notes: <brief, only what matters>

Acceptance criteria:
- <observable requirement 1>
- <observable requirement 2>
- <observable requirement 3>

Boundaries / non-goals:
- Do not merge to main.
- Do not deploy/promote without explicit approval.
- Do not rotate secrets or touch production data.
- Stay inside assigned workspace unless explicitly authorized.
- <task-specific no-go items>

Required working style:
- Investigate first; read repo guidance and relevant existing code before editing.
- Set your own internal goal/plan from this card.
- Implement the scoped goal, then self-review the diff.
- Run relevant targeted gates; if a gate is too expensive or unavailable, explain why.

Verification / gates:
- <command 1>
- <command 2>
- <browser/dogfood proof if relevant>

Receipt required in final response:
- Changed files
- Commands/gates run and results
- Verification evidence
- Remaining risks/blockers
- Recommended reviewer action
```

Kanban policy:
- Implementation cards use `--assignee codex`.
- Code-changing success blocks as `review-required`.
- Failure blocks as `codex-failed`.
- Marshall verifies and either marks done, unblocks with fix instructions, or asks Troy for approval.
