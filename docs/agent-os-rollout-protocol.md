# Agent OS Rollout Protocol

Use this rollout only after the accepted `prior_work_first` core is present and
focused tests pass on the current branch.

## Execution Order

1. Fresh-read local `HEAD`, `TREE`, worktree status, remote branch tip, and PR state.
2. Run focused Agent OS tests and repair only machine-fixable failures.
3. Audit local agent targets read-only first:
   - `Codex`: per-user rule file `~/.codex/AGENTS.md`
   - `Claude Code`: per-user rule file `~/.claude/CLAUDE.md` when the CLI is installed
   - `Cursor`: project-local `.cursorrules` or `.cursor/rules/*.mdc`
   - `Gemini CLI`: only when a stable local rule location is proven
   - `Hermes`: repository `SOUL.md` + `AGENTS.md`, plus runtime activation state
4. Use `Codex` as first canary with deterministic installer apply + readback.
5. Roll out only to supported installed targets that have a deterministic file-based rule location.
6. Before any Hermes restart or hot reload, detect active Hermes processes and active `SignalOps` work. If active work exists, defer runtime activation.

## Acceptance Signals

- `AGENT_OS_RULE_LOADING_PASS`
- `PRIOR_WORK_REUSE_PASS`
- `DAG_PRIMITIVES_PASS`
- `READY_FRONTIER_PASS`
- `CRITICAL_PATH_PASS`
- `DURABLE_RESUME_PASS`
- `ONE_WRITER_PASS`
- `LOCAL_REMOTE_PARITY_PASS`
- `TINY_TASK_OVERHEAD_ACCEPTABLE`
- `NO_DUPLICATE_WRITER_REGRESSION`
- `NO_STALE_AUTHORITY_REGRESSION`

## Stop Condition

Stop only when the remaining step is external-only or when a Hermes runtime
restart would interrupt an active `SignalOps` transaction. In that case,
preserve the continuation boundary and emit explicit defer markers.
