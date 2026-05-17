# MSIG Native Codex Kanban Lane Cleanup Plan

## Keep as active path

```text
Kanban card assigned to codex
  -> dispatcher directly spawns python -m hermes_cli.codex_worker
  -> native codex exec runs in assigned workspace/worktree
  -> worker streams logs + heartbeats
  -> worker blocks review-required or codex-failed with metadata
  -> Marshall verifies and advances/loops/escalates
```

## Mark legacy / remove from active workflow

- `msig-codex-worker` Hermes profile as the implementation route.
- `/home/ubuntu/.hermes/scripts/codex_kanban_bridge.py` as active execution bridge.
- tmux watchdog pattern as default execution path.
- GoalBuddy references for MSIG Codex execution.
- multi-profile MSIG builder/reviewer/browser-verifier split unless reintroduced for a specific proven need.

## Retain as fallback / evidence only

- tmux interactive Codex for emergency recovery or rare high-touch sessions.
- bridge script artifacts as historical smoke evidence until native lane is fully committed and accepted.
- Kanban dashboard/log/tail as the operational transparency layer.

## Follow-up after commit

- Patch `codex` skill to prefer native Kanban lane for MSIG cards.
- Patch/remove stale MSIG sections in `kanban-orchestrator` that mention bridge/profile wrapper as preferred.
- Update any MSIG runbooks to use `assignee=codex`.
- Archive old smoke cards on `msig-platform` once native lane low-risk MSIG smoke passes.
