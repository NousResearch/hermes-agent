# DWF worker lanes (M4 — no Kanban)

> **Runtime:** Hermes dynamic workflows (`delegate_task`, file-backed receipts, worktrees).  
> **Not used:** Hermes Kanban board worker spawn (profiles `kanban-*` remain inventory-only until you opt in).

## Functional lane IDs

| lane_id | host_profile | model (policy) | delegate role | toolsets | mutation | notes |
|---------|--------------|----------------|---------------|----------|----------|-------|
| `dwf-parent` | auto-coder | grok-composer-2.5-fast | orchestrator | per parent session | worktree edits; push gated | Single interactive owner |
| `dwf-research` | auto-coder | grok-composer-2.5-fast | leaf | web, search, file (read) | none | Receipts under `/tmp/hermes-workflows/` |
| `dwf-coder` | auto-coder | grok-composer-2.5-fast | leaf | terminal, file | worktree only | Isolated branch per slice |
| `dwf-verifier` | auto-coder | grok-composer-2.5-fast | leaf | file, terminal (read) | none | Adversarial check of worker claims |
| `dwf-setup-audit` | auto-coder | n/a | script | `setup_gate_ladder_audit.py` | none | M2a/M2b pattern |
| `dwf-routing-smoke` | auto-coder | n/a | script | `routing_smoke.py --no-network` | none | M3 classifier only |

## Kanban profiles (frozen)

`kanban-impl`, `kanban-triage`, `kanban-review`, `kanban-pr`, `kanban-merge-steward`, `kanban-ops-steward` — **do not dispatch** under current operator policy. Listed in `worker-registry.md` for credential hygiene only.

## Preflight before `delegate_task`

```bash
# Parent always auto-coder + Grok (this session)
hermes profile list | grep -E 'auto-coder|grok'
test -f "$HOME/.hermes/profiles/auto-coder/.env" && echo env_ok
```

Worker prompts must state: **no Kanban**, **no push**, **Grok coding policy**, absolute paths, output schema.

## Receipt contract (per DWF run)

```text
/tmp/hermes-workflows/<name>/<run-id>/
  workflow-spec.md
  workers/<task-id>.md
  verifiers/<task-id>.md
  synthesis.md
  receipt.json
```

## Gateway / cron (unchanged)

- Gateway owner: **auto-coder** only (`gateway-minimal` stopped).
- Cron host: **auto-coder** (3 jobs per M2b); cron mutations approval-gated.