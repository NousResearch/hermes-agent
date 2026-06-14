# Hermes OS + Official Hermes Agent Integration Plan

## Direction

Hermes OS remains the control plane and source of truth. Official Hermes Agent is integrated as an optional execution/runtime layer underneath Hermes OS.

## Guardrails

- Do not replace Hermes OS.
- Do not migrate existing projects.
- Do not overwrite the `hermes` command.
- Use `hermes-agent` or equivalent for the official runtime.
- Do not treat runtime memory as authoritative project/task/report storage.

## Task Batches

### Phase 1 - Standalone Runtime

- `task-001`: Verify standalone Hermes Agent runtime.
- `task-002`: Create `hermes-agent` launcher.
- `task-003`: Document command ownership.

### Phase 2 - Runtime Wrapper

- `task-004`: Define runtime adapter contract.
- `task-005`: Build runtime wrapper prototype.

### Phase 3 - Contracts

- `task-006`: Add validated integration schemas.
- `task-007`: Add adapter error taxonomy.

### Phase 4 - Agent Registry

- `task-008`: Create official-agent registry.
- `task-009`: Map Hermes OS tasks to agent kinds.

### Phase 5 - Delegation

- `task-010`: Design delegation engine.
- `task-011`: Build delegation prototype.

### Phase 6 - Memory Boundary

- `task-012`: Define memory boundary.
- `task-013`: Add memory sync guardrails.

### Phase 7 - MCP Integration

- `task-014`: Design MCP permission bridge.
- `task-015`: Build MCP adapter prototype.

### Phase 8 - Dashboard

- `task-016`: Design dashboard runtime status panels.
- `task-017`: Add runtime health endpoint contract.

### Phase 9 - Long-Running Workflows

- `task-018`: Design long-running workflow orchestration.
- `task-019`: Build checkpointed workflow prototype.

### Phase 10 - Kalshi Research

- `task-020`: Define Kalshi research agent architecture.

### Validation

- `task-021`: Create integration smoke tests.
- `task-022`: Create integration rollout plan.
