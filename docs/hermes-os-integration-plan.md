# Hermes OS + Official Hermes Agent Integration Plan

## Direction

Hermes OS remains the control plane and source of truth. Official Hermes Agent is integrated as an optional execution/runtime layer underneath Hermes OS.

## Guardrails

- Do not replace Hermes OS.
- Do not migrate existing projects.
- Do not overwrite the `hermes` command.
- Use `hermes-agent` or equivalent for the official runtime.
- Do not treat runtime memory as authoritative project/task/report storage.

## Command Ownership

- `hermes` belongs to Hermes OS.
- `hermes-agent` belongs to the official Hermes Agent runtime in this repository.
- Shell setup must not alias `hermes` to this repository.
- If `hermes` resolves unexpectedly, check `PATH`, shell aliases, and any symlink under `~/.local/bin`.
- If `hermes-agent` is unavailable, Hermes OS must degrade to direct execution or mark delegation unavailable.

## Runtime Adapter Contract

Hermes OS sends an `AgentRequest`:

- `task_id`
- `project_id`
- `agent_kind`
- `prompt`
- `working_directory`
- `context`
- `tool_policy`
- `runtime_provider`
- `timeout_seconds`
- `dry_run`

Hermes Agent returns an `AgentResponse`:

- `task_id`
- `status`
- `output`
- `artifacts`
- `errors`
- `duration_ms`
- `cost`
- `stdout`
- `stderr`
- `exit_code`

All adapter payloads are validated by `hermes_os_integration.contracts` before Hermes OS stores results.

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

Delegation flow:

```text
Hermes OS task
  -> task type to agent kind mapping
  -> validated AgentRequest
  -> RuntimeWrapper
  -> validated AgentResponse
  -> Hermes OS result storage
```

Retries and escalation are owned by Hermes OS. Runtime memory is never the final record.

### Phase 6 - Memory Boundary

- `task-012`: Define memory boundary.
- `task-013`: Add memory sync guardrails.

Allowed runtime memory:

- Cache.
- Execution-local notes.
- Temporary context.

Disallowed runtime memory:

- Project source of truth.
- Task status source of truth.
- Final report storage.
- Review decision storage.

### Phase 7 - MCP Integration

- `task-014`: Design MCP permission bridge.
- `task-015`: Build MCP adapter prototype.

Hermes OS grants tool permissions. Official runtime consumes only delegated tool capabilities.

Supported MCP categories:

- GitHub.
- Discord.
- Browser.
- Filesystem.
- Documentation.
- Research.
- Kalshi.
- Broker integrations.

### Phase 8 - Dashboard

- `task-016`: Design dashboard runtime status panels.
- `task-017`: Add runtime health endpoint contract.

Dashboard panels should show:

- Runtime status: running, stopped, unavailable.
- Provider, latency, version, and recent errors.
- Agent usage: tasks executed, success rate, retries, cost, tokens.
- Delegation status: active tasks, queue length, completion time.

Dashboard must continue to work if the runtime is unavailable.

### Phase 9 - Long-Running Workflows

- `task-018`: Design long-running workflow orchestration.
- `task-019`: Build checkpointed workflow prototype.

Workflow pattern:

```text
Research -> Validation -> Review -> Report
```

Hermes OS stores checkpoints after each step and can resume from the latest checkpoint.

### Phase 10 - Kalshi Research

- `task-020`: Define Kalshi research agent architecture.

Kalshi target architecture:

```text
Bucket
  -> Research Agent
  -> Evidence Agent
  -> Validation Agent
  -> Portfolio Agent
  -> Experiment Tracker
  -> Dashboard
```

### Validation

- `task-021`: Create integration smoke tests.
- `task-022`: Create integration rollout plan.

Rollout:

1. Dry-run delegation only.
2. Enable one low-risk project.
3. Enable one agent kind at a time.
4. Monitor runtime health and validation failures.
5. Roll back by disabling runtime delegation.
