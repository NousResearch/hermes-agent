# Managed-Agent Sandbox-Capable Workers Plan

Status: in_progress

## Goal
Move Hermes Kanban/managed-agent workers from informal process/workspace isolation toward an explicit sandbox-capable worker contract:

1. stateless worker workspaces with evidence;
2. explicit worker policy contracts;
3. checkpoint/fork/rollback primitives;
4. sandbox capability descriptors that do not overclaim OS/container isolation.

## Non-goals
- Do not build a full Docker/VM jail in this pass.
- Do not change production Hermes config.
- Do not weaken existing Kanban dispatcher behavior.

## Proposed design

### Data model
Add backward-compatible optional task fields:
- `worker_policy`: validated string, default `standard`.
- `checkpoint_policy`: validated string, default `auto` or `off` depending compatibility.

Add migration guards in `kanban_db._migrate_add_optional_columns`.

### Policy model
Define policy descriptors in Python constants/functions:
- `standard`: current behavior plus evidence.
- `read_only`: contract forbids file edits/destructive commands; injected in context/env.
- `code_edit`: permits code edits inside workspace only.
- `test_only`: permits build/test commands, discourage edits except temporary artifacts.
- `sandbox_strict`: requires strongest available workspace isolation; if no OS sandbox, metadata must say `os_sandbox=false`.

### Checkpoint/evidence
Add helper functions:
- `workspace_capabilities(task, workspace)`
- `create_workspace_checkpoint(task, workspace)`
- `collect_workspace_evidence(task, workspace)`

Implementation should be bounded:
- For git repo: capture `git rev-parse HEAD`, `git status --short`, `git diff --stat`, maybe patch path only if bounded.
- For non-git scratch: create a small manifest of file paths/sizes/mtimes capped by count/bytes.
- Store evidence as event payload and run metadata, not huge raw dumps.

### Dispatcher integration
At dispatch:
- Resolve workspace.
- Validate policy.
- Create pre-run checkpoint/evidence.
- Add policy/capability/checkpoint to run metadata/event.
- Inject env vars:
  - `HERMES_KANBAN_WORKER_POLICY`
  - `HERMES_KANBAN_CHECKPOINT_ID`
  - `HERMES_KANBAN_ISOLATION_CAPABILITIES` JSON

### Worker tools/context
- `kanban_show` and worker context should surface policy/capability/checkpoint.
- `kanban_complete` / `kanban_block` should include final workspace evidence in metadata where safe.

### CLI
Add create flags where minimal and backward compatible:
- `--worker-policy <standard|read_only|code_edit|test_only|sandbox_strict>`
- `--checkpoint-policy <off|auto|manifest|git>`

### Tests
Add focused tests with temp DB/workspace:
- default policy is standard.
- invalid policy rejected.
- dispatcher records checkpoint event/metadata.
- git workspace evidence captures status/diff without huge content.
- sandbox capability distinguishes `os_sandbox=false` for normal local workspace.

## File ownership
Single-branch implementation by Hermes/Codex worker to avoid overlap. Review can be self-review plus tests unless changes become large.
