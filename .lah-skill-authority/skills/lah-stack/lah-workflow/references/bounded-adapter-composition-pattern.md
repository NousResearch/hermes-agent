# Bounded Adapter Composition Pattern

Established during `LAH_CODEX_BOUNDED_TASK_EXECUTION_ADAPTER_IMPLEMENTATION_V1`.
Real-Codex integration proven during `LAH_CODEX_BOUNDED_ADAPTER_LIVE_SMOKE_V1`.

## Architecture Verdict

`COMPOSE_WITH_ONE_MINIMAL_NEW_COMPONENT` — compose 12+ existing modules with 2 new thin modules and 1 schema file.

## Role-to-Existing-Component Mapping

| Role | Existing Component | Path |
|------|-------------------|------|
| Codex Invocation | `real-executor-codex` | `tools/real-executor-codex/index.mjs` (37 exports) |
| State Machine | `batch-runner state` | `tools/batch-runner/src/state/state-machine.mjs` (9 states) |
| Checkpoints | `batch-runner checkpoint` | `tools/batch-runner/src/run/checkpoint.mjs` (4 types) |
| Continuation | `batch-runner continuation` | `tools/batch-runner/src/run/continuation.mjs` |
| Resume | `batch-runner resume-run` | `tools/batch-runner/src/run/resume-run.mjs` |
| Transaction | `git-workspace run-transaction` | `tools/git-workspace/run-transaction.mjs` (begin/finalize) |
| Git Policy | `lah-git-policy` | `tools/git-policy/lah-git-policy.mjs` (22 exports) |
| Scope Enforcement | `dirtiness-classifier` | `tools/git-policy/dirtiness-classifier.mjs` |
| Baseline Capture | `baseline-capture` | `tools/git-policy/baseline-capture.mjs` |
| State Stores | `temporal-patterns/persistence` | `tools/temporal-patterns/persistence/factory.mjs` (6 stores) |
| Batch Manifest | `batch-manifest.schema.json` | `tools/batch-runner/schemas/batch-manifest.schema.json` |
| Safety Guards | `lah-guard.mjs` | `lah-guard.mjs` |

## New Modules (minimal)

1. **Bounded Packet Compiler** — validates roadmap, selects one task, compiles bounded packet with deterministic SHA256 hash, idempotency key
2. **Transition Controller** — orchestrates the loop: validate roadmap -> select task -> compile packet -> invoke Codex -> parse output -> enforce scope -> validate receipt -> transition or STOP

## Key Invariants

- Controller invokes Codex at most once per task
- Controller owns the final transition verdict (not Codex)
- Only TASK_PASS authorizes advancement
- Out-of-scope file changes override Codex TASK_PASS
- Roadmap revision mismatch blocks execution
- Completed tasks cannot reopen
- REPLAN_REQUIRED stops without mutating roadmap

## parseCodexJsonlOutput Caveat

`parseCodexJsonlOutput()` from `real-executor-codex` only understands Codex-native event types (`thread.started`, `turn.completed`, `item.completed`, `tool.use`, `tool.result`, `error`). It does NOT accept arbitrary JSON with `verdict`, `files_modified`, or `tests` fields.

For bounded adapter output, parse stdout directly:
1. Try `JSON.parse(stdout)` (single object)
2. Fall back to JSONL: iterate lines, check `evt.type`
3. Extract: `parsed.verdict`, `parsed.files_modified`, `parsed.tests`, `parsed.replan_reason`

## Real Codex Integration — Interface Mismatch

### Problem

The bounded adapter's `runBoundedCodexTask` calls the executor with:
```
executeCodex({ packet, timeoutMs, mission_id, task_id })
```

But `executeCodexWithTimeout` expects:
```
executeCodexWithTimeout(prompt_string, options)
```

The default path receives the packet object as `prompt`, producing `[object Object]`.

### Solution: Codex Prompt Bridge

Create a bridge module that converts packet to prompt:

```
adapter calls executeCodex({ packet, timeoutMs, ... })
  -> bridge.buildPromptFromPacket(packet) -> string prompt
  -> executeCodexWithTimeout(prompt, { cd, sandbox, ... })
  -> return { exitCode, stdout, stderr }
```

The bridge (`codex-prompt-bridge.mjs`) must be injected via `deps.executeCodex`. It is NOT the default executor.

### Pitfall — Bridge must match the adapter's deps shape

The adapter expects `{ exitCode, stdout, stderr }`. The real executor returns `{ events, resultText, tokenUsage, ... }`. The bridge must map:
```javascript
return {
  exitCode: result.exitCode ?? -1,
  stdout: result.resultText || JSON.stringify(result.events),
  stderr: result.stderr || '',
};
```

## Required Commands Runner Pattern

### Problem

After Codex returns exitCode=0, the adapter sets `TASK_PASS`. But the receipt schema requires at least one test result for TASK_PASS. Real Codex output is natural language or JSONL events — NOT structured JSON with `tests: [...]`. CodexTests stays empty and receipt validation fails with `INVALID_RECEIPT`.

### Solution

After verdict determination, if TASK_PASS but tests empty, run the packet's required_commands directly:

```javascript
if (finalVerdict === 'TASK_PASS' && (!codexTests || codexTests.length === 0)) {
  for (const cmd of packet.required_commands || []) {
    try { execSync(cmd, { cwd: workspace_path, timeout: 60000 });
      codexTests.push({ name: cmd, passed: true });
    } catch {
      codexTests.push({ name: cmd, passed: false });
    }
  }
}
```

Each required command becomes a test evidence entry. This populates `codexTests` so receipt validation passes.

### Test Implication

Test 23 (`controller: malformed output produces INVALID_RECEIPT`) changes behavior: with exitCode=0, the runner now populates tests from the packet's required_commands. Expected verdict changes from `INVALID_RECEIPT` to `TASK_PASS`. Update the test.

## Workspace Existence Validation

### Problem

When `workspace_path` points to a non-existent directory, Codex fails with an unclear error or produces invalid output.

### Solution

After packet compilation and before Codex invocation, verify workspace exists:

```javascript
if (!workspacePath || !existsSync(workspacePath) || !statSync(workspacePath).isDirectory()) {
  // Return TASK_BLOCKED with clear error message
}
```

This stops execution before any Codex invocation. Requires importing `statSync` from `node:fs`.

## Live Smoke Protocol

When running a live smoke test (real Hermes -> bounded adapter -> Codex CLI):

1. Create isolated worktree from clean commit
2. Build locked roadmap with one real task (1-2 files, low risk)
3. Inject the Codex prompt bridge via `deps.executeCodex`
4. Run `runBoundedCodexTask(roadmap, { executeCodex: bridge, runDir })`
5. Expect TASK_PASS with test evidence from required_commands runner
6. Only attempt 2 smoke executions max (1 repair cycle)
7. Verify: scope integrity, no external mutations, roadmap not modified
8. Run adapter tests afterwards to confirm no regression
