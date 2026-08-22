# Bounded Codex Progression Wiring + Real Codex Certification

Established during HERMES_CODEX_MULTI_LOT_AUTONOMOUS_EXECUTION_WIRING_V1
(PR #104 → master d9b25a1a, lah-stack-tools). Class of mission: compose an
already-certified bounded adapter into the progression runtime with MINIMAL
wiring, then certify end-to-end against the REAL Codex CLI.

## The minimal wiring pattern (no new orchestrator)

The runtime already had a DI seam: `modules.codexExecutor` imported at
`lah-progression-runtime.mjs` l.448, called as `executeWithGovernance(prompt,
context, options)` at l.859. The bounded adapter exports a DIFFERENT contract:
`runBoundedCodexTask(roadmap, deps, opts)`. Do NOT touch runtime call sites —
write a thin bridge file exposing the runtime's interface and delegating:

```text
Progression Runtime (executeWithGovernance call site — unchanged)
  → tools/progression-runtime/bounded-codex-executor.mjs   (NEW bridge)
  → tools/codex-bounded-adapter/index.mjs runBoundedCodexTask
  → real Codex CLI (or fake via opts.bounded_execute_codex)
```

Bridge responsibilities (all proven):
- `buildBoundedRoadmap(program, mission, repoRoot)` → ONE task per call,
  status LOCKED, `dependencies: []`.
- `computeRoadmapRevision(program)` → deterministic sha256 of stable-json;
  revision mismatch → TASK_BLOCKED before any Codex call.
- Map bounded verdicts to the runtime's expectations (ok:true only for
  TASK_PASS; everything else fail-closed).

**Why `dependencies: []`:** the adapter's `selectCurrentTask` validates
intra-roadmap dependencies against its own task list. Copying the mission's
`depends_on` into the isolated single-task roadmap makes the adapter fail with
"dependency not found". Dependencies are the RUNTIME's authority (it only
invokes READY missions whose depends_on are all PROMOTED) — never duplicate
them into the bounded task.

**Activation flag:** `boundedCodexExecution` default false in the runtime +
`--bounded-codex` CLI flag in the launcher. Persist it in
orchestrator-state.json at run start so `--resume` keeps the same mode
(`boundedCodexExecution: options.boundedCodexExecution === true ||
prevState.bounded_codex_execution === true`). Default false preserves the
legacy `safe-local-executor` path (prove with a LEGACY test).

**Promotion evidence (invariant 12):** Step H of the runtime builds
`evidence.tests`/`evidence.receipts`. In bounded mode, surface the REAL
bounded receipt (verdict, tests, valid flag) instead of the hardcoded
preview-mode `{passed:1, total:1}` — then a TASK_PASS with failing or missing
required_commands evidence cannot ride on a bare executor self-report. This
was observed live: lot_b returned TASK_PASS but `node --test` output was
absent → promotion BLOCKED by the promotion controller.

## Real Codex certification harness (codex CLI v0.146.1, auth chatgpt)

Use a disposable fixture worktree as `workspace_path` (git-free is fine),
never a production file. Program = 3 lots with `depends_on` chain. Drive via
`runProgram(programPath, {boundedCodexExecution:true, enableRealExecution:true,
previewOnly:false})` with moduleOverrides for ledger/bridge/promotion/
controlledRun/verificationGate/jobQueue (same doubles as the wiring tests).

### Pitfall 1 — repository_path vs workspace_path divergence
When a mission declares `workspace_path`, the packet's `repository_path` MUST
follow it (`repository_path: mission.workspace_path || repoRoot`). If
repository_path points at the parent repo while workspace_path points at the
fixture, Codex sees "a real repo + a workspace", correctly abstains from
writing, returns TASK_PASS with `files_modified: []`, and the required_command
fails ("Could not find 'test/...'"). The isolated surface IS the bounded
execution boundary — the prompt must not point at the parent repo.

### Pitfall 2 — Codex prompt must be directive
"Create test/math.test.mjs with node:test tests" → Codex answers with prose,
TASK_PASS, no file written. Use: "You MUST actually write the file X (use the
file edit tool) ... Then run: <cmd> and confirm it passes." Verified: same
packet with directive wording → file created, tests pass.

### Pitfall 3 — TASK_PASS requires ≥1 test result
Adapter receipt validation rejects TASK_PASS with zero tests
(`INVALID_RECEIPT`). Missions that cannot produce test evidence (DESIGN/AUDIT)
must still carry a non-empty required_command (e.g. `node --version`) so the
receipt has one passed test. Empty required_commands → INVALID_RECEIPT, which
is fail-closed but blocks an otherwise-fine DESIGN lot.

### Pitfall 4 — `node --test test/` fails on Node 22
`node --test --test-concurrency=1 test/` → "Cannot find module
'.../test'" (the directory arg is treated as a module on Node 22.22). Use an
explicit file glob: `node --test --test-concurrency=1 test/math.test.mjs`.
This bit the certification program contract, not the wiring — fix the
required_command, not the code.

### Pitfall 5 — Simulated interruption: throw in runPromotionCycle, NOT in the executor
To certify resume: throw inside `runPromotionCycle` AFTER the promote
(leaves next mission ENQUEUED, not BLOCKED) → durable state → `handleResume`
starts at B without re-running A. Throwing inside the Codex executor produces
EXECUTOR_FAILURE → BLOCK (terminal, not resumable) and the resume test
fails with PROGRESSION_BLOCKED.

### Pitfall 6 — Planner missions lack workspace_path/allowed_files
The strategic planner emits missions without bounded fields. Driving the
one-command entrypoint (`runAutonomous`) with real Codex would fall back to
repoRoot (the dev worktree — forbidden). For entrypoint certification,
enrich the generated program artifact (data transformation, not a new
entrypoint): add workspace_path → disposable fixture, allowed_files, and
required_commands BY MISSION TYPE (DESIGN/AUDIT → `node --version`; IMPLEMENTATION
→ check the artifact; VALIDATION/ACCEPTANCE_TRIAL → run the tests). The
launcher also requires transaction isolation: create `release-manifest.json`
with `baseline_sha` in the repository path or pass baselineSha explicitly.

## Test suite shape (12 wiring tests)

nominal (A→PASS→B READY→PROGRAM_COMPLETE, one Codex call per task, order
asserted), task-failure (B never starts), scope-violation (out-of-scope write →
TASK_SCOPE_VIOLATION receipt file on disk), false-pass (TASK_PASS + out-of-scope
files → rejected), revision-mismatch (0 Codex calls), replan (roadmap file
byte-unchanged), resume (A promoted → interrupt in promotion cycle → B+C on
resume, A not re-run), dependency (C before B promotion → never starts),
one-task-authority (roadmap has exactly 1 task), workspace-path (repository_path
follows workspace_path; fallback repoRoot), revision determinism, legacy
(default flag → old executor called).

Use `makeFakeCodex(script)` with calls recording `{mission_id, task_id,
packet_sha: packet.provenance.packet_hash}` — packet hash lives in
`provenance.packet_hash`, not at the packet root.

## Fail-closed observations worth keeping

- Promotion blocked when test evidence absent — PROOF, not a bug; keep it.
- INVALID_RECEIPT on empty tests — same family (adapter's receipt schema).
- Certification reports + bounded receipts per lot → durable evidence under
  `test/reports/<mission>/` (commit them; they are the mission's receipts).

## Rollback (both trivial)

1. Flag off (default) → legacy safe-local-executor path; or
2. Revert the single squash merge commit.
