# Statute PM operational runner proposal

## Objective

Make the DB-centric control plane operational enough that Galt/default can dispatch the next NJ statute wave through a real PM/worker loop, not a deterministic placeholder.

The implementation target is local-control-plane execution, not public network exposure and not a production sandbox. The DB remains the durable state/route/lease/approval ledger; the PM remains the orchestrator.

## Authority model

1. `default` / Galt is the continuously online supervisor.
2. `statutepm` is a finite PM profile. A PM instance is online only while a sprint/wave/parent dispatch is actively being supervised.
3. `statute-worker` instances are finite worker executions. A worker is online only while executing one dispatch.
4. Blocked/stalled/completed/failed finite instances are not online. The durable blocker/dispatch/result rows preserve state.
5. Dangerous actions are represented as DB approval/blocker objects routed to PM first or Galt/default when PM cannot authorize them.

## Dispatch model

1. Galt/default creates a parent dispatch to `statutepm` with a statute wave/sprint packet.
2. A fresh PM runner instance claims the parent dispatch.
3. The PM validates the parent payload and creates scoped worker/reviewer child dispatches.
4. Worker processes claim child dispatches and execute a real Hermes runtime profile in a fresh context derived from the child payload.
5. The PM polls child dispatches/results/blockers until completion, fixable blocker resolution, escalation, or timeout.
6. The PM records the parent result and marks itself offline in a `finally` path.

## Fresh context requirement

Every PM and worker run gets a new process/context seeded from explicit DB payload data:

- contract/repo/runtime paths,
- parent dispatch id,
- allowed paths,
- mandatory constraints,
- acceptance criteria,
- escalation policy,
- prior artifacts only when explicitly listed in the payload,
- approval/blocker routing expectations.

No previous compacted chat transcript is used as operative PM memory.

## Worker modes

### Deterministic mode

Keep deterministic mode for tests/smoke. It writes predictable artifacts and completes the dispatch.

### Agent mode

Add agent mode with lease fencing and a structured completion contract:

1. Claim dispatch and keep the `(dispatch_id, instance_id, lease_epoch)` fence for every DB write.
2. Mark running.
3. Build a fresh worker prompt from payload.
4. Launch a subprocess through an injectable runner. The test runner is fake/in-memory. The default runner uses this checkout's one-shot CLI path: `hermes -p <runtime-profile> chat --query <prompt>` or `hermes -p <runtime-profile> -z <prompt>` when suitable.
5. Use `--ignore-rules` unless the payload explicitly opts into profile/repo rules, so the operative context is DB-payload-derived. Do not pass `--resume` or `--continue`.
6. Set control-plane env for the subprocess: `HERMES_CONTROL_ROOT`, `HERMES_PROFILE_ID`, `HERMES_CONTROL_INSTANCE_ID`, `HERMES_CONTROL_DISPATCH_ID`, `HERMES_CONTROL_LEASE_EPOCH`, `HERMES_APPROVER_PROFILE`, and allowed-path metadata.
7. Do not use `--yolo`. Do not rely on prompt text for safety. Dangerous commands still pass through Hermes' approval system, which already writes DB approvals using the control env. `--accept-hooks` is hook-only and must not be used as a blanket dangerous-command bypass.
8. While the subprocess runs, periodically heartbeat the worker instance and extend the dispatch lease under the same epoch.
9. On timeout, terminate the process group, wait, then kill if needed before marking the dispatch failed/action-required.
10. Capture stdout/stderr/returncode into a private control artifact file under `.hermes-control/` or the configured control root. Store redacted DB summaries; filesystem artifacts are chmod `0600`.
11. Require a structured result contract. The subprocess output must contain a parseable `CONTROL_RESULT_JSON` block or write an agreed result file. Exit code 0 alone is not completion.
12. Validate result status, required artifacts, tests, blockers, and acceptance evidence before completing the dispatch.
13. On nonzero/timeout/exception/malformed result, record a failed result and runtime blocker evidence.
14. Always mark the worker instance offline in `finally`.

The first production-quality implementation must use dependency injection so tests do not spawn a live model.

## PM liveness implementation

Replace bootstrap PM liveness assumptions with finite run instances:

- Add PM instance id generation: `statutepm:<parent_dispatch_id>:<uuid>` when not explicitly provided.
- Add `--fresh-instance`/default finite instance behavior for `hermes control pm run`.
- `StatutePMFlow.run_once()` registers/heartbeats the PM instance only for active work.
- If no dispatch is claimed, the PM marks itself offline and exits cleanly.
- During child polling, the PM periodically heartbeats its instance and extends the parent dispatch lease under the same epoch.
- On completion/failure/block/timeout, the PM records the parent result and marks itself offline.
- Readiness/doctor must stop treating `statutepm:bootstrap` as required online liveness.

## Child dispatch idempotency/recovery

PM-created child/follow-up dispatches need stable idempotency keys:

- key shape: `pm-child:<parent_dispatch_id>:<role>:<ordinal-or-task-id>:<attempt>`;
- retries/fix attempts get explicit attempt suffixes;
- on PM restart, the PM first looks for existing child dispatches for the parent/key and resumes monitoring instead of creating duplicates;
- duplicate parent dispatches are not silently accepted as readiness.

## Blocker and escalation implementation

1. Worker runtime failures become child dispatch failures plus structured blocker/result evidence.
2. PM acceptance loop:
   - PM validates child result contract and acceptance evidence before treating the child as complete.
   - Missing required artifacts/tests/result fields are a blocker, not a successful child completion.
3. PM fix loop:
   - If child result has blockers marked `fixable_by_pm`, PM may create bounded follow-up dispatches or retry once inside configured limits.
   - If blocker requires dangerous permission, missing policy, external spend, deletion, public exposure, dependency install, secret/auth, or human/business judgment, PM opens a blocker for `default` and records an action-required parent result.
4. Approval flow:
   - Use `cp.create_approval()` with requester bound to instance, dispatch, and lease epoch.
   - Approvals must be consumed by the same requester/dispatch/lease before action proceeds.
   - PM may approve only concrete low-risk classes defined in a control policy helper. Anything unknown/high-risk routes to `default`.
5. Reviewer support is deferred for this MVP unless represented as an ordinary worker dispatch. Do not claim a distinct reviewer runner is ready until implemented/tested.

## CLI changes

- `control pm run`:
  - default finite fresh instance id when none supplied;
  - optional `--worker-handler deterministic|agent`;
  - optional `--max-blocker-fix-attempts`;
  - mark PM offline in all terminal paths.

- `control worker run`:
  - `--handler agent` actually executes agent mode;
  - optional `--runtime-profile` override;
  - optional timeout/log artifact options.

## Tests first

Add failing tests before implementation:

1. PM finite liveness: PM registers online while claiming/running and is offline after completion.
2. PM no-dispatch liveness: fresh PM instance does not remain online when no dispatch is claimed.
3. PM creates child dispatch and records parent result from child result.
4. Worker agent mode builds fresh prompt from payload, invokes injected runner, records stdout artifact, completes dispatch, and marks offline.
5. Worker agent nonzero exit records failed result/blocker evidence and marks offline.
6. Dangerous permission/approval helper binds approvals to requester instance, dispatch, and lease epoch; wrong requester cannot consume.
7. Readiness/doctor does not require `statutepm:bootstrap` as live PM liveness.

## Review loop

Before code execution, run independent fresh-context review of this proposal for MVP-critical missing mechanisms, safety gaps, and false readiness claims. Incorporate blocking findings and rerun if material changes occur.

## Verification

Minimum before reporting ready:

```bash
python -m pytest tests/hermes_cli/test_control_db.py tests/hermes_cli/test_control_worker.py tests/hermes_cli/test_statutepm_flow.py tests/hermes_cli/test_control_cli.py tests/hermes_cli/test_control_smoke.py -q -o 'addopts='
```

Then run a non-live end-to-end smoke:

1. temp control root;
2. bootstrap policies/profiles/routes;
3. create parent dispatch to `statutepm`;
4. run PM once with agent worker mode using fake/injected runner in tests or deterministic subprocess smoke;
5. verify parent completed or action-required with durable result;
6. verify PM and worker finite instances offline;
7. verify artifacts/results/blockers/approvals are queryable.

## Out of scope for this implementation

- Public network exposure.
- Restarting Hermes gateways.
- Deleting unrelated dirty files.
- Pushing code.
- Full hostile-code sandboxing. This control plane runs trusted local Hermes worker profiles under existing approval controls.
