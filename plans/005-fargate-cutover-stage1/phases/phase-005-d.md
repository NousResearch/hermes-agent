# Phase 005-D: Flip Fargate `desiredCount=1`; verify cold-start

## Goal
Bring up exactly one Fargate task running the v8 image, observe a clean startup in CloudWatch, hard-block on any `signature expired` or DNS error indicating the same pathology that plagues local Hermes.

## Context
This is the cutover moment. Cloud Hermes goes from 0 → 1 task. Local launchd Hermes STAYS UP until Phase 005-F — for the duration of Phase D + E, you have two Slack listeners on the same workspace. Expected; that's the test bed.

## Dependencies
- Phases 005-B + 005-C complete

## Scope

### Files to Create
None.

### Files to Modify
- ECS service `hermes` in cluster `agentic-stack` — change `desiredCount` from 0 → 1; force-new-deployment to pick up the v8 task def revision

### Explicitly Out of Scope
- Adding additional tasks beyond 1 (Stage 1 is single-task)
- ALB / target group changes
- Auto-scaling rules
- Spot Fargate / Fargate-Spot — defer to cost optimization phase

## Implementation Notes

1. **First register a new task def revision** with the v8 image — DO NOT just `update-service` against the existing revision. Even if Phase B pushed v8 to ECR, the task def may still reference v7 if `:plan-001-E-amd64-v8` wasn't used in the task def's `image` field.
2. **Use `--force-new-deployment`** with `update-service` so even if task def revision didn't change, ECS still pulls + recycles.
3. **Wait for `services-stable`**, then immediately tail CloudWatch logs.
4. **Hard gate on startup log lines.** If you don't see all 4 expected lines within 90s of RUNNING, kill the experiment with `desiredCount=0` and debug from CloudWatch.

## Acceptance Criteria
- [ ] New task definition revision created with `image: agentic-stack/hermes:plan-001-E-amd64-v8` (record the revision number)
- [ ] `aws ecs update-service --cluster agentic-stack --service hermes --task-definition hermes-saas:<rev> --desired-count 1 --force-new-deployment` succeeds
- [ ] `aws ecs wait services-stable` returns within 5 minutes
- [ ] CloudWatch log group `/ecs/hermes-saas` shows all 4 startup lines in order:
  - `Storage backend initialized: NeonBackend (pool=ready)`
  - `[Slack] Authenticated as @bossman2 in workspace hermes (team: T0B16FV0KFF)`
  - `[Slack] Plan 004-A tenant bootstrap: team=T0B16FV0KFF → ac85d33a-c466-4d4c-9747-0a8d69efbe6f`
  - `[Slack] Socket Mode connected (1 workspace(s))`
- [ ] Target group health check at `:8080/health` returns 200 within 90s of task RUNNING
- [ ] Zero `signature expired | DNS resolution | TimeoutError` lines in the first 10 minutes of logs
- [ ] Task RSS < 500 MB after 5-minute warmup (per memory_monitor heartbeat in CloudWatch)

## Verification Steps

```bash
PROFILE='AgenticHub-162471567408'
CLUSTER='agentic-stack'
SERVICE='hermes'

# 1. Register new task def revision (if task def's image field isn't already v8)
# Get current task def + replace image:
CURRENT_TD=$(aws ecs describe-task-definition --task-definition hermes-saas \
  --profile $PROFILE --query 'taskDefinition' --output json)
# ... edit image field, then:
# aws ecs register-task-definition --cli-input-json "<edited>" --profile $PROFILE

# 2. Flip the service
aws ecs update-service --cluster $CLUSTER --service $SERVICE \
  --task-definition hermes-saas:<NEW_REV> --desired-count 1 \
  --force-new-deployment --profile $PROFILE

# 3. Wait + tail
aws ecs wait services-stable --cluster $CLUSTER --services $SERVICE --profile $PROFILE
aws logs tail /ecs/hermes-saas --follow --since 5m --profile $PROFILE
```

## Status
Complete — 2026-05-26

### Adaptations — surfaced a 5-day-pre-existing bug in cloud saas-mode startup

1. **Misassumption corrected**: Plan 005 master plan claimed Fargate was at `desiredCount=0`. Reality: cloud Hermes has been at `desiredCount=1, runningCount=1` since 2026-05-20 — but **silently broken**. ECS health check passed because the entrypoint's port-8080 health server child process kept passing; the actual gateway process never logged past the "No user allowlists configured" warning. Zero "Storage backend initialized", zero "Socket Mode connected", zero "inbound message" lines in 5 days of supposedly-running logs.

2. **Root cause found**: `gateway/run.py:17970` only attaches the stderr log handler when CLI verbosity is non-null. Default `hermes gateway run` has verbosity=None → no handler installed → INFO logs go nowhere. The WARNING at line 3768 only appeared because Python's default warning machinery bypasses the handler chain.

3. **Fix without image rebuild**: overrode task def CMD to `["gateway", "run", "-v"]`. Task def revisions: `:9` (v8 image only) → `:10` (added `PYTHONUNBUFFERED=1`) → `:11` (added `-v` flag). td:11 is the working revision.

4. **The local laptop Hermes was never broken because** launchd's plist passes verbosity differently OR was started without `-v` but the local TTY-attached logging behaves differently. Either way, local has been logging fine; only cloud was silent.

5. **Follow-up: bake `-v` into `entrypoint.saas.sh`** (no image rebuild needed for current run; permanent fix is a small edit + v9 image). Track in Plan 005-G follow-ups.

6. **Follow-up: kanban dispatcher is `embedded` in cloud config** (`INFO gateway.run: kanban dispatcher: embedded in gateway (interval=60.0s)`). Local has `dispatch_in_gateway=false`. This is the OOM-risk path. Cloud may need to set `kanban.dispatch_in_gateway=false` in its config (via env var or config file) before Plan 005-G soak starts. **High-priority Phase 005-G prerequisite.**

7. **All Plan 004-A + Plan 007 startup lines confirmed active in cloud** — tenant bootstrap fired, NeonBackend pool is ready, Slack Socket Mode connected.
