# Plan 005 — Hermes Fargate Cutover (Stage 1: Gateway in cloud, MCP still local)

**Status:** DRAFT 2026-05-23 (prereqs amended 2026-05-25)
**Run after:**
  - 001-E (Fargate scaffolding shipped 2026-05-20; ECR image `agentic-stack/hermes:plan-001-E` @ sha256:23a9a91 already present)
  - PR #13 (`feat/plan-004-self-improvement`) merged to main
  - **Plan 007 (Sessions to Neon) — HARD BLOCKER** added 2026-05-25. Cloud Fargate has no persistent disk; without Neon-backed sessions, every task restart wipes conversation history (a regression vs. today's local SQLite). 007 must ship and pass UAT before 005 starts.
**Blocked by:** Plan 007 (verify `HERMES_MODE=saas` routes session writes to Neon; verify gateway restart preserves an in-flight session's transcript)
**Followed by:** Plan 008 (MCPGateway production wiring) — Plan 005 ships gateway WITHOUT MCPs; 008 lands MCPs as a Fargate sidecar after 005's 48h soak passes
**Estimated effort:** 1 working day end-to-end (4–6 hours active, ~2 hours waiting on AWS + Docker + Slack soak)

## Why this plan exists

Local launchd Hermes on Blake's MacBook keeps hitting a recurring failure class that has nothing to do with Hermes itself:

1. **Clock skew after sleep/wake** → AWS Bedrock SigV4 returns 403 "Signature expired". On 2026-05-22 this killed the `feedback-triage-nightly` cron at 03:44 UTC and caused mid-conversation silent failures (logged in `~/.hermes/logs/gateway.error.log`).
2. **DNS resolution drops** when laptop changes networks → `Cannot connect to host slack.com:443 ssl:default [nodename nor servname provided, or not known]`.
3. **Sleep-triggered websocket zombies** → TCP socket dead but app-layer reader hung; bot appears responsive but goes silent mid-reply.
4. **kanban worker fan-out** OOM'd the laptop on 2026-05-21 (already fixed by disabling the embedded dispatcher in Plan-004 prep work; root architectural fix tracked as Hermes Plan 005-A *separately*).

All four go away when Hermes runs in Fargate: AWS-managed NTP keeps the clock tight, VPC DNS is reliable, container restarts are deterministic, and Fargate's 8 GB memory ceiling forces honest resource accounting.

**This is Stage 1 of a two-stage migration.** Stage 1 moves only the gateway. MCP servers (pipedrive, linear, exa, google_workspace, atlas.mcp) stay local-callable for now; the gateway will simply run without them in cloud mode until Stage 2 (Plan 006 candidate — "MCP cloud landing zone") moves them behind the standalone Skills Service.

## Plan context

What's already built (no new infra work needed for these):

| Component | State as of 2026-05-23 |
|---|---|
| ECR repo `agentic-stack/hermes` | exists; `plan-001-E` image tagged + pushed |
| ECS cluster `agentic-stack` | exists, `hermes` service registered at `desiredCount=0` |
| Task definition `hermes-saas` | shipped in Plan 001-E |
| `infra/terraform/hermes-fargate/` | applied (per Plan 001-E PROGRESS row) |
| `agentic-stack/neon/hermes-saas` secret | exists; contains DSN for `hermes_app` role |
| `HERMES_MODE=saas` → NeonBackend wiring | working (verified 2026-05-22 — tenant bootstrap fires + writes to Neon) |
| `Dockerfile.saas` | shipped Plan 001-E; multi-stage, stateless |
| `gateway/health_server.py` | aiohttp `:8080`, used by Fargate target group |
| Two unmerged commits on `feat/plan-004-self-improvement` | `1828d0d` (tenant bootstrap) + `369d83f` (await get_backend at startup) — both MUST be in the v8 image |

What is NOT in scope here (deliberate):

- Moving MCP servers to cloud → Plan 006 (separate)
- Removing local laptop Hermes entirely → happens at end of Stage 1, but only after cloud Hermes is proven for ≥48h
- Slack thread-routing fix → orthogonal; Plan 004 territory
- Fargate auto-scaling → desired=1 always for now; scale-up is Stage 3
- Cost optimization → defer until after 30-day soak

---

## Phase Index

| Phase | Title | Effort | Risk | Priority | Status |
|---|---|---|---|---|---|
| 005-A | Land + verify PR #13 on main | 30 min | Low | P0 | Not started |
| 005-B | Build + push `plan-001-E-amd64-v8` Docker image | 45 min | Low | P0 | Not started |
| 005-C | Pre-flight: Secrets Manager + IAM + networking validation | 30 min | Med | P0 | Not started |
| 005-D | Flip Fargate `desiredCount=1`; verify cold-start | 30 min | Med | P0 | Not started |
| 005-E | Cloud-side smoke test (Slack mention → Neon row) | 30 min | Low | P0 | Not started |
| 005-F | Unload local launchd; assert single-listener invariant | 15 min | Low | P0 | Not started |
| 005-G | 48-hour soak + cutover sign-off | calendar 48h + 30 min review | Low | P1 | Not started |
| 005-H | Document + open Plan 006 stub for MCP cloud landing | 30 min | Low | P1 | Not started |

## Execution Sequence

A → B → C → D → E → F → G → H

Phases A–F are a single contiguous push (4–6 hrs). Phase G is calendar-time soak. Phase H is paperwork.

**Rollback at any point:** `aws ecs update-service --cluster agentic-stack --service hermes --desired-count 0` and `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.gateway.plist` returns the system to local-only mode within 60 seconds. No data lost (sessions + tenants + feedback all live in Neon, which is the same store regardless of where Hermes runs).

---

## Phase Detail

### Phase 005-A — Land + verify PR #13 on main

**What:** PR #13 (`feat/plan-004-self-improvement`) has two unmerged commits as of 2026-05-23 (`1828d0d` tenant bootstrap, `369d83f` gateway awaits get_backend). The v8 Docker image must build from a main that includes them.

**Files to modify:** none (review + merge only)

**Acceptance criteria:**
- [ ] PR #13 CI is green (`gh run watch` on the branch)
- [ ] PR #13 merged to `main` on `blakeaber/hermes-agent` (squash or rebase, Blake's call)
- [ ] `git log main --oneline -5` shows tenant-bootstrap + get_backend commits

**Verification:**
```bash
cd /Users/blakeaber/Documents/hermes-agent
git fetch origin-fork
gh pr view 13 --json state,mergeable,statusCheckRollup
gh pr merge 13 --squash --auto    # or whichever merge strategy preferred
gh run watch
```

### Phase 005-B — Build + push `plan-001-E-amd64-v8` Docker image

**What:** Build a fresh amd64 image from `main` that incorporates Phase A's commits + today's two patches, push to ECR. Tag must increment from `plan-001-E-amd64-v7` → `plan-001-E-amd64-v8` so the task def update is trivial.

**Files to read (no modification expected):**
- `Dockerfile.saas` — built in Plan 001-E
- `scripts/build-push-saas.sh` — built in Plan 001-E

**Acceptance criteria:**
- [ ] `docker build --platform linux/amd64 -f Dockerfile.saas -t agentic-stack/hermes:plan-001-E-amd64-v8 .` succeeds locally
- [ ] Image health-checks pass: `docker run --rm -p 8080:8080 agentic-stack/hermes:plan-001-E-amd64-v8 &; curl -sf http://localhost:8080/health` returns 200
- [ ] Image pushed to ECR: `aws ecr describe-images --repository-name agentic-stack/hermes --image-ids imageTag=plan-001-E-amd64-v8` returns a row
- [ ] Image digest recorded in `STATUS.md` for rollback reference

**Verification:**
```bash
cd /Users/blakeaber/Documents/hermes-agent
git checkout main && git pull
./scripts/build-push-saas.sh plan-001-E-amd64-v8 2>&1 | tee /tmp/v8-build.log
aws ecr describe-images --repository-name agentic-stack/hermes \
  --image-ids imageTag=plan-001-E-amd64-v8 \
  --query 'imageDetails[0].{digest:imageDigest,pushed:imagePushedAt,sizeMB:imageSizeInBytes}' \
  --profile AgenticHub-162471567408
```

### Phase 005-C — Pre-flight: Secrets Manager + IAM + networking validation

**What:** Before touching ECS, verify every dependency Fargate Hermes needs is reachable from the task IAM role. The most common Fargate-day-1 failure is "task starts, can't read secret, crash loop" — this phase catches that without a 5-minute ECS restart cycle.

**Files to read (no modification expected):**
- Task def JSON: `aws ecs describe-task-definition --task-definition hermes-saas`
- `infra/terraform/hermes-fargate/main.tf` — IAM role + secret refs

**Pre-flight checks (each must pass):**

1. **Secrets exist + are populated:**
   - [ ] `agentic-stack/neon/hermes-saas` contains the `hermes_app` DSN (NOT the owner DSN — verify role)
   - [ ] `agentic-stack/hermes/slack-bot-token` exists and is non-empty
   - [ ] `agentic-stack/hermes/slack-app-token` exists and is non-empty (Socket Mode requires app-level token)
   - [ ] `agentic-stack/hermes/rooben-service-token` matches what's in `~/Documents/hermes-agent/.env` (so cloud + laptop agree on the rooben routing token)

2. **Task IAM role has secrets:GetSecretValue on each ARN above** — `aws iam simulate-principal-policy` for each secret
3. **Task IAM role has bedrock:InvokeModel on `us.anthropic.claude-sonnet-4-6`**
4. **Subnets + security groups allow egress to:** slack.com:443 (Socket Mode), bedrock-runtime.us-east-1.amazonaws.com:443, `ep-weathered-credit-aqq9kjyf.c-8.us-east-1.aws.neon.tech:5432`

**Acceptance criteria:**
- [ ] All 4 secrets present in Secrets Manager + readable from task role
- [ ] Bedrock InvokeModel allowed (simulate-principal-policy returns `allowed`)
- [ ] Neon DSN test from a sample Fargate task or local-with-task-role: `psql ... -c 'SELECT 1'` succeeds
- [ ] Slack `auth.test` succeeds with the cloud-side bot token (proves token didn't get rotated out from under us)

**Verification:**
```bash
# Secrets exist
for s in agentic-stack/neon/hermes-saas agentic-stack/hermes/slack-bot-token agentic-stack/hermes/slack-app-token agentic-stack/hermes/rooben-service-token; do
  aws secretsmanager describe-secret --secret-id "$s" --profile AgenticHub-162471567408 \
    --query '{name:Name,arn:ARN}' --output json
done

# IAM simulation (replace <task-role-arn>)
aws iam simulate-principal-policy --policy-source-arn <task-role-arn> \
  --action-names secretsmanager:GetSecretValue bedrock:InvokeModel \
  --resource-arns <secret-arn> bedrock:::us.anthropic.claude-sonnet-4-6 \
  --profile AgenticHub-162471567408
```

### Phase 005-D — Flip Fargate `desiredCount=1`; verify cold-start

**What:** Update the ECS service to use the v8 image, scale to 1 task, watch CloudWatch logs through cold start. Hermes Plan 001-E shipped a `Dockerfile.saas` health endpoint at `:8080/health` — the ALB target group already wires through it, so we'll know within ~90s if the container is alive.

**Files to read (no modification expected):**
- `infra/terraform/hermes-fargate/main.tf` — service definition

**Acceptance criteria:**
- [ ] `aws ecs update-service --cluster agentic-stack --service hermes --task-definition hermes-saas:<v8-revision> --desired-count 1` succeeds
- [ ] Task transitions to `RUNNING` within 120 seconds
- [ ] CloudWatch log group `/ecs/hermes-saas` shows the four expected startup lines: `Storage backend initialized: NeonBackend (pool=ready)` → `[Slack] Authenticated as @bossman2` → `[Slack] Plan 004-A tenant bootstrap: team=T0B16FV0KFF → ...` → `[Slack] Socket Mode connected (1 workspace(s))`
- [ ] Target group health check at `:8080/health` returns 200 within 90s of RUNNING
- [ ] **No** lines in CloudWatch matching `signature expired|DNS resolution|TimeoutError` for first 10 minutes
- [ ] Fargate task RSS < 500 MB after 5-minute warmup (memory_monitor heartbeat)

**Verification:**
```bash
# Get latest task def revision after Phase A's main + Phase B's image deploy
TD=$(aws ecs describe-task-definition --task-definition hermes-saas \
  --profile AgenticHub-162471567408 --query 'taskDefinition.revision' --output text)

aws ecs update-service --cluster agentic-stack --service hermes \
  --task-definition hermes-saas:$TD --desired-count 1 \
  --force-new-deployment --profile AgenticHub-162471567408

# Watch for RUNNING
aws ecs wait services-stable --cluster agentic-stack --services hermes \
  --profile AgenticHub-162471567408

# Tail logs (first task ARN)
TASK=$(aws ecs list-tasks --cluster agentic-stack --service-name hermes \
  --profile AgenticHub-162471567408 --query 'taskArns[0]' --output text)
aws logs tail /ecs/hermes-saas --follow --profile AgenticHub-162471567408
```

### Phase 005-E — Cloud-side smoke test (Slack mention → Neon row)

**What:** With Fargate Hermes live AND laptop launchd Hermes ALSO still live, we now have two Hermes instances on the same Slack socket. **Both will receive every event.** That's expected for the test duration only — we want to prove the cloud instance handles inbound correctly. Phase F kills the laptop instance immediately after.

**Acceptance criteria:**
- [ ] Slack `@Hermes hello (cloud test 1)` → at least one reply arrives within 10s
- [ ] CloudWatch shows `inbound message: platform=slack ...` matching the test message
- [ ] Neon `skill_output_map` table has a new row matching the Slack message ts (`/opt/homebrew/opt/libpq/bin/psql "<owner-dsn>" -c "SELECT slack_ts, skill_name, registered_at FROM skill_output_map ORDER BY registered_at DESC LIMIT 5;"`)
- [ ] React 👍 to the cloud reply → `skill_feedback` row appears in Neon within 5s
- [ ] Remove the 👍 → row deleted (idempotency)

**If duplicate responses arrive** (both gateways replied): expected for this phase. Phase F removes the laptop side. If ONLY the laptop replied and the cloud reply didn't arrive at all: cloud Socket Mode connection is broken; rollback (Phase D's update-service with desired-count=0) and debug in CloudWatch before retry.

### Phase 005-F — Unload local launchd; assert single-listener invariant

**What:** With cloud Hermes proven, shut down the laptop's launchd Hermes permanently for this cutover. Verify exactly one Slack listener exists workspace-wide.

**Files to modify:**
- `~/Library/LaunchAgents/ai.hermes.gateway.plist` — keep on disk for rollback, but `bootout` to disable

**Acceptance criteria:**
- [ ] `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.gateway.plist` succeeds
- [ ] `launchctl list | grep hermes` returns nothing
- [ ] `ps aux | grep hermes_cli | grep -v grep` returns nothing on the laptop
- [ ] After 30s, `@Hermes hello (single-listener test)` still gets exactly ONE reply (proves cloud is the only listener)
- [ ] `~/Library/LaunchAgents/ai.hermes.gateway.plist.disabled-stage1-cutover-2026-05-23` symlink or rename created (so rollback is `mv` + `bootstrap`)

### Phase 005-G — 48-hour soak + cutover sign-off

**What:** Calendar time. Hermes runs in Fargate for 48 hours with daily-driver use. Failures + recoveries are observed in CloudWatch.

**Acceptance criteria:**
- [ ] 48 hours elapse with `aws ecs describe-services` showing `runningCount=1, desiredCount=1, pendingCount=0` throughout
- [ ] Zero task restarts caused by container exit (CloudWatch event count for `STOPPED` reason `Essential container exited` = 0)
- [ ] At least 10 Slack interactions completed end-to-end (mention → reply round-trip ≤ 30s P95)
- [ ] At least one cron firing of `feedback-triage-nightly` completes without Bedrock 403 (i.e., the clock-skew bug class is gone)
- [ ] No `signature expired` errors in CloudWatch for 48h
- [ ] Average task RSS < 400 MB across the soak window (no memory leak)

**If any acceptance criterion fails:** rollback per the Phase D escape hatch. The laptop launchd plist is still on disk under the `.disabled-stage1-cutover-2026-05-23` rename — bring it back with `mv ... ai.hermes.gateway.plist && launchctl bootstrap ...` and the system is back to local-only in 60s.

### Phase 005-H — Document + open Plan 006 stub for MCP cloud landing

**What:** Write up findings, mark Plan 005 Complete, open a new Plan 006 stub for "MCP cloud landing zone" so Stage 2 has a docket entry.

**Files to modify:**
- `plans/PROGRESS.md` — mark 005 COMPLETE; add 006 stub row
- `plans/005-fargate-cutover-stage1/STATUS.md` — final summary block
- New file: `plans/006-mcp-cloud-landing/006-mcp-cloud-landing.md` — stub only (Phase H is allowed to be a one-pager spec; actual phasing comes in a future architect session)
- `/Users/blakeaber/Documents/agentic-hub/plans/cross-repo-tier-graph.md` — add row for hermes-005 (COMPLETE) and hermes-006 (NOT STARTED, depends-on hermes-003+hermes-005)

**Acceptance criteria:**
- [ ] PROGRESS.md updated + committed
- [ ] STATUS.md final-summary block written with: image digest used, total cost during soak, Slack interaction count, restart count, P95 latency
- [ ] cross-repo-tier-graph.md updated
- [ ] Plan 006 stub file exists with at least: Why, the laptop-MCP problem it solves, candidate phases (registry seeding, Pipedrive/Linear/Google Workspace OAuth-on-Fargate, latency budget, fallback semantics when an MCP is down)

---

## Critical files (summary)

### Read for context
- `Dockerfile.saas` — built in Plan 001-E; do not modify in this plan
- `scripts/build-push-saas.sh` — built in Plan 001-E
- `infra/terraform/hermes-fargate/main.tf` — service + IAM
- `gateway/health.py` + `gateway/health_server.py` — Fargate liveness path
- `infra/terraform/hermes-fargate/*.tfplan` artifacts (existing) — last-applied state
- `~/Library/LaunchAgents/ai.hermes.gateway.plist` — local launchd to disable

### Existing utilities to reuse
- AWS profile `AgenticHub-162471567408` (AdministratorAccess) — used throughout per Plan 004-A productionization session
- Neon owner DSN — for verification only; never persisted to `.env`, only used inline for read queries
- `ScheduleWakeup`/`Monitor` patterns — for the 48h soak in Phase G

## Out of scope (explicit)

- MCP server cloud migration → Plan 006 stub written in Phase H
- Refactoring Slack adapter to Events API (would fix multi-listener architectural debt but is its own plan)
- Cost optimization / Spot Fargate / scale-down idle → defer until post-soak
- Fargate observability beyond raw CloudWatch (no OTEL / Datadog / Sentry wire-up here)
- Hermes auto-deploy on PR merge (manual `update-service` only; CI deploy is Stage 3)
- Migrating `~/.hermes/kanban.db` to DynamoDB (kanban dispatcher is OFF in cloud Hermes by config; reactivation is a Plan 002 / 005-A concern)

## Verification (end-state)

| Phase | Acceptance gate |
|---|---|
| A | PR #13 merged; main has both Plan-004-A commits |
| B | v8 image in ECR with recorded digest |
| C | All 4 secrets readable from task role; Bedrock + Neon reachable |
| D | Fargate task RUNNING; tenant bootstrap log line present |
| E | Slack mention → Neon row round-trip works from cloud Hermes |
| F | Laptop launchd unloaded; exactly one Slack listener workspace-wide |
| G | 48 hours, zero `signature expired`, P95 reply ≤ 30s |
| H | PROGRESS.md updated; Plan 006 stub exists |

## Session-discovered follow-ups (anticipated, track at Phase H)

1. **Plan 006 — MCP cloud landing zone**: skills-service registries empty; Pipedrive/Linear/Google Workspace OAuth tokens currently live in laptop env; Fargate Hermes runs without these tools until 006 lands.
2. **Fargate cost-cap alarm**: add CloudWatch billing alarm at $50/mo before soak begins (defensive; expected steady-state is $30–40 for t-tier 24/7).
3. **`feedback-triage-nightly` cron**: verify it runs in Fargate on the new schedule (`cron/scheduler.py` reads from `cli-config.yaml` `cron.jobs:` block, which is currently empty — so the cron may not exist in cloud-mode at all; Phase G acceptance criterion may need re-scoping).
4. **CloudWatch log retention**: defaults to "never expire" → 90-day retention should be set explicitly during Phase D to cap log costs.

## Rollback playbook (laminated for ops)

```bash
# Emergency rollback — return to laptop-only Hermes in ~60s
aws ecs update-service --cluster agentic-stack --service hermes \
  --desired-count 0 --profile AgenticHub-162471567408

mv ~/Library/LaunchAgents/ai.hermes.gateway.plist.disabled-stage1-cutover-2026-05-23 \
   ~/Library/LaunchAgents/ai.hermes.gateway.plist

launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.gateway.plist
launchctl enable gui/$(id -u)/ai.hermes.gateway
```

No data migration needed — sessions, tenants, feedback all live in Neon regardless of where Hermes runs.
