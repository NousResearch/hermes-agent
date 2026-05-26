# Plan 009 — Cloud LLM Routing Infrastructure (Unblock Plan 005)

**Status:** DRAFT 2026-05-26
**Run after:** None — fully unblocked, just needs focused time
**Blocks:** Plan 005 (Fargate Hermes cutover) — Phase 005-E proved cloud Hermes can't make LLM calls until 009 lands
**Estimated effort:** 1-2 working days

## Why this plan exists

Plan 005 attempted the cloud cutover tonight (2026-05-25/26) and surfaced that **the cloud LLM-routing infrastructure was never finished by Plan 001-E**. Plan 001-E scaffolded ECR repos, secrets, task defs, and services for Portkey + Rooben + Hermes — but several critical setup steps were never completed, leaving the cloud as a paper architecture that has never actually processed an LLM call.

What Plan 005 discovered tonight:

| Gap | Discovered via | Impact |
|---|---|---|
| **Portkey image missing from ECR** | `aws ecr list-images --repository-name agentic-stack/portkey` returned empty | Portkey service has been crash-looping for as long as it's been deployed. Fixed tonight by pulling `portkeyai/gateway:latest` from Docker Hub and pushing to ECR. |
| **Portkey has no provider config** | Portkey image now runs but is vanilla — no virtual-key → provider mappings | Even with image present, Hermes's "bed..." virtual key gets 401 "Missing Authentication header" because Portkey has no config to route it anywhere |
| **Rooben image missing from ECR** | Same `list-images` check returned empty | Cloud Rooben crash-loops (`running=0, desired=1`). Local Rooben works because launchd uses local Python venv. |
| **All LLM-routing secrets are PLACEHOLDERS** | `aws secretsmanager get-secret-value`: anthropic-api-key=`PLACEHOLDER...` (54 chars), hermes-portkey-virtual-key=`bed...` (14 chars) | No service in cloud has ever had real LLM credentials |
| **Hermes saas-mode startup omits stderr handler** | `gateway/run.py:17970` requires CLI `-v` flag, which the entrypoint doesn't pass | Cloud Hermes startup logs went into the void for 5 days. Fixed tonight via CMD override `["gateway", "run", "-v"]` in task def. Permanent fix: bake `-v` into `docker/entrypoint.saas.sh`. |
| **Cloud Hermes had no SLACK_ALLOWED_USERS** | WARNING log + 'Unauthorized user' rejection of Blake's message | Fixed tonight via env var add. Permanent fix: bake into cloud config or use `GATEWAY_ALLOW_ALL_USERS=true` for single-user cloud. |
| **Cloud Hermes had no PYTHONUNBUFFERED=1** | First-load logs were buffered before flush | Fixed tonight via env var add. Permanent fix: bake into Dockerfile.saas. |
| **Cloud kanban dispatcher is ENABLED** | `INFO gateway.run: kanban dispatcher: embedded in gateway (interval=60.0s)` in cloud logs | High OOM risk per Plan 005-G follow-ups (per-worker MCP fan-out). Permanent fix: disable in cloud config. |

**The honest read**: cloud was never working. Local launchd Hermes has been the ONLY working Hermes for the past 5+ days. Plan 005's "cutover" was actually "make cloud work for the first time."

## Strategic decision — Portkey strategy

Discussed tonight: **does Hermes need Portkey at all?**

| Portkey feature | Single-user value | Alternative |
|---|---|---|
| Vendor abstraction | ✅ Real | Provider-specific SDKs (Hermes already has this for local) |
| Observability | ✅ Real | OTel collector (already deployed in agentic-stack) + CloudWatch metrics |
| Cost tracking | ✅ Real (AC6 ≤$100/mo) | AWS Cost Explorer + Bedrock invocation metrics |
| Failover (Bedrock→Anthropic) | ✅ Real | Hermes already has 3-attempt retry + fallback in `agent.conversation_loop` |
| Caching | ⚠️ Nice-to-have | Agent contexts stateful — low hit rate anyway |
| Virtual keys | ⚠️ Nice-to-have | Single-user — no rotation pressure |
| Rate limiting | ❌ Not needed | Single-user |
| Guardrails | ❌ Hermes has its own redactor | Plan 007-D uses `agent.redact` |

**Decision pending Blake**: remove Portkey from architecture entirely, OR finish Portkey config and keep it.

Two paths represented as Phases A and B; pick ONE.

## Phase Index

| Phase | Title | Effort | Risk | Priority | Status |
|---|---|---|---|---|---|
| 009-A-NoPortkey | Remove Portkey; wire direct Anthropic/Bedrock per service | 1 day | Med | Pick one of A/B | Not started |
| 009-B-WithPortkey | Author Portkey config, push real virtual keys, configure provider routing | 1-2 days | High | Pick one of A/B | Not started |
| 009-C | Populate all placeholder secrets with real credentials | 1 hr | Low | P0 | Not started |
| 009-D | Build + push Rooben image (if Rooben stays in architecture) | 2 hr | Low | P1 | Not started |
| 009-E | Bake permanent fixes into `Dockerfile.saas` + `entrypoint.saas.sh` (-v flag, PYTHONUNBUFFERED, allowlist defaults) | 2 hr | Low | P0 | Not started |
| 009-F | Disable kanban dispatcher in cloud config (OOM-risk per Plan 005-G follow-up) | 30 min | Low | P0 | Not started |

## Phase Detail (high-level)

### 009-A (NoPortkey path)

If Blake decides to remove Portkey:
1. Update `agentic-stack-hermes` task def: remove `OPENAI_BASE_URL` + `OPENAI_API_KEY`, add `ANTHROPIC_API_KEY` (from real secret) + set model.provider=anthropic in config
2. Same for `agentic-stack-atlas` task def (Atlas wired through Portkey per Plan 007-B in agentic-hub)
3. Same for Rooben if still in architecture
4. Wire OTel observability for LLM calls (already have OTel collector; need code-level instrumentation)
5. Scale `portkey` service to desired=0, delete Portkey ECR images, delete task def revisions
6. Verify each service makes successful LLM calls

### 009-B (WithPortkey path)

If Blake decides to keep Portkey:
1. Author `config.yaml` for Portkey with:
   - Provider definitions: AWS Bedrock (via task IAM), Anthropic (via real API key)
   - Virtual keys: `bedrock-claude-sonnet`, `anthropic-claude-sonnet`, etc.
   - Routing rules: primary=bedrock, fallback=anthropic
2. Bake config into a custom `Dockerfile.portkey` (or mount via Fargate ECS volumes)
3. Build + push custom image to `agentic-stack/portkey:custom-v1`
4. Update Portkey task def to use new image
5. Update Hermes/Atlas task defs with REAL virtual key values
6. Verify routing end-to-end

### 009-C — Real credentials in Secrets Manager

Populate these (currently placeholders):
- `agentic-stack/anthropic-api-key` — Blake's real Anthropic API key (or a Portkey virtual key if 009-B)
- `agentic-stack/hermes-portkey-virtual-key` — real Portkey vkey (only needed if 009-B)
- `agentic-stack/rooben-portkey-virtual-key` — same

### 009-D — Rooben image (skip if Rooben deprecated)

Only relevant if Rooben Pro stays in cloud architecture. Rooben source lives in `rooben-pro/` repo per MASTER_PLAN. Need to: build Docker image, push to ECR, redeploy service. Pre-Plan-005 spec needed: does cloud still want Rooben, given MCPGateway from Plan 002-C handles the orchestration role?

### 009-E — Permanent fixes for Dockerfile.saas / entrypoint.saas.sh

Move the tonight's task-def overrides to permanent code:
- `entrypoint.saas.sh`: change `exec hermes "$@"` to `exec hermes "$@" -v` (so saas mode always has stderr logging)
- `Dockerfile.saas`: add `ENV PYTHONUNBUFFERED=1`
- Task def template: include `SLACK_ALLOWED_USERS` (or `GATEWAY_ALLOW_ALL_USERS=true` for single-user)

### 009-F — Kanban dispatcher disable in cloud

Cloud config currently shows `kanban dispatcher: embedded in gateway` at startup. Per Plan 005-G follow-up + Plan 002 / 005-A notes, per-worker MCP fan-out OOMs containers under any non-trivial workload. Fix: add `kanban.dispatch_in_gateway: false` to cloud config (via env var or file mount).

## Acceptance criteria (whole-plan)

- [ ] Cloud Hermes processes a real Slack message end-to-end (user mention → Hermes reply → Neon rows for messages + raw_events + skill_feedback)
- [ ] Plan 005-E can complete its acceptance criteria
- [ ] Cloud Hermes survives 48h soak (Plan 005-G) without infrastructure-related restarts
- [ ] No silent failure modes (every error has a visible log line)
- [ ] Decision made + documented: keep Portkey or remove

## Critical files

### Read for context
- `Dockerfile.saas` + `docker/entrypoint.saas.sh` (Hermes container)
- `infra/terraform/hermes-fargate/main.tf` (task def + secrets refs)
- This session's logs in CloudWatch (`/ecs/agentic-stack/hermes`, `/ecs/agentic-stack/portkey`, `/ecs/agentic-stack/rooben`)
- agentic-hub MASTER_PLAN (Portkey + Rooben role definitions)

### Will be modified
- Task definitions: `agentic-stack-hermes`, `agentic-stack-atlas`, possibly `agentic-stack-rooben`
- `Dockerfile.saas`, `docker/entrypoint.saas.sh` (permanent fixes from 009-E)
- AWS Secrets Manager values (009-C)
- Portkey config (only 009-B)

## Out of scope

- Plan 008 (MCPGateway production wiring) — still gated on Plan 005 + 009 completing
- Atlas LLM routing fixes beyond what's needed for Hermes parity
- Multi-tenant Portkey setup (single-user stack; defer until productization decision)
