# Status — Plan 010: Cost Reduction + Skills Service Ops + Kanban Re-enable + Linear Decision

**Status:** IN PROGRESS (3 of 6 phases done; 2 deferred; 1 blocked)
**Last updated:** 2026-05-26
**Blocked by:** Phases A + D defer to post-Plan-005-G soak; Phase E blocked on Plan 008

## Phase Progress

| Phase | Title | Status | Notes |
|---|---|---|---|
| 010-A | NAT Gateway elimination | DEFERRED | Terraform-destructive; requires post-soak verification window. Spec ready in master plan. |
| 010-B | Bedrock OSS model swap | **Complete (2026-05-26)** | Sonnet 4.6 → Llama 3.3 70B (`us.meta.llama3-3-70b-instruct-v1:0`). Baked into v11 image, td:15 deployed. Verified working via `aws bedrock-runtime converse` from same account. |
| 010-C | Skills Service: populate registries | **Complete (2026-05-26)** | Local Skills Service now serves 3 registries: `personal:blake-personal-skills` (~/.hermes/skills), `personal:agentic-hub-skills`, `global:blake-cowork-plugins`. `GET :8001/health` returns `{"status":"ok","registries":3,"scopes":["personal","personal","global"]}` |
| 010-D | Skills Service: cloud Fargate deployment | DEFERRED | Substantial new infrastructure (Dockerfile build, ECR push, new task def + service + Terraform module). Defer to focused session post-soak. |
| 010-E | Re-enable kanban dispatcher in cloud | BLOCKED on Plan 008 | Per plan, gated on Plan 008-C load test confirming MCPGateway pools correctly |
| 010-F | Linear decision: operationalize OR remove | **Complete (2026-05-26)** | REMOVED. Audit confirmed zero Linear tool calls in 5+ months. Local config + .env + AWS secret all cleaned up (AWS secret has 7-day recovery window for paranoia). |

## Resumption context

- Next phases: 010-A + 010-D after Plan 005-G soak (48h, ends 2026-05-28)
- 010-E unblocks after Plan 008 ships

## Adaptations log

### 2026-05-26 — Llama 3.2 90B → 3.3 70B switch
Initial plan said "try Llama 3.1 70B." Discovered:
- `us.meta.llama3-1-70b-instruct-v1:0` exists as foundation model but has NO cross-region inference profile (CRIS) — only direct `meta.llama3-1-70b-instruct-v1:0` access
- Tried `us.meta.llama3-2-90b-instruct-v1:0` instead (CRIS available) — but AWS Bedrock marked it LEGACY: "Access denied. This Model is marked by provider as Legacy and you have not been actively using the model in the last 30 days." AWS deprecation policy: legacy models lose access after 30 days of non-use.
- Landed on `us.meta.llama3-3-70b-instruct-v1:0` — newest Llama generation with CRIS + verified working via direct Bedrock Converse API call ("BEDROCK_LLAMA_OK" response confirmed).

### 2026-05-26 — Plan 010-A deferred
Plan 010-A (NAT Gateway elimination) requires Terraform-destructive apply. Defer until Plan 005-G soak completes so we have a known-good baseline before disrupting cloud egress. Spec is ready (terraform modules + assignPublicIp changes documented in master plan).

### 2026-05-26 — Plan 010-D deferred
Skills Service cloud Fargate deployment needs: Docker image build for hermes-skills-service, ECR push, new task def + service + Terraform module + cloud Hermes config update to point at the cloud Skills Service URL. Substantial work; defer to focused session.

## Verification artifacts
- Linear removal: `aws secretsmanager describe-secret --secret-id agentic-stack/mcp-linear-api-key` shows deletion scheduled for 2026-06-02 (7-day recovery)
- Skills registries: `curl :8001/health` returns the 3-registry response
- Llama swap: CloudWatch log group `/ecs/agentic-stack/hermes` shows task `d4f068ed...` started cleanly on td:15. Next Slack interaction will exercise the new model (verifiable in conversation_loop log line that includes `model=us.meta.llama3-3-70b-instruct-v1:0`)
