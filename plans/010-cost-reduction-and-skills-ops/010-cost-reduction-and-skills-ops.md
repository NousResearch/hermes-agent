# Plan 010 — Cost Reduction + Skills Service Operationalization + Kanban Re-enable + Linear Decision

**Status:** DRAFT 2026-05-26
**Run after:** Plan 005-G soak (Plan 005 must be soaked + signed off before Bedrock-model swap risks the daily-driver)
**Blocks:** Nothing critical
**Estimated effort:** 1-2 working days across phases

## Why this plan exists

Three loose-ends + one decision surfaced tonight that don't fit cleanly into Plans 005-009:

1. **Cost waste discovered during tonight's audit**: AWS month-to-date is **$244** vs MASTER_PLAN AC6 target of ≤$100/mo. Top drivers:
   - $113 NAT Gateway (mostly legacy from pre-VPC-endpoint days, but ~$32/mo base is ongoing)
   - $81 Bedrock Sonnet 4.6 (real usage; can swap to OSS Bedrock models for ~50-70% reduction)
   - $19 Rooben Aurora RDS (deleted tonight, saves $19/mo)
2. **Skills Service is paper-deployed** — Plan 003 marked Complete, code runs at `:8001`, BUT `GET /health` shows `"registries":0` and there's no plan to populate or cloud-deploy it
3. **Kanban dispatcher disabled in both local + cloud** (per cli-config.saas.yaml + cli-config.yaml). Re-enable needs MCP pooling (Plan 008) AND a config flip — no plan owned the "flip the flag back" step
4. **Linear is configured but unused**: `~/.hermes/config.yaml` has Linear MCP wired (`mcp-server-linear` + `LINEAR_API_KEY`), but gateway logs show **zero Linear tool calls** in production. Decision needed: operationalize via Plan 006 OR remove from config

## Phase Index

| Phase | Title | Effort | Risk | Priority | Status |
|---|---|---|---|---|---|
| 010-A | NAT Gateway elimination (or downsizing) | 2-3 hr | Med | P0 cost | Not started |
| 010-B | Bedrock OSS model swap (Sonnet → Llama 3.1 70B Bedrock) | 1 hr | Low | P0 cost | Not started |
| 010-C | Skills Service: populate registries | 2-3 hr | Low | P1 product | Not started |
| 010-D | Skills Service: cloud Fargate deployment | 3-4 hr | Med | P1 product | Not started |
| 010-E | Re-enable kanban dispatcher in cloud (after Plan 008 lands) | 30 min | Low | P1 (gated on Plan 008) | Not started |
| 010-F | Linear decision: operationalize OR remove | 30 min | Low | P2 | Not started |

## Phase Detail

### 010-A — NAT Gateway elimination/downsizing

**Why:** $32/mo base + ~$45/mo data charges, even after VPC endpoints (agentic-hub Plan 009-A) eliminated S3/ECR/Bedrock/Secrets traffic. Remaining outbound is mostly Slack + Anthropic + GitHub.

**Options:**
1. **Remove NAT entirely** (the agentic-hub Plan 009-A.1 candidate). Requires Fargate tasks to have **public IPs** (assignPublicIp: ENABLED) so they can reach Slack/etc directly. Trade-off: each task gets a public IPv4 (small cost ~$3.50/mo per task) + slightly larger attack surface. **Net saving: ~$25-40/mo**.
2. **Keep NAT but right-size**: NAT Gateway is already smallest tier; only data-charge optimization left. Not much further savings without architectural change.
3. **Skip — keep NAT**: Most conservative; accept $32/mo as cost of doing business.

**Recommendation: Option 1.** Single-user stack; security risk is bounded. Update Terraform to `assignPublicIp: ENABLED` on Fargate services + remove NAT Gateway + EIP.

**Files to modify:**
- `agentic-hub/terraform/modules/fargate-hermes/main.tf` (assignPublicIp + remove dependency on NAT route)
- `agentic-hub/terraform/modules/fargate-atlas/main.tf` (same)
- `agentic-hub/terraform/modules/vpc/main.tf` (delete NAT gateway resources)

**Acceptance:**
- [ ] `aws ec2 describe-nat-gateways` returns 0 active gateways in agentic-stack VPC
- [ ] Cloud Hermes still reaches Slack (Socket Mode connects) post-deploy
- [ ] Cloud Atlas still reaches Bedrock + external services
- [ ] Next month's billing shows NAT line item ≈ $0 (vs $113 this month, $32 base going forward)

### 010-B — Bedrock OSS model swap

**Why:** $81 spent on Sonnet 4.6 MTD. Bedrock hosts OSS models (Llama 3.1 70B Instruct, Mistral Large 2, Claude Instant) that are 50-70% cheaper for many tasks. Aligns with MASTER_PLAN's "eventually choosing OSS models" intent.

**Implementation:**
- Change `cli-config.saas.yaml`'s `model.default` from `us.anthropic.claude-sonnet-4-6` to e.g. `us.meta.llama3-1-70b-instruct-v1:0` (or `mistral.mistral-large-2407-v1:0`)
- Bake into new Hermes image (v10)
- Verify task IAM role has `bedrock:InvokeModel` on the new model ARN
- Test quality on representative tasks (Slack chitchat, tool calls, long-context summarization)
- Roll back via config swap + redeploy if quality regresses

**Open decision:** which OSS model first? Recommend Llama 3.1 70B as it has the broadest tool-use / function-calling track record on Bedrock. Mistral Large 2 is a backup if Llama hits issues.

**Acceptance:**
- [ ] Cloud Hermes responds to `@Hermes hello` using Llama (verified in CloudWatch `provider=bedrock, model=us.meta.llama3-1-70b-instruct-v1:0`)
- [ ] Tool calls still work (verify Skills Service round-trip if Plan 010-D lands first; otherwise verify session_search / get_timeline)
- [ ] Quality acceptable for daily-driver use (subjective; Blake's call after 1 day)
- [ ] Next billing shows Bedrock line item ≤50% of Sonnet baseline

### 010-C — Skills Service: populate registries

**Why:** Plan 003 shipped Skills Service code but `:8001/health` reports `"registries":0`. No skills available means no skill-routed responses, no `skill_output_map` correlation for non-`_agent_default` skills, defeats much of Plan 004-A's purpose.

**Implementation:**
- Identify what skills exist locally (look in `~/.hermes/skills/`, `/Users/blakeaber/Documents/agentic-hub/skills/`, `blake-cowork-plugins/` directories)
- Create Git registry (per Plan 003-B/C/D Git-backed registry pattern) for these skills
- Configure Skills Service config to point at the registry
- Verify `:8001/skills/{scope}` returns populated list
- Add at least 3 real skills (e.g., outreach-drafter, daily-digest-summarizer, slack-channel-context)

**Acceptance:**
- [ ] `GET :8001/health` returns `registries: ≥1, scopes: [≥1 names]`
- [ ] At least 3 skills resolvable via `/skills/personal/{user_id}/{skill_name}`
- [ ] Hermes (local AND cloud once 010-D lands) can route to one of these skills end-to-end → skill_output_map row in Neon with the actual skill_name (not `_agent_default`)

### 010-D — Skills Service: cloud Fargate deployment

**Why:** Local Skills Service runs on Blake's laptop port 8001. Cloud Hermes (v9+) can't reach it. Plan 003-F shipped `S3SkillSource` for the SaaS path, but no Fargate service exists yet.

**Implementation:**
- Build Skills Service Docker image (find existing Dockerfile in `hermes-skills-service/` repo; build/push to ECR `agentic-stack/skills-service:v1`)
- Add Terraform module `agentic-hub/terraform/modules/fargate-skills/` modeled on `fargate-hermes/`
- Task def env: `HERMES_MODE=saas`, `S3_SKILLS_BUCKET=hermes-saas-skills` (already exists per Plan 003-F), Neon DSN
- ECS service + internal DNS (e.g., `skills.agentic-stack.internal:8001`)
- Update cloud Hermes config (cli-config.saas.yaml) to point at the cloud Skills Service URL
- Verify reachability from cloud Hermes via a Skill Service ping endpoint

**Acceptance:**
- [ ] `aws ecs describe-services --services skills-service` returns `running=1, desired=1`
- [ ] CloudWatch log shows Skills Service startup + healthy
- [ ] Cloud Hermes can invoke a skill via Skills Service (verified via Neon `skill_output_map` row with the real skill name)
- [ ] Plan 003 PROGRESS row updated from "COMPLETE (local)" to "COMPLETE (local + cloud)"

### 010-E — Re-enable kanban dispatcher in cloud (gated on Plan 008)

**Why:** `kanban.dispatch_in_gateway=false` in both configs since Plan 002/005-A audit. Re-enable once MCP pooling actually works in cloud (Plan 008 acceptance includes load test with subprocess-count assertion).

**Implementation:**
1. Wait for Plan 008-C acceptance (load test passes: 10 concurrent tool calls → 0 new MCP subprocesses)
2. Update `cli-config.saas.yaml`: `kanban.dispatch_in_gateway: true`
3. Bake into new Hermes image
4. Deploy + monitor task RSS for 24h
5. If RSS exceeds 2GB or task restarts due to memory, rollback to `false`

**Acceptance:**
- [ ] Plan 008-C complete + 008-F (cloud smoke test) passing
- [ ] Cloud `cli-config.saas.yaml` has `kanban.dispatch_in_gateway: true`
- [ ] Task RSS < 2GB for 24h with at least 5 kanban tasks executed
- [ ] No "Essential container exited" events in 24h window

### 010-F — Linear decision: operationalize OR remove

**Why:** Linear MCP is configured (`~/.hermes/config.yaml`: `mcp-server-linear` with `LINEAR_API_KEY`) but gateway logs show **zero Linear tool calls** in production usage. Either commit or cut.

**Two paths:**

**A. Operationalize Linear** — execute Plan 006 (Workflow Observability) which has 006-C: "Linear issue enrichment from workflow events" as a built-in surface. Linear becomes the destination for Hermes-generated tasks + RoobenVerifier output. Effort: Plan 006's 4 phases (~3-4 days). Strategic value: project-tracking surface for AI-driven workflows.

**B. Remove Linear** — delete the MCP config from `~/.hermes/config.yaml` + `cli-config.saas.yaml`. Drop `LINEAR_API_KEY` from local `.env` and Secrets Manager. Removes attack surface + complexity for a feature you're not using. Easy to add back later if you decide to. Effort: 15 min.

**Recommendation:** Path B — REMOVE. Reasoning:
- Plan 006 is DRAFT, not been executed; committing to Linear-via-006 adds ~3-4 days of work to validate something you've already not used in 5+ months
- Hermes can always re-integrate Linear later via a quick MCP config add
- Reducing config surface reduces "set up but never used" pattern that bit you in Plan 005

**Acceptance (if Path B):**
- [ ] `~/.hermes/config.yaml` and `cli-config.saas.yaml` no longer reference `mcp-server-linear`
- [ ] `LINEAR_API_KEY` removed from `~/.hermes/.env` and `/Users/blakeaber/Documents/hermes-agent/.env`
- [ ] No regression in Hermes startup or operation
- [ ] Decision documented in Plan 010 STATUS

## Out of scope

- Skills Service web UI (separate work)
- Multi-tenant Skills Service (single user; not needed)
- Replacing Bedrock entirely with another provider (Bedrock is the strategic choice)
- Detailed Linear-to-Hermes workflow design (Plan 006-C scope if Path A chosen)

## Cross-references

- Plan 005 (Fargate cutover) — must soak before 010-A (NAT removal) and 010-B (Bedrock model swap) — both touch production daily-driver path
- Plan 008 (MCPGateway production wiring) — direct prereq for 010-E (kanban re-enable)
- Plan 009 (Cloud LLM Routing Infra) — same architectural family; 010 picks up the cost + skills follow-ups
- agentic-hub Plan 009 — 010-A NAT removal is the "009-A.1 candidate" mentioned in agentic-hub's plan
- Plan 003 (Skills Service) — 010-C/D are the operationalization follow-up that Plan 003 deferred

## Verification (end-state)

| Phase | Acceptance gate |
|---|---|
| A | NAT Gateway deleted; cloud services still reachable; next month's NAT cost ≈ $0 |
| B | OSS model serving Slack interactions; quality acceptable; Bedrock cost ≤50% of baseline |
| C | Skills Service has ≥3 real skills resolvable |
| D | Cloud Skills Service running; cloud Hermes invokes skills end-to-end |
| E | Kanban dispatcher running in cloud without OOM for 24h+ |
| F | Linear decision documented and config reflects the decision |
