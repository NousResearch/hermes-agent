# Plan 009 — Cloud LLM Routing Infrastructure (Unblock Plan 005)

**Status:** DRAFT 2026-05-26 (v2 — revised with cross-repo plan audit findings)
**Run after:** None — fully unblocked, just needs focused time
**Blocks:** Plan 005 (Fargate Hermes cutover) — Phase 005-E proved cloud Hermes can't make LLM calls until 009 lands
**Estimated effort:** 4-8 hours (NoPortkey path) OR 1-2 days (WithPortkey path)

## Why this plan exists — full story now known

Plan 005 attempted the cloud cutover tonight (2026-05-25/26). Cross-repo plan audit afterward revealed the cloud infrastructure is incomplete **by design**, then nobody finished the manual hand-off step. Specifically:

**The Terraform was intentionally designed to leave secrets + images unfilled.**
- `agentic-hub/terraform/modules/secrets/main.tf` lines 3–48 creates all 15 secrets with literal `"PLACEHOLDER — ..."` strings + `lifecycle { ignore_changes = [secret_string] }` — explicitly expecting `aws secretsmanager put-secret-value` after apply.
- `agentic-hub/terraform/modules/ecr/main.tf` creates ECR repos as empty shells. Terraform never pushes images.
- This is conventional Terraform: IaC creates the structure; humans populate sensitive content.

**The runbook for the manual step exists but was never run for the initial deploy.**
- `agentic-hub/runbooks/rebuild.md` Step 2 ("Populate Secrets") + Step 4 ("Build and Push Docker Images") covers exactly what's needed.
- BUT it's framed as a *disaster recovery* procedure, not a *first-time bootstrap*. The first-time bootstrap was assumed implicit and was skipped.

**The plans that would have caught this gap were never written.**
- `agentic-hub/MASTER_PLAN.md` Phase 001-B (Portkey Fargate deploy) and 001-E (Rooben Fargate deploy) were never decomposed into executable plan directories.
- `agentic-hub/plans/005-close-workaround-gaps/STATUS.md:141` says "Wiring Atlas to hosted Portkey is a future plan item" — that future plan item was never written.
- All Plan 001-A/B/E validation criteria used "Terraform apply succeeds" as success, NEVER "service makes a real LLM call from cloud container."

**Local launchd Hermes worked fine via a parallel non-cloud path** (Bedrock + Anthropic key in `.env` + launchd plist), which masked the cloud-side hole for 5+ days.

**Rooben is being deprecated** (post-mortem revealed this changes Plan 009 scope significantly):
- `agentic-hub/plans/005-close-workaround-gaps` (Complete 2026-05-15) archived Rooben as **in-process** — rooben-planning/ is vendored into the gateway, not a separate service.
- `agentic-hub/plans/009-cloud-infra-consolidation` (IN PROGRESS) Phase 009-C will **delete** the cloud Rooben infrastructure (ECR repo, Fargate service, Aurora, secrets).
- **Conclusion: do NOT build a cloud Rooben image. It is going away.**

## Tonight's discoveries already addressed

| Item | Action taken tonight | Permanent fix needed |
|---|---|---|
| Portkey image missing from ECR | Pulled `portkeyai/gateway:latest` from Docker Hub, pushed to `agentic-stack/portkey:latest` | Decide whether to keep Portkey at all (see Phase A vs B below) |
| Hermes saas-mode missing `-v` flag | Task def CMD override to `["gateway", "run", "-v"]` (revision :11) | Phase 009-E (bake into `entrypoint.saas.sh`) |
| Hermes missing PYTHONUNBUFFERED=1 | Added to task def env (revision :10) | Phase 009-E (bake into `Dockerfile.saas`) |
| Hermes missing SLACK_ALLOWED_USERS | Added to task def env (revision :12) | Phase 009-E (bake default into Dockerfile or task def template) |
| Cloud kanban dispatcher enabled (OOM risk) | None yet | Phase 009-F (disable via env var or config mount) |
| Rooben crash-looping (image missing) | Scaled `rooben` service to `desiredCount=0` | Plan 009 scope adjustment: do NOT rebuild — Rooben is being archived |
| Portkey crash-looping then running with no config | Scaled `portkey` service to `desiredCount=0` after image push | Pick Phase A (remove) vs Phase B (configure) below |

## Strategic decision — keep Portkey OR remove?

After the audit, the case for REMOVING Portkey is even stronger than tonight suggested:

**Evidence for removal:**

1. **Zero accumulated value.** No service in cloud has ever successfully called Portkey. There is no cost data, no virtual-key rotation history, no prompt management state. Sunk cost is literally zero.
2. **agentic-hub MASTER_PLAN Phase 001-B (Portkey Fargate deploy) was never decomposed.** Nobody was ever assigned to finish it.
3. **Plan 001-E never owned Portkey config**, only the Hermes container. The Portkey config gap has been hanging unowned for the entire deployment lifetime.
4. **Atlas's Portkey wiring (Plan 007-B in agentic-hub) is probably also broken** — same root cause. Untangling Atlas + Hermes from Portkey together is one bounded change.
5. **OTel collector is already deployed** in agentic-stack — covers the observability story without Portkey.
6. **Local Hermes uses Bedrock directly** with success — proves the direct-provider path is straightforward.
7. **MASTER_PLAN says "Productization posture: explicit none."** Portkey's headline features (rate limiting, virtual keys, multi-tenant cost attribution) aren't needed for single-user.

**Evidence for keeping Portkey:** essentially none beyond the original MASTER_PLAN's aspirational reference. No operational reliance, no production usage.

**Recommendation: take Path A (remove).**

## Phase Index (revised)

| Phase | Title | Effort | Risk | Priority | Status |
|---|---|---|---|---|---|
| 009-A | Remove Portkey from architecture (PATH A) | 4 hr | Med | P0 (recommended) | Not started |
| 009-B | Author Portkey config + push real virtual keys (PATH B alternative) | 1-2 days | High | P0 (alternative) | Not started |
| 009-C | Populate real LLM credentials in Secrets Manager | 1 hr | Low | P0 — gate for both A and B | Not started |
| 009-E | Bake permanent fixes into `Dockerfile.saas` + `entrypoint.saas.sh` | 2 hr | Low | P0 | Not started |
| 009-F | Disable kanban dispatcher in cloud config | 30 min | Low | P0 | Not started |
| 009-G | Promote `agentic-hub/runbooks/rebuild.md` to dual-purpose (first-time bootstrap + disaster recovery) | 1 hr | Low | P1 | Not started |

**Dropped from prior draft:** ~~009-D (Build + push Rooben image)~~ — Rooben is being archived per agentic-hub Plan 005-F + 009-C. Cloud Rooben service was scaled to 0 tonight.

## Phase Detail

### 009-A — Remove Portkey (RECOMMENDED PATH)

**Goal:** Eliminate Portkey from the cloud architecture. Hermes + Atlas talk directly to LLM providers (Bedrock via task IAM, Anthropic via API key).

**Steps:**
1. Update `agentic-stack-hermes` task def: remove `OPENAI_BASE_URL`, replace `OPENAI_API_KEY` secret reference with `ANTHROPIC_API_KEY` (or configure Bedrock provider). Set `model.provider: anthropic` (or bedrock) in mounted config.
2. Update `agentic-stack-atlas` task def: same pattern. (Coordinate with army-of-one Plan 007-B teardown.)
3. Scale `portkey` service to 0 (already done tonight). Delete Portkey ECR images. Delete Portkey task def + service via Terraform.
4. Update `agentic-hub/terraform/modules/secrets/main.tf`: remove `hermes-portkey-virtual-key`, `rooben-portkey-virtual-key`, `portkey-api-key` secrets.
5. Verify each service makes successful LLM calls (Slack mention → Neon row, including assistant reply).

**Acceptance:**
- [ ] `aws ecs describe-services --services portkey` returns "service not found" OR `desiredCount=0` with intent to delete
- [ ] Cloud Hermes processes a real Slack message end-to-end
- [ ] Atlas in cloud still functions for its LLM-using paths (whatever those are)
- [ ] No service has `OPENAI_BASE_URL` env var pointing at portkey internal URL

### 009-B — Configure Portkey (ALTERNATIVE PATH, NOT RECOMMENDED)

Only execute if there's a strategic reason to keep Portkey not captured above. Likely scope:
1. Author `config.yaml` for Portkey with provider definitions (AWS Bedrock via task IAM, Anthropic via real API key) and virtual key → provider mappings
2. Bake config into custom `Dockerfile.portkey` OR mount via Fargate volume
3. Build + push `agentic-stack/portkey:custom-v1`
4. Update Portkey task def to use custom image
5. Populate real virtual key values in Secrets Manager
6. Verify routing end-to-end

### 009-C — Real credentials in Secrets Manager

Per `agentic-hub/runbooks/rebuild.md` Step 2: populate these from your 1Password "agentic-stack" vault:
- `agentic-stack/anthropic-api-key` — real Anthropic API key (used by both Path A and Path B)
- `agentic-stack/hermes-portkey-virtual-key` — Path B only (skip if Path A)
- `agentic-stack/rooben-portkey-virtual-key` — skip (Rooben deprecated)
- Any other LLM-credential secrets the audit surfaces

### 009-E — Permanent fixes for Dockerfile.saas / entrypoint.saas.sh

Move tonight's task-def overrides to permanent code:
- `entrypoint.saas.sh`: change `exec hermes "$@"` to `exec hermes "$@" -v` (saas mode always has stderr logging)
- `Dockerfile.saas`: add `ENV PYTHONUNBUFFERED=1`
- Task def template OR mounted config: include `SLACK_ALLOWED_USERS` (or `GATEWAY_ALLOW_ALL_USERS=true` for single-user cloud)
- Rebuild + push as `plan-001-E-amd64-v9` (or v10 — whatever's next after tonight's v8)

### 009-F — Kanban dispatcher disable in cloud

Cloud config currently shows `kanban dispatcher: embedded in gateway` at startup. Per Plan 002 / 005-A notes, per-worker MCP fan-out OOMs containers. Fix: add `kanban.dispatch_in_gateway: false` to cloud config (via env var, mounted file, or Dockerfile-baked default).

### 009-G — Promote rebuild.md to first-time bootstrap

The runbook gap that caused this whole mess: `agentic-hub/runbooks/rebuild.md` is framed as disaster recovery only. Action:
1. Add a top-section explicitly stating "Use this for first-time bootstrap OR disaster recovery — same steps."
2. Add a pre-flight checklist of all manual steps that must happen after Terraform apply (currently implicit).
3. Cross-link from `agentic-hub/MASTER_PLAN.md` so future plans can find it.
4. Add a "verification" section that includes the "send a real LLM call from a cloud container and verify response" smoke test that was missing from Plan 001-A/B/E AC.

## Acceptance criteria (whole-plan)

- [ ] Cloud Hermes processes a real Slack message end-to-end (Slack mention → Hermes reply → Neon rows for messages + raw_events + skill_feedback)
- [ ] Plan 005-E can complete its acceptance criteria
- [ ] Cloud Hermes survives 48h soak (Plan 005-G) without infrastructure-related restarts
- [ ] No silent failure modes (every error has a visible log line)
- [ ] Decision made + documented: Portkey removed OR kept-and-configured
- [ ] Rooben cloud infrastructure formally retired (deleted task def + ECR repo via Terraform, or queued for 009-C in agentic-hub)
- [ ] `agentic-hub/runbooks/rebuild.md` updated to cover first-time bootstrap with explicit verification step

## Critical files

### Read for context
- `agentic-hub/runbooks/rebuild.md` (THE source of truth for what manual steps were always required)
- `agentic-hub/terraform/modules/secrets/main.tf` (placeholder pattern)
- `agentic-hub/terraform/modules/ecr/main.tf` (empty-repo pattern)
- `agentic-hub/terraform/modules/fargate-portkey/main.tf` (the `:latest` tag reference that never existed)
- `hermes-agent/Dockerfile.saas` + `hermes-agent/docker/entrypoint.saas.sh`
- `hermes-agent/plans/005-fargate-cutover-stage1/phases/phase-005-d.md` (full diagnosis of tonight's discoveries)
- This session's CloudWatch logs (`/ecs/agentic-stack/hermes`, `/ecs/agentic-stack/portkey`)

### Will be modified
- Path A: task defs `agentic-stack-hermes`, `agentic-stack-atlas`; `agentic-hub/terraform/modules/secrets/main.tf`, `modules/fargate-portkey/`; secrets cleanup
- Path B: `Dockerfile.portkey` (new), Portkey config.yaml (new), task defs as above
- Both paths: `hermes-agent/Dockerfile.saas`, `hermes-agent/docker/entrypoint.saas.sh`
- `agentic-hub/runbooks/rebuild.md` (009-G)

## Out of scope

- Plan 008 (MCPGateway production wiring) — still gated on Plan 009 completing
- Atlas LLM routing fixes beyond what's needed for Hermes parity (defer to army-of-one's own plan)
- Multi-tenant LLM-routing setup (single-user; defer until productization decision changes)
- Rebuilding Rooben as a Fargate service (deprecated; in-process is the chosen direction)
