# Plan 008 — MCPGateway Production Wiring (Cloud-Reachable MCP Without Per-Worker Fan-Out)

**Status:** DRAFT 2026-05-25
**Run after:** Plan 005 (Fargate Hermes cutover) complete + 48h soak passed
**Blocks:** nothing critical, but enables tool-call functionality in cloud Hermes
**Estimated effort:** 1 working day (assumes Plan 002-C MCPGateway code is functional; mostly deployment work)

## Context — Why this plan exists

On 2026-05-21 the local launchd Hermes OOM'd the laptop — root cause was each kanban worker spawning its own 7-process MCP stack (pipedrive, linear, exa, google_workspace, atlas.mcp). 61 zombies × 7 procs = ~440 processes, ~25 GB RSS.

**Hermes Plan 002 (Self-Organization)** shipped a `MCPGateway` connection pool in Phase 002-C (Complete 2026-05-19) — exactly to fix this. One central process runs each MCP server once; per-session credential injection is a thin proxy stamp, not a fresh process. Phase 002-D added cloud-routable env vars.

**The gap:** The MCPGateway code shipped but the runtime never adopted it. As of 2026-05-25, every Hermes tool call still spawns local MCP subprocesses per-session. Evidence:
- 2026-05-21 OOM incident happened two days AFTER Plan 002 marked Complete
- `gateway/run.py` has no `MCPGateway` import or initialization at startup
- Per-worker MCP fan-out is the ongoing reason `kanban.dispatch_in_gateway=false` stays pinned

Plan 008 closes that loop: wire MCPGateway into gateway startup, deploy it as a Fargate sidecar (or separate service), and verify cloud Hermes can invoke MCP tools without per-session subprocess fan-out and without unacceptable latency.

## Discovery — what already exists

| Component | State |
|---|---|
| `MCPGateway` class (Phase 002-C) | Code shipped, 16 tests pass per Plan 002 STATUS.md |
| Cloud env-var routing (Phase 002-D) | Code shipped, 18 tests pass |
| Local MCP servers | 5 OAuth/credential setups on Blake's laptop (pipedrive, linear, exa, google_workspace, atlas.mcp) |
| Cloud MCP credentials | NONE — would need to re-OAuth from Fargate's identity, OR proxy through the laptop temporarily |
| Plan 005 (Fargate Hermes) | Ships gateway only; explicitly defers MCP cloud landing to Plan 008 (this plan) |

## Key design decisions

1. **Two-stage deployment.** Stage 1 (this plan): MCPGateway runs as a Fargate sidecar in the same task as Hermes gateway. Single instance, no horizontal scaling. Latency = same-task localhost. Stage 2 (future plan, if scaling demands): MCPGateway moves to its own Fargate service, accessed over VPC.
2. **Credential migration strategy.** For each of the 5 MCP servers, decide per-MCP: (a) re-issue OAuth tokens scoped to a Fargate IAM identity (preferred for long-term), or (b) inject Blake's existing OAuth tokens via Secrets Manager (faster ship, single-user only). Plan defaults to (b) since the stack is single-user; future multi-tenancy work re-opens this.
3. **Connection pooling sanity check.** Re-run a load test with the MCPGateway in place: simulate 10 concurrent tool calls and verify the gateway routes them through the pooled MCP servers without spawning subprocesses per call. If pooling doesn't actually pool, this exposes a Plan 002 bug.
4. **Latency budget.** Tool call P95 latency in cloud Hermes (Fargate Hermes → localhost MCPGateway → MCP server → external API → response) must not exceed 1.5x today's local latency for the same call. If it does, identify the bottleneck before going wider.
5. **Graceful absence.** Cloud Hermes already runs without MCP during Plan 005's 48h soak (Plan 005 Phase E acceptance lists this as expected). MCPGateway adoption must NOT make Hermes hard-depend on it — MCP unavailability should degrade to "no tools available" rather than "gateway crashes."

## Phase Index

| Phase | Title | Effort | Risk | Priority | Status |
|---|---|---|---|---|---|
| 008-A | Audit Plan 002-C MCPGateway: does it actually run? | 2 hr | Low | P0 | Not started |
| 008-B | Wire MCPGateway into `gateway/run.py` startup | 3 hr | Med | P0 | Not started |
| 008-C | Local load test: 10 concurrent tool calls → no per-call subprocess | 2 hr | Med | P0 | Not started |
| 008-D | Migrate 5 MCP credentials to Secrets Manager | 3 hr | Med | P0 | Not started |
| 008-E | Deploy MCPGateway sidecar to Fargate task def | 2 hr | Med | P0 | Not started |
| 008-F | Cloud smoke test: Slack mention → tool call → response, P95 ≤ 1.5x local | 2 hr | Low | P0 | Not started |

## Execution Sequence

A → B → C → (D, E in parallel) → F

## Phase Detail

### Phase 008-A — Audit MCPGateway: does it actually run?

**What:** Plan 002 marked Phase C "Complete (16 tests pass)" but the runtime ignores it. Find out why. Three possible states: (1) MCPGateway exists but is gated behind a feature flag never flipped, (2) it's wired but the call path in `tools/mcp_tool.py` bypasses it, (3) tests pass in isolation but integration with gateway startup is incomplete.

**Files to read:**
- `agent/mcp_gateway.py` or wherever Plan 002-C landed the class
- `gateway/run.py:3854–3877` — adapter init loop (where MCPGateway would plug in)
- `tools/mcp_tool.py` — current MCP invocation path
- `tests/test_mcp_gateway.py` (Plan 002-C tests)

**Acceptance criteria:**
- [ ] Written summary: what state is MCPGateway in (gated / unwired / partial)?
- [ ] Decision recorded: minimum code change to make it active

### Phase 008-B — Wire MCPGateway into gateway startup

**What:** Initialize MCPGateway once at gateway startup, store on `Gateway` instance, route every `tools/mcp_tool.py` call through it. Replace per-session MCP subprocess spawning with pooled-gateway routing.

**Files to modify:** (determined by 008-A audit, but likely)
- `gateway/run.py` — add MCPGateway initialization in startup phase (after storage backend init from Plan 005)
- `tools/mcp_tool.py` — route through gateway instead of direct subprocess
- `gateway/config.py` — add `mcp_gateway.enabled` flag (default True in saas, False in local for now)

**Acceptance criteria:**
- [ ] Gateway startup log shows MCPGateway init line
- [ ] After 5 tool calls, `ps aux | grep -E "pipedrive|linear|exa" | wc -l` returns the same number as before any tool call (i.e., subprocess count is steady-state, not growing per-call)

### Phase 008-C — Local load test

**What:** Use a test harness to fire 10 concurrent `mcp_tool` calls against the same MCP (e.g., `linear`) and verify the gateway serializes/parallelizes correctly without spawning fresh MCP subprocesses. This is the test that Plan 002-C should have run as part of its Phase-C acceptance but apparently didn't (otherwise the 2026-05-21 OOM wouldn't have happened).

**Files to create:**
- `tests/integration/test_mcp_gateway_load.py` — 10-concurrent-call test with subprocess-count assertion

**Acceptance criteria:**
- [ ] Test fires 10 concurrent calls to one MCP
- [ ] Test asserts subprocess count ≤ (baseline + 0)
- [ ] Test asserts P95 latency for the 10 calls ≤ 2x P95 for a single call

### Phase 008-D — Migrate 5 MCP credentials to Secrets Manager

**What:** For each MCP server, identify the credential (OAuth token, API key, etc.), copy into `agentic-stack/hermes/mcp/<mcp-name>` in AWS Secrets Manager, update MCPGateway config to load from Secrets Manager when running in cloud.

**Files to modify:**
- `gateway/config.py` — add MCP credential resolution: env var → Secrets Manager → laptop file
- Per-MCP server config files (likely under `~/.hermes/mcp/` or `agentic-hub/skills/`)

**Acceptance criteria:**
- [ ] All 5 secrets present in AWS Secrets Manager under `agentic-stack/hermes/mcp/*`
- [ ] Local Hermes can still load from filesystem (no regression for local mode)
- [ ] Cloud Hermes (set `HERMES_MODE=saas`) loads credentials from Secrets Manager — verified by a debug print at gateway startup

### Phase 008-E — Deploy MCPGateway sidecar to Fargate task def

**What:** Add the MCPGateway as a second container in the existing `hermes-saas` Fargate task definition. Same task = same network namespace = localhost access. Update Plan 005's deployment to use the new task def revision.

**Files to modify:**
- `infra/terraform/hermes-fargate/main.tf` — add `mcp-gateway` container_definitions entry
- `Dockerfile.mcp-gateway` (new) — minimal container image for MCPGateway

**Acceptance criteria:**
- [ ] `aws ecs describe-tasks` shows 2 containers per task (hermes + mcp-gateway), both RUNNING
- [ ] CloudWatch log group `/ecs/hermes-saas-mcp-gateway` shows MCPGateway init lines
- [ ] Task memory still under 8 GB limit with both containers running

### Phase 008-F — Cloud smoke test

**What:** From Slack, invoke `@Hermes search Linear for issues about UAT` (real tool call). Measure round-trip latency. Compare to local baseline measured pre-cutover.

**Acceptance criteria:**
- [ ] Tool call succeeds end-to-end from cloud Hermes
- [ ] P95 round-trip latency ≤ 1.5x local baseline
- [ ] CloudWatch shows the call hit MCPGateway (not a fresh subprocess)
- [ ] All 5 MCP servers reachable from cloud (run one canary call per MCP)

## Critical files (summary)

### Read for context
- `agent/mcp_gateway.py` (Plan 002-C)
- `gateway/run.py:3854–3877` (adapter init)
- `tools/mcp_tool.py`
- `infra/terraform/hermes-fargate/main.tf`

### Will be modified
- `gateway/run.py` (Phase 008-B)
- `tools/mcp_tool.py` (Phase 008-B)
- `gateway/config.py` (Phases 008-B, 008-D)
- `infra/terraform/hermes-fargate/main.tf` (Phase 008-E)

### Will be created
- `tests/integration/test_mcp_gateway_load.py` (Phase 008-C)
- `Dockerfile.mcp-gateway` (Phase 008-E)
- 5× AWS Secrets Manager entries (Phase 008-D)

## Out of scope (deliberate)

- Horizontal scaling of MCPGateway (Stage 2; only matters at >50 concurrent sessions)
- Per-tenant MCP credential isolation (would require re-OAuth per tenant; single-user stack doesn't need this)
- New MCP server additions (Pipedrive/Linear/Exa/Google Workspace/Atlas are the 5; others come as separate work)
- MCP server health checking / auto-restart (defer to ops layer)
- Cross-region failover (post-MVP)
