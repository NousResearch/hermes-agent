# Hermes Target Architecture

Date: 2026-05-20

## Target Principle

Hermes should evolve through a staged control plane, not a one-shot rewrite.
The current gateway, tool registry, memory system, plugin layer, cron, kanban,
API server, and Operator wrappers are valuable rails. The target architecture
adds policy, ledgering, evaluation, and repeatable optimization stages around
them.

## Core Shape

```
User / CLI / Telegram / API / Cron
        |
        v
Hermes Gateway + CLI Surfaces
        |
        v
Optimization Control Plane
  - objective ledger
  - stage state machine
  - risk and approval policy
  - model/tool router
  - tool/control registry
  - judge process
        |
        v
Execution Rails
  - AIAgent
  - tools.registry
  - plugin hooks
  - API /v1/runs
  - kanban workgraphs
  - cron no-agent jobs
  - Operator scripts
        |
        v
Evidence And Memory
  - tests and smoke checks
  - structured logs
  - audit JSONL
  - sessions/state DB
  - curated memory
  - holographic facts
```

## Module Boundaries

### Core Agent

Owns:

- Provider/model calls.
- Prompt building.
- Tool execution loop.
- Context/memory injection.
- Iteration budget and failure handling.

Primary files:

- `run_agent.py`
- `agent/`
- `model_tools.py`

Do not embed optimization business logic here. Add seams only when the core
needs a generic lifecycle hook or routing input.

### Tool Registry

Owns:

- Tool schemas.
- Tool handlers.
- Toolset membership.
- Availability checks.
- Tool dispatch metadata.

Primary files:

- `tools/registry.py`
- `tools/*.py`
- `toolsets.py`

Target upgrade:

- Add generated registry metadata outside the dispatch path first.
- Later add risk/cost/permission metadata once the read-only inventory is
  stable and tested.

### Gateway

Owns:

- Messaging/API runtime.
- Platform adapters.
- Active sessions.
- Delivery routing.
- Gateway slash commands.
- Cron ticker integration.
- Runtime health state.

Primary files:

- `gateway/run.py`
- `gateway/session.py`
- `gateway/status.py`
- `gateway/platforms/*`

Target boundary:

- Gateway remains the live runtime nucleus.
- Control-plane orchestration lives in plugin/API/CLI surfaces, not directly in
  `gateway/run.py`.

### CLI And Ops

Owns:

- User-facing commands.
- Setup/config/status/doctor/logs.
- Plugin management.
- Dashboard entrypoints.

Primary files:

- `hermes_cli/main.py`
- `hermes_cli/commands.py`
- `hermes_cli/status.py`
- `hermes_cli/doctor.py`
- `hermes_cli/gateway.py`
- `hermes_cli/plugins.py`

Target upgrade:

- Add `hermes control inventory --json --redact` or equivalent.
- Later add `hermes ops` as a compact wrapper over status, doctor, gateway,
  logs, cron, and health-loop receipts.

### Plugin Layer

Owns:

- New capabilities that should not hardcode into the gateway or core agent.
- Hook-based policy enforcement.
- Optional tools and providers.

Primary file:

- `hermes_cli/plugins.py`

Target upgrade:

- Implement the optimization control plane as a plugin-backed layer.
- Use hooks for `pre_tool_call`, `post_tool_call`,
  `pre_approval_request`, `post_approval_response`, and session lifecycle.

### Memory

Owns:

- Always-on curated memory.
- External memory providers.
- Session search.
- Structured facts.

Primary files:

- `tools/memory_tool.py`
- `tools/session_search_tool.py`
- `agent/memory_manager.py`
- `agent/memory_provider.py`
- `plugins/memory/*`

Target upgrade:

- Keep a tiered memory strategy.
- Add reconciliation for markdown memory replace/remove against structured
  facts.
- Add deletion and privacy checklist.

### Runtime And Operator Layer

Owns:

- Launchd services.
- Keychain-backed environment loading.
- Health guardian.
- Local operator scripts.

Primary files:

- `/Users/agent1/Operator/scripts/hermes-gateway.sh`
- `/Users/agent1/Operator/scripts/hermes-env.sh`
- `/Users/agent1/Operator/scripts/agent-health-guardian.sh`
- `/Users/agent1/Operator/scripts/hermes-command-center.sh`

Target boundary:

- Keep scripts thin and auditable.
- Do not duplicate gateway lifecycle logic in multiple wrappers.
- Treat `ai.hermes.gateway` as canonical.

## Optimization Control Plane Components

### Objective Ledger

Tracks:

- Objective ID.
- Request/source.
- Stage.
- Risk class.
- Tool/model policy.
- Files/artifacts touched.
- Commands run.
- Validation evidence.
- Judge results.
- Remaining work.

First implementation should be local-only, append-safe, and readable as JSONL
or SQLite. Existing Mission Control can be wrapped rather than replaced.

### Stage Engine

Standard lifecycle:

1. Intake.
2. Discovery.
3. Design.
4. Plan.
5. Dry-run or scaffold.
6. Implementation.
7. Validation.
8. Judge.
9. Report.
10. Memory/docs update when approved.

Each stage has:

- Allowed tools.
- Required evidence.
- Required approval policy.
- Exit criteria.

### Policy Gate

Risk classes:

- `R0`: read-only local inspection.
- `R1`: docs or non-runtime metadata changes.
- `R2`: workspace-scoped code changes.
- `R3`: local runtime/service changes.
- `R4`: external sends, account writes, publishing, deploys, credential edits.
- `R5`: spend, trading, money movement, destructive/system operations.

Default:

- `R0-R1`: allowed with logging.
- `R2`: allowed after repo safety check.
- `R3`: confirmation or drain-aware procedure.
- `R4-R5`: explicit typed confirmation; many actions remain denied by default.

### Model And Tool Router

Route by stage:

- Intake/preflight: deterministic code or cheap/local route.
- Architecture/planning: strongest model available.
- Implementation: constrained toolsets and workspace-specific context.
- Verification: tests and smoke checks first, LLM only for ambiguous diagnosis.
- Security judging: independent pass with high scrutiny.
- Reporting: concise summarization.

Do not rely only on provider failover. Add stage-aware desired quality, cost,
latency, tool, and risk policy.

### Tool Registry Overlay

Read-only first. Merge:

- Built-in registry tools.
- Toolsets.
- Plugins and plugin manifests.
- MCP servers.
- Quick commands.
- Cron jobs.
- Launchd services.
- Operator scripts.
- Credential presence booleans.
- Health probes.
- Cost and risk labels.

The registry should produce stable JSON and Markdown without exposing secrets.

### Audit Trail

For gated or mutating actions, append:

- Timestamp.
- Session/source.
- Actor/surface.
- Risk tier.
- Tool/command.
- Target paths/external target, redacted.
- Approval prompt and decision.
- Exit code/status.
- Before/after hashes when feasible.
- Validation receipt paths.

Use `0600` files under `~/.hermes/audit/`.

## Extensibility Plan

Phase order:

1. Control-plane docs and judge rubric.
2. Read-only inventory registry.
3. Policy/risk labels.
4. Audit log.
5. Memory hygiene and deletion workflow.
6. Ops command/dashboard.
7. Domain workflow packs.

Workflow packs should define:

- Stage template.
- Allowed tools.
- Required confirmations.
- Evidence files.
- Smoke checks.
- Report format.

Candidate packs:

- Hermes runtime maintenance.
- Research brief.
- Local file processing.
- Business intelligence report.
- Content draft pipeline.
- Coding assistant review/fix.
- App/TestFlight release readiness.
- AWS scout, strictly read-only by default.

## Non-Goals

- Do not replace `GatewayRunner` with a second orchestrator.
- Do not rewrite the tool registry before inventorying it.
- Do not make external actions autonomous by default.
- Do not move secrets into config files or docs.
- Do not enable paid/provider-heavy features without an explicit gate.
