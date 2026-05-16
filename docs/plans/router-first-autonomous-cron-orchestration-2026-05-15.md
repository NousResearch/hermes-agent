# Router-First Autonomous Cron Orchestration: Plan

## Goal
Move routine autonomous work — cron maintenance, agent heartbeats, Zulip cleanup, Paperclip queue execution, and Hermes supervision — to a router-first model where `blockrun/auto` can use Sparky/GN100 local inference for simple work and escalate automatically to stronger models for coding, tool-heavy, or reasoning-heavy work, with Hermes reviewing routing outcomes.

## Background

### Locked decisions
- Broad migration target: router-first with explicit opt-out pins, not a narrow lightweight-only migration.
- `exec` jobs are not model work by themselves. Leave plain script-only exec jobs alone; model routing matters when those scripts spawn or wake agent work.
- Escalation policy: allow automatic escalation to stronger models when classification requires it, with routing logs and Hermes review so the supervisor can improve over time.
- CronRouter remains scheduler/executor/hint layer; OpenClaw/ClawRouter own actual model/provider selection.

### Current GN100/Sparky routing state
- GN100 is reachable through the Mac-local Ollama tunnel at `127.0.0.1:11436`, with `qwen3.6:35b-a3b` installed.
- OpenClaw has a direct provider `gn100/qwen3.6:35b-a3b` in `~/.openclaw/openclaw.json`.
- OpenClaw default agent model was set to `blockrun/auto` with direct GN100 and stronger cloud/CLI fallbacks.
- `openclaw-cli-proxy` is launched by `ai.openclaw.claude-cli-proxy` and now uses `OLLAMA_API_BASE=http://127.0.0.1:11436`; existing local aliases like `tool-local` and `local/ollama-qwen3.5-*` resolve to the GN100 model in the runtime-main worktree.
- Hermes uses `.hermes/config.yaml` with `model.default: "blockrun/auto"` and `provider: "openrouter"`; `.hermes/.env` routes OpenAI/OpenRouter-compatible calls through `http://127.0.0.1:11435/v1`. Treat `11435` as the deployed canonical Hermes endpoint unless the validation step proves direct `8402` should replace it.

### Cron execution seams
- Cron payloads are a discriminated union of `systemEvent`, `agentTurn`, and `exec` in `src/cron/types.ts:145`; `agentTurn` carries optional `model`, `fallbacks`, `thinking`, `timeoutSeconds`, `lightContext`, and `toolsAllow` in `src/cron/types.ts:168`; `exec` carries `command`, `workdir`, `env`, and `timeoutSeconds` in `src/cron/types.ts:136`.
- Main-session cron jobs use `executeMainSessionCronJob`; detached jobs use `executeDetachedCronJob` in `src/cron/service/timer.ts:1106`. Non-main `exec` jobs call `runDeterministicExecJob` in `src/cron/service/timer.ts:1272`; non-main `agentTurn` jobs call `runIsolatedAgentJob` in `src/cron/service/timer.ts:1285`.
- Isolated cron agent turns are prepared in `src/cron/isolated-agent/run.ts:220`, then dispatched through `executeCronRun` in `src/cron/isolated-agent/run-executor.ts:273`.
- `createCronPromptExecutor` wraps execution in `runWithModelFallback(...)` in `src/cron/isolated-agent/run-executor.ts:115`, then chooses CLI-backed execution via `runCliAgent(...)` in `src/cron/isolated-agent/run-executor.ts:139` or embedded provider execution via `runEmbeddedPiAgent(...)` in `src/cron/isolated-agent/run-executor.ts:174`.
- Model selection starts with configured defaults in `src/cron/isolated-agent/model-selection.ts:55`, then applies agent/subagent overrides, hook overrides, explicit `payload.model`, session overrides, and `pacing.providerTarget`/inferred routing in `src/cron/isolated-agent/model-selection.ts:81`, `:108`, `:133`, `:178`, and `:220`.
- Current guardrail rejects/filters unattended `auto` and `blockrun/auto` in `src/cron/model-guard.ts:9`; this is the central contradiction with router-first unattended cron.
- `payload.fallbacks` wins when present; otherwise fallback policy depends on selection source in `src/cron/isolated-agent/run-fallback-policy.ts:17` and `:25`.

### CronRouter and routing hint seams
- CronRouter supports `routingHints` with `budget` and `modelChain`, but treats them as advisory metadata only; runtime model selection belongs downstream in OpenClaw/ClawRouter (`lionroot-cronrouter/src/registry/job-types.ts:12`).
- Gateway cron loading preserves `task`, `source`, `budget`, and `modelChain` from `~/.openclaw/cron/jobs.json` in `lionroot-cronrouter/src/registry/jobs-gateway.ts:141`.
- Paperclip maintenance jobs are generated from a central `PAPERCLIP_MAINTENANCE_SPECS` table and tagged as gateway/paperclip/maintenance in `lionroot-cronrouter/src/registry/jobs.ts:17`.
- Prior plans explicitly state CronRouter is a scheduler/executor router, not a model router; OpenClaw/ClawRouter own provider/model selection (`docs/plans/local-worker-pool-capacity-routing-2026-05-14.md:29`, `:47`, `:168`; `docs/plans/model-burndown-routing-2026-05-09.md:245`).

### Paperclip and Hermes seams
- Paperclip maintenance dispatch enters through `lionroot-command-post/command-post/dashboard/app/api/paperclip/maintenance/dispatch/route.ts:25-65` and calls `dispatchMaintenanceWork(...)`.
- `TASK_CONFIG` maps task kind to tracker, assignee, runner, repo selection, and Zulip target in `lionroot-command-post/command-post/dashboard/lib/paperclip/maintenance-bridge.ts:65-225`.
- `paperclip-heartbeat` work wakes an existing Paperclip/OpenClaw agent via `invokeAgentHeartbeat(assignee.id)` in `maintenance-bridge.ts:960-974`; this should inherit OpenClaw router-first defaults.
- `worker-repoprompt` work enqueues maintenance execution via `maintenance-bridge.ts:978-1038` and `maintenance-worker-client.ts:75-117`; the worker accepts `/api/paperclip/maintenance/execute` in `scripts/maintenance-worker.ts:604-632`.
- Worker execution currently defaults safe remediation provider order to `codex`, then `claude`, overridable by `SAFE_REMEDIATION_PROVIDER_ORDER` in `repoprompt-executor.ts:315-332`; Codex uses CLI defaults at `:286-296`, while Claude pins `--model sonnet` at `:298-307`.
- Hermes supports model aliases as exact `(model, provider, base_url)` tuples in `.hermes/hermes-agent/hermes_cli/model_switch.py:157-214`, and direct alias `base_url` can override runtime base URL in `.hermes/hermes-agent/hermes_cli/model_switch.py:850-860`.

### Prior art and constraints
- `docs/plans/model-burndown-routing-2026-05-09.md:16` describes the existing control loop: Command Post/CodexBar telemetry → routing hints file → OpenClaw fallback ordering/filtering → agent/cron execution.
- `docs/plans/model-burndown-routing-2026-05-09.md:72` and `:88` keep Codex preferred for coding while allowing unattended/non-coding automation to shift across Claude/Gemini/local based on capability and burndown state.
- `.workflow/multi-machine-plan-v2-skeleton-2026-05-04.md:76` called out that CronRouter had zero burndown awareness and gateway jobs fired LLM calls regardless of burndown.
- `.workflow/orchestration-2026-05-03-cron-gateway-sweep.md:79` records a standalone cron linter validating session target, payload kind, and schedule shape.
- `.workflow/item-7-fn6-paperclip-rollout-guide.md:172` defines Paperclip bounded parallel dispatch with per-repo lanes, blocker-aware waves, telemetry, and rollout monitoring; `PAPERCLIP_MAX_PARALLEL_ISSUES=1` remains the safety default in `.workflow/item-7-fn6-paperclip-rollout-guide.md:3`.

## Approach

Implement router-first orchestration by changing the existing OpenClaw cron policy and observability seams, not by adding a second model router. `blockrun/auto` becomes an allowed unattended cron model route; raw `auto` remains blocked as ambiguous. Existing explicit non-router model selections become auditable opt-out pins using the marker syntax `[router-pin: <reason>]` in the cron job description or payload message.

The first migration wave should make normal cron `agentTurn`, main-session wake, Paperclip heartbeat, and Hermes supervisor work router-first. Plain deterministic `exec` jobs remain script jobs. Coding-heavy Paperclip worker remediation remains pinned to Codex/Claude in phase one because it currently depends on deterministic code-edit CLIs and sandbox behavior, but that pin must become visible in execution summaries and policy checks.

Hermes should review routing outcomes, not approve every escalation synchronously. Automatic escalation through ClawRouter/OpenClaw is allowed when classification indicates coding, tool-heavy, or reasoning-heavy work. Hermes receives logs/status and flags routing failures: simple work not using local when available, coding work failing to escalate, missing resolved-model metadata, repeated local failures, or unexplained explicit pins.

## Work Items

### Item 1 — Lock router-first cron guard semantics
**Goal:** Replace the current all-auto cron block with a narrower policy: raw `auto` is ambiguous and blocked; `blockrun/auto` is an allowed router-first model ref.

**Done when:**
- `src/cron/model-guard.ts` exposes a single shared classifier used by cron selection, fallback filtering, and policy doctor.
- Recommended helper shape is available or equivalently covered: `classifyCronModelRef`, `isAllowedCronRouterRef`, and `isAmbiguousCronAutoRef`.
- `blockrun/auto` and `BLOCKRUN/AUTO` are preserved in fallback lists and allowed as selected model.
- Raw `auto` remains rejected or filtered everywhere unattended cron currently needs protection.
- Empty or undefined model refs continue to mean “no explicit override,” not an error.

**Key files:**
- `src/cron/model-guard.ts:9`
- `src/cron/isolated-agent/run-fallback-policy.ts:17`
- `src/cron/isolated-agent/run-fallback-policy.test.ts`
- `src/cron/isolated-agent/run.payload-fallbacks.test.ts`

**Dependencies:** None.

**Size:** S.

### Item 2 — Preserve router-first default through cron model selection
**Goal:** Ensure cron model selection does not immediately override a router-first default with inferred or pacing-derived Claude/Codex/Gemini targets.

**Done when:**
- `resolveCronModelSelection` allows selected provider/model `blockrun/auto`.
- When current/default selection is router-first, `pacing.providerTarget` and inferred provider routing are advisory metadata, not forced model replacement.
- Explicit model sources still pin the selected model.
- Tests prove router-first default survives providerTarget inference and raw `auto` remains rejected.

**Key files:**
- `src/cron/isolated-agent/model-selection.ts:55`
- `src/cron/isolated-agent/model-selection.ts:220`
- `src/cron/provider-routing.ts:66`
- `src/gateway/protocol/schema/cron.ts:69`
- `src/cron/isolated-agent/run.cron-model-override.test.ts`

**Dependencies:** Item 1.

**Size:** M.

### Item 3 — Update cron fallback and policy-doctor behavior
**Goal:** Make policy validation and fallback behavior match router-first semantics.

**Done when:**
- `payload.fallbacks` can include `blockrun/auto`; only raw `auto` is filtered.
- Policy doctor uses the shared Item 1 classifier, not duplicated string matching.
- Policy doctor reports raw `auto` as critical.
- Policy doctor reports explicit non-router `payload.model` as an opt-out pin warning unless job text contains `[router-pin: <reason>]`.
- Policy doctor warns when router-first jobs have no usable fallback chain.
- Existing delivery, timeout, session target, and payload-shape checks remain unchanged.

**Key files:**
- `src/cron/isolated-agent/run-fallback-policy.ts:17`
- `src/cron/policy-doctor.ts`
- `src/cron/policy-doctor.test.ts`
- `.workflow/orchestration-2026-05-03-cron-gateway-sweep.md:79`

**Dependencies:** Items 1 and 2.

**Size:** M.

### Item 4 — Add routing outcome observability for Hermes review
**Goal:** Make router decisions inspectable enough for Hermes to review whether work used local GN100 when appropriate and escalated when required.

**Done when:**
- Before loosening production schedules, a live probe confirms whether ClawRouter/openclaw-cli-proxy returns the resolved upstream model in OpenAI-compatible responses.
- If the resolved upstream model is exposed, OpenClaw calls `updateLastDecisionResolvedModel()` at the response-parsing/fallback call site.
- If it is not exposed, the implementation adds a small protocol/logging seam before broad cron migration proceeds.
- Router-first cron runs record selected model (`blockrun/auto`) and resolved upstream model when available.
- Missing resolved-model metadata is detectable as a warning condition, not silently accepted.

**Key files:**
- `src/agents/routing-decisions.ts`
- `src/agents/model-fallback.ts:411`
- `src/cron/isolated-agent/run-executor.ts:115`
- `src/cron/isolated-agent/run.ts:486`

**Dependencies:** Items 1-3.

**Size:** M, or L if the live probe shows ClawRouter needs a protocol/logging addition.

### Item 5 — Make Paperclip routing policy explicit
**Goal:** Keep Paperclip heartbeat work router-first and make coding-heavy worker remediation an explicit, visible opt-out pin.

**Done when:**
- Paperclip heartbeat dispatch responses/reporting identify routing policy as router-first intent and explain that the actual model is resolved by OpenClaw/ClawRouter defaults.
- If Item 4 resolved-model evidence is available, Paperclip reports include or link to it; if not, Paperclip marks the field as policy intent only.
- `worker-repoprompt` / safe remediation output identifies routing policy as pinned with `[router-pin: safe remediation requires deterministic Codex/Claude CLI]` or equivalent structured reason.
- `SAFE_REMEDIATION_PROVIDER_ORDER` remains supported.
- No plain deterministic `exec` job is rewritten just to add model routing.
- A follow-up issue or plan note captures any future `router-agent` worker provider work; it is not part of phase one.

**Key files:**
- `lionroot-command-post/command-post/dashboard/lib/paperclip/types.ts:206`
- `lionroot-command-post/command-post/dashboard/lib/paperclip/maintenance-bridge.ts:65`
- `lionroot-command-post/command-post/dashboard/lib/paperclip/maintenance-bridge.ts:960`
- `lionroot-command-post/command-post/dashboard/lib/paperclip/repoprompt-executor.ts:315`
- `lionroot-command-post/command-post/dashboard/scripts/maintenance-worker.ts:604`

**Dependencies:** Items 3 and 4.

**Size:** M.

### Item 6 — Add Hermes routing review loop
**Goal:** Have Hermes supervise router outcomes without blocking automatic escalation.

**Done when:**
- Hermes config makes the deployed `blockrun/auto` endpoint explicit. On this machine the starting assumption is `http://127.0.0.1:11435/v1` through `openclaw-cli-proxy`.
- Endpoint validation probes both `http://127.0.0.1:11435/v1/models` and `http://127.0.0.1:8402/v1/models` when both are present; choose `11435` unless direct `8402` is intentionally the supported deployment route.
- A scheduled Hermes review job inspects routing outcomes and reports `[ACK]` or `[ESCALATION_NOTICE]` using the existing supervisor protocol.
- The review checks for: local-capable simple work not using local, coding/reasoning work not escalating, missing resolved model, repeated local failures, and unexplained opt-out pins.
- Hermes remains a verifier/router supervisor, not a content-generation worker.

**Key files:**
- `.hermes/config.yaml:1`
- `.hermes/cron/jobs.json`
- `.hermes/hermes-agent/hermes_cli/model_switch.py:157`
- `.hermes/hermes-agent/hermes_cli/model_switch.py:850`
- `.hermes/hermes-agent/website/docs/integrations/providers.md:1008`

**Dependencies:** Item 4.

**Size:** M.

### Item 7 — Operational migration, CronRouter regression, and smoke verification
**Goal:** Apply the policy to enabled jobs, preserve CronRouter boundaries, and prove router-first orchestration works before leaving it unattended.

**Done when:**
- Enabled cron jobs are audited after Items 1-4 land and classified as router-first eligible, explicit opt-out pin, script-only exec, or exec that wakes/spawns agent work.
- Router-first eligible jobs either inherit default `blockrun/auto` or explicitly use it where needed.
- Explicit pins have `[router-pin: <reason>]` markers.
- CronRouter docs/tests confirm it preserves hints and does not inject `payload.model` or choose runtime models.
- Runtime smoke proves: simple prompt resolves to GN100/Sparky local path; coding/reasoning-heavy prompt escalates; Hermes review reports healthy or actionable status; Paperclip heartbeat uses router-first policy; worker remediation remains pinned and visible.
- Rollback is tested on one sample job by re-pinning an affected cron `agentTurn` job to its previous explicit model.

**Key files:**
- `~/.openclaw/cron/jobs.json`
- `~/.openclaw/openclaw.json`
- `.hermes/cron/jobs.json`
- `lionroot-cronrouter/docs/architecture.md`
- `lionroot-cronrouter/src/registry/job-types.ts:12`
- `lionroot-cronrouter/src/registry/jobs-gateway.ts:141`
- `lionroot-cronrouter/src/executors/gateway-executor.ts`
- `lionroot-cronrouter/src/registry/paperclip-automation-jobs.test.ts`
- `docs/plans/router-first-autonomous-cron-orchestration-2026-05-15.md`

**Dependencies:** Items 1-6.

**Size:** L.

## Verification Plan
- OpenClaw targeted tests:
  - `pnpm test src/cron/isolated-agent/run.cron-model-override.test.ts`
  - `pnpm test src/cron/isolated-agent/run.payload-fallbacks.test.ts`
  - `pnpm test src/cron/isolated-agent/run-fallback-policy.test.ts`
  - `pnpm test src/cron/policy-doctor.test.ts`
- CronRouter targeted tests:
  - Run the existing Paperclip automation and gateway executor tests in `lionroot-cronrouter`.
- Command Post targeted tests:
  - Run Paperclip maintenance bridge/worker tests that cover dispatch, worker execution, and response shape.
- Runtime smoke:
  - Direct GN100 route: OpenClaw model run against `gn100/qwen3.6:35b-a3b`.
  - Router simple route: `blockrun/auto` prompt that should resolve to local/GN100.
  - Router escalation route: coding/tool-heavy prompt that should escalate to Codex/Claude/Gemini.
  - Hermes route: supervisor prompt using its configured endpoint.
  - Paperclip heartbeat: one safe maintenance heartbeat or dry-run showing router-first policy metadata.

## Rollback
- Re-pin affected cron `agentTurn` jobs by setting explicit `payload.model` to the previous Claude/Gemini/Codex model.
- Restore OpenClaw default model from the pre-migration backup.
- Disable the Hermes routing review cron job if it creates noisy false positives.
- Keep CronRouter unchanged; do not roll back by moving model routing into CronRouter.
- For Paperclip worker failures, keep or restore `SAFE_REMEDIATION_PROVIDER_ORDER=codex,claude` and leave heartbeat router-first migration intact.

## Open Questions
- Which exact existing cron jobs should remain explicit opt-out pins after the inventory is produced?
- Does ClawRouter currently expose the resolved upstream model in a way OpenClaw can persist for Hermes review, or does Item 4 need a small protocol/logging addition?

## References
- `docs/plans/model-burndown-routing-2026-05-09.md`
- `docs/plans/local-worker-pool-capacity-routing-2026-05-14.md`
- `.workflow/multi-machine-plan-v2-skeleton-2026-05-04.md`
- `.workflow/orchestration-2026-05-03-cron-gateway-sweep.md`
- `.workflow/item-7-fn6-paperclip-rollout-guide.md`
