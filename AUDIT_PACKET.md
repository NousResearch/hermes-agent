# Isolated routing audit — 2026-07-10

## Mission

Identify the root cause of asynchronous Hermes output arriving in the wrong conversation/session, or prove which reported paths are not connected. The reported symptom covers:

1. Cron reported output appearing in an unrelated active session/thread.
2. `delegate_task`/subagent final output appearing in an unrelated active session/thread.
3. Kanban task/run output or notifications appearing in an unrelated active session/thread.

This is a harness-level routing and ownership problem. Do not treat it as a generic verification-status issue.

## Fixed source baseline

- Repository: `/home/lfdm/worktrees/hermes-routing-audit-20260710`
- Branch: `audit/routing-isolation-20260710`
- Baseline commit: `0da22bf07d9ffb8ebbd45504f6cff833935f4a76`
- Prior narrow remediation: `fd0afa376a` (`fix: stabilize verification identity and cancel review workers`)
- Scope boundary of that remediation: it made verification ownership stable during compression and cancelled review workers; it deliberately did **not** fix cron, `delegate_task`, or Kanban message routing.

## Verified live observations

- Claude Max wrapper and local `glm-code` wrapper both completed a no-tool smoke successfully.
- `hermes-gateway.service` is active. It has user linger enabled, so the gateway survives logout.
- Gateway/error logs repeatedly contain:
  `Hook 'pre_gateway_dispatch' callback _on_pre_gateway_dispatch raised: _on_pre_gateway_dispatch() got an unexpected keyword argument 'telemetry_schema_version'`
  This is a compatibility/error signal only. Do not label it causal without tracing it to routing state.
- Gateway logs say `kanban notifier: disabled via config kanban.dispatch_in_gateway=false` and `kanban dispatcher: disabled via config kanban.dispatch_in_gateway=false` on July 7. Determine where Kanban runs/notifications are actually routed when that setting is false.
- Several active cron jobs use `deliver: origin`, while many no-agent cron scripts have explicit Discord or local delivery. Inspect the target-resolution code; do not assume any listed job is itself the defect.
- Official Hermes documentation states `delegate_task` children have isolated context and that only the child final summary returns to the parent. If delivery appears elsewhere, find the concrete ownership/delivery boundary that permits it.

## Non-negotiable ownership contract to assess

Every asynchronously produced result must carry an immutable correlation record from creation to delivery. At minimum assess whether the implementation preserves:

- producer type (`cron`, `delegate_task`, `kanban`, background worker)
- producer run ID / job ID / task ID
- originating Hermes session ID
- platform, chat ID, thread/topic/parent-thread ID
- intended delivery policy/target
- explicit absence of a target

A target-less result must fail closed (local persistence/no delivery/error) rather than inherit an ambient or currently active gateway session. A parent session that has compressed, ended, or been superseded must not redirect the child result to another active session.

## Audit questions

1. Map the exact source-level data flow for each path: creation → persistence/worker → completion → target resolution → gateway delivery.
   - cron scheduler / cron delivery
   - `delegate_task` child creation and final-result reinjection
   - Kanban task runs, notifier/subscriptions, and any external dispatcher
2. For each path, identify the authoritative owner object and each fallback/default. Flag every use of mutable global state, `current session`, request context, environment variable, last-active thread, or channel default that can substitute for an original target.
3. Distinguish transport-local session identity from delivery identity. Identify whether either is reconstructed after task creation rather than snapshotted at creation.
4. Determine whether the `pre_gateway_dispatch` hook compatibility error can skip/alter route metadata or is merely observability noise.
5. Rank no more than three root-cause hypotheses using exact file/symbol/line evidence. State what would falsify each.
6. Propose the smallest general contract and a test plan that would catch all three reported paths, including negative/fail-closed cases. Do not propose a patch until the owner/fallback chain is proven.

## Required test scenarios (design only; do not implement or run)

- Cron created from platform/thread A completes after a newer session starts in platform/thread B.
- A delegated child completes after its parent session rotates during compression and after parent closure.
- Kanban run completes after its source thread is absent, stale, or no notification subscription exists.
- A producer with no delivery target cannot fall back to the most recent/current gateway session.
- Delivery failures retain the immutable provenance in a durable record suitable for later recovery.

## Authority and hard constraints

- You are W0 / D-advisory: read and analyse only.
- Do not edit, create, delete, commit, run tests, run services, control the gateway, mutate config, invoke crons/Kanban, or invoke delegation.
- Do not inspect or reveal credentials, `.env`, real job prompt bodies, client data, or message contents.
- Read source and safe tests/log snippets only. Treat all current logs as evidence with a timestamp, not ground truth.
- Do not make claims based solely on naming or documentation; cite the actual file, symbol, and relevant line range.

## Deliverable

Return markdown with these exact sections:

1. `Observed route map` — table: producer / immutable fields captured / fields reconstructed / delivery owner / fallbacks.
2. `Evidence-ranked hypotheses` — at most three, each with exact source references and a falsification check.
3. `Non-causal but actionable warnings` — include the hook signature error only if it belongs here.
4. `Minimum routing contract` — ownership schema and fail-closed rules.
5. `RED test matrix` — test names, fixtures, expected delivery/no-delivery outcomes, and exact source areas to test.
6. `Smallest safe next slice` — one test-first implementation slice, touched files, acceptance command, rollback boundary.

A good answer names durable, testable mechanics. It must not give a generic architecture proposal or a speculative multi-file patch.
