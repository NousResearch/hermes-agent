# Dashboard Ultimate Gap Assessment

This document separates two different ideas that were previously getting blurred:

- **Design-system foundation complete:** V1-V7 created the kit, adapter, recipes, scaffold, gallery, and validation rails.
- **Ultimate operating-dashboard goal complete:** every TLC/Hermes dashboard is package-native, live-data-backed, visually premium, agent-governed, production-tested, and connected into a central Hermes command layer.

Current assessment: the foundation is strong, but the ultimate goal is still only partially built.

## Where We Are

| Capability | Current State | Estimated Ultimate Completion |
|---|---|---:|
| Component kit | Real package with shell, metrics, tables, charts, launcher, operations, executive components | 85% |
| Static dashboard adapter | Real and used by Khashi VC, Media Engine, and other dashboards | 80% |
| Full-page recipes | V7 added eight approved recipes and recipe-aware scaffolding | 85% |
| Package-native dashboard adoption | Hermes OS is closest; Khashi and Media Engine are still adapter/static surfaces | 30% |
| Live executive control plane | Reference route exists, but live project cost/capacity/health data is shallow | 25% |
| Production visual QA | Local/static visual checks exist; live production screenshot checks are limited | 35% |
| Premium visual polish | Improved consistency, but not yet a polished, product-grade redesign across dashboards | 40% |
| Agent governance | Planning docs and validators exist; Codex/Hermes are not yet forced through recipe/design review gates everywhere | 55% |
| Cross-project registry | Manifests and launcher exist; source-of-truth and production-sync still need hardening | 65% |
| Data contracts | Some route-level and dashboard plugin signals exist; per-project API contracts are inconsistent | 30% |
| Cost/capacity observability | Concepts and dashboards exist in places; no universal operating telemetry contract yet | 35% |
| Deployment confidence | Shared deploy rails exist in some projects; not all dashboards have identical promotion/rollback/verification paths | 50% |

Overall ultimate-dashboard estimate: **45-55%**.

## Where We Are Trying To Be

The ultimate goal is a Hermes-controlled dashboard ecosystem where:

- Every active TLC project exposes a production dashboard.
- Every dashboard is reachable from the launcher and from Hermes executive summary.
- Every dashboard uses a package-native `@hermes/dashboard-kit` implementation unless there is a documented reason not to.
- Every dashboard has a live data contract with health, cost, capacity, queue, activity, and action-needed signals.
- Hermes can summarize what is happening across all projects without opening every dashboard.
- Codex/Hermes can create new dashboards from approved recipes, not from improvisation.
- Production deployments include screenshot, accessibility, registry, health, and smoke validation.
- Dashboard quality is judged by operating usefulness, not only by whether components render.

## Main Gaps

### 1. Static Adapter Dependency

Current gap:

- Khashi VC and Media Engine are still mostly static/generated dashboards with `hdk-*` adapter classes.
- This improves consistency but does not fully solve layout quality, state management, reusable data fetching, or richer interactions.

What needs to happen:

- Convert priority dashboards to package-native React routes.
- Keep static adapters as a bridge only until parity is proven.

### 2. Weak Data Contracts

Current gap:

- The design system knows how to display cost, capacity, health, queues, and research, but each project exposes those signals differently.
- The executive summary cannot become a true source of truth until projects expose standard dashboard APIs.

What needs to happen:

- Define `DashboardSnapshot`, `HealthSnapshot`, `CostSnapshot`, `CapacitySnapshot`, `QueueSnapshot`, and `ActionNeeded` contracts.
- Add adapters in each project that map local data into those contracts.

### 3. Executive Layer Is Still A Reference

Current gap:

- `/executive-summary` exists, but it is not yet the full TLC command layer.
- It needs live signals from Khashi VC, Media Engine, Media Business Ops, Hermes OS, and future businesses.

What needs to happen:

- Add per-project signal endpoints.
- Add cross-project rollups, action queues, trends, and launch links.
- Add “what changed since yesterday” and “what needs attention now.”

### 4. Visual Quality Is Tested, But Not Fully Scored

Current gap:

- Playwright and axe checks catch obvious breakage.
- They do not score whether a dashboard looks premium, dense enough, clear enough, or decision-ready.

What needs to happen:

- Add a dashboard quality scorecard.
- Add screenshots for live production URLs.
- Add before/after design reviews for the highest-value dashboards.

### 5. Agent Governance Is Advisory, Not Mandatory

Current gap:

- The docs tell agents what to do, but dashboard tasks can still bypass recipes, contracts, or review gates.

What needs to happen:

- Require every new dashboard to declare a V7 recipe.
- Require a data contract before UI build.
- Require validation before marking complete.

### 6. Cross-Project Deployment Is Not Uniform Enough

Current gap:

- Some dashboards deploy cleanly; others still depend on project-specific routes, secrets, Caddy config, compose services, or manual production verification.

What needs to happen:

- Standardize deploy manifests.
- Standardize health endpoints.
- Standardize promotion and rollback checks.
- Add dashboard registry verification after deployment.

### 7. Observability Is Fragmented

Current gap:

- Cost, tokens, API calls, CPU, storage, queue depth, failed jobs, and scheduler pressure are not normalized across projects.

What needs to happen:

- Add a common operations telemetry model.
- Feed the Cost And Capacity recipe from real data.
- Roll up telemetry into Hermes executive summary.

### 8. Design Tokens Are Not Yet Brand/Domain Mature

Current gap:

- The kit has theme-compatible tokens, but not a mature multi-brand system for TLC, Khashi VC, Media Engine, and future businesses.

What needs to happen:

- Add theme packs or domain skins.
- Keep layout consistent while allowing project-level identity.
- Validate contrast and visual coherence across themes.

### 9. Package Distribution Is Still Local-Workspace Or File-Based

Current gap:

- The kit exists as a workspace package, but adoption across independent projects can still be brittle.

What needs to happen:

- Decide whether to publish internally, use Git dependency, or standardize a workspace/submodule approach.
- Add version/change management that downstream dashboards can follow.

### 10. Dashboard Data Is Not Yet Decision-Grade

Current gap:

- Many dashboards display state, but not always the reason, confidence, freshness, owner, next action, or business impact.

What needs to happen:

- Require every major dashboard panel to answer:
  - What happened?
  - Why does it matter?
  - How fresh is it?
  - Who owns it?
  - What should happen next?

## Recommended Future Versions

### V8: Package-Native Migration

Convert the most important dashboards from static adapter surfaces into package-native dashboard apps.

Priority:

1. Media Engine Ops
2. Khashi VC ROC
3. Hermes Executive Summary
4. Media Business Operations
5. Business Mapper / Meal Assistant as needed

### V9: Live Data Contracts And Executive Signals

Create standard project dashboard contracts and wire real endpoints into Hermes.

Contracts:

- `DashboardSnapshot`
- `HealthSnapshot`
- `CostSnapshot`
- `CapacitySnapshot`
- `QueueSnapshot`
- `ActionNeeded`
- `ResearchSignal`
- `DeploymentSignal`

### V10: Premium Visual QA And Design Review

Move beyond “does it render” to “is it excellent and decision-ready.”

Add:

- Live production screenshots.
- Visual baselines.
- Dashboard quality scorecard.
- Manual review checklist.
- Recipe compliance scoring.

### V11: Agent-Enforced Dashboard Creation

Make Codex/Hermes follow the dashboard process by default.

Add:

- Required recipe selection.
- Required data contract.
- Required validation plan.
- Required screenshot evidence.
- Required migration/rollback note when touching production dashboards.

### V12: Hermes Central Command

Turn Hermes into the CEO/control-plane dashboard for TLC.

Add:

- Cross-project daily brief.
- Action-needed queue.
- Health/cost/capacity rollups.
- Project drilldowns.
- Agent task routing.
- Design/research/build status.
- Business impact summaries.

### V13: Multi-Brand Theme And Product Polish

Make dashboards feel like one operating system while allowing each business to retain identity.

Add:

- TLC base theme.
- Khashi VC research theme.
- Media Engine publishing theme.
- Media Business Ops analytics theme.
- Brand-level typography and density rules.

### V14: Dashboard Marketplace / Plugin System

Let each project register dashboards, panels, commands, and executive signals dynamically.

Add:

- Dashboard plugin manifest.
- Panel registry.
- Command registry.
- Health/cost/capacity signal registry.
- Permission-aware command execution.

## Practical Next Build Order

1. Build V8 for Media Engine first.
2. Build V8 for Khashi VC second.
3. Build V9 signal contracts while those migrations are happening.
4. Upgrade `/executive-summary` into the real Hermes central command layer.
5. Add V10 production screenshot QA.

This order gives the fastest visible improvement while also building toward the larger Hermes CEO/control-plane vision.
