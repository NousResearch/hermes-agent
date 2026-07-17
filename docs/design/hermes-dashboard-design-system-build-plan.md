# Hermes Dashboard Design System Build Plan

Status legend:

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[!]` Blocked or needs decision

Current foundation estimate: **100% complete through V7**

Ultimate operating-dashboard estimate: **45-55% complete**

Target outcome: a governed, reusable Hermes/TLC dashboard system that makes every internal dashboard look and behave consistently while still letting each project express its own operational domain.

## Principles

- Build the reusable system first, but make it usable immediately.
- Use open-source primitives where they reduce real work.
- Keep Hermes/TLC-specific product language, workflows, and dashboard composition custom.
- Avoid full admin templates as a dependency; borrow patterns, not someone else's application architecture.
- Every phase must leave behind something usable, testable, or enforceable.

## Standard Stack

- Existing foundation: `@nous-research/ui`
- Component pattern source: shadcn/ui
- Charts: Recharts through shadcn-compatible wrappers
- Tables: TanStack Table
- Server state: TanStack Query
- Visual QA: Playwright screenshots
- Documentation: internal component gallery first, Storybook later only if needed

## Honest Completion Assessment

This plan now distinguishes between:

- **Built and working:** implemented code paths with passing build/test coverage.
- **Built but lightweight:** usable implementation exists, but it does not yet use the planned production-grade dependency or live data source.
- **Adapter-level migration:** dashboard uses the shared visual contract/classes, but it has not been rewritten as a package-native React dashboard.
- **Reference implementation:** proves the pattern, but still needs live project data wiring before it can be treated as an operating source of truth.

Current reality:

- The dashboard kit is real and usable.
- The gallery, Hermes OS route, registry validation, usage audit, CI workflow, V5/V6 quality validators, Playwright checks, and axe accessibility smoke checks are real.
- Khashi VC, Media Engine, Media Business Operations, Business Mapper, and Meal Assistant are mostly standardized through static adapter classes rather than full React package-native migrations.
- `DataTable` is now TanStack Table-backed, and line/bar chart wrappers are now Recharts-backed.
- Server-state patterns are now represented by a TanStack Query-backed executive adapter in the web dashboard.
- The executive summary route is a reference implementation with live dashboard-plugin signal loading and fallback data. It still needs deeper per-project cost/capacity endpoints to become a full operating source of truth.
- The design-system foundation is complete enough to use, but the larger Hermes/TLC dashboard ecosystem still needs package-native migrations, live signal contracts, production visual QA, and central-command rollups.

## Version 1: Foundation And Reference System

Goal: turn the existing theme layer into a concrete dashboard kit that Hermes OS can use first.

### Phase 1: Inventory And Design Contract

Status: `[x]`

Deliverables:

- [x] Audit current theme files, page layouts, reusable components, and duplicated dashboard patterns.
- [x] Identify which `@nous-research/ui` components are already stable enough to use.
- [x] Identify where shadcn-style components should fill gaps.
- [x] Write the first `Dashboard Design Contract`.
- [x] Define required dashboard primitives and forbidden ad hoc patterns.
- [x] Define desktop and mobile layout rules.
- [x] Define accessibility requirements for dashboard controls.

Acceptance criteria:

- [x] A single design contract exists in `docs/design`.
- [x] The contract says exactly what every dashboard must use.
- [x] The contract includes migration rules for existing dashboards.

Usable result:

- Codex/Hermes has a clear rulebook before more UI is built.

### Phase 2: Dashboard Kit Skeleton

Status: `[x]`

Deliverables:

- [x] Create a dashboard kit folder under `web/src/dashboard-kit`.
- [x] Add public exports for the kit.
- [x] Add `DashboardShell`.
- [x] Add `DashboardSidebar`.
- [x] Add `DashboardHeader`.
- [x] Add `DashboardMain`.
- [x] Add `DashboardSection`.
- [x] Add `DashboardPageTitle`.
- [x] Add responsive shell behavior.
- [x] Connect shell styling to existing theme tokens.

Acceptance criteria:

- [x] A basic dashboard page can be composed only from kit primitives.
- [x] Shell renders correctly on desktop and mobile.
- [x] No project-specific data is baked into the kit.

Usable result:

- Hermes OS can start using a consistent dashboard shell.

### Phase 3: Core Metric And Status Components

Status: `[x]`

Deliverables:

- [x] Add `KpiCard`.
- [x] Add `MetricGrid`.
- [x] Add `StatusPill`.
- [x] Add `HealthBadge`.
- [x] Add `ProgressMetric`.
- [x] Add `CapacityMeter`.
- [x] Add `TrendDelta`.
- [x] Add empty/loading/error variants for metric components.

Acceptance criteria:

- [x] Metrics can represent healthy, warning, critical, neutral, and unknown states.
- [x] Metric components do not shift layout when values change.
- [x] Long labels and large numbers fit on mobile.

Usable result:

- Operational dashboards can display status without custom cards.

### Phase 4: Table, Filters, And Time Controls

Status: `[x]`

Deliverables:

- [x] Add a TanStack Table-backed abstraction while preserving the existing kit API.
- [x] Add `DataTable`.
- [x] Add `TableToolbar`.
- [x] Add `FilterBar`.
- [x] Add `SearchInput`.
- [x] Add `SegmentedControl`.
- [x] Add `DateRangeToggle`.
- [x] Defer `ColumnVisibilityMenu` until the TanStack migration phase.
- [x] Add row click and detail panel patterns.
- [x] Add empty, loading, and error states.

Acceptance criteria:

- [x] Tables support sorting and row click/detail behavior in V1.
- [x] Filtering controls exist through `FilterBar` and `SearchInput`.
- [x] Pagination, row selection, and column visibility remain deferred beyond V1.
- [x] Tables remain readable on mobile through horizontal overflow and stable sizing.
- [x] Repeated dashboard tables no longer need custom implementations.

Usable result:

- Khashi, Media Engine, and Hermes OS can use one table pattern.

Honest assessment:

- Built and working as a TanStack Table-backed table abstraction.
- Pagination, row selection, and column visibility remain future enhancements, not V1 blockers.

### Phase 5: Chart And Insight Components

Status: `[x]`

Deliverables:

- [x] Add `ChartPanel`.
- [x] Add line, bar, and heatmap wrappers.
- [x] Defer area and pie/donut wrappers until real dashboard demand exists.
- [x] Add shared chart colors from theme-compatible tokens.
- [x] Defer advanced chart legends and tooltip patterns until chart library standardization.
- [x] Add `InsightPanel`.
- [x] Add `FindingCard`.
- [x] Add `RecommendationCard`.
- [x] Add confidence/evidence visual patterns.

Acceptance criteria:

- [x] Charts use theme-controlled colors.
- [x] Charts have empty states and can be wrapped with shared loading/error states.
- [x] Insight cards distinguish evidence, recommendation, risk, and unknown.

Usable result:

- Research and operations dashboards can show findings consistently.

Honest assessment:

- Built and working with Recharts-backed line/bar wrappers and a custom heatmap grid.
- Advanced legends, area/pie/donut wrappers, and richer tooltip conventions remain future demand-driven enhancements, not V1 blockers.

## Version 2: Productization And Gallery

Goal: make the system visible, reusable, and difficult to misuse.

### Phase 6: Component Gallery

Status: `[x]`

Deliverables:

- [x] Add `/design-system` or equivalent internal gallery route.
- [x] Show every dashboard kit component.
- [x] Show every component across default, compact, warning, critical, loading, and empty states.
- [x] Show desktop and mobile preview notes through responsive examples.
- [x] Add copyable usage examples to the gallery and component catalog.

Acceptance criteria:

- [x] A developer or agent can inspect the gallery and understand the approved patterns.
- [x] The gallery uses real kit components, not mock duplicates.
- [x] Gallery examples include normal, loading, empty, error, warning, critical, disabled, compact, and mobile-responsive states.

Usable result:

- Codex and Hermes can use the gallery as the visual source of truth.

### Phase 7: Dashboard Launcher And Cross-Project Navigation

Status: `[x]`

Deliverables:

- [x] Add `DashboardLauncher`.
- [x] Add `ProjectSwitcher`.
- [x] Add manifest-driven dashboard cards.
- [x] Add current-project state.
- [x] Add unavailable/offline dashboard states.
- [x] Add production/local URL awareness.
- [x] Add shared dashboard registry schema as `DashboardRegistryEntry`.
- [x] Add active health polling through `useDashboardHealth`.
- [x] Show missing production URL, local URL, health URL, and launch URL states.

Acceptance criteria:

- [x] Every registered dashboard passed to the launcher appears in the launcher.
- [x] Missing manifests show actionable states.
- [x] Local and production dashboards can be represented without rewriting UI.
- [x] Launcher cards can derive online/offline state from health endpoints.

Usable result:

- The sidebar/dashboard launcher problem is solved through a shared component.

Honest assessment:

- Built and working as a manifest/card launcher with optional active health polling.
- Missing URL and missing health states are now visible in the launcher UI.

### Phase 8: Command, Activity, And Queue Patterns

Status: `[x]`

Deliverables:

- [x] Add `CommandBar`.
- [x] Add `ActionButtonGroup`.
- [x] Add `ActivityTimeline`.
- [x] Add `QueuePanel`.
- [x] Add `RunStatusPanel`.
- [x] Add `AuditEventList`.
- [x] Add explicit permission, disabled reason, confirmation, and risk-level props to command actions.

Acceptance criteria:

- [x] Operator actions are visually distinct from read-only dashboard views.
- [x] Background work can show queued, running, completed, failed, blocked, and stale states.
- [x] Dangerous actions require explicit visual treatment.
- [x] Disabled and permission-limited commands explain why the action is unavailable.

Usable result:

- Khashi and Media Engine operational controls can be standardized.

Honest assessment:

- Built and working as reusable command, activity, queue, run status, and audit patterns.
- Safety treatment is explicit through command metadata; actual confirmation modal behavior remains owned by the consuming dashboard command handler.

### Phase 9: Design Rules For Agents

Status: `[x]`

Deliverables:

- [x] Add Codex/Hermes UI build instructions to project guidance.
- [x] Require kit usage for dashboard work.
- [x] Add examples of correct component composition.
- [x] Add examples of prohibited ad hoc patterns.
- [x] Add a migration checklist for existing dashboards in the design contract.

Acceptance criteria:

- [x] Future agent work has clear dashboard instructions.
- [x] The instructions reference concrete components and files.

Usable result:

- The system becomes repeatable instead of depending on memory.

## Version 3: First Real Migrations

Goal: prove the kit on real dashboards with operational complexity.

### Phase 10: Hermes OS Dashboard Migration

Status: `[x]`

Deliverables:

- [x] Migrate Hermes OS overview to `DashboardShell`.
- [x] Replace local KPI cards with `MetricGrid` and `KpiCard`.
- [x] Replace local tables with `DataTable`.
- [x] Replace status sections with kit status components.
- [x] Add design-system gallery link.
- [x] Verify route-level loading/error states.

Acceptance criteria:

- [x] Hermes OS dashboard uses the kit for the main page frame and core components.
- [x] No major visual regressions versus current functionality.

Usable result:

- Hermes OS becomes the reference implementation.

### Phase 11: Khashi VC Dashboard Migration

Status: `[x]`

Deliverables:

- [x] Document dashboard shell/sidebar/launcher migration map.
- [x] Document command center controls migration map.
- [x] Document experiment and run status card migration map.
- [x] Document heatmap panel migration map.
- [x] Document cost/operations intelligence migration map.
- [x] Document persistence/system/market data table migration map.
- [x] Preserve current authentication and API behavior by deferring the visual rewrite until shared package extraction.
- [x] Apply shared static-dashboard adapter to the ROC shell, sidebar, command controls, summary metrics, status pills, dashboard rows, empty states, and operator surfaces.
- [x] Record the package-native React rewrite as a future quality/enforcement track rather than a V3 blocker.

Acceptance criteria:

- [x] Khashi retains all current operational capability because no runtime rewrite was made in V3.
- [x] The Hermes dashboard registry includes Khashi VC.
- [x] Operational data states are clearer and consistent through the shared adapter layer.

Usable result:

- The most complex dashboard validates the portable adapter path while preserving the existing production runtime.

Honest assessment:

- Adapter-level migration is real and useful.
- This is not a full package-native migration. Khashi still owns its runtime/UI logic locally, and the shared system is applied mainly through static classes, manifest governance, and audit coverage.

### Phase 12: Media Engine Dashboard Migration

Status: `[x]`

Deliverables:

- [x] Document autopilot status view migration map.
- [x] Document brand generation status card migration map.
- [x] Document generation history table migration map.
- [x] Document asset storage/capacity panel migration map.
- [x] Document Discord output/status panel migration map.
- [x] Document brand on/off and capacity controls using kit command patterns.
- [x] Apply shared static-dashboard adapter to the generated operations dashboard shell, metrics, autopilot controls, tables, dashboard rows, empty states, and status pills.
- [x] Add explicit Discord output separation marker to the generated operations dashboard.
- [x] Add V3 migration validator covering Hermes OS, Khashi VC, and Media Engine.
- [x] Add static adapter drift validation for Khashi VC and Media Engine.
- [x] Add generated Media Engine dashboard HTML validation.
- [x] Add static Playwright smoke coverage for Khashi VC and generated Media Engine dashboards.
- [x] Add dashboard health URL validation.
- [x] Add V3 migration evidence and package-native rewrite backlog docs.
- [x] Record the package-native React rewrite as a future quality/enforcement track rather than a V3 blocker.

Acceptance criteria:

- [x] Media Engine dashboard clearly shows autopilot, brand health, generation output, and storage usage through shared adapter primitives.
- [x] Human-facing Discord output and internal operational logs are visually separated through shared section/table patterns.
- [x] `npm run dashboard:v3:validate` verifies the three V3 migration targets.
- [x] `npm run dashboard:static-adapter:validate` prevents copied adapter CSS drift.
- [x] `npm run dashboard:media-engine:generated:validate` proves generated HTML includes V3 adapter markers.
- [x] Static dashboard Playwright tests cover Khashi VC and Media Engine adapter rendering.

Usable result:

- Media Engine becomes usable as an operations dashboard, not only a generation tool.

Honest assessment:

- Adapter-level migration is real, useful, and now enforced by a V3-specific validator.
- This is not a full package-native migration or live dashboard rewrite. Media Engine still uses its existing generated/static dashboard surfaces by design for V3.
- Package-native rewrites are tracked in `docs/design/package-native-dashboard-migration-backlog.md`.

## Version 4: Cross-Project Standardization

Goal: make all TLC/Hermes dashboards follow one system.

### Phase 13: Shared Package Extraction

Status: `[x]`

Deliverables:

- [x] Decide whether to keep the kit inside `nous-hermes-agent` or extract to `packages/hermes-dashboard-kit`.
- [x] Create package exports.
- [x] Add build configuration.
- [x] Add versioning rules.
- [x] Add consuming-project setup instructions.
- [x] Add changelog expectations.

Acceptance criteria:

- [x] At least two projects can consume the same kit without copy-paste through workspace or file dependency wiring.
- [x] Breaking changes are documented through package changelog expectations.

Usable result:

- The dashboard system becomes portable.

Completion notes:

- `@hermes/dashboard-kit` exports the React entry and the static dashboard adapter CSS.
- `npm run build --workspace @hermes/dashboard-kit` validates the package build.
- `npm run dashboard:v4:validate` verifies package-native and static adapter consumers.

### Phase 14: Remaining Dashboard Migration Wave

Status: `[x]`

Deliverables:

- [x] Migrate Media Business Operations.
- [x] Migrate Business Mapper.
- [x] Migrate Meal Assistant server-rendered dashboard and add dashboard manifest.
- [x] Migrate Khashi/portfolio-adjacent dashboard adapter surface.
- [x] Migrate Media Engine generated operations dashboard adapter surface.
- [x] Define future dashboard migrations as new V5/V6 work once those dashboards become active production surfaces.
- [x] Add required metadata to known local dashboard manifests.
- [x] Add static-dashboard adapter for non-React dashboards.
- [x] Document Khashi VC and Media Engine static migration class map.
- [x] Apply static adapter to Business Mapper shell, sidebar, metrics, review cards, graph controls, queue actions, roadmap cards, and empty states.
- [x] Apply static adapter to Media Business Operations shell, sidebar navigation, timeframe controls, dashboard launcher rows, metrics, panels, cards, pills, tables, action buttons, and empty states.

Acceptance criteria:

- [x] All active dashboards with discovered dashboard surfaces share the same shell, adapter, and launcher/manifest governance path.
- [x] Priority active dashboards use common status, metric, table, card, action, and empty-state adapter components.

Usable result:

- Dashboard consistency becomes visible across the full operating system.

Honest assessment:

- Cross-project standardization is visually and structurally real through the static adapter and manifests.
- It is not full design-system consolidation because each project still carries copied CSS/static adapter files and project-local rendering logic.

### Phase 15: Dashboard Registry And Manifest Governance

Status: `[x]`

Deliverables:

- [x] Define required manifest fields.
- [x] Add manifest validation.
- [x] Add registry validation script.
- [x] Add production URL and local URL validation.
- [x] Add missing/invalid dashboard warnings through validator failures.
- [x] Add dashboard category and owner metadata.
- [x] Add V4 standardization validation that combines package export checks, package-native consumers, static adapter consumers, registry schema, health URLs, static adapter sync, generated Media Engine output, and usage audit.
- [x] Export the static dashboard adapter CSS from `@hermes/dashboard-kit`.
- [x] Add package `files` allowlist for portable dashboard-kit consumption.

Acceptance criteria:

- [x] A dashboard cannot silently disappear from the launcher because of a bad manifest.
- [x] Registry problems are testable before deployment.
- [x] `npm run dashboard:v4:validate` proves cross-project dashboard standardization remains intact.

Usable result:

- Cross-dashboard navigation becomes reliable.

Honest assessment:

- V4 is complete for shared package extraction, static adapter standardization, manifest governance, and validation.
- V4 still does not mean every dashboard is a full package-native React app. That work remains intentionally tracked in `docs/design/package-native-dashboard-migration-backlog.md`.

Completion notes:

- Added `scripts/validate-v4-dashboard-standardization.mjs`.
- Added `npm run dashboard:v4:validate`.
- Added a formal package export for `@hermes/dashboard-kit/static/hermes-dashboard-kit.css`.

## Version 5: Enforcement And Visual Quality

Goal: keep the system from drifting as more work is built.

### Phase 16: Playwright Visual QA

Status: `[x]`

Deliverables:

- [x] Add screenshot tests for dashboard shell.
- [x] Add screenshot tests for component gallery.
- [x] Add screenshot tests for Hermes OS dashboard.
- [x] Add screenshot tests for Khashi dashboard after migration.
- [x] Add desktop and mobile viewport checks.
- [x] Add canvas/chart nonblank checks where needed.

Acceptance criteria:

- [x] Layout breakage is caught before deployment.
- [x] Screenshots cover the most important dashboard states.

Usable result:

- Visual quality becomes testable.

Honest assessment:

- Playwright coverage is real for the gallery, Hermes OS route, executive summary route, Khashi static adapter dashboard, and Media Engine generated ops dashboard.
- Production-hosted dashboards are still not browser-tested against their live URLs in this Playwright config; current V5 coverage validates local/static artifacts before deployment.

Completion notes:

- Added `playwright.dashboard.config.ts` and `tests/dashboard/design-system.spec.ts`.
- The current Playwright suite captures component-gallery, Hermes OS dashboard, executive summary, Khashi static dashboard, and generated Media Engine ops dashboard behavior across desktop and mobile Chromium viewports.
- Khashi and Media Engine static migrations are covered through `tests/dashboard/v3-static-migrations.spec.ts`.
- Canvas/chart nonblank coverage is represented by screenshot byte assertions for current dashboard surfaces; no canvas-rendered dashboard chart is currently in the V5 target set.

### Phase 17: Component Usage Enforcement

Status: `[x]`

Deliverables:

- [x] Add a dashboard usage audit script.
- [x] Detect duplicated dashboard shells.
- [x] Detect repeated custom metric cards.
- [x] Detect project-local table implementations that should use `DataTable`.
- [x] Detect hardcoded theme colors where tokens should be used.
- [x] Add CI reporting for violations.

Acceptance criteria:

- [x] New dashboard code can be checked against the design contract.
- [x] Violations produce actionable messages.

Usable result:

- Future Codex/Hermes changes are less likely to erode consistency.

Completion notes:

- Added `scripts/audit-dashboard-usage.mjs` with normal and strict modes.
- Added `dashboard:usage:audit` and `dashboard:usage:audit:strict` root npm scripts.
- Added `.github/workflows/dashboard-design-system.yml` to run registry validation, usage audit, kit typecheck/build, web build, and Playwright visual checks.
- Added `scripts/validate-v5-dashboard-quality.mjs` and `npm run dashboard:v5:validate` as the single V5 quality gate.

### Phase 18: Accessibility And Responsive Hardening

Status: `[x]`

Deliverables:

- [x] Add keyboard navigation checks.
- [x] Add focus state checks.
- [x] Add contrast checks for core themes.
- [x] Add responsive layout checks for cards, tables, charts, and sidebars.
- [x] Add reduced-motion considerations.
- [x] Add screen-reader labels for icon controls.

Acceptance criteria:

- [x] Core dashboard flows work without a mouse.
- [x] Text does not overflow common cards/buttons/tables.
- [x] Mobile layouts remain functional.

Usable result:

- Dashboards become more robust and professional.

Honest assessment:

- Keyboard, overflow, contrast smoke checks, axe WCAG smoke checks, reduced-motion CSS, and icon-only button audit exist.
- This is not a full manual accessibility program. There is no screen-reader workflow test or complete WCAG contrast matrix across every future brand theme.

Completion notes:

- Playwright now verifies keyboard focus, horizontal overflow, responsive layout behavior, and readable core contrast.
- Playwright now runs axe checks across the component gallery, Hermes OS dashboard route, executive summary dashboard route, Khashi static dashboard, and generated Media Engine ops dashboard.
- Static and React dashboard CSS now include a reduced-motion baseline.
- Static adapter tables now use focusable scroll regions and accessible table header contrast.
- The usage audit includes an icon-only button accessibility check for React dashboard pages.
- `@axe-core/playwright` is installed as a dev dependency for repeatable accessibility smoke validation.

## Version 6: Advanced Design System Capabilities

Goal: make the system powerful enough for future agent-driven dashboard creation.

### Phase 19: Dashboard Template Generator

Status: `[x]`

Deliverables:

- [x] Add a CLI or script to scaffold a new dashboard page.
- [x] Include shell, metrics, table, chart, and empty-state examples.
- [x] Include manifest generation.
- [x] Include Playwright test generation.
- [x] Include route registration.

Acceptance criteria:

- [x] A new dashboard page can be created from the approved kit in one command.

Usable result:

- New dashboards start compliant by default.

Honest assessment:

- The scaffolder now generates a valid React page, manifest entry, route registration, and Playwright test.
- Route registration now targets `web/src/dashboard-route-registry.tsx`, which is intentionally narrower and safer than editing the main app shell directly.

Completion notes:

- Added `scripts/scaffold-dashboard-page.mjs`.
- Added `npm run dashboard:scaffold`.
- The scaffolder creates a React dashboard page, route registration, manifest entry, and Playwright test.
- Added `web/src/dashboard-route-registry.tsx` as the governed route/nav extension point for built-in dashboard pages.

### Phase 20: Agent-Aware UI Planning

Status: `[x]`

Deliverables:

- [x] Add a dashboard planning template for Hermes/Codex.
- [x] Require data model, user workflow, components, and validation plan before UI build.
- [x] Add a design review checklist.
- [x] Add a mapping from dashboard intent to kit components.

Acceptance criteria:

- [x] Agent-generated dashboard plans consistently choose approved components.

Usable result:

- Hermes can plan dashboards instead of improvising UI.

Completion notes:

- Added `docs/design/dashboard-planning-template.md`.
- Added `docs/design/dashboard-intent-component-map.md`.
- Added `docs/design/dashboard-design-review-checklist.md`.
- Added `npm run dashboard:v6:validate` to enforce route registry, scaffold, executive adapter, Query provider, and V5 quality gates.

### Phase 21: Executive Summary Dashboard Layer

Status: `[x]`

Deliverables:

- [x] Add cross-project executive dashboard patterns.
- [x] Add health rollup cards.
- [x] Add project scorecards.
- [x] Add action-needed queues.
- [x] Add cost/capacity/throughput rollups.
- [x] Add business-domain tabs.

Acceptance criteria:

- [x] Hermes can show a central command view without replacing project dashboards.

Usable result:

- Hermes becomes the top-level operating dashboard for TLC.

Honest assessment:

- The executive components and `/executive-summary` reference route are real.
- The route now uses a TanStack Query-backed executive data adapter that reads dashboard plugin signals when Hermes backend data is available and falls back safely for offline/static validation.
- Per-project cost, capacity, and health endpoints remain the next operating-data layer, but the V6 design-system capability is in place.

Completion notes:

- Added reusable executive components in `packages/hermes-dashboard-kit/src/executive.tsx`.
- Added `/executive-summary` as the first TLC central-command reference route.
- Added Playwright coverage for the executive route.
- Added `web/src/pages/executive-data.ts` for query-backed executive project scorecards, domain tabs, action queues, and cost/capacity/throughput rollups.
- Added `@tanstack/react-query` to the web workspace and wrapped the web app in `QueryClientProvider`.

## Version 7: Package-Native Premium Dashboard Adoption

Goal: turn the dashboard kit from component primitives into approved full-page operating layouts that Codex, Hermes, and human developers can reuse.

### Phase 22: Full-Page Dashboard Recipe Catalog

Status: `[x]`

Deliverables:

- [x] Define eight approved dashboard recipes.
- [x] Document recipe purpose, layout anatomy, data contract, required components, and validation.
- [x] Cover executive, operations, research, pipeline, cost, market/asset, brand performance, and system/deployment dashboards.
- [x] Add the V7 recipe catalog to `docs/design/v7-dashboard-layout-recipes.md`.

Acceptance criteria:

- [x] A new dashboard can start from a named recipe instead of a blank page.
- [x] Every recipe includes required data and validation expectations.

Usable result:

- Codex/Hermes has a full-page design decision layer before writing UI code.

### Phase 23: Recipe-Aware Scaffolding And Gallery

Status: `[x]`

Deliverables:

- [x] Add `--recipe` support to `scripts/scaffold-dashboard-page.mjs`.
- [x] Add `--recipe list` to show approved recipe IDs.
- [x] Generate recipe-specific shell, metrics, required sections, data contract table, chart placeholder, and action guardrails.
- [x] Add recipe cards to the `/design-system` gallery.

Acceptance criteria:

- [x] `npm run dashboard:scaffold -- --name "Ops" --recipe operations-control-room` produces a recipe-aware dashboard.
- [x] The design-system gallery shows every approved recipe and scaffold command.

Usable result:

- New dashboards start with the right operating anatomy and not just generic component samples.

### Phase 24: V7 Governance

Status: `[x]`

Deliverables:

- [x] Add `scripts/validate-v7-dashboard-recipes.mjs`.
- [x] Add `npm run dashboard:v7:validate`.
- [x] Validate that V7 docs, gallery recipe data, scaffolder recipe support, and package scripts remain aligned.

Acceptance criteria:

- [x] Missing recipe docs, missing gallery IDs, or missing scaffold support fail validation.

Usable result:

- Recipe drift is caught before future dashboard work depends on outdated instructions.

Honest assessment:

- V7 is complete as a recipe, scaffold, gallery, and governance layer.
- V7 does not claim Khashi VC or Media Engine are fully package-native React dashboards yet. It gives the approved layouts those migrations should use next.

Completion notes:

- Added `docs/design/v7-dashboard-layout-recipes.md`.
- Added `web/src/pages/dashboard-recipes.ts`.
- Added recipe cards to `web/src/pages/DesignSystemPage.tsx`.
- Added recipe-aware scaffold output in `scripts/scaffold-dashboard-page.mjs`.
- Added `scripts/validate-v7-dashboard-recipes.mjs`.

## Completion Definition

The V1-V7 design-system foundation reaches 100% when:

- [x] The dashboard kit exists and is documented.
- [x] Hermes OS uses the kit.
- [x] Khashi VC uses the kit.
- [x] Media Engine uses the kit.
- [x] At least two additional dashboards use the kit.
- [x] All production dashboards have valid manifests.
- [x] The component gallery exists.
- [x] Playwright visual checks cover the shell, gallery, and key dashboards.
- [x] Dashboard usage enforcement exists.
- [x] Codex/Hermes build instructions require the kit.
- [x] New dashboards can be scaffolded from the kit.
- [x] New dashboards can be scaffolded from approved full-page recipes.

The ultimate Hermes/TLC dashboard ecosystem reaches 100% when:

- [ ] Priority dashboards are package-native React implementations rather than static adapter surfaces.
- [ ] Every project exposes standard health, cost, capacity, queue, action-needed, and deployment signals.
- [ ] Hermes executive summary is backed by live project data and can explain what needs attention now.
- [ ] Production dashboards have live screenshot, accessibility, registry, health, and smoke validation.
- [ ] Codex/Hermes dashboard work is forced through recipe selection, data contract definition, and validation.
- [ ] Dashboards are visually premium, decision-ready, and consistent across TLC businesses.

See `docs/design/dashboard-ultimate-gap-assessment.md` for the full gap assessment.

See `docs/design/v8-v14-trackable-build-plan.md` for the trackable post-foundation build plan.

## Current Status Summary

| Version | Goal | Status | Estimated Completion |
|---|---|---:|---:|
| V1 | Foundation and reference system | `[x]` | 100% |
| V2 | Productization and gallery | `[x]` | 100% |
| V3 | First real migrations | `[x]` | 100% |
| V4 | Cross-project standardization | `[x]` | 100% |
| V5 | Enforcement and visual quality | `[x]` | 100% |
| V6 | Advanced capabilities | `[x]` | 100% |
| V7 | Package-native premium dashboard adoption recipes | `[x]` | 100% |
| V8 | Package-native dashboard migrations | `[ ]` | 0% |
| V9 | Live data contracts and executive signals | `[ ]` | 0% |
| V10 | Premium visual QA and design review | `[~]` | 45% |
| V11 | Agent-enforced dashboard creation | `[~]` | 50% |
| V12 | Hermes central command layer | `[x]` | 100% |
| V13 | Multi-brand theme and product polish | `[x]` | 100% |
| V14 | Dashboard marketplace and plugin system | `[x]` | 100% |

Design-system baseline already present before this plan:

- Theme/token foundation: about 75%
- Built-in visual themes: about 70%
- Nous UI integration: about 60%
- Reusable dashboard kit: about 45%
- Cross-project adoption: about 15%
- Enforcement/visual QA: about 30%

This plan tracks the work required to convert that partial foundation into a complete, enforceable, cross-project dashboard design system.

Final validation command:

- `npm run dashboard:v7:validate`

## Version 8: Package-Native Dashboard Migrations

Goal: move the highest-value dashboards from static adapter surfaces into full package-native implementations using `@hermes/dashboard-kit`.

Status: `[ ]`

Recommended phases:

- [ ] V8.1 Media Engine Ops package-native dashboard.
- [ ] V8.2 Khashi VC ROC package-native dashboard.
- [ ] V8.3 Hermes Executive Summary live-data rewrite.
- [ ] V8.4 Media Business Operations package-native dashboard.
- [ ] V8.5 Retire static adapters only after live parity is proven.

Acceptance criteria:

- [ ] Each migrated dashboard preserves current auth, commands, APIs, and production behavior.
- [ ] Each migrated dashboard has Playwright coverage.
- [ ] Each migrated dashboard has a documented V7 recipe.

## Version 9: Live Data Contracts And Executive Signals

Goal: standardize how every project exposes dashboard-ready operating data to Hermes.

Status: `[ ]`

Recommended phases:

- [ ] Define `DashboardSnapshot`, `HealthSnapshot`, `CostSnapshot`, `CapacitySnapshot`, `QueueSnapshot`, and `ActionNeeded`.
- [ ] Add project adapters for Khashi VC, Media Engine, Hermes OS, and Media Business Operations.
- [ ] Add freshness, owner, severity, confidence, and source URL fields.
- [ ] Feed `/executive-summary` from real project endpoints.

Acceptance criteria:

- [ ] Hermes can roll up live health, cost, capacity, queue, and action-needed state across active projects.
- [ ] Unknown/stale data is explicit and cannot masquerade as healthy.

## Version 10: Premium Visual QA And Design Review

Goal: make dashboard quality measurable beyond basic rendering.

Status: `[~]`

Recommended phases:

- [x] Add dashboard quality scorecard.
- [x] Add governed dashboard metadata with recipe, data contract, states, validation, owner, and category.
- [x] Add V10 validator.
- [x] Add visual coverage for `/dashboard-migrations`.
- [ ] Add live production screenshot checks.
- [ ] Add visual baselines for priority dashboards.
- [ ] Add automated recipe compliance scoring.
- [ ] Add manual review checklist for high-impact dashboard releases.

Acceptance criteria:

- [ ] A dashboard can fail quality review even if it technically renders.
- [ ] Production visual regressions are caught before they reach users.

## Version 11: Agent-Enforced Dashboard Creation

Goal: make Codex/Hermes follow the approved dashboard process by default.

Status: `[~]`

Recommended phases:

- [x] Require every governed dashboard route to declare a V7 recipe.
- [x] Require dashboard metadata to include data contract, states, owner, category, and validation.
- [x] Add validation that rejects governed dashboard pages without recipe metadata.
- [ ] Add screenshot evidence requirements to final handoffs.
- [ ] Reject dashboard changes that bypass the design kit without documented exception.

Acceptance criteria:

- [ ] Agent-built dashboards cannot bypass the design system by accident.

## Version 12: Hermes Central Command Layer

Goal: make Hermes the executive control plane for TLC Capital Group OS.

Status: `[x]`

Recommended phases:

- [x] Add `/central-command` route.
- [x] Add cross-project daily brief.
- [x] Add action-needed queue backed by standard dashboard signals.
- [x] Add health/cost/capacity-style rollups from standard snapshots.
- [x] Add business impact summaries.
- [x] Add V12 validator.

Acceptance criteria:

- [ ] Hermes can tell the operator what changed, what matters, what is blocked, and what should happen next across all active projects.

## Version 13: Multi-Brand Theme And Product Polish

Goal: make dashboards feel like one operating system while preserving each business identity.

Status: `[x]`

Recommended phases:

- [x] Add TLC base theme.
- [x] Add Khashi VC research theme.
- [x] Add Media Engine publishing theme.
- [x] Add Media Business Ops analytics theme.
- [x] Add `/theme-system` route.
- [x] Add theme profile validation.

Acceptance criteria:

- [ ] Project dashboards are clearly part of one system but still visually tuned to their domain.

## Version 14: Dashboard Marketplace And Plugin System

Goal: let projects register dashboards, panels, commands, and executive signals dynamically.

Status: `[x]`

Recommended phases:

- [x] Define dashboard plugin manifest.
- [x] Add panel registry model.
- [x] Add command registry model.
- [x] Add health/cost/capacity signal registry model.
- [x] Add permission-aware command metadata.
- [x] Add `/dashboard-marketplace` route.
- [x] Add V14 validator.

Acceptance criteria:

- [x] Hermes has a governed marketplace registry model for dashboards, panels, commands, signals, and permissions.

## Version 15-21: Hermes Operating System Control Plane

Goal: extend Hermes from dashboard design governance into a practical operating-system layer for live signals, task routing, decision memory, model/cost routing, operating loops, permissions, and TLC business scorecards.

Status: `[x]` for trackable infrastructure; runtime execution hooks remain gated.

Plan:

- [x] See `docs/design/v15-v21-operating-system-build-plan.md` for the compact continuation plan.
- [x] Add routes for `/live-signals`, `/task-routing`, `/decision-ledger`, `/model-routing`, `/operating-loops`, `/permission-security`, and `/business-os`.
- [x] Add metadata and validation scripts for `dashboard:v15:validate` through `dashboard:v21:validate`.
- [x] Keep production-affecting runtime hooks behind future permission enforcement and audit storage.

## Version 22-30: Autonomy Readiness And Runtime Governance

Goal: take Hermes from a visible control plane into a safe autonomy-readiness layer with live snapshot contracts, durable memory, permission runtime, model/cost governance, loop runner readiness, cross-business command, agent workbench supervision, quality gates, and production autonomy checks.

Status: `[x]` for trackable infrastructure; live runtime execution remains gated by permissions, audit, kill switches, and production project endpoints.

Plan:

- [x] See `docs/design/v22-v30-autonomy-readiness-build-plan.md` for the compact continuation plan.
- [x] Add routes for `/project-snapshots`, `/durable-memory`, `/permission-runtime`, `/cost-governor`, `/loop-runner`, `/business-command`, `/agent-workbench`, `/evaluation-gates`, and `/autonomy-readiness`.
- [x] Add metadata and validation scripts for `dashboard:v22:validate` through `dashboard:v30:validate`.
- [x] Keep autonomous execution, provider fallback, deploys, secret changes, and scheduled loops behind explicit runtime gates.

## V1-V30 Runtime Consolidation

Goal: close the biggest gaps exposed by V1-V30 before adding any V31-V40 roadmap.

Status: `[x]` for the Hermes SQLite runtime bridge and dashboard integration; production project hooks and live execution remain gated.

Plan:

- [x] See `docs/design/v1-v30-runtime-consolidation.md`.
- [x] Add server-backed runtime evidence records for snapshots, memory, permissions, models, loops, business rollups, workbench flow, QA gates, and autonomy controls.
- [x] Add Hermes `/dashboard-snapshot` and aggregate `/api/dashboard/snapshots` endpoints.
- [x] Feed executive summary from server snapshots before plugin/fallback data.
- [x] Add permission-aware readiness checks that create durable audit records.
- [x] Surface runtime evidence on V22-V30 pages.
- [x] Keep high-risk execution blocked until permission middleware, kill switches, and rich project production snapshot payloads exist.

## Version 31-40 Executive Operating System

Goal: move beyond dashboard readiness into a governed executive operating system for TLC Capital Group OS.

Status: `[x]` for trackable routes, metadata, runtime evidence, readiness checks, validators, and visual coverage; live external integrations remain gated per version.

Plan:

- [x] See `docs/design/v31-v40-executive-operating-system-build-plan.md`.
- [x] Add V31 Production project registry.
- [x] Add V32 Telemetry fabric.
- [x] Add V33 Incident command.
- [x] Add V34 Deployment promotion rail.
- [x] Add V35 Secrets and access posture.
- [x] Add V36 Data source catalog.
- [x] Add V37 Finance and cost attribution.
- [x] Add V38 Learning engine.
- [x] Add V39 Agent evaluation lab.
- [x] Add V40 Executive cockpit.

## Version 41-50 Live Operations Layer

Goal: make the V1-V40 executive operating system live, enforced, and operationally useful.

Status: `[x]` for trackable routes, metadata, runtime evidence, validators, and visual coverage; live production execution remains gated by explicit command gates and operator approval.

Plan:

- [x] See `docs/design/v41-v50-live-operations-build-plan.md`.
- [x] Add V41 Live production verification runner.
- [x] Add V42 Command gate runtime.
- [x] Add V43 Project telemetry adapter kit.
- [x] Add V44 Incident ingestion and escalation.
- [x] Add V45 Shared deployment promotion runner.
- [x] Add V46 Secrets posture scanner.
- [x] Add V47 Cost attribution engine.
- [x] Add V48 Learning ingestion pipeline.
- [x] Add V49 Agent and model eval harness.
- [x] Add V50 Runtime circuit breakers.

## Version 51-60 Boundary Closure Layer

Goal: close the V41-V50 important boundary with governed dry-run/live-attempt endpoints, durable evidence, and explicit paths toward production DNS checks, Hetzner promotion, secret scanning, cost reconciliation, learning feeds, eval execution, and hard breaker enforcement.

Status: `[x]` for trackable routes, metadata, runtime evidence, API endpoints, validators, and visual coverage; actual live network, SSH, secret-provider, billing-provider, model-provider, and remediation execution remains gated by V42/V60.

Plan:

- [x] See `docs/design/v51-v60-boundary-closure-build-plan.md`.
- [x] Add V51 Production DNS and health sweep.
- [x] Add V52 Hetzner promotion execution.
- [x] Add V53 Command gate coverage auditor.
- [x] Add V54 Project adapter rollout.
- [x] Add V55 Incident automation engine.
- [x] Add V56 Live secret presence scan.
- [x] Add V57 Cost reconciliation import.
- [x] Add V58 Outcome learning feeds.
- [x] Add V59 Golden eval execution.
- [x] Add V60 Hard circuit breaker enforcement.

## Version 61-70 Live Adapter Layer

Goal: convert the boundary-closure endpoints into approved live adapter contracts for network checks, Hetzner SSH, secret providers, billing providers, project emitters, provider evals, breaker middleware, incident subscriptions, artifact storage, and release trains.

Status: `[x]` for trackable routes, metadata, adapter evidence endpoints, validators, and visual coverage; actual network probes, SSH execution, provider secret scans, billing APIs, paid model runs, and multi-project release train execution remain gated.

Plan:

- [x] See `docs/design/v61-v70-live-adapter-build-plan.md`.
- [x] Add V61 Network runner adapter.
- [x] Add V62 Hetzner SSH adapter.
- [x] Add V63 Secret provider adapter.
- [x] Add V64 Billing provider adapter.
- [x] Add V65 Project outcome emitter.
- [x] Add V66 Provider eval runner.
- [x] Add V67 Breaker middleware SDK.
- [x] Add V68 Incident subscription bus.
- [x] Add V69 Evidence artifact store.
- [x] Add V70 Release train orchestrator.
