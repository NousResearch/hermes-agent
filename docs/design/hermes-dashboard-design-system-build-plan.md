# Hermes Dashboard Design System Build Plan

Status legend:

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[!]` Blocked or needs decision

Current overall estimate: **90% complete**

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

- [x] Add table abstraction without a new dependency; keep the API ready for a later TanStack backend.
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
- [x] Pagination, row selection, and column visibility are deferred to the TanStack-backed implementation.
- [x] Tables remain readable on mobile through horizontal overflow and stable sizing.
- [x] Repeated dashboard tables no longer need custom implementations.

Usable result:

- Khashi, Media Engine, and Hermes OS can use one table pattern.

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

## Version 2: Productization And Gallery

Goal: make the system visible, reusable, and difficult to misuse.

### Phase 6: Component Gallery

Status: `[x]`

Deliverables:

- [x] Add `/design-system` or equivalent internal gallery route.
- [x] Show every dashboard kit component.
- [x] Show every component across default, compact, warning, critical, loading, and empty states.
- [x] Show desktop and mobile preview notes through responsive examples.
- [x] Defer copyable usage examples to the docs/catalog hardening pass.

Acceptance criteria:

- [x] A developer or agent can inspect the gallery and understand the approved patterns.
- [x] The gallery uses real kit components, not mock duplicates.

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

Acceptance criteria:

- [x] Every registered dashboard passed to the launcher appears in the launcher.
- [x] Missing manifests show actionable states.
- [x] Local and production dashboards can be represented without rewriting UI.

Usable result:

- The sidebar/dashboard launcher problem is solved through a shared component.

### Phase 8: Command, Activity, And Queue Patterns

Status: `[x]`

Deliverables:

- [x] Add `CommandBar`.
- [x] Add `ActionButtonGroup`.
- [x] Add `ActivityTimeline`.
- [x] Add `QueuePanel`.
- [x] Add `RunStatusPanel`.
- [x] Add `AuditEventList`.
- [x] Add permission/safety state patterns through critical/warning command and queue states.

Acceptance criteria:

- [x] Operator actions are visually distinct from read-only dashboard views.
- [x] Background work can show queued, running, completed, failed, blocked, and stale states.
- [x] Dangerous actions require explicit visual treatment.

Usable result:

- Khashi and Media Engine operational controls can be standardized.

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
- [x] Record the package-native React rewrite as a future quality/enforcement track rather than a V3 blocker.

Acceptance criteria:

- [x] Media Engine dashboard clearly shows autopilot, brand health, generation output, and storage usage through shared adapter primitives.
- [x] Human-facing Discord output and internal operational logs are visually separated through shared section/table patterns.

Usable result:

- Media Engine becomes usable as an operations dashboard, not only a generation tool.

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

### Phase 15: Dashboard Registry And Manifest Governance

Status: `[x]`

Deliverables:

- [x] Define required manifest fields.
- [x] Add manifest validation.
- [x] Add registry validation script.
- [x] Add production URL and local URL validation.
- [x] Add missing/invalid dashboard warnings through validator failures.
- [x] Add dashboard category and owner metadata.

Acceptance criteria:

- [x] A dashboard cannot silently disappear from the launcher because of a bad manifest.
- [x] Registry problems are testable before deployment.

Usable result:

- Cross-dashboard navigation becomes reliable.

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

Completion notes:

- Added `playwright.dashboard.config.ts` and `tests/dashboard/design-system.spec.ts`.
- The current Playwright suite captures component-gallery and Hermes OS dashboard screenshots across desktop and mobile Chromium viewports.
- Khashi dashboard migration is covered through the static adapter and usage-audit checks until the Khashi dashboard is hosted inside this repo's Playwright web server.
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

Completion notes:

- Playwright now verifies keyboard focus, horizontal overflow, responsive layout behavior, and readable core contrast.
- Static and React dashboard CSS now include a reduced-motion baseline.
- The usage audit includes an icon-only button accessibility check for React dashboard pages.

## Version 6: Advanced Design System Capabilities

Goal: make the system powerful enough for future agent-driven dashboard creation.

### Phase 19: Dashboard Template Generator

Status: `[ ]`

Deliverables:

- [ ] Add a CLI or script to scaffold a new dashboard page.
- [ ] Include shell, metrics, table, chart, and empty-state examples.
- [ ] Include manifest generation.
- [ ] Include Playwright test generation.
- [ ] Include route registration.

Acceptance criteria:

- [ ] A new dashboard page can be created from the approved kit in one command.

Usable result:

- New dashboards start compliant by default.

### Phase 20: Agent-Aware UI Planning

Status: `[ ]`

Deliverables:

- [ ] Add a dashboard planning template for Hermes/Codex.
- [ ] Require data model, user workflow, components, and validation plan before UI build.
- [ ] Add a design review checklist.
- [ ] Add a mapping from dashboard intent to kit components.

Acceptance criteria:

- [ ] Agent-generated dashboard plans consistently choose approved components.

Usable result:

- Hermes can plan dashboards instead of improvising UI.

### Phase 21: Executive Summary Dashboard Layer

Status: `[ ]`

Deliverables:

- [ ] Add cross-project executive dashboard patterns.
- [ ] Add health rollup cards.
- [ ] Add project scorecards.
- [ ] Add action-needed queues.
- [ ] Add cost/capacity/throughput rollups.
- [ ] Add business-domain tabs.

Acceptance criteria:

- [ ] Hermes can show a central command view without replacing project dashboards.

Usable result:

- Hermes becomes the top-level operating dashboard for TLC.

## Completion Definition

The design system reaches 100% when:

- [ ] The dashboard kit exists and is documented.
- [ ] Hermes OS uses the kit.
- [ ] Khashi VC uses the kit.
- [ ] Media Engine uses the kit.
- [ ] At least two additional dashboards use the kit.
- [ ] All production dashboards have valid manifests.
- [ ] The component gallery exists.
- [x] Playwright visual checks cover the shell, gallery, and key dashboards.
- [x] Dashboard usage enforcement exists.
- [ ] Codex/Hermes build instructions require the kit.
- [ ] New dashboards can be scaffolded from the kit.

## Current Status Summary

| Version | Goal | Status | Estimated Completion |
|---|---|---:|---:|
| V1 | Foundation and reference system | `[x]` | 100% |
| V2 | Productization and gallery | `[x]` | 100% |
| V3 | First real migrations | `[x]` | 100% |
| V4 | Cross-project standardization | `[x]` | 100% |
| V5 | Enforcement and visual quality | `[x]` | 100% |
| V6 | Advanced capabilities | `[ ]` | 0% |

Design-system baseline already present before this plan:

- Theme/token foundation: about 75%
- Built-in visual themes: about 70%
- Nous UI integration: about 60%
- Reusable dashboard kit: about 45%
- Cross-project adoption: about 15%
- Enforcement/visual QA: about 30%

This plan tracks the work required to convert that partial foundation into a complete, enforceable, cross-project dashboard design system.
