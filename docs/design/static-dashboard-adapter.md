# Static Dashboard Adapter

Some TLC dashboards are not React apps yet. Khashi VC ROC and Media Engine Ops currently render static HTML/vanilla JavaScript dashboards, so the React package alone is not enough to migrate them without a rewrite.

The shared package now includes:

```text
packages/hermes-dashboard-kit/static/hermes-dashboard-kit.css
```

Use this CSS adapter when a dashboard needs the Hermes visual system before it is converted to React.

## Required Static Classes

| Purpose | Static Class |
|---|---|
| Body styling | `hdk-body` |
| App shell | `hdk-shell` |
| Sidebar | `hdk-sidebar` |
| Main content | `hdk-main` |
| Header | `hdk-header` |
| Section | `hdk-section` |
| Card | `hdk-card` |
| Metric grid | `hdk-metric-grid` |
| KPI label | `hdk-kpi-label` |
| KPI value | `hdk-kpi-value` |
| KPI detail | `hdk-kpi-detail` |
| Button | `hdk-button` |
| Sidebar nav item | `hdk-nav-item` |
| Status pill | `hdk-pill` plus `success`, `warning`, or `critical` |
| Table wrapper | `hdk-table-wrap` |
| Table | `hdk-table` |
| Empty/loading/error | `hdk-empty`, `hdk-loading`, `hdk-error` |

## Khashi VC Static Migration

Start with:

- `public/roc/index.html`
- `public/roc/styles.css`
- `public/roc/app.js`

Recommended first pass:

- Add `hdk-body` to `<body>`.
- Add `hdk-shell` to `.app-shell`.
- Add `hdk-sidebar` to `.sidebar`.
- Add `hdk-main` to `.workspace`.
- Add `hdk-button` to operator action buttons.
- Add `hdk-nav-item` to `.nav-item`.
- Add `hdk-metric-grid` to `#summary`.
- Map existing `.metric-card`/summary cards to `hdk-card`.
- Map dashboard status chips to `hdk-pill`.

Do not change Khashi API calls, auth, scheduler controls, or experiment behavior in this pass.

## Media Engine Static Migration

Start with:

- `.lavish/media-operator-control-plane.html`
- `core/operations/unified-publishing-dashboard.js`

Recommended first pass:

- Add `hdk-body` to `<body>`.
- Use `hdk-header`, `hdk-section`, `hdk-card`, and `hdk-metric-grid` in generated HTML.
- Map autopilot buttons to `hdk-button`.
- Map dashboard registry rows to `hdk-card`.
- Map status tags to `hdk-pill`.
- Map generated tables/lists to `hdk-table` or `hdk-section`.

Do not change Discord delivery, autopilot control semantics, generated asset pruning, or publishing behavior in this pass.
