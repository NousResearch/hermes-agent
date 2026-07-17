# Dashboard Quality Scorecard

V10 defines how Hermes/TLC dashboards are judged after they technically render.

Minimum passing score: **80/100**.

## Categories

| Category | Points | What Good Looks Like |
|---|---:|---|
| Recipe fit | 15 | The dashboard declares and follows one V7 recipe. |
| Data contract | 15 | Required live/fallback data groups are visible and stale/unknown states are explicit. |
| Decision clarity | 15 | The page answers what happened, why it matters, freshness, owner, and next action. |
| Visual hierarchy | 15 | Headings, metrics, tables, charts, and commands have clear scan order and no generic clutter. |
| State coverage | 10 | Loading, empty, error, warning, critical, stale, and permission-limited states are represented. |
| Accessibility | 10 | Keyboard, contrast, axe smoke checks, labels, and reduced-motion behavior pass. |
| Responsiveness | 10 | Desktop and mobile layouts avoid overflow and preserve usable controls. |
| Production readiness | 10 | Production URL, health check, screenshot evidence, and rollback/verification notes exist. |

## Failure Conditions

A dashboard fails V10 review even with a high score when:

- It exposes secret values.
- It has no clear owner or next action for critical states.
- It uses a static adapter without a retirement/parity plan.
- It hides stale/unknown data behind a healthy-looking state.
- It cannot be reached from the dashboard launcher.

## Required Evidence

Every priority dashboard should have:

- V7 recipe id.
- Data contract list.
- Required states list.
- Validation commands.
- Screenshot or Playwright evidence.
- Production URL or documented local-only reason.
- Owner and category.

## Initial Priority Dashboards

- `/executive-summary`
- `/dashboard-migrations`
- `/design-system`
- Khashi VC ROC production dashboard
- Media Engine Ops production dashboard

Khashi VC and Media Engine stay in the scorecard even before package-native migration so their production quality can be measured while adapter retirement is planned.
