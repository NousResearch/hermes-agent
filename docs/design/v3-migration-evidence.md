# V3 Migration Evidence

V3 is complete as an adapter-level migration. It proves the dashboard design system on real operational surfaces without forcing Khashi VC or Media Engine through a full React rewrite.

## Targets

| Target | Type | Evidence | Validation |
|---|---|---|---|
| Hermes OS | React package-native | `web/src/pages/HermesOsPage.tsx` uses dashboard-kit primitives. | `npm run dashboard:visual:check` |
| Khashi VC ROC | Static adapter | `public/roc/index.html`, `public/roc/app.js`, and `public/roc/hermes-dashboard-kit.css` use `hdk-*` classes. | `npm run dashboard:v3:validate` |
| Media Engine Ops | Static generated adapter | `core/operations/unified-publishing-dashboard.js` generates `hdk-*` dashboard HTML with autopilot and Discord-output sections. | `npm run dashboard:media-engine:generated:validate` |

## Validation Commands

Run from `projects/nous-hermes-agent`:

```bash
npm run dashboard:v3:validate
npm run dashboard:usage:audit:strict
npm run dashboard:registry:validate
npm run dashboard:health:validate
npm run dashboard:visual:check
```

Optional live health check:

```bash
npm run dashboard:health:validate:live
```

The live check depends on deployed dashboards and local dashboard servers being reachable. The static health check verifies that every registered dashboard declares a health URL.

## What V3 Does Not Claim

- Khashi VC is not package-native React yet.
- Media Engine Ops is not package-native React yet.
- Static adapter validation checks required classes and generated HTML markers, not every runtime interaction.
- Production URL visual regression is not included by default because it depends on deployed auth/session state.

## Why This Is Still Complete

V3's goal was to prove the kit on real dashboards with operational complexity while preserving current production behavior. That is satisfied by:

- Hermes OS as the package-native reference implementation.
- Khashi VC as the most complex static adapter target.
- Media Engine as a generated static adapter target.
- Enforced adapter sync, registry validation, generated HTML validation, and Playwright smoke coverage.
