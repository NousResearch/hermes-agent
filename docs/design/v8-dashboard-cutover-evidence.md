# V8 Dashboard Cutover Evidence

This file tracks the evidence required before any static production dashboard adapter can be retired.

## Status

| Dashboard | Package-Native Route | Snapshot Endpoint | Local Playwright | Production Screenshot | Rollback Path | Retirement |
|---|---|---|---|---|---|---|
| Media Engine Ops | `/package-native/media-engine` | `https://media.tlccapitalgroup.com/dashboard-snapshot` | `[x]` | `[ ]` | `[x]` | blocked |
| Khashi VC ROC | `/package-native/khashi-vc` | `https://roc.tlccapitalgroup.com/api/dashboard-snapshot` | `[x]` | `[ ]` | `[x]` | blocked |

## Evidence Captured

- Package-native shadow routes are registered in `web/src/dashboard-route-registry.tsx`.
- Route governance metadata exists in `web/src/dashboard-page-metadata.ts`.
- Local visual checks cover `/dashboard-migrations`, `/package-native/media-engine`, and `/package-native/khashi-vc`.
- Project-owned snapshot endpoints exist in the Media Engine and Khashi VC projects.
- Adapter retirement is disabled in `docs/design/package-native-parity-registry.json` until production screenshots are captured.

## Latest Production Reachability Check

Checked on July 20, 2026.

| URL | Result | Meaning |
|---|---|---|
| `https://media.tlccapitalgroup.com/dashboard` | `HTTP 200` | Current static Media Engine production dashboard is reachable. |
| `https://media.tlccapitalgroup.com/dashboard-snapshot` | `HTTP 200` | Media Engine production snapshot endpoint is reachable. |
| `https://media.tlccapitalgroup.com/api/dashboard-snapshot` | `HTTP 200` | Media Engine API snapshot alias is reachable. |
| `https://roc.tlccapitalgroup.com/` | `HTTP 200` | Current static Khashi VC production dashboard is reachable. |
| `https://roc.tlccapitalgroup.com/dashboard-snapshot` | `HTTP 404` | Root snapshot path is not exposed in production. |
| `https://roc.tlccapitalgroup.com/api/dashboard-snapshot` | `HTTP 401` | Authenticated Khashi API snapshot endpoint exists and requires credentials. |
| `https://roc.tlccapitalgroup.com/api/dashboard/snapshot` | `HTTP 401` | Authenticated Khashi API snapshot alias exists and requires credentials. |
| `https://agent.tlccapitalgroup.com/api/status` | `GET HTTP 200` | Nous Hermes Agent production dashboard is live, reports `auth_required=true`, and exposes the `basic` auth provider. |
| `https://agent.tlccapitalgroup.com/package-native/media-engine` | `HTTP 302` | Production protects the route path and redirects to login; authenticated route rendering still requires screenshot verification. |
| `https://agent.tlccapitalgroup.com/package-native/khashi-vc` | `HTTP 302` | Production protects the route path and redirects to login; authenticated route rendering still requires screenshot verification. |

## Production Evidence Still Required

- Desktop screenshot of the production Media Engine package-native route.
- Mobile screenshot of the production Media Engine package-native route.
- Desktop screenshot of the production Khashi VC package-native route.
- Mobile screenshot of the production Khashi VC package-native route.
- Health and snapshot endpoint check captured at the same deployment version.
- Confirmation that auth and command gating behave correctly in production.
- Confirmation that the deployed authenticated SPA contains the new `/package-native/media-engine` and `/package-native/khashi-vc` route bundle, not only a server-side login redirect.

## Production Check Command

Run unauthenticated reachability checks:

```bash
npm run dashboard:v8:production:check
```

Run authenticated screenshot checks:

```bash
HERMES_AGENT_DASHBOARD_USERNAME=<username> \
HERMES_AGENT_DASHBOARD_PASSWORD=<password> \
npm run dashboard:v8:production:check
```

Authenticated screenshots are written under `test-results/v8-production-cutover`.

Latest command result on July 20, 2026:

```text
npm run dashboard:v8:production:check
6 passed
4 skipped
```

The skipped tests are the authenticated screenshot checks. They require `HERMES_AGENT_DASHBOARD_USERNAME` and `HERMES_AGENT_DASHBOARD_PASSWORD`.

## Cutover Rule

Do not remove or demote the static dashboards until every parity flag for the target is true:

- `authPreserved`
- `commandsPreserved`
- `apiBehaviorPreserved`
- `snapshotEndpointExists`
- `packageNativeShadowRoute`
- `playwrightCoverage`
- `productionScreenshotEvidence`
- `rollbackPath`

Only then may `retirementAllowed` be changed to `true`.
