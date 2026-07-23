# Package-Native Cutover Checklist

Canonical machine-readable checklist: `docs/design/package-native-cutover-checklist.json`.

A static adapter dashboard can only be retired after the package-native replacement has production evidence. Local screenshots and local builds are necessary, but not sufficient.

## Required Evidence

| Evidence | Why It Matters |
| --- | --- |
| Desktop production screenshot | Confirms the deployed bundle renders the package-native route for real operators. |
| Mobile production screenshot | Confirms the layout survives constrained viewports. |
| Authentication preserved | Confirms the package-native route did not weaken login or role boundaries. |
| Command gating preserved | Confirms high-risk controls are still disabled or permission-gated. |
| Snapshot freshness | Confirms the route is showing current project state, not stale fixtures. |
| Rollback path | Confirms the static dashboard can be restored if package-native promotion fails. |
| Static route retained | Prevents first promotion from becoming an irreversible cutover. |

## Current Position

Media Engine Ops and Khashi VC ROC are still blocked from adapter retirement because authenticated production screenshot evidence and same-version snapshot freshness have not been captured.

Run:

```bash
HERMES_AGENT_DASHBOARD_USERNAME=<username> \
HERMES_AGENT_DASHBOARD_PASSWORD=<password> \
npm run dashboard:v8:production:check
```

Screenshots are written to `test-results/v8-production-cutover`.
