# V8 Static Adapter Rollback Plan

Use this rollback plan if a package-native dashboard replacement fails after promotion.

## Scope

This applies to:

- Media Engine Ops
- Khashi VC ROC

The static dashboards remain the source of production truth until V8 cutover evidence is complete.

## Preconditions

- Confirm the failing route and deployment version.
- Confirm whether the failure is visual, auth, command, data, or deployment related.
- Do not delete the static adapter files during the first package-native promotion.

## Rollback Steps

1. Restore production routing to the existing static dashboard surface.
2. Disable or hide the package-native route from production navigation if it creates operator confusion.
3. Restart the affected dashboard service.
4. Verify the static dashboard health endpoint.
5. Verify the static dashboard loads under the production URL.
6. Confirm critical commands are available only to the correct auth role.
7. Record the failed route, deployment version, root cause, and rollback timestamp.

## Media Engine Ops

Current static surface:

- Production URL: `https://media.tlccapitalgroup.com/dashboard`
- Snapshot URL: `https://media.tlccapitalgroup.com/dashboard-snapshot`
- Local source: `../media-engine/core/operations/unified-publishing-dashboard.js`
- Static adapter: `../media-engine/core/operations/hermes-dashboard-kit.css`

Rollback verification:

- The operations dashboard renders.
- Autopilot controls remain permission-gated.
- Generation, approval, Discord delivery, storage, and production-run status remain visible.

## Khashi VC ROC

Current static surface:

- Production URL: `https://roc.tlccapitalgroup.com/`
- Snapshot URL: `https://roc.tlccapitalgroup.com/api/dashboard-snapshot`
- Local source: `../khashi-vc/public/roc`
- Static adapter: `../khashi-vc/public/roc/hermes-dashboard-kit.css`

Rollback verification:

- The ROC dashboard renders.
- Command buttons remain permission-gated.
- Run monitor, market data, coverage, cost, persistence, activity, and system views remain visible.

## Failure Classes

| Failure | Rollback Trigger |
|---|---|
| Auth failure | Operator cannot log in or role boundaries change. |
| Data failure | Snapshot, health, queue, cost, or action data is missing or stale without warning. |
| Command failure | A command can run without proper permission metadata or fails silently. |
| Visual failure | Desktop or mobile route has overflow, blank panels, unreadable text, or inaccessible controls. |
| Deployment failure | Route returns 5xx, stale assets, or incorrect version. |

## Recovery Rule

After rollback, keep `retirementAllowed` set to `false` and add the incident to the release review notes before another cutover attempt.
