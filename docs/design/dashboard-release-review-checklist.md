# Dashboard Release Review Checklist

Use this checklist for high-impact dashboard releases, adapter retirements, and production design-system changes.

## Required Release Evidence

- [ ] V7 recipe id is documented.
- [ ] Data contracts are listed and match the route metadata.
- [ ] Required states are checked: normal or ready, loading, empty or missing, warning, critical or failed, stale, permission-limited when relevant, and mobile.
- [ ] `npm run dashboard:recipe:score:strict` passes.
- [ ] `npm run dashboard:design-system:status -- --strict` passes.
- [ ] `npm run dashboard:visual:check` passes or a local-only screenshot reason is documented.
- [ ] Production URL, health URL, and snapshot URL are listed when the dashboard is production-facing.
- [ ] Production screenshot evidence is captured through `/production-screenshot-runner` or an approved production sweep.
- [ ] Rollback path is documented before retiring a static adapter.

## Failure Conditions

A release should not proceed when:

- A dashboard bypasses `@hermes/dashboard-kit` without a documented design-system exception.
- A copied static adapter is drifted from the canonical CSS.
- A dashboard hides stale or unknown data behind a healthy state.
- A production route has no health or snapshot evidence.
- A high-risk command lacks permission metadata or rollback notes.

## Manual Review Questions

1. Can the operator tell what changed, what matters, and what action is needed?
2. Are stale, unknown, failed, and permission-limited states visible?
3. Does the page use the approved recipe without mixing unrelated dashboard patterns?
4. Are source dashboard links, owners, and freshness visible?
5. Is the screenshot evidence recent enough to trust?

